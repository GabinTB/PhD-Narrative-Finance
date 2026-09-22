"""Orchestration: the ONE code path for live and historical narrative scoring.

    result = score_dates(
        dates, config,
        table=table, primitive_embeddings=P,
        source=ParquetHeadlineSource(...),        # or InMemoryHeadlineSource for experiments
        calibration=TauSeriesProvider(...),       # tau_asof + mu_asof, resolved per day
        writer=ParquetMonthWriter(out_dir, diagnostics_dir),
        null_sink=MonthlyNullPartitionWriter(...),   # feeds the f0_monthly_partitions family
    )

A live run passes today's date; a backtest passes a range; a notebook
passes a handful of days and keeps the result in memory. Nothing else
differs. Per day, in order:

    1. mu / mu_hat as of the day (calibration.mu_for), tau as of the day
       (calibration.tau_for) -- both refuse to look ahead;
    2. the mode-corrected scoring matrix for that mu (cached per mu date);
    3. for each bounded chunk of headline embeddings: apply_mode, S = H @ P.T
       with paraphrase pooling, then select + aggregate into the day's
       accumulators (compiled kernel when built, numpy otherwise -- same
       numbers), sample the chunk's null draws into the month partition, and
       drop S;
    4. close the day: day x narrative frame (one row per narrative and
       sentiment label), optional day x primitive diagnostics, the day
       diagnostics row with RSS before/after, hand to the writer and the sink.

Sentiment split (config.sentiment_split = sign): every headline contributes
to the "all" rows; headlines with a sentiment score contribute additionally
to their label's rows (pos: s > eps, neg: s < -eps, neu: |s| <= eps when
eps > 0). Headlines with missing sentiment (NaN, or exactly 0 when eps == 0)
appear in "all" only. N_HEADLINES on every row is the day's total, so
ATTENTION = TOTAL_SCORE / N_HEADLINES decomposes across labels;
N_LABELLED is the row's own headline count.

Every number comes from selection.py / aggregation.py / f0.py (or the
kernel asserted equal to them); this module only sequences calls.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal, Protocol

import numpy as np
import polars as pl

from narrative_scoring._kernels import HAVE_SELECT, select_aggregate_rowwise
from narrative_scoring.aggregation import DayAccumulator, headline_narrative_scores
from narrative_scoring.calibration import (
    CalibrationProvider,
    LookaheadError,
    MuRecord,
    TauRecord,
)
from narrative_scoring.config import (
    PERCENTILE_AXIS,
    SENTIMENT_ALL,
    AggRule,
    RunMetadata,
    ScoringConfig,
    SentimentSplit,
    n_candidates_for,
)
from narrative_scoring.corrections import Correction, apply_mode
from narrative_scoring.f0 import n_keep, n_trim, sample_null_draws, trim_threshold
from narrative_scoring.partitions import NullDrawSink
from narrative_scoring.primitives import (
    PrimitiveTable,
    embeddings_digest,
    primitive_scores,
    scoring_matrix,
)
from narrative_scoring.schema import (
    DAY_DIAGNOSTICS_SCHEMA,
    EMBEDDING_DIM,
    NARRATIVE_DAILY_SCHEMA,
    PRIMITIVE_DAILY_SCHEMA,
)
from narrative_scoring.selection import select
from narrative_scoring.streaming import (
    Chunk,
    HeadlineSource,
    MemoryBudgetExceeded,
    prefetch,
    rss_gb,
)

log = logging.getLogger(__name__)

_LABEL_CODES = {"pos": 1, "neg": 2, "neu": 3}


# ---------------------------------------------------------------------------
# Results and sinks
# ---------------------------------------------------------------------------

@dataclass
class DayResult:
    day: date
    narrative_daily: pl.DataFrame
    primitive_daily: pl.DataFrame | None
    diagnostics: dict[str, Any]


class OutputWriter(Protocol):
    def write_day(self, result: DayResult) -> None: ...

    def close(self, metadata: RunMetadata) -> None: ...


@dataclass
class ScoringResult:
    """What ``score_dates`` returns: the day x narrative panel plus run-level context."""

    narrative_daily: pl.DataFrame
    primitive_daily: pl.DataFrame | None
    day_diagnostics: pl.DataFrame
    metadata: RunMetadata | None          # None when every day was null-only (cold start)
    peak_rss_gb: float
    skipped_days: list[date] = field(default_factory=list)
    null_only_days: list[date] = field(default_factory=list)

    @property
    def n_days(self) -> int:
        return self.day_diagnostics.height


class ParquetMonthWriter:
    """``YYYY-MM.parquet`` per month under ``out_dir`` (narrative_daily) and under
    ``diagnostics_dir`` (day_diagnostics), plus ``run_metadata.json`` in both.

    Days are written in the order received (``score_dates`` sorts them); a
    month is flushed when the first day of a later month arrives or on close.
    Primitive-grain diagnostics, when produced, go to ``out_dir/primitive_daily/``.
    """

    def __init__(self, out_dir: Path, diagnostics_dir: Path | None = None):
        self.out_dir = Path(out_dir)
        self.diag_dir = (Path(diagnostics_dir) if diagnostics_dir
                         else self.out_dir / "day_diagnostics")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.diag_dir.mkdir(parents=True, exist_ok=True)
        self._month: tuple[int, int] | None = None
        self._narr: list[pl.DataFrame] = []
        self._prim: list[pl.DataFrame] = []
        self._diag: list[dict[str, Any]] = []

    def write_day(self, r: DayResult) -> None:
        ym = (r.day.year, r.day.month)
        if self._month is not None and ym != self._month:
            self._flush()
        self._month = ym
        self._narr.append(r.narrative_daily)
        if r.primitive_daily is not None:
            self._prim.append(r.primitive_daily)
        self._diag.append(r.diagnostics)

    def _flush(self) -> None:
        if self._month is None or not self._narr:
            return
        name = f"{self._month[0]}-{self._month[1]:02d}.parquet"
        _atomic_parquet(pl.concat(self._narr), self.out_dir / name)
        _atomic_parquet(pl.DataFrame(self._diag, schema=DAY_DIAGNOSTICS_SCHEMA),
                        self.diag_dir / name)
        if self._prim:
            (self.out_dir / "primitive_daily").mkdir(exist_ok=True)
            _atomic_parquet(pl.concat(self._prim), self.out_dir / "primitive_daily" / name)
        self._narr, self._prim, self._diag = [], [], []

    def close(self, metadata: RunMetadata) -> None:
        import json

        self._flush()
        text = json.dumps(metadata.to_dict(), indent=2, sort_keys=True, default=str) + "\n"
        (self.out_dir / "run_metadata.json").write_text(text)
        (self.diag_dir / "run_metadata.json").write_text(text)


def _atomic_parquet(df: pl.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(".parquet.tmp")
    try:
        df.write_parquet(tmp, compression="zstd")
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(path)


# ---------------------------------------------------------------------------
# One block: steps 3-5 into the day's accumulators
# ---------------------------------------------------------------------------

@dataclass
class _Funnel:
    n_candidates: int = 0
    n_f0_survivors: int = 0
    n_retained_pre_jump: int = 0
    n_retained: int = 0
    n_jump_trimmed: int = 0
    jump_gap_sum: float = 0.0


@dataclass
class _DayState:
    narr: dict[str, DayAccumulator]
    prim: dict[str, DayAccumulator] | None
    funnel: _Funnel = field(default_factory=_Funnel)
    n_with_sentiment: int = 0


def sentiment_labels(sentiment: np.ndarray | None, n: int, config: ScoringConfig) -> np.ndarray:
    """int8 per headline: 0 = unlabelled, 1 = pos, 2 = neg, 3 = neu."""
    codes = np.zeros(n, dtype=np.int8)
    if sentiment is None or config.sentiment_split is SentimentSplit.NONE:
        return codes
    s = np.asarray(sentiment, dtype=np.float64)
    eps = config.neutral_eps
    ok = np.isfinite(s)
    codes[ok & (s > eps)] = 1
    codes[ok & (s < -eps)] = 2
    if eps > 0:
        codes[ok & (np.abs(s) <= eps)] = 3
    return codes


def _accumulate_block(
    S: np.ndarray, tau32: float, n_candidates: int, config: ScoringConfig,
    prim_to_narr: np.ndarray, n_narr: int, narr: DayAccumulator, prim: DayAccumulator | None,
    use_kernel: bool, threads: int, funnel: _Funnel | None, n_trim_rows: int,
) -> np.ndarray | None:
    """Steps 3-5 for one block into ``narr``/``prim``. Returns the trim thresholds when asked."""
    n_head = S.shape[0]
    jump_min = config.jump_min_candidates if config.jump_cut else -1
    thresholds = None
    if use_kernel:
        k = select_aggregate_rowwise(
            np.ascontiguousarray(S, dtype=np.float32), tau32, n_candidates, jump_min,
            prim_to_narr, n_narr, config.narrative_agg is AggRule.MEDIAN,
            prim is not None, threads, n_trim_rows,
        )
        narr.add_arrays(k["narr_count"], k["narr_total"], k["narr_sumsq"], k["narr_peak"])
        if prim is not None:
            prim.add_arrays(k["prim_count"], k["prim_total"], k["prim_sumsq"], k["prim_peak"])
        n_unassigned = int(k["n_unassigned"])
        if n_trim_rows:
            thresholds = k["trim_threshold"]
        if funnel is not None:
            funnel.n_candidates += int(k["n_candidates"])
            funnel.n_f0_survivors += int(k["n_f0_survivors"])
            funnel.n_retained_pre_jump += int(k["n_retained_pre_jump"])
            funnel.n_retained += int(k["n_retained"])
            funnel.n_jump_trimmed += int(k["n_jump_trimmed"])
            funnel.jump_gap_sum += float(k["jump_gap_sum"])
    else:
        sel = select(S, tau32, n_candidates, jump_cut=config.jump_cut,
                     jump_min_candidates=config.jump_min_candidates)
        _, nid, nval = headline_narrative_scores(
            sel.rows, sel.cols, sel.vals, prim_to_narr, n_narr, config.narrative_agg)
        narr.add_values(nid, nval)
        if prim is not None:
            prim.add_values(sel.cols.astype(np.int64), sel.vals)
        n_unassigned = sel.n_unassigned
        if n_trim_rows:
            thresholds = trim_threshold(S, config.trim_frac)
        if funnel is not None:
            funnel.n_candidates += int(sel.n_candidates.sum())
            funnel.n_f0_survivors += int(sel.n_f0_survivors.sum())
            funnel.n_retained_pre_jump += sel.n_retained_pre_jump
            funnel.n_retained += int(sel.n_retained.sum())
            funnel.n_jump_trimmed += sel.n_jump_trimmed
            funnel.jump_gap_sum += sel.jump_gap_sum
    narr.n_headlines += n_head
    narr.n_unassigned += n_unassigned
    if prim is not None:
        prim.n_headlines += n_head
        prim.n_unassigned += n_unassigned
    return thresholds


# ---------------------------------------------------------------------------
# The canonical entry point
# ---------------------------------------------------------------------------

def score_dates(
    dates: Iterable[date],
    config: ScoringConfig,
    *,
    table: PrimitiveTable,
    primitive_embeddings: np.ndarray,
    source: HeadlineSource,
    calibration: CalibrationProvider,
    writer: OutputWriter | None = None,
    null_sink: NullDrawSink | None = None,
    tau_missing: Literal["raise", "null_only"] = "raise",
    keep_primitive_daily: bool = False,
    collect: bool = True,
    use_kernel: bool | None = None,
    threads: int = 8,
    rss_budget_gb: float | None = 30.0,
    prefetch_depth: int = 2,
    seed: int = 0,
    code_version: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> ScoringResult:
    """Score every day in ``dates`` (sorted, de-duplicated) with point-in-time inputs.

    Args:
        dates: calendar days to score. One day = a live run; a range = a backtest.
        config: the numerical configuration (recorded in the result's metadata).
        table: the primitive table the embeddings were built from.
        primitive_embeddings: ``embed_primitive_texts`` output for ``table``.
        source: where the day's headline embeddings (and optional sentiment) come from.
        calibration: mu_asof and tau as of each day.
        writer: optional sink receiving each closed day; ``close`` gets the metadata.
        null_sink: optional monthly null-partition builder fed with every chunk's draws.
        tau_missing: what to do on a day with no tau row old enough: ``raise`` (default)
            or ``null_only`` -- the day still feeds ``null_sink`` (which only needs mu)
            but produces no narrative_daily / day_diagnostics rows. This is the cold
            start of the live loop, nothing else.
        keep_primitive_daily: also produce day x primitive diagnostics.
        collect: keep the day frames in memory for the returned result.
        use_kernel: force the compiled kernel (True), numpy (False) or auto (None).
        threads: kernel thread count; BLAS keeps its own pool (phases are sequential).
        rss_budget_gb: raise ``MemoryBudgetExceeded`` naming the day when RSS exceeds it.
        seed: recorded; the scorer itself is deterministic (null sampling is seeded by the sink).
    Days with no headlines are skipped (listed in ``skipped_days``) but still closed
    in the null sink, so a month with quiet days can complete.
    """
    if use_kernel is None:
        use_kernel = HAVE_SELECT
    elif use_kernel and not HAVE_SELECT:
        raise RuntimeError("use_kernel=True but the compiled kernel is not built; "
                           "run `python -m narrative_scoring._kernels.build`")
    if primitive_embeddings.shape != (table.n_primitives * table.n_texts, EMBEDDING_DIM):
        raise ValueError("primitive_embeddings do not match the primitive table")

    days = sorted(set(dates))
    if not days:
        raise ValueError("no dates to score")
    n_prim, n_narr = table.n_primitives, table.n_narratives
    prim_to_narr = table.primitive_to_narrative
    n_candidates = n_candidates_for(config.q, n_prim)
    labels = config.sentiment_labels()
    narr_nodes, prim_nodes = table.narrative_nodes, table.primitive_nodes
    n_trim_rows = n_trim(n_prim, config.trim_frac) if null_sink is not None else 0
    n_keep_rows = n_keep(n_prim, config.trim_frac)
    log.info("scoring %d day(s) [%s .. %s] mode=%s pooling=%s q=%g (k=%d) jump=%s split=%s "
             "path=%s null_sink=%s",
             len(days), days[0], days[-1], config.mode.value, config.paraphrase_pooling.value,
             config.q, n_candidates, config.jump_cut, config.sentiment_split.value,
             "kernel" if use_kernel else "numpy", null_sink is not None)

    matrix_cache: dict[date | None, np.ndarray] = {}

    def scoring_matrix_for(mu: MuRecord | None) -> np.ndarray:
        key = mu.date if mu is not None else None
        if key not in matrix_cache:
            matrix_cache.clear()          # one live matrix at a time; mu changes monotonically
            matrix_cache[key] = scoring_matrix(
                primitive_embeddings, table, config.mode, config.paraphrase_pooling,
                mu.mu if mu else None, mu.mu_hat if mu else None)
        return matrix_cache[key]

    narr_frames: list[pl.DataFrame] = []
    prim_frames: list[pl.DataFrame] = []
    diags: list[dict[str, Any]] = []
    seen_days: set[date] = set()
    null_only_days: list[date] = []
    peak_rss = 0.0
    first_tau: TauRecord | None = None
    current: dict[str, Any] | None = None

    def open_day(day: date) -> dict[str, Any] | None:
        try:
            mu = calibration.mu_for(day) if config.mode is not Correction.RAW else None
        except LookaheadError:
            if tau_missing != "null_only":
                raise
            log.info("%s: no mu_asof row yet; day skipped entirely (nothing to correct with)",
                     day)
            return None
        try:
            tau_rec: TauRecord | None = calibration.tau_for(day)
        except LookaheadError:
            if tau_missing != "null_only" or null_sink is None:
                raise
            tau_rec = None
            null_only_days.append(day)
            log.info("%s: no tau row old enough; null-only day (partitions fed, no scores)", day)
        return {
            "day": day, "t0": time.perf_counter(), "rss_before": rss_gb(),
            "mu": mu, "tau_rec": tau_rec,
            # Cast once so the numpy path and the kernel compare float32 to float32.
            "tau32": float(np.float32(tau_rec.tau)) if tau_rec is not None else None,
            "P_scoring": scoring_matrix_for(mu),
            "state": _DayState(
                narr={lab: DayAccumulator(n_narr) for lab in (SENTIMENT_ALL, *labels)},
                prim=({lab: DayAccumulator(n_prim) for lab in (SENTIMENT_ALL, *labels)}
                      if keep_primitive_daily else None),
            ),
        }

    def close_day(ctx: dict[str, Any]) -> None:
        nonlocal peak_rss
        day, state, mu, tau_rec = ctx["day"], ctx["state"], ctx["mu"], ctx["tau_rec"]
        rss_before, rss_after = ctx["rss_before"], rss_gb()
        peak_rss = max(peak_rss, rss_before, rss_after)
        if rss_budget_gb is not None and rss_after > rss_budget_gb:
            raise MemoryBudgetExceeded(
                f"RSS {rss_after:.2f} GB exceeded budget {rss_budget_gb:.2f} GB scoring {day}")
        if tau_rec is None:                      # null-only day: nothing to write
            if null_sink is not None:
                null_sink.close_day(day, rss_after)
            return
        all_acc = state.narr[SENTIMENT_ALL]
        n_head = all_acc.n_headlines

        nd = pl.concat([
            acc.frame(narr_nodes, day, sentiment=lab, n_headlines=n_head)
            for lab, acc in state.narr.items()
        ]).select(list(NARRATIVE_DAILY_SCHEMA.keys()))
        pd_ = None
        if state.prim is not None:
            pd_ = pl.concat([
                acc.frame(prim_nodes, day, sentiment=lab, n_headlines=n_head)
                for lab, acc in state.prim.items()
            ]).select(list(PRIMITIVE_DAILY_SCHEMA.keys()))

        f = state.funnel
        diag = {
            "DATE": day, "N_HEADLINES": n_head, "N_UNASSIGNED": all_acc.n_unassigned,
            "N_F0_SURVIVORS_PRE_Q": f.n_f0_survivors, "N_Q_CANDIDATES": f.n_candidates,
            "N_RETAINED_PRE_JUMP": f.n_retained_pre_jump,
            "N_RETAINED_POST_Q_TAU": f.n_retained,
            "MEAN_RETAINED_PER_HEADLINE": f.n_retained / max(n_head, 1),
            "PCT_TAU_PRUNED_WITHIN_Q": 1.0 - f.n_retained_pre_jump / max(f.n_candidates, 1),
            "PCT_JUMP_APPLIED": f.n_jump_trimmed / max(n_head, 1),
            "MEAN_JUMP_GAP": (f.jump_gap_sum / f.n_jump_trimmed) if f.n_jump_trimmed else None,
            "NARRATIVES_TOUCHED": int((all_acc.count > 0).sum()),
            "N_WITH_SENTIMENT": state.n_with_sentiment,
            "MU_DATE": mu.date if mu else None, "MU_NORM": mu.norm if mu else None,
            "TAU": ctx["tau32"], "TAU_MONTH_END": tau_rec.month_end, "N_EFF": tau_rec.n_eff,
            "CONFIG_ID": config.digest(), "F0_CONFIG_ID": config.f0_digest(),
            "TAU_SOURCE_ID": calibration.tau_source_id, "MU_ASOF_ID": calibration.mu_asof_id,
            "RSS_GB_BEFORE": rss_before, "RSS_GB_AFTER": rss_after,
            "SECONDS": time.perf_counter() - ctx["t0"],
        }
        log.info("%s: %d headlines, %d unassigned, %.2f retained/headline, RSS %.2f -> %.2f GB, "
                 "%.1fs", day, n_head, diag["N_UNASSIGNED"], diag["MEAN_RETAINED_PER_HEADLINE"],
                 rss_before, rss_after, diag["SECONDS"])
        result = DayResult(day, nd, pd_, diag)
        if writer is not None:
            writer.write_day(result)
        if collect:
            narr_frames.append(nd)
            if pd_ is not None:
                prim_frames.append(pd_)
        diags.append(diag)
        if null_sink is not None:
            null_sink.close_day(day, rss_after)

    unusable: set[date] = set()
    for day, chunk in prefetch(_day_chunks(source, days), depth=prefetch_depth):
        if day in unusable:
            continue
        if current is None or current["day"] != day:
            if current is not None:
                close_day(current)
                current = None
            ctx = open_day(day)
            if ctx is None:
                unusable.add(day)
                continue
            current = ctx
            seen_days.add(day)
            first_tau = first_tau or current["tau_rec"]
        mu, state = current["mu"], current["state"]
        X = chunk.embeddings
        H = apply_mode(X, config.mode, mu.mu if mu else None, mu.mu_hat if mu else None)
        S = primitive_scores(H, current["P_scoring"], n_prim, table.n_texts,
                             config.paraphrase_pooling)
        if current["tau32"] is None:             # null-only day
            draws = sample_null_draws(S, trim_threshold(S, config.trim_frac),
                                      config.null_draws_per_headline, null_sink.rng_for(day))
            null_sink.add_draws(day, draws, S.shape[0] * n_keep_rows, S.shape[0])
            del H, S, X, chunk
            continue
        thresholds = _accumulate_block(
            S, current["tau32"], n_candidates, config, prim_to_narr, n_narr,
            state.narr[SENTIMENT_ALL], state.prim[SENTIMENT_ALL] if state.prim else None,
            use_kernel, threads, state.funnel, n_trim_rows)
        if labels:
            codes = sentiment_labels(chunk.sentiment, S.shape[0], config)
            state.n_with_sentiment += int((codes > 0).sum())
            for lab in labels:
                mask = codes == _LABEL_CODES[lab]
                if mask.any():
                    _accumulate_block(
                        S[mask], current["tau32"], n_candidates, config, prim_to_narr, n_narr,
                        state.narr[lab], state.prim[lab] if state.prim else None,
                        use_kernel, threads, None, 0)
        if null_sink is not None:
            draws = sample_null_draws(S, thresholds, config.null_draws_per_headline,
                                      null_sink.rng_for(day))
            null_sink.add_draws(day, draws, S.shape[0] * n_keep_rows, S.shape[0])
        del H, S, X, chunk
    if current is not None:
        close_day(current)

    skipped = [d for d in days if d not in seen_days]
    for d in skipped:
        if d in unusable:
            continue                             # never closed: it had no mu row at all
        log.info("%s: no headlines, skipped", d)
        if null_sink is not None and _mu_available(calibration, config, d):
            null_sink.close_day(d, rss_gb())     # a quiet day still counts towards its month
    if null_sink is not None:
        null_sink.flush_open()
    if first_tau is None:
        if tau_missing == "null_only":
            # the replay's cold start: nothing scorable in this batch is not an error
            log.info("%d null-only day(s), %d day(s) without a mu row, %d quiet day(s); "
                     "no scores produced", len(null_only_days), len(unusable),
                     len(skipped) - len(unusable))
            return ScoringResult(
                narrative_daily=pl.DataFrame(schema=NARRATIVE_DAILY_SCHEMA),
                primitive_daily=None,
                day_diagnostics=pl.DataFrame(schema=DAY_DIAGNOSTICS_SCHEMA),
                metadata=None, peak_rss_gb=peak_rss, skipped_days=skipped,
                null_only_days=null_only_days)
        raise RuntimeError(f"no headlines in any of the {len(days)} requested day(s)")

    metadata = RunMetadata(
        config=config.to_dict(), percentile_axis=PERCENTILE_AXIS, n_candidates=n_candidates,
        n_primitives=n_prim, n_narratives=n_narr, n_primitive_texts=len(table.texts),
        k_paraphrases=table.k_paraphrases, embedding_dim=EMBEDDING_DIM,
        taxonomy_name=table.name, taxonomy_sha1=table.taxonomy_sha1,
        paraphrase_sha1=table.paraphrase_sha1,
        primitive_embeddings_digest=embeddings_digest(primitive_embeddings),
        mu_asof_id=calibration.mu_asof_id, mu_policy=calibration.mu_policy,
        tau_source_id=calibration.tau_source_id, tau_policy=calibration.tau_policy,
        n_eff=first_tau.n_eff,
        seed=seed, code_version=code_version if code_version is not None else _git_revision(),
        config_id=config.digest(), f0_config_id=config.f0_digest(),
        extra={"source": source.describe(), "scoring_path": "kernel" if use_kernel else "numpy",
               "first_tau_record": first_tau.to_dict(), **(extra_metadata or {})},
    )
    if writer is not None:
        writer.close(metadata)

    empty_narr = pl.DataFrame(schema=NARRATIVE_DAILY_SCHEMA)
    return ScoringResult(
        narrative_daily=pl.concat(narr_frames) if narr_frames else empty_narr,
        primitive_daily=pl.concat(prim_frames) if prim_frames else None,
        day_diagnostics=pl.DataFrame(diags, schema=DAY_DIAGNOSTICS_SCHEMA),
        metadata=metadata, peak_rss_gb=peak_rss, skipped_days=skipped,
        null_only_days=null_only_days,
    )


def _mu_available(calibration: CalibrationProvider, config: ScoringConfig, day: date) -> bool:
    """A day can only count towards its null partition if it could have been corrected."""
    if config.mode is Correction.RAW:
        return True
    try:
        calibration.mu_for(day)
        return True
    except LookaheadError:
        return False


def _day_chunks(source: HeadlineSource, days: list[date]) -> Iterator[tuple[date, Chunk]]:
    """Day-ordered (day, chunk) stream over ``days``; days without headlines yield nothing."""
    for day in days:
        for chunk in source.iter_day(day):
            yield day, chunk


def date_range(start: date, end: date) -> Iterator[date]:
    """Every calendar day in [start, end]."""
    from datetime import timedelta

    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def _git_revision() -> str | None:
    try:
        from datalake.meta import git_commit
        return git_commit(Path(__file__).resolve().parents[3])
    except Exception:  # noqa: BLE001 - metadata only
        return None
