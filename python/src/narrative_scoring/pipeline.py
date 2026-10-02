"""Orchestration: the ONE code path for live and historical narrative scoring.

    result = score_dates(
        dates, config,
        table=table, primitive_embeddings=P,
        source=PartitionHeadlineSource(...),      # or InMemoryHeadlineSource for experiments
        calibration=TauSeriesProvider(...),       # tau_asof + mu_asof, resolved per day
        writer=ParquetMonthWriter(out_dir, diagnostics_dir),
        null_sink=MonthlyNullPartitionWriter(...),   # feeds the f0_monthly_partitions family
        assets=AssetUniverse(...),                # optional asset layer
    )

A live run passes today's date; a backtest passes a range; a notebook
passes a handful of days and keeps the result in memory. Nothing else
differs. Per day, in order:

    1. mu / mu_hat as of the day (calibration.mu_for), tau as of the day
       (calibration.tau_for) -- both refuse to look ahead;
    2. the mode-corrected scoring matrix for that mu (cached per mu date);
    3. for each bounded chunk of the day's headlines (EVERY headline): mode.correct,
       S = H @ P.T with paraphrase pooling, the bipolar pole mask when
       ``config.mask_bipolar`` (in place on S, selection.apply_pole_mask), then select +
       aggregate (compiled kernel when built, numpy otherwise -- same numbers), each row
       into the accumulators of its sentiment label; the chunk's null draws (every row)
       into the month partition; the asset layer, when given; then S is dropped;
    4. close the day: day x narrative rows per label, optional day x primitive
       diagnostics, one day_diagnostics row per label with RSS before/after, the asset
       rows; hand them to the writer and the day to the sink.

Sentiment layer (``config.tags``, tags.py): each chunk carries the tag code of every row;
a row feeds its tag's narrative (and asset) accumulators, an untagged row (no interval,
or no score) feeds none, is counted (N_UNTAGGED) and, like every row, feeds the null
model. N_HEADLINES is the day's total on every row, so ATTENTION = TOTAL_SCORE /
N_HEADLINES adds up across labels (plus the untagged share); N_LABELLED is the label's
own count. Without tags there is one label, "all", and the output is the historical
all-headlines run, bit for bit (same kernel call).

Every number comes from selection.py / aggregation.py / f0.py / assets.py (or the
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

from datalake.periods import partition_file, period_key
from narrative_scoring._kernels import HAVE_SELECT, select_aggregate_rowwise
from narrative_scoring.aggregation import DayAccumulator, headline_narrative_scores
from narrative_scoring.assets import AssetDay, AssetUniverse
from narrative_scoring.calibration import CalibrationProvider, LookaheadError, TauRecord
from narrative_scoring.config import (
    PERCENTILE_AXIS,
    AggRule,
    RunMetadata,
    ScoringConfig,
    n_candidates_for,
)
from narrative_scoring.f0 import n_keep, n_trim, sample_null_draws, trim_threshold
from narrative_scoring.partitions import NullDrawSink
from narrative_scoring.primitives import (
    BipolarPairs,
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
from narrative_scoring.selection import apply_pole_mask, select, select_with_pole_mask
from narrative_scoring.streaming import (
    Chunk,
    HeadlineSource,
    MemoryBudgetExceeded,
    prefetch,
    rss_gb,
)
from nlp.corrections import Correction
from nlp.reference_vector import ReferenceValue

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Results and sinks
# ---------------------------------------------------------------------------

@dataclass
class DayResult:
    day: date
    narrative_daily: pl.DataFrame
    primitive_daily: pl.DataFrame | None
    diagnostics: list[dict[str, Any]]          # one row per sentiment label
    asset_attention: pl.DataFrame | None = None
    narrative_asset: pl.DataFrame | None = None


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
    asset_attention: pl.DataFrame | None = None
    narrative_asset: pl.DataFrame | None = None

    @property
    def n_days(self) -> int:
        return self.day_diagnostics["DATE"].n_unique()


class ParquetMonthWriter:
    """``{period key}.parquet`` per period of ``freq`` (default M: ``YYYY-MM.parquet``)
    under ``out_dir`` (narrative_daily), ``diagnostics_dir`` (day_diagnostics) and, with
    an asset layer, ``asset_attention_dir`` / ``narrative_asset_dir``; plus
    ``run_metadata.json`` in each.

    Days are written in the order received (``score_dates`` sorts them); a
    period is flushed when the first day of a later period arrives or on close.
    Primitive-grain diagnostics, when produced, go to ``out_dir/primitive_daily/``.
    """

    def __init__(self, out_dir: Path, diagnostics_dir: Path | None = None, freq: str = "M",
                 asset_attention_dir: Path | None = None,
                 narrative_asset_dir: Path | None = None):
        self.freq = freq
        self.out_dir = Path(out_dir)
        self.diag_dir = (Path(diagnostics_dir) if diagnostics_dir
                         else self.out_dir / "day_diagnostics")
        self.att_dir = Path(asset_attention_dir) if asset_attention_dir else None
        self.cross_dir = Path(narrative_asset_dir) if narrative_asset_dir else None
        for d in (self.out_dir, self.diag_dir, self.att_dir, self.cross_dir):
            if d is not None:
                d.mkdir(parents=True, exist_ok=True)
        self._month: str | None = None           # the open period's key
        self._narr: list[pl.DataFrame] = []
        self._prim: list[pl.DataFrame] = []
        self._diag: list[dict[str, Any]] = []
        self._att: list[pl.DataFrame] = []
        self._cross: list[pl.DataFrame] = []

    def write_day(self, r: DayResult) -> None:
        ym = period_key(r.day, self.freq)
        if self._month is not None and ym != self._month:
            self._flush()
        self._month = ym
        self._narr.append(r.narrative_daily)
        if r.primitive_daily is not None:
            self._prim.append(r.primitive_daily)
        self._diag.extend(r.diagnostics)
        if r.asset_attention is not None:
            self._att.append(r.asset_attention)
        if r.narrative_asset is not None:
            self._cross.append(r.narrative_asset)

    def _flush(self) -> None:
        if self._month is None or not self._narr:
            return
        name = partition_file(self._month)
        _atomic_parquet(pl.concat(self._narr), self.out_dir / name)
        _atomic_parquet(pl.DataFrame(self._diag, schema=DAY_DIAGNOSTICS_SCHEMA),
                        self.diag_dir / name)
        if self._prim:
            (self.out_dir / "primitive_daily").mkdir(exist_ok=True)
            _atomic_parquet(pl.concat(self._prim), self.out_dir / "primitive_daily" / name)
        if self.att_dir is not None:
            _atomic_parquet(pl.concat(self._att), self.att_dir / name)
        if self.cross_dir is not None:
            _atomic_parquet(pl.concat(self._cross), self.cross_dir / name)
        self._narr, self._prim, self._diag, self._att, self._cross = [], [], [], [], []

    def close(self, metadata: RunMetadata) -> None:
        import json

        self._flush()
        text = json.dumps(metadata.to_dict(), indent=2, sort_keys=True, default=str) + "\n"
        for d in (self.out_dir, self.diag_dir, self.att_dir, self.cross_dir):
            if d is not None:
                (d / "run_metadata.json").write_text(text)


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
    n_pole_masked: int = 0
    n_mask_changed_retention: int = 0


_FUNNEL_KEYS = ("n_candidates", "n_f0_survivors", "n_retained_pre_jump", "n_retained",
                "n_jump_trimmed", "jump_gap_sum", "n_pole_masked", "n_mask_changed_retention")


@dataclass
class _DayState:
    narr: list[DayAccumulator]                 # one per sentiment label
    prim: list[DayAccumulator] | None
    funnels: list[_Funnel]
    assets: AssetDay | None = None
    n_total: int = 0                           # every headline of the day
    n_untagged: int = 0


@dataclass
class _Block:
    """What one block hands back besides the accumulators."""

    thresholds: np.ndarray | None = None       # per-row F0 trim threshold
    n_masked: np.ndarray | None = None         # per-row masked score count
    trip_n: np.ndarray | None = None           # per-row (narrative, score) pairs
    trip_narr: np.ndarray | None = None
    trip_score: np.ndarray | None = None


def calibration_for(calibration: CalibrationProvider, config: ScoringConfig,
                    day: date) -> tuple[ReferenceValue | None, TauRecord | None] | None:
    """(mu, tau) as of ``day``. None when there is no usable mu row (the day cannot be
    corrected, so it is never scored); tau None when no tau row is old enough yet."""
    try:
        mu = calibration.mu_for(day) if config.mode is not Correction.RAW else None
    except LookaheadError:
        return None
    try:
        tau: TauRecord | None = calibration.tau_for(day)
    except LookaheadError:
        tau = None
    return mu, tau


def _kernel_into(k: dict[str, Any], state: _DayState, tagged: bool) -> None:
    """Fold a kernel result into the day's per-label accumulators and funnels."""
    n_labels = len(state.narr)
    for t in range(n_labels):
        pick = (lambda v: v[t]) if tagged else (lambda v: v)
        state.narr[t].add_arrays(pick(k["narr_count"]), pick(k["narr_total"]),
                                 pick(k["narr_sumsq"]), pick(k["narr_peak"]))
        if state.prim is not None:
            state.prim[t].add_arrays(pick(k["prim_count"]), pick(k["prim_total"]),
                                     pick(k["prim_sumsq"]), pick(k["prim_peak"]))
        f = state.funnels[t]
        for name in _FUNNEL_KEYS:
            setattr(f, name, getattr(f, name) + pick(k[name]))
        n_un = int(pick(k["n_unassigned"]))
        state.narr[t].n_unassigned += n_un
        if state.prim is not None:
            state.prim[t].n_unassigned += n_un


def _numpy_slot(S: np.ndarray, tau32: float, n_candidates: int, config: ScoringConfig,
                prim_to_narr: np.ndarray, n_narr: int, pairs: BipolarPairs | None,
                n_trim_rows: int, narr: DayAccumulator | None, prim: DayAccumulator | None,
                funnel: _Funnel | None) -> tuple[Any, np.ndarray | None, np.ndarray | None,
                                                 tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """The numpy path over one block of rows of one label (``narr`` None: untagged rows,
    selected only for the mask, the trim and the null draws)."""
    if pairs is not None:
        ms = select_with_pole_mask(S, tau32, n_candidates, pairs, prim_to_narr,
                                   jump_cut=config.jump_cut,
                                   jump_min_candidates=config.jump_min_candidates)
        sel, n_masked = ms.selection, ms.n_masked
        if funnel is not None:
            funnel.n_pole_masked += ms.n_pole_masked
            funnel.n_mask_changed_retention += ms.n_changed_retention
    else:
        sel = select(S, tau32, n_candidates, jump_cut=config.jump_cut,
                     jump_min_candidates=config.jump_min_candidates)
        n_masked = None
    rows, nid, nval = headline_narrative_scores(
        sel.rows, sel.cols, sel.vals, prim_to_narr, n_narr, config.narrative_agg)
    if narr is not None:
        narr.add_values(nid, nval)
        narr.n_unassigned += sel.n_unassigned
        if prim is not None:
            prim.add_values(sel.cols.astype(np.int64), sel.vals)
            prim.n_unassigned += sel.n_unassigned
    thresholds = trim_threshold(S, config.trim_frac) if n_trim_rows else None
    if funnel is not None:
        funnel.n_candidates += int(sel.n_candidates.sum())
        funnel.n_f0_survivors += int(sel.n_f0_survivors.sum())
        funnel.n_retained_pre_jump += sel.n_retained_pre_jump
        funnel.n_retained += int(sel.n_retained.sum())
        funnel.n_jump_trimmed += sel.n_jump_trimmed
        funnel.jump_gap_sum += sel.jump_gap_sum
    return sel, thresholds, n_masked, (rows, nid, nval)


def _triplets_2d(n_rows: int, n_narr: int, rows: np.ndarray, nid: np.ndarray,
                 val: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(row, narrative, score) triplets in the kernel's per-row layout."""
    order = np.lexsort((nid, rows))
    rows, nid, val = rows[order].astype(np.int64), nid[order], val[order]
    trip_n = np.bincount(rows, minlength=n_rows).astype(np.int32)
    first = np.repeat(np.cumsum(trip_n) - trip_n, trip_n)
    pos = np.arange(rows.size) - first
    trip_narr = np.full((n_rows, n_narr), -1, dtype=np.int32)
    trip_score = np.full((n_rows, n_narr), np.nan, dtype=np.float64)
    trip_narr[rows, pos] = nid
    trip_score[rows, pos] = val
    return trip_n, trip_narr, trip_score


def _cat(parts: list[np.ndarray], dtype: Any) -> np.ndarray:
    return np.concatenate(parts) if parts else np.empty(0, dtype)


def _accumulate_block(
    S: np.ndarray, tau32: float, n_candidates: int, config: ScoringConfig,
    prim_to_narr: np.ndarray, n_narr: int, state: _DayState, codes: np.ndarray | None,
    use_kernel: bool, threads: int, n_trim_rows: int, pairs: BipolarPairs | None = None,
    want_triplets: bool = False,
) -> _Block:
    """Steps 3-5 for one block into the day's accumulators, each row into its label's
    (``codes``: int8 per row, -1 untagged; None: one label, every row). With ``pairs``
    the bipolar pole mask is applied to ``S`` IN PLACE first, so the caller's null draws
    come from the masked row."""
    n_head = S.shape[0]
    jump_min = config.jump_min_candidates if config.jump_cut else -1
    out = _Block()
    if pairs is not None and (S.dtype != np.float32 or not S.flags.c_contiguous):
        raise ValueError("the pole mask needs S as a C-contiguous float32 block")
    labels_n = (np.full(1, n_head, dtype=np.int64) if codes is None
                else np.bincount(codes[codes >= 0].astype(np.int64),
                                 minlength=len(state.narr)))
    if use_kernel:
        tag_kw = {} if codes is None else {"row_tag": codes, "n_tags": len(state.narr)}
        k = select_aggregate_rowwise(
            np.ascontiguousarray(S, dtype=np.float32), tau32, n_candidates, jump_min,
            prim_to_narr, n_narr, config.narrative_agg is AggRule.MEDIAN,
            state.prim is not None, threads, n_trim_rows,
            *((pairs.a_start, pairs.a_len, pairs.b_start, pairs.b_len) if pairs is not None
              else (None, None, None, None)),
            want_triplets=want_triplets, **tag_kw,
        )
        _kernel_into(k, state, codes is not None)
        if n_trim_rows:
            out.thresholds = k["trim_threshold"]
        if pairs is not None:
            out.n_masked = k["n_masked"]
        if want_triplets:
            out.trip_n, out.trip_narr, out.trip_score = (k["trip_n"], k["trip_narr"],
                                                         k["trip_score"])
    elif codes is None:                  # one label: the historical numpy call, in place
        _, out.thresholds, out.n_masked, trip = _numpy_slot(
            S, tau32, n_candidates, config, prim_to_narr, n_narr, pairs, n_trim_rows,
            state.narr[0], state.prim[0] if state.prim is not None else None,
            state.funnels[0])
        if want_triplets:
            out.trip_n, out.trip_narr, out.trip_score = _triplets_2d(n_head, n_narr, *trip)
    else:                                # per label (and untagged) on its rows, row-wise
        thresholds = np.full(n_head, np.inf, dtype=np.float32) if n_trim_rows else None
        n_masked = np.zeros(n_head, dtype=np.int64) if pairs is not None else None
        rows_all, nid_all, val_all = [], [], []
        for t in [*range(len(state.narr)), -1]:
            idx = np.flatnonzero(codes == t)
            if not idx.size:
                continue
            sub = np.ascontiguousarray(S[idx])
            tagged = t >= 0
            _, thr, msk, (r, nid, val) = _numpy_slot(
                sub, tau32, n_candidates, config, prim_to_narr, n_narr, pairs, n_trim_rows,
                state.narr[t] if tagged else None,
                state.prim[t] if tagged and state.prim is not None else None,
                state.funnels[t] if tagged else None)
            S[idx] = sub                  # the masked rows, for the null draws
            if thresholds is not None:
                thresholds[idx] = thr
            if n_masked is not None:
                n_masked[idx] = msk
            if tagged:
                rows_all.append(idx[r])
                nid_all.append(nid)
                val_all.append(val)
        out.thresholds, out.n_masked = thresholds, n_masked
        if want_triplets:
            out.trip_n, out.trip_narr, out.trip_score = _triplets_2d(
                n_head, n_narr, _cat(rows_all, np.int64), _cat(nid_all, np.int64),
                _cat(val_all, np.float64))
    for t, n in enumerate(labels_n):
        state.narr[t].n_headlines += int(n)
        if state.prim is not None:
            state.prim[t].n_headlines += int(n)
    state.n_total += n_head
    if codes is not None:
        state.n_untagged += int((codes < 0).sum())
    return out


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
    sentiment_artifact_id: str | None = None,
    embeddings_provenance: dict[str, Any] | None = None,
    assets: AssetUniverse | None = None,
) -> ScoringResult:
    """Score every day in ``dates`` (sorted, de-duplicated) with point-in-time inputs.

    Args:
        dates: calendar days to score. One day = a live run; a range = a backtest.
        config: the numerical configuration (recorded in the result's metadata); its
            ``tags`` set the sentiment labels, and the source must then give each chunk
            its rows' tag codes.
        table: the primitive table the embeddings were built from.
        primitive_embeddings: ``embed_primitive_texts`` output for ``table``.
        source: where the day's headline embeddings (tags, assets) come from.
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
        sentiment_artifact_id: the headline_sentiment artifact behind the tags; recorded in
            the metadata and in day_diagnostics.
        embeddings_provenance: what produced the primitive and headline embeddings
            (``artifacts.embeddings_provenance``); recorded in the run metadata.
        assets: the asset layer (assets.py); the source must then give each chunk its
            rows' assets.
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
    pairs = table.bipolar_pairs if config.mask_bipolar else None
    n_candidates = n_candidates_for(config.q, n_prim)
    labels = config.labels
    tagged = config.tags is not None
    sentiment_source_id = (f"{sentiment_artifact_id or config.tags.source}:{config.tags.rule}"
                           if tagged else None)
    narr_nodes, prim_nodes = table.narrative_nodes, table.primitive_nodes
    narrative_keys = narr_nodes["narrative_key"].to_list()
    n_trim_rows = n_trim(n_prim, config.trim_frac) if null_sink is not None else 0
    n_keep_rows = n_keep(n_prim, config.trim_frac)
    log.info("scoring %d day(s) [%s .. %s] mode=%s pooling=%s q=%g (k=%d) jump=%s "
             "mask_bipolar=%s (%d pairs) labels=%s assets=%s path=%s null_sink=%s",
             len(days), days[0], days[-1], config.mode.value, config.paraphrase_pooling.value,
             config.q, n_candidates, config.jump_cut, config.mask_bipolar,
             pairs.n_pairs if pairs is not None else 0, labels,
             assets.n_assets if assets is not None else None,
             "kernel" if use_kernel else "numpy", null_sink is not None)

    matrix_cache: dict[date | None, np.ndarray] = {}

    def scoring_matrix_for(mu: ReferenceValue | None) -> np.ndarray:
        key = mu.date if mu is not None else None
        if key not in matrix_cache:
            matrix_cache.clear()          # one live matrix at a time; mu changes monotonically
            matrix_cache[key] = scoring_matrix(
                primitive_embeddings, table, config.mode, config.paraphrase_pooling,
                mu.mu if mu else None, mu.mu_hat if mu else None)
        return matrix_cache[key]

    narr_frames: list[pl.DataFrame] = []
    prim_frames: list[pl.DataFrame] = []
    att_frames: list[pl.DataFrame] = []
    cross_frames: list[pl.DataFrame] = []
    diags: list[dict[str, Any]] = []
    seen_days: set[date] = set()
    null_only_days: list[date] = []
    peak_rss = 0.0
    first_tau: TauRecord | None = None
    current: dict[str, Any] | None = None

    def open_day(day: date) -> dict[str, Any] | None:
        cal = calibration_for(calibration, config, day)
        if cal is None:
            if tau_missing != "null_only":
                calibration.mu_for(day)                  # re-raises the LookaheadError
            log.info("%s: no mu_asof row yet; day skipped entirely (nothing to correct with)",
                     day)
            return None
        mu, tau_rec = cal
        if tau_rec is None:
            if tau_missing != "null_only" or null_sink is None:
                calibration.tau_for(day)                 # re-raises the LookaheadError
            null_only_days.append(day)
            log.info("%s: no tau row old enough; null-only day (partitions fed, no scores)", day)
        return {
            "day": day, "t0": time.perf_counter(), "rss_before": rss_gb(),
            "mu": mu, "tau_rec": tau_rec,
            # Cast once so the numpy path and the kernel compare float32 to float32.
            "tau32": float(np.float32(tau_rec.tau)) if tau_rec is not None else None,
            "P_scoring": scoring_matrix_for(mu),
            "state": _DayState(
                narr=[DayAccumulator(n_narr) for _ in labels],
                prim=[DayAccumulator(n_prim) for _ in labels] if keep_primitive_daily else None,
                funnels=[_Funnel() for _ in labels],
                assets=(AssetDay(len(labels), n_narr, assets.n_assets, assets.min_relevance)
                        if assets is not None else None),
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
        n_head = state.n_total                   # the attention denominator, every label
        in_universe, n_unmapped = (assets.as_of(day) if assets is not None else (None, None))
        nd, pd_, diag_rows = [], [], []
        seconds = time.perf_counter() - ctx["t0"]
        for t, label in enumerate(labels):
            acc, f = state.narr[t], state.funnels[t]
            n_scored = acc.n_headlines
            nd.append(acc.frame(narr_nodes, day, sentiment=label, n_headlines=n_head).select(
                list(NARRATIVE_DAILY_SCHEMA.keys())))
            if state.prim is not None:
                pd_.append(state.prim[t].frame(prim_nodes, day, sentiment=label,
                                               n_headlines=n_head).select(
                    list(PRIMITIVE_DAILY_SCHEMA.keys())))
            diag_rows.append({
                "DATE": day, "SENTIMENT": label, "N_HEADLINES": n_head, "N_SCORED": n_scored,
                "N_UNTAGGED": state.n_untagged, "N_UNASSIGNED": acc.n_unassigned,
                "N_F0_SURVIVORS_PRE_Q": f.n_f0_survivors, "N_Q_CANDIDATES": f.n_candidates,
                "N_RETAINED_PRE_JUMP": f.n_retained_pre_jump,
                "N_RETAINED_POST_Q_TAU": f.n_retained,
                "MEAN_RETAINED_PER_HEADLINE": f.n_retained / max(n_scored, 1),
                "PCT_TAU_PRUNED_WITHIN_Q": 1.0 - f.n_retained_pre_jump / max(f.n_candidates, 1),
                "PCT_JUMP_APPLIED": f.n_jump_trimmed / max(n_scored, 1),
                "MEAN_JUMP_GAP": (f.jump_gap_sum / f.n_jump_trimmed) if f.n_jump_trimmed
                else None,
                "NARRATIVES_TOUCHED": int((acc.count > 0).sum()),
                "N_POLE_MASKED": f.n_pole_masked if pairs is not None else None,
                "N_MASK_CHANGED_RETENTION": (f.n_mask_changed_retention if pairs is not None
                                             else None),
                "N_UNMAPPED": n_unmapped,
                "MU_DATE": mu.date if mu else None, "MU_NORM": mu.norm if mu else None,
                "TAU": ctx["tau32"], "TAU_MONTH_END": tau_rec.month_end,
                "N_EFF": tau_rec.n_eff,
                "CONFIG_ID": config.digest(), "F0_CONFIG_ID": config.f0_digest(),
                "TAU_SOURCE_ID": calibration.tau_source_id,
                "MU_ASOF_ID": calibration.mu_asof_id,
                "SENTIMENT_SOURCE_ID": sentiment_source_id,
                "RSS_GB_BEFORE": rss_before, "RSS_GB_AFTER": rss_after, "SECONDS": seconds,
            })
        att = cross = None
        if state.assets is not None:
            att, cross = state.assets.frames(
                day, labels, narrative_keys, assets.entity_ids, in_universe, n_head,
                [a.n_headlines for a in state.narr])
        nd_frame = pl.concat(nd)
        pd_frame = pl.concat(pd_) if pd_ else None
        log.info("%s: %d headlines (%s), %d untagged, RSS %.2f -> %.2f GB, %.1fs", day, n_head,
                 ", ".join(f"{lab} {a.n_headlines}" for lab, a in zip(labels, state.narr)),
                 state.n_untagged, rss_before, rss_after, seconds)
        result = DayResult(day, nd_frame, pd_frame, diag_rows, att, cross)
        if writer is not None:
            writer.write_day(result)
        if collect:
            narr_frames.append(nd_frame)
            if pd_frame is not None:
                prim_frames.append(pd_frame)
            if att is not None:
                att_frames.append(att)
                cross_frames.append(cross)
        diags.extend(diag_rows)
        if null_sink is not None:
            null_sink.close_day(day, rss_after)

    unusable: set[date] = set()
    want_triplets = assets is not None
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
        if tagged and chunk.tags is None:
            raise ValueError(f"{day}: the config has sentiment tags but the source gives none")
        if assets is not None and chunk.asset_indptr is None:
            raise ValueError(f"{day}: an asset layer was given but the source gives no assets")
        mu, state = current["mu"], current["state"]
        X = chunk.embeddings
        H, _ = config.mode.correct(X, mu, as_of=day)     # raises on a future-dated mu
        S = primitive_scores(H, current["P_scoring"], n_prim, table.n_texts,
                             config.paraphrase_pooling)
        if current["tau32"] is None:             # null-only day
            n_masked = apply_pole_mask(S, pairs) if pairs is not None else None
            draws = sample_null_draws(S, trim_threshold(S, config.trim_frac),
                                      config.null_draws_per_headline, null_sink.rng_for(day),
                                      n_masked=n_masked)
            null_sink.add_draws(day, draws, _n_available(S.shape[0], n_keep_rows, n_masked),
                                S.shape[0])
            del H, S, X, chunk
            continue
        codes = chunk.tags if tagged else None
        blk = _accumulate_block(
            S, current["tau32"], n_candidates, config, prim_to_narr, n_narr, state, codes,
            use_kernel, threads, n_trim_rows, pairs, want_triplets)
        if state.assets is not None:
            label = codes if codes is not None else np.zeros(chunk.n, dtype=np.int8)
            state.assets.add_attention(label, chunk.asset_indptr, chunk.asset_idx,
                                       chunk.asset_rel)
            state.assets.add_cross(label, blk.trip_n, blk.trip_narr, blk.trip_score,
                                   chunk.asset_indptr, chunk.asset_idx, chunk.asset_rel)
        if null_sink is not None:
            draws = sample_null_draws(S, blk.thresholds, config.null_draws_per_headline,
                                      null_sink.rng_for(day), n_masked=blk.n_masked)
            null_sink.add_draws(day, draws, _n_available(S.shape[0], n_keep_rows, blk.n_masked),
                                S.shape[0])
        del H, S, X, chunk, blk
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
        sentiment_artifact_id=sentiment_artifact_id,
        embeddings_provenance=embeddings_provenance,
        config_id=config.digest(), f0_config_id=config.f0_digest(),
        extra={"source": source.describe(), "scoring_path": "kernel" if use_kernel else "numpy",
               "first_tau_record": first_tau.to_dict(),
               **({"assets": assets.params()} if assets is not None else {}),
               **(extra_metadata or {})},
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
        asset_attention=pl.concat(att_frames) if att_frames else None,
        narrative_asset=pl.concat(cross_frames) if cross_frames else None,
    )


def _n_available(n_head: int, n_keep_rows: int, n_masked: np.ndarray | None) -> int:
    """Null draws available in a block: n_keep per headline, minus the masked scores (all
    below the trim threshold, since sample_null_draws checks the threshold is finite)."""
    return n_head * n_keep_rows - (int(n_masked.sum()) if n_masked is not None else 0)


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
    """Day-ordered (day, chunk) stream over ``days``; a day without headlines yields
    nothing."""
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
