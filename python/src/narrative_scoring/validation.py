"""Dense reference implementations, derived views and summaries. Not production scoring.

``reference_select`` and ``reference_day`` are the slowest, most literal
transcription of the selection and aggregation rules; the vectorised numpy
path and the compiled kernel are asserted against them in the test suite.
``attention`` is the downstream Sadka-style measure, derived here and never
stored by the scorer; ``sum_sentiment_runs`` adds the four sentiment-bucket runs of
one setup back into the all-headlines panel, without rescoring; ``merge_poles`` adds the
two pole rows of every bipolar pair into one narrative row (exact on a mask_bipolar run).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import polars as pl

from narrative_scoring.config import (
    SENTIMENT_ALL,
    SENTIMENT_BUCKETS,
    AggRule,
    ScoringConfig,
)


def reference_select(
    S: np.ndarray, tau: float, n_candidates: int, *, jump_cut: bool = False,
    jump_min_candidates: int = 10,
) -> np.ndarray:
    """(n_head, n_prim) with the retained scores and NaN everywhere else."""
    S = np.asarray(S, dtype=np.float32)
    tau32 = np.float32(tau)
    out = np.full(S.shape, np.nan, dtype=np.float32)
    for i, row in enumerate(S):
        kth = np.sort(row)[::-1][n_candidates - 1]          # k-th largest of the FULL row
        candidates = np.flatnonzero(row >= kth)
        retained = [j for j in candidates if row[j] >= tau32]
        if jump_cut and len(retained) > jump_min_candidates:
            retained = sorted(retained, key=lambda j: -row[j])
            vals = np.asarray([row[j] for j in retained], dtype=np.float64)
            gaps = vals[:-1] - vals[1:]
            if gaps.max() > 0.0:
                cut = int(np.argmax(gaps))                    # first largest gap
                retained = retained[:cut + 1]
        for j in retained:
            out[i, j] = row[j]
    return out


def reference_headline_narrative(
    A: np.ndarray, prim_to_narr: np.ndarray, n_narr: int, rule: AggRule,
) -> np.ndarray:
    """(n_head, n_narr) headline x narrative scores from a NaN-masked primitive matrix."""
    out = np.full((A.shape[0], n_narr), np.nan, dtype=np.float64)
    for i, row in enumerate(A):
        for j in range(n_narr):
            blk = row[prim_to_narr == j]
            blk = blk[np.isfinite(blk)].astype(np.float64)
            if blk.size:
                out[i, j] = np.median(blk) if rule is AggRule.MEDIAN else blk.mean()
    return out


def reference_day(N: np.ndarray) -> dict[str, np.ndarray]:
    """Day statistics from a (n_head, n_nodes) NaN-masked headline x node matrix."""
    has = np.isfinite(N)
    cnt = has.sum(axis=0)
    stats: dict[str, np.ndarray] = {"SUPPORT": cnt}
    with np.errstate(invalid="ignore"):
        stats["TOTAL_SCORE"] = np.where(cnt > 0, np.nansum(N, axis=0), np.nan)
        stats["INTENSITY"] = np.where(cnt > 0, np.nanmean(N, axis=0), np.nan)
        stats["STD_SCORE"] = np.where(cnt > 0, np.nanstd(N, axis=0, ddof=0), np.nan)
        stats["PEAK"] = np.where(cnt > 0, np.nanmax(N, axis=0), np.nan)
    return stats


@dataclass
class SentimentRun:
    """One sentiment-bucket run, as ``sum_sentiment_runs`` needs it: its rows and the
    identity of what produced them (from its run metadata)."""

    frame: pl.DataFrame               # narrative_daily (or primitive_daily) rows
    config: ScoringConfig
    sentiment_artifact_id: str | None
    tau_asof_id: str | None
    mu_asof_id: str | None
    name: str = ""

    @classmethod
    def from_result(cls, result: Any, *, primitive: bool = False) -> SentimentRun:
        """From a ``pipeline.ScoringResult`` (its ``metadata`` must be set)."""
        m = result.metadata
        frame = result.primitive_daily if primitive else result.narrative_daily
        return cls(frame, ScoringConfig.from_dict(m.config), m.sentiment_artifact_id,
                   m.tau_source_id, m.mu_asof_id, name=str(m.config.get("sentiment")))

    @classmethod
    def from_artifact(cls, artifact: Any, *, primitive: bool = False) -> SentimentRun:
        """From a narrative_daily artifact: its partition files (or primitive_daily/) and
        its ``run_metadata.json``."""
        path = Path(artifact.path)
        m = json.loads((path / "run_metadata.json").read_text())
        files = sorted((path / "primitive_daily" if primitive else path).glob("*.parquet"))
        frame = pl.concat([pl.read_parquet(f) for f in files]) if files else pl.DataFrame()
        return cls(frame, ScoringConfig.from_dict(m["config"]), m.get("sentiment_artifact_id"),
                   m.get("tau_source_id"), m.get("mu_asof_id"), name=artifact.artifact_id)


def check_sentiment_runs(runs: Sequence[SentimentRun]) -> None:
    """Raise unless ``runs`` are the four buckets of ONE setup: each bucket exactly once,
    the same sentiment artifact / source / rule / thresholds, the same config apart from
    the bucket, and the same tau_asof and mu_asof. (A FinBERT negative run and a RavenBERT
    positive run must not add up silently.)"""
    buckets = [r.config.sentiment for r in runs]
    if sorted(b.value for b in buckets) != sorted(b.value for b in SENTIMENT_BUCKETS):
        raise ValueError(f"need each of {[b.value for b in SENTIMENT_BUCKETS]} exactly once, "
                         f"got {[b.value for b in buckets]}")

    def same(what: str, values: list[Any]) -> None:
        if len({json.dumps(v, sort_keys=True, default=str) for v in values}) != 1:
            raise ValueError(f"the runs differ in {what}: "
                             + ", ".join(f"{r.name or r.config.sentiment.value}={v!r}"
                                         for r, v in zip(runs, values)))

    same("sentiment artifact", [r.sentiment_artifact_id for r in runs])
    same("mask_bipolar", [r.config.mask_bipolar for r in runs])
    for field_name in ("sentiment_source", "sentiment_rule", "neg_max", "pos_min", "min_conf"):
        same(field_name, [getattr(r.config, field_name) for r in runs])
    same("config (bucket removed)", [r.config.digest_without_filter() for r in runs])
    same("tau_asof", [r.tau_asof_id for r in runs])
    same("mu_asof", [r.mu_asof_id for r in runs])
    days = [frozenset(r.frame["DATE"].unique().to_list()) for r in runs]
    if len(set(days)) != 1:
        raise ValueError("the runs cover different day sets: "
                         + ", ".join(f"{r.config.sentiment.value}: {len(d)} day(s)"
                                     for r, d in zip(runs, days)))
    totals = (pl.concat([r.frame.select("DATE", "N_HEADLINES").unique()
                         .with_columns(pl.lit(i).alias("_run")) for i, r in enumerate(runs)])
              .group_by("DATE").agg(pl.col("N_HEADLINES").n_unique().alias("n"),
                                    pl.col("N_HEADLINES").unique().alias("values")))
    bad = totals.filter(pl.col("n") > 1)
    if bad.height:
        raise ValueError(f"N_HEADLINES differs across the runs on {bad.height} day(s), e.g. "
                         f"{bad.sort('DATE').head(3).to_dicts()}")


def sum_sentiment_runs(runs: Sequence[SentimentRun]) -> pl.DataFrame:
    """The all-headlines panel rebuilt from its four sentiment-bucket runs. Pure; never
    rescores. ``check_sentiment_runs`` first; then an OUTER combination on the key columns
    (DATE, node): a row missing from a run, or with null statistics, counts as SUPPORT 0,
    TOTAL 0. Per key, with n_i = SUPPORT_i, T_i = TOTAL_SCORE_i and
    Q_i = n_i (STD_i^2 + INTENSITY_i^2) (recovered from the stored ddof=0 statistics)::

        SUPPORT     = sum n_i
        TOTAL_SCORE = sum T_i
        INTENSITY   = sum T_i / sum n_i
        STD_SCORE   = sqrt((sum Q_i - (sum T_i)^2 / sum n_i) / sum n_i)      (ddof=0)
        PEAK        = max PEAK_i
        N_LABELLED  = sum over runs of the run's headlines that day (a per-day count, taken
                      from any of the run's rows that day, so a missing row loses nothing)
        N_HEADLINES = the day's total (checked equal across runs)

    The result carries SENTIMENT = "all". It is NOT compared against the all-headlines
    run: a headline's float32 scores depend slightly (~1e-7) on the rows batched with it,
    so a score at the tau / top-k boundary can be retained in one run and not in the
    other (accepted drift, owner decision). Exact equality holds on the toy lakes of the
    test suite, where it is asserted.
    """
    runs = list(runs)
    check_sentiment_runs(runs)
    frames = [r.frame for r in runs]
    cols = frames[0].columns
    key_cols = [c for c in cols if c not in _STAT_COLUMNS and c != "SENTIMENT"]
    labelled = (pl.concat([f.select("DATE", "N_LABELLED").with_columns(pl.lit(i).alias("_run"))
                           for i, f in enumerate(frames)])
                .group_by("_run", "DATE").agg(pl.col("N_LABELLED").max())
                .group_by("DATE").agg(pl.col("N_LABELLED").sum()))
    agg = (
        pl.concat([f.select(cols) for f in frames])
        .with_columns(*_moment_columns())
        .group_by(key_cols, maintain_order=True)
        .agg(
            pl.col("SUPPORT").sum().alias("SUPPORT"),
            pl.col("_total").sum().alias("_total"),
            pl.col("_sumsq").sum().alias("_sumsq"),
            pl.col("PEAK").max().alias("PEAK"),
            pl.col("N_HEADLINES").drop_nulls().first().alias("N_HEADLINES"),
        )
        .join(labelled, on="DATE", how="left")
    )
    out = _recombined(agg).with_columns(pl.lit(SENTIMENT_ALL).alias("SENTIMENT"))
    schema = frames[0].schema
    return out.select([pl.col(c).cast(schema[c]) for c in cols])


def _moment_columns() -> list[pl.Expr]:
    """Per row: the recoverable sums ``_total`` = TOTAL_SCORE and ``_sumsq`` =
    n (STD^2 + INTENSITY^2) (from the stored ddof=0 statistics), 0 where SUPPORT is 0."""
    n = pl.col("SUPPORT").cast(pl.Float64)
    sumsq = n * (pl.col("STD_SCORE").cast(pl.Float64) ** 2
                 + pl.col("INTENSITY").cast(pl.Float64) ** 2)
    return [sumsq.fill_null(0.0).alias("_sumsq"),
            pl.col("TOTAL_SCORE").fill_null(0.0).alias("_total"),
            pl.col("SUPPORT").fill_null(0)]


def _recombined(agg: pl.DataFrame) -> pl.DataFrame:
    """TOTAL_SCORE / INTENSITY / STD_SCORE (ddof=0) from summed SUPPORT, _total, _sumsq;
    null where SUPPORT is 0."""
    n = pl.col("SUPPORT").cast(pl.Float64)
    has = pl.col("SUPPORT") > 0
    return agg.with_columns(
        pl.when(has).then(pl.col("_total")).otherwise(None).alias("TOTAL_SCORE"),
        pl.when(has).then(pl.col("_total") / n).otherwise(None).cast(pl.Float32)
        .alias("INTENSITY"),
        pl.when(has)
        .then(((pl.col("_sumsq") - pl.col("_total") ** 2 / n) / n).clip(lower_bound=0.0).sqrt())
        .otherwise(None).cast(pl.Float32).alias("STD_SCORE"),
    ).drop(["_total", "_sumsq"])


_PAIR_KEY = ["reservoir", "dimension", "TYPE"]


def merge_poles(narrative_daily: pl.DataFrame) -> pl.DataFrame:
    """The narrative-level view: the two pole rows of every bipolar pair added into one row.
    Pure; never rescores. A pair is a (reservoir, dimension, TYPE) group holding exactly two
    distinct signed poles; monopolar (one signed pole) and unsigned rows pass through
    unchanged. Per (DATE, SENTIMENT, reservoir, dimension, TYPE), with the recovered sums of
    squares, the same arithmetic as ``sum_sentiment_runs``::

        SUPPORT, TOTAL_SCORE    summed
        INTENSITY, STD_SCORE    recombined (ddof=0)
        PEAK                    max
        N_HEADLINES, N_LABELLED the day's counts (equal on both pole rows; checked)
        n_primitives            summed
        narrative = TYPE, pole = "", narrative_key = reservoir|dimension|TYPE

    Exact only on a ``mask_bipolar`` run: there a headline retains at most one pole of a
    pair, so SUPPORT adds up. On an unmasked run a headline hitting both poles is counted
    twice and the sum overstates the narrative (the reason the mask exists).
    """
    if "TYPE" not in narrative_daily.columns:
        raise ValueError("narrative_daily has no TYPE column (written before it existed, e.g. "
                         "v2.2.0); its pairs cannot be identified")
    cols = narrative_daily.columns
    schema = narrative_daily.schema
    pairs = (narrative_daily.filter(pl.col("pole").fill_null("") != "")
             .group_by(_PAIR_KEY).agg(pl.col("pole").n_unique().alias("_n_poles"))
             .filter(pl.col("_n_poles") == 2).drop("_n_poles"))
    rest = narrative_daily.join(pairs, on=_PAIR_KEY, how="anti")
    pair_rows = narrative_daily.join(pairs, on=_PAIR_KEY, how="semi")
    merged = (
        pair_rows.with_columns(*_moment_columns())
        .group_by(["DATE", "SENTIMENT", *_PAIR_KEY], maintain_order=True)
        .agg(
            pl.col("SUPPORT").sum(), pl.col("_total").sum(), pl.col("_sumsq").sum(),
            pl.col("PEAK").max(), pl.col("n_primitives").sum(),
            pl.col("N_HEADLINES").first(), pl.col("N_LABELLED").first(),
            (pl.col("N_HEADLINES").n_unique() + pl.col("N_LABELLED").n_unique())
            .alias("_n_counts"),
            pl.len().alias("_n_rows"),
        )
    )
    bad = merged.filter((pl.col("_n_counts") != 2) | (pl.col("_n_rows") != 2))
    if bad.height:
        raise ValueError(f"{bad.height} pair-day(s) without exactly two pole rows sharing "
                         f"N_HEADLINES / N_LABELLED, e.g. {bad.head(3).to_dicts()}")
    merged = _recombined(merged.drop(["_n_counts", "_n_rows"])).with_columns(
        pl.col("TYPE").alias("narrative"), pl.lit("").alias("pole"),
        pl.concat_str([pl.col(c) for c in _PAIR_KEY], separator="|").alias("narrative_key"),
    )
    out = pl.concat([rest.select(cols),
                     merged.select([pl.col(c).cast(schema[c]) for c in cols])])
    return out.sort(["DATE", "SENTIMENT", "narrative_key"], maintain_order=True)


_STAT_COLUMNS = {"SUPPORT", "TOTAL_SCORE", "INTENSITY", "STD_SCORE", "PEAK", "N_HEADLINES",
                 "N_LABELLED"}


def attention(narrative_daily: pl.DataFrame) -> pl.DataFrame:
    """Derived, never stored: ATTENTION = TOTAL_SCORE / N_HEADLINES (null where SUPPORT == 0)."""
    return narrative_daily.with_columns(
        (pl.col("TOTAL_SCORE") / pl.col("N_HEADLINES")).alias("ATTENTION")
    )


def summarize(result: Any, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """One comparison row per run, every config field always present."""
    row: dict[str, Any] = result.metadata.flat()
    diag = result.day_diagnostics
    nd = result.narrative_daily
    row["n_days"] = diag.height
    row["n_headlines"] = int(diag["N_HEADLINES"].sum())
    row["n_scored"] = int(diag["N_SCORED"].sum())
    n_head = max(row["n_scored"], 1)            # funnel rates are per scored headline
    row["unassigned_share"] = float(diag["N_UNASSIGNED"].sum()) / n_head
    row["f0_survivors_per_headline"] = float(diag["N_F0_SURVIVORS_PRE_Q"].sum()) / n_head
    row["candidates_per_headline"] = float(diag["N_Q_CANDIDATES"].sum()) / n_head
    row["retained_per_headline"] = float(diag["N_RETAINED_POST_Q_TAU"].sum()) / n_head
    row["tau_pruned_within_q"] = 1.0 - float(diag["N_RETAINED_PRE_JUMP"].sum()) / max(
        float(diag["N_Q_CANDIDATES"].sum()), 1.0)
    jumped = (diag["PCT_JUMP_APPLIED"] * diag["N_SCORED"]).sum()
    row["jump_applied_share"] = float(jumped) / n_head
    row["mean_narratives_touched"] = float(diag["NARRATIVES_TOUCHED"].mean())
    row["mean_support"] = float(nd["SUPPORT"].mean()) if nd.height else float("nan")
    row["mean_intensity_present"] = (
        float(nd["INTENSITY"].drop_nulls().mean() or float("nan")) if nd.height else float("nan")
    )
    row["narrative_day_rows"] = nd.height
    row["peak_rss_gb"] = result.peak_rss_gb
    row.update(extra or {})
    return row
