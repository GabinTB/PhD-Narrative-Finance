"""Dense reference implementations, derived views and summaries. Not production scoring.

``reference_select`` and ``reference_day`` are the slowest, most literal
transcription of the selection and aggregation rules; the vectorised numpy
path and the compiled kernel are asserted against them in the test suite.
``attention`` is the downstream Sadka-style measure, derived here and never
stored by the scorer; ``combine_sentiment`` re-aggregates stored rows across
sentiment labels (all, neg-only, neu+pos, ...) without rescoring.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import polars as pl

from narrative_scoring.config import AggRule


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


def combine_sentiment(df: pl.DataFrame, labels: Sequence[str]) -> pl.DataFrame:
    """Post-hoc re-aggregation of stored rows across SENTIMENT labels. Pure; never rescore.

    Per (DATE, node) over the requested labels, with n_i = SUPPORT_i,
    T_i = TOTAL_SCORE_i and Q_i = SUMSQ_i = n_i (STD_i^2 + INTENSITY_i^2)
    (recovered from the stored ddof=0 statistics)::

        SUPPORT     = sum n_i
        TOTAL_SCORE = sum T_i
        INTENSITY   = sum T_i / sum n_i
        STD_SCORE   = sqrt((sum Q_i - (sum T_i)^2 / sum n_i) / sum n_i)      (ddof=0)
        PEAK        = max PEAK_i
        N_LABELLED  = sum N_LABELLED_i ;  N_HEADLINES unchanged (the day's total)

    Combining every stored label reproduces the "all" row, up to float32
    rounding of the stored INTENSITY / STD_SCORE (tested). The result carries
    SENTIMENT = "+".join(labels). Works for narrative_daily and primitive_daily.
    """
    labels = list(labels)
    if not labels:
        raise ValueError("no labels")
    key_cols = [c for c in df.columns if c not in _STAT_COLUMNS and c != "SENTIMENT"]
    sub = df.filter(pl.col("SENTIMENT").is_in(labels))
    missing = set(labels) - set(sub["SENTIMENT"].unique().to_list())
    if missing:
        raise ValueError(f"labels not present in the frame: {sorted(missing)}")
    n = pl.col("SUPPORT").cast(pl.Float64)
    sumsq = n * (pl.col("STD_SCORE").cast(pl.Float64) ** 2
                 + pl.col("INTENSITY").cast(pl.Float64) ** 2)
    agg = (
        sub.with_columns(sumsq.fill_null(0.0).alias("_sumsq"),
                         pl.col("TOTAL_SCORE").fill_null(0.0).alias("_total"))
        .group_by(key_cols, maintain_order=True)
        .agg(
            pl.col("SUPPORT").sum().alias("SUPPORT"),
            pl.col("_total").sum().alias("_total"),
            pl.col("_sumsq").sum().alias("_sumsq"),
            pl.col("PEAK").max().alias("PEAK"),
            pl.col("N_HEADLINES").first().alias("N_HEADLINES"),
            pl.col("N_LABELLED").sum().alias("N_LABELLED"),
        )
    )
    n = pl.col("SUPPORT").cast(pl.Float64)
    has = pl.col("SUPPORT") > 0
    out = agg.with_columns(
        pl.lit("+".join(labels)).alias("SENTIMENT"),
        pl.when(has).then(pl.col("_total")).otherwise(None).alias("TOTAL_SCORE"),
        pl.when(has).then(pl.col("_total") / n).otherwise(None).cast(pl.Float32)
        .alias("INTENSITY"),
        pl.when(has)
        .then(((pl.col("_sumsq") - pl.col("_total") ** 2 / n) / n).clip(lower_bound=0.0).sqrt())
        .otherwise(None).cast(pl.Float32).alias("STD_SCORE"),
    ).drop(["_total", "_sumsq"])
    return out.select([c for c in df.columns if c in out.columns])


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
    nd = result.narrative_daily.filter(pl.col("SENTIMENT") == "all")
    row["n_days"] = diag.height
    row["n_headlines"] = int(diag["N_HEADLINES"].sum())
    n_head = max(row["n_headlines"], 1)
    row["unassigned_share"] = float(diag["N_UNASSIGNED"].sum()) / n_head
    row["f0_survivors_per_headline"] = float(diag["N_F0_SURVIVORS_PRE_Q"].sum()) / n_head
    row["candidates_per_headline"] = float(diag["N_Q_CANDIDATES"].sum()) / n_head
    row["retained_per_headline"] = float(diag["N_RETAINED_POST_Q_TAU"].sum()) / n_head
    row["tau_pruned_within_q"] = 1.0 - float(diag["N_RETAINED_PRE_JUMP"].sum()) / max(
        float(diag["N_Q_CANDIDATES"].sum()), 1.0)
    jumped = (diag["PCT_JUMP_APPLIED"] * diag["N_HEADLINES"]).sum()
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
