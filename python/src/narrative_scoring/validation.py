"""Dense reference implementations, derived views and summaries. Not production scoring.

``reference_select`` and ``reference_day`` are the slowest, most literal
transcription of the selection and aggregation rules; the vectorised numpy
path and the compiled kernel are asserted against them in the test suite.
``attention`` is the downstream Sadka-style measure, derived here and never
stored by the scorer; ``sum_tags`` adds the sentiment labels of one run into its
all-tagged-headlines panel, without rescoring; ``merge_poles`` adds the two pole rows of
every bipolar pair into one narrative row (exact on a mask_bipolar run).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from narrative_scoring.config import SENTIMENT_ALL, AggRule


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


def sum_tags(frame: pl.DataFrame, diagnostics: pl.DataFrame | None = None) -> pl.DataFrame:
    """One run's sentiment labels added into one panel (SENTIMENT = "all"). Pure; never
    rescores. ``frame`` holds the narrative_daily (or primitive_daily) rows of ONE run, every
    label of a day sharing N_HEADLINES (checked). Per key (DATE, node), with n_i = SUPPORT_i,
    T_i = TOTAL_SCORE_i and Q_i = n_i (STD_i^2 + INTENSITY_i^2) (the stored ddof=0
    statistics)::

        SUPPORT     = sum n_i
        TOTAL_SCORE = sum T_i
        INTENSITY   = sum T_i / sum n_i
        STD_SCORE   = sqrt((sum Q_i - (sum T_i)^2 / sum n_i) / sum n_i)      (ddof=0)
        PEAK        = max PEAK_i
        N_LABELLED  = sum over labels of the label's headlines that day
        N_HEADLINES = the day's total (every label and untagged)

    The labels are disjoint, so the result is the panel of the TAGGED headlines: it equals
    an untagged run's exactly when no headline was untagged. With ``diagnostics`` (the
    run's day_diagnostics) the check N_LABELLED + N_UNTAGGED = N_HEADLINES is enforced, so
    the gap is always explicit."""
    if frame.is_empty():
        return frame
    cols = frame.columns
    heads = frame.group_by("DATE").agg(pl.col("N_HEADLINES").n_unique().alias("n"))
    if (heads["n"] > 1).any():
        raise ValueError("N_HEADLINES differs across the labels of a day: not one run")
    key_cols = [c for c in cols if c not in _STAT_COLUMNS and c != "SENTIMENT"]
    labelled = (frame.group_by("DATE", "SENTIMENT").agg(pl.col("N_LABELLED").max())
                .group_by("DATE").agg(pl.col("N_LABELLED").sum()))
    if diagnostics is not None:
        untagged = diagnostics.group_by("DATE").agg(pl.col("N_UNTAGGED").first(),
                                                    pl.col("N_HEADLINES").first())
        chk = labelled.join(untagged, on="DATE", how="inner")
        bad = chk.filter(pl.col("N_LABELLED") + pl.col("N_UNTAGGED") != pl.col("N_HEADLINES"))
        if bad.height:
            raise ValueError(f"N_LABELLED + N_UNTAGGED != N_HEADLINES on {bad.height} day(s), "
                             f"e.g. {bad.sort('DATE').head(3).to_dicts()}")
    agg = (
        frame.with_columns(*_moment_columns())
        .group_by(key_cols, maintain_order=True)
        .agg(
            pl.col("SUPPORT").sum().alias("SUPPORT"),
            pl.col("_total").sum().alias("_total"),
            pl.col("_sumsq").sum().alias("_sumsq"),
            pl.col("PEAK").max().alias("PEAK"),
            pl.col("N_HEADLINES").first().alias("N_HEADLINES"),
        )
        .join(labelled, on="DATE", how="left")
    )
    out = _recombined(agg).with_columns(pl.lit(SENTIMENT_ALL).alias("SENTIMENT"))
    schema = frame.schema
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
    squares, the same arithmetic as ``sum_tags``::

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
    per_day = diag.unique("DATE", keep="first")          # one row per label: count days once
    row["n_days"] = per_day.height
    row["n_headlines"] = int(per_day["N_HEADLINES"].sum())
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
