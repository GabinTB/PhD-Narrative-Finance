"""Aggregation: retained primitives -> headline x narrative -> day x narrative.

Headline level (spec step 5, first hop). For each headline, its retained
primitives are grouped by narrative and pooled with a NaN-skipping mean
(default) or median. A narrative with no retained primitive for that
headline is simply absent -- never zero.

Day level (owner ruling R1: the storable artifact is day x narrative). Over
the day's headline x narrative scores, per narrative:

    SUPPORT      = number of headlines with a score
    TOTAL_SCORE  = sum of those scores
    INTENSITY    = TOTAL_SCORE / SUPPORT                      (mean)
    STD_SCORE    = sqrt(sum(x^2)/n - mean^2)                  (POPULATION std, ddof=0)
    PEAK         = max score
    N_HEADLINES  = headlines scored that day (all narratives share it)

A narrative with SUPPORT = 0 has null TOTAL/INTENSITY/STD/PEAK. STD_SCORE
with one observation is 0.0 (population std of a single value), not null.
The downstream attention measure is ATTENTION = TOTAL_SCORE / N_HEADLINES;
it is NOT stored here (validation.attention derives it).

Accumulators hold count / sum / sum-of-squares / max in float64, which are
associative, so the day aggregate is independent of how headlines were
chunked or in which order chunks arrived (exactly for count and max, to
float64 rounding for the sums). The same accumulator serves the optional
primitive-grain diagnostics.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl

from narrative_scoring.config import AggRule


def headline_narrative_scores(
    rows: np.ndarray, cols: np.ndarray, vals: np.ndarray,
    prim_to_narr: np.ndarray, n_narr: int, rule: AggRule,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pool each headline's retained primitives into narratives.

    Inputs are the retained triplets of a block. Returns (headline_row,
    narrative_id, score) in float64 for every (headline, narrative) with at
    least one retained primitive. Scores are computed in float64 from the
    float32 primitive scores so the numpy path and the compiled kernel agree
    to rounding of the final division only.
    """
    if rows.size == 0:
        e = np.empty(0, dtype=np.int64)
        return e, e, np.empty(0, dtype=np.float64)
    nid = prim_to_narr[cols].astype(np.int64)
    key = rows.astype(np.int64) * n_narr + nid
    v64 = vals.astype(np.float64)
    if rule is AggRule.MEAN:
        order = np.argsort(key, kind="stable")
        key_s, val_s = key[order], v64[order]
        starts = np.flatnonzero(np.r_[True, key_s[1:] != key_s[:-1]])
        lens = np.diff(np.r_[starts, key_s.size])
        score = np.add.reduceat(val_s, starts) / lens
    elif rule is AggRule.MEDIAN:
        order = np.lexsort((v64, key))                  # by key, then value ascending
        key_s, val_s = key[order], v64[order]
        starts = np.flatnonzero(np.r_[True, key_s[1:] != key_s[:-1]])
        lens = np.diff(np.r_[starts, key_s.size])
        lo = starts + (lens - 1) // 2
        hi = starts + lens // 2
        score = 0.5 * (val_s[lo] + val_s[hi])
    else:
        raise ValueError(f"unknown aggregation rule: {rule!r}")
    gkey = key_s[starts]
    return gkey // n_narr, gkey % n_narr, score


class DayAccumulator:
    """count / total / sum-of-squares / peak per node, for one calendar day."""

    __slots__ = ("count", "total", "sumsq", "peak", "n_headlines", "n_unassigned")

    def __init__(self, n_nodes: int):
        self.count = np.zeros(n_nodes, dtype=np.int64)
        self.total = np.zeros(n_nodes, dtype=np.float64)
        self.sumsq = np.zeros(n_nodes, dtype=np.float64)
        self.peak = np.full(n_nodes, -np.inf, dtype=np.float64)
        self.n_headlines = 0
        self.n_unassigned = 0

    @property
    def n_nodes(self) -> int:
        return self.count.shape[0]

    def add_values(self, node_ids: np.ndarray, values: np.ndarray) -> None:
        """Fold in (node, value) observations. Nodes not present contribute nothing."""
        if node_ids.size == 0:
            return
        n = self.n_nodes
        v = np.asarray(values, dtype=np.float64)
        ids = np.asarray(node_ids, dtype=np.int64)
        self.count += np.bincount(ids, minlength=n)
        self.total += np.bincount(ids, weights=v, minlength=n)
        self.sumsq += np.bincount(ids, weights=v * v, minlength=n)
        order = np.argsort(ids, kind="stable")
        ids_s, v_s = ids[order], v[order]
        starts = np.flatnonzero(np.r_[True, ids_s[1:] != ids_s[:-1]])
        np.maximum.at(self.peak, ids_s[starts], np.maximum.reduceat(v_s, starts))

    def add_arrays(self, count: np.ndarray, total: np.ndarray, sumsq: np.ndarray,
                   peak: np.ndarray) -> None:
        """Fold in already-reduced accumulators (from the compiled kernel or another chunk)."""
        self.count += count
        self.total += total
        self.sumsq += sumsq
        np.maximum(self.peak, peak, out=self.peak)

    def merge(self, other: "DayAccumulator") -> None:
        self.add_arrays(other.count, other.total, other.sumsq, other.peak)
        self.n_headlines += other.n_headlines
        self.n_unassigned += other.n_unassigned

    def stats(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(intensity, std_ddof0, total, peak) as float64 with NaN where count == 0."""
        has = self.count > 0
        n = np.maximum(self.count, 1).astype(np.float64)
        mean = self.total / n
        # population variance; clamp the cancellation error at zero
        var = np.maximum(self.sumsq / n - mean * mean, 0.0)
        nan = np.nan
        return (
            np.where(has, mean, nan), np.where(has, np.sqrt(var), nan),
            np.where(has, self.total, nan), np.where(has, self.peak, nan),
        )

    def frame(self, nodes: pl.DataFrame, day: date, *, sentiment: str = "all",
              n_headlines: int | None = None) -> pl.DataFrame:
        """One row per node for ``day``; null (not zero) where SUPPORT == 0.

        ``n_headlines`` is the DAY's headline count (the attention denominator,
        shared by every sentiment label); ``self.n_headlines`` is the number of
        headlines this accumulator actually saw (N_LABELLED).
        """
        intensity, std, total, peak = self.stats()
        day_total = self.n_headlines if n_headlines is None else n_headlines
        return nodes.with_columns(
            pl.lit(day).cast(pl.Date).alias("DATE"),
            pl.lit(sentiment).alias("SENTIMENT"),
            pl.Series("SUPPORT", self.count, dtype=pl.Int32),
            pl.Series("TOTAL_SCORE", total, dtype=pl.Float64).fill_nan(None),
            pl.Series("INTENSITY", intensity).cast(pl.Float32).fill_nan(None),
            pl.Series("STD_SCORE", std).cast(pl.Float32).fill_nan(None),
            pl.Series("PEAK", peak).cast(pl.Float32).fill_nan(None),
            pl.lit(day_total, dtype=pl.Int32).alias("N_HEADLINES"),
            pl.lit(self.n_headlines, dtype=pl.Int32).alias("N_LABELLED"),
        )
