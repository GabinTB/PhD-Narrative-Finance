"""tau_asof: the production F0 floor as a point-in-time monthly series.

Conventions mirror mu_asof: a 1-month delay and, by default, a ROLLING
5-year window (``expanding`` is the alternative over the same partitions).

For each cutoff (a partition MONTH_END at or before today - 1M), the job is
self-contained::

    mu            = latest mu_asof row with DATE <= cutoff           (as-of)
    N_eff, ...    = warmup.compute_warmup(table, P, config, mu)       (recomputed here)
    window        = partitions with month_end(cutoff - 5Y) < MONTH_END <= cutoff
                    (60 partitions; or every partition <= cutoff when expanding)
    while pool(window) < min_month_draws and older partitions exist:
        window   += one earlier partition                          (recorded)
    D             = TDigest.merge(window partitions, chronological order)
    W             = Welford.merged(window partitions)
    p_tail        = (1 - alpha) ** (1 / N_eff)
    tau_empirical = D.quantile(p_tail)                                (the mechanism of record)
    tau_gauss     = W.mean + W.std * norm.isf(alpha / N_eff)          (cross-check ONLY)
    pool_q        = D.quantile([0.99, 0.999, 0.9993, 0.9999])

One row per cutoff (schema.TAU_ASOF_SCHEMA) pinning N_EFF, EFFECTIVE_RANK,
LAMBDA1_SHARE, MU_DATE, MU_ARTIFACT_ID, N_PARTITIONS, POOL_COUNT,
WINDOW_MONTHS_USED, WINDOW_EXTENDED_BY. The build is a pure function of its
inputs: identical inputs give byte-identical rows (tested).

Cold start: while fewer than 60 partitions exist the window is simply
shorter; those rows are VALID and identifiable by N_PARTITIONS.

Why no bucket histogram: p_tail is about 1 - 7e-4 (N_eff ~ 11-14, alpha
0.01) and the diagnostics go to 0.9999. Cosine null draws live in
~[0.1, 0.6]; 0.01-wide buckets put the whole tail above tau into ~2 buckets
and 0.1-wide buckets into one, so neither can place a quantile 1e-4 from the
top. The t-digest keeps ~1000 centroids concentrated at the tails.

The Gaussian tau is a structural robustness check, not a sampling-noise
one: a gap above ``config.gap_alert_threshold`` is logged as a warning and
nothing else happens.

Reader: ``TauSeriesProvider.tau_for(day)`` returns the latest row with
MONTH_END strictly before ``day``, else ``LookaheadError`` -- the same
guarantee as ``resolve_mu_asof``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Sequence

import numpy as np
import pandas as pd
import polars as pl

from narrative_scoring.calibration import LookaheadError, MuRecord, TauRecord, resolve_mu_asof
from narrative_scoring.config import ScoringConfig
from narrative_scoring.corrections import Correction
from narrative_scoring.f0 import TDigest, Welford, gaussian_tau, tail_probability
from narrative_scoring.partitions import month_end, partition_digest, partition_welford
from narrative_scoring.primitives import PrimitiveTable
from narrative_scoring.schema import TAU_ASOF_SCHEMA
from narrative_scoring.warmup import compute_warmup

log = logging.getLogger(__name__)

WINDOW_DEFAULT = "5Y"
WINDOW_EXPANDING = "expanding"
DELAY = pd.DateOffset(months=1)
POOL_QUANTILES = (0.99, 0.999, 0.9993, 0.9999)


def window_offset(window: str) -> pd.DateOffset | None:
    if window == WINDOW_EXPANDING:
        return None
    if window.endswith("Y") and window[:-1].isdigit() and int(window[:-1]) > 0:
        return pd.DateOffset(years=int(window[:-1]))
    if window.endswith("M") and window[:-1].isdigit() and int(window[:-1]) > 0:
        return pd.DateOffset(months=int(window[:-1]))
    raise ValueError(f"window must be 'expanding', 'NY' or 'NM', got {window!r}")


def window_start(cutoff: date, offset: pd.DateOffset) -> date:
    """Exclusive lower bound of a rolling window: the month-end ``offset`` before ``cutoff``.

    Month arithmetic is done on the first of the month so that a 5Y window
    ending 2013-04-30 starts after 2008-04-30 (60 partitions), whatever the
    day counts of the months involved.
    """
    first = (pd.Timestamp(cutoff.replace(day=1)) - offset).date()
    return month_end(first.year, first.month)


def latest_cutoff(partitions: pl.DataFrame, today: date) -> date | None:
    """Latest partition MONTH_END at or before today - 1 month."""
    limit = (pd.Timestamp(today) - DELAY).date()
    ok = partitions.filter(pl.col("MONTH_END") <= limit)
    return None if ok.is_empty() else ok["MONTH_END"].max()


def _months_span(start_exclusive: date, cutoff: date) -> int:
    a, b = start_exclusive, cutoff
    return (b.year - a.year) * 12 + (b.month - a.month)


def build_tau_rows(
    partitions: pl.DataFrame,
    config: ScoringConfig,
    table: PrimitiveTable,
    P: np.ndarray,
    mu_df: pl.DataFrame | None,
    *,
    mu_asof_id: str | None = None,
    cutoffs: Sequence[date] | None = None,
    window: str = WINDOW_DEFAULT,
    seed: int = 0,
    partitions_id: str = "",
    code_version: str | None = None,
    today: date | None = None,
) -> pl.DataFrame:
    """One TAU_ASOF row per cutoff. Default cutoffs: every MONTH_END <= today - 1M."""
    parts = partitions.sort("MONTH_END")
    if parts.is_empty():
        raise ValueError("no partitions")
    bad = parts.filter(pl.col("F0_CONFIG_ID") != config.f0_digest())
    if bad.height:
        raise ValueError(f"{bad.height} partition(s) built under another F0 config")
    if parts.filter(pl.col("TAXONOMY_SHA1") != table.taxonomy_sha1).height:
        raise ValueError("partition taxonomy hash does not match the primitive table")
    if config.mode is not Correction.RAW and mu_df is None:
        raise ValueError(f"mode {config.mode.value} needs mu_asof to compute N_eff")

    if cutoffs is None:
        limit = (pd.Timestamp(today or date.today()) - DELAY).date()
        cutoffs = parts.filter(pl.col("MONTH_END") <= limit)["MONTH_END"].to_list()
    offset = window_offset(window)
    all_rows = parts.to_dicts()
    ends = [r["MONTH_END"] for r in all_rows]

    rows = []
    for cutoff in sorted(set(cutoffs)):
        # -- warmup at this cutoff: the mu row available as of the cutoff ---------
        mu: MuRecord | None = None
        if config.mode is not Correction.RAW:
            mu = resolve_mu_asof(mu_df, cutoff)
        warm, _ = compute_warmup(table, P, config, mu, mu_asof_id=mu_asof_id)
        p_tail = tail_probability(config.alpha, warm.n_eff)

        # -- nominal window, then the min_month_draws extension ------------------
        hi = max(i for i, e in enumerate(ends) if e <= cutoff)
        if offset is None:
            lo = 0
        else:
            start = window_start(cutoff, offset)
            lo = min((i for i, e in enumerate(ends) if e > start), default=hi)
        nominal_lo = lo
        pool = sum(int(r["WELFORD_COUNT"]) for r in all_rows[lo:hi + 1])
        while pool < config.min_month_draws and lo > 0:
            lo -= 1
            pool += int(all_rows[lo]["WELFORD_COUNT"])
        extended_by = nominal_lo - lo
        sel_rows = all_rows[lo:hi + 1]
        if extended_by:
            log.warning("tau_asof %s: pool below min_month_draws=%d; window extended by %d "
                        "month(s) back to %s (pool now %d)", cutoff, config.min_month_draws,
                        extended_by, sel_rows[0]["MONTH_END"], pool)
        elif pool < config.min_month_draws:
            log.warning("tau_asof %s: pool %d below min_month_draws=%d and history exhausted",
                        cutoff, pool, config.min_month_draws)

        if pool == 0:
            log.warning("tau_asof %s: no null draws in any partition of the window; no row",
                        cutoff)
            continue
        digest = TDigest.merge([partition_digest(r) for r in sel_rows])
        moments = Welford.merged(partition_welford(r) for r in sel_rows)
        tau_emp = digest.quantile(p_tail)
        tau_g = gaussian_tau(moments.mean, moments.std, warm.n_eff, config.alpha)
        gap = abs(tau_g - tau_emp)
        if gap > config.gap_alert_threshold:
            log.warning("tau_asof %s: |tau_gauss - tau_empirical| = %.4f above %.3f "
                        "(empirical %.4f, gaussian %.4f); structural, informational only",
                        cutoff, gap, config.gap_alert_threshold, tau_emp, tau_g)
        qs = digest.quantiles(np.asarray(POOL_QUANTILES))
        # the window actually used: from just before the first partition merged to the cutoff
        w_start = month_end(*_prev_month(sel_rows[0]["MONTH_END"]))
        rows.append({
            "MONTH_END": cutoff, "TAU_EMPIRICAL": float(tau_emp), "TAU_GAUSS": float(tau_g),
            "ABS_GAP": float(gap),
            "POOL_Q99": float(qs[0]), "POOL_Q999": float(qs[1]),
            "POOL_Q9993": float(qs[2]), "POOL_Q9999": float(qs[3]),
            "P_TAIL": float(p_tail), "POOL_COUNT": int(digest.count),
            "POOL_MEAN": float(moments.mean), "POOL_STD": float(moments.std),
            "N_PARTITIONS": len(sel_rows), "WINDOW_START": w_start, "WINDOW": window,
            "WINDOW_MONTHS_USED": _months_span(w_start, cutoff),
            "WINDOW_EXTENDED_BY": int(extended_by), "MIN_MONTH_DRAWS": config.min_month_draws,
            "N_EFF": float(warm.n_eff), "EFFECTIVE_RANK": float(warm.effective_rank),
            "LAMBDA1_SHARE": float(warm.lambda1_share),
            "MU_DATE": mu.date if mu is not None else None, "MU_ARTIFACT_ID": mu_asof_id or "",
            "ALPHA": config.alpha, "MODE": config.mode.value,
            "POOLING": config.paraphrase_pooling.value, "SEED": seed,
            "F0_CONFIG_ID": config.f0_digest(), "TAXONOMY_SHA1": table.taxonomy_sha1,
            "PARAPHRASE_SHA1": table.paraphrase_sha1, "EMBEDDINGS_DIGEST": warm.embeddings_digest,
            "PARTITIONS_ID": partitions_id, "CODE_VERSION": code_version or "",
        })
    return pl.DataFrame(rows, schema=TAU_ASOF_SCHEMA)


def _prev_month(d: date) -> tuple[int, int]:
    return (d.year - 1, 12) if d.month == 1 else (d.year, d.month - 1)


@dataclass
class TauSeriesProvider:
    """Production CalibrationProvider: tau from the tau_asof series, mu from mu_asof, per day.

    An empty series is allowed (cold start): every ``tau_for`` then raises
    ``LookaheadError``, which the replay driver turns into a null-only day.
    """

    tau_df: pl.DataFrame
    mu_df: pl.DataFrame | None
    tau_source_id: str
    mu_asof_id: str | None = None
    tau_policy: str = "tau_asof"

    def __post_init__(self) -> None:
        self.tau_df = self.tau_df.sort("MONTH_END")
        self._mu_cache: dict[date, MuRecord] = {}
        self._tau_cache: dict[date, TauRecord] = {}

    @property
    def mu_policy(self) -> str:
        return "none" if self.mu_df is None else "as_of_day"

    def mu_for(self, day: date) -> MuRecord | None:
        if self.mu_df is None:
            return None
        rec = self._mu_cache.get(day)
        if rec is None:
            rec = resolve_mu_asof(self.mu_df, day)
            self._mu_cache[day] = rec
        return rec

    def tau_for(self, day: date) -> TauRecord:
        rec = self._tau_cache.get(day)
        if rec is not None:
            return rec
        ok = self.tau_df.filter(pl.col("MONTH_END") < day)
        if ok.is_empty():
            raise LookaheadError(f"tau_asof has no cutoff strictly before {day}")
        r = ok.row(-1, named=True)
        rec = TauRecord(
            tau=r["TAU_EMPIRICAL"], gaussian_tau=r["TAU_GAUSS"], n_eff=r["N_EFF"],
            alpha=r["ALPHA"], trim_frac=float("nan"), n_draws=r["POOL_COUNT"], seed=r["SEED"],
            calibrated_from=r["WINDOW_START"], calibrated_through=r["MONTH_END"],
            mode=r["MODE"], paraphrase_pooling=r["POOLING"], mu_date=r["MU_DATE"],
            source_id=self.tau_source_id, month_end=r["MONTH_END"],
        )
        self._tau_cache[day] = rec
        return rec
