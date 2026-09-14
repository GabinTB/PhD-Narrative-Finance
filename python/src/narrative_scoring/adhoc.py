"""Ad-hoc frozen-mu scoring harness, for offline comparison notebooks.

``scoring.score_headlines_to_datalake`` implements the production pipeline: a
lagged, per-day, per-source expanding null pool and a correction that is
re-resolved from mu_asof every single day. That is the right design for a
committed datalake artifact spanning decades, but it is overkill for a
one-off notebook comparing a handful of scoring configurations (corrections,
pooling modes, paraphrase styles, F0/garbage toggles) over a single
historical window (e.g. the GFC or Covid crash).

This module implements a deliberately simplified harness for that use case,
built on the SAME tested primitives as the production path
(``scoring.score_day``, ``null_model.compute_n_eff/compute_tau``,
``null_model.trim_null_draws_batch``):

  - The correction (mu / mu_hat) is resolved ONCE, from mu_asof at a fixed
    calibration cutoff date, and reused for every headline scored --
    calibration headlines, the scoring window, and the description matrix
    alike. This preserves corrections.py's consistency rule (same mu applied
    to headlines and descriptions) with a single snapshot instead of one per
    day; it is only valid because the calibration cutoff is chosen well
    before the scoring window starts (mu_asof drifts slowly), which holds
    for both the GFC (cutoff 2005, window 2007-2009) and Covid (cutoff 2005,
    window 2020) test notebooks this module was written for.
  - The null pool is a single POOLED reservoir (not per-source, not
    lagged/expanding) built once from headlines up to the calibration
    cutoff. Pooling across sources rather than per-source is a further
    simplification versus the production model -- reasonable for a
    comparison notebook, not a substitute for the per-source floor when
    committing a real primitive_scores artifact.

Do not use this module to write a primitive_scores artifact; that is what
scoring.score_headlines_to_datalake is for.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from narrative_scoring.corrections import Correction
from narrative_scoring.descriptions import DescriptionEmbeddings, PoolingMode
from narrative_scoring.null_model import (
    ReservoirPool,
    compute_gaussian_tau,
    compute_n_eff,
    compute_tau,
    trim_null_draws_batch,
)
from narrative_scoring.scoring import (
    _correct_descriptions,
    load_month,
    representative_matrix,
    score_day,
)

log = logging.getLogger(__name__)


def resolve_frozen_mu(
    mu_df: pl.DataFrame | None, cutoff: date
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """mu/mu_hat at the latest mu_asof date <= cutoff, reused for the whole notebook.

    Unlike ``scoring.resolve_mu`` (an exact-date lookup, used by the
    production per-day loop where every calendar day is looked up), this is
    an asof lookup: ``cutoff`` need not itself be a date with an mu_asof row
    (e.g. a weekend), which matters here since only one cutoff is resolved
    for the whole notebook. RAW correction needs neither -> (None, None).
    """
    if mu_df is None:
        return None, None
    candidates = mu_df.filter(pl.col("DATE") <= cutoff)
    if candidates.is_empty():
        raise ValueError(f"mu_asof has no entry at or before cutoff {cutoff}")
    row = candidates.sort("DATE").row(-1, named=True)
    return np.asarray(row["MU"], dtype=np.float32), np.asarray(row["MU_HAT"], dtype=np.float32)


def centroid_from_raw(desc: DescriptionEmbeddings) -> DescriptionEmbeddings:
    """Derive a CENTROID DescriptionEmbeddings from a MAX/MEDIAN (raw K-vector) one.

    embed_descriptions writes identical (n_prim, K, dim) raw vectors for MAX
    and MEDIAN pooling -- pooling only changes behavior at score time. So a
    single non-CENTROID taxonomy_embeddings artifact carries everything
    needed to compare all three pooling modes, without three separate embed
    runs: MAX and MEDIAN read the raw vectors directly (``as_pooling``
    below); CENTROID is derived here by averaging + renormalizing.
    """
    if desc.mode is PoolingMode.CENTROID:
        return desc
    c = desc.vectors.mean(axis=1)
    norms = np.linalg.norm(c, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    vectors = (c / norms).astype(np.float32)
    return DescriptionEmbeddings(
        primitives=desc.primitives, mode=PoolingMode.CENTROID, vectors=vectors, k=1
    )


def as_pooling(desc: DescriptionEmbeddings, mode: PoolingMode) -> DescriptionEmbeddings:
    """Reinterpret a raw (n_prim, K, dim) DescriptionEmbeddings under a different
    non-CENTROID pooling mode (MAX <-> MEDIAN); same underlying vectors."""
    if desc.mode is PoolingMode.CENTROID or mode is PoolingMode.CENTROID:
        raise ValueError("as_pooling only switches between MAX and MEDIAN")
    return DescriptionEmbeddings(
        primitives=desc.primitives, mode=mode, vectors=desc.vectors, k=desc.k
    )


def _month_range(start: date, end: date) -> list[tuple[int, int]]:
    months = []
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        months.append((y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return months


@dataclass
class NullCalibration:
    F0: np.ndarray
    n_eff: float
    tau: float
    gaussian_tau: float
    n_draws: int


def calibrate_null_model(
    headlines_dir: Path,
    embeddings_dir: Path,
    D_tax: DescriptionEmbeddings,
    correction: Correction,
    mu: np.ndarray | None,
    mu_hat: np.ndarray | None,
    calib_start: date,
    calib_end: date,
    *,
    alpha: float = 0.01,
    trim_frac: float = 0.10,
    cap: int = 2_000_000,
    chunk_size: int = 50_000,
    threads: int = 8,
    seed: int = 0,
) -> NullCalibration:
    """Pool trimmed null draws from every headline in [calib_start, calib_end],
    under the given (frozen) correction, and fit tau/N_eff from the pool.
    """
    D_corr = _correct_descriptions(D_tax.vectors, D_tax.mode, correction, mu, mu_hat)
    n_eff = compute_n_eff(representative_matrix(D_corr, D_tax.mode))

    pool = ReservoirPool(cap=cap, seed=seed)
    n_days = 0
    for y, m in _month_range(calib_start, calib_end):
        name = f"{y}-{m:02d}.parquet"
        hl_path, emb_path = headlines_dir / name, embeddings_dir / name
        if not hl_path.exists() or not emb_path.exists():
            continue
        month_df = load_month(hl_path, emb_path, threads=threads)
        if month_df.is_empty():
            continue
        month_df = month_df.with_columns(
            pl.col("TIMESTAMP_UTC").str.slice(0, 10).str.to_date().alias("_DATE")
        )
        for dt, day_df in month_df.group_by("_DATE", maintain_order=True):
            d = dt[0] if isinstance(dt, tuple) else dt
            if d < calib_start or d > calib_end:
                continue
            _, raw_chunks, _, _ = score_day(
                day_df, d, D_tax, None, correction, mu, mu_hat,
                use_f0=False, use_garbage=False, chunk_size=chunk_size,
            )
            for S in raw_chunks:
                pool.add(trim_null_draws_batch(S, trim_frac=trim_frac).ravel())
            n_days += 1

    if pool.n_filled == 0:
        raise RuntimeError(
            f"no headlines found for null calibration in [{calib_start}, {calib_end}]"
        )

    F0 = pool.draws
    log.info("calibrated null model: %d day(s), %d draw(s), N_eff=%.1f", n_days, F0.shape[0], n_eff)
    return NullCalibration(
        F0=F0,
        n_eff=n_eff,
        tau=compute_tau(F0, n_eff, alpha=alpha),
        gaussian_tau=compute_gaussian_tau(F0, n_eff, alpha=alpha),
        n_draws=int(F0.shape[0]),
    )


@dataclass
class ScoringResult:
    scores: pl.DataFrame           # concatenated PRIMITIVE_SCORES_SCHEMA over the window
    n_days: int
    n_garbage_rejected: int
    config: dict[str, Any] = field(default_factory=dict)


def score_window(
    headlines_dir: Path,
    embeddings_dir: Path,
    D_tax: DescriptionEmbeddings,
    D_garbage: DescriptionEmbeddings | None,
    correction: Correction,
    mu: np.ndarray | None,
    mu_hat: np.ndarray | None,
    tau: float,
    *,
    start: date,
    end: date,
    use_f0: bool = True,
    use_garbage: bool = True,
    rel_floor: float = 0.65,
    trim_frac: float = 0.10,
    chunk_size: int = 50_000,
    threads: int = 8,
    config: dict[str, Any] | None = None,
) -> ScoringResult:
    """Score every headline in [start, end] under one fixed configuration.

    Every day uses the SAME frozen mu/mu_hat and the SAME pooled tau (passed
    as score_day's ``fallback_tau`` with an empty per-source override) --
    see the module docstring for why that is a valid simplification here.
    """
    if use_garbage and D_garbage is None:
        log.warning("use_garbage=True but D_garbage is None; disabling garbage layer")
        use_garbage = False

    monthly: list[pl.DataFrame] = []
    n_days = 0
    n_garbage_rejected = 0

    for y, m in _month_range(start, end):
        name = f"{y}-{m:02d}.parquet"
        hl_path, emb_path = headlines_dir / name, embeddings_dir / name
        if not hl_path.exists() or not emb_path.exists():
            continue
        month_df = load_month(hl_path, emb_path, threads=threads)
        if month_df.is_empty():
            continue
        month_df = month_df.with_columns(
            pl.col("TIMESTAMP_UTC").str.slice(0, 10).str.to_date().alias("_DATE")
        )
        for dt, day_df in month_df.group_by("_DATE", maintain_order=True):
            d = dt[0] if isinstance(dt, tuple) else dt
            if d < start or d > end:
                continue
            stats, _, _, n_rej = score_day(
                day_df, d, D_tax, D_garbage, correction, mu, mu_hat,
                use_f0=use_f0, use_garbage=use_garbage,
                tau_by_source={}, fallback_tau=tau, rel_floor=rel_floor,
                trim_frac=trim_frac, chunk_size=chunk_size,
            )
            monthly.append(stats)
            n_garbage_rejected += n_rej
            n_days += 1

    if not monthly:
        raise RuntimeError(f"no headlines found in [{start}, {end}]")

    scores = pl.concat(monthly)
    return ScoringResult(
        scores=scores, n_days=n_days, n_garbage_rejected=n_garbage_rejected, config=config or {}
    )


def summarize(result: ScoringResult) -> dict[str, Any]:
    """Headline comparison stats for one ScoringResult -- for the summary table."""
    df = result.scores
    active = df.filter(pl.col("SUPPORT") > 0)
    n_prim_days = df.height
    return {
        **result.config,
        "n_days": result.n_days,
        "n_primitive_days": n_prim_days,
        "activation_rate": active.height / n_prim_days if n_prim_days else 0.0,
        "mean_intensity_active": active["INTENSITY"].mean() if active.height else 0.0,
        "mean_support_active": active["SUPPORT"].mean() if active.height else 0.0,
        "mean_peak_active": active["PEAK"].mean() if active.height else 0.0,
        "n_garbage_rejected": result.n_garbage_rejected,
    }
