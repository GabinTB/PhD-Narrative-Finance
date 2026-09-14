"""Per-day / per-month headline scoring.

Per calendar day:

  1. Load that day's headline embeddings (join headline_embeddings on
     RP_STORY_ID to ravenpack_headlines for TIMESTAMP_UTC and SOURCE_NAME).
  2. Resolve mu_t / mu_hat_t from the mu_asof artifact by exact date lookup.
     If the date is missing (the first month, before the delay cutoff), skip
     the day and count it.
  3. Apply the correction to headlines AND to the description matrix using
     the same mu_t (see corrections.py's consistency rule).
  4. Compute S = X_corr @ D_corr.T.  For MAX/MEDIAN pooling, D_corr is
     (n_prim, K, 384) -- reshape to (n_prim*K, 384), score, reshape back to
     (n_head, n_prim, K), pool over the last axis.
  5. If the garbage layer is enabled, compute S_garbage the same way and
     apply the rejection mask.
  6. If the F0 layer is enabled, apply the gate using tau for that day's
     source (falling back to a pooled tau when a source is unseen or thin).
  7. Aggregate to daily primitive intensity:

        INTENSITY = mean of retained (non-zero) scores for that primitive
                    across that day's headlines
        SUPPORT   = count of headlines with a non-zero retained score
        PEAK      = max retained score that day

     A primitive with zero support that day yields INTENSITY=0.0,
     SUPPORT=0, PEAK=0.0 -- never a null.

Memory: headlines are chunked within a day (default 50,000 rows) so the
(n_head, n_prim[, K]) score tensor never exceeds a few GB.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import duckdb
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from narrative_scoring.corrections import Correction, apply_correction
from narrative_scoring.descriptions import DescriptionEmbeddings, PoolingMode, from_frame
from narrative_scoring.garbage import apply_garbage_filter
from narrative_scoring.null_model import (
    ReservoirPool,
    apply_gate,
    compute_gaussian_tau,
    compute_n_eff,
    compute_tau,
    trim_null_draws_batch,
)
from narrative_scoring.schema import PRIMITIVE_SCORES_SCHEMA

if TYPE_CHECKING:
    from datalake import Artifact, DatalakeIndex
    from datalake.verify import Finding

log = logging.getLogger(__name__)

KIND = "primitive_scores"
SOURCE_HEADLINES_KIND = "ravenpack_headlines"
SOURCE_EMBEDDINGS_KIND = "headline_embeddings"
SOURCE_MU_ASOF_KIND = "mu_asof"
SOURCE_TAXONOMY_EMBEDDINGS_KIND = "taxonomy_embeddings"

PIPELINE = "PhD-Narrative-Finance"
PIPELINE_REPO = "https://github.com/GabinTB/PhD-Narrative-Finance"

_ARROW_SCHEMA: pa.Schema | None = None


def _arrow_schema() -> pa.Schema:
    global _ARROW_SCHEMA
    if _ARROW_SCHEMA is None:
        _ARROW_SCHEMA = pl.DataFrame(schema=PRIMITIVE_SCORES_SCHEMA).to_arrow().schema
    return _ARROW_SCHEMA


# ---------------------------------------------------------------------------
# Scoring math (unit-tested directly)
# ---------------------------------------------------------------------------

def score_chunk(X: np.ndarray, D: np.ndarray, mode: PoolingMode) -> np.ndarray:
    """X: (n_head, dim) corrected headline embeddings.

    D: (n_prim, dim) for CENTROID, (n_prim, K, dim) for MAX/MEDIAN.
    Returns S: (n_head, n_prim).
    """
    if mode is PoolingMode.CENTROID:
        return X @ D.T

    n_prim, k, dim = D.shape
    D_flat = D.reshape(n_prim * k, dim)
    raw = X @ D_flat.T                            # (n_head, n_prim*k)
    raw = raw.reshape(X.shape[0], n_prim, k)
    if mode is PoolingMode.MAX:
        return raw.max(axis=-1)
    if mode is PoolingMode.MEDIAN:
        return np.median(raw, axis=-1)
    raise ValueError(f"unknown pooling mode: {mode!r}")


def representative_matrix(D: np.ndarray, mode: PoolingMode) -> np.ndarray:
    """A single (n_prim, dim) unit-norm matrix for N_eff purposes.

    CENTROID already is one; MAX/MEDIAN average+renormalize their K vectors
    per primitive so the Gram matrix in compute_n_eff is well-defined.
    """
    if mode is PoolingMode.CENTROID:
        return D
    c = D.mean(axis=1)
    norms = np.linalg.norm(c, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return (c / norms).astype(np.float32)


def daily_stats(chunks: list[np.ndarray], primitives: list[str], date: Any) -> pl.DataFrame:
    """Combine one or more (n_head_chunk, n_prim) gated score matrices for one
    day into a PRIMITIVE_SCORES_SCHEMA frame.

    Accumulating per-chunk sums/counts/maxes in float64 and combining gives
    results equal to a single pass (sum, count, and max are associative), so
    chunking is a pure memory-bound optimization, not an approximation.
    """
    n_prim = len(primitives)
    total_sum = np.zeros(n_prim, dtype=np.float64)
    total_support = np.zeros(n_prim, dtype=np.int64)
    total_peak = np.zeros(n_prim, dtype=np.float64)

    for S in chunks:
        if S.shape[0] == 0:
            continue
        nonzero = S != 0
        total_sum += S.sum(axis=0, dtype=np.float64)
        total_support += nonzero.sum(axis=0)
        chunk_peak = np.where(nonzero.any(axis=0), S.max(axis=0), 0.0)
        total_peak = np.maximum(total_peak, chunk_peak)

    with np.errstate(invalid="ignore", divide="ignore"):
        intensity = np.where(
            total_support > 0, total_sum / np.maximum(total_support, 1), 0.0
        )

    return pl.DataFrame(
        {
            "DATE": [date] * n_prim,
            "PRIMITIVE": primitives,
            "INTENSITY": intensity.astype(np.float32),
            "SUPPORT": total_support.astype(np.int32),
            "PEAK": total_peak.astype(np.float32),
        },
        schema=PRIMITIVE_SCORES_SCHEMA,
    )


def gate_with_source_tau(
    S: np.ndarray,
    sources: list[str],
    tau_by_source: dict[str, float],
    fallback_tau: float,
    rel_floor: float = 0.65,
) -> np.ndarray:
    """Per-row F0 gate, tau resolved per headline's source (grouped, not looped)."""
    out = np.zeros_like(S)
    sources_arr = np.asarray(sources)
    for src in set(sources):
        mask = sources_arr == src
        tau = tau_by_source.get(src, fallback_tau)
        out[mask] = apply_gate(S[mask], tau, rel_floor)
    return out


# ---------------------------------------------------------------------------
# Correction helpers over (possibly 3D) description matrices
# ---------------------------------------------------------------------------

def _correct_descriptions(
    D: np.ndarray, mode: PoolingMode, correction: Correction, mu, mu_hat
) -> np.ndarray:
    if mode is PoolingMode.CENTROID:
        out, _ = apply_correction(D, correction, mu=mu, mu_hat=mu_hat)
        return out
    n_prim, k, dim = D.shape
    flat, _ = apply_correction(D.reshape(n_prim * k, dim), correction, mu=mu, mu_hat=mu_hat)
    return flat.reshape(n_prim, k, dim)


# ---------------------------------------------------------------------------
# Day loop (used both by tests, indirectly, and by the datalake driver)
# ---------------------------------------------------------------------------

def score_day(
    day_df: pl.DataFrame,           # RP_STORY_ID, TIMESTAMP_UTC, SOURCE_NAME, EMBEDDING
    date: Any,
    D_tax: DescriptionEmbeddings,
    D_garbage: DescriptionEmbeddings | None,
    correction: Correction,
    mu: np.ndarray | None,
    mu_hat: np.ndarray | None,
    *,
    use_f0: bool,
    use_garbage: bool,
    tau_by_source: dict[str, float] | None = None,
    fallback_tau: float | None = None,
    rel_floor: float = 0.65,
    trim_frac: float = 0.10,
    chunk_size: int = 50_000,
) -> tuple[pl.DataFrame, list[np.ndarray], list[str], int]:
    """Score one calendar day of headlines.

    Returns (daily stats frame, per-chunk trimmed-null-draws-by-source-ready
    raw taxonomy score chunks [pre-gate], per-chunk sources, n_garbage_rejected).
    The raw score chunks + sources are handed back so the caller can feed
    trimmed draws into the lagged null pool for future days.
    """
    primitives = D_tax.primitives
    D_tax_corr = _correct_descriptions(D_tax.vectors, D_tax.mode, correction, mu, mu_hat)
    D_g_corr = (
        _correct_descriptions(D_garbage.vectors, D_garbage.mode, correction, mu, mu_hat)
        if (use_garbage and D_garbage is not None)
        else None
    )

    n_rows = day_df.height
    gated_chunks: list[np.ndarray] = []
    raw_chunks: list[np.ndarray] = []
    source_chunks: list[str] = []
    n_garbage_rejected = 0

    for start in range(0, max(n_rows, 1), chunk_size):
        if start >= n_rows:
            break
        chunk = day_df.slice(start, chunk_size)
        X = np.asarray(chunk["EMBEDDING"].to_list(), dtype=np.float32)
        X_corr, _ = apply_correction(X, correction, mu=mu, mu_hat=mu_hat)

        S = score_chunk(X_corr, D_tax_corr, D_tax.mode)
        raw_chunks.append(S)
        sources = chunk["SOURCE_NAME"].to_list()
        source_chunks.extend(sources)

        reject = np.zeros(S.shape[0], dtype=bool)
        if use_garbage and D_g_corr is not None:
            S_g = score_chunk(X_corr, D_g_corr, D_garbage.mode)
            reject, _ = apply_garbage_filter(S, S_g)
            n_garbage_rejected += int(reject.sum())

        S_gated = S.copy()
        S_gated[reject] = 0.0

        if use_f0:
            S_gated = gate_with_source_tau(
                S_gated, sources, tau_by_source or {}, fallback_tau or 0.0, rel_floor
            )
            S_gated[reject] = 0.0  # a garbage-rejected row stays rejected

        gated_chunks.append(S_gated)

    stats = daily_stats(gated_chunks, primitives, date)
    return stats, raw_chunks, source_chunks, n_garbage_rejected


# ---------------------------------------------------------------------------
# mu_asof lookup
# ---------------------------------------------------------------------------

def resolve_mu(mu_asof_df: pl.DataFrame, date: Any) -> tuple[np.ndarray, np.ndarray] | None:
    """Exact-date lookup into an mu_asof frame; None if the date is absent."""
    row = mu_asof_df.filter(pl.col("DATE") == date)
    if row.is_empty():
        return None
    r = row.row(0, named=True)
    return np.asarray(r["MU"], dtype=np.float32), np.asarray(r["MU_HAT"], dtype=np.float32)


# ---------------------------------------------------------------------------
# Month loading (DuckDB join, mirrors mu_asof.compute_daily_sums)
# ---------------------------------------------------------------------------

def load_month(headlines_path: Path, embeddings_path: Path, threads: int = 8) -> pl.DataFrame:
    """RP_STORY_ID, TIMESTAMP_UTC, SOURCE_NAME, EMBEDDING for one month, joined."""
    conn = duckdb.connect()
    try:
        conn.execute(f"PRAGMA threads={threads}")
        df = conn.sql(
            f"""
            SELECT h.RP_STORY_ID, h.TIMESTAMP_UTC, h.SOURCE_NAME, e.EMBEDDING
            FROM read_parquet('{headlines_path}') h
            JOIN read_parquet('{embeddings_path}') e USING (RP_STORY_ID)
            ORDER BY h.TIMESTAMP_UTC
            """
        ).pl()
    finally:
        conn.close()
    return df


# ---------------------------------------------------------------------------
# Datalake-aware entry point
# ---------------------------------------------------------------------------

@dataclass
class ScoringConfig:
    taxonomy_version: str
    family: str = "evergreen"
    paraphrase_style: str = "headlined"
    pooling: PoolingMode = PoolingMode.CENTROID
    correction: Correction = Correction.R2
    use_f0: bool = True
    use_garbage: bool = True
    alpha: float = 0.01
    trim_frac: float = 0.10
    rel_floor: float = 0.65
    null_delay: str = "1M"
    null_cap: int = 5_000_000
    chunk_size: int = 50_000
    start: int = 20070101
    end: int = 20091231


def run_scoring_range(
    out_dir: Path,
    headlines_dir: Path,
    embeddings_dir: Path,
    mu_df: pl.DataFrame | None,
    D_tax: DescriptionEmbeddings,
    D_garbage: DescriptionEmbeddings | None,
    correction: Correction,
    start: int,
    end: int,
    *,
    use_f0: bool = True,
    use_garbage: bool = True,
    alpha: float = 0.01,
    trim_frac: float = 0.10,
    rel_floor: float = 0.65,
    null_delay: str = "1M",
    null_cap: int = 5_000_000,
    chunk_size: int = 50_000,
    threads: int = 8,
    skip_existing: bool = False,
) -> float:
    """Core month-by-month scoring loop, independent of the datalake `run()`
    context so both a fresh run and a manual resume (progress-logged hashing,
    see scripts/score_headlines.py) can drive it against the same out_dir.

    Streams months chronologically, maintaining a lagged, source-keyed null
    pool (null_model.py), and writes one parquet per month plus a single
    null_model.json diagnostic record into out_dir.  Returns the last
    computed N_eff (for logging / diagnostics).
    """
    from ravenpack.headlines.mu_asof import parse_delay

    delay_offset = parse_delay(null_delay)
    null_pool = _ReservoirNullPool(cap=null_cap)
    pending: dict[Any, dict[str, np.ndarray]] = {}
    n_skipped_no_mu = 0
    last_n_eff = 0.0

    months = [(y, m) for y in range(start // 10000, end // 10000 + 1) for m in range(1, 13)]
    for y, m in months:
        name = f"{y}-{m:02d}.parquet"
        out_path = out_dir / name
        if skip_existing and out_path.exists():
            continue
        hl_path = headlines_dir / name
        emb_path = embeddings_dir / name
        if not hl_path.exists() or not emb_path.exists():
            continue

        month_df = load_month(hl_path, emb_path, threads=threads)
        if month_df.is_empty():
            continue
        month_df = month_df.with_columns(
            pl.col("TIMESTAMP_UTC").str.slice(0, 10).str.to_date().alias("_DATE")
        )

        month_stats: list[pl.DataFrame] = []
        for date, day_df in month_df.group_by("_DATE", maintain_order=True):
            d = date[0] if isinstance(date, tuple) else date

            mu = mu_hat = None
            if correction is not Correction.RAW:
                resolved = resolve_mu(mu_df, d) if mu_df is not None else None
                if resolved is None:
                    n_skipped_no_mu += 1
                    continue
                mu, mu_hat = resolved

            D_tax_corr = _correct_descriptions(D_tax.vectors, D_tax.mode, correction, mu, mu_hat)
            rep = representative_matrix(D_tax_corr, D_tax.mode)
            n_eff = compute_n_eff(rep)
            last_n_eff = n_eff

            cutoff = d - delay_offset
            for pending_date in [dt for dt in pending if dt < cutoff]:
                for src, draws in pending[pending_date].items():
                    null_pool.add(src, draws)
                del pending[pending_date]

            tau_by_source: dict[str, float] = {}
            fallback_tau = None
            if use_f0:
                fallback_draws = null_pool.pooled_draws()
                if fallback_draws.size:
                    fallback_tau = compute_tau(fallback_draws, n_eff, alpha=alpha)
                for src in null_pool.sources():
                    draws = null_pool.draws_for(src)
                    if draws.size:
                        tau_by_source[src] = compute_tau(draws, n_eff, alpha=alpha)

            stats, raw_chunks, source_chunks, _n_rejected = score_day(
                day_df, d, D_tax, D_garbage, correction, mu, mu_hat,
                use_f0=use_f0, use_garbage=use_garbage,
                tau_by_source=tau_by_source, fallback_tau=fallback_tau,
                rel_floor=rel_floor, trim_frac=trim_frac, chunk_size=chunk_size,
            )
            month_stats.append(stats)

            if use_f0 and raw_chunks:
                S_all = np.concatenate(raw_chunks, axis=0)
                trimmed = trim_null_draws_batch(S_all, trim_frac=trim_frac)
                sources_arr = np.asarray(source_chunks)
                day_bucket = pending.setdefault(d, {})
                for src in set(source_chunks):
                    mask = sources_arr == src
                    day_bucket[src] = trimmed[mask].ravel()

        if not month_stats:
            continue
        out = pl.concat(month_stats)
        tmp = out_path.with_suffix(".parquet.tmp")
        try:
            pq.write_table(out.to_arrow().cast(_arrow_schema()), tmp, compression="zstd")
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        tmp.replace(out_path)
        log.info("wrote %s (%d days)", name, out["DATE"].n_unique())

    if n_skipped_no_mu:
        log.warning("%d day(s) skipped: no mu_asof entry before the delay cutoff", n_skipped_no_mu)

    _write_null_model_json(
        out_dir, null_pool,
        alpha=alpha, trim_frac=trim_frac, rel_floor=rel_floor, n_eff=last_n_eff,
    )
    return last_n_eff


def load_taxonomy_embeddings(
    tax_art: "Artifact", use_garbage: bool
) -> tuple[DescriptionEmbeddings, DescriptionEmbeddings | None, bool]:
    """Load taxonomy.parquet (+ garbage.parquet, if requested and present)."""
    hp_tax = tax_art.meta.hyperparams
    D_tax = from_frame(
        pl.read_parquet(tax_art.path / "taxonomy.parquet"), PoolingMode(hp_tax["pooling"])
    )
    D_garbage = None
    if use_garbage and (tax_art.path / "garbage.parquet").is_file():
        D_garbage = from_frame(
            pl.read_parquet(tax_art.path / "garbage.parquet"), PoolingMode(hp_tax["pooling"])
        )
    elif use_garbage:
        log.warning("use_garbage=True but taxonomy_embeddings has no garbage.parquet; disabling")
        use_garbage = False
    return D_tax, D_garbage, use_garbage


def score_headlines_to_datalake(
    index: "DatalakeIndex",
    taxonomy_embeddings_artifact_id: str,
    pipeline_version: str,
    *,
    correction: Correction = Correction.R2,
    start: int = 20070101,
    end: int = 20091231,
    use_f0: bool = True,
    use_garbage: bool = True,
    alpha: float = 0.01,
    trim_frac: float = 0.10,
    rel_floor: float = 0.65,
    null_delay: str = "1M",
    null_cap: int = 5_000_000,
    chunk_size: int = 50_000,
    pipeline: str = PIPELINE,
    pipeline_repo: str | None = PIPELINE_REPO,
    repo_dir: Path | None = None,
    threads: int = 8,
) -> "Artifact":
    """Score every headline in [start, end] against a taxonomy version.

    Resolves ravenpack_headlines / headline_embeddings / mu_asof / the given
    taxonomy_embeddings artifact and delegates the day-by-day work to
    ``run_scoring_range``.  For resuming a partial artifact, see
    scripts/score_headlines.py, which drives ``run_scoring_range`` directly
    against the existing artifact directory (mirroring ingest_ravenpack.py's
    manual resume, with progress-logged hashing over the GDrive mount).
    """
    tax_art = index.get(taxonomy_embeddings_artifact_id)
    hp_tax = tax_art.meta.hyperparams

    headlines_art = index.latest(SOURCE_HEADLINES_KIND)
    embeddings_art = index.latest(SOURCE_EMBEDDINGS_KIND)
    mu_art = index.latest(SOURCE_MU_ASOF_KIND) if correction is not Correction.RAW else None
    mu_df = pl.read_parquet(mu_art.glob("*.parquet")) if mu_art is not None else None

    D_tax, D_garbage, use_garbage = load_taxonomy_embeddings(tax_art, use_garbage)

    hyperparams: dict[str, Any] = {
        "taxonomy_version": hp_tax.get("taxonomy_version"),
        "family": hp_tax.get("family"),
        "paraphrase_style": hp_tax.get("paraphrase_style"),
        "pooling": hp_tax.get("pooling"),
        "correction": correction.value,
        "use_f0": use_f0,
        "use_garbage": use_garbage,
        "alpha": alpha,
        "trim_frac": trim_frac,
        "rel_floor": rel_floor,
        "null_delay": null_delay,
        "start": start,
        "end": end,
    }
    sources = [headlines_art, embeddings_art, tax_art] + ([mu_art] if mu_art else [])
    notes = f"source_taxonomy_embeddings={tax_art.artifact_id}"

    with index.run(
        kind=KIND,
        pipeline=pipeline,
        pipeline_version=pipeline_version,
        pipeline_repo=pipeline_repo,
        repo_dir=repo_dir,
        hyperparams=hyperparams,
        notes=notes,
        sources=sources,
        verifier=KIND,
        hash_pattern="*.parquet",
    ) as run:
        run_scoring_range(
            run.out_dir, headlines_art.path, embeddings_art.path, mu_df,
            D_tax, D_garbage, correction, start, end,
            use_f0=use_f0, use_garbage=use_garbage, alpha=alpha,
            trim_frac=trim_frac, rel_floor=rel_floor, null_delay=null_delay,
            null_cap=null_cap, chunk_size=chunk_size, threads=threads,
            skip_existing=False,
        )

        n_months = len(list(run.out_dir.glob("*.parquet")))
        if n_months == 0:
            raise RuntimeError(f"scoring produced no monthly output for [{start}, {end}]")
        run.note(f"{n_months} monthly score files")

    return index.get(run.artifact_id)


class _ReservoirNullPool:
    """Source-keyed collection of ReservoirPool, used by the datalake driver."""

    def __init__(self, cap: int = 5_000_000, seed: int = 0):
        self.cap = cap
        self.seed = seed
        self._pools: dict[str, ReservoirPool] = {}

    def add(self, source: str, draws: np.ndarray) -> None:
        if draws.size == 0:
            return
        if source not in self._pools:
            self._pools[source] = ReservoirPool(cap=self.cap, seed=self.seed)
        self._pools[source].add(draws)

    def draws_for(self, source: str) -> np.ndarray:
        pool = self._pools.get(source)
        return pool.draws if pool is not None else np.empty(0, dtype=np.float64)

    def pooled_draws(self) -> np.ndarray:
        if not self._pools:
            return np.empty(0, dtype=np.float64)
        return np.concatenate([p.draws for p in self._pools.values() if p.draws.size])

    def sources(self) -> list[str]:
        return list(self._pools.keys())


def _write_null_model_json(
    out_dir: Path,
    null_pool: "_ReservoirNullPool",
    *,
    alpha: float,
    trim_frac: float,
    rel_floor: float,
    n_eff: float,
) -> None:
    """Diagnostic record: final per-source tau/N_eff/draw-counts + Gaussian cross-check.

    tau is time-varying under a non-RAW correction (D_corr, and so N_eff,
    changes with mu_t each day); the snapshot here reflects the last scoring
    date processed, recorded for post-hoc interpretability, not reused as a
    fixed threshold by anything downstream.
    """
    payload: dict[str, Any] = {
        "alpha": alpha, "trim_frac": trim_frac, "rel_floor": rel_floor,
        "n_eff": n_eff, "sources": {},
    }
    for src in null_pool.sources():
        draws = null_pool.draws_for(src)
        if draws.size == 0:
            continue
        entry: dict[str, Any] = {
            "n_draws": int(draws.size),
            "mean": float(draws.mean()),
            "std": float(draws.std()),
        }
        if n_eff > 0:
            entry["tau"] = compute_tau(draws, n_eff, alpha=alpha)
            entry["gaussian_tau"] = compute_gaussian_tau(draws, n_eff, alpha=alpha)
        payload["sources"][src] = entry

    pooled = null_pool.pooled_draws()
    if pooled.size:
        payload["pooled"] = {
            "n_draws": int(pooled.size),
            "mean": float(pooled.mean()),
            "std": float(pooled.std()),
        }
        if n_eff > 0:
            payload["pooled"]["tau"] = compute_tau(pooled, n_eff, alpha=alpha)
            payload["pooled"]["gaussian_tau"] = compute_gaussian_tau(pooled, n_eff, alpha=alpha)

    tmp = (out_dir / "null_model.json").with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(out_dir / "null_model.json")


# ---------------------------------------------------------------------------
# Content verifier (registered via pyproject.toml entry points)
# ---------------------------------------------------------------------------

def verify_artifact(artifact: "Artifact") -> "list[Finding]":
    """Verify a primitive_scores artifact matches its declared scope.

    Registered as the ``primitive_scores`` entry point. Checks: one parquet
    per month in the declared range; schema match; INTENSITY in [-1, 1];
    SUPPORT >= 0; null_model.json present with a finite tau/n_eff; the
    primitive set matching the taxonomy version recorded in hyperparams.
    """
    from datalake.verify import Finding, Severity

    findings: list[Finding] = []
    aid = artifact.artifact_id
    hp = artifact.meta.hyperparams
    start = hp.get("start")
    end = hp.get("end")

    if not hp.get("taxonomy_version"):
        findings.append(Finding(
            Severity.ERROR, aid,
            "hyperparams missing taxonomy_version; primitive set cannot be verified",
        ))

    if start is None or end is None:
        findings.append(Finding(
            Severity.WARNING, aid, "no start/end in hyperparams; cannot verify date coverage"
        ))
    else:
        present = {p.name for p in artifact.path.glob("*.parquet")}
        expected = {
            f"{y}-{m:02d}.parquet"
            for y in range(start // 10000, end // 10000 + 1)
            for m in range(1, 13)
        }
        missing = expected - present
        if missing:
            findings.append(Finding(
                Severity.WARNING, aid,
                f"{len(missing)} of {len(expected)} declared months missing: "
                f"{', '.join(sorted(missing)[:6])}" + (" ..." if len(missing) > 6 else ""),
            ))
        unexpected = present - expected
        if unexpected:
            findings.append(Finding(
                Severity.ERROR, aid,
                f"{len(unexpected)} parquet(s) outside declared range: "
                f"{', '.join(sorted(unexpected)[:6])}",
            ))

    null_model_path = artifact.path / "null_model.json"
    if not null_model_path.is_file():
        findings.append(Finding(Severity.ERROR, aid, "null_model.json missing"))
    else:
        try:
            payload = json.loads(null_model_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            findings.append(Finding(Severity.ERROR, aid, f"null_model.json unreadable: {exc}"))
            payload = None
        if payload is not None:
            pooled = payload.get("pooled")
            sources = payload.get("sources", {})
            if not pooled and not sources:
                findings.append(Finding(
                    Severity.ERROR, aid,
                    "null_model.json has no pooled or per-source draws recorded",
                ))

    for path in sorted(artifact.path.glob("*.parquet"))[:3]:
        try:
            df = pl.read_parquet(path)
        except Exception as exc:  # noqa: BLE001
            findings.append(Finding(Severity.ERROR, aid, f"{path.name}: unreadable parquet: {exc}"))
            continue
        if df.is_empty():
            findings.append(Finding(Severity.ERROR, aid, f"{path.name}: file is empty"))
            continue
        if df.schema != PRIMITIVE_SCORES_SCHEMA:
            findings.append(Finding(
                Severity.ERROR, aid,
                f"{path.name}: schema mismatch (got {dict(df.schema)}, "
                f"expected {dict(PRIMITIVE_SCORES_SCHEMA)})",
            ))
            continue
        bad_intensity = int(((df["INTENSITY"] < -1.0) | (df["INTENSITY"] > 1.0)).sum())
        if bad_intensity:
            findings.append(Finding(
                Severity.ERROR, aid,
                f"{path.name}: {bad_intensity} INTENSITY value(s) outside [-1, 1]",
            ))
        bad_support = int((df["SUPPORT"] < 0).sum())
        if bad_support:
            findings.append(Finding(
                Severity.ERROR, aid, f"{path.name}: {bad_support} SUPPORT value(s) < 0"
            ))

    return findings
