"""Asof pooled-mean headline embedding pipeline (mu_asof).

Computes, for each observed calendar day t, the pooled mean embedding vector
over headlines strictly before the cutoff date ``t - delay`` -- expanding
(all history) or rolling (a fixed-length window) -- plus its unit direction.

mu_asof(t) never uses data from [t - delay, t]: mean-centering or direction-
removing downstream cosine scores against mu_asof(t) keeps the pipeline
stateless-windowed and look-ahead-free.

Output layout: one parquet per artifact (a daily time series, not a monthly
corpus), named ``mu_asof_delay-{delay}_mode-{mode}.parquet``, with columns
(``schema.MU_ASOF_SCHEMA``):

    DATE      Date
    MU        Array(Float32, 384)   -- raw pooled mean vector (not normalized)
    MU_HAT    Array(Float32, 384)   -- unit mean direction: MU / ||MU||
    N         Int64                 -- headlines contributing to this day

MU is what mean-centering (subtract mu, renormalize) needs; MU_HAT is what
direction removal (project out mu_hat) needs. Both are stored so downstream
code picks what it needs without recomputing.

Two-pass design, adapted from the lab repo's ``mu_asof.py``:

  1. One DuckDB join+aggregate PER MONTH, accumulated in Python: the
     embedding parquet has only RP_STORY_ID and EMBEDDING (no timestamp), so
     each month's embeddings file is joined to that SAME month's headlines
     file (identically named ``YYYY-MM.parquet`` in both artifacts) to get
     each headline's day. UNNEST + generate_subscripts explodes each
     embedding into (day, pos, val) rows, then a hash aggregation over
     (day, pos) gives that month's per-day, per-dimension sums and counts in
     one shot. Joining month-by-month rather than the full corpus at once is
     deliberate: a single cross-corpus join (312 x 312 files) was tried
     first and spilled past 100GB of DuckDB temp storage over GDrive FUSE.
     A calendar day never spans two monthly files, so restricting each join
     to one month is exact, not an approximation -- and keeps each join
     trivially small.
  2. Cumulative sums over a complete daily calendar grid (missing days
     zero-filled), then an O(n_days) asof lookup per day. Everything here
     operates on a (n_days, 384) array -- trivial memory/compute.

A fresh ``duckdb.connect()`` is used per month rather than the module-level
``duckdb.sql()`` or one shared connection: a shared connection accumulates
state across calls and was the source of an OOM bug in an earlier pipeline.
"""
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import duckdb
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from ravenpack.headlines.schema import EMBEDDING_DIM, MU_ASOF_SCHEMA

if TYPE_CHECKING:
    from datalake import Artifact, DatalakeIndex
    from datalake.verify import Finding

log = logging.getLogger(__name__)

# Datalake artifact kind produced by this module, and the kinds it reads from.
KIND = "mu_asof"
SOURCE_HEADLINES_KIND = "ravenpack_headlines"
SOURCE_EMBEDDINGS_KIND = "headline_embeddings"

PIPELINE = "PhD-Narrative-Finance"
PIPELINE_REPO = "https://github.com/GabinTB/PhD-Narrative-Finance"

# Contiguity tolerance for verify_artifact: weekends + a holiday or two.
_MAX_DATE_GAP_DAYS = 5

# Float32 tolerance for the unit-norm check on MU_HAT.
_UNIT_NORM_TOL = 1e-5

_PERIOD_RE = re.compile(r"^(\d+)([dWM])$")


# ---------------------------------------------------------------------------
# Arrow schema for the ParquetWriter -- derived from MU_ASOF_SCHEMA so the
# written table and the writer schema are guaranteed to agree.
# ---------------------------------------------------------------------------

_ARROW_SCHEMA: pa.Schema | None = None


def _arrow_schema() -> pa.Schema:
    global _ARROW_SCHEMA
    if _ARROW_SCHEMA is None:
        _ARROW_SCHEMA = pl.DataFrame(schema=MU_ASOF_SCHEMA).to_arrow().schema
    return _ARROW_SCHEMA


# ---------------------------------------------------------------------------
# Period parsing -- ported exactly from the lab repo's mu_asof.py
# ---------------------------------------------------------------------------

def parse_period(spec: str, allowed_units: str) -> tuple[int, str]:
    m = _PERIOD_RE.match(spec)
    if not m or int(m.group(1)) <= 0:
        raise argparse.ArgumentTypeError(
            f"invalid period '{spec}', expected e.g. '5d', '2W', '3M' (n > 0)"
        )
    n, unit = int(m.group(1)), m.group(2)
    if unit not in allowed_units:
        raise argparse.ArgumentTypeError(
            f"period '{spec}' uses unit '{unit}', allowed units here are {list(allowed_units)}"
        )
    return n, unit


def offset_of(n: int, unit: str) -> pd.DateOffset:
    return {
        "d": pd.DateOffset(days=n),
        "W": pd.DateOffset(weeks=n),
        "M": pd.DateOffset(months=n),
    }[unit]


def parse_delay(spec: str) -> pd.DateOffset:
    """Minimum granularity is 1 day: 'Xd', 'XW', 'XM' all allowed."""
    n, unit = parse_period(spec, allowed_units="dWM")
    return offset_of(n, unit)


def parse_mode(spec: str) -> str | pd.DateOffset:
    """'expanding', or a rolling window >= 1 week: 'XW', 'XM' (no days)."""
    if spec == "expanding":
        return "expanding"
    n, unit = parse_period(spec, allowed_units="WM")
    return offset_of(n, unit)


def _offset_days(offset: pd.DateOffset, anchor: pd.Timestamp) -> int:
    """Calendar-day length of a DateOffset, anchored at a fixed reference date.

    Exact for day/week offsets; month offsets are measured by their actual
    calendar length from the anchor, matching how offsets are applied
    elsewhere in this module (``t - delay``, ``cutoff - window``).
    """
    return (anchor - (anchor - offset)).days


def validate_window_after_delay(delay: pd.DateOffset, mode: str | pd.DateOffset) -> None:
    """Raise if a rolling window would not fully clear the delay period.

    mu_asof(t) is defined as data strictly before ``t - delay``, over the
    last ``window`` of history. If the window is not longer than the delay,
    the window's start would fall inside [t - delay, t) -- i.e. the estimate
    would include data the delay was meant to exclude. No-op for expanding
    mode, which has no window to compare.
    """
    if mode == "expanding":
        return
    anchor = pd.Timestamp("2000-01-01")
    delay_days = _offset_days(delay, anchor)
    window_days = _offset_days(mode, anchor)
    if window_days <= delay_days:
        raise ValueError(
            f"rolling window ({window_days}d) must be longer than the delay "
            f"({delay_days}d); a window this short would look inside the delay period"
        )


# ---------------------------------------------------------------------------
# Pass 1: per-calendar-day sum vector and count, one month at a time, in SQL
# ---------------------------------------------------------------------------

def compute_daily_sums(
    headlines_dir: Path,
    embeddings_dir: Path,
    threads: int = 8,
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    """Per-month DuckDB join + aggregate, accumulated across months in Python.

    The embedding parquet carries only RP_STORY_ID and EMBEDDING (no
    timestamp), so each month's embeddings file is joined to that SAME
    month's headlines file (both artifacts name their monthly files
    identically: ``YYYY-MM.parquet``) to attribute each embedding to a
    calendar day. UNNEST explodes each embedding to 384 rows, but GROUP BY
    (day, pos) is a hash aggregation with only ~31 * 384 distinct groups per
    month, so each month's join+aggregate is trivially small -- DuckDB never
    needs to spill.

    A day never spans two monthly files, so restricting the join to
    same-month files is exact (not an approximation): the per-day
    (sum_vector, count) pairs from each month are simply merged into a
    running Python dict keyed by day. This deliberately avoids a single
    cross-corpus join (all headlines x all embeddings at once), which spills
    to disk past available memory once the corpus spans decades.

    MAX(n_vals) doubles as the per-day headline count: every headline
    contributes exactly one value at each position, so it's free, no second
    scan needed for counts.

    Args:
        headlines_dir:  Directory of the ravenpack_headlines artifact
                        (RP_STORY_ID, TIMESTAMP_UTC), one parquet per month.
        embeddings_dir: Directory of the headline_embeddings artifact
                        (RP_STORY_ID, EMBEDDING), one parquet per month with
                        the same file names as headlines_dir.
        threads:        DuckDB PRAGMA threads, applied to each month's
                        (fresh) connection.

    Returns:
        (days, sum_matrix, count_vector): days is the sorted index of
        calendar days with at least one headline; sum_matrix has shape
        (len(days), 384) in float64; count_vector has shape (len(days),).
    """
    headlines_dir = Path(headlines_dir)
    embeddings_dir = Path(embeddings_dir)
    embedding_files = sorted(embeddings_dir.glob("*.parquet"))
    if not embedding_files:
        raise RuntimeError(f"no embedding parquet files found under {embeddings_dir}")

    day_sum: dict[Any, np.ndarray] = {}
    day_cnt: dict[Any, int] = {}

    for i, emb_file in enumerate(embedding_files, 1):
        hl_file = headlines_dir / emb_file.name
        if not hl_file.exists():
            log.warning("no headlines file matching %s, skipping", emb_file.name)
            continue

        conn = duckdb.connect()
        try:
            conn.execute(f"PRAGMA threads={threads}")
            result = conn.sql(f"""
                WITH joined AS (
                    SELECT
                        CAST(h.TIMESTAMP_UTC AS DATE) AS day,
                        e.EMBEDDING
                    FROM read_parquet('{hl_file}') h
                    JOIN read_parquet('{emb_file}') e
                      ON h.RP_STORY_ID = e.RP_STORY_ID
                ),
                exploded AS (
                    SELECT
                        day,
                        UNNEST(EMBEDDING)                  AS val,
                        generate_subscripts(EMBEDDING, 1)  AS pos
                    FROM joined
                ),
                per_pos AS (
                    SELECT day, pos, SUM(val)::DOUBLE AS s, COUNT(*) AS n_vals
                    FROM exploded
                    GROUP BY day, pos
                )
                SELECT
                    day,
                    array_agg(s ORDER BY pos) AS day_sum,
                    MAX(n_vals)                AS n
                FROM per_pos
                GROUP BY day
                ORDER BY day
            """).pl()
        finally:
            conn.close()

        for row in result.iter_rows(named=True):
            d = row["day"]
            v = np.asarray(row["day_sum"], dtype=np.float64)
            if d in day_sum:
                # A day should never actually span two monthly files, but
                # merge rather than overwrite in case it ever does.
                day_sum[d] += v
                day_cnt[d] += row["n"]
            else:
                day_sum[d] = v
                day_cnt[d] = row["n"]

        if i % 20 == 0 or i == len(embedding_files):
            log.info(
                "  joined %d/%d months (%s) -> %d calendar days so far",
                i, len(embedding_files), emb_file.name, len(day_sum),
            )

    if not day_sum:
        raise RuntimeError(
            f"no headlines found joining headlines={headlines_dir} "
            f"embeddings={embeddings_dir}"
        )

    # day_sum/day_cnt are keyed by datetime.date (DuckDB's CAST(... AS DATE)
    # comes back as datetime.date via polars). pd.DatetimeIndex(...) converts
    # those keys to pd.Timestamp for the returned index, so look back up by
    # d.date() rather than d itself -- indexing day_sum[d] directly raises
    # KeyError (Timestamp != date, even for the same calendar day).
    days = pd.DatetimeIndex(sorted(day_sum.keys()))
    sum_matrix = np.stack([day_sum[d.date()] for d in days]).astype(np.float64)
    count_vector = np.array([day_cnt[d.date()] for d in days], dtype=np.int64)

    log.info(
        "%d calendar days, %d total headlines, %s -> %s",
        len(days), int(count_vector.sum()), days.min().date(), days.max().date(),
    )
    return days, sum_matrix, count_vector


# ---------------------------------------------------------------------------
# Pass 2: complete daily grid, cumulative sums, asof lookup
# ---------------------------------------------------------------------------

def build_cumulative(
    days: pd.DatetimeIndex,
    sums: np.ndarray,
    counts: np.ndarray,
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    """Zero-fill every calendar day between the first and last observed day,
    so a delay/window in weeks or months maps to an exact calendar-date
    lookup rather than an N-observations-back lookup.

    Returns (full_grid, cumsum_ext, cumcount_ext), prefixed with a zero
    row/entry: index 0 means "no history yet", index i >= 1 means
    "cumulative up to and including full_grid[i - 1]".
    """
    full_grid = pd.date_range(days.min(), days.max(), freq="D")
    n = len(full_grid)

    sum_full = np.zeros((n, EMBEDDING_DIM), dtype=np.float64)
    count_full = np.zeros(n, dtype=np.int64)
    idx = full_grid.get_indexer(days)
    sum_full[idx] = sums
    count_full[idx] = counts

    cumsum_ext = np.vstack(
        [np.zeros((1, EMBEDDING_DIM), dtype=np.float64), np.cumsum(sum_full, axis=0)]
    )
    cumcount_ext = np.concatenate([[0], np.cumsum(count_full)])
    return full_grid, cumsum_ext, cumcount_ext


def _asof_index(date: pd.Timestamp, grid_start: pd.Timestamp, n: int) -> int:
    """Index into cumsum_ext/cumcount_ext for `date`, clipped to [0, n].

    Before grid_start clips to 0 (no history); at/after the last grid day
    clips to n (all available history).
    """
    offset = (pd.Timestamp(date).normalize() - grid_start).days
    return int(np.clip(offset + 1, 0, n))


def compute_mu_asof(
    asof_dates: pd.DatetimeIndex,
    grid_start: pd.Timestamp,
    cumsum_ext: np.ndarray,
    cumcount_ext: np.ndarray,
    delay: pd.DateOffset,
    mode: str | pd.DateOffset,
) -> pl.DataFrame:
    """Apply delay + window to the cumulative arrays for each asof date.

    For each t in asof_dates: cutoff = t - delay. In expanding mode the
    window is [grid_start, cutoff); in rolling mode it is
    [cutoff - window, cutoff). A date with zero headlines in its window (no
    history yet, or an all-zero-count window) is skipped, not emitted with a
    NaN/zero row.

    The unit direction MU_HAT is formed by normalizing the pooled SUM vector
    and only then dividing by N is applied separately for MU -- normalizing
    the sum first avoids computing the mean and re-normalizing it, and gives
    the identical direction since MU_HAT = mu_t/||mu_t|| = (mu_t/n)/||mu_t/n||
    for n > 0.

    Returns:
        pl.DataFrame matching schema.MU_ASOF_SCHEMA (DATE, MU, MU_HAT, N).
    """
    n = cumcount_ext.shape[0] - 1
    rows_date: list[Any] = []
    rows_mu: list[np.ndarray] = []
    rows_mu_hat: list[np.ndarray] = []
    rows_n: list[int] = []
    n_skipped = 0
    n_degenerate = 0

    for t in asof_dates:
        cutoff = t - delay
        end = _asof_index(cutoff, grid_start, n)
        start = 0 if mode == "expanding" else _asof_index(cutoff - mode, grid_start, n)

        n_t = int(cumcount_ext[end] - cumcount_ext[start])
        if n_t == 0:
            n_skipped += 1
            continue

        mu_t = cumsum_ext[end] - cumsum_ext[start]
        norm = np.linalg.norm(mu_t)
        if norm == 0.0:
            # Contributing embeddings summed to exactly zero: no defined unit
            # direction. Never seen with real embeddings, but guards against
            # a division by zero rather than silently emitting NaN.
            n_degenerate += 1
            continue

        rows_date.append(pd.Timestamp(t).date())
        rows_mu.append((mu_t / n_t).astype(np.float32))
        rows_mu_hat.append((mu_t / norm).astype(np.float32))
        rows_n.append(n_t)

    if n_skipped:
        log.warning(
            "%d/%d asof dates skipped: no headlines available before the delay cutoff",
            n_skipped, len(asof_dates),
        )
    if n_degenerate:
        log.warning(
            "%d/%d asof dates skipped: pooled sum vector was exactly zero (no defined direction)",
            n_degenerate, len(asof_dates),
        )

    return pl.DataFrame(
        {"DATE": rows_date, "MU": rows_mu, "MU_HAT": rows_mu_hat, "N": rows_n},
        schema=MU_ASOF_SCHEMA,
    )


# ---------------------------------------------------------------------------
# Datalake-aware entry point
# ---------------------------------------------------------------------------

def mu_asof_to_datalake(
    index: "DatalakeIndex",
    delay: str,
    mode: str,
    pipeline_version: str,
    *,
    pipeline_repo: str | None = None,
    threads: int = 8,
) -> "Artifact":
    """Full mu_asof pipeline: resolve sources, run both passes, register the artifact.

    Resolves the latest ``ravenpack_headlines`` and ``headline_embeddings``
    artifacts via ``index.latest()`` -- never hardcoded paths -- and records
    both as lineage sources. Writes a single parquet
    (``mu_asof_delay-{delay}_mode-{mode}.parquet``) into the run's output
    directory, since mu_asof is one daily time series, not a monthly corpus.

    Args:
        index:              Datalake index to register the artifact in.
        delay:               Delay spec, e.g. "1M" ('Xd'/'XW'/'XM').
        mode:                Window spec: "expanding" or 'XW'/'XM'.
        pipeline_version:    Semantic version of this pipeline.
        pipeline_repo:       URL of the producing repo.
        threads:             DuckDB PRAGMA threads for pass 1 (applied
                             per-month; see compute_daily_sums).

    Returns:
        The completed Artifact.
    """
    delay_offset = parse_delay(delay)
    mode_value = parse_mode(mode)
    validate_window_after_delay(delay_offset, mode_value)

    headlines_artifact = index.latest(SOURCE_HEADLINES_KIND)
    embeddings_artifact = index.latest(SOURCE_EMBEDDINGS_KIND)

    hyperparams: dict[str, Any] = {
        "delay": delay,
        "mode": mode,
        "dim": EMBEDDING_DIM,
        "source_headlines": headlines_artifact.artifact_id,
        "source_embeddings": embeddings_artifact.artifact_id,
    }
    notes = (
        f"source_headlines={headlines_artifact.artifact_id} "
        f"source_embeddings={embeddings_artifact.artifact_id}"
    )

    with index.run(
        kind=KIND,
        pipeline=PIPELINE,
        pipeline_version=pipeline_version,
        pipeline_repo=pipeline_repo,
        hyperparams=hyperparams,
        notes=notes,
        sources=[headlines_artifact, embeddings_artifact],
        verifier=KIND,
        hash_pattern="*.parquet",
    ) as run:
        log.info("pass 1/2: daily embedding sums (SQL join, per month)")
        days, sums, counts = compute_daily_sums(
            headlines_artifact.path,
            embeddings_artifact.path,
            threads=threads,
        )

        grid, cumsum_ext, cumcount_ext = build_cumulative(days, sums, counts)

        log.info(
            "pass 2/2: mu_asof for %d asof dates, delay=%s mode=%s",
            len(days), delay, mode,
        )
        result = compute_mu_asof(
            days, grid[0], cumsum_ext, cumcount_ext, delay_offset, mode_value,
        )

        if result.is_empty():
            raise RuntimeError(
                f"mu_asof produced no rows for delay={delay} mode={mode}; "
                "every asof date was skipped (no headlines before the delay cutoff)"
            )

        out_name = f"mu_asof_delay-{delay}_mode-{mode}.parquet"
        out_path = run.out_dir / out_name
        tmp = out_path.with_suffix(".parquet.tmp")
        try:
            pq.write_table(result.to_arrow().cast(_arrow_schema()), tmp, compression="zstd")
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        tmp.replace(out_path)

        run.note(f"{len(result)} asof dates -> {out_name}")

    return index.get(run.artifact_id)


# ---------------------------------------------------------------------------
# Content verifier (registered via pyproject.toml entry points)
# ---------------------------------------------------------------------------

def verify_artifact(artifact: "Artifact") -> "list[Finding]":
    """Verify a mu_asof artifact's content matches its declared scope.

    Registered as the ``mu_asof`` entry point. Checks:
      - hyperparams record both source artifact IDs (-> ERROR if missing)
      - exactly one parquet file, readable, non-empty, matching MU_ASOF_SCHEMA
      - N is positive on every row (-> ERROR otherwise)
      - MU_HAT is unit-norm within float32 tolerance, |1 - ||mu_hat||^2| < 1e-5
        (-> ERROR otherwise)
      - DATE coverage is contiguous: no gap longer than 5 calendar days
        (weekends + holidays) (-> WARNING otherwise)
    """
    from datalake.verify import Finding, Severity

    findings: list[Finding] = []
    aid = artifact.artifact_id
    hp = artifact.meta.hyperparams

    if not hp.get("source_headlines") or not hp.get("source_embeddings"):
        findings.append(Finding(
            Severity.ERROR, aid,
            "hyperparams missing source_headlines/source_embeddings artifact id(s)",
        ))

    files = sorted(artifact.path.glob("*.parquet"))
    if not files:
        findings.append(Finding(Severity.ERROR, aid, "no parquet file found"))
        return findings
    if len(files) > 1:
        findings.append(Finding(
            Severity.WARNING, aid,
            f"expected exactly one parquet file, found {len(files)}: "
            f"{', '.join(p.name for p in files)}",
        ))

    path = files[0]
    try:
        df = pl.read_parquet(path)
    except Exception as exc:  # noqa: BLE001 - report any read failure
        findings.append(Finding(Severity.ERROR, aid, f"{path.name}: unreadable parquet: {exc}"))
        return findings

    if df.is_empty():
        findings.append(Finding(Severity.ERROR, aid, f"{path.name}: file is empty"))
        return findings

    if df.schema != MU_ASOF_SCHEMA:
        findings.append(Finding(
            Severity.ERROR, aid,
            f"{path.name}: schema mismatch (got {dict(df.schema)}, "
            f"expected {dict(MU_ASOF_SCHEMA)})",
        ))
        return findings

    n_nonpositive = int((df["N"] <= 0).sum())
    if n_nonpositive:
        findings.append(Finding(
            Severity.ERROR, aid, f"{n_nonpositive} row(s) with N <= 0",
        ))

    mu_hat = np.asarray(df["MU_HAT"].to_list(), dtype=np.float64)
    if mu_hat.size:
        sq_norms = np.sum(mu_hat**2, axis=1)
        n_bad = int(np.sum(np.abs(1.0 - sq_norms) >= _UNIT_NORM_TOL))
        if n_bad:
            findings.append(Finding(
                Severity.ERROR, aid,
                f"{n_bad} MU_HAT row(s) not unit-norm "
                f"(|1 - ||mu_hat||^2| >= {_UNIT_NORM_TOL})",
            ))

    dates = sorted(df["DATE"].to_list())
    max_gap = max(
        (cur - prev).days for prev, cur in zip(dates, dates[1:])
    ) if len(dates) > 1 else 0
    if max_gap > _MAX_DATE_GAP_DAYS:
        findings.append(Finding(
            Severity.WARNING, aid,
            f"date coverage has a gap of {max_gap} calendar days "
            f"(> {_MAX_DATE_GAP_DAYS})",
        ))

    return findings
