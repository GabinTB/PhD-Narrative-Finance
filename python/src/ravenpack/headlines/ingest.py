"""RavenPack Annotations 1.0 ingestion pipeline.

Reads raw per-year zip files (one CSV per month inside each zip) and writes
one structured parquet per month under the output directory.

Raw layout (expected on disk):
    {raw_dir}/RavenPackAnalytics_AllEntities_1.0_{year}.zip
        {stem}/{year}-{month:02d}.csv   <- one CSV per month inside the zip

Output layout:
    {out_dir}/{year}-{month:02d}.parquet

Processing per month:
    1. Stream-read the CSV in chunks (raw_chunk_rows at a time) to bound RAM.
    2. Dedup to one row per RP_STORY_ID: entity-level columns collapse to
       aligned lists (RP_ENTITY_ID, ENTITY_TYPE, ENTITY_NAME,
       EVENT_SENTIMENT_SCORE); scalar columns keep first-occurrence values.
    3. Write with zstd compression, atomically (tmp + rename).

Resumable: existing output files are skipped unless overwrite=True.

Column schema is defined in schema.py.  To add columns: edit RAW_SCHEMA
there -- this module derives its usecols list automatically.
"""
from __future__ import annotations

import logging
import os
import time
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ravenpack.headlines.schema import (
    ENTITY_LIST_COLS,
    RAW_COLUMNS,
    RAW_SCHEMA,
    SCALAR_COLS,
    STRUCTURED_SCHEMA,
)

if TYPE_CHECKING:  # avoids a hard datalake dependency for the pure path
    from datalake import Artifact, DatalakeIndex

log = logging.getLogger(__name__)

# Datalake artifact kind produced by this module.
KIND = "ravenpack_headlines"

_PAGE = os.sysconf("SC_PAGE_SIZE")

# Zip / member naming convention for RavenPack Annotations 1.0.
_ZIP_NAME = "RavenPackAnalytics_AllEntities_1.0_{year}.zip"
_MEMBER_NAME = "{stem}/{year}-{month:02d}.csv"


# ---------------------------------------------------------------------------
# Memory monitoring
# ---------------------------------------------------------------------------

def _rss_gb() -> float:
    """Current resident memory in GB.  Falls when memory is released."""
    with open("/proc/self/statm") as fh:
        return int(fh.read().split()[1]) * _PAGE / 1e9


# ---------------------------------------------------------------------------
# Deduplication: one row per RP_STORY_ID
# ---------------------------------------------------------------------------

def _dedup_stories(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse entity-event rows to one row per RP_STORY_ID.

    Entity-level columns become aligned lists; scalar columns keep the
    first occurrence.  The list order within each story follows TIMESTAMP_UTC
    so it is deterministic across runs.

    The caller is responsible for passing a frame whose RP_STORY_ID values
    are complete (not split across chunk boundaries).
    """
    if df.empty:
        return df

    agg: dict[str, object] = {col: list for col in ENTITY_LIST_COLS}
    agg.update({col: "first" for col in SCALAR_COLS})
    # TIMESTAMP_UTC is the sort key above but is excluded from SCALAR_COLS
    # (it's one of the two explicit leading columns) -- pandas' dict-form
    # .agg() drops any column not listed, and reindex() below would then
    # silently fill it back in as all-null rather than raising. Aggregate it
    # explicitly, taking the first (earliest, given the sort) timestamp.
    agg["TIMESTAMP_UTC"] = "first"

    return (
        df.sort_values("TIMESTAMP_UTC")
        .groupby("RP_STORY_ID", sort=False, as_index=False)
        .agg(agg)
        .reindex(columns=df.columns)
    )


# ---------------------------------------------------------------------------
# Streaming reader: one month, chunk by chunk
# ---------------------------------------------------------------------------

def _iter_month_deduped(
    zf: zipfile.ZipFile,
    member: str,
    raw_chunk_rows: int,
    tag: str,
) -> Iterator[pd.DataFrame]:
    """Stream-read one month's CSV, yield deduped story frames chunk by chunk.

    Reads raw_chunk_rows raw rows at a time.  All rows of a single
    RP_STORY_ID share a TIMESTAMP_UTC and are contiguous in RavenPack files,
    so the trailing story from each chunk is carried forward and prepended to
    the next chunk to keep groupby exact across chunk boundaries.

    Yields at least one non-empty frame per chunk where deduplicated rows
    exist.  Callers may see multiple yields per chunk if chunk_rows is large.
    """
    carry: pd.DataFrame | None = None
    n_raw = 0

    # Cast numeric columns after read to avoid object-typed columns when
    # values are missing in early rows.  Non-numeric columns left as str.
    dtype_map: dict[str, object] = {
        col: "string"
        for col, arrow_type in RAW_SCHEMA.items()
        if str(arrow_type) == "string"
    }

    with zf.open(member) as fh:
        reader = pd.read_csv(
            fh,
            usecols=RAW_COLUMNS,
            dtype=dtype_map,
            low_memory=False,
            chunksize=raw_chunk_rows,
        )
        for block in reader:
            n_raw += len(block)

            # Cast non-string columns explicitly.
            for col, arrow_type in RAW_SCHEMA.items():
                if str(arrow_type) != "string" and col in block.columns:
                    block[col] = pd.to_numeric(block[col], errors="coerce")

            if carry is not None and not carry.empty:
                block = pd.concat([carry, block], ignore_index=True)
                carry = None

            if block.empty:
                continue

            # Hold back the last story: its remaining rows may be in the
            # next chunk.
            last_id = block["RP_STORY_ID"].iloc[-1]
            tail_mask = block["RP_STORY_ID"] == last_id
            carry = block.loc[tail_mask].copy()
            body = block.loc[~tail_mask]

            if not body.empty:
                out = _dedup_stories(body)
                log.info(
                    "%s  read %d raw rows -> %d stories | RSS=%.1fGB",
                    tag, n_raw, len(out), _rss_gb(),
                )
                yield out
                del out
            del block, body

    # Flush carry (last story of the month).
    if carry is not None and not carry.empty:
        yield _dedup_stories(carry)


# ---------------------------------------------------------------------------
# Arrow table builder
# ---------------------------------------------------------------------------

def _to_arrow_table(df: pd.DataFrame) -> pa.Table:
    """Convert a deduped pandas frame to an Arrow table matching STRUCTURED_SCHEMA.

    Entity-level columns are already Python lists (from groupby agg(list));
    PyArrow converts them to list<T> arrays directly.  Scalar columns are
    cast to their declared types.

    Types are resolved from RAW_SCHEMA (the source-of-truth dict) rather than
    by looking up fields in STRUCTURED_SCHEMA by name to avoid any dependency
    on field-position assumptions.
    """
    arrays: dict[str, pa.Array] = {}

    # Scalar columns (including the two explicit leading columns).
    # Pandas nullable string/object dtypes after groupby agg("first") may hold
    # pd.NA or numpy nan; convert to Python-native types before Arrow cast.
    scalar_cols = ["TIMESTAMP_UTC", "RP_STORY_ID"] + SCALAR_COLS
    for col in scalar_cols:
        raw_type = RAW_SCHEMA[col]
        series = df[col]
        if pa.types.is_string(raw_type) or pa.types.is_large_string(raw_type):
            # Convert to Python str, replace NA/NaN with None.
            values = [
                None if (v is None or (isinstance(v, float) and pd.isna(v)))
                else str(v)
                for v in series
            ]
        else:
            # Numeric: coerce to Python float/int, NA -> None.
            values = [
                None if (v is None or (isinstance(v, float) and pd.isna(v)))
                else v
                for v in series
            ]
        arrays[col] = pa.array(values, type=raw_type)

    # Entity-level list columns: each cell is already a Python list after dedup.
    for col in ENTITY_LIST_COLS:
        inner_type = RAW_SCHEMA[col]
        list_type = pa.list_(inner_type)

        def _cast_row(row: object) -> list:
            if not isinstance(row, list):
                return []
            result = []
            for v in row:
                if v is None or (isinstance(v, float) and pd.isna(v)):
                    result.append(None)
                else:
                    result.append(v)
            return result

        arrays[col] = pa.array(
            [_cast_row(row) for row in df[col]],
            type=list_type,
        )

    return pa.table(arrays, schema=STRUCTURED_SCHEMA)


# ---------------------------------------------------------------------------
# Month writer
# ---------------------------------------------------------------------------

def _write_month(
    df_iter: Iterator[pd.DataFrame],
    out_path: Path,
    tag: str,
    log_every: int,
) -> int:
    """Write one month's structured parquet from a stream of deduped frames.

    Atomic: writes to a .tmp file first, renames on success.  A partial run
    never leaves a file that looks complete.
    """
    tmp = out_path.with_suffix(".parquet.tmp")
    writer = pq.ParquetWriter(tmp, STRUCTURED_SCHEMA, compression="zstd")
    n_written = 0
    next_report = log_every
    t0 = time.monotonic()

    try:
        for df in df_iter:
            if df.empty:
                continue
            writer.write_table(_to_arrow_table(df))
            n_written += len(df)
            if n_written >= next_report:
                rate = n_written / max(time.monotonic() - t0, 1e-9)
                log.info(
                    "%s  %d stories | %.0f/s | RSS=%.1fGB",
                    tag, n_written, rate, _rss_gb(),
                )
                next_report = n_written + log_every
            del df
    except BaseException:
        writer.close()
        tmp.unlink(missing_ok=True)
        raise

    writer.close()

    if n_written == 0:
        tmp.unlink(missing_ok=True)
        return 0

    tmp.replace(out_path)
    return n_written


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def ingest_range(
    raw_dir: Path,
    out_dir: Path,
    start_year: int,
    end_year: int,                    # inclusive
    raw_chunk_rows: int = 2_000_000,
    log_every: int = 100_000,
    overwrite: bool = False,
) -> None:
    """Ingest all months in [start_year, end_year] from raw zip files.

    Args:
        raw_dir:        Directory containing the per-year zip files.
        out_dir:        Destination for structured parquets (created if absent).
        start_year:     First year to process (inclusive).
        end_year:       Last year to process (inclusive).
        raw_chunk_rows: Raw CSV rows read per chunk.  Lower this if RSS
                        climbs during the read phase on high-volume months.
        log_every:      Emit a progress line roughly every N stories written.
        overwrite:      Re-process months whose output file already exists.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    all_months = [
        (y, m)
        for y in range(start_year, end_year + 1)
        for m in range(1, 13)
    ]
    if overwrite:
        todo = all_months
    else:
        todo = [
            (y, m)
            for y, m in all_months
            if not (out_dir / f"{y}-{m:02d}.parquet").exists()
        ]

    n_all = len(all_months)
    n_done = n_all - len(todo)
    log.info(
        "ingest: %d/%d months already done, %d to process (overwrite=%s)",
        n_done, n_all, len(todo), overwrite,
    )
    if not todo:
        log.info("nothing to do")
        return

    total = len(todo)
    processed = 0
    open_year: int | None = None
    zf: zipfile.ZipFile | None = None

    try:
        for i, (year, month) in enumerate(todo, 1):
            tag = f"[{i}/{total}] {year}-{month:02d}"

            zip_path = raw_dir / _ZIP_NAME.format(year=year)
            if not zip_path.exists():
                log.warning("%s  skip: zip not found at %s", tag, zip_path)
                continue

            if open_year != year:
                if zf is not None:
                    zf.close()
                zf = zipfile.ZipFile(zip_path)
                open_year = year

            stem = zip_path.stem
            member = _MEMBER_NAME.format(stem=stem, year=year, month=month)
            if member not in zf.namelist():
                log.warning("%s  skip: member %s not in zip", tag, member)
                continue

            out_path = out_dir / f"{year}-{month:02d}.parquet"
            t_month = time.monotonic()
            log.info(
                "%s  start (raw_chunk_rows=%d)",
                tag, raw_chunk_rows,
            )

            n = _write_month(
                _iter_month_deduped(zf, member, raw_chunk_rows, tag),
                out_path,
                tag,
                log_every,
            )

            if n == 0:
                log.warning("%s  skip: no stories after dedup", tag)
                continue

            elapsed = time.monotonic() - t_month
            log.info(
                "%s  wrote %s (%d stories in %.0fs, %.0f/s) | RSS=%.1fGB",
                tag, out_path.name, n, elapsed,
                n / max(elapsed, 1e-9), _rss_gb(),
            )
            processed += 1

    finally:
        if zf is not None:
            zf.close()

    log.info("ingest done: %d/%d months written to %s", processed, total, out_dir)


# ---------------------------------------------------------------------------
# Datalake-aware entry point
# ---------------------------------------------------------------------------

def ingest_to_datalake(
    index: DatalakeIndex,
    raw_dir: Path,
    start_year: int,
    end_year: int,
    *,
    pipeline: str = "PhD-Narrative-Finance",
    pipeline_version: str,
    pipeline_repo: str | None = None,
    repo_dir: Path | None = None,
    raw_chunk_rows: int = 2_000_000,
    log_every: int = 100_000,
) -> Artifact:
    """Ingest into a new datalake artifact, with provenance recorded.

    Wraps `ingest_range` in a datalake run: the output directory is allocated
    by the index, sidecars and hashes are written on completion, and a crash
    leaves the artifact registered as partial rather than silently absent.

    Note there is no `overwrite` parameter.  Datalake artifacts are immutable:
    re-ingesting produces a new artifact rather than mutating an existing one.
    Resumption within a single run still works (`ingest_range` skips months
    already written into this run's output directory), so an interrupted
    ingest can be continued by re-entering the same artifact directory
    manually if needed.

    The raw zips themselves are not registered as a datalake artifact: they
    are vintaged source data living outside the tree.  Their identity is
    recorded in the run's hyperparameters via `raw_dir`, and the checksums of
    what was produced from them are recorded in the sidecar.

    Args:
        index:            Datalake index to register the artifact in.
        raw_dir:          Directory containing the per-year zip files.
        start_year:       First year to ingest (inclusive).
        end_year:         Last year to ingest (inclusive).
        pipeline:         Producing repo name, recorded in provenance.
        pipeline_version: Semantic version of this pipeline.
        pipeline_repo:    URL of the producing repo.
        repo_dir:         Directory to read the git SHA from (default cwd).
        raw_chunk_rows:   Raw CSV rows read per chunk.
        log_every:        Progress line frequency, in stories written.

    Returns:
        The completed Artifact.
    """
    hyperparams = {
        "start_year": start_year,
        "end_year": end_year,
        # "raw_dir": str(raw_dir),
        # "columns": ",".join(RAW_COLUMNS),
    }
    notes = f"raw_dir={raw_dir} columns={','.join(RAW_COLUMNS)}"

    with index.run(
        kind=KIND,
        pipeline=pipeline,
        pipeline_version=pipeline_version,
        pipeline_repo=pipeline_repo,
        repo_dir=repo_dir,
        hyperparams=hyperparams,
        notes=notes,
        verifier=KIND,
        hash_pattern="*.parquet",
    ) as run:
        ingest_range(
            raw_dir=raw_dir,
            out_dir=run.out_dir,
            start_year=start_year,
            end_year=end_year,
            raw_chunk_rows=raw_chunk_rows,
            log_every=log_every,
            overwrite=False,
        )
        n_months = len(list(run.out_dir.glob("*.parquet")))
        if n_months == 0:
            raise RuntimeError(
                f"ingest produced no output for {start_year}-{end_year}; "
                f"check that {raw_dir} contains the expected zip files"
            )
        run.note(f"{n_months} monthly files ingested")

    return index.get(run.artifact_id)

# ---------------------------------------------------------------------------
# Content verifiers (registered via pyproject.toml entry points)
# ---------------------------------------------------------------------------

def verify_artifact(artifact: "Artifact") -> list:
    """Verify a ravenpack_headlines artifact's content matches its declared scope.

    Registered as the `ravenpack_headlines` entry point.  Checks:
      - one parquet per month in [start_year, end_year], none missing
      - no parquet outside the declared range
      - a sample of files are non-empty and match STRUCTURED_SCHEMA
    """
    import polars as pl

    from datalake.verify import Finding, Severity

    findings: list = []
    aid = artifact.artifact_id
    hp = artifact.meta.hyperparams
    start_year = hp.get("start_year")
    end_year = hp.get("end_year")

    if start_year is None or end_year is None:
        findings.append(Finding(
            Severity.WARNING, aid,
            "no start_year/end_year in hyperparams; cannot verify date coverage",
        ))
        return findings

    present = {p.name for p in artifact.path.glob("*.parquet")}
    expected = {
        f"{y}-{m:02d}.parquet"
        for y in range(start_year, end_year + 1)
        for m in range(1, 13)
    }

    missing = expected - present
    if missing:
        # Missing months are a WARNING not ERROR: raw zips genuinely lack some
        # months, but a complete artifact should surface the gap.
        findings.append(Finding(
            Severity.WARNING, aid,
            f"{len(missing)} of {len(expected)} declared months missing: "
            f"{', '.join(sorted(missing)[:6])}"
            + (" ..." if len(missing) > 6 else ""),
        ))

    unexpected = present - expected
    if unexpected:
        findings.append(Finding(
            Severity.ERROR, aid,
            f"{len(unexpected)} parquet(s) outside declared range "
            f"[{start_year}, {end_year}]: {', '.join(sorted(unexpected)[:6])}",
        ))

    # Deep-check a sample: first, middle, last present file.
    sample_names = sorted(present)
    if sample_names:
        sample = {
            sample_names[0],
            sample_names[len(sample_names) // 2],
            sample_names[-1],
        }
        expected_cols = set(STRUCTURED_SCHEMA.names)
        for name in sorted(sample):
            path = artifact.path / name
            try:
                df = pl.read_parquet(path)
            except Exception as exc:
                findings.append(Finding(
                    Severity.ERROR, aid, f"{name}: unreadable parquet: {exc}",
                ))
                continue
            if df.is_empty():
                findings.append(Finding(
                    Severity.ERROR, aid, f"{name}: file is empty",
                ))
                continue
            actual_cols = set(df.columns)
            if actual_cols != expected_cols:
                findings.append(Finding(
                    Severity.ERROR, aid,
                    f"{name}: schema mismatch "
                    f"(missing={expected_cols - actual_cols}, "
                    f"extra={actual_cols - expected_cols})",
                ))

    return findings


def verify_canonical(artifact: "Artifact") -> list:
    """Verify a canonical_narratives artifact has the expected structure.

    Registered as the `canonical_narratives` entry point.  Hand-curated
    taxonomy artifacts must carry a taxonomy.yaml and a grounding_table.csv;
    their absence means the artifact is incomplete regardless of hash state.
    """
    from datalake.verify import Finding, Severity

    findings: list = []
    aid = artifact.artifact_id

    required = {"taxonomy.yaml", "grounding_table.csv"}
    present = {p.name for p in artifact.path.iterdir() if p.is_file()}
    missing = required - present
    if missing:
        findings.append(Finding(
            Severity.ERROR, aid,
            f"canonical narratives artifact missing required file(s): "
            f"{', '.join(sorted(missing))}",
        ))

    # A taxonomy.yaml that is present but empty is worse than absent.
    tax = artifact.path / "taxonomy.yaml"
    if tax.is_file() and tax.stat().st_size == 0:
        findings.append(Finding(
            Severity.ERROR, aid, "taxonomy.yaml is empty",
        ))

    return findings
