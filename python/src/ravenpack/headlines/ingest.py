"""RavenPack Annotations 1.0 ingestion pipeline.

Reads raw per-year zip files (one CSV per month inside each zip, the vendor's
fixed layout) and writes one structured parquet per PARTITION of the chosen
``partition_freq`` (``datalake.layout``; default M = today's monthly files).

Raw layout (expected on disk):
    {raw_dir}/RavenPackAnalytics_AllEntities_1.0_{year}.zip
        {stem}/{year}-{month:02d}.csv   <- one CSV per month inside the zip

Output layout (``datalake.layout.Layout``):
    {out_dir}/{period key}.parquet     2008-01 (M), 2008-01-15 (D), 2008-W03 (W),
                                       2008Q1 (Q), 2008 (Y)

Routing raw rows to partitions: for M / Q / Y a raw month belongs to exactly one
partition and all its rows go there (M reproduces the historical files byte for
byte); for D / W rows go by their TIMESTAMP_UTC date, and a row dated outside
its raw month raises instead of being silently misplaced. A partition is
written (atomically) once every raw month it spans has been read; ranges are
rounded out to whole periods, so a partition always holds a complete period.

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
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from datalake.jobs import TEMP_SUFFIX, Job, JobContext, Unit, register_job
from datalake.layout import Layout, add_layout_args, layout_from_args, layout_from_hyperparams
from datalake.periods import parse_key, partition_file, period_key, periods
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
# Partition writers and the public entry point
# ---------------------------------------------------------------------------

def whole_periods(layout: Layout) -> Layout:
    """The layout with its range rounded out to whole periods."""
    first, last = layout.expected()[0], layout.expected()[-1]
    return Layout(layout.freq, first.first, last.last)


class _PartitionWriters:
    """Lazily opened atomic writers, one per partition key, closed when complete."""

    def __init__(self, out_dir: Path, layout: Layout, log_every: int) -> None:
        self.out_dir, self.layout, self.log_every = out_dir, layout, log_every
        self.open: dict[str, tuple[pq.ParquetWriter, Path, list[int]]] = {}
        self.written: list[str] = []

    def append(self, key: str, df: pd.DataFrame) -> None:
        if df.empty:
            return
        if key not in self.open:
            tmp = self.out_dir / f"{key}.parquet.tmp"
            self.open[key] = (pq.ParquetWriter(tmp, STRUCTURED_SCHEMA, compression="zstd"),
                              tmp, [0])
        writer, _, count = self.open[key]
        writer.write_table(_to_arrow_table(df))
        count[0] += len(df)

    def close_through(self, day: date) -> None:
        """Finalise every open partition whose last day is <= ``day``."""
        for key in [k for k in self.open if _last_day(k) <= day]:
            writer, tmp, count = self.open.pop(key)
            writer.close()
            if count[0] == 0:
                tmp.unlink(missing_ok=True)
                continue
            if self.layout.freq in ("W", "Q", "Y"):      # spans several raw months
                ids = pq.read_table(tmp, columns=["RP_STORY_ID"]).column(0)
                if len(ids) != len(ids.unique()):
                    tmp.unlink(missing_ok=True)
                    raise ValueError(f"partition {key}: a story id appears in more than one "
                                     "raw month; refusing to write duplicate stories")
            tmp.replace(self.out_dir / f"{key}.parquet")
            self.written.append(key)
            log.info("wrote %s.parquet (%d stories) | RSS=%.1fGB", key, count[0], _rss_gb())

    def abort(self) -> None:
        for writer, tmp, _ in self.open.values():
            writer.close()
            tmp.unlink(missing_ok=True)
        self.open.clear()


def _last_day(key: str) -> date:
    return parse_key(key).last


def ingest_range(
    raw_dir: Path,
    out_dir: Path,
    start_year: int | None = None,
    end_year: int | None = None,                    # inclusive
    raw_chunk_rows: int = 2_000_000,
    log_every: int = 100_000,
    overwrite: bool = False,
    *,
    layout: Layout | None = None,
    keys: set[str] | None = None,
) -> None:
    """Ingest every partition of ``layout`` (or the monthly layout of [start_year,
    end_year], the historical call) from the raw zip files.

    Args:
        raw_dir:        Directory containing the per-year zip files.
        out_dir:        Destination for structured parquets (created if absent).
        start_year / end_year: legacy monthly range (inclusive years).
        raw_chunk_rows: Raw CSV rows read per chunk.  Lower this if RSS
                        climbs during the read phase on high-volume months.
        log_every:      Emit a progress line roughly every N stories written.
        overwrite:      Re-process partitions whose output file already exists.
        layout:         Partition frequency and range (rounded out to whole periods).
        keys:           Only these partitions of ``layout`` (default: all of them).
    """
    if layout is None:
        if start_year is None or end_year is None:
            raise ValueError("give a layout or start_year/end_year")
        layout = Layout("M", date(start_year, 1, 1), date(end_year, 12, 31))
    layout = whole_periods(layout)
    out_dir.mkdir(parents=True, exist_ok=True)

    parts = layout.expected()
    todo = {p.key for p in parts if (keys is None or p.key in keys)
            and (overwrite or not layout.path_of(out_dir, p).exists())}
    asked = len(parts) if keys is None else len(keys)
    log.info("ingest: %d/%d %s partitions already done, %d to process (overwrite=%s)",
             asked - len(todo), asked, layout.freq, len(todo), overwrite)
    if not todo:
        log.info("nothing to do")
        return
    raw_months = sorted({(m.first.year, m.first.month) for p in parts if p.key in todo
                         for m in periods(p.first, p.last, "M")})

    writers = _PartitionWriters(out_dir, layout, log_every)
    open_year: int | None = None
    zf: zipfile.ZipFile | None = None
    t0, n_total = time.monotonic(), 0
    try:
        for i, (year, month) in enumerate(raw_months, 1):
            tag = f"[{i}/{len(raw_months)}] {year}-{month:02d}"
            month_last = periods(date(year, month, 1), date(year, month, 1), "M")[0].last
            zip_path = raw_dir / _ZIP_NAME.format(year=year)
            if not zip_path.exists():
                log.warning("%s  skip: zip not found at %s", tag, zip_path)
                writers.close_through(month_last)
                continue
            if open_year != year:
                if zf is not None:
                    zf.close()
                zf = zipfile.ZipFile(zip_path)
                open_year = year
            member = _MEMBER_NAME.format(stem=zip_path.stem, year=year, month=month)
            if member not in zf.namelist():
                log.warning("%s  skip: member %s not in zip", tag, member)
                writers.close_through(month_last)
                continue

            log.info("%s  start (raw_chunk_rows=%d)", tag, raw_chunk_rows)
            month_key = f"{year}-{month:02d}"
            for df in _iter_month_deduped(zf, member, raw_chunk_rows, tag):
                if df.empty:
                    continue
                if layout.freq in ("M", "Q", "Y"):       # the raw month's own partition
                    key = period_key(date(year, month, 1), layout.freq)
                    if key in todo:
                        writers.append(key, df)
                else:                                    # D / W: by the row's date
                    days = df["TIMESTAMP_UTC"].astype(str).str.slice(0, 10)
                    outside = days.str.slice(0, 7) != month_key
                    if outside.any():
                        raise ValueError(f"{tag}: {int(outside.sum())} row(s) dated outside "
                                         f"{month_key} (e.g. {days[outside].iloc[0]})")
                    for day_str, part in df.groupby(days, sort=True):
                        key = period_key(date.fromisoformat(day_str), layout.freq)
                        if key in todo:
                            writers.append(key, part)
                n_total += len(df)
            writers.close_through(month_last)
        writers.close_through(layout.end)
    except BaseException:
        writers.abort()
        raise
    finally:
        if zf is not None:
            zf.close()

    elapsed = time.monotonic() - t0
    log.info("ingest done: %d partition(s) written to %s (%d stories, %.0f/s)",
             len(writers.written), out_dir, n_total, n_total / max(elapsed, 1e-9))


# ---------------------------------------------------------------------------
# The Job (datalake.jobs) and the datalake-aware entry point
# ---------------------------------------------------------------------------

PIPELINE_VERSION = "v0.2.0"   # v0.2.0: partition layout (partition_freq/start/end) in the id
RAW_SUBDIR = ("RavenPack", "headlines_edge_v1.0")


def default_raw_dir() -> Path:
    """``$RAW_DATA_PATH/RavenPack/headlines_edge_v1.0``."""
    root = os.environ.get("RAW_DATA_PATH")
    if not root:
        raise ValueError("RAW_DATA_PATH is not set; pass the raw directory explicitly")
    return Path(root).joinpath(*RAW_SUBDIR)


def _raw_months(first: date, last: date) -> frozenset[tuple[int, int]]:
    return frozenset((m.first.year, m.first.month) for m in periods(first, last, "M"))


@register_job
class IngestJob(Job):
    """RavenPack Annotations zips -> ravenpack_headlines partitions."""

    kind = KIND
    pipeline_version = PIPELINE_VERSION

    def __init__(self, raw_dir: Path, layout: Layout, *, temp: bool = False,
                 pipeline_version: str | None = None, raw_chunk_rows: int = 2_000_000,
                 log_every: int = 100_000) -> None:
        self.raw_dir, self.layout, self.temp = Path(raw_dir), whole_periods(layout), temp
        if pipeline_version is not None:
            self.pipeline_version = pipeline_version
        self.raw_chunk_rows, self.log_every = raw_chunk_rows, log_every
        self._read: set[str] = set()        # partitions whose raw months were read this run

    def params(self) -> dict:
        return self.layout.hyperparams()

    def notes(self) -> str:
        return f"raw_dir={self.raw_dir} columns={','.join(RAW_COLUMNS)}"

    def units(self) -> list[Unit]:
        return [Unit(p.key) for p in self.layout.expected()]

    def is_done(self, unit: Unit, out_dir: Path) -> bool:
        return (out_dir / partition_file(unit.key)).exists()

    def run_unit(self, unit: Unit, ctx: JobContext) -> None:
        """Read the raw months of ``unit`` once and write every partition they complete.

        For M / Q / Y that is the unit alone; for D / W it is every day / week
        of those raw months, so a raw CSV is read once, not once per day. A
        partition with no stories writes no file (as before); it is not re-read
        in the same run.
        """
        if unit.key in self._read:
            return
        need = _raw_months(*_bounds(unit.key))
        group = {p.key for p in self.layout.expected()
                 if _raw_months(p.first, p.last) <= need
                 and not (ctx.out_dir / partition_file(p.key)).exists()}
        ingest_range(raw_dir=self.raw_dir, out_dir=ctx.out_dir,
                     raw_chunk_rows=self.raw_chunk_rows, log_every=self.log_every,
                     overwrite=False, layout=self.layout, keys=group)
        self._read |= group

    def finalize(self, ctx: JobContext) -> None:
        n_parts = len(self.layout.existing(ctx.out_dir))
        if n_parts == 0:
            raise RuntimeError(
                f"ingest produced no output for {self.layout.start}..{self.layout.end}; "
                f"check that {self.raw_dir} contains the expected zip files")
        expected = len(self.layout.expected())
        if n_parts < expected:
            ctx.log.warning("%d/%d partitions present: some raw months are missing or empty",
                            n_parts, expected)
        ctx.note(f"{n_parts} {self.layout.freq} partition(s) ingested")

    @classmethod
    def from_artifact(cls, artifact: Artifact, index: DatalakeIndex, *,
                      raw_dir: Path | str | None = None, **kwargs) -> IngestJob:
        layout = layout_from_hyperparams(artifact.meta.hyperparams)
        if layout.start is None or layout.end is None:
            raise ValueError(f"{artifact.artifact_id} declares no range in its hyperparams")
        version = artifact.meta.pipeline_version
        return cls(Path(raw_dir) if raw_dir else default_raw_dir(), layout,
                   temp=version.endswith(TEMP_SUFFIX), **kwargs)

    @classmethod
    def add_cli_args(cls, parser) -> None:
        add_layout_args(parser)
        parser.add_argument("--raw-dir", default=None,
                            help="default: $RAW_DATA_PATH/RavenPack/headlines_edge_v1.0")
        parser.add_argument("--raw-chunk-rows", type=int, default=2_000_000)
        parser.add_argument("--log-every", type=int, default=100_000)
        parser.add_argument("--temp", action="store_true", help="agent-created (__TEMP)")

    @classmethod
    def from_args(cls, args, index: DatalakeIndex) -> IngestJob:
        raw_dir = Path(args.raw_dir) if args.raw_dir else default_raw_dir()
        if not raw_dir.is_dir():
            raise ValueError(f"raw directory does not exist: {raw_dir}")
        return cls(raw_dir, layout_from_args(args), temp=args.temp,
                   raw_chunk_rows=args.raw_chunk_rows, log_every=args.log_every)


def _bounds(key: str) -> tuple[date, date]:
    period = parse_key(key)
    return period.first, period.last


def ingest_to_datalake(
    index: DatalakeIndex,
    raw_dir: Path,
    start_year: int | None = None,
    end_year: int | None = None,
    *,
    layout: Layout | None = None,
    pipeline: str = "PhD-Narrative-Finance",
    pipeline_version: str = PIPELINE_VERSION,
    pipeline_repo: str | None = None,
    repo_dir: Path | None = None,
    raw_chunk_rows: int = 2_000_000,
    log_every: int = 100_000,
) -> Artifact:
    """Ingest into a new datalake artifact through ``IngestJob`` (library entry point).

    The raw zips are not an artifact: vintaged source data outside the tree,
    recorded by ``raw_dir`` in the run notes. Datalake artifacts are immutable,
    so there is no ``overwrite``: an interrupted run is finished with
    ``jobs resume <artifact_id>`` (or ``JobRunner.resume``).

    Args:
        index:            Datalake index to register the artifact in.
        raw_dir:          Directory containing the per-year zip files.
        start_year / end_year: legacy monthly range (inclusive years).
        layout:           Partition frequency and range (default: monthly over
                          [start_year, end_year]); recorded in the hyperparams.
        pipeline / pipeline_repo: provenance (default: the Job's).
        pipeline_version: Semantic version of this pipeline.
        repo_dir:         Directory to read the git SHA from (default cwd).
        raw_chunk_rows:   Raw CSV rows read per chunk.
        log_every:        Progress line frequency, in stories written.
    """
    from datalake.jobs import JobRunner

    if layout is None:
        if start_year is None or end_year is None:
            raise ValueError("give a layout or start_year/end_year")
        layout = Layout("M", date(start_year, 1, 1), date(end_year, 12, 31))
    job = IngestJob(raw_dir, layout, pipeline_version=pipeline_version,
                    raw_chunk_rows=raw_chunk_rows, log_every=log_every)
    job.pipeline = pipeline
    if pipeline_repo is not None:
        job.pipeline_repo = pipeline_repo
    runner = JobRunner(index, repo_dir=repo_dir, allow_dirty=True, handle_signals=False)
    return runner.start(job)

# ---------------------------------------------------------------------------
# Content verifiers (registered via pyproject.toml entry points)
# ---------------------------------------------------------------------------

def verify_artifact(artifact: "Artifact") -> list:
    """Verify a ravenpack_headlines artifact's content matches its declared scope.

    Registered as the `ravenpack_headlines` entry point.  Checks:
      - one parquet per partition of the declared layout, none missing
      - no parquet outside the declared range
      - a sample of files are non-empty and match STRUCTURED_SCHEMA
    """
    import polars as pl

    from datalake.verify import Finding, Severity

    findings: list = []
    aid = artifact.artifact_id
    layout = layout_from_hyperparams(artifact.meta.hyperparams)
    if layout.start is None or layout.end is None:
        findings.append(Finding(
            Severity.WARNING, aid, "no declared range in hyperparams; cannot verify coverage",
        ))
        return findings

    present = {p.name for p in artifact.path.glob("*.parquet")}
    expected = {f"{p.key}.parquet" for p in layout.expected()}

    missing = expected - present
    if missing:
        # Missing partitions are a WARNING not ERROR: raw zips genuinely lack some
        # months, but a complete artifact should surface the gap.
        findings.append(Finding(
            Severity.WARNING, aid,
            f"{len(missing)} of {len(expected)} declared {layout.freq} partitions missing: "
            f"{', '.join(sorted(missing)[:6])}"
            + (" ..." if len(missing) > 6 else ""),
        ))

    unexpected = present - expected
    if unexpected:
        findings.append(Finding(
            Severity.ERROR, aid,
            f"{len(unexpected)} parquet(s) outside declared range "
            f"[{layout.start}, {layout.end}]: {', '.join(sorted(unexpected)[:6])}",
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
