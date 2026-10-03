"""RavenPack corpus job for the dated reference vector (datalake kind ``mu_asof``).

The reference vector itself -- pooling (mean / min / max), delay, window and
the point-in-time rule -- is ``nlp.reference_vector.ReferenceVector``. This
module is the RavenPack I/O around it: per-day aggregates of the headline
embeddings computed in DuckDB one month at a time, then
``ReferenceVector.from_daily(...).series(days)`` for every observed day.

For each observed day t the value pools the headlines of the days in
``(cutoff - window, cutoff]``, ``cutoff = t - delay`` (delay >= 1 day, so
day t itself is never used; see ``nlp.reference_vector``).

Output layout: one parquet per artifact (a daily time series, not a monthly
corpus), named ``mu_asof_delay-{delay}_mode-{mode}[_pooling-{pooling}].parquet``
(the pooling suffix only for min / max), with columns
(``nlp.reference_vector.REFERENCE_SCHEMA``):

    DATE      Date
    MU        Array(Float32, 384)   -- raw pooled vector (mean / min / max; not normalized)
    MU_HAT    Array(Float32, 384)   -- unit direction: MU / ||MU||
    N         Int64                 -- headlines contributing to this day

MU is what mean-centering (subtract mu, renormalize) needs; MU_HAT is what
direction removal (project out mu_hat) needs. Both are stored so downstream
code picks what it needs without recomputing.

Two-pass design, adapted from the lab repo's ``mu_asof.py``:

  1. One DuckDB join+aggregate PER PARTITION (e.g. month), accumulated in Python: the
     embedding parquet has only RP_STORY_ID and EMBEDDING (no timestamp), so
     each month's embeddings file is joined to that SAME month's headlines
     file (identically named ``YYYY-MM.parquet`` in both artifacts) to get
     each headline's day. UNNEST + generate_subscripts explodes each
     embedding into (day, pos, val) rows, then a hash aggregation over
     (day, pos) gives that month's per-day, per-dimension SUM (mean pooling)
     or MIN / MAX, and counts, in one shot. Joining month-by-month rather
     than the full corpus at once is deliberate: a single cross-corpus join
     (312 x 312 files) was tried
     first and spilled past 100GB of DuckDB temp storage over GDrive FUSE.
     A calendar day never spans two monthly files, so restricting each join
     to one month is exact, not an approximation -- and keeps each join
     trivially small.
  2. ``ReferenceVector.from_daily``: prefix sums (mean) or a sparse table
     (rolling min / max) over the complete daily grid, then one vectorised
     query for every asof date. Everything here operates on a (n_days, 384)
     array -- trivial memory/compute.

A fresh ``duckdb.connect()`` is used per month rather than the module-level
``duckdb.sql()`` or one shared connection: a shared connection accumulates
state across calls and was the source of an OOM bug in an earlier pipeline.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import duckdb
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from datalake.jobs import TEMP_SUFFIX, Job, JobContext, Unit, register_job
from datalake.periods import partition_file
from nlp.reference_vector import (
    POOLINGS,
    REFERENCE_SCHEMA,
    ReferenceVector,
    parse_delay,
    parse_window,
    validate_window_after_delay,
)
from ravenpack.annotations.access import (
    DEFAULT_DAY_TZ,
    HEADLINES_KIND,
    day_tz_params,
    duckdb_day,
    latest_headlines,
    pin_utc,
    require_headlines,
)
from ravenpack.headlines.schema import EMBEDDING_DIM

if TYPE_CHECKING:
    from datalake import Artifact, DatalakeIndex
    from datalake.verify import Finding

log = logging.getLogger(__name__)

# Datalake artifact kind produced by this module, and the kinds it reads from.
KIND = "mu_asof"
SOURCE_HEADLINES_KIND = HEADLINES_KIND     # rp_headlines
SOURCE_EMBEDDINGS_KIND = "headline_embeddings"

PIPELINE = "PhD-Narrative-Finance"
PIPELINE_REPO = "https://github.com/GabinTB/PhD-Narrative-Finance"

# Contiguity tolerance for verify_artifact: weekends + a holiday or two.
_MAX_DATE_GAP_DAYS = 5

# Float32 tolerance for the unit-norm check on MU_HAT.
_UNIT_NORM_TOL = 1e-5


# ---------------------------------------------------------------------------
# Arrow schema for the ParquetWriter -- derived from REFERENCE_SCHEMA so the
# written table and the writer schema are guaranteed to agree.
# ---------------------------------------------------------------------------

_ARROW_SCHEMA: pa.Schema | None = None


def _arrow_schema() -> pa.Schema:
    global _ARROW_SCHEMA
    if _ARROW_SCHEMA is None:
        _ARROW_SCHEMA = pl.DataFrame(schema=REFERENCE_SCHEMA).to_arrow().schema
    return _ARROW_SCHEMA


# ---------------------------------------------------------------------------
# Pass 1: per-calendar-day aggregate vector and count, one month at a time, in SQL
# ---------------------------------------------------------------------------

_SQL_AGG = {"mean": "SUM(val)::DOUBLE", "min": "MIN(val)::DOUBLE", "max": "MAX(val)::DOUBLE"}


def _merge_day(pooling: str, cur: np.ndarray, new: np.ndarray) -> np.ndarray:
    if pooling == "mean":
        return cur + new
    return np.minimum(cur, new) if pooling == "min" else np.maximum(cur, new)


def compute_daily_stats(
    headlines_dir: Path,
    embeddings_dir: Path,
    pooling: str = "mean",
    threads: int = 8,
    checkpoint_dir: Path | None = None,
    day_tz: str = DEFAULT_DAY_TZ,
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

    The per-day (aggregate, count) pairs of each month are merged into a running
    Python dict keyed by day, so a calendar day spanning two monthly files (a
    ``day_tz`` other than UTC, the files being UTC months) is still exact. This
    deliberately avoids a single cross-corpus join (all headlines x all embeddings
    at once), which spills to disk past
    available memory once the corpus spans decades.

    MAX(n_vals) doubles as the per-day headline count: every headline
    contributes exactly one value at each position, so it's free, no second
    scan needed for counts.

    Args:
        headlines_dir:  Directory of the rp_headlines artifact
                        (RP_STORY_ID, TIMESTAMP_UTC in UTC), one parquet per month.
        embeddings_dir: Directory of the headline_embeddings artifact
                        (RP_STORY_ID, EMBEDDING), one parquet per month with
                        the same file names as headlines_dir.
        pooling:        "mean" (per-day SUM), "min" or "max" (per-day
                        elementwise MIN / MAX).
        threads:        DuckDB PRAGMA threads, applied to each month's
                        (fresh) connection.
        checkpoint_dir: When given, each month's per-day aggregates are saved
                        there (``YYYY-MM.npz``, atomically) as soon as its query
                        finishes, and months already saved are loaded instead
                        of queried -- an interrupted pass resumes where it stopped.
        day_tz:         Time zone of the calendar days (default UTC).

    Returns:
        (days, stat_matrix, count_vector): days is the sorted index of
        calendar days with at least one headline; stat_matrix has shape
        (len(days), 384) in float64 (sums, mins or maxs); count_vector has
        shape (len(days),).
    """
    if pooling not in POOLINGS:
        raise ValueError(f"pooling must be one of {POOLINGS}, got {pooling!r}")
    headlines_dir = Path(headlines_dir)
    embeddings_dir = Path(embeddings_dir)
    embedding_files = sorted(embeddings_dir.glob("*.parquet"))
    if not embedding_files:
        raise RuntimeError(f"no embedding parquet files found under {embeddings_dir}")

    day_stat: dict[Any, np.ndarray] = {}
    day_cnt: dict[Any, int] = {}

    if checkpoint_dir is not None:
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    n_reused = 0

    for i, emb_file in enumerate(embedding_files, 1):
        hl_file = headlines_dir / emb_file.name
        if not hl_file.exists():
            log.warning("no headlines file matching %s, skipping", emb_file.name)
            continue

        ckpt = (Path(checkpoint_dir) / f"{emb_file.stem}.npz") if checkpoint_dir else None
        if ckpt is not None and ckpt.exists():
            saved = np.load(ckpt)
            result = pl.DataFrame({"day": saved["day"].astype("datetime64[D]"),
                                   "day_stat": list(saved["stat"]), "n": saved["n"]})
            n_reused += 1
        else:
            result = _month_stats(hl_file, emb_file, pooling, threads, day_tz)
            if ckpt is not None:
                _save_checkpoint(result, ckpt)
        _accumulate(pooling, result, day_stat, day_cnt)

        if i % 20 == 0 or i == len(embedding_files):
            log.info(
                "  joined %d/%d months (%s) -> %d calendar days so far",
                i, len(embedding_files), emb_file.name, len(day_stat),
            )
    if n_reused:
        log.info("  %d month(s) reused from checkpoints", n_reused)
    return _finish_daily(day_stat, day_cnt, headlines_dir, embeddings_dir)


def _save_checkpoint(result: pl.DataFrame, ckpt: Path) -> None:
    """One partition's per-day aggregates, atomically (float64: a lossless round trip)."""
    tmp = ckpt.with_suffix(".tmp.npz")
    stat = (np.asarray(result["day_stat"].to_list(), dtype=np.float64)
            .reshape(result.height, -1) if result.height
            else np.empty((0, EMBEDDING_DIM), dtype=np.float64))
    np.savez(tmp, day=result["day"].to_numpy().astype("datetime64[D]"), stat=stat,
             n=result["n"].to_numpy())
    tmp.replace(ckpt)


def _month_stats(hl_file: Path, emb_file: Path, pooling: str, threads: int,
                 day_tz: str = DEFAULT_DAY_TZ) -> pl.DataFrame:
    """One month's per-day aggregate (day, day_stat, n) via a fresh DuckDB connection
    (session pinned to UTC; days in ``day_tz``)."""
    conn = pin_utc(duckdb.connect())
    try:
        conn.execute(f"PRAGMA threads={threads}")
        result = conn.sql(f"""
            WITH joined AS (
                SELECT
                    {duckdb_day("h.TIMESTAMP_UTC", day_tz)} AS day,
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
                SELECT day, pos, {_SQL_AGG[pooling]} AS s, COUNT(*) AS n_vals
                FROM exploded
                GROUP BY day, pos
            )
            SELECT
                day,
                array_agg(s ORDER BY pos) AS day_stat,
                MAX(n_vals)                AS n
            FROM per_pos
            GROUP BY day
            ORDER BY day
        """).pl()
    finally:
        conn.close()
    return result


def _accumulate(pooling: str, result: pl.DataFrame, day_stat: dict[Any, np.ndarray],
                day_cnt: dict[Any, int]) -> None:
    for row in result.iter_rows(named=True):
        d = row["day"]
        v = np.asarray(row["day_stat"], dtype=np.float64)
        if d in day_stat:
            # A day should never actually span two monthly files, but
            # merge rather than overwrite in case it ever does.
            day_stat[d] = _merge_day(pooling, day_stat[d], v)
            day_cnt[d] += row["n"]
        else:
            day_stat[d] = v
            day_cnt[d] = row["n"]


def _finish_daily(day_stat: dict[Any, np.ndarray], day_cnt: dict[Any, int],
                  headlines_dir: Path, embeddings_dir: Path,
                  ) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    if not day_stat:
        raise RuntimeError(
            f"no headlines found joining headlines={headlines_dir} "
            f"embeddings={embeddings_dir}"
        )

    # day_stat/day_cnt are keyed by datetime.date (DuckDB's CAST(... AS DATE)
    # comes back as datetime.date via polars). pd.DatetimeIndex(...) converts
    # those keys to pd.Timestamp for the returned index, so look back up by
    # d.date() rather than d itself -- indexing day_stat[d] directly raises
    # KeyError (Timestamp != date, even for the same calendar day).
    days = pd.DatetimeIndex(sorted(day_stat.keys()))
    stat_matrix = np.stack([day_stat[d.date()] for d in days]).astype(np.float64)
    count_vector = np.array([day_cnt[d.date()] for d in days], dtype=np.int64)

    log.info(
        "%d calendar days, %d total headlines, %s -> %s",
        len(days), int(count_vector.sum()), days.min().date(), days.max().date(),
    )
    return days, stat_matrix, count_vector


# ---------------------------------------------------------------------------
# Datalake-aware entry point
# ---------------------------------------------------------------------------

def output_name(delay: str, mode: str, pooling: str = "mean") -> str:
    """Series file name; mean keeps the historical name, min / max are suffixed."""
    suffix = "" if pooling == "mean" else f"_pooling-{pooling}"
    return f"mu_asof_delay-{delay}_mode-{mode}{suffix}.parquet"


PIPELINE_VERSION = "v0.3.0"   # v0.3.0: reads rp_headlines (UTC datetimes), day_tz;
                              # v0.2.0: pooling in the hyperparams (mean/min/max)


@register_job
class MuAsofJob(Job):
    """Dated reference vector of the corpus: one checkpoint per partition, then the series."""

    kind = KIND
    pipeline_version = PIPELINE_VERSION

    def __init__(self, headlines: Artifact, embeddings: Artifact, delay: str, mode: str, *,
                 pooling: str = "mean", threads: int = 8, temp: bool = False,
                 pipeline_version: str | None = None, day_tz: str = DEFAULT_DAY_TZ) -> None:
        validate_window_after_delay(parse_delay(delay), parse_window(mode))
        require_headlines(headlines)
        self.day_tz = day_tz
        if pooling not in POOLINGS:
            raise ValueError(f"pooling must be one of {POOLINGS}, got {pooling!r}")
        _check_same_layout(headlines, embeddings)
        from datalake.lineage import require_lineage

        require_lineage([(embeddings, {SOURCE_HEADLINES_KIND: headlines.artifact_id})],
                        what="mu_asof")
        self.headlines, self.embeddings = headlines, embeddings
        self.delay, self.mode, self.pooling = delay, mode, pooling
        self.threads, self.temp = threads, temp
        if pipeline_version is not None:
            self.pipeline_version = pipeline_version

    def params(self) -> dict[str, Any]:
        return {"delay": self.delay, "mode": self.mode, "pooling": self.pooling,
                "dim": EMBEDDING_DIM, "source_headlines": self.headlines.artifact_id,
                "source_embeddings": self.embeddings.artifact_id,
                **day_tz_params(self.day_tz)}

    def sources(self) -> list[Any]:
        return [self.headlines, self.embeddings]

    def notes(self) -> str:
        return (f"source_headlines={self.headlines.artifact_id} "
                f"source_embeddings={self.embeddings.artifact_id}")

    def units(self) -> list[Unit]:
        """Every embeddings partition with a headlines partition of the same key."""
        keys = [p.stem for p in sorted(self.embeddings.path.glob("*.parquet"))]
        missing = [k for k in keys if not (self.headlines.path / partition_file(k)).exists()]
        if missing:
            log.warning("%d embeddings partition(s) have no headlines partition, skipped "
                        "(e.g. %s)", len(missing), missing[0])
        return [Unit(k) for k in keys if k not in set(missing)]

    def is_done(self, unit: Unit, out_dir: Path) -> bool:
        return (out_dir / CHECKPOINT_DIR / f"{unit.key}.npz").exists()

    def run_unit(self, unit: Unit, ctx: JobContext) -> None:
        """Pass 1 for one partition: per-day aggregates, checkpointed."""
        name = partition_file(unit.key)
        ckpt_dir = ctx.out_dir / CHECKPOINT_DIR
        ckpt_dir.mkdir(exist_ok=True)
        result = _month_stats(self.headlines.path / name, self.embeddings.path / name,
                              self.pooling, self.threads, self.day_tz)
        _save_checkpoint(result, ckpt_dir / f"{unit.key}.npz")
        ctx.log.info("%s  %d day(s) aggregated", unit.key, result.height)

    def finalize(self, ctx: JobContext) -> None:
        """Pass 2 from the checkpoints (nothing re-queried), then drop them."""
        _build_series(ctx.run, self.headlines, self.embeddings, self.delay, self.mode,
                      self.pooling, self.threads, self.day_tz)

    @classmethod
    def from_artifact(cls, artifact: Artifact, index: DatalakeIndex, *,
                      threads: int = 8) -> MuAsofJob:
        """Delay, window, pooling and the exact sources come from the artifact."""
        hp = artifact.meta.hyperparams
        return cls(index.get(hp["source_headlines"]), index.get(hp["source_embeddings"]),
                   hp["delay"], hp["mode"], pooling=hp.get("pooling", "mean"),
                   threads=threads, temp=artifact.meta.pipeline_version.endswith(TEMP_SUFFIX),
                   day_tz=hp.get("day_tz", DEFAULT_DAY_TZ))

    @classmethod
    def add_cli_args(cls, parser: Any) -> None:
        parser.add_argument("--delay", required=True,
                            help="minimum 1 day: '1d', '2W', '3M'; mu(t) uses data before "
                                 "t - delay")
        parser.add_argument("--mode", required=True,
                            help="'expanding', or a rolling window >= 1 week: '4W', '6M'")
        parser.add_argument("--pooling", default="mean", choices=list(POOLINGS))
        parser.add_argument("--headlines-artifact", default=None,
                            help="default: the latest rp_headlines")
        parser.add_argument("--embeddings-artifact", default=None,
                            help="default: the latest headline_embeddings")
        parser.add_argument("--threads", type=int, default=8, help="DuckDB PRAGMA threads")
        parser.add_argument("--day-tz", default=DEFAULT_DAY_TZ,
                            help="time zone of the calendar days (default UTC; any other "
                                 "value enters the artifact id)")
        parser.add_argument("--temp", action="store_true", help="agent-created (__TEMP)")

    @classmethod
    def from_args(cls, args: Any, index: DatalakeIndex) -> MuAsofJob:
        return cls(*_sources(index, args.headlines_artifact, args.embeddings_artifact),
                   args.delay, args.mode, pooling=args.pooling, threads=args.threads,
                   temp=args.temp, day_tz=args.day_tz)


def _sources(index: DatalakeIndex, headlines_id: str | None,
             embeddings_id: str | None) -> tuple[Artifact, Artifact]:
    return (index.get(headlines_id) if headlines_id else latest_headlines(index),
            index.get(embeddings_id) if embeddings_id
            else index.latest(SOURCE_EMBEDDINGS_KIND))


def mu_asof_to_datalake(
    index: "DatalakeIndex",
    delay: str,
    mode: str,
    pipeline_version: str = PIPELINE_VERSION,
    *,
    pooling: str = "mean",
    pipeline_repo: str | None = None,
    threads: int = 8,
    headlines_id: str | None = None,
    embeddings_id: str | None = None,
    repo_dir: Path | None = None,
    day_tz: str = DEFAULT_DAY_TZ,
) -> "Artifact":
    """Dated reference-vector series for the corpus, registered as a ``mu_asof`` artifact
    (library entry point; runs ``MuAsofJob``).

    Sources default to the latest ``rp_headlines`` and
    ``headline_embeddings`` artifacts (never hardcoded paths) and are recorded
    as lineage. Writes a single parquet (``output_name``): the series is one
    daily time series, not a partitioned corpus.

    Args:
        index:              Datalake index to register the artifact in.
        delay:               Delay spec, e.g. "1M" ('Xd'/'XW'/'XM').
        mode:                Window spec: "expanding" or 'XW'/'XM'.
        pipeline_version:    Semantic version of this pipeline.
        pooling:             "mean", "min" or "max".
        pipeline_repo:       URL of the producing repo.
        threads:             DuckDB PRAGMA threads for pass 1 (per partition).
        headlines_id / embeddings_id: explicit sources (default: latest).
    """
    from datalake.jobs import JobRunner

    job = MuAsofJob(*_sources(index, headlines_id, embeddings_id), delay, mode,
                    pooling=pooling, threads=threads, pipeline_version=pipeline_version,
                    day_tz=day_tz)
    job.pipeline_repo = pipeline_repo
    return JobRunner(index, repo_dir=repo_dir, allow_dirty=True,
                     handle_signals=False).start(job)


CHECKPOINT_DIR = "_daily"


def _check_same_layout(headlines_artifact: Artifact, embeddings_artifact: Artifact) -> None:
    """Pass 1 pairs the two artifacts' partitions file by file (same key)."""
    from datalake.layout import layout_of

    a, b = layout_of(headlines_artifact).freq, layout_of(embeddings_artifact).freq
    if a != b:
        raise ValueError(f"headlines ({a}) and embeddings ({b}) must share a partition "
                         "frequency to be joined partition by partition")


def _build_series(run: Any, headlines_artifact: Artifact, embeddings_artifact: Artifact,
                  delay: str, mode: str, pooling: str, threads: int,
                  day_tz: str = DEFAULT_DAY_TZ) -> None:
    """Both passes into ``run.out_dir``; pass 1 is checkpointed per partition so an
    interrupted run resumes without re-querying finished partitions."""
    _check_same_layout(headlines_artifact, embeddings_artifact)
    ckpt_dir = run.out_dir / CHECKPOINT_DIR
    log.info("pass 1/2: daily embedding %s aggregates (SQL join, per month)", pooling)
    days, stats, counts = compute_daily_stats(
        headlines_artifact.path, embeddings_artifact.path, pooling=pooling, threads=threads,
        checkpoint_dir=ckpt_dir, day_tz=day_tz,
    )
    log.info("pass 2/2: reference series for %d asof dates, delay=%s mode=%s pooling=%s",
             len(days), delay, mode, pooling)
    aggregate = {"mean": "sums", "min": "mins", "max": "maxs"}[pooling]
    reference = ReferenceVector.from_daily(
        days, counts, **{aggregate: stats}, pooling=pooling, delay=delay, window=mode,
    )
    result = reference.series(days)
    if result.is_empty():
        raise RuntimeError(
            f"mu_asof produced no rows for delay={delay} mode={mode}; "
            "every asof date was skipped (no headlines before the delay cutoff)"
        )
    out_name = output_name(delay, mode, pooling)
    out_path = run.out_dir / out_name
    tmp = out_path.with_suffix(".parquet.tmp")
    try:
        pq.write_table(result.to_arrow().cast(_arrow_schema()), tmp, compression="zstd")
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(out_path)
    for f in ckpt_dir.glob("*.npz"):                  # the series is written: drop checkpoints
        f.unlink()
    ckpt_dir.rmdir()
    run.note(f"{len(result)} asof dates -> {out_name}")


def resume_mu_asof(index: DatalakeIndex, artifact_id: str, *, threads: int = 8,
                   repo_dir: Path | None = None) -> Artifact:
    """Finish a partial ``mu_asof`` artifact from its own hyperparams (``jobs resume``).

    Delay, window, pooling and the exact source artifacts come from the artifact
    (never "latest"); partitions checkpointed before the interruption are not
    re-queried; the artifact is completed in place.
    """
    from datalake.jobs import JobRunner

    art = index.get(artifact_id)
    if art.kind != KIND:
        raise ValueError(f"{artifact_id} is a {art.kind}, not a {KIND}")
    if not art.partial:
        raise ValueError(f"{artifact_id} is complete; nothing to resume")
    return JobRunner(index, repo_dir=repo_dir, allow_dirty=True, handle_signals=False
                     ).resume_job(artifact_id, MuAsofJob.from_artifact(art, index,
                                                                       threads=threads))


# ---------------------------------------------------------------------------
# Content verifier (registered via pyproject.toml entry points)
# ---------------------------------------------------------------------------

def verify_artifact(artifact: "Artifact") -> "list[Finding]":
    """Verify a mu_asof artifact's content matches its declared scope.

    Registered as the ``mu_asof`` entry point. Checks:
      - hyperparams record both source artifact IDs (-> ERROR if missing)
      - exactly one parquet file, readable, non-empty, matching REFERENCE_SCHEMA
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

    if df.schema != REFERENCE_SCHEMA:
        findings.append(Finding(
            Severity.ERROR, aid,
            f"{path.name}: schema mismatch (got {dict(df.schema)}, "
            f"expected {dict(REFERENCE_SCHEMA)})",
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
