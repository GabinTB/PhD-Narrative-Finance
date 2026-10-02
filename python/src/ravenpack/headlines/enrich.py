"""Add RELEVANCE and RP_SOURCE_ID to an existing ``ravenpack_headlines`` artifact, in place.

The headlines artifact predates these two columns, and the embeddings, sentiment,
mu_asof and narrative_daily artifacts all cite its id. Re-ingesting would create a new
id and break that lineage, so the columns are added to the SAME artifact as a new
execution (``JobRunner.update_job`` -> ``DatalakeIndex.run(extend=...)``): its id and
hyperparams are unchanged, every partition is rewritten with the new columns, the
directory is rehashed, and the execution history records the enrichment with its own
commit and version (the artifact's major.minor, next patch: existing columns are
unchanged, which the per-partition check below proves).

Per partition (one unit), the raw months it spans are read once, in file order
(``sentiment_vendor.raw_month_batches``), and each story's raw rows are collapsed to
lists in that order. The ingest wrote its lists in the same order (checked on 2003-06
and 2014-06: every story), so:

    rebuilt RP_ENTITY_ID / ENTITY_TYPE / ENTITY_NAME lists   must equal the stored ones
    rebuilt EVENT_SENTIMENT_SCORE                           must equal at its two raw
                                                            decimals (the stored float32
                                                            went through another parser)
    RELEVANCE (UInt8, 0-100)                                then aligned by construction
    RP_SOURCE_ID                                            single-valued per story

The checked lists are compared through a 64-bit hash per story and list (one seed for
both sides), so a month's rebuilt lists are never all held in memory: only the hashes,
RELEVANCE and RP_SOURCE_ID are (with the lists themselves, 2014-06 and its 23M raw rows
peaked at 9.8 GB, about 3x more for the 2022-2024 months).

Any mismatch raises, naming the partition and an example story: nothing is guessed. An
M / Q / Y partition must hold exactly the raw months' stories; a D / W partition holds
a subset of them.

The swap is never destructive. The enriched file is written to the staging directory,
read back and checked against the original (row order, every existing column, list
lengths); only then does the original move to the backup directory and the enriched
file take its place. A crash between the two moves leaves the original in the backup
directory, where the next run reads it from. Backups are never deleted here.

    uv run python -m ravenpack.headlines.enrich start  <headlines id> [--raw-dir DIR]
    uv run python -m ravenpack.headlines.enrich resume <headlines id> [--raw-dir DIR]
    uv run jobs status <headlines id>    /    uv run jobs pause <headlines id>

``jobs resume`` cannot finish an interrupted enrichment (it rebuilds an IngestJob of
another major.minor and refuses); use ``resume`` above.
"""
from __future__ import annotations

import argparse
import logging
import os
import re
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from datalake.jobs import TEMP_SUFFIX, Job, JobContext, JobError, Unit
from datalake.layout import layout_from_hyperparams
from datalake.periods import Period, parse_key, partition_file, periods
from ravenpack.headlines.ingest import KIND
from ravenpack.headlines.schema import STRUCTURED_SCHEMA, STRUCTURED_SCHEMA_POLARS
from ravenpack.headlines.sentiment_vendor import BLOCK_BYTES, raw_month_batches

if TYPE_CHECKING:
    from datalake import Artifact, DatalakeIndex

log = logging.getLogger(__name__)

ID_COL = "RP_STORY_ID"
NEW_COLUMNS = ("RELEVANCE", "RP_SOURCE_ID")
#: rebuilt from the raw rows and compared with the stored lists
CHECKED_LISTS = ("RP_ENTITY_ID", "ENTITY_TYPE", "ENTITY_NAME", "EVENT_SENTIMENT_SCORE")
RAW_FIELDS: dict[str, pa.DataType] = {
    ID_COL: pa.string(), "RP_ENTITY_ID": pa.string(), "ENTITY_TYPE": pa.string(),
    "ENTITY_NAME": pa.string(), "EVENT_SENTIMENT_SCORE": pa.float64(),
    "RELEVANCE": pa.float64(), "RP_SOURCE_ID": pa.string(),
}
_LISTS = (*CHECKED_LISTS, "RELEVANCE")
_RAW = "_raw_"                                  # prefix of the rebuilt columns in a join
_HASH_SEED = 20261001                           # one seed for both sides of the comparison
STAGING_DIR, BACKUP_DIR = "_staging", "_backup"  # under the datalake root, per artifact id
_VERSION = re.compile(r"^(v?\d+\.\d+\.)(\d+)$")


# ---------------------------------------------------------------------------
# Pure part: raw rows -> per-story lists -> the enriched partition
# ---------------------------------------------------------------------------

def _check_relevance(df: pl.DataFrame, where: str) -> None:
    rel = df["RELEVANCE"]
    bad = df.filter(rel.is_null() | (rel != rel.round(0)) | (rel < 0) | (rel > 100))
    if bad.height:
        raise ValueError(f"{where}: {bad.height} raw row(s) with RELEVANCE not an integer in "
                         f"[0, 100], e.g. story {bad[ID_COL][0]} ({bad['RELEVANCE'][0]!r})")


def _ess_cents(col: pl.Expr) -> pl.Expr:
    """EVENT_SENTIMENT_SCORE as integer hundredths (the raw CSV has two decimals)."""
    return col.list.eval((pl.element().cast(pl.Float64) * 100).round(0).cast(pl.Int32))


def list_hashes(prefix: str = "") -> list[pl.Expr]:
    """``_h_<col>``: the per-row hash of each checked list (ESS at its two decimals)."""
    out = []
    for c in CHECKED_LISTS:
        col = pl.col(prefix + c)
        if c == "EVENT_SENTIMENT_SCORE":
            col = _ess_cents(col)
        out.append(col.hash(seed=_HASH_SEED).alias(f"_h_{c}"))
    return out


def _collapse(df: pl.DataFrame) -> pl.DataFrame:
    """One row per story of ``df`` (first-appearance order): the hashes of its checked
    lists, its RELEVANCE list in row order and its RP_SOURCE_ID."""
    lists = df.group_by(ID_COL, maintain_order=True).agg(
        *(pl.col(c) for c in _LISTS),
        pl.col("RP_SOURCE_ID").first(),
        pl.col("RP_SOURCE_ID").n_unique().alias("_n_sources"),
    )
    return lists.with_columns(list_hashes()).drop(list(CHECKED_LISTS))


def raw_story_lists(batches: Iterable[pa.RecordBatch | pl.DataFrame], *,
                    where: str = "") -> pl.DataFrame:
    """Per story of the raw rows: RP_STORY_ID, the hashes of its checked lists
    (``list_hashes``), its RELEVANCE list in file order, RP_SOURCE_ID. A story's rows are
    contiguous in the vendor files (the ingest relies on it too); the last story of a batch
    is carried into the next, and a story seen in two places raises. RP_SOURCE_ID must be
    single-valued per story."""
    parts: list[pl.DataFrame] = []
    carry: pl.DataFrame | None = None
    for b in batches:
        df = b if isinstance(b, pl.DataFrame) else pl.from_arrow(b)
        if not df.height:
            continue
        _check_relevance(df, where)
        df = df.with_columns(pl.col("RELEVANCE").cast(pl.UInt8))
        if carry is not None:
            df = pl.concat([carry, df])
        ids = df[ID_COL]
        n_runs = int(ids.ne_missing(ids.shift(1)).sum())
        if n_runs != ids.n_unique():               # group_by would merge them silently
            raise ValueError(f"{where}: the raw rows of {n_runs - ids.n_unique()} "
                             "stor(y/ies) are not contiguous")
        tail = df[ID_COL] == df[ID_COL][-1]
        carry = df.filter(tail)
        body = df.filter(~tail)
        if body.height:
            parts.append(_collapse(body))
    if carry is not None and carry.height:
        parts.append(_collapse(carry))
    if not parts:
        return pl.DataFrame(schema={ID_COL: pl.String, "RELEVANCE": pl.List(pl.UInt8),
                                    "RP_SOURCE_ID": pl.String,
                                    **{f"_h_{c}": pl.UInt64 for c in CHECKED_LISTS}})
    out = pl.concat(parts)
    split = out.filter(pl.col(ID_COL).is_duplicated())
    if split.height:
        raise ValueError(f"{where}: the raw rows of {split[ID_COL].n_unique()} stor(y/ies) are "
                         f"not contiguous, e.g. {split[ID_COL][0]}")
    many = out.filter(pl.col("_n_sources") > 1)
    if many.height:
        raise ValueError(f"{where}: RP_SOURCE_ID varies within {many.height} stor(y/ies), "
                         f"e.g. {many[ID_COL][0]}")
    return out.drop("_n_sources")


def enrich_frame(orig: pl.DataFrame, lists: pl.DataFrame, *, exact: bool,
                 where: str = "") -> pl.DataFrame:
    """``orig`` (one stored partition, without the new columns) plus RELEVANCE and
    RP_SOURCE_ID, in ``orig``'s row order and STRUCTURED_SCHEMA column order.

    ``lists`` is ``raw_story_lists`` of the partition's raw months. Raises unless every
    story of ``orig`` is found there with lists equal to its stored ones (and, when
    ``exact``, unless ``lists`` holds no other story)."""
    have = set(orig.columns)
    want = set(STRUCTURED_SCHEMA.names) - set(NEW_COLUMNS)
    if have != want:
        raise ValueError(f"{where}: the stored partition has columns {sorted(have)}, expected "
                         f"{sorted(want)} (already enriched, or another layout)")
    if orig[ID_COL].is_duplicated().any():
        raise ValueError(f"{where}: duplicate {ID_COL} in the stored partition")
    if exact:
        extra = lists.join(orig.select(ID_COL), on=ID_COL, how="anti")
        if extra.height:
            raise ValueError(f"{where}: {extra.height} raw stor(y/ies) absent from the "
                             f"partition, e.g. {extra[ID_COL][0]}")
    raw = lists.rename({c: _RAW + c for c in lists.columns if c != ID_COL}).with_columns(
        pl.lit(True).alias("_found"))
    j = orig.join(raw, on=ID_COL, how="left", maintain_order="left")
    missing = j.filter(pl.col("_found").is_null())
    if missing.height:
        raise ValueError(f"{where}: {missing.height} stor(y/ies) of the partition have no raw "
                         f"rows, e.g. {missing[ID_COL][0]}")
    j = j.with_columns(list_hashes())
    for c in CHECKED_LISTS:
        diff = j.filter(pl.col(f"_h_{c}") != pl.col(f"{_RAW}_h_{c}"))
        if diff.height:
            raise ValueError(f"{where}: {c} rebuilt from the raw rows differs from the stored "
                             f"list in {diff.height} stor(y/ies), e.g. {diff[ID_COL][0]}")
    return j.with_columns(
        pl.col(_RAW + "RELEVANCE").cast(pl.List(pl.UInt8)).alias("RELEVANCE"),
        pl.col(_RAW + "RP_SOURCE_ID").alias("RP_SOURCE_ID"),
    ).select(STRUCTURED_SCHEMA.names)


def write_enriched(df: pl.DataFrame, path: Path) -> None:
    """Exactly STRUCTURED_SCHEMA (the Arrow schema a fresh ingest writes), zstd."""
    pq.write_table(df.to_arrow().cast(STRUCTURED_SCHEMA), path, compression="zstd")


def check_enriched(orig: pl.DataFrame, path: Path, *, where: str = "") -> None:
    """The file at ``path`` is ``orig`` plus the new columns: same rows in the same order,
    every existing column equal, RELEVANCE aligned in length with RP_ENTITY_ID."""
    back = pl.read_parquet(path)
    if back.columns != STRUCTURED_SCHEMA.names:
        raise ValueError(f"{where}: written columns {back.columns} != {STRUCTURED_SCHEMA.names}")
    if back.schema != STRUCTURED_SCHEMA_POLARS:
        raise ValueError(f"{where}: written schema {back.schema} != {STRUCTURED_SCHEMA_POLARS}")
    if back.height != orig.height:
        raise ValueError(f"{where}: {back.height} rows written, {orig.height} stored")
    for c in orig.columns:
        if not back[c].equals(orig[c]):
            raise ValueError(f"{where}: column {c} changed in the enriched file")
    lengths = back["RELEVANCE"].list.len() != back["RP_ENTITY_ID"].list.len()
    if lengths.any():
        raise ValueError(f"{where}: RELEVANCE and RP_ENTITY_ID differ in length in "
                         f"{int(lengths.sum())} row(s)")


# ---------------------------------------------------------------------------
# The Job: one unit per partition, run as an update of the headlines artifact
# ---------------------------------------------------------------------------

def next_patch(version: str) -> str:
    """``v0.1.0`` -> ``v0.1.1``: the same major.minor, so the extend run is accepted."""
    m = _VERSION.match(version)
    if not m:
        raise JobError(f"not a semantic version: {version!r}")
    return f"{m.group(1)}{int(m.group(2)) + 1}"


def is_enriched(path: Path) -> bool:
    return path.exists() and set(NEW_COLUMNS) <= set(pq.read_schema(path).names)


class EnrichJob(Job):
    """Rewrite every partition of a headlines artifact with RELEVANCE and RP_SOURCE_ID.

    Not registered: it shares its kind with IngestJob and only ever runs as an update
    of an existing artifact (``JobRunner.update_job`` / ``resume_job``)."""

    kind = KIND
    hash_pattern = "*.parquet"

    def __init__(self, headlines: Artifact, raw_dir: Path, *, backup_dir: Path,
                 staging_dir: Path, block_bytes: int = BLOCK_BYTES) -> None:
        if headlines.kind != KIND:
            raise ValueError(f"{headlines.artifact_id} is a {headlines.kind}, not a {KIND}")
        version = headlines.meta.pipeline_version
        self.temp = version.endswith(TEMP_SUFFIX)
        self.pipeline_version = next_patch(version.removesuffix(TEMP_SUFFIX))
        self.headlines, self.raw_dir = headlines, Path(raw_dir)
        self.backup_dir, self.staging_dir = Path(backup_dir), Path(staging_dir)
        self.block_bytes = block_bytes
        self.layout = layout_from_hyperparams(headlines.meta.hyperparams)
        if self.layout.start is None or self.layout.end is None:
            raise ValueError(f"{headlines.artifact_id} declares no range in its hyperparams")
        self._cache: dict[tuple[int, int], pl.DataFrame] = {}

    def params(self) -> dict:
        return self.headlines.meta.hyperparams

    def notes(self) -> str:
        return (f"enrich: + {', '.join(NEW_COLUMNS)} from raw_dir={self.raw_dir}; "
                f"originals moved to {self.backup_dir}")

    def units(self) -> list[Unit]:
        out = []
        for p in self.layout.expected():
            name = partition_file(p.key)
            if (self.headlines.path / name).exists() or (self.backup_dir / name).exists():
                out.append(Unit(p.key))
        return out

    def is_done(self, unit: Unit, out_dir: Path) -> bool:
        return is_enriched(out_dir / partition_file(unit.key))

    def _month_lists(self, year: int, month: int, where: str) -> pl.DataFrame:
        key = (year, month)
        if key not in self._cache:
            if len(self._cache) >= 2:
                self._cache.pop(next(iter(self._cache)))
            batches = raw_month_batches(self.raw_dir, year, month, self.block_bytes,
                                        fields=RAW_FIELDS)
            self._cache[key] = raw_story_lists(batches, where=f"{where} raw {year}-{month:02d}")
        return self._cache[key]

    def _lists(self, period: Period, where: str) -> pl.DataFrame:
        frames = [self._month_lists(m.first.year, m.first.month, where)
                  for m in periods(period.first, period.last, "M")]
        return frames[0] if len(frames) == 1 else pl.concat(frames)

    def run_unit(self, unit: Unit, ctx: JobContext) -> None:
        name = partition_file(unit.key)
        target, backup = ctx.out_dir / name, self.backup_dir / name
        if target.exists():
            if backup.exists():
                raise JobError(f"{name}: an unenriched file in the artifact AND a backup "
                               f"in {self.backup_dir}; inspect both before continuing")
            source = target
        elif backup.exists():                   # crashed between the two moves
            source = backup
        else:
            raise FileNotFoundError(f"{name}: neither in the artifact nor in {self.backup_dir}")
        period = parse_key(unit.key)
        orig = pl.read_parquet(source)
        enriched = enrich_frame(orig, self._lists(period, unit.key),
                                exact=period.freq in ("M", "Q", "Y"), where=unit.key)
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        staged = self.staging_dir / f"{name}.tmp"
        write_enriched(enriched, staged)
        check_enriched(orig, staged, where=unit.key)
        if source == target:
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            os.replace(target, backup)
        os.replace(staged, target)
        ctx.log.info("%s: enriched (%d stories); original in %s", name, orig.height, backup)

    def finalize(self, ctx: JobContext) -> None:
        names = [partition_file(u.key) for u in self.units()]
        left = [n for n in names if not is_enriched(ctx.out_dir / n)]
        if left:
            raise JobError(f"{len(left)} partition(s) not enriched, e.g. {left[0]}")
        ctx.note(f"enriched {len(names)} partition(s) with {', '.join(NEW_COLUMNS)}; "
                 f"originals kept in {self.backup_dir}")

    @classmethod
    def from_artifact(cls, artifact: Artifact, index: DatalakeIndex, **kwargs) -> EnrichJob:
        return cls(artifact, **kwargs)


def default_dirs(index: DatalakeIndex, artifact_id: str) -> tuple[Path, Path]:
    """(backup_dir, staging_dir): ``<root>/_backup/<id>`` and ``<root>/_staging/<id>``,
    on the lake's own file system so the moves are renames."""
    return index.root / BACKUP_DIR / artifact_id, index.root / STAGING_DIR / artifact_id


def enrich_job(index: DatalakeIndex, artifact_id: str, raw_dir: Path,
               **kwargs) -> EnrichJob:
    backup_dir, staging_dir = default_dirs(index, artifact_id)
    return EnrichJob(index.get(artifact_id), raw_dir, backup_dir=backup_dir,
                     staging_dir=staging_dir, **kwargs)


def main(argv: list[str] | None = None) -> int:
    from dotenv import find_dotenv, load_dotenv

    from datalake import DatalakeIndex
    from datalake.jobs import JobRunner
    from ravenpack.headlines.ingest import default_raw_dir

    load_dotenv(find_dotenv(usecwd=True))
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("start", "resume"))
    ap.add_argument("artifact_id")
    ap.add_argument("--raw-dir", default=None,
                    help="default: $RAW_DATA_PATH/RavenPack/headlines_edge_v1.0")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="run from a dirty git tree (recorded in the run notes)")
    args = ap.parse_args(argv)
    raw_dir = Path(args.raw_dir) if args.raw_dir else default_raw_dir()
    if not raw_dir.is_dir():
        raise SystemExit(f"raw directory does not exist: {raw_dir}")
    repo_dir = Path(__file__).resolve().parents[4]
    with DatalakeIndex(os.environ["DATALAKE_ROOT"], create=False) as dl:
        runner = JobRunner(dl, repo_dir=repo_dir, allow_dirty=args.allow_dirty)
        job = enrich_job(dl, args.artifact_id, raw_dir)
        art = (runner.update_job(args.artifact_id, job) if args.cmd == "start"
               else runner.resume_job(args.artifact_id, job))
    print(f"{art.artifact_id} partial={art.partial}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
