"""RavenPack Annotations ingestion: one raw pass, every field, two sibling tables.

Raw: one zip per year, one CSV per month (``feeds.FeedSpec``); a CSV row is one
detection (story x entity x event). Output, one parquet per month in each of two
sibling artifacts (filed under the feed's group, e.g. ``derived/RavenPack/``):

  {headlines_kind}  one row per story: RP_STORY_ID, the story-level fields,
                    N_ENTITIES (its story x entity rows), N_DETECTIONS (its raw rows);
                    plus ``{YYYY-MM}.report.json`` (counts, providers), hashed
  {entities_kind}   one row per story x entity: RP_STORY_ID, RP_ENTITY_ID,
                    TIMESTAMP_UTC, the entity-level fields, every detection field as
                    a list aligned across fields and ordered by RP_STORY_EVENT_INDEX

Stories keep raw order (= timestamp order); a story's entities keep the order of
their first raw row. Exploding the lists and joining the stories gives back every
raw row and every field (tested).

Refused, failing the month (nothing written): a raw column missing from or unknown
to the feed's field registry; a value that does not parse to its type; a field
breaking its level (a story field varying inside a story, an entity field inside a
story x entity); a story whose rows are not contiguous; duplicate (story, entity,
event index). Memory is bounded by the CSV block size (one block plus the carried
last story).

Commit per month: the entities file is renamed first, then the report, then the
headlines file, which is the job's done-marker; a crash in between re-runs the month
and overwrites the entities file.
"""
from __future__ import annotations

import json
import logging
import time
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import polars as pl
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from datalake.jobs import TEMP_SUFFIX, Job, JobContext, Unit
from datalake.layout import Layout, add_layout_args, layout_from_args, layout_from_hyperparams
from datalake.lineage import kind_of
from datalake.periods import parse_key, periods
from ravenpack.annotations.feeds import RPA1_ALL, FeedSpec, feed
from ravenpack.annotations.fields import (
    DATETIME,
    ENTITY_KEY,
    EVENT_INDEX,
    RAW_DATETIME_FORMAT,
    STORY_KEY,
    TIMESTAMP,
    FieldSet,
)

if TYPE_CHECKING:
    from datalake import Artifact, DatalakeIndex

log = logging.getLogger(__name__)

PIPELINE_VERSION = "v1.0.0"
BLOCK_BYTES = 256 << 20          # CSV bytes parsed per block (~1M raw rows)
REPORT_SUFFIX = ".report.json"
_HASH_SEED = 0x5EED
# Raw strings read as null: pandas' read_csv defaults, kept so the tables match the
# ravenpack_headlines lineage (owner decision, 2026-10-02). Note that "NA" is also
# Namibia's country code: COUNTRY_CODE = NA is null here, as it always was.
NULL_STRINGS = ("", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan",
                "1.#IND", "1.#QNAN", "<NA>", "N/A", "NA", "NULL", "NaN", "None", "n/a", "nan",
                "null")


class IngestError(ValueError):
    """The raw month cannot be ingested as declared (schema, parse, level, order)."""


class LevelError(IngestError):
    """A field is not constant at its declared level."""


# ---------------------------------------------------------------------------
# Reading: raw CSV blocks -> typed polars frames, whole stories only
# ---------------------------------------------------------------------------

def _arrow_reader(stream: Any, columns: list[str], block_bytes: int) -> pacsv.CSVStreamingReader:
    return pacsv.open_csv(
        stream,
        read_options=pacsv.ReadOptions(block_size=block_bytes),
        parse_options=pacsv.ParseOptions(newlines_in_values=True),
        convert_options=pacsv.ConvertOptions(
            column_types={c: pa.string() for c in columns},
            strings_can_be_null=True, null_values=list(NULL_STRINGS),
            quoted_strings_can_be_null=True),
    )


def _check_header(names: list[str], fields: FieldSet, where: str) -> None:
    unknown = sorted(set(names) - set(fields.raw_columns))
    missing = sorted(set(fields.raw_columns) - set(names))
    if unknown or missing or len(names) != len(set(names)):
        raise IngestError(f"{where}: raw header does not match the field registry "
                          f"(unknown {unknown}, missing {missing})")


def _parse_expr(name: str, dtype: pl.DataType) -> pl.Expr:
    col = pl.col(name)
    if dtype == pl.String:
        return col
    if dtype == DATETIME:
        return col.str.to_datetime(RAW_DATETIME_FORMAT, time_unit="us", time_zone="UTC",
                                   strict=False)
    num = col.cast(pl.Float64, strict=False)
    if dtype.is_integer():
        ranges = ((pl.Int8, -128, 127), (pl.UInt8, 0, 255), (pl.Int32, -(2**31), 2**31 - 1))
        lo, hi = next((a, b) for t, a, b in ranges if dtype == t)
        ok = (num == num.round(0)) & num.is_between(lo, hi)
        return pl.when(ok).then(num).otherwise(None).cast(dtype)
    return num.cast(dtype)


def typed(frame: pl.DataFrame, fields: FieldSet, where: str) -> pl.DataFrame:
    """Cast a raw all-string frame to the registry types; any value that does not parse
    (or an integer field holding a fraction or out of range) raises."""
    by = fields.by_name
    out = frame.select([_parse_expr(c, by[c].dtype).alias(c) for c in fields.raw_columns])
    lost = {c: frame[c].is_not_null() & out[c].is_null() for c in fields.raw_columns}
    bad = {c: int(m.sum()) for c, m in lost.items() if m.any()}
    if bad:
        name = next(iter(bad))
        example = frame.filter(lost[name])[name][0]
        raise IngestError(f"{where}: values that do not parse to their type: {bad} "
                          f"(e.g. {name} = {example!r})")
    return out


def _story_runs(ids: pl.Series) -> int:
    return int((ids != ids.shift(1)).fill_null(True).sum())


def iter_month(stream: Any, fields: FieldSet, where: str,
               block_bytes: int = BLOCK_BYTES) -> Iterator[pl.DataFrame]:
    """Typed frames of one raw month, each holding whole stories (the last story of a
    block is carried into the next). Raises when a story's rows are not contiguous."""
    reader = _arrow_reader(stream, fields.raw_columns, block_bytes)
    _check_header(reader.schema.names, fields, where)
    seen = pl.Series("h", [], dtype=pl.UInt64)
    carry: pl.DataFrame | None = None

    def emit(block: pl.DataFrame) -> pl.DataFrame:
        nonlocal seen
        ids = block[STORY_KEY]
        if ids.null_count():
            raise IngestError(f"{where}: {ids.null_count()} row(s) without {STORY_KEY}")
        uniq = ids.unique()
        if _story_runs(ids) != len(uniq):
            raise IngestError(f"{where}: a story's rows are not contiguous in the raw file")
        hashes = uniq.hash(_HASH_SEED)
        again = hashes.is_in(seen.implode())
        if again.any():
            raise IngestError(f"{where}: story {uniq.filter(again)[0]} reappears after "
                              "other stories (rows not contiguous)")
        seen = pl.concat([seen, hashes.rename("h")], rechunk=False)
        return typed(block, fields, where)

    for batch in reader:
        block = pl.from_arrow(pa.Table.from_batches([batch]))
        assert isinstance(block, pl.DataFrame)
        if block.is_empty():
            continue
        block = block.select(fields.raw_columns)
        if carry is not None:
            block = pl.concat([carry, block])
        ids = block[STORY_KEY]
        other = ids != ids[-1]
        trailing = int(other.reverse().arg_max()) if other.any() else block.height
        cut = block.height - trailing           # only the LAST run is carried: an earlier
        carry, body = block.slice(cut), block.slice(0, cut)    # run of that id stays in body
        if not body.is_empty():
            yield emit(body)
    if carry is not None and not carry.is_empty():
        yield emit(carry)


# ---------------------------------------------------------------------------
# Building the two tables from a block of whole stories
# ---------------------------------------------------------------------------

def _level_violations(frame: pl.DataFrame, keys: list[str], cols: list[str]) -> pl.DataFrame:
    return (frame.group_by(keys).agg([pl.col(c).n_unique().alias(c) for c in cols])
            .select(keys + [(pl.col(c) > 1).alias(c) for c in cols]))


def build(rows: pl.DataFrame, fields: FieldSet, where: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(stories, story x entity) frames of typed raw rows holding whole stories."""
    story_f, entity_f, det_f = fields.story_fields, fields.entity_fields, fields.detection_fields
    for keys, cols, label in (([STORY_KEY], story_f, "story"),
                              ([STORY_KEY, ENTITY_KEY], entity_f, "story x entity")):
        viol = _level_violations(rows, keys, cols)
        bad = {c: int(viol[c].sum()) for c in cols if viol[c].any()}
        if bad:
            c = next(iter(bad))
            ex = viol.filter(pl.col(c)).row(0, named=True)
            raise LevelError(f"{where}: {label}-level field(s) vary within a {label}: {bad} "
                             f"(e.g. {c} at {', '.join(str(ex[k]) for k in keys)})")
    if rows[ENTITY_KEY].null_count():
        raise IngestError(f"{where}: {rows[ENTITY_KEY].null_count()} row(s) without "
                          f"{ENTITY_KEY}")
    dup = rows.select(pl.struct(STORY_KEY, ENTITY_KEY, EVENT_INDEX).is_duplicated().sum()).item()
    if dup:
        raise IngestError(f"{where}: {dup} duplicated (story, entity, event index) row(s)")

    pairs = (rows.group_by([STORY_KEY, ENTITY_KEY], maintain_order=True)
             .agg(pl.col(TIMESTAMP).first(), *[pl.col(c).first() for c in entity_f],
                  *[pl.col(c).sort_by(EVENT_INDEX) for c in det_f]))
    stories = (rows.group_by(STORY_KEY, maintain_order=True)
               .agg(*[pl.col(c).first() for c in story_f],
                    pl.col(ENTITY_KEY).n_unique().cast(pl.Int32).alias("N_ENTITIES"),
                    pl.len().cast(pl.Int32).alias("N_DETECTIONS")))
    hs, es = fields.headlines_schema, fields.entities_schema
    return stories.select(hs.names()).cast(hs), pairs.select(es.names()).cast(es)


# ---------------------------------------------------------------------------
# Writing one month, atomically
# ---------------------------------------------------------------------------

def _arrow_schema(schema: pl.Schema) -> pa.Schema:
    return pl.DataFrame(schema=schema).to_arrow(compat_level=pl.CompatLevel.oldest()).schema


@dataclass
class MonthReport:
    month: str
    feed: str
    member: str
    raw_rows: int = 0
    stories: int = 0
    pairs: int = 0
    event_rows: int = 0                         # raw rows carrying an event (CATEGORY)
    stories_outside_month: int = 0              # timestamp outside the raw month
    rows_by_provider: dict[str, int] = field(default_factory=dict)
    stories_by_provider: dict[str, int] = field(default_factory=dict)

    def add(self, rows: pl.DataFrame, stories: pl.DataFrame, pairs: pl.DataFrame) -> None:
        self.raw_rows += rows.height
        self.stories += stories.height
        self.pairs += pairs.height
        if "CATEGORY" in rows.columns:
            self.event_rows += int(rows["CATEGORY"].is_not_null().sum())
        months = stories[TIMESTAMP].dt.strftime("%Y-%m")
        self.stories_outside_month += int((months != self.month).sum())
        for target, frame in ((self.rows_by_provider, rows), (self.stories_by_provider, stories)):
            if "PROVIDER_ID" in frame.columns:
                for prov, n in frame["PROVIDER_ID"].fill_null("(null)").value_counts().iter_rows():
                    target[prov] = target.get(prov, 0) + int(n)


class MonthWriter:
    """Both tables of one month into ``{key}.parquet.tmp``; ``commit`` renames them."""

    def __init__(self, fields: FieldSet, headlines_dir: Path, entities_dir: Path,
                 report: MonthReport) -> None:
        key = report.month
        self.fields, self.report = fields, report
        self.final = (Path(entities_dir) / f"{key}.parquet",
                      Path(headlines_dir) / f"{key}{REPORT_SUFFIX}",
                      Path(headlines_dir) / f"{key}.parquet")
        self.tmp = tuple(p.with_name(p.name + ".tmp") for p in self.final)
        self.schemas = (_arrow_schema(fields.entities_schema),
                        _arrow_schema(fields.headlines_schema))
        self.writers = (pq.ParquetWriter(self.tmp[0], self.schemas[0], compression="zstd"),
                        pq.ParquetWriter(self.tmp[2], self.schemas[1], compression="zstd"))
        self.n_written = [0, 0]

    def write(self, rows: pl.DataFrame, stories: pl.DataFrame, pairs: pl.DataFrame) -> None:
        for i, (frame, schema) in enumerate(((pairs, self.schemas[0]),
                                             (stories, self.schemas[1]))):
            table = frame.to_arrow(compat_level=pl.CompatLevel.oldest())
            self.writers[i].write_table(table.cast(schema))
            self.n_written[i] += frame.height
        self.report.add(rows, stories, pairs)

    def commit(self, n_detections: int, n_pairs_from_stories: int) -> None:
        for w in self.writers:
            w.close()
        r = self.report
        problems = []
        if self.n_written != [r.pairs, r.stories]:
            problems.append(f"rows written {self.n_written} != counted {[r.pairs, r.stories]}")
        if n_detections != r.raw_rows:
            problems.append(f"sum N_DETECTIONS {n_detections} != raw rows {r.raw_rows}")
        if n_pairs_from_stories != r.pairs:
            problems.append(f"sum N_ENTITIES {n_pairs_from_stories} != pairs {r.pairs}")
        if problems:
            self.abort()
            raise IngestError(f"{r.month}: tables do not reconcile: {'; '.join(problems)}")
        self.tmp[1].write_text(json.dumps(asdict(r), indent=2, sort_keys=True) + "\n")
        for tmp, final in zip(self.tmp, self.final):      # entities, report, headlines last
            tmp.replace(final)

    def abort(self) -> None:
        for w in self.writers:
            try:
                w.close()
            except Exception:  # noqa: BLE001 - best effort on the failure path
                pass
        for tmp in self.tmp:
            tmp.unlink(missing_ok=True)


def ingest_month(spec: FeedSpec, zf: zipfile.ZipFile, member: str, key: str,
                 headlines_dir: Path, entities_dir: Path,
                 block_bytes: int = BLOCK_BYTES) -> MonthReport:
    """Read one raw month and write both of its partitions (atomic, see module doc)."""
    report = MonthReport(month=key, feed=spec.name, member=member)
    writer = MonthWriter(spec.fields, headlines_dir, entities_dir, report)
    n_det = n_pairs = 0
    t0 = time.monotonic()
    try:
        with zf.open(member) as stream:
            for rows in iter_month(stream, spec.fields, f"{spec.name} {key}", block_bytes):
                stories, pairs = build(rows, spec.fields, f"{spec.name} {key}")
                writer.write(rows, stories, pairs)
                n_det += int(stories["N_DETECTIONS"].sum())
                n_pairs += int(stories["N_ENTITIES"].sum())
                log.info("%s: %d raw rows, %d stories so far", key, report.raw_rows,
                         report.stories)
        writer.commit(n_det, n_pairs)
    except BaseException:
        writer.abort()
        raise
    secs = time.monotonic() - t0
    log.info("%s done: %d raw rows -> %d stories, %d story x entity rows in %.0fs "
             "(%.0f rows/s)", key, report.raw_rows, report.stories, report.pairs, secs,
             report.raw_rows / max(secs, 1e-9))
    return report


# ---------------------------------------------------------------------------
# The Job: the headlines artifact, the entities sibling opened in session()
# ---------------------------------------------------------------------------

def last_raw_month(spec: FeedSpec, raw_dir: Path) -> date | None:
    """First day of the latest month available in the raw zips (None: none)."""
    years = sorted(int(p.stem.rsplit("_", 1)[-1]) for p in Path(raw_dir).glob(
        spec.zip_name.format(year="*")) if p.stem.rsplit("_", 1)[-1].isdigit())
    for year in reversed(years):
        zp = spec.zip_path(raw_dir, year)
        with zipfile.ZipFile(zp) as zf:
            names = set(zf.namelist())
        months = [m for m in range(1, 13) if spec.member(zp, year, m) in names]
        if months:
            return date(year, max(months), 1)
    return None


def monthly(layout: Layout) -> Layout:
    if layout.freq != "M":
        raise ValueError(f"the raw is monthly: partition_freq must be M, got {layout.freq}")
    first, last = layout.expected()[0], layout.expected()[-1]
    return Layout("M", first.first, last.last)


class AnnotationsIngestJob(Job):
    """A RavenPack Annotations feed -> headlines + entities siblings (one month per unit).

    A feed is one subclass setting ``feed`` (its kinds and group follow); see
    ``RPA1IngestJob``."""

    feed: ClassVar[FeedSpec]
    pipeline_version = PIPELINE_VERSION
    hash_pattern = "*"

    def __init__(self, raw_dir: Path, layout: Layout, *, temp: bool = False,
                 block_bytes: int = BLOCK_BYTES, recorded: dict[str, Any] | None = None,
                 after: date | None = None) -> None:
        self.raw_dir, self.layout, self.temp = Path(raw_dir), monthly(layout), temp
        self.block_bytes = block_bytes
        self.recorded = recorded            # the artifact's hyperparams (resume / update)
        self.after = after                  # an update plans only the months after this day
        self._zip: tuple[int, zipfile.ZipFile] | None = None
        self.entities_dir: Path | None = None

    # -- identity -------------------------------------------------------------

    def params(self) -> dict[str, Any]:
        return {"feed": self.feed.name, **self.layout.hyperparams()}

    def notes(self) -> str:
        return f"raw_dir={self.raw_dir}"

    # -- units ----------------------------------------------------------------

    def units(self) -> list[Unit]:
        return [Unit(p.key) for p in self.layout.expected()
                if self.after is None or p.first > self.after]

    def is_done(self, unit: Unit, out_dir: Path) -> bool:
        return (out_dir / f"{unit.key}.parquet").exists()      # written last: done-marker

    @contextmanager
    def session(self, ctx: JobContext) -> Iterator[None]:
        """Open (or resume / extend) the entities sibling of this headlines artifact."""
        siblings = [c for c in ctx.index.children(ctx.artifact_id)
                    if kind_of(c) == self.feed.entities_kind]
        if len(siblings) > 1:
            raise ValueError(f"{ctx.artifact_id} has several {self.feed.entities_kind} "
                             f"siblings: {siblings}")
        existing = ctx.index.get(siblings[0]) if siblings else None
        hp = {**(self.recorded or self.params()), "headlines_id": ctx.artifact_id}
        with ctx.index.run(
                kind=self.feed.entities_kind, pipeline=self.pipeline,
                pipeline_version=self.version, pipeline_repo=self.pipeline_repo,
                hyperparams=hp, sources=[ctx.artifact_id], verifier=self.feed.entities_kind,
                hash_pattern="*.parquet", notes=self.notes(),
                resume=existing.artifact_id if existing and existing.partial else None,
                extend=existing.artifact_id if existing and not existing.partial else None,
        ) as run:
            self.entities_dir = run.out_dir
            try:
                yield
            finally:
                self._close_zip()

    def _zipfile(self, year: int) -> zipfile.ZipFile | None:
        if self._zip is not None and self._zip[0] == year:
            return self._zip[1]
        self._close_zip()
        path = self.feed.zip_path(self.raw_dir, year)
        if not path.exists():
            return None
        self._zip = (year, zipfile.ZipFile(path))
        return self._zip[1]

    def _close_zip(self) -> None:
        if self._zip is not None:
            self._zip[1].close()
            self._zip = None

    def run_unit(self, unit: Unit, ctx: JobContext) -> None:
        if self.entities_dir is None:
            raise RuntimeError("run_unit outside the job's session")
        month = parse_key(unit.key).first
        zf = self._zipfile(month.year)
        member = (self.feed.member(self.feed.zip_path(self.raw_dir, month.year), month.year,
                                   month.month) if zf is not None else None)
        if zf is None or member not in zf.namelist():
            ctx.log.warning("%s: no raw file (%s), month skipped", unit.key,
                            member or self.feed.zip_path(self.raw_dir, month.year))
            return
        ingest_month(self.feed, zf, member, unit.key, ctx.out_dir, self.entities_dir,
                     self.block_bytes)

    def finalize(self, ctx: JobContext) -> None:
        present = Layout("M").existing(ctx.out_dir)
        if not present:
            raise RuntimeError(f"no month ingested for {self.layout.start}..{self.layout.end}; "
                               f"check {self.raw_dir}")
        missing = [u.key for u in self.units() if u.key not in present]
        if missing:
            ctx.log.warning("%d month(s) without raw data: %s", len(missing), missing[:12])
        ctx.note(f"{len(present)} month(s) present")

    # -- rebuild ----------------------------------------------------------------

    @classmethod
    def _rebuild(cls, artifact: Artifact, *, raw_dir: Path | str | None, end: date | None,
                 after: date | None = None, **kwargs: Any) -> AnnotationsIngestJob:
        hp = artifact.meta.hyperparams
        if hp.get("feed") != cls.feed.name:
            raise ValueError(f"{artifact.artifact_id} is feed {hp.get('feed')!r}, "
                             f"this job ingests {cls.feed.name!r}")
        layout = layout_from_hyperparams(hp)
        if layout.start is None or layout.end is None:
            raise ValueError(f"{artifact.artifact_id} declares no range in its hyperparams")
        raw = Path(raw_dir) if raw_dir else cls.feed.raw_dir()
        if end is not None:
            layout = Layout("M", layout.start, max(layout.end, end))
        return cls(raw, layout, temp=artifact.meta.pipeline_version.endswith(TEMP_SUFFIX),
                   recorded=hp, after=after, **kwargs)

    @classmethod
    def from_artifact(cls, artifact: Artifact, index: DatalakeIndex, *,
                      raw_dir: Path | str | None = None, **kwargs: Any) -> AnnotationsIngestJob:
        """Resume. An interrupted update (several executions) runs to the latest raw
        month, like the update did."""
        raw = Path(raw_dir) if raw_dir else cls.feed.raw_dir()
        end = last_raw_month(cls.feed, raw) if len(artifact.meta.runs) > 1 else None
        return cls._rebuild(artifact, raw_dir=raw, end=end, **kwargs)

    @classmethod
    def for_update(cls, artifact: Artifact, index: DatalakeIndex, *,
                   raw_dir: Path | str | None = None, **kwargs: Any) -> AnnotationsIngestJob:
        """Add the raw months after the artifact's last one (existing months untouched)."""
        raw = Path(raw_dir) if raw_dir else cls.feed.raw_dir()
        present = Layout("M").existing(artifact.path)
        after = parse_key(list(present)[-1]).last if present else None
        return cls._rebuild(artifact, raw_dir=raw, end=last_raw_month(cls.feed, raw),
                            after=after, **kwargs)

    @classmethod
    def add_cli_args(cls, parser: Any) -> None:
        add_layout_args(parser, default_freq="M")
        parser.add_argument("--raw-dir", default=None,
                            help=f"default: $RAW_DATA_PATH/{'/'.join(cls.feed.raw_subdir)}")
        parser.add_argument("--block-mb", type=int, default=BLOCK_BYTES >> 20,
                            help="raw CSV megabytes parsed per block (bounds memory)")
        parser.add_argument("--temp", action="store_true", help="agent-created (__TEMP)")

    @classmethod
    def from_args(cls, args: Any, index: DatalakeIndex) -> AnnotationsIngestJob:
        raw = Path(args.raw_dir) if args.raw_dir else cls.feed.raw_dir()
        if not raw.is_dir():
            raise ValueError(f"raw directory does not exist: {raw}")
        return cls(raw, layout_from_args(args), temp=args.temp, block_bytes=args.block_mb << 20)


class RPA1IngestJob(AnnotationsIngestJob):
    """RavenPack Analytics Annotations 1.0, all entities (web, Dow Jones, ... providers)."""

    feed = RPA1_ALL
    kind = RPA1_ALL.headlines_kind
    group = RPA1_ALL.group


# ---------------------------------------------------------------------------
# Verifiers (registered via pyproject.toml entry points)
# ---------------------------------------------------------------------------

def _sample(names: list[str]) -> list[str]:
    return sorted({names[0], names[len(names) // 2], names[-1]}) if names else []


def _months(artifact: Artifact) -> dict[str, Path]:
    return Layout("M").existing(artifact.path)


def _coverage(artifact: Artifact) -> list[tuple[str, str]]:
    """(severity, message) about the months present vs the declared start."""
    out: list[tuple[str, str]] = []
    layout = layout_from_hyperparams(artifact.meta.hyperparams)
    present = _months(artifact)
    if not present:
        return [("error", "no month file")]
    keys = list(present)
    if layout.start is not None and parse_key(keys[0]).first < layout.start:
        out.append(("error", f"months before the declared start {layout.start}: {keys[0]}"))
    expected = [p.key for p in periods(parse_key(keys[0]).first, parse_key(keys[-1]).first, "M")]
    gaps = [k for k in expected if k not in present]
    if gaps:
        out.append(("warning", f"{len(gaps)} month(s) missing inside the range: {gaps[:6]}"))
    return out


def _finding(artifact: Artifact, severity: str, message: str) -> Any:
    from datalake.verify import Finding, Severity

    return Finding(Severity.ERROR if severity == "error" else Severity.WARNING,
                   artifact.artifact_id, message)


def verify_headlines(artifact: Artifact) -> list:
    """Months covered, schema, report present and reconciled (sampled months)."""
    spec = feed(artifact.meta.hyperparams["feed"])
    findings = [_finding(artifact, s, m) for s, m in _coverage(artifact)]
    present = _months(artifact)
    for key in present:
        if not (artifact.path / f"{key}{REPORT_SUFFIX}").is_file():
            findings.append(_finding(artifact, "error", f"{key}: report missing"))
    for key in _sample(list(present)):
        try:
            frame = pl.read_parquet(present[key])
            report = json.loads((artifact.path / f"{key}{REPORT_SUFFIX}").read_text())
        except Exception as exc:  # noqa: BLE001
            findings.append(_finding(artifact, "error", f"{key}: unreadable: {exc}"))
            continue
        if frame.schema != spec.fields.headlines_schema:
            findings.append(_finding(artifact, "error", f"{key}: schema differs from the "
                                     "field registry"))
            continue
        counts = (frame.height, int(frame["N_DETECTIONS"].sum()), int(frame["N_ENTITIES"].sum()))
        if counts != (report["stories"], report["raw_rows"], report["pairs"]):
            findings.append(_finding(artifact, "error", f"{key}: (stories, raw rows, pairs) "
                                     f"{counts} do not match the report"))
        if frame[STORY_KEY].n_unique() != frame.height:
            findings.append(_finding(artifact, "error", f"{key}: duplicated story ids"))
    return findings


def verify_entities(artifact: Artifact) -> list:
    """Schema, aligned detection lists, every story present in the headlines sibling."""
    hp = artifact.meta.hyperparams
    spec = feed(hp["feed"])
    findings = [_finding(artifact, s, m) for s, m in _coverage(artifact)]
    sibling = artifact.path.parent.parent / spec.headlines_kind / hp.get("headlines_id", "")
    present = _months(artifact)
    for key in _sample(list(present)):
        try:
            frame = pl.read_parquet(present[key])
        except Exception as exc:  # noqa: BLE001
            findings.append(_finding(artifact, "error", f"{key}: unreadable: {exc}"))
            continue
        if frame.schema != spec.fields.entities_schema:
            findings.append(_finding(artifact, "error", f"{key}: schema differs from the "
                                     "field registry"))
            continue
        lengths = frame.select([pl.col(c).list.len() for c in spec.fields.detection_fields])
        first = lengths.columns[0]
        differs = pl.any_horizontal([pl.col(c) != pl.col(first) for c in lengths.columns])
        misaligned = lengths.select((differs | (pl.col(first) < 1)).sum()).item()
        if misaligned:
            findings.append(_finding(artifact, "error", f"{key}: {misaligned} row(s) with "
                                     "misaligned or empty detection lists"))
        stories = sibling / f"{key}.parquet"
        if not stories.is_file():
            findings.append(_finding(artifact, "warning", f"{key}: headlines sibling not "
                                     f"found at {stories}; story coverage not checked"))
            continue
        ids = pl.read_parquet(stories, columns=[STORY_KEY])[STORY_KEY]
        if set(frame[STORY_KEY].unique().to_list()) != set(ids.to_list()):
            findings.append(_finding(artifact, "error", f"{key}: story ids differ from the "
                                     "headlines sibling"))
    return findings


__all__ = ["AnnotationsIngestJob", "RPA1IngestJob", "IngestError", "LevelError", "MonthReport",
           "build", "ingest_month", "iter_month", "last_raw_month", "typed",
           "verify_entities", "verify_headlines", "PIPELINE_VERSION"]
