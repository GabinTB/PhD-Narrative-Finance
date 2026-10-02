"""Capital IQ Key Developments (WRDS) -> ``ciq_keydev`` raw artifact, one parquet per month,
one row per event VERSION.

CIQ keeps its history as validity periods (``speffectivedate`` -> ``sptodate``, open
when current): an edit, a re-dating, or the reuse of a ``keydevid`` for another event
adds a version and closes the previous one. That history starts in 2018-04 (WRDS's first
``speffectivedate``); the oldest version of every event is therefore taken as valid since
the beginning (``first_version``).

Source (WRDS ``ciq_keydev``, plans checked with ``EXPLAIN``, every query indexed):

    wrds_keydev   event x company x role link versions (45.6 M rows). Indexed on
                  ``announcedate``, ``keydevid`` and ``speffectivedate``.
    ciqkeydev     event versions with the text (35 M rows), indexed on ``keydevid`` only.

Base load (``jobs start ciq_keydev``): unit = month. The month's event ids come from
``wrds_keydev.announcedate``; every version of those events and of their company links is
fetched by ``keydevid``, and the versions announced in the month are written to
``YYYY-MM.parquet`` (``KEYDEV_SCHEMA``): text, every CIQ date, validity, and
``companies``, the link versions overlapping the event version's validity (list of
structs with their own validity). A month without events is an empty file.

Update (``jobs update <id>``): one unit. The events with a link version newer than the
latest ``speffectivedate`` already stored (``wrds_keydev_speffectivedate_idx``) are
re-fetched with all their versions; ``update-<vintage>.parquet`` holds the versions that
are new or changed (``row_sha256``; a closed ``sptodate`` is a change), empty when nothing
changed. Base files are never rewritten.

Reading (``read_keydev``): current versions by default; ``as_of=t`` gives what CIQ showed
at ``t`` (event and link versions valid at ``t``, entered by ``t``); ``versions=True``
every version. Lookups (event type -> category, role, time zone) are written once by the
base load as ``dim-<table>.parquet``.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from datalake.jobs import TEMP_SUFFIX, Job, JobContext, JobError, Unit
from datalake.layout import Layout, add_layout_args, layout_from_args, layout_from_hyperparams
from datalake.periods import PeriodError, parse_key, partition_file, period_of

if TYPE_CHECKING:
    from datalake import Artifact, DatalakeIndex

log = logging.getLogger(__name__)

KIND = "ciq_keydev"
PIPELINE_VERSION = "v0.2.0"     # v0.2.0: one row per event version (v0.1.0 mixed versions)
SOURCE = "wrds_ciq_keydev"
GROUP = "CapIQ"
ID_BATCH = 20_000

PLAN_PREFIX, UPDATE_PREFIX, DIM_PREFIX = "plan-", "update-", "dim-"
SOURCES_METADATA_KEY = b"ciq_keydev.source"
KEY = ("keydevid", "speffectivedate")

Clock = Callable[[], datetime]
Query = Callable[[str, dict[str, Any]], pd.DataFrame]

COMPANY_TYPE = pa.struct([
    ("companyid", pa.int64()), ("gvkey", pa.string()), ("companyname", pa.string()),
    ("keydeveventtypeid", pa.int32()), ("eventtype", pa.string()),
    ("keydevtoobjectroletypeid", pa.int32()), ("objectroletype", pa.string()),
    ("speffectivedate", pa.timestamp("us")), ("sptodate", pa.timestamp("us")),
    ("first_version", pa.bool_())])

KEYDEV_SCHEMA = pa.schema([
    ("keydevid", pa.int64()),
    ("speffectivedate", pa.timestamp("us")),       # version valid from (CIQ database time)
    ("sptodate", pa.timestamp("us")),              # ... until (null: current version)
    ("first_version", pa.bool_()),                 # oldest version: valid since the start
    ("headline", pa.string()),
    ("situation", pa.string()),
    ("announcedate", pa.date32()),                 # local announcement date (partition key)
    ("announceddate", pa.timestamp("us")),
    ("announceddateutc", pa.timestamp("us")),
    ("announceddatetimezoneid", pa.int32()),
    ("entereddate", pa.timestamp("us")),
    ("entereddateutc", pa.timestamp("us")),
    ("lastmodifieddate", pa.timestamp("us")),
    ("lastmodifieddateutc", pa.timestamp("us")),
    ("mostimportantdateutc", pa.timestamp("us")),
    ("sourcetypename", pa.string()),
    ("companies", pa.list_(COMPANY_TYPE)),
    ("row_sha256", pa.string()),                   # content signature (updates)
    ("vintage", pa.timestamp("us", tz="UTC")),     # execution that wrote the row
])

_EVENT_TIMES = ("announceddate", "announceddateutc", "entereddate", "entereddateutc",
                "lastmodifieddate", "lastmodifieddateutc", "mostimportantdateutc")
_COMPANY_FIELDS = [f.name for f in COMPANY_TYPE]
_LINK_FIELDS = _COMPANY_FIELDS[:7]                 # the link itself, without its validity
_LINK_COLUMNS = """keydevid, companyid, companyname, gvkey, keydeveventtypeid, eventtype,
       keydevtoobjectroletypeid, objectroletype, sourcetypename, announcedate,
       speffectivedate, sptodate,
       min(speffectivedate) over (partition by keydevid, companyid, keydevtoobjectroletypeid,
                                  keydeveventtypeid) as first_speffectivedate"""
_EVENT_COLUMNS = """keydevid, headline, situation, announceddate, announceddateutc,
       announceddatetimezoneid, entereddate, entereddateutc, lastmodifieddate,
       lastmodifieddateutc, mostimportantdateutc, speffectivedate, sptodate,
       min(speffectivedate) over (partition by keydevid) as first_speffectivedate"""
_MONTH_IDS = """select distinct keydevid from ciq_keydev.wrds_keydev
                where announcedate >= %(start)s and announcedate < %(stop)s"""

# Month: every version of the month's events / links, then the versions dated in the month
# (the window sees all versions, so the oldest one is found whatever its month).
LINKS_SQL = f"""
select * from (select {_LINK_COLUMNS} from ciq_keydev.wrds_keydev
               where keydevid in ({_MONTH_IDS})) x
where announcedate >= %(start)s and announcedate < %(stop)s
"""
EVENTS_SQL = f"""
select * from (select {_EVENT_COLUMNS} from ciq_keydev.ciqkeydev
               where keydevid in ({_MONTH_IDS})) x
where announceddate >= %(start)s and announceddate < %(stop)s
"""
# Update: events with a link version newer than ``since``, then all their versions.
CHANGED_IDS_SQL = """
select distinct keydevid from ciq_keydev.wrds_keydev where speffectivedate > %(since)s
"""
LINKS_BY_IDS_SQL = f"""
select {_LINK_COLUMNS} from ciq_keydev.wrds_keydev where keydevid = any(%(ids)s::numeric[])
"""
EVENTS_BY_IDS_SQL = f"""
select {_EVENT_COLUMNS} from ciq_keydev.ciqkeydev where keydevid = any(%(ids)s::numeric[])
"""

DIM_TABLES = ("ciqkeydevcategorytype", "ciqkeydevobjectroletype", "ciqkeydevtimezone")


def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def vintage_tag(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_vintage(tag: str) -> datetime:
    return datetime.strptime(tag, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)


def _naive_utc(ts: datetime) -> datetime:
    return ts.astimezone(timezone.utc).replace(tzinfo=None) if ts.tzinfo else ts


# ---------------------------------------------------------------------------
# Source
# ---------------------------------------------------------------------------

class KeyDevSource:
    """The job's queries on a WRDS connection (``query`` injectable: ``(sql, params) ->
    DataFrame``; default ``WRDSClient.from_env().raw.raw_sql``)."""

    def __init__(self, query: Query | None = None) -> None:
        self._query, self._conn = query, None

    def _run(self, sql: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
        if self._query is None:
            from wrds_client.client import WRDSClient

            self._conn = WRDSClient.from_env().raw
            self._query = lambda s, p: self._conn.raw_sql(s, params=p or None)
        return self._query(sql, params or {})

    def month(self, start: date, stop: date) -> tuple[pd.DataFrame, pd.DataFrame]:
        """(links, events): the versions announced in [start, stop)."""
        p = {"start": start.isoformat(), "stop": stop.isoformat()}
        return self._run(LINKS_SQL, p), self._run(EVENTS_SQL, p)

    def changed_ids(self, since: datetime) -> list[int]:
        df = self._run(CHANGED_IDS_SQL, {"since": since.isoformat(sep=" ")})
        return sorted({int(x) for x in df["keydevid"]})

    def by_ids(self, ids: Sequence[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
        """(links, events): every version of ``ids``."""
        p = {"ids": [int(i) for i in ids]}
        return self._run(LINKS_BY_IDS_SQL, p), self._run(EVENTS_BY_IDS_SQL, p)

    def dim(self, table: str) -> pd.DataFrame:
        if table not in DIM_TABLES:
            raise ValueError(f"not a lookup table: {table}")
        return self._run(f"select * from ciq_keydev.{table}")

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None


# ---------------------------------------------------------------------------
# Versions -> rows
# ---------------------------------------------------------------------------

def _int(s: pd.Series, dtype: str = "Int64") -> pd.Series:
    return pd.to_numeric(s, errors="coerce").round().astype(dtype)


def _signature(row: dict[str, Any]) -> str:
    payload = {k: v for k, v in row.items() if k not in ("row_sha256", "vintage")}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


_LINKS_SCHEMA = {"keydevid": pl.Int64, "companyid": pl.Int64, "gvkey": pl.String,
                 "companyname": pl.String, "keydeveventtypeid": pl.Int32,
                 "eventtype": pl.String, "keydevtoobjectroletypeid": pl.Int32,
                 "objectroletype": pl.String, "speffectivedate": pl.Datetime("us"),
                 "sptodate": pl.Datetime("us"), "first_version": pl.Boolean,
                 "sourcetypename": pl.String}


def _links_frame(links: pd.DataFrame) -> pl.DataFrame:
    if links.empty:
        return pl.DataFrame(schema=_LINKS_SCHEMA)
    lk = links.copy()
    lk["keydevid"] = _int(lk["keydevid"])
    lk["companyid"] = _int(lk["companyid"])
    for c in ("keydeveventtypeid", "keydevtoobjectroletypeid"):
        lk[c] = _int(lk[c], "Int32")
    for c in ("speffectivedate", "sptodate", "first_speffectivedate"):
        lk[c] = pd.to_datetime(lk[c])
    lk["first_version"] = lk["speffectivedate"] == lk["first_speffectivedate"]
    return pl.from_pandas(lk[list(_LINKS_SCHEMA)]).cast(_LINKS_SCHEMA)


def build_versions(links: pd.DataFrame, events: pd.DataFrame,
                   vintage: datetime) -> tuple[pa.Table, dict[str, int]]:
    """One row per event version, with the link versions overlapping its validity."""
    ev = events.copy()
    ev["keydevid"] = _int(ev["keydevid"])
    ev["announceddatetimezoneid"] = _int(ev["announceddatetimezoneid"], "Int32")
    for c in (*_EVENT_TIMES, "speffectivedate", "sptodate", "first_speffectivedate"):
        ev[c] = pd.to_datetime(ev[c])
    ev = ev[ev["announceddate"].notna()]
    ev["first_version"] = ev["speffectivedate"] == ev["first_speffectivedate"]
    ev["announcedate"] = ev["announceddate"].dt.date
    first_seen = (pl.from_pandas(ev[["keydevid", "first_speffectivedate"]])
                    .group_by("keydevid").agg(pl.col("first_speffectivedate").min()
                                              .alias("e_first")))
    pev = pl.from_pandas(ev[["keydevid", "speffectivedate", "sptodate", "first_version",
                             "headline", "situation", "announcedate", *_EVENT_TIMES,
                             "announceddatetimezoneid"]])
    plk = _links_frame(links).rename({"speffectivedate": "l_from", "sptodate": "l_to",
                                      "first_version": "l_first"})
    overlap = ((pl.col("sptodate").is_null() | (pl.col("l_from") < pl.col("sptodate")))
               & (pl.col("l_to").is_null() | (pl.col("l_to") > pl.col("speffectivedate"))))
    # A link counts from the beginning only if CIQ recorded it with the event's oldest
    # version; a link recorded later was added later (e.g. a second company in 2023).
    pairs = (pev.select(["keydevid", "speffectivedate", "sptodate"])
                .join(plk, on="keydevid", how="inner")
                .filter(overlap)
                .join(first_seen, on="keydevid", how="left")
                .with_columns(pl.col("l_first") & (pl.col("l_from") <= pl.col("e_first")))
                .sort(["keydevid", "speffectivedate", "keydevtoobjectroletypeid", "companyid",
                       "l_from"]))
    companies = pairs.group_by(["keydevid", "speffectivedate"], maintain_order=True).agg(
        pl.struct([pl.col(c) for c in _LINK_FIELDS]
                  + [pl.col("l_from").alias("speffectivedate"),
                     pl.col("l_to").alias("sptodate"),
                     pl.col("l_first").alias("first_version")]).alias("companies"),
        pl.col("sourcetypename").drop_nulls().first())
    merged = (pev.join(companies, on=["keydevid", "speffectivedate"], how="left")
                 .sort(["keydevid", "speffectivedate"]))
    matched = pairs.select(["keydevid", "companyid", "keydevtoobjectroletypeid",
                            "keydeveventtypeid", "l_from"]).unique().height
    counts = {"versions": merged.height,
              "current": int(merged["sptodate"].is_null().sum()),
              "versions_without_links": int(merged["companies"].is_null().sum()),
              "link_versions_unmatched": plk.unique().height - matched}
    rows = merged.to_dicts()
    for r in rows:
        r["companies"] = r["companies"] or []
        r["row_sha256"] = _signature(r)
        r["vintage"] = vintage
    if not rows:
        return KEYDEV_SCHEMA.empty_table(), counts
    return pa.Table.from_pylist(rows, schema=KEYDEV_SCHEMA), counts


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------

def _write_atomic(path: Path, write: Callable[[Path], None]) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        write(tmp)
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def write_table(path: Path, table: pa.Table, **source: Any) -> None:
    meta = {SOURCES_METADATA_KEY: json.dumps({"source": SOURCE, **source}, sort_keys=True,
                                             default=str).encode()}
    table = table.select(KEYDEV_SCHEMA.names).cast(KEYDEV_SCHEMA).replace_schema_metadata(meta)
    _write_atomic(path, lambda tmp: pq.write_table(table, tmp, compression="zstd"))


def plan_files(root: Path) -> list[Path]:
    return sorted(Path(root).glob(f"{PLAN_PREFIX}*.json"))


def data_files(root: Path) -> list[Path]:
    """Base months then update deltas, in resolution order (vintage)."""
    base, deltas = [], []
    for path in sorted(Path(root).glob("*.parquet")):
        if path.name.startswith(UPDATE_PREFIX):
            deltas.append(path)
        elif not path.name.startswith(DIM_PREFIX):
            try:
                parse_key(path.stem)
            except PeriodError:
                continue
            base.append(path)
    return base + deltas


def latest_speffectivedate(root: Path) -> datetime | None:
    """The newest version start recorded (event or link version)."""
    files = data_files(root)
    if not files:
        return None
    lf = pl.scan_parquet(files)
    ev = lf.select(pl.col("speffectivedate").max()).collect().item()
    lk = lf.select(pl.col("companies").list.eval(
        pl.element().struct.field("speffectivedate")).list.max().max()).collect().item()
    found = [t for t in (ev, lk) if t is not None]
    return max(found) if found else None


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------

def _valid_at(field: Callable[[str], pl.Expr], t: datetime) -> pl.Expr:
    """A version valid at ``t``: started by then (the oldest one counts from the
    beginning: CIQ's history only starts in 2018) and not closed yet."""
    return ((field("first_version") | (field("speffectivedate") <= t))
            & (field("sptodate").is_null() | (field("sptodate") > t)))


def read_state(root: Path, *, start: date | None = None, end: date | None = None,
               as_of: datetime | None = None, versions: bool = False,
               columns: list[str] | None = None) -> pl.DataFrame:
    """Event versions (latest copy of each by vintage).

    Default: the current versions. ``as_of``: the versions valid at ``as_of`` and entered
    by then, their companies filtered the same way. ``versions=True``: every version.
    ``start`` / ``end`` filter on the announcement date."""
    cols = list(dict.fromkeys([*KEY, "sptodate", "first_version", "announcedate",
                               "entereddateutc", "vintage",
                               *(columns or KEYDEV_SCHEMA.names)]))
    frames = [pl.read_parquet(p, columns=cols).with_columns(pl.lit(i).alias("_order"))
              for i, p in enumerate(data_files(root))]
    if not frames:
        return pl.from_arrow(KEYDEV_SCHEMA.empty_table()).select(cols)
    df = (pl.concat(frames, how="vertical_relaxed")
            .sort(["vintage", "_order"], maintain_order=True)
            .group_by(list(KEY), maintain_order=True).last()
            .drop("_order"))
    if as_of is not None:
        t = _naive_utc(as_of)
        df = df.filter(_valid_at(pl.col, t) & (pl.col("entereddateutc") <= t))
        if "companies" in df.columns:
            df = df.with_columns(pl.col("companies").list.eval(pl.element().filter(
                _valid_at(lambda f: pl.element().struct.field(f), t))))
    elif not versions:
        df = df.filter(pl.col("sptodate").is_null())
    if start is not None:
        df = df.filter(pl.col("announcedate") >= start)
    if end is not None:
        df = df.filter(pl.col("announcedate") <= end)
    return df.sort(["announcedate", *KEY]).select(cols)


def read_keydev(artifact: Artifact, *, start: date | str | None = None,
                end: date | str | None = None, as_of: datetime | str | None = None,
                versions: bool = False, columns: list[str] | None = None,
                explode: bool = False) -> pl.DataFrame:
    """The events of a ``ciq_keydev`` artifact (see ``read_state``). ``explode=True``: one
    row per event x company, the struct fields as ``company_*`` columns."""
    start = date.fromisoformat(start) if isinstance(start, str) else start
    end = date.fromisoformat(end) if isinstance(end, str) else end
    as_of = datetime.fromisoformat(as_of) if isinstance(as_of, str) else as_of
    df = read_state(artifact.path, start=start, end=end, as_of=as_of, versions=versions,
                    columns=columns)
    if explode and "companies" in df.columns:
        df = (df.explode("companies", empty_as_null=True)
                .with_columns(pl.col("companies").struct.rename_fields(
                    [f"company_{f}" for f in _COMPANY_FIELDS]))
                .unnest("companies"))
    return df


# ---------------------------------------------------------------------------
# Job
# ---------------------------------------------------------------------------

def _monthly(layout: Layout) -> Layout:
    """Whole months covering ``layout`` (the job is monthly whatever the requested freq)."""
    return Layout("M", period_of(layout.start, "M").first, period_of(layout.end, "M").last)


class KeyDevJob(Job):
    """Capital IQ Key Developments (WRDS) -> one parquet per month of event versions."""

    kind = KIND
    pipeline_version = PIPELINE_VERSION
    group = GROUP
    layer = "raw"
    hash_pattern = "*.parquet"

    def __init__(self, layout: Layout, plan: dict[str, Any], *, temp: bool = False,
                 source: KeyDevSource | None = None) -> None:
        self.layout, self.plan, self.temp = _monthly(layout), plan, temp
        self.source = source or KeyDevSource()
        self.vintage = parse_vintage(plan["vintage"])
        self._counts: dict[str, int] = {}

    @staticmethod
    def make_plan(mode: str, vintage: datetime, **extra: Any) -> dict[str, Any]:
        return {"mode": mode, "vintage": vintage_tag(vintage),
                "links": "ciq_keydev.wrds_keydev", "events": "ciq_keydev.ciqkeydev", **extra}

    @classmethod
    def new(cls, layout: Layout, *, temp: bool = False, source: KeyDevSource | None = None,
            clock: Clock = utc_now) -> KeyDevJob:
        return cls(_monthly(layout), cls.make_plan("base", clock()), temp=temp, source=source)

    @property
    def mode(self) -> str:
        return self.plan["mode"]

    def params(self) -> dict[str, Any]:
        return {"source": SOURCE, **self.layout.hyperparams()}

    def notes(self) -> str:
        extra = f" since={self.plan['since']}" if self.mode == "update" else ""
        return f"{self.mode} vintage={self.plan['vintage']}{extra}"

    # -- units ---------------------------------------------------------------

    def units(self) -> list[Unit]:
        if self.mode == "update":
            return [Unit("changes", "update")]
        return [Unit(p.key) for p in self.layout.expected()]

    def _path(self, out_dir: Path, key: str) -> Path:
        if self.mode == "update":
            return out_dir / f"{UPDATE_PREFIX}{self.plan['vintage']}.parquet"
        return out_dir / partition_file(key)

    def is_done(self, unit: Unit, out_dir: Path) -> bool:
        return self._path(out_dir, unit.key).exists()

    @contextmanager
    def session(self, ctx: JobContext) -> Iterator[None]:
        path = ctx.out_dir / f"{PLAN_PREFIX}{self.plan['vintage']}.json"
        if not path.exists():
            _write_atomic(path, lambda tmp: tmp.write_text(
                json.dumps(self.plan, indent=2, sort_keys=True)))
        try:
            if self.mode == "base":
                for table in DIM_TABLES:
                    dim = ctx.out_dir / f"{DIM_PREFIX}{table}.parquet"
                    if not dim.exists():
                        frame = pa.Table.from_pandas(self.source.dim(table),
                                                     preserve_index=False)
                        _write_atomic(dim, lambda tmp, f=frame: pq.write_table(
                            f, tmp, compression="zstd"))
            yield
        finally:
            self.source.close()

    def _count(self, counts: dict[str, int]) -> None:
        for k, v in counts.items():
            self._counts[k] = self._counts.get(k, 0) + v

    def run_unit(self, unit: Unit, ctx: JobContext) -> None:
        if self.mode == "update":
            self._update(ctx)
            return
        period = parse_key(unit.key)
        links, events = self.source.month(period.first, period.last + timedelta(days=1))
        table, counts = build_versions(links, events, self.vintage)
        write_table(self._path(ctx.out_dir, unit.key), table, month=unit.key,
                    vintage=self.plan["vintage"], **counts)
        self._count(counts)
        if counts["versions_without_links"] or counts["link_versions_unmatched"]:
            ctx.log.warning("%s: %d version(s) without links, %d link version(s) unmatched",
                            unit.key, counts["versions_without_links"],
                            counts["link_versions_unmatched"])
        ctx.log.info("%s: %d version(s), %d current", unit.key, counts["versions"],
                     counts["current"])

    def _update(self, ctx: JobContext) -> None:
        since = datetime.fromisoformat(self.plan["since"])
        ids = self.source.changed_ids(since)
        seen: dict[tuple[int, Any], str] = {}
        if ids:
            known = read_state(ctx.out_dir, versions=True, columns=[*KEY, "row_sha256"])
            seen = dict(zip(zip(known["keydevid"].to_list(),
                                known["speffectivedate"].to_list()),
                            known["row_sha256"].to_list()))
        parts = []
        for i in range(0, len(ids), ID_BATCH):
            links, events = self.source.by_ids(ids[i:i + ID_BATCH])
            table, counts = build_versions(links, events, self.vintage)
            keep = [seen.get((k, s)) != h for k, s, h in zip(
                table.column("keydevid").to_pylist(),
                table.column("speffectivedate").to_pylist(),
                table.column("row_sha256").to_pylist())]
            parts.append(table.filter(pa.array(keep, type=pa.bool_())))
            self._count(counts)
        out = pa.concat_tables(parts) if parts else KEYDEV_SCHEMA.empty_table()
        write_table(self._path(ctx.out_dir, "changes"), out, since=self.plan["since"],
                    vintage=self.plan["vintage"], changed_events=len(ids))
        self._count({"written": out.num_rows, "changed_events": len(ids)})
        ctx.log.info("update since %s: %d event(s) changed, %d version row(s) written",
                     self.plan["since"], len(ids), out.num_rows)

    def finalize(self, ctx: JobContext) -> None:
        if self.mode == "base":
            keys = pl.concat([pl.read_parquet(p, columns=list(KEY))
                              for p in data_files(ctx.out_dir)
                              if not p.name.startswith(UPDATE_PREFIX)]
                             or [pl.DataFrame(schema={"keydevid": pl.Int64,
                                                      "speffectivedate": pl.Datetime("us")})])
            dup = keys.filter(pl.struct(list(KEY)).is_duplicated())
            if dup.height:
                raise JobError(f"event versions in several rows: {dup.head(5).rows()}")
        ctx.note(f"{self.mode} {self.plan['vintage']}: {self._counts}")

    # -- rebuild -------------------------------------------------------------

    @classmethod
    def from_artifact(cls, artifact: Artifact, index: DatalakeIndex, *,
                      source: KeyDevSource | None = None, **kwargs: Any) -> KeyDevJob:
        """The unfinished execution of a partial artifact (its latest plan)."""
        layout = layout_from_hyperparams(artifact.meta.hyperparams)
        temp = artifact.meta.pipeline_version.endswith(TEMP_SUFFIX)
        plans = plan_files(artifact.path)
        if not plans:                          # killed before its plan was written
            return cls.new(layout, temp=temp, source=source, **kwargs)
        return cls(layout, json.loads(plans[-1].read_text()), temp=temp, source=source)

    @classmethod
    def for_update(cls, artifact: Artifact, index: DatalakeIndex, *,
                   source: KeyDevSource | None = None, clock: Clock = utc_now,
                   **kwargs: Any) -> KeyDevJob:
        """Fetch the events changed since the newest version already stored."""
        layout = layout_from_hyperparams(artifact.meta.hyperparams)
        vintage = clock()
        plans = [json.loads(p.read_text()) for p in plan_files(artifact.path)]
        if plans and vintage_tag(vintage) <= plans[-1]["vintage"]:
            raise JobError("the update vintage must be later than the latest one")
        since = latest_speffectivedate(artifact.path)
        if since is None:
            raise JobError(f"{artifact.artifact_id} holds no version to update from")
        plan = cls.make_plan("update", vintage, since=since.isoformat(sep=" "))
        return cls(layout, plan, temp=artifact.meta.pipeline_version.endswith(TEMP_SUFFIX),
                   source=source)

    @classmethod
    def add_cli_args(cls, parser: Any) -> None:
        add_layout_args(parser, default_freq="M")
        parser.add_argument("--temp", action="store_true", help="agent-created (__TEMP)")

    @classmethod
    def from_args(cls, args: Any, index: DatalakeIndex) -> KeyDevJob:
        if args.partition_freq != "M":
            raise JobError("ciq_keydev is monthly: --partition-freq must be M")
        return cls.new(layout_from_args(args), temp=args.temp)


# ---------------------------------------------------------------------------
# Verifier (registered via pyproject.toml entry points)
# ---------------------------------------------------------------------------

def verify_artifact(artifact: Artifact) -> list:
    """Schema, partition dates, unique (keydevid, speffectivedate), plans and deltas."""
    from datalake.verify import Finding, Severity

    aid = artifact.artifact_id
    findings: list[Finding] = []

    def err(msg: str) -> None:
        findings.append(Finding(Severity.ERROR, aid, msg))

    plans = {json.loads(p.read_text())["vintage"]: json.loads(p.read_text())
             for p in plan_files(artifact.path)}
    if not plans:
        err("no plan file")
    base = []
    for path in data_files(artifact.path):
        if not pq.read_schema(path).remove_metadata().equals(KEYDEV_SCHEMA):
            err(f"{path.name}: schema differs from KEYDEV_SCHEMA")
            continue
        df = pl.read_parquet(path, columns=[*KEY, "announcedate"])
        if df.select(pl.struct(list(KEY)).is_duplicated().any()).item():
            err(f"{path.name}: duplicate (keydevid, speffectivedate)")
        if path.name.startswith(UPDATE_PREFIX):
            tag = path.stem[len(UPDATE_PREFIX):]
            if plans.get(tag, {}).get("mode") != "update":
                err(f"{path.name}: no update plan for vintage {tag}")
            continue
        period = parse_key(path.stem)
        if df.height and (df["announcedate"].min() < period.first
                          or df["announcedate"].max() > period.last):
            err(f"{path.name}: announcedate outside {period.first}..{period.last}")
        base.append(df.select(list(KEY)))
    if base and pl.concat(base).select(pl.struct(list(KEY)).is_duplicated().any()).item():
        err("an event version in several base months")
    return findings


__all__ = ["KEYDEV_SCHEMA", "KIND", "PIPELINE_VERSION", "KeyDevJob", "KeyDevSource",
           "build_versions", "latest_speffectivedate", "read_keydev", "read_state",
           "verify_artifact"]
