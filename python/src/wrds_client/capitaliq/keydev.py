"""Capital IQ Key Developments (WRDS) -> ``ciq_keydev`` raw artifact, one parquet per month.

Source (WRDS ``ciq_keydev``, checked live with ``EXPLAIN``):

    wrds_keydev   one row per event x company x role (45.6 M rows, 42 GB). The only table
                  indexed on a date (``announcedate``): the month's event ids and company
                  links are read here, without its (repeated) text columns.
    ciqkeydev     one row per event (35 M rows, 21 GB), indexed on ``keydevid`` only: the
                  headline / situation text and every date, fetched once per event with a
                  server-side semi-join on the month's ids.

Base load (``jobs start ciq_keydev``): unit = month of ``announcedate`` (the announcement's
local date). ``YYYY-MM.parquet`` holds one row per event (``KEYDEV_SCHEMA``): text, dates,
source type and ``companies``, a list of structs (company id, GVKEY, name, event type,
role) ordered by role then company. A month without events is an empty file.

Update (``jobs update <id>``): re-pulls the last ``REPULL_MONTHS`` months of the declared
range up to the update's ``end`` (default: the last complete month) and writes
``update-<vintage>-YYYY-MM.parquet`` with the events that are new or whose content changed
(``row_sha256``); empty when nothing changed. Base files are never rewritten. Events CIQ
back-fills into older months, or deletes, are not picked up: rebuild for those.

Point in time: partitions follow the announcement date, but CIQ back-filled its history
(``entereddateutc`` can be years later). ``read_keydev(as_of=...)`` keeps the events entered
by ``as_of``.

Lookups (event type -> category, role, time zone) are written once by the base load as
``dim-<table>.parquet``.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Callable, Iterator
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
PIPELINE_VERSION = "v0.1.0"
SOURCE = "wrds_ciq_keydev"
GROUP = "CapIQ"
REPULL_MONTHS = 3

PLAN_PREFIX, UPDATE_PREFIX, DIM_PREFIX = "plan-", "update-", "dim-"
SOURCES_METADATA_KEY = b"ciq_keydev.source"

Clock = Callable[[], datetime]
Query = Callable[[str, dict[str, Any]], pd.DataFrame]

COMPANY_TYPE = pa.struct([
    ("companyid", pa.int64()), ("gvkey", pa.string()), ("companyname", pa.string()),
    ("keydeveventtypeid", pa.int32()), ("eventtype", pa.string()),
    ("keydevtoobjectroletypeid", pa.int32()), ("objectroletype", pa.string())])

KEYDEV_SCHEMA = pa.schema([
    ("keydevid", pa.int64()),
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

LINKS_SQL = """
select keydevid, companyid, companyname, gvkey, keydeveventtypeid, eventtype,
       keydevtoobjectroletypeid, objectroletype, sourcetypename, announcedate
from ciq_keydev.wrds_keydev
where announcedate >= %(start)s and announcedate < %(stop)s
"""

EVENTS_SQL = """
select keydevid, headline, situation, announceddate, announceddateutc,
       announceddatetimezoneid, entereddate, entereddateutc, lastmodifieddate,
       lastmodifieddateutc, mostimportantdateutc
from ciq_keydev.ciqkeydev
where keydevid in (select distinct keydevid from ciq_keydev.wrds_keydev
                   where announcedate >= %(start)s and announcedate < %(stop)s)
"""

DIM_TABLES = ("ciqkeydevcategorytype", "ciqkeydevobjectroletype", "ciqkeydevtimezone")


def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def vintage_tag(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_vintage(tag: str) -> datetime:
    return datetime.strptime(tag, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)


def last_complete_month_end(now: datetime) -> date:
    return now.date().replace(day=1) - timedelta(days=1)


# ---------------------------------------------------------------------------
# Source
# ---------------------------------------------------------------------------

class KeyDevSource:
    """The two monthly queries and the lookups, on a WRDS connection (``query`` injectable:
    ``(sql, params) -> DataFrame``; default ``WRDSClient.from_env().raw.raw_sql``)."""

    def __init__(self, query: Query | None = None) -> None:
        self._query, self._conn = query, None

    def _run(self, sql: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
        if self._query is None:
            from wrds_client.client import WRDSClient

            self._conn = WRDSClient.from_env().raw
            self._query = lambda s, p: self._conn.raw_sql(s, params=p or None)
        return self._query(sql, params or {})

    def links(self, start: date, stop: date) -> pd.DataFrame:
        return self._run(LINKS_SQL, {"start": start.isoformat(), "stop": stop.isoformat()})

    def events(self, start: date, stop: date) -> pd.DataFrame:
        return self._run(EVENTS_SQL, {"start": start.isoformat(), "stop": stop.isoformat()})

    def dim(self, table: str) -> pd.DataFrame:
        if table not in DIM_TABLES:
            raise ValueError(f"not a lookup table: {table}")
        return self._run(f"select * from ciq_keydev.{table}")

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None


# ---------------------------------------------------------------------------
# Month frame
# ---------------------------------------------------------------------------

def _int(s: pd.Series, dtype: str = "Int64") -> pd.Series:
    return pd.to_numeric(s, errors="coerce").round().astype(dtype)


def _signature(row: dict[str, Any]) -> str:
    payload = {k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in row.items()
               if k not in ("row_sha256", "vintage")}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def build_month(links: pd.DataFrame, events: pd.DataFrame,
                vintage: datetime) -> tuple[pa.Table, dict[str, int]]:
    """One row per event with its ``companies`` list; counts of orphans on either side."""
    lk = links.copy()
    lk["keydevid"] = _int(lk["keydevid"])
    lk["companyid"] = _int(lk["companyid"])
    for c in ("keydeveventtypeid", "keydevtoobjectroletypeid"):
        lk[c] = _int(lk[c], "Int32")
    lk["announcedate"] = pd.to_datetime(lk["announcedate"]).dt.date
    lk = lk.sort_values(["keydevid", "keydevtoobjectroletypeid", "companyid"], kind="stable")
    ev = events.copy()
    ev["keydevid"] = _int(ev["keydevid"])
    ev["announceddatetimezoneid"] = _int(ev["announceddatetimezoneid"], "Int32")
    for c in _EVENT_TIMES:
        ev[c] = pd.to_datetime(ev[c])

    plk = pl.from_pandas(lk[["keydevid", *_COMPANY_FIELDS, "sourcetypename", "announcedate"]])
    grouped = plk.group_by("keydevid", maintain_order=True).agg(
        pl.struct(_COMPANY_FIELDS).alias("companies"),
        pl.col("sourcetypename").drop_nulls().first(),
        pl.col("announcedate").first())
    pev = pl.from_pandas(ev[["keydevid", "headline", "situation", *_EVENT_TIMES,
                             "announceddatetimezoneid"]])
    merged = pev.join(grouped, on="keydevid", how="inner").sort("keydevid")
    no_event = ~grouped["keydevid"].is_in(pev["keydevid"].implode())
    counts = {"events": merged.height,
              "events_without_links": pev.height - merged.height,
              "links_without_event": int(no_event.sum())}
    rows = merged.to_dicts()
    for r in rows:
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


def update_name(vintage: str, month: str) -> str:
    return f"{UPDATE_PREFIX}{vintage}-{month}.parquet"


def data_files(root: Path) -> list[Path]:
    """Base months then update deltas, in resolution order (vintage, then month)."""
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


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------

def read_state(root: Path, *, start: date | None = None, end: date | None = None,
               as_of: datetime | None = None, columns: list[str] | None = None) -> pl.DataFrame:
    """Latest version of every event (by vintage), optionally within [start, end] and
    entered by ``as_of`` (``entereddateutc``)."""
    cols = list(dict.fromkeys(["keydevid", "announcedate", "entereddateutc", "vintage",
                               *(columns or KEYDEV_SCHEMA.names)]))
    frames = [pl.read_parquet(p, columns=cols).with_columns(pl.lit(i).alias("_order"))
              for i, p in enumerate(data_files(root))]
    if not frames:
        return pl.from_arrow(KEYDEV_SCHEMA.empty_table()).select(cols)
    df = (pl.concat(frames, how="vertical_relaxed")
            .sort(["vintage", "_order"], maintain_order=True)
            .group_by("keydevid", maintain_order=True).last()
            .drop("_order"))
    if start is not None:
        df = df.filter(pl.col("announcedate") >= start)
    if end is not None:
        df = df.filter(pl.col("announcedate") <= end)
    if as_of is not None:
        naive = as_of.astimezone(timezone.utc).replace(tzinfo=None) if as_of.tzinfo else as_of
        df = df.filter(pl.col("entereddateutc") <= naive)
    return df.sort(["announcedate", "keydevid"]).select(cols)


def read_keydev(artifact: Artifact, *, start: date | str | None = None,
                end: date | str | None = None, as_of: datetime | str | None = None,
                columns: list[str] | None = None, explode: bool = False) -> pl.DataFrame:
    """The events of a ``ciq_keydev`` artifact. ``explode=True``: one row per event x
    company (the struct fields as columns)."""
    start = date.fromisoformat(start) if isinstance(start, str) else start
    end = date.fromisoformat(end) if isinstance(end, str) else end
    as_of = datetime.fromisoformat(as_of) if isinstance(as_of, str) else as_of
    df = read_state(artifact.path, start=start, end=end, as_of=as_of, columns=columns)
    if explode and "companies" in df.columns:
        df = df.explode("companies", empty_as_null=True).unnest("companies")
    return df


# ---------------------------------------------------------------------------
# Job
# ---------------------------------------------------------------------------

def _monthly(layout: Layout) -> Layout:
    """Whole months covering ``layout`` (the job is monthly whatever the requested freq)."""
    return Layout("M", period_of(layout.start, "M").first, period_of(layout.end, "M").last)


class KeyDevJob(Job):
    """Capital IQ Key Developments (WRDS) -> one parquet per month (raw layer)."""

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
        self._counts = {"events": 0, "changed": 0, "events_without_links": 0,
                        "links_without_event": 0}

    @staticmethod
    def make_plan(mode: str, vintage: datetime, first: date, last: date) -> dict[str, Any]:
        return {"mode": mode, "vintage": vintage_tag(vintage), "first_month": first.isoformat(),
                "last_month": last.isoformat(), "links": "ciq_keydev.wrds_keydev",
                "events": "ciq_keydev.ciqkeydev"}

    @classmethod
    def new(cls, layout: Layout, *, temp: bool = False, source: KeyDevSource | None = None,
            clock: Clock = utc_now) -> KeyDevJob:
        m = _monthly(layout)
        return cls(m, cls.make_plan("base", clock(), m.start, m.end), temp=temp, source=source)

    @property
    def mode(self) -> str:
        return self.plan["mode"]

    def params(self) -> dict[str, Any]:
        return {"source": SOURCE, **self.layout.hyperparams()}

    def notes(self) -> str:
        return (f"{self.mode} vintage={self.plan['vintage']} months="
                f"{self.plan['first_month'][:7]}..{self.plan['last_month'][:7]}")

    # -- units ---------------------------------------------------------------

    def units(self) -> list[Unit]:
        months = Layout("M", date.fromisoformat(self.plan["first_month"]),
                        date.fromisoformat(self.plan["last_month"])).expected()
        return [Unit(p.key) for p in months]

    def _path(self, out_dir: Path, key: str) -> Path:
        if self.mode == "update":
            return out_dir / update_name(self.plan["vintage"], key)
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

    def run_unit(self, unit: Unit, ctx: JobContext) -> None:
        period = parse_key(unit.key)
        stop = period.last + timedelta(days=1)
        links = self.source.links(period.first, stop)
        events = self.source.events(period.first, stop)
        table, counts = build_month(links, events, self.vintage)
        if self.mode == "update":
            known = read_state(ctx.out_dir, start=period.first, end=period.last,
                               columns=["keydevid", "row_sha256"])
            seen = dict(zip(known["keydevid"].to_list(), known["row_sha256"].to_list()))
            keep = [seen.get(k) != s for k, s in zip(table.column("keydevid").to_pylist(),
                                                    table.column("row_sha256").to_pylist())]
            table = table.filter(pa.array(keep, type=pa.bool_()))
            self._counts["changed"] += table.num_rows
        write_table(self._path(ctx.out_dir, unit.key), table, month=unit.key,
                    vintage=self.plan["vintage"], **counts)
        for k in ("events", "events_without_links", "links_without_event"):
            self._counts[k] += counts[k]
        if counts["events_without_links"] or counts["links_without_event"]:
            ctx.log.warning("%s: %d event(s) without links, %d link group(s) without event",
                            unit.key, counts["events_without_links"],
                            counts["links_without_event"])
        ctx.log.info("%s: %d event(s)%s", unit.key, counts["events"],
                     f", {table.num_rows} new/changed" if self.mode == "update" else "")

    def finalize(self, ctx: JobContext) -> None:
        if self.mode == "base":
            ids = pl.concat([pl.read_parquet(p, columns=["keydevid"])
                             for p in data_files(ctx.out_dir)
                             if not p.name.startswith(UPDATE_PREFIX)]
                            or [pl.DataFrame(schema={"keydevid": pl.Int64})])
            dup = ids.filter(pl.col("keydevid").is_duplicated())["keydevid"].unique().to_list()
            if dup:
                raise JobError(f"keydevid in several months: {sorted(dup)[:10]}")
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
                   source: KeyDevSource | None = None, end: date | None = None,
                   clock: Clock = utc_now, **kwargs: Any) -> KeyDevJob:
        """Re-pull the last ``REPULL_MONTHS`` months already pulled (base or earlier update)
        up to ``end`` (default: the last complete month); writes new / changed events only."""
        layout = _monthly(layout_from_hyperparams(artifact.meta.hyperparams))
        vintage = clock()
        plans = [json.loads(p.read_text()) for p in plan_files(artifact.path)]
        if plans and vintage_tag(vintage) <= plans[-1]["vintage"]:
            raise JobError("the update vintage must be later than the latest one")
        end = end or last_complete_month_end(vintage)
        pulled = max([layout.end, *(date.fromisoformat(p["last_month"]) for p in plans)])
        first = period_of(pulled, "M").first
        for _ in range(REPULL_MONTHS - 1):
            first = period_of(first - timedelta(days=1), "M").first
        first = max(first, layout.start)
        if end < first:
            raise JobError(f"update end {end} is before the re-pull window start {first}")
        plan = cls.make_plan("update", vintage, first, period_of(end, "M").last)
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
    """Schema, months, partition dates, unique ids, plans and deltas."""
    from datalake.verify import Finding, Severity

    aid = artifact.artifact_id
    findings: list[Finding] = []

    def err(msg: str) -> None:
        findings.append(Finding(Severity.ERROR, aid, msg))

    plans = {json.loads(p.read_text())["vintage"]: json.loads(p.read_text())
             for p in plan_files(artifact.path)}
    if not plans:
        err("no plan file")
    base_ids: list[int] = []
    for path in data_files(artifact.path):
        if not pq.read_schema(path).remove_metadata().equals(KEYDEV_SCHEMA):
            err(f"{path.name}: schema differs from KEYDEV_SCHEMA")
            continue
        df = pl.read_parquet(path, columns=["keydevid", "announcedate"])
        if df["keydevid"].is_duplicated().any():
            err(f"{path.name}: duplicate keydevid")
        if path.name.startswith(UPDATE_PREFIX):
            tag, month = path.stem[len(UPDATE_PREFIX):].rsplit("-", 2)[0], path.stem[-7:]
            if plans.get(tag, {}).get("mode") != "update":
                err(f"{path.name}: no update plan for vintage {tag}")
        else:
            month = path.stem
            base_ids += df["keydevid"].to_list()
        period = parse_key(month)
        if df.height and (df["announcedate"].min() < period.first
                          or df["announcedate"].max() > period.last):
            err(f"{path.name}: announcedate outside {period.first}..{period.last}")
    if len(base_ids) != len(set(base_ids)):
        err("keydevid in several base months")
    return findings


__all__ = ["KEYDEV_SCHEMA", "KIND", "PIPELINE_VERSION", "KeyDevJob", "KeyDevSource",
           "build_month", "read_keydev", "read_state", "verify_artifact"]
