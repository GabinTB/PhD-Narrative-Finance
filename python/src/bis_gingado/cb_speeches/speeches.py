"""BIS central-banker speeches -> ``cb_speeches`` artifact (historical load + updates).

Source: BIS's bulk ``speeches.zip`` (every speech since 1996, one CSV;
``bis_gingado.cb_speeches.bis``), downloaded once per execution, resumed across the cut
connections the BIS server is prone to.

Base load (``jobs start cb_speeches``): one parquet per partition of the
chosen ``partition_freq`` (``datalake.layout``, default Y), rows routed by
speech date. Unit = partition; the bulk file is downloaded once per run and
kept in memory while the partitions are written. A partition without speeches
is an empty file (so every expected partition is present when complete).
Speeches dated outside [start, end] are not written (counted in the run note).

Update (``jobs update <artifact_id>``): a new execution of the same artifact,
one unit. An unchanged ETag (HEAD only) or zip sha256 writes an empty delta;
otherwise the bulk file is diffed against the artifact's current state and the
delta ``update-<vintage>.parquet`` holds ``new`` / ``revised`` / ``removed``
rows (speeches dated from the declared ``start`` on, including after ``end``).
Existing files are never rewritten.

Files (all hashed):
    <period key>.parquet              base partitions (change = base)
    update-<vintage>.parquet          update deltas (possibly empty)
    raw-speeches-<sha12>.zip          every raw zip read, content-addressed
    plan-<vintage>.json               what an execution fetches (mode, URL)

Every parquet carries the manifest of the zip it was read from (URL, ETag,
Last-Modified, sha256, size, fetch time) in its key-value metadata.

``read_speeches(artifact, as_of=...)`` resolves the latest version of every
speech recorded at or before ``as_of`` (removed ones dropped). Base rows carry
the base load's vintage, not BIS's publication time: point-in-time
availability is exact only from the first update on.
"""
from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from bis_gingado.cb_speeches.bis import (
    DEFAULT_URL,
    BISClient,
    Manifest,
    SpeechDataError,
    normalise,
    read_zip,
)
from bis_gingado.cb_speeches.schema import (
    CHANGES,
    SOURCES_METADATA_KEY,
    SPEECH_SCHEMA,
    STATE_COLUMNS,
)
from datalake.jobs import TEMP_SUFFIX, Job, JobContext, Unit
from datalake.layout import Layout, add_layout_args, layout_from_args, layout_from_hyperparams
from datalake.periods import PeriodError, parse_key, partition_file

if TYPE_CHECKING:
    from datalake import Artifact, DatalakeIndex

log = logging.getLogger(__name__)

KIND = "cb_speeches"
PIPELINE_VERSION = "v0.2.0"      # v0.2.0: bulk speeches.zip (v0.1.0: yearly zips)
SOURCE = "bis_cbspeeches"

PLAN_PREFIX, UPDATE_PREFIX, RAW_PREFIX = "plan-", "update-", "raw-"

Clock = Callable[[], datetime]


def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def vintage_tag(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_vintage(tag: str) -> datetime:
    return datetime.strptime(tag, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)


def whole_periods(layout: Layout) -> Layout:
    first, last = layout.expected()[0], layout.expected()[-1]
    return Layout(layout.freq, first.first, last.last)


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


def write_table(path: Path, frame: pd.DataFrame, schema: pa.Schema,
                metadata: dict[bytes, bytes]) -> None:
    """Write ``frame`` with exactly ``schema`` (zstd, atomic, metadata attached)."""
    table = pa.Table.from_pandas(frame[schema.names], schema=schema, preserve_index=False)
    table = table.replace_schema_metadata(metadata)
    _write_atomic(path, lambda tmp: pq.write_table(table, tmp, compression="zstd"))


def _sources_metadata(manifests: list[Manifest], **extra: Any) -> dict[bytes, bytes]:
    payload = {"sources": [m.to_dict() for m in manifests], **extra}
    return {SOURCES_METADATA_KEY: json.dumps(payload, sort_keys=True).encode()}


def file_sources(path: Path) -> dict[str, Any]:
    """The raw-zip manifest recorded in a speeches parquet."""
    meta = pq.read_schema(path).metadata or {}
    raw = meta.get(SOURCES_METADATA_KEY)
    return json.loads(raw) if raw else {"sources": []}


def raw_zip_path(out_dir: Path, manifest: Manifest) -> Path:
    stem = manifest.zip.removesuffix(".zip")
    return out_dir / f"{RAW_PREFIX}{stem}-{manifest.sha256[:12]}.zip"


def plan_files(root: Path) -> list[Path]:
    return sorted(Path(root).glob(f"{PLAN_PREFIX}*.json"))


def data_files(root: Path) -> list[Path]:
    """Base partitions then deltas, in resolution order (vintage, then name)."""
    base, deltas = [], []
    for path in sorted(Path(root).glob("*.parquet")):
        if path.name.startswith(UPDATE_PREFIX):
            deltas.append(path)
        else:
            try:
                parse_key(path.stem)
            except PeriodError:
                continue
            base.append(path)
    return base + deltas


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------

def _resolve(frames: list[pl.DataFrame], as_of: datetime | None) -> pl.DataFrame:
    if not frames:
        return pl.DataFrame(schema=pl.from_arrow(SPEECH_SCHEMA.empty_table()).schema)
    df = pl.concat([f.with_columns(pl.lit(i).alias("_order")) for i, f in enumerate(frames)],
                   how="vertical_relaxed")
    if as_of is not None:
        df = df.filter(pl.col("vintage") <= as_of)
    df = (df.sort(["vintage", "_order"], maintain_order=True)
            .group_by("speech_id", maintain_order=True).last()
            .filter(pl.col("change") != "removed")
            .drop("_order"))
    return df.sort(["date", "speech_id"]) if "date" in df.columns else df


def read_state(root: Path, as_of: datetime | None = None,
               columns: tuple[str, ...] = STATE_COLUMNS) -> pl.DataFrame:
    """Current state of a speeches directory: latest version per speech_id."""
    cols = list(dict.fromkeys(("speech_id", "vintage", "change", *columns)))
    return _resolve([pl.read_parquet(p, columns=cols) for p in data_files(root)], as_of)


def read_speeches(artifact: Artifact, as_of: datetime | str | None = None,
                  columns: list[str] | None = None) -> pl.DataFrame:
    """The speeches of ``artifact`` as known at ``as_of`` (default: latest)."""
    if isinstance(as_of, str):
        as_of = datetime.fromisoformat(as_of)
    if as_of is not None and as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=timezone.utc)
    return read_state(artifact.path, as_of, tuple(columns or SPEECH_SCHEMA.names))


# ---------------------------------------------------------------------------
# Job
# ---------------------------------------------------------------------------

class SpeechesJob(Job):
    """BIS central-banker speeches (bulk zip) -> cb_speeches partitions / updates."""

    kind = KIND
    pipeline_version = PIPELINE_VERSION
    hash_pattern = "*"

    def __init__(self, layout: Layout, plan: dict[str, Any], *, temp: bool = False,
                 transport: Any = None, timeout: float = 300.0, clock: Clock = utc_now,
                 retry_wait_s: float = 2.0) -> None:
        self.layout, self.plan, self.temp = whole_periods(layout), plan, temp
        self.clock = clock
        self.client = BISClient(plan["url"], timeout=timeout, transport=transport,
                                retry_wait_s=retry_wait_s)
        self.vintage = parse_vintage(plan["vintage"])
        self._loaded: tuple[Manifest, pd.DataFrame] | None = None
        self._outside = 0
        self._changes: dict[str, int] = {}

    # -- construction -------------------------------------------------------

    @staticmethod
    def make_plan(mode: str, url: str, vintage: datetime) -> dict[str, Any]:
        return {"mode": mode, "vintage": vintage_tag(vintage), "url": url}

    @classmethod
    def new(cls, layout: Layout, *, url: str = DEFAULT_URL, temp: bool = False,
            transport: Any = None, clock: Clock = utc_now, **kwargs: Any) -> SpeechesJob:
        """A base load of every speech dated in ``layout``."""
        plan = cls.make_plan("base", url, clock())
        return cls(whole_periods(layout), plan, temp=temp, transport=transport, clock=clock,
                   **kwargs)

    @property
    def mode(self) -> str:
        return self.plan["mode"]

    @property
    def plan_name(self) -> str:
        return f"{PLAN_PREFIX}{self.plan['vintage']}.json"

    def params(self) -> dict[str, Any]:
        return {"source": SOURCE, **self.layout.hyperparams()}

    def notes(self) -> str:
        return f"{self.mode} vintage={self.plan['vintage']} url={self.plan['url']}"

    # -- units ---------------------------------------------------------------

    def units(self) -> list[Unit]:
        if self.mode == "update":
            return [Unit("all", "update")]
        return [Unit(p.key) for p in self.layout.expected()]

    def _delta_path(self, out_dir: Path) -> Path:
        return out_dir / f"{UPDATE_PREFIX}{self.plan['vintage']}.parquet"

    def is_done(self, unit: Unit, out_dir: Path) -> bool:
        if self.mode == "update":
            return self._delta_path(out_dir).exists()
        return (out_dir / partition_file(unit.key)).exists()

    @contextmanager
    def session(self, ctx: JobContext) -> Iterator[None]:
        path = ctx.out_dir / self.plan_name
        if not path.exists():
            _write_atomic(path, lambda tmp: tmp.write_text(
                json.dumps(self.plan, indent=2, sort_keys=True)))
        try:
            yield
        finally:
            self.client.close()

    def _load(self, out_dir: Path) -> tuple[Manifest, pd.DataFrame]:
        """The bulk file, downloaded once per run and kept as a raw file."""
        if self._loaded is None:
            manifest, content = self.client.download(self.clock())
            raw = raw_zip_path(out_dir, manifest)
            if not raw.exists():
                _write_atomic(raw, lambda tmp: tmp.write_bytes(content))
            df = normalise(read_zip(content, manifest.zip), manifest.zip)
            inside = (df["date"] >= self.layout.start) & (df["date"] <= self.layout.end)
            self._outside = int((~inside).sum())
            if self._outside and self.mode == "base":
                log.warning("%s: %d speech(es) dated outside %s..%s are not written",
                            manifest.zip, self._outside, self.layout.start, self.layout.end)
            self._loaded = (manifest, df)
        return self._loaded

    def _rows(self, df: pd.DataFrame, change: str) -> pd.DataFrame:
        out = df.copy()
        out["vintage"] = pd.Timestamp(self.vintage)
        out["change"] = change
        return out

    def run_unit(self, unit: Unit, ctx: JobContext) -> None:
        if self.mode == "update":
            self._update(ctx)
        else:
            self._base_unit(unit, ctx)

    # -- base load -----------------------------------------------------------

    def _base_unit(self, unit: Unit, ctx: JobContext) -> None:
        """Write partition ``unit`` from the bulk file (downloaded once per run)."""
        manifest, rows = self._load(ctx.out_dir)
        period = parse_key(unit.key)
        part = rows[(rows["date"] >= period.first) & (rows["date"] <= period.last)]
        write_table(ctx.out_dir / partition_file(unit.key), self._rows(part, "base"),
                    SPEECH_SCHEMA, _sources_metadata([manifest]))
        ctx.log.info("%s: %d speech(es)", unit.key, len(part))

    # -- update --------------------------------------------------------------

    def _latest_manifest(self, out_dir: Path) -> dict[str, Any] | None:
        latest = None
        for path in data_files(out_dir):
            for m in file_sources(path)["sources"]:
                if m.get("sha256") and (latest is None
                                        or m["fetched_at"] >= latest["fetched_at"]):
                    latest = m
        return latest

    def _update(self, ctx: JobContext) -> None:
        out = self._delta_path(ctx.out_dir)
        recorded = self._latest_manifest(ctx.out_dir)
        head = self.client.head()
        empty = pd.DataFrame({c: pd.Series(dtype=object) for c in SPEECH_SCHEMA.names})
        if recorded and head.etag and head.etag == recorded.get("etag"):
            write_table(out, empty, SPEECH_SCHEMA,
                        _sources_metadata([], unchanged=recorded, checked=head.to_dict()))
            ctx.log.info("%s unchanged (ETag)", head.zip)
            return
        manifest, fresh = self._load(ctx.out_dir)
        if recorded and recorded.get("sha256") == manifest.sha256:
            write_table(out, empty, SPEECH_SCHEMA, _sources_metadata([manifest], unchanged=True))
            ctx.log.info("%s unchanged (sha256)", manifest.zip)
            return
        state = read_state(ctx.out_dir, columns=("speech_id", "content_sha256", "date"))
        known = dict(zip(state["speech_id"].to_list(), state["content_sha256"].to_list()))
        fresh = fresh[(fresh["date"] >= self.layout.start)]
        is_new = ~fresh["speech_id"].isin(known.keys())
        is_rev = ~is_new & (fresh["content_sha256"] != fresh["speech_id"].map(known))
        gone = state.filter(~pl.col("speech_id").is_in(fresh["speech_id"].tolist()))
        removed = pd.DataFrame({
            "speech_id": gone["speech_id"].to_list(), "date": gone["date"].to_list(),
            **{c: None for c in ("url", "title", "description", "author", "text",
                                 "content_sha256")},
            "source_zip": manifest.zip})
        delta = pd.concat([self._rows(fresh[is_new], "new"), self._rows(fresh[is_rev], "revised"),
                           self._rows(removed, "removed")], ignore_index=True)
        write_table(out, delta, SPEECH_SCHEMA, _sources_metadata([manifest]))
        self._changes = {"new": int(is_new.sum()), "revised": int(is_rev.sum()),
                         "removed": len(removed)}
        ctx.log.info("%s: %s", manifest.zip, self._changes)

    # -- end -----------------------------------------------------------------

    def finalize(self, ctx: JobContext) -> None:
        if self.mode == "update":
            ctx.note(f"update {self.plan['vintage']}: {self._changes or 'no change'}")
            return
        ids = pl.concat([pl.read_parquet(p, columns=["speech_id"])
                         for p in data_files(ctx.out_dir)] or
                        [pl.DataFrame({"speech_id": []}, schema={"speech_id": pl.String})])
        dup = ids.filter(pl.col("speech_id").is_duplicated())["speech_id"].unique().to_list()
        if dup:
            raise SpeechDataError(f"speech ids in several partitions: {sorted(dup)[:10]}")
        ctx.note(f"base {self.plan['vintage']}: {ids.height} speeches; outside the range "
                 f"(not written): {self._outside}")

    # -- rebuild -------------------------------------------------------------

    @classmethod
    def from_artifact(cls, artifact: Artifact, index: DatalakeIndex, *,
                      transport: Any = None, **kwargs: Any) -> SpeechesJob:
        """The unfinished execution of a partial artifact (its latest plan)."""
        layout = layout_from_hyperparams(artifact.meta.hyperparams)
        temp = artifact.meta.pipeline_version.endswith(TEMP_SUFFIX)
        plans = plan_files(artifact.path)
        if not plans:                         # killed before its plan was written
            return cls.new(layout, temp=temp, transport=transport, **kwargs)
        plan = json.loads(plans[-1].read_text())
        return cls(layout, plan, temp=temp, transport=transport, **kwargs)

    @classmethod
    def for_update(cls, artifact: Artifact, index: DatalakeIndex, *, transport: Any = None,
                   url: str | None = None, clock: Clock = utc_now,
                   **kwargs: Any) -> SpeechesJob:
        """Check the bulk file for new, revised or removed speeches."""
        layout = layout_from_hyperparams(artifact.meta.hyperparams)
        plans = plan_files(artifact.path)
        first = json.loads(plans[0].read_text())
        vintage = clock()
        if vintage_tag(vintage) <= json.loads(plans[-1].read_text())["vintage"]:
            raise SpeechDataError("the update vintage must be later than the latest one")
        plan = cls.make_plan("update", url or first["url"], vintage)
        return cls(layout, plan, temp=artifact.meta.pipeline_version.endswith(TEMP_SUFFIX),
                   transport=transport, clock=clock, **kwargs)

    @classmethod
    def add_cli_args(cls, parser: Any) -> None:
        add_layout_args(parser, default_freq="Y")
        parser.add_argument("--url", default=DEFAULT_URL,
                            help="BIS bulk speeches.zip (every speech since 1996)")
        parser.add_argument("--temp", action="store_true", help="agent-created (__TEMP)")

    @classmethod
    def from_args(cls, args: Any, index: DatalakeIndex) -> SpeechesJob:
        return cls.new(layout_from_args(args), url=args.url, temp=args.temp)


# ---------------------------------------------------------------------------
# Verifier (registered via pyproject.toml entry points)
# ---------------------------------------------------------------------------

def verify_artifact(artifact: Artifact) -> list:
    """Schema, ids, partition dates, raw-zip hashes, plans and deltas."""
    import hashlib

    from datalake.verify import Finding, Severity

    aid = artifact.artifact_id
    findings: list[Finding] = []

    def err(msg: str) -> None:
        findings.append(Finding(Severity.ERROR, aid, msg))

    plans = {json.loads(p.read_text())["vintage"]: json.loads(p.read_text())
             for p in plan_files(artifact.path)}
    if not plans:
        err("no plan file")
    base_ids: list[str] = []
    for path in data_files(artifact.path):
        schema = pq.read_schema(path).remove_metadata()
        if not schema.equals(SPEECH_SCHEMA):
            err(f"{path.name}: schema differs from SPEECH_SCHEMA")
            continue
        df = pl.read_parquet(path, columns=["speech_id", "date", "change", "vintage"])
        if not set(df["change"].unique().to_list()) <= set(CHANGES):
            err(f"{path.name}: unknown change values")
        if path.name.startswith(UPDATE_PREFIX):
            tag = path.stem[len(UPDATE_PREFIX):]
            if plans.get(tag, {}).get("mode") != "update":
                err(f"{path.name}: no update plan for vintage {tag}")
            if df["speech_id"].is_duplicated().any():
                err(f"{path.name}: duplicate speech_id")
            continue
        period = parse_key(path.stem)
        if df.height and ((df["date"].min() < period.first) or (df["date"].max() > period.last)):
            err(f"{path.name}: speech dates outside {period.first}..{period.last}")
        if (df["change"] != "base").any():
            err(f"{path.name}: base partition holds non-base rows")
        base_ids += df["speech_id"].to_list()
    if len(base_ids) != len(set(base_ids)):
        err("duplicate speech_id across base partitions")
    for raw in artifact.path.glob(f"{RAW_PREFIX}*.zip"):
        digest = hashlib.sha256(raw.read_bytes()).hexdigest()
        if not raw.stem.endswith(digest[:12]):
            err(f"{raw.name}: content does not match its name")
    return findings


__all__ = ["KIND", "PIPELINE_VERSION", "SpeechesJob", "file_sources", "read_speeches",
           "read_state", "verify_artifact"]
