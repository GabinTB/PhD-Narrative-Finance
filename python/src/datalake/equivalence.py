"""Equivalence checks: may an artifact's files stand for new inputs or new code?

``DatalakeIndex.relink`` registers an existing artifact's files under new inputs
(a re-ingested, restructured or moved parent) or a new pipeline version, without
recomputing them. That is only sound when what the artifact was computed from is
unchanged; these checks establish it and their results are recorded on the new
artifact (``meta.relink["checks"]``).

A check is a callable ``check(ctx: RelinkContext) -> CheckResult`` with a ``name``.
Built-ins (generic: any kind, any data source):

  FilesIdentical(parent)              the new parent has the old parent's files
                                      (names and recorded digests)
  ProjectionEqual(columns, parent)    per partition (file stem) or over the whole
                                      artifact, the old and new parent hold the same
                                      multiset of rows projected on ``columns``
                                      (columns may be renamed: {old name: new name})
  KeysCover(key, parent)              per partition, the artifact's key set equals the
                                      new parent's key set (no orphan, no gap)
  RerunSample(units)                  the kind's Job, with the new code and the new
                                      inputs, recomputes ``units`` into a scratch
                                      directory; outputs equal the artifact's files

Row digests are order independent: (number of rows, wrapping sum of a 64-bit row
hash), computed per file with bounded memory (only the projected columns of one
file are read at a time).

Kind-specific default checks register under the entry-point group
``narrative_finance.equivalence`` (``{kind} = "pkg.mod:checks"``), a callable
``checks(ctx) -> list[Check]`` run in addition to the explicit ones.
"""
from __future__ import annotations

import logging
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from importlib.metadata import entry_points
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np

from datalake.index import DatalakeError

if TYPE_CHECKING:
    from datalake.artifact import Artifact
    from datalake.index import DatalakeIndex

log = logging.getLogger(__name__)

EQUIVALENCE_ENTRYPOINT_GROUP = "narrative_finance.equivalence"
PARQUET = "*.parquet"
_SEEDS = dict(seed=0, seed_1=1, seed_2=2, seed_3=3)


@dataclass
class CheckResult:
    name: str
    passed: bool
    details: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RelinkContext:
    """What a check sees.

    ``old`` is the registered artifact being relinked; ``new`` the provisional new
    artifact (its meta and directory; not registered while checks run; in a dry run
    its path is the old directory, whose files are the same); ``replace`` maps old
    parent ids to new parent ids."""

    index: DatalakeIndex
    old: Artifact
    new: Artifact
    replace: dict[str, str] = field(default_factory=dict)

    def parents(self, parent: str | None) -> tuple[Artifact, Artifact]:
        """(old parent, new parent) for ``parent`` (an old parent id; may be omitted
        when exactly one parent is replaced)."""
        if parent is None:
            if len(self.replace) != 1:
                raise DatalakeError("several parents are replaced: name the one to check "
                                    f"(one of {sorted(self.replace)})")
            parent = next(iter(self.replace))
        if parent not in self.replace:
            raise DatalakeError(f"{parent} is not a replaced parent ({sorted(self.replace)})")
        return self.index.get(parent), self.index.get(self.replace[parent])


Check = Callable[[RelinkContext], CheckResult]


# ---------------------------------------------------------------------------
# Digests
# ---------------------------------------------------------------------------

def _rows_digest(frame: Any) -> tuple[int, int]:
    """(rows, wrapping uint64 sum of row hashes): equal for equal row multisets."""
    if frame.height == 0:
        return 0, 0
    hashes = frame.hash_rows(**_SEEDS).to_numpy().astype(np.uint64, copy=False)
    with np.errstate(over="ignore"):
        total = int(np.sum(hashes, dtype=np.uint64))
    return frame.height, total


def _projected(path: Path, columns: Sequence[str], rename: dict[str, str] | None,
               schema: dict[str, Any] | None, unique: bool = False) -> Any:
    """``columns`` of one parquet file (read under ``rename[c]`` names, returned under
    ``columns`` names and cast to ``schema`` when given)."""
    import polars as pl

    rename = rename or {}
    src = [rename.get(c, c) for c in columns]
    frame = pl.read_parquet(path, columns=src).rename(dict(zip(src, columns)))
    if schema is not None:
        frame = frame.cast({c: t for c, t in schema.items()
                            if c in frame.columns and frame.schema[c] != t})
    return frame.unique() if unique else frame


DEFAULT_SAMPLE = 6        # partitions a content check reads by default (0: all of them)


def sample_stems(stems: Sequence[str], n: int) -> list[str]:
    """``n`` stems spread over the sorted ``stems`` (first and last included,
    deterministic); every stem when ``n`` is 0 or not smaller than their number."""
    keys = sorted(stems)
    if n <= 0 or n >= len(keys):
        return keys
    if n == 1:
        return [keys[0]]
    idx = sorted({round(i * (len(keys) - 1) / (n - 1)) for i in range(n)})
    return [keys[i] for i in idx]


def _sampled(a: dict[str, Path], b: dict[str, Path], n: int) -> tuple[dict[str, Path],
                                                                      dict[str, Path], str]:
    """Both sides restricted to ``n`` sampled stems of the union (a stem missing on one
    side still fails the comparison); plus a note naming the sample."""
    keys = sample_stems(set(a) | set(b), n)
    if len(keys) == len(set(a) | set(b)):
        return a, b, ""
    pick = set(keys)
    note = f" [sample {len(keys)}/{len(set(a) | set(b))} partitions: {', '.join(keys)}]"
    return ({k: v for k, v in a.items() if k in pick},
            {k: v for k, v in b.items() if k in pick}, note)


def _files(art_path: Path, pattern: str) -> dict[str, Path]:
    return {p.stem: p for p in sorted(art_path.glob(pattern)) if p.is_file()}


def _compare_digests(a: dict[str, tuple[int, int]], b: dict[str, tuple[int, int]],
                     by: str, what: str) -> tuple[bool, str]:
    if by == "all":
        ta = (sum(n for n, _ in a.values()), sum(h for _, h in a.values()) % 2**64)
        tb = (sum(n for n, _ in b.values()), sum(h for _, h in b.values()) % 2**64)
        if ta == tb:
            return True, f"{ta[0]:,} {what}, {len(a)} vs {len(b)} file(s), digests equal"
        return False, f"{what} differ: {ta[0]:,} vs {tb[0]:,} rows (digest mismatch)"
    only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
    bad = [k for k in sorted(set(a) & set(b)) if a[k] != b[k]]
    if not only_a and not only_b and not bad:
        return True, f"{len(a)} partition(s), {sum(n for n, _ in a.values()):,} {what}, all equal"
    parts = []
    if only_a:
        parts.append(f"only in old: {only_a[:5]}{' ...' if len(only_a) > 5 else ''}")
    if only_b:
        parts.append(f"only in new: {only_b[:5]}{' ...' if len(only_b) > 5 else ''}")
    if bad:
        parts.append(f"{len(bad)} partition(s) differ, e.g. "
                     + ", ".join(f"{k} ({a[k][0]:,} vs {b[k][0]:,} rows)" for k in bad[:3]))
    return False, "; ".join(parts)


# ---------------------------------------------------------------------------
# Built-in checks
# ---------------------------------------------------------------------------

@dataclass
class FilesIdentical:
    """The new parent carries the old parent's files: same names, same digests."""

    parent: str | None = None

    @property
    def name(self) -> str:
        return "files" + (f"@{self.parent}" if self.parent else "")

    def __call__(self, ctx: RelinkContext) -> CheckResult:
        old_p, new_p = ctx.parents(self.parent)
        a, b = old_p.file_hashes, new_p.file_hashes
        if not a:
            return CheckResult(self.name, False, f"{old_p.artifact_id} records no file hashes")
        if a == b:
            return CheckResult(self.name, True, f"{len(a)} file(s), identical digests")
        diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
        return CheckResult(self.name, False, f"{len(diff)} file(s) differ, e.g. {diff[:5]}")


TIMESTAMP_TEXT_FORMAT = "%Y-%m-%d %H:%M:%S%.f"


def _as_date(col: Any, dtype: Any, tz: str) -> Any:
    """The calendar day in ``tz`` of a timestamp column: a datetime (naive = UTC) is
    converted to ``tz``; a text timestamp is read as UTC (``TIMESTAMP_TEXT_FORMAT``,
    or a bare date) and converted the same way; a date stays as is."""
    import polars as pl

    if dtype == pl.Date:
        return col
    if dtype == pl.String:
        col = col.str.to_datetime(TIMESTAMP_TEXT_FORMAT, time_unit="us", time_zone="UTC",
                                  strict=False).fill_null(
            col.str.to_datetime("%Y-%m-%d", time_unit="us", time_zone="UTC", strict=False))
    elif isinstance(dtype, pl.Datetime) and dtype.time_zone is None:
        col = col.dt.replace_time_zone("UTC")
    return col.dt.convert_time_zone(tz).dt.date()


@dataclass
class ProjectionEqual:
    """Old and new parent hold the same rows projected on ``columns``.

    ``rename`` maps an old column name to its name in the new parent; the new
    columns are cast to the old dtypes (a failing cast fails the check). ``by``:
    "stem" compares partition by partition (same file stems), "all" the whole
    artifact (layouts may differ).

    ``as_date`` columns are compared as calendar days in ``tz`` on both sides (text or
    datetime timestamps, see ``_as_date``; an unparseable value fails the check).
    ``old_nulls`` strings read as null on the OLD side only (a legacy writer's null
    text, e.g. "<NA>"); how many values that touched is reported."""

    columns: list[str]
    parent: str | None = None
    rename: dict[str, str] = field(default_factory=dict)
    by: str = "stem"
    pattern: str = PARQUET
    as_date: list[str] = field(default_factory=list)
    tz: str = "UTC"
    old_nulls: list[str] = field(default_factory=list)
    sample: int = DEFAULT_SAMPLE

    @property
    def name(self) -> str:
        cols = ",".join(f"{c}:{self.rename[c]}" if c in self.rename else c for c in self.columns)
        extra = "".join([f" as_date={'+'.join(self.as_date)}@{self.tz}" if self.as_date else "",
                         f" old_nulls={'+'.join(self.old_nulls)}" if self.old_nulls else ""])
        return f"projection{'@' + self.parent if self.parent else ''}[{cols}] by {self.by}{extra}"

    def _side(self, path: Path, *, old: bool, schema: dict[str, Any] | None,
              counter: list[int]) -> Any:
        import polars as pl

        rename = None if old else self.rename
        cast_to = None if schema is None else {c: t for c, t in schema.items()
                                               if c not in self.as_date}
        frame = _projected(path, self.columns, rename, cast_to)
        if old and self.old_nulls:
            text = [c for c in self.columns if frame.schema[c] == pl.String]
            hit = frame.select([pl.col(c).is_in(self.old_nulls).sum() for c in text])
            counter[0] += int(sum(hit.row(0))) if text else 0
            frame = frame.with_columns([pl.when(pl.col(c).is_in(self.old_nulls)).then(None)
                                        .otherwise(pl.col(c)).alias(c) for c in text])
        if self.as_date:
            before = frame.select([pl.col(c).null_count() for c in self.as_date]).row(0)
            frame = frame.with_columns([_as_date(pl.col(c), frame.schema[c], self.tz).alias(c)
                                        for c in self.as_date])
            after = frame.select([pl.col(c).null_count() for c in self.as_date]).row(0)
            if after != before:
                raise ValueError(f"{path.name}: timestamps that do not parse in "
                                 f"{self.as_date}")
        return frame

    def __call__(self, ctx: RelinkContext) -> CheckResult:
        import polars as pl

        old_p, new_p = ctx.parents(self.parent)
        old_files, new_files = _files(old_p.path, self.pattern), _files(new_p.path, self.pattern)
        if not old_files or not new_files:
            return CheckResult(self.name, False, "no partition file on one side")
        note = ""
        if self.by == "stem":
            old_files, new_files, note = _sampled(old_files, new_files, self.sample)
        first = next(iter(old_files.values()))
        schema = dict(pl.read_parquet_schema(first))
        missing = [c for c in self.columns if c not in schema]
        if missing:
            return CheckResult(self.name, False, f"columns {missing} not in {old_p.artifact_id}")
        schema = {c: schema[c] for c in self.columns}
        nulled = [0]
        try:
            a = {k: _rows_digest(self._side(p, old=True, schema=None, counter=nulled))
                 for k, p in old_files.items()}
            b = {k: _rows_digest(self._side(p, old=False, schema=schema, counter=nulled))
                 for k, p in new_files.items()}
        except Exception as exc:  # noqa: BLE001 - a read / cast failure is a failed check
            return CheckResult(self.name, False, f"{type(exc).__name__}: {exc}")
        ok, details = _compare_digests(a, b, self.by, "rows")
        if self.old_nulls:
            details += (f"; {nulled[0]:,} old value(s) in {self.old_nulls} read as null")
        return CheckResult(self.name, ok, details + note)


@dataclass
class KeysCover:
    """The artifact's ``key`` set equals the new parent's ``parent_key`` set, per
    partition (``by="stem"``) or overall (``by="all"``)."""

    key: str
    parent: str | None = None
    parent_key: str | None = None
    by: str = "stem"
    pattern: str = PARQUET
    sample: int = DEFAULT_SAMPLE

    @property
    def name(self) -> str:
        pk = f":{self.parent_key}" if self.parent_key else ""
        return f"keys{'@' + self.parent if self.parent else ''}[{self.key}{pk}] by {self.by}"

    def __call__(self, ctx: RelinkContext) -> CheckResult:
        import polars as pl

        _, new_p = ctx.parents(self.parent)
        child_files = _files(ctx.new.path, self.pattern)
        parent_files = _files(new_p.path, self.pattern)
        note = ""
        if self.by == "stem":
            child_files, parent_files, note = _sampled(child_files, parent_files, self.sample)
        if not child_files or not parent_files:
            return CheckResult(self.name, False, "no partition file on one side")
        try:
            schema = {self.key: pl.read_parquet_schema(next(iter(child_files.values())))[self.key]}
            rename = {self.key: self.parent_key} if self.parent_key else None
            a = {k: _rows_digest(_projected(p, [self.key], None, None, unique=True))
                 for k, p in child_files.items()}
            b = {k: _rows_digest(_projected(p, [self.key], rename, schema, unique=True))
                 for k, p in parent_files.items()}
        except Exception as exc:  # noqa: BLE001
            return CheckResult(self.name, False, f"{type(exc).__name__}: {exc}")
        ok, details = _compare_digests(a, b, self.by, "keys")
        return CheckResult(self.name, ok, details + note)


@dataclass
class RerunSample:
    """Recompute ``units`` with the kind's Job (current code, new inputs) into a
    scratch directory and compare with the artifact's files: parquet as frames
    (exact, or within ``rtol`` / ``atol``), other files byte for byte.

    Only ``run_unit`` is called: a job that needs its ``session`` (e.g. one writing
    sibling artifacts) cannot be rerun here and fails the check with that reason.
    Bounded by the owner's choice of units (e.g. one month)."""

    units: list[str]
    rtol: float = 0.0
    atol: float = 0.0

    @property
    def name(self) -> str:
        return f"rerun[{','.join(self.units)}]"

    def __call__(self, ctx: RelinkContext) -> CheckResult:
        import polars as pl
        from polars.testing import assert_frame_equal

        from datalake.jobs import JobContext, _UnitAdapter, job_class

        try:
            job = job_class(ctx.new.kind).from_artifact(ctx.new, ctx.index)
            for backend in job.backends():
                backend.check_unchanged()
            known = {u.key: u for u in job.units()}
            missing = [u for u in self.units if u not in known]
            if missing:
                return CheckResult(self.name, False, f"unknown unit(s) {missing}")
            compared = 0
            with tempfile.TemporaryDirectory(prefix="relink-rerun-") as tmp:
                scratch = Path(tmp)
                run = SimpleNamespace(out_dir=scratch, artifact_id=f"{ctx.new.artifact_id}__RERUN",
                                      note=lambda text: None,
                                      record=SimpleNamespace(pipeline_commit=None))
                adapter = _UnitAdapter(logging.getLogger(f"relink.{ctx.new.kind}"),
                                       {"kind": ctx.new.kind, "artifact": run.artifact_id,
                                        "host": "-", "unit": "-"})
                jctx = JobContext(run, ctx.index, adapter)
                for key in self.units:
                    job.run_unit(known[key], jctx)
                for out in sorted(p for p in scratch.rglob("*") if p.is_file()):
                    rel = out.relative_to(scratch)
                    ref = ctx.old.path / rel
                    if not ref.is_file():
                        return CheckResult(self.name, False, f"rerun wrote {rel}, absent from "
                                           f"{ctx.old.artifact_id}")
                    if out.suffix == ".parquet":
                        assert_frame_equal(pl.read_parquet(out), pl.read_parquet(ref),
                                           check_exact=self.rtol == 0 and self.atol == 0,
                                           rel_tol=self.rtol, abs_tol=self.atol)
                    elif out.read_bytes() != ref.read_bytes():
                        return CheckResult(self.name, False, f"{rel} differs")
                    compared += 1
        except AssertionError as exc:
            return CheckResult(self.name, False, f"outputs differ: {str(exc)[:300]}")
        except Exception as exc:  # noqa: BLE001
            return CheckResult(self.name, False, f"rerun failed ({type(exc).__name__}: {exc})")
        if compared == 0:
            return CheckResult(self.name, False, "the rerun wrote no file")
        return CheckResult(self.name, True, f"{compared} file(s) recomputed, equal")


# ---------------------------------------------------------------------------
# Defaults per kind, and the CLI spec
# ---------------------------------------------------------------------------

def default_checks(ctx: RelinkContext) -> list[Check]:
    """Checks contributed by the ``narrative_finance.equivalence`` entry point of the
    artifact's kind (none when the kind registers nothing)."""
    for ep in entry_points(group=EQUIVALENCE_ENTRYPOINT_GROUP):
        if ep.name == ctx.old.kind:
            return list(ep.load()(ctx))
    return []


def parse_check(spec: str) -> Check:
    """A check from its CLI spec::

        files[@PARENT]
        projection[@PARENT]=COL,COL,OLDCOL:NEWCOL[,by=all][,as_date=C+C][,tz=Europe/Paris]
                                                 [,old_nulls=<NA>+None][,sample=N]
        keys[@PARENT]=KEY[:PARENT_KEY][,by=all][,sample=N]
        rerun=UNIT,UNIT[,rtol=1e-6][,atol=0]

    ``@PARENT`` (an old parent id) may be omitted when one parent is replaced.
    ``sample``: partitions read by a ``by=stem`` check, spread over the range (first and
    last included; default ``DEFAULT_SAMPLE``; 0 = every partition, an explicit choice:
    it costs about what recomputing costs). The sampled stems are named in the result."""
    head, _, body = spec.partition("=")
    name, _, parent = head.partition("@")
    tokens = [t for t in body.split(",") if t] if body else []
    opts = dict(t.split("=", 1) for t in tokens if "=" in t)
    args = [t for t in tokens if "=" not in t]
    parent_id = parent or None
    if name == "files" and not args:
        return FilesIdentical(parent_id)
    if name == "projection" and args:
        columns, rename = [], {}
        for a in args:
            old, _, new = a.partition(":")
            columns.append(old)
            if new:
                rename[old] = new
        plus = lambda key: [v for v in opts.get(key, "").split("+") if v]  # noqa: E731
        return ProjectionEqual(columns, parent_id, rename, by=opts.get("by", "stem"),
                               as_date=plus("as_date"), tz=opts.get("tz", "UTC"),
                               old_nulls=plus("old_nulls"),
                               sample=int(opts.get("sample", DEFAULT_SAMPLE)))
    if name == "keys" and len(args) == 1:
        key, _, parent_key = args[0].partition(":")
        return KeysCover(key, parent_id, parent_key or None, by=opts.get("by", "stem"),
                         sample=int(opts.get("sample", DEFAULT_SAMPLE)))
    if name == "rerun" and args and not parent:
        return RerunSample(args, rtol=float(opts.get("rtol", 0.0)),
                           atol=float(opts.get("atol", 0.0)))
    raise DatalakeError(f"bad check spec {spec!r} (files[@P] | projection[@P]=COLS | "
                        "keys[@P]=KEY[:PKEY] | rerun=UNITS)")


__all__ = ["Check", "CheckResult", "RelinkContext", "FilesIdentical", "ProjectionEqual",
           "KeysCover", "RerunSample", "default_checks", "parse_check",
           "EQUIVALENCE_ENTRYPOINT_GROUP"]
