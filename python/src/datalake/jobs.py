"""One lifecycle for every long pipeline: start, status, check, pause, resume, end.

A ``Job`` declares WHAT it computes; ``JobRunner`` owns HOW it runs:

    kind / pipeline / pipeline_version     the datalake artifact it produces
    group                                  optional folder group of NEW artifacts
                                           ({layer}/{group}/{kind}/{id}); siblings
                                           opened in ``session`` follow it
    layer                                  optional layer of NEW artifacts (default: the
                                           index's layer for the kind, i.e. derived)
    params()                             the artifact's hyperparams (its identity)
    sources() / model_card() / backends()  lineage, model provenance, live model checks
    units()                                ordered work units (the partitions of its range)
    is_done(unit, out_dir)                 skip test (resume, idempotence)
    run_unit(unit, ctx)                    the work
    finalize(ctx)                          optional last step (e.g. a series from checkpoints)
    from_artifact(artifact, index)         rebuild the job from the artifact alone (resume)
    for_update(artifact, index)            optional: the job adding new data to a complete
                                           artifact (update)

Runner guarantees, identical for every job:

  * start -> ``DatalakeIndex.run``; resume -> ``run(resume=...)`` after the
    job is rebuilt from the artifact, the code's pipeline_version is checked
    against the artifact's (same major.minor, else refused), and the models
    are checked against the recorded identities (``backend.check_unchanged``);
  * update -> ``run(extend=...)`` on a COMPLETE artifact, with the same version
    and model checks: a new RunRecord, new files only (existing files are never
    rewritten). A crashed update leaves the artifact partial; ``resume``
    finishes it (``from_artifact`` recognises the unfinished update);
  * the unit loop skips done units, re-checks the models before every unit,
    runs it, then updates ``job.json`` (status, units done / total, current
    unit, rate, ETA, host, pid, heartbeat, last error, version, commit);
  * a ``job.lock`` (created O_EXCL, heartbeat-refreshed by a thread) blocks a
    second process on the same artifact; a lock whose heartbeat is older than
    ``stale_after_s`` is taken over (logged). The lake lives on a network
    mount: this is a same-host lock plus a cross-host heartbeat guard, not a
    distributed lock;
  * pause is graceful: SIGTERM / SIGINT, or a ``PAUSE`` file in the artifact
    directory (works from another machine), stop the job at the next unit
    boundary; the artifact stays partial and resumable, status ``paused``;
  * logging defaults on: ``job.log`` in the artifact directory (plain or JSON
    lines) plus whatever console handler the caller configured;
  * a non-TEMP run refuses a dirty git tree unless ``allow_dirty`` (recorded
    in the run notes).

``job.json`` / ``job.lock`` / ``job.log`` / ``PAUSE`` are not hashed and not
part of the outputs.
"""
from __future__ import annotations

import json
import logging
import os
import re
import signal
import socket
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from datalake.artifact import utc_now_iso
from datalake.index import DatalakeError, DatalakeIndex
from datalake.meta import JOB_CONTROL_FILES, git_commit

if TYPE_CHECKING:
    from datalake.artifact import Artifact, ModelCard

log = logging.getLogger(__name__)

STATE_FILE, LOCK_FILE, LOG_FILE, PAUSE_FILE = JOB_CONTROL_FILES     # never hashed
JOB_ENTRYPOINT_GROUP = "narrative_finance.jobs"
TEMP_SUFFIX = "__TEMP"


class JobError(DatalakeError):
    """A job cannot start, resume or continue (version, lock, dirty tree, model)."""


class JobLocked(JobError):
    """Another live process holds the artifact."""


class JobPaused(Exception):
    """Raised inside the unit loop to stop at a unit boundary (not an error)."""


@dataclass(frozen=True)
class Unit:
    """One work unit: ``key`` is stable across runs (usually a period key)."""

    key: str
    label: str = ""


class JobContext:
    """What a unit sees: the output directory, a unit-tagged logger, the run handle."""

    def __init__(self, run: Any, index: DatalakeIndex, logger: logging.LoggerAdapter) -> None:
        self.run, self.index, self.log = run, index, logger
        self.out_dir: Path = run.out_dir
        self.artifact_id: str = run.artifact_id

    def note(self, text: str) -> None:
        self.run.note(text)


class Job(ABC):
    """A resumable pipeline producing one datalake artifact of ``kind``."""

    kind: ClassVar[str]
    pipeline: ClassVar[str] = "PhD-Narrative-Finance"
    pipeline_repo: ClassVar[str | None] = "https://github.com/GabinTB/PhD-Narrative-Finance"
    pipeline_version: ClassVar[str]
    hash_pattern: ClassVar[str] = "*.parquet"
    group: ClassVar[str | None] = None
    layer: ClassVar[str | None] = None       # None: the index default for the kind
    temp: bool = False

    @property
    def version(self) -> str:
        """The version recorded on the artifact (TEMP runs carry the suffix)."""
        return self.pipeline_version + (TEMP_SUFFIX if self.temp else "")

    @property
    def verifier(self) -> str | None:
        return self.kind

    @abstractmethod
    def params(self) -> dict[str, Any]:
        """The artifact's hyperparams (identity; resume requires them identical)."""

    def sources(self) -> list[Any]:
        return []

    def model_card(self) -> ModelCard | None:
        return None

    def backends(self) -> list[Any]:
        """Objects exposing ``check_unchanged()`` and ``identity()`` (remote models)."""
        return []

    def notes(self) -> str:
        return ""

    @abstractmethod
    def units(self) -> list[Unit]:
        """Ordered work units."""

    @abstractmethod
    def is_done(self, unit: Unit, out_dir: Path) -> bool:
        """True when ``unit`` is already written in ``out_dir``."""

    @abstractmethod
    def run_unit(self, unit: Unit, ctx: JobContext) -> None:
        """Compute and write one unit (atomically).

        It may also write later units that share its inputs (e.g. every day of
        the raw month it reads); the loop then counts and skips them."""

    def finalize(self, ctx: JobContext) -> None:
        """Optional last step after every unit is done."""

    def session(self, ctx: JobContext) -> AbstractContextManager[Any]:
        """Context held around the unit loop and ``finalize`` (default: nothing).

        A job writing several artifacts in one pass opens the siblings here; an
        exception (crash or pause) leaves them partial exactly like the main one."""
        return nullcontext()

    @classmethod
    @abstractmethod
    def from_artifact(cls, artifact: Artifact, index: DatalakeIndex, **kwargs: Any) -> Job:
        """Rebuild the job from the artifact alone (hyperparams, card, lineage)."""

    @classmethod
    def for_update(cls, artifact: Artifact, index: DatalakeIndex, **kwargs: Any) -> Job:
        """The job adding new data to the complete ``artifact`` (default: unsupported)."""
        raise JobError(f"{cls.kind} does not support updates")

    @classmethod
    def add_cli_args(cls, parser: Any) -> None:
        """Arguments of ``jobs start <kind>`` (none by default)."""

    @classmethod
    def from_args(cls, args: Any, index: DatalakeIndex) -> Job:
        """Build a fresh job from ``jobs start <kind>`` arguments."""
        raise JobError(f"{cls.kind} cannot be started from the jobs CLI")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type[Job]] = {}


def register_job(cls: type[Job]) -> type[Job]:
    """Register a Job class for its kind (decorator; entry points do the same)."""
    _REGISTRY[cls.kind] = cls
    return cls


def job_class(kind: str) -> type[Job]:
    if kind not in _REGISTRY:
        from importlib.metadata import entry_points

        for ep in entry_points(group=JOB_ENTRYPOINT_GROUP):
            if ep.name == kind:
                _REGISTRY[kind] = ep.load()
                break
    if kind not in _REGISTRY:
        raise JobError(f"no Job registered for kind {kind!r}")
    return _REGISTRY[kind]


def registered_jobs() -> dict[str, type[Job]]:
    from importlib.metadata import entry_points

    for ep in entry_points(group=JOB_ENTRYPOINT_GROUP):
        if ep.name not in _REGISTRY:
            try:
                _REGISTRY[ep.name] = ep.load()
            except Exception as exc:  # noqa: BLE001 - a broken plugin must not hide the rest
                log.warning("job entry point %s failed to load: %s", ep.name, exc)
    return dict(_REGISTRY)


# ---------------------------------------------------------------------------
# Versions
# ---------------------------------------------------------------------------

_SEMVER = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)")


def major_minor(version: str) -> tuple[int, int]:
    m = _SEMVER.match(version)
    if not m:
        raise JobError(f"not a semantic version: {version!r}")
    return int(m.group(1)), int(m.group(2))


def check_resume_version(artifact_version: str, code_version: str) -> None:
    """Resume only with the same major.minor (a patch never changes outputs)."""
    if major_minor(artifact_version) != major_minor(code_version):
        raise JobError(f"artifact was produced by {artifact_version}, this code is "
                       f"{code_version}: a different major.minor cannot resume it")


# ---------------------------------------------------------------------------
# State, lock, logging
# ---------------------------------------------------------------------------

@dataclass
class JobState:
    kind: str
    artifact_id: str
    status: str = "running"            # running | paused | failed | complete
    units_total: int = 0
    units_done: int = 0
    current_unit: str | None = None
    started_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    heartbeat_at: str = field(default_factory=utc_now_iso)
    host: str = field(default_factory=socket.gethostname)
    pid: int = field(default_factory=os.getpid)
    pipeline_version: str = ""
    commit: str | None = None
    rate_units_per_hour: float | None = None
    eta: str | None = None
    last_error: str | None = None
    backends: list[dict[str, Any]] = field(default_factory=list)
    unit_seconds: dict[str, float] = field(default_factory=dict)

    def write(self, out_dir: Path) -> None:
        """Written by the job's own thread only (the heartbeat lives in ``job.lock``)."""
        self.updated_at = self.heartbeat_at = utc_now_iso()
        _write_atomic(Path(out_dir) / STATE_FILE,
                      json.dumps(asdict(self), indent=2, sort_keys=True, default=str))

    @classmethod
    def read(cls, out_dir: Path) -> JobState | None:
        path = Path(out_dir) / STATE_FILE
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


def _age_s(iso: str) -> float:
    return (datetime.now(timezone.utc) - datetime.fromisoformat(iso)).total_seconds()


def _write_atomic(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` via a writer-unique temp file and an atomic rename."""
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        tmp.write_text(text)
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


class JobLock:
    """``job.lock`` holder with a heartbeat thread (see module docstring for limits)."""

    def __init__(self, out_dir: Path, *, heartbeat_s: float, stale_after_s: float) -> None:
        self.path = Path(out_dir) / LOCK_FILE
        self.heartbeat_s, self.stale_after_s = heartbeat_s, stale_after_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _payload(self) -> str:
        return json.dumps({"host": socket.gethostname(), "pid": os.getpid(),
                           "heartbeat_at": utc_now_iso()})

    def acquire(self) -> None:
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            try:
                held = json.loads(self.path.read_text())
                age = _age_s(held["heartbeat_at"])
            except (ValueError, KeyError, OSError):
                held, age = {}, float("inf")
            if age < self.stale_after_s:
                raise JobLocked(f"{self.path.parent.name} is held by {held.get('host')} "
                                f"pid {held.get('pid')} (heartbeat {age:.0f}s ago)") from None
            log.warning("taking over a stale lock on %s (holder %s pid %s, heartbeat %.0fs ago)",
                        self.path.parent.name, held.get("host"), held.get("pid"), age)
            _write_atomic(self.path, self._payload())
        else:
            with os.fdopen(fd, "w") as fh:
                fh.write(self._payload())
        self._thread = threading.Thread(target=self._beat, daemon=True, name="job-heartbeat")
        self._thread.start()

    def _beat(self) -> None:
        while not self._stop.wait(self.heartbeat_s):
            try:
                _write_atomic(self.path, self._payload())    # a reader never sees half a lock
            except OSError as exc:            # the mount hiccuped; the next beat retries
                log.warning("heartbeat write failed: %s", exc)

    def release(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self.path.unlink(missing_ok=True)


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps({"time": self.formatTime(record), "level": record.levelname,
                           "logger": record.name, "message": record.getMessage(),
                           **{k: getattr(record, k) for k in ("kind", "artifact", "unit", "host")
                              if hasattr(record, k)}}, default=str)


class _UnitAdapter(logging.LoggerAdapter):
    def process(self, msg: str, kwargs: Any) -> tuple[str, Any]:
        kwargs.setdefault("extra", {}).update(self.extra)
        return msg, kwargs


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

@dataclass
class JobStatus:
    artifact_id: str
    kind: str
    status: str                 # running | stale | paused | failed | complete | partial
    units_done: int | None
    units_total: int | None
    current_unit: str | None
    heartbeat_age_s: float | None
    host: str | None
    pid: int | None
    last_error: str | None
    eta: str | None


class JobRunner:
    """Runs Jobs against one datalake index (see module docstring for guarantees).

    Args:
        index:         the datalake index artifacts are registered in.
        log_file:      write ``job.log`` in the artifact directory (default True).
        log_level:     level of the job.log handler.
        log_json:      JSON lines instead of plain text in job.log.
        heartbeat_s / stale_after_s: lock heartbeat period and staleness threshold.
        allow_dirty:   let a non-TEMP run start from a dirty git tree (recorded).
        repo_dir:      git work tree whose commit is recorded.
        handle_signals: install SIGTERM / SIGINT handlers requesting a pause.
    """

    def __init__(self, index: DatalakeIndex, *, log_file: bool = True, log_level: str = "INFO",
                 log_json: bool = False, heartbeat_s: float = 30.0, stale_after_s: float = 600.0,
                 allow_dirty: bool = False, repo_dir: Path | None = None,
                 handle_signals: bool = True) -> None:
        self.index = index
        self.log_file, self.log_level, self.log_json = log_file, log_level, log_json
        self.heartbeat_s, self.stale_after_s = heartbeat_s, stale_after_s
        self.allow_dirty, self.repo_dir = allow_dirty, repo_dir
        self.handle_signals = handle_signals
        self._pause_requested = threading.Event()

    # -- public API ---------------------------------------------------------

    def start(self, job: Job) -> Artifact:
        """Run ``job`` into a new artifact (to completion, or until paused)."""
        return self._execute(job, resume=None, notes=self._dirty_guard(job, job.notes()))

    def resume(self, artifact_id: str, **job_kwargs: Any) -> Artifact:
        """Rebuild the job from the partial artifact and finish it in place."""
        art = self._partial(artifact_id)
        cls = job_class(art.kind)
        check_resume_version(art.meta.pipeline_version, cls.pipeline_version)
        return self.resume_job(artifact_id, cls.from_artifact(art, self.index, **job_kwargs))

    def resume_job(self, artifact_id: str, job: Job) -> Artifact:
        """Finish a partial artifact with an already built ``job`` (library callers).

        The artifact's recorded hyperparams stay its identity (a legacy layout
        such as start_year / end_year is not rewritten); the version and model
        checks are the same as ``resume``.
        """
        art = self._partial(artifact_id)
        if art.kind != job.kind:
            raise JobError(f"{artifact_id} is a {art.kind}, not a {job.kind}")
        check_resume_version(art.meta.pipeline_version, job.pipeline_version)
        for backend in job.backends():
            backend.check_unchanged()
        return self._execute(job, resume=artifact_id, notes=self._dirty_guard(job, ""),
                             hyperparams=art.meta.hyperparams)

    def update(self, artifact_id: str, **job_kwargs: Any) -> Artifact:
        """Add new data to a complete artifact (``Job.for_update``), as a new execution."""
        art = self._complete(artifact_id)
        cls = job_class(art.kind)
        check_resume_version(art.meta.pipeline_version, cls.pipeline_version)
        return self.update_job(artifact_id, cls.for_update(art, self.index, **job_kwargs))

    def update_job(self, artifact_id: str, job: Job) -> Artifact:
        """``update`` with an already built ``job`` (library callers)."""
        art = self._complete(artifact_id)
        if art.kind != job.kind:
            raise JobError(f"{artifact_id} is a {art.kind}, not a {job.kind}")
        check_resume_version(art.meta.pipeline_version, job.pipeline_version)
        for backend in job.backends():
            backend.check_unchanged()
        return self._execute(job, resume=None, extend=artifact_id,
                             notes=self._dirty_guard(job, job.notes()),
                             hyperparams=art.meta.hyperparams)

    def _complete(self, artifact_id: str) -> Artifact:
        art = self.index.get(artifact_id)
        if art.partial:
            raise JobError(f"{artifact_id} is partial; finish it with resume before updating")
        return art

    def _partial(self, artifact_id: str) -> Artifact:
        art = self.index.get(artifact_id)
        if not art.partial:
            raise JobError(f"{artifact_id} is complete; nothing to resume")
        return art

    def _dirty_guard(self, job: Job, notes: str) -> str:
        commit = git_commit(self.repo_dir)
        if commit and commit.endswith("-dirty") and not job.temp:
            if not self.allow_dirty:
                raise JobError("refusing a non-TEMP run from a dirty git tree "
                               f"({commit}); commit first or pass allow_dirty")
            notes = f"{notes} [allow_dirty: run from {commit}]".strip()
        return notes

    def pause(self, artifact_id: str) -> None:
        """Ask the process running ``artifact_id`` to stop at the next unit boundary."""
        art = self.index.get(artifact_id)
        (art.path / PAUSE_FILE).write_text(utc_now_iso())

    def status(self, artifact_id: str) -> JobStatus:
        art = self.index.get(artifact_id)
        state = JobState.read(art.path)
        lock_age = None
        if (art.path / LOCK_FILE).exists():
            try:
                lock_age = _age_s(json.loads((art.path / LOCK_FILE).read_text())["heartbeat_at"])
            except (ValueError, KeyError, OSError):
                lock_age = float("inf")
        if not art.partial:
            status = "complete"
        elif lock_age is not None:
            status = "running" if lock_age < self.stale_after_s else "stale"
        elif state is not None and state.status in ("paused", "failed"):
            status = state.status
        else:
            status = "partial"                         # interrupted with no live holder
        return JobStatus(artifact_id, art.kind, status,
                         state.units_done if state else None,
                         state.units_total if state else None,
                         state.current_unit if state else None, lock_age,
                         state.host if state else None, state.pid if state else None,
                         state.last_error if state else None, state.eta if state else None)

    def check(self, artifact_id: str, **job_kwargs: Any) -> list[str]:
        """Problems that would stop a resume now: version, model identity, content."""
        from datalake.verify import discover_verifiers

        art = self.index.get(artifact_id)
        problems: list[str] = []
        try:
            cls = job_class(art.kind)
            check_resume_version(art.meta.pipeline_version, cls.pipeline_version)
            job = cls.from_artifact(art, self.index, **job_kwargs)
            for backend in job.backends():
                backend.check_unchanged()
        except Exception as exc:  # noqa: BLE001 - report, never raise
            problems.append(f"{type(exc).__name__}: {exc}")
        verifier = discover_verifiers().get(art.meta.verifier) if art.meta.verifier else None
        if verifier is not None:
            problems += [f"{f.severity.name}: {f.message}" for f in (verifier(art) or [])
                         if f.severity.name == "ERROR"]
        return problems

    def request_pause(self) -> None:
        self._pause_requested.set()

    # -- execution ----------------------------------------------------------

    def _execute(self, job: Job, *, resume: str | None, notes: str,
                 hyperparams: dict[str, Any] | None = None,
                 extend: str | None = None) -> Artifact:
        run_kwargs = dict(kind=job.kind, pipeline=job.pipeline,
                          pipeline_version=job.version, pipeline_repo=job.pipeline_repo,
                          hyperparams=hyperparams if hyperparams is not None else job.params(),
                          sources=job.sources(),
                          model_card=job.model_card(), verifier=job.verifier,
                          repo_dir=self.repo_dir, notes=notes, hash_pattern=job.hash_pattern,
                          layer=job.layer)
        artifact_id: str | None = resume or extend
        previous = self._install_signals()
        try:
            with self.index.default_group(job.group), \
                    self.index.run(resume=resume, extend=extend, **run_kwargs) as run:
                artifact_id = run.artifact_id
                self._loop(job, run, new_execution=extend is not None)
            return self.index.get(artifact_id)
        except JobPaused:
            log.info("job %s paused at a unit boundary", artifact_id)
            return self.index.get(artifact_id)
        finally:
            self._restore_signals(previous)

    def _loop(self, job: Job, run: Any, *, new_execution: bool = False) -> None:
        out_dir: Path = run.out_dir
        (out_dir / PAUSE_FILE).unlink(missing_ok=True)
        units = job.units()
        state = JobState(kind=job.kind, artifact_id=run.artifact_id,
                         units_total=len(units), pipeline_version=job.version,
                         commit=run.record.pipeline_commit,
                         backends=[b.identity() for b in job.backends()])
        previous = None if new_execution else JobState.read(out_dir)   # an update starts afresh
        if previous is not None:
            state.started_at, state.unit_seconds = previous.started_at, previous.unit_seconds

        root = logging.getLogger()
        previous_level = root.level
        handler = self._log_handler(out_dir)
        adapter = _UnitAdapter(logging.getLogger(f"job.{job.kind}"),
                               {"kind": job.kind, "artifact": run.artifact_id,
                                "host": state.host, "unit": "-"})
        lock = JobLock(out_dir, heartbeat_s=self.heartbeat_s, stale_after_s=self.stale_after_s)
        lock.acquire()
        try:
            ctx = JobContext(run, self.index, adapter)
            done_before = {u.key for u in units if job.is_done(u, out_dir)}
            state.units_done = len(done_before)
            state.write(out_dir)
            adapter.info("%s: %d/%d unit(s) already done", "resume" if previous else "start",
                         state.units_done, state.units_total)
            t_run, n_run = time.monotonic(), 0
            with job.session(ctx):           # e.g. sibling artifacts open across units
                for unit in units:
                    if job.is_done(unit, out_dir):
                        if unit.key not in done_before:    # written by an earlier unit's run
                            state.units_done += 1
                        continue
                    if self._pause_requested.is_set() or (out_dir / PAUSE_FILE).exists():
                        state.status, state.current_unit, state.eta = "paused", None, None
                        state.write(out_dir)
                        adapter.info("pause requested: stopping before %s", unit.key)
                        raise JobPaused(unit.key)
                    state.current_unit = unit.key
                    state.write(out_dir)
                    adapter.extra["unit"] = unit.key
                    for backend in job.backends():
                        backend.check_unchanged()
                    t0 = time.monotonic()
                    job.run_unit(unit, ctx)
                    elapsed = time.monotonic() - t0
                    n_run += 1
                    state.units_done += 1
                    state.unit_seconds[unit.key] = round(elapsed, 3)
                    rate = n_run / max(time.monotonic() - t_run, 1e-9) * 3600
                    left = state.units_total - state.units_done
                    state.rate_units_per_hour = round(rate, 3)
                    state.eta = (datetime.now(timezone.utc).timestamp() + left / rate * 3600
                                 if rate > 0 else None)
                    state.eta = (datetime.fromtimestamp(state.eta, timezone.utc).isoformat(
                        timespec="seconds") if state.eta else None)
                    state.write(out_dir)
                    adapter.info("unit %s done in %.1fs (%d/%d)", unit.key, elapsed,
                                 state.units_done, state.units_total)
                adapter.extra["unit"] = "-"
                state.current_unit = None
                job.finalize(ctx)
            state.status = "complete"
            state.write(out_dir)
            adapter.info("job complete: %d unit(s)", state.units_total)
        except JobPaused:
            raise
        except BaseException as exc:
            state.status, state.last_error = "failed", f"{type(exc).__name__}: {exc}"
            state.eta = None
            state.write(out_dir)
            adapter.error("job failed at %s: %s", state.current_unit, state.last_error)
            raise
        finally:
            lock.release()
            if handler is not None:
                root.removeHandler(handler)
                handler.close()
                root.setLevel(previous_level)

    def _log_handler(self, out_dir: Path) -> logging.Handler | None:
        if not self.log_file:
            return None
        handler = logging.FileHandler(out_dir / LOG_FILE)
        handler.setLevel(self.log_level)
        handler.setFormatter(_JsonFormatter() if self.log_json else logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s"))
        root = logging.getLogger()
        root.addHandler(handler)
        if root.level > logging.getLevelName(self.log_level):
            root.setLevel(self.log_level)
        return handler

    def _install_signals(self) -> dict[int, Any]:
        if not self.handle_signals or threading.current_thread() is not threading.main_thread():
            return {}
        previous = {}
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous[sig] = signal.signal(sig, lambda *_: self._pause_requested.set())
        return previous

    def _restore_signals(self, previous: dict[int, Any]) -> None:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def list_jobs(index: DatalakeIndex, runner: JobRunner | None = None,
              kinds: Iterable[str] | None = None) -> list[JobStatus]:
    """Status of every partial artifact (of the registered job kinds, or ``kinds``)."""
    runner = runner or JobRunner(index, handle_signals=False)
    kinds = list(kinds) if kinds is not None else list(registered_jobs())
    out = []
    for kind in kinds:
        for art in index.list(kind, include_partial=True):
            if art.partial:
                out.append(runner.status(art.artifact_id))
    return out
