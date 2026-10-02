"""DatalakeIndex: the registry and query layer over the datalake tree.

The datalake is a POSIX tree of immutable artifact directories:

    {root}/{layer}/{kind}/{artifact_id}/            (no group)
    {root}/{layer}/{group}/{kind}/{artifact_id}/    (group, e.g. the data source)
        meta.json
        README.md
        2000-01.parquet
        ...

The group is a folder only, never part of the id: `move()` re-files an artifact
under another group without touching its content, and `relink()` registers an
existing artifact's files (hard links) under new inputs (`datalake.relink`).

`index.db` at the root is a SQLite cache over the meta.json sidecars.  It
exists to make lookups and lineage queries fast; it holds no information that
is not also in the sidecars, so `reindex()` can rebuild it from scratch.

Typical use in a pipeline script:

    dl = DatalakeIndex("/mnt/storage/datalake")
    source = dl.latest("ravenpack_headlines")

    with dl.run(
        kind="headline_embeddings",
        pipeline="PhD-Narrative-Finance",
        pipeline_version="v0.1.0",
        hyperparams={"pooling": "cls"},
        sources=[source],
        model_card=RAVENBERT_V1_2,
        verifier="headline_embeddings",
    ) as run:
        embed_corpus(input_dir=source.path, out_dir=run.out_dir)

On clean exit the run's RunRecord is marked complete, outputs are hashed, both
sidecars written, and lineage edges registered.  On exception the record is
left partial so the crash is visible rather than silent.

Each entry to run() appends a new RunRecord to the artifact's meta.  A fresh
artifact starts with one record; a completed artifact that is later extended
appends another, preserving each execution's own commit and timestamps.
"""
from __future__ import annotations

import builtins
import json
import logging
import os
import re
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from datalake.artifact import (
    Artifact,
    ModelCard,
    RunMeta,
    RunRecord,
    utc_now_iso,
)
from datalake.meta import (
    JOB_CONTROL_FILES,
    META_FILENAME,
    git_commit,
    hash_directory,
    read_meta,
    write_sidecars,
)

log = logging.getLogger(__name__)

INDEX_FILENAME = "index.db"
_SCHEMA_PATH = Path(__file__).parent / "schema.sql"

VALID_LAYERS = ("raw", "derived", "output")

# Which layer a kind lands in.  Kinds not listed default to 'derived'.
_LAYER_OVERRIDES: dict[str, str] = {}

# A group is one folder name (CamelCase allowed: RavenPack, WRDS, CapIQ).
_GROUP_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

_LOCK_FILE = JOB_CONTROL_FILES[1]
# A job.lock whose heartbeat is younger than this belongs to a live job (jobs.JobRunner default).
LIVE_LOCK_S = 600.0


class DatalakeError(RuntimeError):
    """Raised for datalake-level failures: unknown artifact, bad layer, etc."""


# ---------------------------------------------------------------------------
# RunHandle: what the `run()` context manager yields
# ---------------------------------------------------------------------------

class RunHandle:
    """Handle to an in-progress run.

    `out_dir` is the directory the pipeline writes into; it exists by the time
    the handle is yielded.  `record` is this execution's RunRecord, already
    appended to `meta.runs`.
    """

    def __init__(
        self,
        index: DatalakeIndex,
        meta: RunMeta,
        record: RunRecord,
        out_dir: Path,
        layer: str,
    ):
        self._index = index
        self.meta = meta
        self.record = record
        self.out_dir = out_dir
        self.layer = layer
        self._file_hashes: dict[str, dict[str, Any]] | None = None

    @property
    def artifact_id(self) -> str:
        return self.meta.artifact_id

    def note(self, text: str) -> None:
        """Append a free-text note to this execution's record."""
        self.record.notes = (
            f"{self.record.notes} {text}".strip() if self.record.notes else text
        )

    def finalize(self, pattern: str = "*") -> dict[str, dict[str, Any]]:
        """Hash all outputs and record which files THIS execution produced.

        The returned hashes cover the whole artifact directory (so the index
        and sidecar record every file).  But `record.produced` is set to only
        the files not already claimed by an earlier execution, so the per-run
        audit trail attributes each file to the run that actually wrote it.
        """
        if self._file_hashes is None:
            self._file_hashes = hash_directory(self.out_dir, pattern=pattern)
            already_claimed: set[str] = set()
            for earlier in self.meta.runs:
                if earlier is not self.record:
                    already_claimed.update(earlier.produced)
            self.record.produced = sorted(
                set(self._file_hashes.keys()) - already_claimed
            )
        return self._file_hashes


# ---------------------------------------------------------------------------
# DatalakeIndex
# ---------------------------------------------------------------------------

class DatalakeIndex:
    """Registry and query interface over a datalake root."""

    def __init__(self, root: Path | str, *, create: bool = True):
        self.root = Path(root)
        if create:
            self.root.mkdir(parents=True, exist_ok=True)
        elif not self.root.exists():
            raise DatalakeError(f"datalake root does not exist: {self.root}")

        self.db_path = self.root / INDEX_FILENAME
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(_SCHEMA_PATH.read_text())
        self._migrate()
        self._conn.commit()
        self._default_group: str | None = None

    def _migrate(self) -> None:
        """Bring an index created by an older schema up to date (idempotent, additive)."""
        columns = {r["name"] for r in self._conn.execute("PRAGMA table_info(artifacts)")}
        if "grp" not in columns:
            self._conn.execute("ALTER TABLE artifacts ADD COLUMN grp TEXT")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_artifacts_group "
                           "ON artifacts (layer, grp, kind)")

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> DatalakeIndex:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- path layout -------------------------------------------------------

    def layer_for(self, kind: str) -> str:
        return _LAYER_OVERRIDES.get(kind, "derived")

    def artifact_dir(self, layer: str, kind: str, artifact_id: str,
                     group: str | None = None) -> Path:
        if layer not in VALID_LAYERS:
            raise DatalakeError(f"invalid layer {layer!r}, expected one of {VALID_LAYERS}")
        base = self.root / layer
        return (base / group if group else base) / kind / artifact_id

    @contextmanager
    def default_group(self, group: str | None) -> Iterator[None]:
        """New artifacts created by ``run()`` inside this block (without an explicit
        ``group``) go under ``group``: how a job's sibling artifacts follow its group."""
        previous, self._default_group = self._default_group, group
        try:
            yield
        finally:
            self._default_group = previous

    def check_group(self, group: str | None, kind: str, layer: str) -> None:
        """Refuse a (group, kind) whose folder would collide with another one.

        Directly under a layer, a name is either a kind folder (artifacts without a
        group) or a group folder: a group may not be named like a kind filed at the
        root of that layer, nor a root-level kind like an existing group."""
        if group is None:
            clash = self._conn.execute(
                "SELECT 1 FROM artifacts WHERE layer = ? AND grp = ? LIMIT 1",
                (layer, kind)).fetchone()
            if clash:
                raise DatalakeError(f"kind {kind!r} is the name of a group folder in {layer}/")
            return
        if not _GROUP_NAME.match(group):
            raise DatalakeError(f"invalid group name {group!r} (one folder name: letters, "
                                "digits, '.', '_', '-')")
        clash = self._conn.execute(
            "SELECT 1 FROM artifacts WHERE layer = ? AND grp IS NULL AND kind = ? LIMIT 1",
            (layer, group)).fetchone()
        if clash:
            raise DatalakeError(f"group {group!r} is the name of a kind folder in {layer}/")

    # -- registration ------------------------------------------------------

    def _upsert(
        self,
        meta: RunMeta,
        layer: str,
        path: Path,
        file_hashes: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """Write (or update) the index row for an artifact.

        The full RunMeta is stored as a JSON blob (meta_json) so that
        _row_to_artifact can reconstruct the complete run history without
        re-reading the sidecar.  The denormalised columns exist only for fast
        filtering (kind, model, partial, deprecated).
        """
        card = meta.model_card
        self._conn.execute(
            """
            INSERT INTO artifacts (
                artifact_id, layer, kind, path,
                pipeline, pipeline_version, pipeline_commit, pipeline_repo,
                model_id, model_version, model_commit,
                hyperparams_json, model_card_json, meta_json,
                run_start, run_end, partial, deprecated, deprecation_reason, notes, grp
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT (artifact_id) DO UPDATE SET
                path               = excluded.path,
                grp                = excluded.grp,
                meta_json          = excluded.meta_json,
                run_start          = excluded.run_start,
                run_end            = excluded.run_end,
                partial            = excluded.partial,
                deprecated         = excluded.deprecated,
                deprecation_reason = excluded.deprecation_reason,
                pipeline_commit    = excluded.pipeline_commit,
                notes              = excluded.notes
            """,
            (
                meta.artifact_id, layer, meta.kind, str(path),
                meta.pipeline, meta.pipeline_version, meta.pipeline_commit,
                meta.pipeline_repo,
                card.model_id if card else None,
                card.version if card else None,
                card.commit if card else None,
                json.dumps(meta.hyperparams, sort_keys=True),
                json.dumps(card.to_dict()) if card else None,
                json.dumps(meta.to_dict()),
                meta.run_start, meta.run_end,
                int(meta.partial), int(meta.deprecated),
                meta.deprecation_reason, meta.notes, meta.group,
            ),
        )

        if file_hashes is not None:
            self._conn.execute(
                "DELETE FROM file_hashes WHERE artifact_id = ?", (meta.artifact_id,)
            )
            self._conn.executemany(
                """
                INSERT INTO file_hashes (artifact_id, filename, algorithm, digest, size_bytes)
                VALUES (?,?,?,?,?)
                """,
                [
                    (meta.artifact_id, name, entry["algorithm"],
                     entry["digest"], entry["size_bytes"])
                    for name, entry in file_hashes.items()
                ],
            )

        self._conn.execute("DELETE FROM lineage WHERE child_id = ?", (meta.artifact_id,))
        if meta.sources:
            self._conn.executemany(
                "INSERT OR IGNORE INTO lineage (child_id, parent_id) VALUES (?,?)",
                [(meta.artifact_id, parent) for parent in meta.sources],
            )

        self._conn.commit()

    # -- the run context manager -------------------------------------------

    @contextmanager
    def run(
        self,
        kind: str,
        pipeline: str,
        pipeline_version: str,
        *,
        hyperparams: dict[str, Any] | None = None,
        sources: Sequence[Artifact | str] | None = None,
        model_card: ModelCard | None = None,
        pipeline_repo: str | None = None,
        verifier: str | None = None,
        repo_dir: Path | None = None,
        layer: str | None = None,
        notes: str = "",
        hash_pattern: str = "*",
        extend: str | None = None,
        resume: str | None = None,
        group: str | None = None,
    ) -> Iterator[RunHandle]:
        """Register, execute, and finalise one pipeline execution.

        By default this creates a NEW artifact.  Pass `extend=<artifact_id>` to
        append a new RunRecord to an existing complete artifact (e.g. adding
        more years to a finished corpus) -- this preserves the earlier
        execution's commit and timestamps rather than overwriting them.

        Pass `resume=<artifact_id>` to continue a PARTIAL artifact (a crashed or
        killed run) in place: its kind and hyperparams must equal the ones given
        here, or DatalakeError is raised -- a resume can never change what the
        artifact means. The crashed execution stays in the history as partial
        and a new RunRecord is appended; on clean exit the artifact is complete.

        On clean exit the execution's RunRecord is marked complete, outputs are
        hashed, and both sidecars are written.  On exception the record is left
        partial so the crash is inspectable.

        Args:
            kind:             Artifact kind, e.g. "headline_embeddings".
            pipeline:         Producing repo name.
            pipeline_version: Semantic version of the pipeline.
            hyperparams:      Parameters that define the artifact's identity.
            sources:          Input Artifacts (or IDs) for lineage.
            model_card:       Model used, if any.
            pipeline_repo:    URL of the producing repo.
            verifier:         Entry-point name for content verification.
            repo_dir:         Directory to read the git SHA from (default cwd).
            layer:            Override the default layer for this kind.
            notes:            Free text recorded on this execution's record.
            hash_pattern:     Glob for which output files to hash.
            extend:           Artifact ID to append a new execution to.
            resume:           Partial artifact ID to continue in place.
            group:            Folder group of a NEW artifact ({layer}/{group}/{kind}/{id});
                              default: the enclosing ``default_group`` scope, else none.
                              Ignored on extend / resume (the artifact stays where it is).
        """
        source_ids = [
            s.artifact_id if isinstance(s, Artifact) else str(s)
            for s in (sources or [])
        ]

        if extend is not None and resume is not None:
            raise DatalakeError("give at most one of extend / resume")
        if resume is not None:
            existing = self.get(resume)
            if not existing.partial:
                raise DatalakeError(f"cannot resume a complete artifact: {resume}. "
                                    "Use extend=... to add a new execution.")
            if existing.kind != kind:
                raise DatalakeError(f"{resume} is a {existing.kind}, not a {kind}")
            if existing.meta.hyperparams != dict(hyperparams or {}):
                raise DatalakeError(
                    f"resume of {resume} with different hyperparams: recorded "
                    f"{existing.meta.hyperparams}, given {dict(hyperparams or {})}")
            meta = existing.meta
            resolved_layer = existing.layer
            out_dir = existing.path
            notes = (f"resume of execution {len(meta.runs)}. {notes}").strip()
        elif extend is not None:
            existing = self.get(extend)
            if existing.partial:
                raise DatalakeError(
                    f"cannot extend a partial artifact: {extend}. "
                    "Finish or deprecate it first."
                )
            meta = existing.meta
            resolved_layer = existing.layer
            out_dir = existing.path
        else:
            meta = RunMeta(
                kind=kind,
                pipeline=pipeline,
                pipeline_version=pipeline_version,
                pipeline_repo=pipeline_repo,
                hyperparams=dict(hyperparams or {}),
                sources=source_ids,
                model_card=model_card,
                verifier=verifier,
                group=group if group is not None else self._default_group,
            )
            resolved_layer = layer or self.layer_for(kind)
            self.check_group(meta.group, kind, resolved_layer)
            out_dir = self.artifact_dir(resolved_layer, kind, meta.artifact_id, meta.group)
            out_dir.mkdir(parents=True, exist_ok=True)

        record = RunRecord(
            run_start=utc_now_iso(),
            pipeline_version=pipeline_version,
            pipeline_commit=git_commit(repo_dir),
            partial=True,
            notes=notes,
        )
        meta.runs.append(record)

        # Persist the partial record before yielding: a hard kill leaves the
        # artifact visible as an incomplete run, not an unexplained directory.
        write_sidecars(out_dir, meta, None)
        self._upsert(meta, resolved_layer, out_dir, file_hashes={})

        handle = RunHandle(self, meta, record, out_dir, resolved_layer)
        log.info(
            "run start: %s (execution %d) -> %s",
            meta.artifact_id, len(meta.runs), out_dir,
        )

        try:
            yield handle
        except BaseException:
            write_sidecars(out_dir, handle.meta, None)
            self._upsert(handle.meta, resolved_layer, out_dir, file_hashes={})
            log.error("run failed, left partial: %s", meta.artifact_id)
            raise

        file_hashes = handle.finalize(pattern=hash_pattern)
        record.run_end = utc_now_iso()
        record.partial = False
        write_sidecars(out_dir, handle.meta, file_hashes)
        self._upsert(handle.meta, resolved_layer, out_dir, file_hashes)
        log.info(
            "run done: %s (%d files, %d execution(s))",
            handle.meta.artifact_id, len(file_hashes), len(handle.meta.runs),
        )

    # -- lookups -----------------------------------------------------------

    def _row_to_artifact(self, row: sqlite3.Row) -> Artifact:
        meta = RunMeta.from_dict(json.loads(row["meta_json"]))
        hashes = {
            r["filename"]: r["digest"]
            for r in self._conn.execute(
                "SELECT filename, digest FROM file_hashes WHERE artifact_id = ?",
                (row["artifact_id"],),
            )
        }
        return Artifact(
            artifact_id=row["artifact_id"],
            layer=row["layer"],
            path=Path(row["path"]),
            meta=meta,
            file_hashes=hashes,
        )

    def _parents_of(self, artifact_id: str) -> list[str]:
        return [
            r["parent_id"]
            for r in self._conn.execute(
                "SELECT parent_id FROM lineage WHERE child_id = ? ORDER BY parent_id",
                (artifact_id,),
            )
        ]

    def get(self, artifact_id: str) -> Artifact:
        """Look up one artifact by ID.  Raises DatalakeError if absent."""
        row = self._conn.execute(
            "SELECT * FROM artifacts WHERE artifact_id = ?", (artifact_id,)
        ).fetchone()
        if row is None:
            raise DatalakeError(f"no artifact with id {artifact_id!r}")
        return self._row_to_artifact(row)

    def list(
        self,
        kind: str | None = None,
        *,
        model: str | None = None,
        version: str | None = None,
        layer: str | None = None,
        group: str | None = None,
        include_partial: bool = False,
        include_deprecated: bool = False,
    ) -> list[Artifact]:
        """List artifacts, newest first.

        Partial and deprecated artifacts are excluded by default: consuming
        either one silently is the failure this layer exists to prevent.
        ``group`` filters on the folder group (None: any; "": artifacts without one).
        """
        clauses: list[str] = []
        params: list[Any] = []
        if kind is not None:
            clauses.append("kind = ?")
            params.append(kind)
        if model is not None:
            clauses.append("model_id = ?")
            params.append(model)
        if version is not None:
            clauses.append("model_version = ?")
            params.append(version)
        if layer is not None:
            clauses.append("layer = ?")
            params.append(layer)
        if group == "":
            clauses.append("grp IS NULL")
        elif group is not None:
            clauses.append("grp = ?")
            params.append(group)
        if not include_partial:
            clauses.append("partial = 0")
        if not include_deprecated:
            clauses.append("deprecated = 0")

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._conn.execute(
            f"SELECT * FROM artifacts {where} ORDER BY run_start DESC, artifact_id DESC",
            params,
        ).fetchall()
        return [self._row_to_artifact(r) for r in rows]

    def latest(
        self,
        kind: str,
        *,
        model: str | None = None,
        version: str | None = None,
        **filters: Any,
    ) -> Artifact:
        """Most recent complete, non-deprecated artifact of `kind`.

        Raises DatalakeError when nothing matches, rather than returning None.
        """
        matches = self.list(kind, model=model, version=version, **filters)
        if not matches:
            detail = f"kind={kind!r}"
            if model:
                detail += f", model={model!r}"
            if version:
                detail += f", version={version!r}"
            raise DatalakeError(f"no complete artifact found for {detail}")
        return matches[0]

    def exists(self, artifact_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM artifacts WHERE artifact_id = ?", (artifact_id,)
        ).fetchone()
        return row is not None

    # -- lineage -----------------------------------------------------------

    def parents(self, artifact_id: str) -> builtins.list[str]:
        """Direct inputs of an artifact."""
        return self._parents_of(artifact_id)

    def children(self, artifact_id: str) -> builtins.list[str]:
        """Artifacts that directly consumed this one."""
        return [
            r["child_id"]
            for r in self._conn.execute(
                "SELECT child_id FROM lineage WHERE parent_id = ? ORDER BY child_id",
                (artifact_id,),
            )
        ]

    def descendants(self, artifact_id: str) -> builtins.list[str]:
        """All artifacts transitively downstream of this one.

        This is the "what is now stale?" query.  Cycles are impossible in a
        well-formed lineage graph but guarded against anyway.
        """
        seen: set[str] = set()
        frontier = [artifact_id]
        while frontier:
            current = frontier.pop()
            for child in self.children(current):
                if child not in seen:
                    seen.add(child)
                    frontier.append(child)
        return sorted(seen)

    def ancestors(self, artifact_id: str) -> builtins.list[str]:
        """All artifacts transitively upstream of this one."""
        seen: set[str] = set()
        frontier = [artifact_id]
        while frontier:
            current = frontier.pop()
            for parent in self.parents(current):
                if parent not in seen:
                    seen.add(parent)
                    frontier.append(parent)
        return sorted(seen)

    # -- mutation ----------------------------------------------------------

    def deprecate(self, artifact_id: str, reason: str) -> None:
        """Mark an artifact deprecated.

        Files stay on disk and the ID stays resolvable via `get()`; only
        `list()` and `latest()` stop returning it by default.  Nothing is ever
        deleted by this layer.
        """
        artifact = self.get(artifact_id)
        artifact.meta.deprecated = True
        artifact.meta.deprecation_reason = reason
        try:
            _, file_hashes = read_meta(artifact.path)
        except FileNotFoundError:
            file_hashes = {}
        self._upsert(artifact.meta, artifact.layer, artifact.path, file_hashes)
        write_sidecars(artifact.path, artifact.meta, file_hashes)
        log.info("deprecated %s: %s", artifact_id, reason)

    def annotate(self, artifact_id: str, notes: str) -> Artifact:
        """Append a note to an existing artifact's metadata (sidecars + index row rewritten).

        Only ``notes`` can be added after the fact: the artifact id is derived
        from the hyperparameters, so those are frozen at creation. Files are
        never touched.
        """
        artifact = self.get(artifact_id)
        artifact.meta.notes = (
            f"{artifact.meta.notes} {notes}".strip() if artifact.meta.notes else notes
        )
        try:
            _, file_hashes = read_meta(artifact.path)
        except FileNotFoundError:
            file_hashes = {}
        self._upsert(artifact.meta, artifact.layer, artifact.path, file_hashes)
        write_sidecars(artifact.path, artifact.meta, file_hashes)
        return self.get(artifact_id)

    def move(self, artifact_id: str, group: str | None, *, dry_run: bool = False,
             allow_partial: bool = False) -> tuple[Path, Path]:
        """Re-file an artifact under ``group`` (None: no group). Returns (old, new) path.

        Content, id, hashes and lineage are untouched: the directory is renamed (one
        atomic step, the commit point), then ``meta.json`` / README and the index row
        get the new group and path. A crash in between is repaired by ``reindex()``
        (the folder wins). Refused: a live ``job.lock``; a partial artifact (the
        siblings of a running job hold no lock) unless ``allow_partial``; an existing
        target; a target on another device (a rename never becomes a copy).
        """
        art = self.get(artifact_id)
        source = art.path
        target = self.artifact_dir(art.layer, art.kind, artifact_id, group)
        if group == art.group and source == target:
            return source, target
        self.check_group(group, art.kind, art.layer)
        if not source.is_dir():
            raise DatalakeError(f"{artifact_id}: directory missing from disk: {source}")
        lock = source / _LOCK_FILE
        if lock.exists() and _lock_age_s(lock) < LIVE_LOCK_S:
            raise DatalakeError(f"{artifact_id} is held by a live job ({lock}); pause it first")
        if art.partial and not allow_partial:
            raise DatalakeError(f"{artifact_id} is partial (a running job may be writing it); "
                                "pass allow_partial once no job uses it")
        if target.exists():
            raise DatalakeError(f"target already exists: {target}")
        _same_device(source, target)
        if dry_run:
            return source, target
        target.parent.mkdir(parents=True, exist_ok=True)
        os.rename(source, target)
        meta, file_hashes = read_meta(target)
        meta.group = group
        write_sidecars(target, meta, file_hashes)
        self._upsert(meta, art.layer, target, file_hashes)
        for empty in (source.parent, source.parent.parent):     # kind folder, group folder
            if empty != self.root / art.layer:
                try:
                    empty.rmdir()                                 # only when empty
                except OSError:
                    break
        log.info("moved %s: %s -> %s", artifact_id, source, target)
        return source, target

    def relink(self, artifact_id: str, **kwargs: Any) -> Any:
        """Register ``artifact_id``'s files under new inputs / code (``datalake.relink``)."""
        from datalake.relink import relink

        return relink(self, artifact_id, **kwargs)

    # -- reindex -----------------------------------------------------------

    def reindex(self) -> int:
        """Rebuild the index from the meta.json sidecars on disk.

        The index holds no information the sidecars lack, so this is always
        safe: the recovery path after losing index.db, and the way to pick up
        artifacts copied from another machine.

        Returns the number of artifacts registered.
        """
        self._conn.execute("DELETE FROM lineage")
        self._conn.execute("DELETE FROM file_hashes")
        self._conn.execute("DELETE FROM artifacts")
        self._conn.commit()

        count = 0
        for layer in VALID_LAYERS:
            layer_dir = self.root / layer
            if not layer_dir.is_dir():
                continue
            for artifact_dir in _artifact_dirs(layer_dir):
                rel = artifact_dir.relative_to(layer_dir).parts
                if len(rel) not in (2, 3):
                    log.warning("skipping %s: not {kind}/{id} nor {group}/{kind}/{id}",
                                artifact_dir)
                    continue
                try:
                    meta, file_hashes = read_meta(artifact_dir)
                except (OSError, ValueError) as exc:
                    log.warning("skipping unreadable sidecar %s: %s", artifact_dir, exc)
                    continue
                folder_group = rel[0] if len(rel) == 3 else None
                if meta.group != folder_group:           # moved by hand: the folder wins
                    log.warning("%s: sidecar group %r, folder group %r (folder kept)",
                                meta.artifact_id, meta.group, folder_group)
                    meta.group = folder_group
                if rel[-2] != meta.kind:
                    log.warning("%s: a %s filed under the %r folder", meta.artifact_id,
                                meta.kind, rel[-2])
                self._upsert(meta, layer, artifact_dir, file_hashes)
                count += 1

        log.info("reindexed %d artifacts under %s", count, self.root)
        return count

    # -- DuckDB integration ------------------------------------------------

    def register(
        self,
        conn: Any,
        kind_or_artifact: str | Artifact,
        *,
        alias: str | None = None,
        pattern: str = "*.parquet",
        **filters: Any,
    ) -> str:
        """Register an artifact as a DuckDB view.

        Lets notebooks query by kind rather than by path:

            dl.register(conn, "primitive_scores", alias="scores")
            conn.execute("SELECT * FROM scores WHERE DATE >= '2010-01-01'")

        Returns the view name.
        """
        artifact = (
            kind_or_artifact
            if isinstance(kind_or_artifact, Artifact)
            else self.latest(kind_or_artifact, **filters)
        )
        view = alias or artifact.kind
        if not view.replace("_", "").isalnum():
            raise DatalakeError(f"unsafe view name {view!r}")
        conn.execute(
            f"CREATE OR REPLACE VIEW {view} AS "
            f"SELECT * FROM read_parquet('{artifact.glob(pattern)}')"
        )
        return view


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _artifact_dirs(layer_dir: Path) -> list[Path]:
    """Every directory under ``layer_dir`` holding a meta.json, sorted; never descends
    into an artifact (its own sub-directories are content, not artifacts)."""
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(layer_dir):
        if META_FILENAME in filenames and Path(dirpath) != layer_dir:
            found.append(Path(dirpath))
            dirnames[:] = []
        else:
            dirnames.sort()
    return sorted(found)


def _lock_age_s(lock: Path) -> float:
    """Seconds since a job.lock's last heartbeat (inf when unreadable)."""
    from datetime import datetime, timezone

    try:
        beat = json.loads(lock.read_text())["heartbeat_at"]
        return (datetime.now(timezone.utc) - datetime.fromisoformat(beat)).total_seconds()
    except (OSError, ValueError, KeyError):
        return float("inf")


def _same_device(source: Path, target: Path) -> None:
    """Refuse a rename / hard link across filesystems (it would need a copy)."""
    anchor = target
    while not anchor.exists():
        anchor = anchor.parent
    if source.stat().st_dev != anchor.stat().st_dev:
        raise DatalakeError(f"{source} and {target} are on different filesystems: "
                            "refusing to copy")
