"""DatalakeIndex: the registry and query layer over the datalake tree.

The datalake is a POSIX tree of immutable artifact directories:

    {root}/{layer}/{kind}/{artifact_id}/
        meta.json
        README.md
        2000-01.parquet
        ...

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
    ) as run:
        embed_corpus(input_dir=source.path, out_dir=run.out_dir)

On clean exit the run hashes its outputs, writes both sidecars, flips
partial to false, and registers lineage edges.  On exception it leaves the
artifact registered as partial so the crash is visible rather than silent.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from datalake.artifact import Artifact, ModelCard, RunMeta, utc_now_iso
from datalake.meta import (
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

# Which layer a kind lands in.  Kinds not listed default to 'derived', which
# is correct for everything the pipelines produce; 'raw' is for vintaged
# source data registered by hand, 'output' for final research artifacts.
_LAYER_OVERRIDES: dict[str, str] = {}


class DatalakeError(RuntimeError):
    """Raised for datalake-level failures: unknown artifact, bad layer, etc."""


# ---------------------------------------------------------------------------
# RunHandle: what the `run()` context manager yields
# ---------------------------------------------------------------------------

class RunHandle:
    """Handle to an in-progress run.

    `out_dir` is the directory the pipeline should write its outputs into.  It
    exists by the time the handle is yielded.
    """

    def __init__(self, index: DatalakeIndex, meta: RunMeta, out_dir: Path, layer: str):
        self._index = index
        self.meta = meta
        self.out_dir = out_dir
        self.layer = layer
        self._file_hashes: dict[str, dict[str, Any]] | None = None

    @property
    def artifact_id(self) -> str:
        return self.meta.artifact_id

    def note(self, text: str) -> None:
        """Append a free-text note recorded in the sidecars."""
        self.meta.notes = f"{self.meta.notes} {text}".strip() if self.meta.notes else text

    def finalize(self, pattern: str = "*") -> dict[str, dict[str, Any]]:
        """Hash outputs and cache the result.

        Called automatically on clean exit; call it explicitly only to inspect
        hashes before the context closes.
        """
        if self._file_hashes is None:
            self._file_hashes = hash_directory(self.out_dir, pattern=pattern)
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
        # WAL lets a long-running writer coexist with readers (a notebook
        # querying the index while a pipeline registers a new artifact).
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(_SCHEMA_PATH.read_text())
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> DatalakeIndex:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- path layout -------------------------------------------------------

    def layer_for(self, kind: str) -> str:
        return _LAYER_OVERRIDES.get(kind, "derived")

    def artifact_dir(self, layer: str, kind: str, artifact_id: str) -> Path:
        if layer not in VALID_LAYERS:
            raise DatalakeError(f"invalid layer {layer!r}, expected one of {VALID_LAYERS}")
        return self.root / layer / kind / artifact_id

    # -- registration ------------------------------------------------------

    def _upsert(
        self,
        meta: RunMeta,
        layer: str,
        path: Path,
        file_hashes: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        card = meta.model_card
        self._conn.execute(
            """
            INSERT INTO artifacts (
                artifact_id, layer, kind, path,
                pipeline, pipeline_version, pipeline_commit, pipeline_repo,
                model_id, model_version, model_commit,
                hyperparams_json, model_card_json,
                run_start, run_end, partial, deprecated, deprecation_reason, notes
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT (artifact_id) DO UPDATE SET
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
                meta.run_start, meta.run_end,
                int(meta.partial), int(meta.deprecated),
                meta.deprecation_reason, meta.notes,
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
        repo_dir: Path | None = None,
        layer: str | None = None,
        notes: str = "",
        hash_pattern: str = "*",
    ) -> Iterator[RunHandle]:
        """Register, execute, and finalise one pipeline run.

        The yielded handle's `out_dir` is where the pipeline writes.  On clean
        exit outputs are hashed, sidecars written, and partial cleared.  On
        exception the artifact stays registered with partial=1 so the failure
        is inspectable rather than invisible.

        Args:
            kind:             Artifact kind, e.g. "headline_embeddings".
            pipeline:         Producing repo name.
            pipeline_version: Semantic version of the pipeline.
            hyperparams:      Parameters that define this run's identity.
            sources:          Input Artifacts (or their IDs) for lineage.
            model_card:       Model used, if any.
            pipeline_repo:    URL of the producing repo.
            repo_dir:         Directory to read the git SHA from (default cwd).
            layer:            Override the default layer for this kind.
            notes:            Free text recorded in the sidecars.
            hash_pattern:     Glob for which output files to hash.
        """
        source_ids = [
            s.artifact_id if isinstance(s, Artifact) else str(s)
            for s in (sources or [])
        ]

        meta = RunMeta(
            kind=kind,
            pipeline=pipeline,
            pipeline_version=pipeline_version,
            pipeline_commit=git_commit(repo_dir),
            pipeline_repo=pipeline_repo,
            hyperparams=dict(hyperparams or {}),
            sources=source_ids,
            model_card=model_card,
            run_start=utc_now_iso(),
            partial=True,
            notes=notes,
        )

        resolved_layer = layer or self.layer_for(kind)
        out_dir = self.artifact_dir(resolved_layer, kind, meta.artifact_id)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Write the partial sidecar and register before yielding: if the
        # process is killed hard (OOM, power loss) the artifact is already
        # visible as an incomplete run rather than an unexplained directory.
        write_sidecars(out_dir, meta, None)
        self._upsert(meta, resolved_layer, out_dir, file_hashes={})

        handle = RunHandle(self, meta, out_dir, resolved_layer)
        log.info("run start: %s -> %s", meta.artifact_id, out_dir)

        try:
            yield handle
        except BaseException:
            # Leave partial=1.  Refresh the sidecar so any note added during
            # the run survives, then let the exception propagate.
            write_sidecars(out_dir, handle.meta, None)
            self._upsert(handle.meta, resolved_layer, out_dir, file_hashes={})
            log.error("run failed, left partial: %s", meta.artifact_id)
            raise

        file_hashes = handle.finalize(pattern=hash_pattern)
        handle.meta.run_end = utc_now_iso()
        handle.meta.partial = False
        write_sidecars(out_dir, handle.meta, file_hashes)
        self._upsert(handle.meta, resolved_layer, out_dir, file_hashes)
        log.info(
            "run done: %s (%d files)", handle.meta.artifact_id, len(file_hashes)
        )

    # -- lookups -----------------------------------------------------------

    def _row_to_artifact(self, row: sqlite3.Row) -> Artifact:
        card_json = row["model_card_json"]
        meta = RunMeta(
            kind=row["kind"],
            pipeline=row["pipeline"],
            pipeline_version=row["pipeline_version"],
            pipeline_commit=row["pipeline_commit"],
            pipeline_repo=row["pipeline_repo"],
            hyperparams=json.loads(row["hyperparams_json"]),
            sources=self._parents_of(row["artifact_id"]),
            model_card=ModelCard.from_dict(json.loads(card_json)) if card_json else None,
            run_start=row["run_start"],
            run_end=row["run_end"],
            partial=bool(row["partial"]),
            deprecated=bool(row["deprecated"]),
            deprecation_reason=row["deprecation_reason"],
            notes=row["notes"],
        )
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
        include_partial: bool = False,
        include_deprecated: bool = False,
    ) -> list[Artifact]:
        """List artifacts, newest first.

        Partial and deprecated artifacts are excluded by default: consuming
        either one silently is the failure this layer exists to prevent.
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

        Raises DatalakeError when nothing matches, rather than returning None:
        a pipeline that silently proceeds without its input is worse than one
        that stops.
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

    def parents(self, artifact_id: str) -> list[str]:
        """Direct inputs of an artifact."""
        return self._parents_of(artifact_id)

    def children(self, artifact_id: str) -> list[str]:
        """Artifacts that directly consumed this one."""
        return [
            r["child_id"]
            for r in self._conn.execute(
                "SELECT child_id FROM lineage WHERE parent_id = ? ORDER BY child_id",
                (artifact_id,),
            )
        ]

    def descendants(self, artifact_id: str) -> list[str]:
        """All artifacts transitively downstream of this one.

        This is the "what is now stale?" query: re-embed the corpus, pass the
        old embedding artifact ID here, and get back every score, beta and
        output built on it.

        Cycles are impossible in a well-formed lineage graph but are guarded
        against anyway, since a corrupted index should not hang a CLI.
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

    def ancestors(self, artifact_id: str) -> list[str]:
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

        The files stay on disk and the ID stays resolvable via `get()`; only
        `list()` and `latest()` stop returning it by default.  Nothing in the
        datalake is ever deleted by this layer.
        """
        artifact = self.get(artifact_id)
        artifact.meta.deprecated = True
        artifact.meta.deprecation_reason = reason
        self._upsert(artifact.meta, artifact.layer, artifact.path, None)
        # Keep the on-disk sidecars in step with the index.
        try:
            _, file_hashes = read_meta(artifact.path)
        except FileNotFoundError:
            file_hashes = {}
        write_sidecars(artifact.path, artifact.meta, file_hashes)
        log.info("deprecated %s: %s", artifact_id, reason)

    # -- reindex -----------------------------------------------------------

    def reindex(self) -> int:
        """Rebuild the index from the meta.json sidecars on disk.

        The index holds no information the sidecars lack, so this is always
        safe: it is the recovery path after losing or corrupting index.db, and
        the way to pick up artifacts copied in from another machine.

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
            for meta_path in sorted(layer_dir.glob(f"*/*/{META_FILENAME}")):
                artifact_dir = meta_path.parent
                try:
                    meta, file_hashes = read_meta(artifact_dir)
                except (OSError, ValueError) as exc:
                    log.warning("skipping unreadable sidecar %s: %s", meta_path, exc)
                    continue
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

        Lets notebooks and scripts query by kind rather than by path:

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
