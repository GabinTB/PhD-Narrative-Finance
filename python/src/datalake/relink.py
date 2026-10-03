"""Relink: register an artifact's files under new inputs or new code, checked.

When a parent is re-ingested, restructured or re-registered (new id) but what a
child was computed from is unchanged, recomputing the child is waste. ``relink``
registers a NEW artifact that

  * has the child's kind, pipeline, model card, verifier and group;
  * records the new parents as sources (``replace`` = {old parent: new parent});
    hyperparams equal to a replaced id are substituted (inside lists and dicts too),
    then ``set_params`` applies explicit changes (e.g. a derived digest);
  * optionally carries a new ``pipeline_version`` (relink under new code);
  * holds the same files, as hard links (zero copy; same filesystem required), with
    the old sidecar's digests (each link is checked to be the same inode, so nothing
    is re-hashed; ``datalake verify --artifact`` re-hashes on demand);
  * records ``meta.relink`` = {from, replace, checks, commit}, NOT a lineage source:
    ``require_lineage`` keeps comparing true inputs.

Every replacement is validated before any check runs (``validate_replace``): the
new parent is complete and not deprecated; it has the old parent's kind unless the
kind change is declared (``kind_changes``); both parents share a partition frequency
and the new one holds every partition the child holds; at least one content check
(``files`` or ``projection``) compares each old parent with its new one; a child with
period partitions needs a ``keys`` check. A replaced id the child never recorded is
refused unless ``assert_sources`` (legacy lineage, owner-asserted): then the asserted
parent must also hold exactly the child's partitions with the same row counts
(parquet metadata). Asserted ids are recorded in ``meta.relink["asserted"]``.

The equivalence checks (``datalake.equivalence``) run before anything is
registered: the explicit ones plus the kind's defaults. Any failure removes the
new directory and registers nothing. The old artifact is never modified, except
when ``deprecate_old`` is passed explicitly. A relinked artifact is complete; a
job that supports updates rebuilds itself from its meta (``Job.for_update``), so
``jobs update`` works on it like on any complete artifact.

Chains (a parent and its children all relinked): relink parents first, then
pass every replacement a child depends on (``relink_order`` lists descendants in
topological order).
"""
from __future__ import annotations

import logging
import os
import shutil
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from datalake.artifact import Artifact, RunMeta, RunRecord, utc_now_iso
from datalake.equivalence import (
    Check,
    CheckResult,
    FilesIdentical,
    KeysCover,
    ProjectionEqual,
    RelinkContext,
    default_checks,
)
from datalake.index import DatalakeError, _same_device
from datalake.meta import (
    JOB_CONTROL_FILES,
    META_FILENAME,
    README_FILENAME,
    git_commit,
    read_meta,
    write_sidecars,
)
from datalake.periods import PeriodError, parse_key

if TYPE_CHECKING:
    from datalake.index import DatalakeIndex

log = logging.getLogger(__name__)

_NOT_LINKED = frozenset({META_FILENAME, README_FILENAME, *JOB_CONTROL_FILES})


class RelinkError(DatalakeError):
    """A relink was refused, or its equivalence checks failed (``results``)."""

    def __init__(self, message: str, results: Sequence[CheckResult] = ()) -> None:
        super().__init__(message)
        self.results = list(results)


@dataclass
class RelinkResult:
    old_id: str
    new_id: str
    path: Path
    hyperparams_diff: dict[str, tuple[Any, Any]]
    sources_diff: dict[str, str]
    n_files: int
    n_bytes: int
    checks: list[CheckResult] = field(default_factory=list)
    registered: bool = False

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)


def substitute(value: Any, replace: dict[str, str]) -> Any:
    """``value`` with every string equal to a replaced id swapped (recursively)."""
    if isinstance(value, str):
        return replace.get(value, value)
    if isinstance(value, list):
        return [substitute(v, replace) for v in value]
    if isinstance(value, tuple):
        return tuple(substitute(v, replace) for v in value)
    if isinstance(value, dict):
        return {k: substitute(v, replace) for k, v in value.items()}
    return value


def _embedded(value: Any, ids: Sequence[str]) -> list[str]:
    """Replaced ids still appearing INSIDE a longer string (substitution cannot fix)."""
    found: list[str] = []
    if isinstance(value, str):
        found += [i for i in ids if i in value and value != i]
    elif isinstance(value, (list, tuple)):
        for v in value:
            found += _embedded(v, ids)
    elif isinstance(value, dict):
        for v in value.values():
            found += _embedded(v, ids)
    return found


def _content_files(source: Path) -> Iterator[Path]:
    """Every content file of an artifact directory, sub-directories included (not the
    sidecars, job control files or stray .tmp files)."""
    for dirpath, dirnames, filenames in os.walk(source):
        dirnames.sort()
        top = Path(dirpath) == source
        for name in sorted(filenames):
            if (top and name in _NOT_LINKED) or name.endswith(".tmp"):
                continue
            yield Path(dirpath) / name


def _link_tree(source: Path, target: Path) -> tuple[int, int]:
    """Hard-link every content file of ``source`` into a new ``target`` (same relative
    paths). Returns (files, bytes)."""
    target.mkdir(parents=True)
    n_files = n_bytes = 0
    for src in _content_files(source):
        dst = target / src.relative_to(source)
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.link(src, dst)
        n_files += 1
        n_bytes += src.stat().st_size
    return n_files, n_bytes


def _partitions(art: Artifact) -> dict[str, Path]:
    """Period-keyed parquet files of an artifact (stem -> path); other files ignored."""
    out = {}
    for p in sorted(art.path.glob("*.parquet")):
        try:
            parse_key(p.stem)
        except PeriodError:
            continue
        out[p.stem] = p
    return out


def _freqs(parts: dict[str, Path]) -> set[str]:
    return {parse_key(k).freq for k in parts}


def _row_counts(parts: dict[str, Path]) -> dict[str, int]:
    import pyarrow.parquet as pq

    return {k: pq.ParquetFile(p).metadata.num_rows for k, p in parts.items()}


def validate_replace(index: DatalakeIndex, old: Artifact, replace: dict[str, str],
                     checks: Sequence[Check], *, assert_sources: bool = False,
                     kind_changes: dict[str, str] | None = None) -> list[str]:
    """Problems that refuse ``replace`` for ``old`` before any check runs (empty: valid).
    See the module docstring for the rules."""
    kind_changes = dict(kind_changes or {})
    problems: list[str] = []
    child_parts = _partitions(old)
    for old_id, new_id in replace.items():
        recorded = old_id in old.meta.sources
        if not recorded and not assert_sources:
            problems.append(f"[source] {old_id} is not a recorded source of {old.artifact_id} "
                            f"(sources: {old.meta.sources}); pass assert_sources if it was "
                            "built from it without recording it")
            continue
        if not index.exists(old_id) or not index.exists(new_id):
            problems.append(f"[registered] {old_id if not index.exists(old_id) else new_id} "
                            "is not registered")
            continue
        op, np_ = index.get(old_id), index.get(new_id)
        if np_.partial or np_.deprecated:
            problems.append(f"[state] new parent {new_id} is "
                            f"{'partial' if np_.partial else 'deprecated'}")
        if np_.kind != op.kind and kind_changes.get(op.kind) != np_.kind:
            problems.append(f"[kind] {old_id} is a {op.kind}, {new_id} a {np_.kind}: declare "
                            f"the change (kind_changes {{{op.kind!r}: {np_.kind!r}}})")
        op_parts, np_parts = _partitions(op), _partitions(np_)
        if _freqs(op_parts) != _freqs(np_parts):
            problems.append(f"[layout] partition frequencies differ: {sorted(_freqs(op_parts))} "
                            f"vs {sorted(_freqs(np_parts))}")
        missing = sorted(set(child_parts) - set(np_parts))
        if missing:
            problems.append(f"[layout] {new_id} lacks {len(missing)} partition(s) the child "
                            f"holds: {missing[:5]}")
        if not recorded:
            if set(op_parts) != set(child_parts):
                problems.append(f"[asserted] {old_id} and {old.artifact_id} hold different "
                                f"partitions ({len(op_parts)} vs {len(child_parts)})")
            else:
                a, b = _row_counts(op_parts), _row_counts(child_parts)
                bad = [k for k in child_parts if a[k] != b[k]]
                if bad:
                    problems.append(f"[asserted] row counts differ in {len(bad)} partition(s), "
                                    f"e.g. {bad[0]}: {a[bad[0]]:,} vs {b[bad[0]]:,}")
        single = len(replace) == 1
        content = [c for c in checks if isinstance(c, (FilesIdentical, ProjectionEqual))
                   and (c.parent == old_id or (c.parent is None and single))]
        if not content:
            problems.append(f"[checks] no content check (files@ / projection@) compares "
                            f"{old_id} with {new_id}")
    if child_parts and replace and not any(isinstance(c, KeysCover) for c in checks):
        problems.append(f"[checks] {old.artifact_id} has period partitions: a keys check is "
                        "required")
    return problems


def relink(index: DatalakeIndex, artifact_id: str, *, replace: dict[str, str],
           set_params: dict[str, Any] | None = None, pipeline_version: str | None = None,
           checks: Sequence[Check] = (), notes: str = "", deprecate_old: bool = False,
           dry_run: bool = False, repo_dir: Path | None = None, assert_sources: bool = False,
           kind_changes: dict[str, str] | None = None) -> RelinkResult:
    """Register ``artifact_id``'s files as a new artifact under ``replace`` /
    ``set_params`` / ``pipeline_version``, after the equivalence checks.

    Raises ``RelinkError`` when refused or when a check fails (nothing registered,
    no directory left). ``dry_run`` runs the checks against the old directory and
    returns what would be registered, writing nothing."""
    old = index.get(artifact_id)
    replace = dict(replace)
    if old.partial:
        raise RelinkError(f"{artifact_id} is partial: finish it before relinking")
    if not replace and not set_params and pipeline_version is None:
        raise RelinkError("nothing to relink: give replace, set_params or pipeline_version")
    asserted = [p for p in replace if p not in old.meta.sources]
    if asserted and not assert_sources:              # first: it decides everything below
        raise RelinkError(f"{artifact_id}: relink refused:\n  [source] {asserted} are not "
                          f"recorded sources (sources: {old.meta.sources}); pass "
                          "assert_sources if it was built from them without recording it")

    hyperparams = substitute(dict(old.meta.hyperparams), replace)
    hyperparams.update(set_params or {})
    leftovers = _embedded(hyperparams, list(replace))
    if leftovers:
        log.warning("%s: hyperparams still embed replaced id(s) %s inside longer values; "
                    "pass set_params for them", artifact_id, sorted(set(leftovers)))
    meta = RunMeta(
        kind=old.kind, pipeline=old.meta.pipeline,
        pipeline_version=pipeline_version or old.meta.pipeline_version,
        pipeline_repo=old.meta.pipeline_repo, hyperparams=hyperparams,
        sources=[replace.get(s, s) for s in old.meta.sources]
        + [replace[a] for a in asserted],
        model_card=old.meta.model_card, verifier=old.meta.verifier, group=old.group,
    )
    new_id = meta.artifact_id
    if new_id == artifact_id or index.exists(new_id):
        raise RelinkError(f"the relinked artifact would be {new_id}, which already exists "
                          "(nothing changes the identity, or it was relinked already today)")
    new_dir = index.artifact_dir(old.layer, old.kind, new_id, old.group)
    if new_dir.exists():
        raise RelinkError(f"target directory already exists: {new_dir}")
    _same_device(old.path, new_dir)

    _, file_hashes = read_meta(old.path)
    hp_diff = {k: (old.meta.hyperparams.get(k), hyperparams.get(k))
               for k in sorted(set(old.meta.hyperparams) | set(hyperparams))
               if old.meta.hyperparams.get(k) != hyperparams.get(k)}

    digests = {k: v["digest"] for k, v in file_hashes.items()}
    pre = RelinkContext(index, old, Artifact(new_id, old.layer, old.path, meta, digests), replace)
    all_checks = list(checks) + default_checks(pre)
    if not all_checks:
        raise RelinkError("no equivalence check given and none registered for "
                          f"{old.kind}: refusing an unchecked relink")
    problems = validate_replace(index, old, replace, all_checks,
                                assert_sources=assert_sources, kind_changes=kind_changes)
    if problems:
        raise RelinkError(f"{artifact_id}: relink refused:\n  " + "\n  ".join(problems))

    if dry_run:
        content = list(_content_files(old.path))
        n_files, n_bytes = len(content), sum(p.stat().st_size for p in content)
        work_dir = old.path
    else:
        n_files, n_bytes = _link_tree(old.path, new_dir)
        work_dir = new_dir
    ctx = RelinkContext(index, old, Artifact(new_id, old.layer, work_dir, meta, digests), replace)
    results: list[CheckResult] = []
    try:
        for check in all_checks:
            try:
                results.append(check(ctx))
            except Exception as exc:  # noqa: BLE001 - a crashing check is a failed check
                results.append(CheckResult(getattr(check, "name", repr(check)), False,
                                           f"{type(exc).__name__}: {exc}"))
        if not dry_run:
            for name, entry in file_hashes.items():
                a, b = (old.path / name).stat(), (new_dir / name).stat()
                if (a.st_dev, a.st_ino) != (b.st_dev, b.st_ino):
                    raise RelinkError(f"{name} is not a hard link of the old file")
                if b.st_size != entry["size_bytes"]:
                    raise RelinkError(f"{name}: size {b.st_size} differs from the recorded "
                                      f"{entry['size_bytes']} (old artifact changed on disk?)")
    except BaseException:
        if not dry_run:
            shutil.rmtree(new_dir, ignore_errors=True)
        raise

    result = RelinkResult(artifact_id, new_id, new_dir, hp_diff,
                          {k: v for k, v in replace.items()}, n_files, n_bytes, results)
    if not result.passed:
        if not dry_run:
            shutil.rmtree(new_dir, ignore_errors=True)
        failed = "; ".join(f"{c.name}: {c.details}" for c in results if not c.passed)
        raise RelinkError(f"{artifact_id}: equivalence check(s) failed, nothing registered: "
                          f"{failed}", results)
    if dry_run:
        return result

    commit = git_commit(repo_dir)
    now = utc_now_iso()
    meta.relink = {"from": artifact_id, "from_version": old.meta.pipeline_version,
                   "replace": replace, "asserted": asserted,
                   "kind_changes": dict(kind_changes or {}),
                   "checks": [c.to_dict() for c in results], "commit": commit}
    meta.notes = notes
    meta.runs = [RunRecord(run_start=now, run_end=now, pipeline_version=meta.pipeline_version,
                           pipeline_commit=commit, partial=False, produced=sorted(file_hashes),
                           notes=(f"relinked from {artifact_id} (hard links, "
                                  f"{len(results)} check(s) passed). {notes}").strip())]
    write_sidecars(new_dir, meta, file_hashes)
    index._upsert(meta, old.layer, new_dir, file_hashes)
    result.registered = True
    log.info("relinked %s -> %s (%d files, %d checks)", artifact_id, new_id, n_files,
             len(results))
    if deprecate_old:
        index.deprecate(artifact_id, f"superseded by relink {new_id}")
    return result


def relink_order(index: DatalakeIndex, artifact_id: str) -> list[str]:
    """Descendants of ``artifact_id`` in topological order (parents before children):
    the order in which they would be relinked or recomputed."""
    todo: set[str] = set(index.descendants(artifact_id))
    ordered: list[str] = []
    done = {artifact_id}
    while todo:
        ready = sorted(a for a in todo
                       if all(p in done or p not in todo for p in index.parents(a)))
        if not ready:                      # a cycle cannot happen; guard anyway
            ready = sorted(todo)
        ordered += ready
        done.update(ready)
        todo.difference_update(ready)
    return ordered


__all__ = ["RelinkError", "RelinkResult", "relink", "relink_order", "substitute",
           "validate_replace"]
