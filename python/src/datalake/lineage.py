"""Lineage checks: was an artifact built from the inputs a run is about to combine it with?

An artifact records the artifacts it was built from in ``meta.sources`` (ids that
start with their kind, ``kind__...``). ``require_lineage(child, expected)`` compares,
for each expected kind, the id the run uses with the id(s) the child recorded:

  * recorded and different -> ``LineageError`` (mixing inputs would be silent);
  * nothing of that kind recorded (legacy or migrated artifacts) -> a warning: the
    lineage cannot be verified, the run goes on.

Only validation: nothing here changes an id, a lookup key or an output.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING

from datalake.index import DatalakeError

if TYPE_CHECKING:
    from datalake.artifact import Artifact

log = logging.getLogger(__name__)


class LineageError(DatalakeError):
    """An artifact was built from other inputs than the ones a run combines it with."""


def kind_of(artifact_id: str) -> str:
    return artifact_id.split("__", 1)[0]


def recorded_sources(artifact: Artifact, kind: str) -> list[str]:
    """The ids of kind ``kind`` the artifact records as sources."""
    return [s for s in artifact.meta.sources if kind_of(s) == kind]


def lineage_problems(artifact: Artifact,
                     expected: Mapping[str, str | None]) -> tuple[list[str], list[str]]:
    """(mismatches, unverifiable) of ``artifact`` against ``{kind: expected id}``.

    Kinds whose expected id is None are not checked.
    """
    mismatches, unverifiable = [], []
    for kind, want in expected.items():
        if want is None:
            continue
        got = recorded_sources(artifact, kind)
        if not got:
            unverifiable.append(kind)
        elif want not in got:
            mismatches.append(f"{artifact.artifact_id} was built from {kind} "
                              f"{', '.join(got)}, but this run uses {want}")
    return mismatches, unverifiable


def require_lineage(checks: list[tuple[Artifact | None, Mapping[str, str | None]]],
                    *, what: str = "run") -> None:
    """Raise ``LineageError`` listing every mismatch of ``checks`` (artifact, expected);
    warn once per artifact whose lineage is not recorded. ``None`` artifacts are skipped."""
    errors: list[str] = []
    for artifact, expected in checks:
        if artifact is None:
            continue
        mismatches, unverifiable = lineage_problems(artifact, expected)
        errors += mismatches
        if unverifiable:
            log.warning("%s: %s records no %s source; its lineage cannot be verified",
                        what, artifact.artifact_id, " / ".join(unverifiable))
    if errors:
        raise LineageError(f"{what}: inconsistent inputs, refusing to mix them:\n  "
                           + "\n  ".join(errors))


__all__ = ["LineageError", "kind_of", "recorded_sources", "lineage_problems",
           "require_lineage"]
