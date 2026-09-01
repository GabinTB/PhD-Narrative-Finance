"""Datalake verification.

Answers three questions that matter before anyone trusts a result:

  1. Do the files on disk still hash to what was recorded when they were
     written?  (Corruption, partial sync, manual edit.)
  2. Are there artifacts that never completed, or stray .tmp files from a
     crashed write?
  3. Does every lineage edge point at something that exists?

Verification never mutates anything.  It reports; the human decides.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

from datalake.index import DatalakeIndex
from datalake.meta import META_FILENAME, hash_file, read_meta

log = logging.getLogger(__name__)


class Severity(str, Enum):
    ERROR = "error"      # artifact cannot be trusted
    WARNING = "warning"  # suspicious, may be benign


@dataclass
class Finding:
    severity: Severity
    artifact_id: str
    message: str

    def __str__(self) -> str:
        return f"[{self.severity.value.upper()}] {self.artifact_id}: {self.message}"


@dataclass
class VerifyReport:
    findings: list[Finding] = field(default_factory=list)
    n_artifacts_checked: int = 0
    n_files_checked: int = 0

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity is Severity.ERROR]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity is Severity.WARNING]

    @property
    def ok(self) -> bool:
        return not self.errors

    def add(self, severity: Severity, artifact_id: str, message: str) -> None:
        self.findings.append(Finding(severity, artifact_id, message))

    def summary(self) -> str:
        if self.ok and not self.warnings:
            return (
                f"OK: {self.n_artifacts_checked} artifacts, "
                f"{self.n_files_checked} files verified"
            )
        return (
            f"{len(self.errors)} error(s), {len(self.warnings)} warning(s) "
            f"across {self.n_artifacts_checked} artifacts "
            f"({self.n_files_checked} files checked)"
        )


def verify_hashes(
    index: DatalakeIndex,
    report: VerifyReport,
    *,
    artifact_ids: list[str] | None = None,
) -> None:
    """Re-hash every registered file and compare against the recorded digest."""
    artifacts = (
        [index.get(a) for a in artifact_ids]
        if artifact_ids
        else index.list(include_partial=True, include_deprecated=True)
    )

    for artifact in artifacts:
        report.n_artifacts_checked += 1

        if not artifact.path.is_dir():
            report.add(
                Severity.ERROR, artifact.artifact_id,
                f"directory missing from disk: {artifact.path}",
            )
            continue

        if not (artifact.path / META_FILENAME).is_file():
            report.add(
                Severity.ERROR, artifact.artifact_id,
                "meta.json missing; artifact cannot be reindexed if the "
                "database is lost",
            )

        for stray in artifact.path.glob("*.tmp"):
            report.add(
                Severity.WARNING, artifact.artifact_id,
                f"stray temporary file from an interrupted write: {stray.name}",
            )

        try:
            _, recorded = read_meta(artifact.path)
        except (OSError, ValueError) as exc:
            report.add(
                Severity.ERROR, artifact.artifact_id,
                f"meta.json unreadable: {exc}",
            )
            continue

        for filename, entry in recorded.items():
            path = artifact.path / filename
            if not path.is_file():
                report.add(
                    Severity.ERROR, artifact.artifact_id,
                    f"recorded file missing from disk: {filename}",
                )
                continue

            digest, size = hash_file(path)
            report.n_files_checked += 1

            if digest != entry["digest"]:
                report.add(
                    Severity.ERROR, artifact.artifact_id,
                    f"digest mismatch for {filename}: recorded "
                    f"{entry['digest'][:16]}..., found {digest[:16]}...",
                )
            elif size != entry["size_bytes"]:
                # Same digest with a different size is not physically possible;
                # it means the recorded metadata itself was corrupted.
                report.add(
                    Severity.ERROR, artifact.artifact_id,
                    f"size mismatch for {filename} despite matching digest: "
                    f"recorded {entry['size_bytes']}, found {size}",
                )

        on_disk = {
            p.name for p in artifact.path.iterdir()
            if p.is_file() and p.name not in {META_FILENAME, "README.md"}
            and not p.name.endswith(".tmp")
        }
        unrecorded = on_disk - set(recorded)
        if unrecorded:
            report.add(
                Severity.WARNING, artifact.artifact_id,
                f"{len(unrecorded)} file(s) on disk not recorded in meta.json: "
                f"{', '.join(sorted(unrecorded)[:5])}"
                + (" ..." if len(unrecorded) > 5 else ""),
            )


def verify_completeness(index: DatalakeIndex, report: VerifyReport) -> None:
    """Flag artifacts that never completed."""
    for artifact in index.list(include_partial=True, include_deprecated=True):
        if artifact.partial:
            report.add(
                Severity.WARNING, artifact.artifact_id,
                f"incomplete run started {artifact.meta.run_start}; "
                "not safe to consume downstream",
            )
        if artifact.meta.pipeline_commit is None:
            report.add(
                Severity.WARNING, artifact.artifact_id,
                "no pipeline commit recorded; this run is not reproducible "
                "from source",
            )
        elif artifact.meta.pipeline_commit.endswith("-dirty"):
            report.add(
                Severity.WARNING, artifact.artifact_id,
                "produced from a dirty working tree; the recorded commit does "
                "not fully describe the code that ran",
            )


def verify_lineage(index: DatalakeIndex, report: VerifyReport) -> None:
    """Flag lineage edges pointing at unregistered artifacts."""
    for artifact in index.list(include_partial=True, include_deprecated=True):
        for parent in index.parents(artifact.artifact_id):
            if not index.exists(parent):
                report.add(
                    Severity.WARNING, artifact.artifact_id,
                    f"input {parent!r} is not registered in this index; "
                    "lineage cannot be traced past it",
                )
                continue
            parent_artifact = index.get(parent)
            if parent_artifact.partial:
                report.add(
                    Severity.ERROR, artifact.artifact_id,
                    f"built on incomplete input {parent!r}",
                )
            elif parent_artifact.deprecated:
                report.add(
                    Severity.WARNING, artifact.artifact_id,
                    f"built on deprecated input {parent!r}",
                )


def verify(
    index: DatalakeIndex,
    *,
    check_hashes: bool = True,
    check_lineage: bool = True,
    artifact_ids: list[str] | None = None,
) -> VerifyReport:
    """Run all verification passes.  Never mutates the datalake."""
    report = VerifyReport()

    if check_hashes:
        verify_hashes(index, report, artifact_ids=artifact_ids)
    else:
        report.n_artifacts_checked = len(
            index.list(include_partial=True, include_deprecated=True)
        )

    verify_completeness(index, report)

    if check_lineage:
        verify_lineage(index, report)

    return report
