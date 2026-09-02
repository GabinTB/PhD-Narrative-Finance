"""Datalake layer: artifact registry, provenance, and verification.

    from datalake import DatalakeIndex, ModelCard

    dl = DatalakeIndex("/mnt/storage/datalake")
    source = dl.latest("ravenpack_headlines")

    with dl.run(kind="headline_embeddings", pipeline="PhD-Narrative-Finance",
                pipeline_version="v0.1.0", sources=[source],
                verifier="headline_embeddings") as run:
        ...  # write outputs into run.out_dir
"""
from datalake.artifact import (
    Artifact,
    ModelCard,
    RunMeta,
    RunRecord,
    utc_now_iso,
)
from datalake.index import DatalakeError, DatalakeIndex, RunHandle
from datalake.meta import git_commit, hash_file, read_meta, write_sidecars
from datalake.verify import (
    Finding,
    Severity,
    VerifyReport,
    discover_verifiers,
    verify,
)

__all__ = [
    "Artifact",
    "DatalakeError",
    "DatalakeIndex",
    "Finding",
    "ModelCard",
    "RunHandle",
    "RunMeta",
    "RunRecord",
    "Severity",
    "VerifyReport",
    "discover_verifiers",
    "git_commit",
    "hash_file",
    "read_meta",
    "utc_now_iso",
    "verify",
    "write_sidecars",
]
