"""Datalake layer: artifact registry, provenance, and verification.

    from datalake import DatalakeIndex, ModelCard

    dl = DatalakeIndex("/mnt/storage/datalake")
    source = dl.latest("ravenpack_headlines")

    with dl.run(kind="headline_embeddings", pipeline="PhD-Narrative-Finance",
                pipeline_version="v0.1.0", sources=[source]) as run:
        ...  # write outputs into run.out_dir
"""
from datalake.artifact import Artifact, ModelCard, RunMeta, utc_now_iso
from datalake.index import DatalakeError, DatalakeIndex, RunHandle
from datalake.meta import git_commit, hash_file, read_meta, write_sidecars
from datalake.verify import Severity, VerifyReport, verify

__all__ = [
    "Artifact",
    "DatalakeError",
    "DatalakeIndex",
    "ModelCard",
    "RunHandle",
    "RunMeta",
    "Severity",
    "VerifyReport",
    "git_commit",
    "hash_file",
    "read_meta",
    "utc_now_iso",
    "verify",
    "write_sidecars",
]
