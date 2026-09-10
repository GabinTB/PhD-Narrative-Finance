#!/usr/bin/env python3
"""Embed RavenPack headlines with RavenBERT into a datalake artifact.

Usage:
    # Fresh run -- reads the latest ravenpack_headlines artifact
    python scripts/embed_headlines.py --start-year 2000 --end-year 2025

    # Resume a partial run -- all params inferred from the artifact
    python scripts/embed_headlines.py \\
        --resume headline_embeddings__ravenbert-1.0__v0.1.0__..._20260909

The RavenBERT weights directory is always taken from
$RAVENBERT_EMBEDDING_MODEL_PATH (no CLI flag).  Resuming writes into the existing
artifact directory, skipping months already written, and finalises the artifact
on completion.

Artifacts are immutable once complete.  To supersede a finished artifact:

    datalake deprecate <artifact_id> --reason "..."
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from datalake import DatalakeError, DatalakeIndex

PIPELINE = "PhD-Narrative-Finance"
PIPELINE_REPO = "https://github.com/GabinTB/PhD-Narrative-Finance"
PIPELINE_VERSION = "v0.1.0"

log = logging.getLogger(__name__)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--env", default=".env")
    ap.add_argument("--datalake-root", help="overrides $DATALAKE_ROOT")
    ap.add_argument("-v", "--verbose", action="store_true")

    sub = ap.add_subparsers(dest="command", required=False)

    fresh = sub.add_parser("run", help="start a fresh embedding run")
    fresh.add_argument("--start-year", type=int)
    fresh.add_argument("--end-year", type=int)
    fresh.add_argument("--pipeline-version", default=PIPELINE_VERSION)
    fresh.add_argument("--source-artifact", help="ravenpack_headlines artifact id")
    fresh.add_argument("--device", help="torch device: cuda|mps|cpu (default: auto)")
    fresh.add_argument("--batch-size", type=int, default=128)
    fresh.add_argument("--write-chunk-rows", type=int, default=200_000)
    fresh.add_argument("--log-every", type=int, default=100_000)

    resume = sub.add_parser("resume", help="resume a partial run")
    resume.add_argument("artifact_id", help="ID of the partial artifact to resume")
    resume.add_argument("--device", help="torch device: cuda|mps|cpu (default: auto)")
    resume.add_argument("--batch-size", type=int, default=128)
    resume.add_argument("--write-chunk-rows", type=int, default=200_000)
    resume.add_argument("--log-every", type=int, default=100_000)

    # Backwards-compatible: no subcommand + --start-year/--end-year = fresh run
    ap.add_argument("--start-year", type=int)
    ap.add_argument("--end-year", type=int)
    ap.add_argument("--pipeline-version", default=PIPELINE_VERSION)
    ap.add_argument("--source-artifact")
    ap.add_argument("--device")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--write-chunk-rows", type=int, default=200_000)
    ap.add_argument("--log-every", type=int, default=100_000)
    ap.add_argument("--resume", metavar="ARTIFACT_ID")

    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    load_dotenv(find_dotenv(usecwd=True))
    if args.env != ".env":
        load_dotenv(args.env, override=True)

    root = args.datalake_root or os.environ.get("DATALAKE_ROOT")
    if not root:
        log.error("DATALAKE_ROOT must be set (in .env, environment, or --datalake-root)")
        return 1

    model_path_str = os.environ.get("RAVENBERT_EMBEDDING_MODEL_PATH")
    if not model_path_str:
        log.error("RAVENBERT_EMBEDDING_MODEL_PATH must be set (in .env or environment)")
        return 1
    model_path = Path(model_path_str)
    if not model_path.is_dir():
        log.error("RavenBERT model directory does not exist: %s", model_path)
        return 1

    with DatalakeIndex(root) as index:
        if args.resume:
            artifact = _resume(
                index, args.resume, model_path,
                args.device, args.batch_size, args.write_chunk_rows, args.log_every,
            )
        elif args.start_year and args.end_year:
            artifact = _fresh(
                index, model_path,
                args.start_year, args.end_year, args.pipeline_version,
                args.source_artifact, args.device,
                args.batch_size, args.write_chunk_rows, args.log_every,
            )
        else:
            ap.print_help()
            return 1

    if artifact is None:
        return 1

    print(f"\nartifact: {artifact.artifact_id}")
    print(f"path:     {artifact.path}")
    print(f"files:    {len(artifact.file_hashes)}")
    return 0


def _fresh(
    index, model_path, start_year, end_year, pipeline_version,
    source_artifact, device, batch_size, write_chunk_rows, log_every,
):
    from ravenpack.headlines.embed import embed_to_datalake

    return embed_to_datalake(
        index,
        model_path=model_path,
        pipeline_version=pipeline_version,
        source_artifact_id=source_artifact,
        start_year=start_year,
        end_year=end_year,
        device=device,
        pipeline=PIPELINE,
        pipeline_repo=PIPELINE_REPO,
        batch_size=batch_size,
        write_chunk_rows=write_chunk_rows,
        log_every=log_every,
    )


def _resume(index, artifact_id, model_path, device, batch_size, write_chunk_rows, log_every):
    """Resume a partial run, inferring all params from the artifact metadata."""
    from datalake.artifact import utc_now_iso
    from datalake.meta import META_FILENAME, README_FILENAME, git_commit, hash_file, write_sidecars
    from ravenpack.headlines.embed import SOURCE_KIND, _load_model, embed_range

    try:
        artifact = index.get(artifact_id)
    except DatalakeError:
        log.error("artifact not found: %s", artifact_id)
        return None

    if not artifact.partial:
        log.error(
            "artifact %s is already complete. "
            "Start a new run or deprecate the old one first.",
            artifact_id,
        )
        return None

    recorded = artifact.meta.hyperparams
    start_year = recorded.get("start_year")
    end_year = recorded.get("end_year")
    if start_year is None or end_year is None:
        log.error("artifact %s has no start_year/end_year in hyperparams.", artifact_id)
        return None

    # Resolve the source (structured headlines) artifact directory.
    source_id = recorded.get("source_artifact")
    try:
        source = index.get(source_id) if source_id else index.latest(SOURCE_KIND)
    except DatalakeError:
        log.error("cannot resolve source artifact (%s) to resume from", source_id)
        return None
    source_dir = source.path

    out_dir = artifact.path
    already_done = len(list(out_dir.glob("*.parquet")))
    expected_months = (end_year - start_year + 1) * 12
    log.info(
        "resuming %s | start_year=%d end_year=%d | %d/%d months done | source=%s",
        artifact_id, start_year, end_year, already_done, expected_months,
        source.artifact_id,
    )

    model = _load_model(model_path, device=device)
    embed_range(
        model,
        source_dir=source_dir,
        out_dir=out_dir,
        start_year=start_year,
        end_year=end_year,
        batch_size=batch_size,
        write_chunk_rows=write_chunk_rows,
        log_every=log_every,
        overwrite=False,
    )

    n_months = len(list(out_dir.glob("*.parquet")))
    if n_months == 0:
        log.error("still no output after resume -- check source dir and year range")
        return None
    if n_months < expected_months:
        log.warning(
            "%d/%d months present -- some source months may be missing",
            n_months, expected_months,
        )

    # Hash with progress -- runs over GDrive FUSE and takes several minutes.
    _EXCLUDE = frozenset({META_FILENAME, README_FILENAME})
    parquet_files = sorted(
        p for p in out_dir.iterdir()
        if p.is_file() and p.name.endswith(".parquet") and p.name not in _EXCLUDE
    )
    log.info(
        "hashing %d files -- this may take several minutes over a network mount",
        len(parquet_files),
    )
    file_hashes: dict = {}
    for i, path in enumerate(parquet_files, 1):
        digest, size = hash_file(path)
        file_hashes[path.name] = {"algorithm": "blake2b", "digest": digest, "size_bytes": size}
        if i % 20 == 0 or i == len(parquet_files):
            log.info("  hashed %d/%d files", i, len(parquet_files))

    # Update the last RunRecord directly -- partial, run_end, pipeline_commit,
    # produced are fields on RunRecord, not RunMeta (which exposes them as
    # computed properties).  Setting them on RunMeta would be silently ignored.
    last_record = artifact.meta.runs[-1]
    last_record.partial = False
    last_record.run_end = utc_now_iso()
    last_record.pipeline_commit = git_commit()
    last_record.produced = sorted(file_hashes.keys())
    last_record.notes = (
        (last_record.notes + " " if last_record.notes else "")
        + f"Resumed after interruption. {n_months}/{expected_months} monthly files."
    ).strip()

    write_sidecars(out_dir, artifact.meta, file_hashes)
    index._upsert(artifact.meta, artifact.layer, out_dir, file_hashes)

    log.info(
        "resume complete: %s (%d/%d files)",
        artifact.artifact_id, n_months, expected_months,
    )
    return index.get(artifact.artifact_id)


if __name__ == "__main__":
    sys.exit(main())
