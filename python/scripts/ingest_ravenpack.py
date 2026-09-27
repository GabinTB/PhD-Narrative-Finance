#!/usr/bin/env python3
"""Ingest RavenPack Annotations 1.0 zips into a datalake artifact.

Deprecated entry point: ``jobs start ravenpack_headlines ...`` / ``jobs resume <id>``
do the same through the shared Job lifecycle (status, pause, lock, job.log).

Usage:
    # Fresh run
    python scripts/ingest_ravenpack.py --start-year 2000 --end-year 2025

    # Resume a partial run -- all params inferred from the artifact
    python scripts/ingest_ravenpack.py \\
        --resume ravenpack_headlines__v0.1.0__end_year2025_start_year2000__20260901

Resuming writes into the existing artifact directory, skipping months already
written, and finalises the artifact on completion.

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
from ravenpack.headlines.ingest import PIPELINE_VERSION

PIPELINE = "PhD-Narrative-Finance"
PIPELINE_REPO = "https://github.com/GabinTB/PhD-Narrative-Finance"

log = logging.getLogger(__name__)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--env", default=".env")
    ap.add_argument("--datalake-root", help="overrides $DATALAKE_ROOT")
    ap.add_argument("--raw-dir", help="overrides $RAW_DATA_PATH-derived location")
    ap.add_argument("-v", "--verbose", action="store_true")

    sub = ap.add_subparsers(dest="command", required=False)

    # Fresh run subcommand
    fresh = sub.add_parser("run", help="start a fresh ingest run")
    fresh.add_argument("--start-year", type=int, required=True)
    fresh.add_argument("--end-year", type=int, required=True)
    fresh.add_argument("--pipeline-version", default=PIPELINE_VERSION)
    fresh.add_argument("--raw-chunk-rows", type=int, default=2_000_000)
    fresh.add_argument("--log-every", type=int, default=100_000)

    # Resume subcommand
    resume = sub.add_parser("resume", help="resume a partial run")
    resume.add_argument("artifact_id", help="ID of the partial artifact to resume")
    resume.add_argument("--raw-chunk-rows", type=int, default=2_000_000)
    resume.add_argument("--log-every", type=int, default=100_000)

    # Backwards-compatible: no subcommand + --start-year/--end-year = fresh run
    ap.add_argument("--start-year", type=int)
    ap.add_argument("--end-year", type=int)
    ap.add_argument("--pipeline-version", default=PIPELINE_VERSION)
    ap.add_argument("--raw-chunk-rows", type=int, default=2_000_000)
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

    # Resolve raw_dir now -- resume will use the recorded one as fallback
    raw_dir: Path | None = None
    if args.raw_dir:
        raw_dir = Path(args.raw_dir)
    else:
        raw_data_path = os.environ.get("RAW_DATA_PATH")
        if raw_data_path:
            raw_dir = Path(raw_data_path) / "RavenPack" / "headlines_edge_v1.0"

    with DatalakeIndex(root) as index:
        resume_id = args.resume or (args.artifact_id if args.command == "resume" else None)
        if resume_id:
            artifact = _resume(index, resume_id, raw_dir, args.raw_chunk_rows, args.log_every)
        elif args.start_year and args.end_year:
            if raw_dir is None or not raw_dir.is_dir():
                log.error("raw directory does not exist: %s", raw_dir)
                return 1
            artifact = _fresh(
                index, raw_dir,
                args.start_year, args.end_year,
                args.pipeline_version, args.raw_chunk_rows, args.log_every,
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


def _fresh(index, raw_dir, start_year, end_year, pipeline_version, raw_chunk_rows, log_every):
    from ravenpack.headlines.ingest import ingest_to_datalake
    return ingest_to_datalake(
        index,
        raw_dir=raw_dir,
        start_year=start_year,
        end_year=end_year,
        pipeline=PIPELINE,
        pipeline_version=pipeline_version,
        pipeline_repo=PIPELINE_REPO,
        raw_chunk_rows=raw_chunk_rows,
        log_every=log_every,
    )


def _resume(index, artifact_id, raw_dir_override, raw_chunk_rows, log_every):
    """Resume a partial run through the Job lifecycle (same as ``jobs resume``)."""
    from datalake.jobs import JobRunner

    raw_dir = raw_dir_override if raw_dir_override is not None and raw_dir_override.is_dir() \
        else None
    try:
        return JobRunner(index, allow_dirty=True).resume(
            artifact_id, raw_dir=raw_dir, raw_chunk_rows=raw_chunk_rows, log_every=log_every)
    except (DatalakeError, ValueError) as exc:
        log.error("cannot resume %s: %s", artifact_id, exc)
        return None

if __name__ == "__main__":
    sys.exit(main())