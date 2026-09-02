#!/usr/bin/env python3
"""Ingest RavenPack Annotations 1.0 zips into a datalake artifact.

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
        if args.resume:
            artifact = _resume(index, args.resume, raw_dir, args.raw_chunk_rows, args.log_every)
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
    """Resume a partial run, inferring all params from the artifact metadata."""
    from datalake.artifact import utc_now_iso
    from datalake.meta import git_commit, hash_directory, write_sidecars
    from ravenpack.headlines.ingest import ingest_range

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

    # Infer everything from recorded metadata.
    recorded = artifact.meta.hyperparams
    start_year = recorded.get("start_year")
    end_year = recorded.get("end_year")

    if start_year is None or end_year is None:
        log.error(
            "artifact %s has no start_year/end_year in hyperparams -- "
            "cannot infer range. Pass --start-year and --end-year explicitly.",
            artifact_id,
        )
        return None

    # Raw dir: use override if provided, else reconstruct from recorded notes
    # or fall back to environment.
    if raw_dir_override is not None and raw_dir_override.is_dir():
        raw_dir = raw_dir_override
        log.info("using raw_dir from CLI: %s", raw_dir)
    else:
        raw_data_path = os.environ.get("RAW_DATA_PATH")
        if not raw_data_path:
            log.error("RAW_DATA_PATH not set and no --raw-dir provided")
            return None
        raw_dir = Path(raw_data_path) / "RavenPack" / "headlines_edge_v1.0"
        log.info("using raw_dir from environment: %s", raw_dir)

    if not raw_dir.is_dir():
        log.error("raw directory does not exist: %s", raw_dir)
        return None

    out_dir = artifact.path
    already_done = len(list(out_dir.glob("*.parquet")))
    expected_months = (end_year - start_year + 1) * 12
    remaining = expected_months - already_done

    log.info(
        "resuming %s | start_year=%d end_year=%d | "
        "%d/%d months done, %d remaining",
        artifact_id, start_year, end_year,
        already_done, expected_months, remaining,
    )

    ingest_range(
        raw_dir=raw_dir,
        out_dir=out_dir,
        start_year=start_year,
        end_year=end_year,
        raw_chunk_rows=raw_chunk_rows,
        log_every=log_every,
        overwrite=False,
    )

    n_months = len(list(out_dir.glob("*.parquet")))
    if n_months == 0:
        log.error("still no output after resume -- check raw_dir and year range")
        return None

    if n_months < expected_months:
        log.warning(
            "%d/%d months present -- some months may be missing from raw zips",
            n_months, expected_months,
        )

    # Finalise: hash outputs, clear partial, update sidecars and index.
    file_hashes = hash_directory(out_dir, pattern="*.parquet")
    artifact.meta.partial = False
    artifact.meta.run_end = utc_now_iso()
    artifact.meta.pipeline_commit = git_commit()
    artifact.meta.notes = (
        (artifact.meta.notes + " " if artifact.meta.notes else "")
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