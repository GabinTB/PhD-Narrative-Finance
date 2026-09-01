#!/usr/bin/env python3
"""Ingest RavenPack Annotations 1.0 zips into a datalake artifact.

Usage:
    python scripts/ingest_ravenpack.py --start-year 2000 --end-year 2025
    python scripts/ingest_ravenpack.py --start-year 2010 --end-year 2010 -v

Reads raw zips from $RAW_DATA_PATH/RavenPack/headlines_edge_v1/ and writes a
new artifact under $DATALAKE_ROOT/derived/ravenpack_headlines/.

Artifacts are immutable: re-running produces a new artifact rather than
overwriting an existing one.  To supersede an old ingest, deprecate it:

    datalake deprecate <artifact_id> --reason "..."
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from datalake import DatalakeIndex
from ravenpack.headlines.ingest import ingest_to_datalake

PIPELINE = "PhD-Narrative-Finance"
PIPELINE_REPO = "https://github.com/GabinTB/PhD-Narrative-Finance"
PIPELINE_VERSION = "v0.1.0"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--env", default=".env")
    ap.add_argument("--start-year", type=int, required=True)
    ap.add_argument("--end-year", type=int, required=True)
    ap.add_argument("--datalake-root", help="overrides $DATALAKE_ROOT")
    ap.add_argument("--raw-dir", help="overrides $RAW_DATA_PATH-derived location")
    ap.add_argument(
        "--pipeline-version", default=PIPELINE_VERSION,
        help="bump when the ingestion logic changes in a way that alters output",
    )
    ap.add_argument(
        "--raw-chunk-rows", type=int, default=2_000_000,
        help="raw CSV rows per read chunk; lower if RSS climbs during reads",
    )
    ap.add_argument("--log-every", type=int, default=100_000)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    load_dotenv(args.env)

    root = args.datalake_root or os.environ.get("DATALAKE_ROOT")
    if not root:
        logging.error("DATALAKE_ROOT must be set (in .env, environment, or --datalake-root)")
        return 1

    if args.raw_dir:
        raw_dir = Path(args.raw_dir)
    else:
        raw_data_path = os.environ.get("RAW_DATA_PATH")
        if not raw_data_path:
            logging.error("RAW_DATA_PATH must be set, or pass --raw-dir")
            return 1
        raw_dir = Path(raw_data_path) / "RavenPack" / "headlines_edge_v1.0"

    if not raw_dir.is_dir():
        logging.error("raw directory does not exist: %s", raw_dir)
        return 1

    if args.start_year > args.end_year:
        logging.error(
            "start-year (%d) is after end-year (%d)", args.start_year, args.end_year
        )
        return 1

    with DatalakeIndex(root) as index:
        artifact = ingest_to_datalake(
            index,
            raw_dir=raw_dir,
            start_year=args.start_year,
            end_year=args.end_year,
            pipeline=PIPELINE,
            pipeline_version=args.pipeline_version,
            pipeline_repo=PIPELINE_REPO,
            raw_chunk_rows=args.raw_chunk_rows,
            log_every=args.log_every,
        )

    print(f"\nartifact: {artifact.artifact_id}")
    print(f"path:     {artifact.path}")
    print(f"files:    {len(artifact.file_hashes)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
