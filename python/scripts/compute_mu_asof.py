#!/usr/bin/env python3
"""Compute the asof pooled-mean headline embedding (mu_asof) into a datalake artifact.

Usage:
    python scripts/compute_mu_asof.py --delay 1M --mode expanding --pipeline-version v0.1.0
    python scripts/compute_mu_asof.py --delay 1d --mode 8W --pipeline-version v0.1.0 -v

Resolves the latest ``ravenpack_headlines`` and ``headline_embeddings``
artifacts via the datalake index (never hardcoded paths). There is no
--resume: if an artifact with the same hyperparams already exists,
``dl.latest()`` finds it downstream; to recompute, deprecate the old one
first with ``datalake deprecate <artifact_id> --reason "..."``.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

from dotenv import find_dotenv, load_dotenv

from datalake import DatalakeIndex

PIPELINE_VERSION = "v0.1.0"

log = logging.getLogger(__name__)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--env", default=".env")
    ap.add_argument("--datalake-root", help="overrides $DATALAKE_ROOT")
    ap.add_argument(
        "--delay", required=True,
        help="minimum 1 day: '1d', '2W', '3M'. mu(t) uses data strictly before t - delay.",
    )
    ap.add_argument(
        "--mode", required=True,
        help="'expanding', or a rolling window >= 1 week: '4W', '6M' (no day-granularity windows).",
    )
    ap.add_argument("--pipeline-version", default=PIPELINE_VERSION)
    ap.add_argument("--threads", type=int, default=8, help="DuckDB PRAGMA threads")
    ap.add_argument("-v", "--verbose", action="store_true")
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

    from ravenpack.headlines.mu_asof import PIPELINE_REPO, mu_asof_to_datalake

    with DatalakeIndex(root) as index:
        artifact = mu_asof_to_datalake(
            index,
            delay=args.delay,
            mode=args.mode,
            pipeline_version=args.pipeline_version,
            pipeline_repo=PIPELINE_REPO,
            threads=args.threads,
        )

    print(f"\nartifact: {artifact.artifact_id}")
    print(f"path:     {artifact.path}")
    print(f"files:    {len(artifact.file_hashes)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
