#!/usr/bin/env python3
"""Run HFT analytics for a MIC + product + date range.

Usage:
    uv run python scripts/run_hft_analytics.py \\
        --mic XEUR --product FESX \\
        --from 20240101 --to 20240131 \\
        --markout-period 1000000 \\
        --pipeline-version v0.1.0
"""
from __future__ import annotations

import argparse
import logging
import sys

import deutsche_boerse  # noqa: F401 -- sets DATA_BASE_DIR before any dbg_cdm import
from deutsche_boerse.analytics.hft_event_detection import run_hft_analytics
from datalake import DatalakeIndex

PIPELINE = "PhD-Narrative-Finance"
PIPELINE_REPO = "https://github.com/GabinTB/PhD-Narrative-Finance"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--mic", required=True, help="Market identifier code (e.g. XEUR)")
    ap.add_argument("--product", required=True, help="Product name (e.g. FESX)")
    ap.add_argument("--from", dest="from_ccyymmdd", type=int, required=True)
    ap.add_argument("--to", dest="to_ccyymmdd", type=int, required=True)
    ap.add_argument(
        "--markout-period", type=int, default=1_000_000,
        help="Markout window in nanoseconds (default: 1ms)",
    )
    ap.add_argument("--epsilon", type=int, default=10)
    ap.add_argument("--pipeline-version", default="v0.1.0")
    ap.add_argument("--skip-missing", action="store_true", default=True)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    import os
    datalake_root = os.environ.get("DATALAKE_ROOT")
    if not datalake_root:
        logging.error("DATALAKE_ROOT must be set")
        return 1

    with DatalakeIndex(datalake_root) as index:
        artifact = run_hft_analytics(
            index=index,
            mic=args.mic,
            product=args.product,
            from_ccyymmdd=args.from_ccyymmdd,
            to_ccyymmdd=args.to_ccyymmdd,
            markout_period=args.markout_period,
            pipeline=PIPELINE,
            pipeline_version=args.pipeline_version,
            pipeline_repo=PIPELINE_REPO,
            epsilon=args.epsilon,
            skip_missing=args.skip_missing,
        )

    print(f"\nartifact: {artifact.artifact_id}")
    print(f"path:     {artifact.path}")
    print(f"files:    {len(artifact.file_hashes)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
