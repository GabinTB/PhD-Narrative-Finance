#!/usr/bin/env python3
"""Embed a taxonomy version's primitive descriptions into a datalake artifact.

Usage:
    python scripts/embed_taxonomy.py \\
        --taxonomy-version v4.3 --family evergreen --paraphrase-style headlined \\
        --pooling centroid --pipeline-version v0.1.0

The RavenBERT weights directory is taken from $RAVENBERT_EMBEDDING_MODEL_PATH
and the taxonomy root from $RAW_DATA_PATH/Evergreen_Taxonomy (evergreen) or
$RAW_DATA_PATH/RavenPack_Taxonomy (ravenpack) -- no CLI flag for either.

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

from datalake import DatalakeIndex

PIPELINE_VERSION = "v0.1.0"

log = logging.getLogger(__name__)

_TAXONOMY_ROOT_DIRNAME = {
    "evergreen": "Evergreen_Taxonomy",
    "ravenpack": "RavenPack_Taxonomy",
}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--env", default=".env")
    ap.add_argument("--datalake-root", help="overrides $DATALAKE_ROOT")
    ap.add_argument("--taxonomy-version", required=True, help='e.g. "v4.3"')
    ap.add_argument("--family", choices=["evergreen", "ravenpack"], default="evergreen")
    ap.add_argument("--paraphrase-style", choices=["pure", "headlined"], default="headlined")
    ap.add_argument("--pooling", choices=["centroid", "max", "median"], default="centroid")
    ap.add_argument("--no-garbage", action="store_true", help="skip the garbage catcher")
    ap.add_argument("--device", help="torch device: cuda|mps|cpu (default: auto)")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--pipeline-version", default=PIPELINE_VERSION)
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

    model_path_str = os.environ.get("RAVENBERT_EMBEDDING_MODEL_PATH")
    if not model_path_str:
        log.error("RAVENBERT_EMBEDDING_MODEL_PATH must be set (in .env or environment)")
        return 1
    model_path = Path(model_path_str)
    if not model_path.is_dir():
        log.error("RavenBERT model directory does not exist: %s", model_path)
        return 1

    raw_data_path_str = os.environ.get("RAW_DATA_PATH")
    if not raw_data_path_str:
        log.error("RAW_DATA_PATH must be set (in .env or environment)")
        return 1
    taxonomy_root = Path(raw_data_path_str) / _TAXONOMY_ROOT_DIRNAME[args.family]
    if not taxonomy_root.is_dir():
        log.error("taxonomy root does not exist: %s", taxonomy_root)
        return 1

    from narrative_scoring.descriptions import PoolingMode, embed_taxonomy_to_datalake

    with DatalakeIndex(root) as index:
        artifact = embed_taxonomy_to_datalake(
            index,
            raw_data_path=taxonomy_root,
            model_path=model_path,
            taxonomy_version=args.taxonomy_version,
            pipeline_version=args.pipeline_version,
            family=args.family,
            paraphrase_style=args.paraphrase_style,
            pooling=PoolingMode(args.pooling),
            include_garbage=not args.no_garbage,
            device=args.device,
            batch_size=args.batch_size,
        )

    print(f"\nartifact: {artifact.artifact_id}")
    print(f"path:     {artifact.path}")
    print(f"files:    {len(artifact.file_hashes)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
