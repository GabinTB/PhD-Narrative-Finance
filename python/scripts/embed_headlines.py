#!/usr/bin/env python3
"""Embed RavenPack headlines with RavenBERT into a datalake artifact.

Usage:
    # Fresh run on TEI in fp16 (defaults) -- reads the latest ravenpack_headlines
    python scripts/embed_headlines.py --start-year 2000 --end-year 2025

    # Another engine / dtype
    python scripts/embed_headlines.py --start-year 2026 --end-year 2026 \\
        --backend local --dtype float32 --device cuda

    # Resume a partial run -- everything comes from the artifact: engine, dtype,
    # model (a remote server must still serve the recorded model), source, years
    python scripts/embed_headlines.py \\
        --resume headline_embeddings__ravenbert-1.0__v0.2.0__..._20260925

Backends: ``tei`` ($TEI_BASE_URL / $TEI_API_KEY), ``embedx`` ($EMBEDX_BASE_URL /
$EMBEDX_MODEL / $EMBEDX_API_KEY), ``local`` ($RAVENBERT_EMBEDDING_MODEL_PATH).
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

from dotenv import find_dotenv, load_dotenv

from datalake import DatalakeError, DatalakeIndex
from ravenpack.headlines.embed import PIPELINE_VERSION

PIPELINE = "PhD-Narrative-Finance"
PIPELINE_REPO = "https://github.com/GabinTB/PhD-Narrative-Finance"

_BACKEND_HELP = "inference engine: tei (default) | local | embedx"
_DTYPE_HELP = "compute dtype: float16 (default) | float32"
_DEVICE_HELP = "local backend only: torch device cuda|mps|cpu (default: auto)"


def _add_engine_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--backend", default="tei", choices=["tei", "local", "embedx"],
                   help=_BACKEND_HELP)
    p.add_argument("--dtype", default="float16", help=_DTYPE_HELP)
    p.add_argument("--device", help=_DEVICE_HELP)
    p.add_argument("--batch-size", type=int, default=None,
                   help="texts per request / forward pass (default: the backend's)")

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
    _add_engine_args(fresh)
    fresh.add_argument("--write-chunk-rows", type=int, default=200_000)
    fresh.add_argument("--log-every", type=int, default=100_000)

    resume = sub.add_parser("resume", help="resume a partial run")
    resume.add_argument("artifact_id", help="ID of the partial artifact to resume")
    resume.add_argument("--batch-size", type=int, default=None)
    resume.add_argument("--device", help="local backend only: torch device")
    resume.add_argument("--write-chunk-rows", type=int, default=200_000)
    resume.add_argument("--log-every", type=int, default=100_000)

    # Backwards-compatible: no subcommand + --start-year/--end-year = fresh run
    ap.add_argument("--start-year", type=int)
    ap.add_argument("--end-year", type=int)
    ap.add_argument("--pipeline-version", default=PIPELINE_VERSION)
    ap.add_argument("--source-artifact")
    _add_engine_args(ap)
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

    resume_id = args.resume or (args.artifact_id if args.command == "resume" else None)
    with DatalakeIndex(root) as index:
        if resume_id:
            artifact = _resume(index, resume_id, args.write_chunk_rows, args.log_every,
                               getattr(args, "batch_size", None), getattr(args, "device", None))
        elif args.start_year and args.end_year:
            if args.device and args.backend != "local":
                log.error("--device applies to the local backend only")
                return 1
            from ravenpack.headlines.embed import build_embedder

            embedder = build_embedder(args.backend, args.dtype, batch_size=args.batch_size,
                                      device=args.device)
            artifact = _fresh(
                index, embedder,
                args.start_year, args.end_year, args.pipeline_version,
                args.source_artifact, args.write_chunk_rows, args.log_every,
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
    index, embedder, start_year, end_year, pipeline_version,
    source_artifact, write_chunk_rows, log_every,
):
    from ravenpack.headlines.embed import embed_to_datalake

    return embed_to_datalake(
        index,
        pipeline_version=pipeline_version,
        embedder=embedder,
        source_artifact_id=source_artifact,
        start_year=start_year,
        end_year=end_year,
        pipeline=PIPELINE,
        pipeline_repo=PIPELINE_REPO,
        write_chunk_rows=write_chunk_rows,
        log_every=log_every,
    )


def _resume(index, artifact_id, write_chunk_rows, log_every, batch_size, device):
    """Resume a partial run from the artifact alone (engine, model, params, source)."""
    from ravenpack.headlines.embed import resume_embedding

    kwargs = {k: v for k, v in (("batch_size", batch_size), ("device", device)) if v}
    try:
        return resume_embedding(index, artifact_id, write_chunk_rows=write_chunk_rows,
                                log_every=log_every, **kwargs)
    except (DatalakeError, ValueError) as exc:          # includes IncompatibleModelError
        log.error("cannot resume %s: %s", artifact_id, exc)
        return None


if __name__ == "__main__":
    sys.exit(main())
