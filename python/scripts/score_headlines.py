#!/usr/bin/env python3
"""Score RavenPack headlines against a taxonomy_embeddings artifact.

Usage:
    # Fresh run -- taxonomy embeddings resolved by artifact id
    python scripts/score_headlines.py \\
        --taxonomy-embeddings taxonomy_embeddings__ravenbert-1.0__v0.1.0__..._20260910 \\
        --correction r2 --start 20070101 --end 20091231

    # Resume a partial run -- all params inferred from the artifact
    python scripts/score_headlines.py \\
        --resume primitive_scores__v0.1.0__..._20260910

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

    ap.add_argument("--taxonomy-embeddings", help="taxonomy_embeddings artifact id")
    ap.add_argument("--correction", choices=["raw", "r1", "r2"], default="r2")
    ap.add_argument("--start", type=int, help="e.g. 20070101")
    ap.add_argument("--end", type=int, help="e.g. 20091231")
    ap.add_argument("--no-f0", action="store_true", help="disable the F0 null-model gate")
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--trim-frac", type=float, default=0.10)
    ap.add_argument("--rel-floor", type=float, default=0.65)
    ap.add_argument("--null-delay", default="1M")
    ap.add_argument("--null-cap", type=int, default=5_000_000)
    ap.add_argument("--chunk-size", type=int, default=50_000)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--pipeline-version", default=PIPELINE_VERSION)
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

    with DatalakeIndex(root) as index:
        if args.resume:
            artifact = _resume(
                index, args.resume,
                chunk_size=args.chunk_size, threads=args.threads,
            )
        elif args.taxonomy_embeddings and args.start and args.end:
            artifact = _fresh(index, args)
        else:
            ap.print_help()
            return 1

    if artifact is None:
        return 1

    print(f"\nartifact: {artifact.artifact_id}")
    print(f"path:     {artifact.path}")
    print(f"files:    {len(artifact.file_hashes)}")
    return 0


def _fresh(index, args):
    from narrative_scoring.corrections import Correction
    from narrative_scoring.scoring import score_headlines_to_datalake

    return score_headlines_to_datalake(
        index,
        taxonomy_embeddings_artifact_id=args.taxonomy_embeddings,
        pipeline_version=args.pipeline_version,
        correction=Correction(args.correction),
        start=args.start,
        end=args.end,
        use_f0=not args.no_f0,
        alpha=args.alpha,
        trim_frac=args.trim_frac,
        rel_floor=args.rel_floor,
        null_delay=args.null_delay,
        null_cap=args.null_cap,
        chunk_size=args.chunk_size,
        pipeline=PIPELINE,
        pipeline_repo=PIPELINE_REPO,
        threads=args.threads,
    )


def _resume(index, artifact_id, *, chunk_size, threads):
    """Resume a partial run, inferring all params from the artifact metadata."""
    from datalake.artifact import utc_now_iso
    from datalake.meta import META_FILENAME, README_FILENAME, git_commit, hash_file, write_sidecars
    from narrative_scoring.corrections import Correction
    from narrative_scoring.scoring import (
        SOURCE_EMBEDDINGS_KIND,
        SOURCE_HEADLINES_KIND,
        SOURCE_MU_ASOF_KIND,
        SOURCE_TAXONOMY_EMBEDDINGS_KIND,
        load_taxonomy_embeddings,
        run_scoring_range,
    )

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

    hp = artifact.meta.hyperparams
    start = hp.get("start")
    end = hp.get("end")
    if start is None or end is None:
        log.error("artifact %s has no start/end in hyperparams.", artifact_id)
        return None

    correction = Correction(hp.get("correction", "r2"))
    use_f0 = bool(hp.get("use_f0", True))

    # Resolve source artifacts from recorded lineage, by kind.
    resolved = {}
    for source_id in artifact.meta.sources:
        try:
            src = index.get(source_id)
        except DatalakeError:
            continue
        resolved[src.kind] = src

    headlines_art = resolved.get(SOURCE_HEADLINES_KIND) or index.latest(SOURCE_HEADLINES_KIND)
    embeddings_art = resolved.get(SOURCE_EMBEDDINGS_KIND) or index.latest(SOURCE_EMBEDDINGS_KIND)
    tax_art = resolved.get(SOURCE_TAXONOMY_EMBEDDINGS_KIND)
    if tax_art is None:
        log.error("cannot resolve taxonomy_embeddings source artifact to resume from")
        return None
    mu_art = resolved.get(SOURCE_MU_ASOF_KIND)
    if mu_art is None and correction is not Correction.RAW:
        try:
            mu_art = index.latest(SOURCE_MU_ASOF_KIND)
        except DatalakeError:
            log.error(
                "correction=%s requires an mu_asof artifact but none was found", correction.value
            )
            return None

    import polars as pl

    mu_df = pl.read_parquet(mu_art.glob("*.parquet")) if mu_art is not None else None
    D_tax = load_taxonomy_embeddings(tax_art)

    out_dir = artifact.path
    already_done = len(list(out_dir.glob("*.parquet")))
    log.info(
        "resuming %s | start=%d end=%d | %d months already done | taxonomy=%s",
        artifact_id, start, end, already_done, tax_art.artifact_id,
    )

    run_scoring_range(
        out_dir, headlines_art.path, embeddings_art.path, mu_df,
        D_tax, correction, start, end,
        use_f0=use_f0,
        alpha=hp.get("alpha", 0.01), trim_frac=hp.get("trim_frac", 0.10),
        rel_floor=hp.get("rel_floor", 0.65), null_delay=hp.get("null_delay", "1M"),
        chunk_size=chunk_size, threads=threads, skip_existing=True,
    )

    n_months = len(list(out_dir.glob("*.parquet")))
    if n_months == 0:
        log.error("still no output after resume -- check source artifacts and date range")
        return None

    # Hash with progress -- runs over GDrive FUSE and takes several minutes.
    _EXCLUDE = frozenset({META_FILENAME, README_FILENAME})
    out_files = sorted(
        p for p in out_dir.iterdir()
        if p.is_file() and p.name not in _EXCLUDE and not p.name.endswith(".tmp")
    )
    log.info(
        "hashing %d files -- this may take several minutes over a network mount", len(out_files)
    )
    file_hashes: dict = {}
    for i, path in enumerate(out_files, 1):
        digest, size = hash_file(path)
        file_hashes[path.name] = {"algorithm": "blake2b", "digest": digest, "size_bytes": size}
        if i % 20 == 0 or i == len(out_files):
            log.info("  hashed %d/%d files", i, len(out_files))

    last_record = artifact.meta.runs[-1]
    last_record.partial = False
    last_record.run_end = utc_now_iso()
    last_record.pipeline_commit = git_commit()
    last_record.produced = sorted(file_hashes.keys())
    last_record.notes = (
        (last_record.notes + " " if last_record.notes else "")
        + f"Resumed after interruption. {n_months} monthly parquet(s)."
    ).strip()

    write_sidecars(out_dir, artifact.meta, file_hashes)
    index._upsert(artifact.meta, artifact.layer, out_dir, file_hashes)

    log.info("resume complete: %s (%d monthly files)", artifact.artifact_id, n_months)
    return index.get(artifact.artifact_id)


if __name__ == "__main__":
    sys.exit(main())
