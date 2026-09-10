#!/usr/bin/env python3
"""One-shot migration of the legacy .archive headline embeddings into a datalake artifact.

Source:
    $DATALAKE_ROOT/derived/.archive/headline_embeddings/YYYY-MM.parquet   (2000-01 .. 2025-12)

These legacy parquets carry many columns (TIMESTAMP_UTC, HEADLINE, entity lists,
EMBEDDING, ...).  We keep only RP_STORY_ID + EMBEDDING and write fresh parquets
(schema.EMBEDDING_SCHEMA) into a new `headline_embeddings` datalake artifact.
Files are NOT copied -- each month is read, projected to two columns, and
re-written.

The artifact is finalised manually (not via `index.run`) so that
`RunRecord.pipeline_commit` is None -- the original producing commit is unknown.
It carries the same RavenBERT ModelCard as the fresh `embed_headlines.py`
pipeline, so `dl.latest("headline_embeddings", model="ravenbert", version="1.0")`
resolves both the migrated and the freshly-computed artifacts.

Usage:
    python scripts/migrate_archive_embeddings.py --start-year 2000 --end-year 2000   # smoke
    python scripts/migrate_archive_embeddings.py --start-year 2000 --end-year 2025   # full
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import polars as pl
from dotenv import find_dotenv, load_dotenv

from datalake import DatalakeIndex, RunMeta, RunRecord
from datalake.artifact import utc_now_iso
from datalake.meta import META_FILENAME, README_FILENAME, hash_file, write_sidecars
from ravenpack.headlines.embed import KIND, build_model_card, model_dir_sha256
from ravenpack.headlines.schema import EMBEDDING_DIM, EMBEDDING_SCHEMA

LAYER = "derived"
PIPELINE = "PhD-Narrative-Finance-Laboratory"
PIPELINE_VERSION = "v0.1.0"
MIGRATION_NOTES = (
    "Migrated from .archive/headline_embeddings. Original run: embed_headlines.py "
    "from PhD-Narrative-Finance-Laboratory repo, commit unknown."
)

log = logging.getLogger(__name__)


def _migrate_month(src: Path, out_path: Path) -> int:
    """Read one archive month, project to two columns, write atomically.

    Returns the row count written.  Eager read: one archive month decompresses to
    a few GB at most; switch to scan/sink if this ever OOMs on a high-volume month.
    """
    df = pl.read_parquet(src, columns=["RP_STORY_ID", "EMBEDDING"]).with_columns(
        pl.col("EMBEDDING").cast(pl.Array(pl.Float16, EMBEDDING_DIM))
    )
    if df.schema != EMBEDDING_SCHEMA:
        raise ValueError(
            f"{src.name}: schema after select/cast is {dict(df.schema)}, "
            f"expected {dict(EMBEDDING_SCHEMA)}"
        )
    tmp = out_path.with_suffix(".parquet.tmp")
    try:
        df.write_parquet(tmp, compression="zstd")
        tmp.replace(out_path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return df.height


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--start-year", type=int, default=2000)
    ap.add_argument("--end-year", type=int, default=2025)
    ap.add_argument("--env", default=".env")
    ap.add_argument("--datalake-root", help="overrides $DATALAKE_ROOT")
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

    archive_dir = Path(root) / "derived" / ".archive" / "headline_embeddings"
    if not archive_dir.is_dir():
        log.error("archive directory not found: %s", archive_dir)
        return 1

    log.info("hashing RavenBERT model directory %s ...", model_path)
    card = build_model_card(model_path, model_dir_sha256(model_path))

    meta = RunMeta(
        kind=KIND,
        pipeline=PIPELINE,
        pipeline_version=PIPELINE_VERSION,
        pipeline_repo=None,
        hyperparams={"start_year": args.start_year, "end_year": args.end_year},
        sources=[],
        model_card=card,
        verifier=KIND,
    )

    with DatalakeIndex(root) as index:
        out_dir = index.artifact_dir(LAYER, KIND, meta.artifact_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        log.info("migrating into %s", out_dir)

        months = [
            (y, m)
            for y in range(args.start_year, args.end_year + 1)
            for m in range(1, 13)
        ]
        n_written = 0
        for i, (y, m) in enumerate(months, 1):
            name = f"{y}-{m:02d}.parquet"
            src = archive_dir / name
            out_path = out_dir / name
            if out_path.exists():
                continue
            if not src.exists():
                log.warning("[%d/%d] %s  archive month missing, skipping", i, len(months), name)
                continue
            rows = _migrate_month(src, out_path)
            n_written += 1
            log.info("[%d/%d] %s  wrote %d stories", i, len(months), name, rows)

        present = sorted(
            p for p in out_dir.iterdir()
            if p.is_file() and p.name.endswith(".parquet")
        )
        if not present:
            log.error("no parquet written -- check archive dir and year range")
            return 1
        log.info("%d month(s) written this run; %d present total", n_written, len(present))

        # Hash with progress -- runs over GDrive FUSE and is slow.
        _EXCLUDE = frozenset({META_FILENAME, README_FILENAME})
        parquet_files = [p for p in present if p.name not in _EXCLUDE]
        log.info(
            "hashing %d files -- this may take several minutes over a network mount",
            len(parquet_files),
        )
        file_hashes: dict = {}
        for i, path in enumerate(parquet_files, 1):
            digest, size = hash_file(path)
            file_hashes[path.name] = {
                "algorithm": "blake2b", "digest": digest, "size_bytes": size,
            }
            if i % 20 == 0 or i == len(parquet_files):
                log.info("  hashed %d/%d files", i, len(parquet_files))

        # Finalise manually: single complete RunRecord, pipeline_commit unknown.
        record = RunRecord(
            run_start=utc_now_iso(),
            pipeline_version=PIPELINE_VERSION,
            pipeline_commit=None,
            run_end=utc_now_iso(),
            partial=False,
            produced=sorted(file_hashes.keys()),
            notes=MIGRATION_NOTES,
        )
        meta.runs.append(record)

        write_sidecars(out_dir, meta, file_hashes)
        index._upsert(meta, LAYER, out_dir, file_hashes)

    print(f"\nartifact: {meta.artifact_id}")
    print(f"path:     {out_dir}")
    print(f"files:    {len(file_hashes)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
