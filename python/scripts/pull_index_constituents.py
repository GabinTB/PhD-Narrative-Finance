#!/usr/bin/env python3
"""Index constituents from WRDS as universe snapshot files (`wrds_client.indices`).

    python scripts/pull_index_constituents.py search "STOXX 600"
    python scripts/pull_index_constituents.py pull --source compustat --id 150376 \\
        --index STOXX600 --from 2006-01-01 --to 2026-09-01 --out DIR

`search` lists matching indices in every source (Compustat North America and global, CRSP),
with their id and whether constituents exist; a source this WRDS account is not licensed for
is reported, not fatal. `pull` writes `{INDEX}_constituents-{first}_to_{last}.parquet`
(first business day of each month) into `--out`, which `scripts/ingest_universe.py
--index-dir DIR --index INDEX` then enriches and registers. Nothing is written to the
datalake here.
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from wrds_client import WRDSClient
from wrds_client.indices import (
    SOURCES,
    IndexNotAvailable,
    build_constituents,
    search_indices,
    write_constituents,
)

log = logging.getLogger(__name__)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    s = sub.add_parser("search", help="find an index id by name")
    s.add_argument("text")
    p = sub.add_parser("pull", help="write one index's constituents file")
    p.add_argument("--source", required=True, choices=SOURCES)
    p.add_argument("--id", required=True, help="index id from `search` (gvkeyx or indno)")
    p.add_argument("--index", required=True, help="universe name used in the file name")
    p.add_argument("--from", dest="start", required=True, type=date.fromisoformat)
    p.add_argument("--to", dest="end", required=True, type=date.fromisoformat)
    p.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    load_dotenv(find_dotenv(usecwd=True))
    with WRDSClient.from_env() as client:
        if args.command == "search":
            found = search_indices(client, args.text)
            print(found.to_string(index=False) if not found.empty else "no match")
            return 0
        try:
            frame = build_constituents(client, args.source, args.id, args.start, args.end)
        except IndexNotAvailable as exc:
            log.error("%s", exc)
            return 1
    path = write_constituents(frame, args.out, args.index)
    print(f"{path}\n{len(frame)} rows, {frame['snapshot_date'].nunique()} snapshots")
    return 0


if __name__ == "__main__":
    sys.exit(main())
