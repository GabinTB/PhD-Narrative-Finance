#!/usr/bin/env python3
"""Register a universe master list as a datalake artifact, enriched via WRDS/Deutsche Boerse.

Usage:
    python scripts/ingest_universe.py --file your_universe.csv [--name sp500_2020]
    python scripts/ingest_universe.py --file your_universe.csv --name sp500_2020 \\
        --dbga-mic XETR --dbga-market-segment-id 688

`your_universe.{csv,parquet}` has mandatory columns `snapshot_date,name,ticker`
plus at least one of `isin`/`cusip`; every other identifier/classification
column (cusip, sedol, cik, figi, gvkey, ciq_secid, dbga_secid, country_name,
country_iso, region, gics_*) is optional and, by default, backfilled here:

- WRDS CapitalIQ (isin/cusip/cik/gvkey/ciq_secid/country_*/region/gics_*):
  runs whenever WRDS credentials are available (`--resolve-wrds`, default on).
- WRDS/LSEG (sedol): runs alongside CapitalIQ unless `--no-resolve-sedol`.
- Deutsche Boerse (dbga_secid): only runs if BOTH `--dbga-mic` and
  `--dbga-market-segment-id` are given -- there's no way to derive which
  market/segment to scan from an ISIN alone (see
  `deutsche_boerse.identifiers`; this does a full market-day scan, one API
  call per distinct `snapshot_date` in the universe, not one per row).

`--name` is an optional human-readable tag (e.g. "sp500_2020") for finding
this registration later by name -- see `universe.load_universe_by_name`.
Names are not enforced unique; re-registering the same name creates a new
artifact and lookups return the most recent one.

By default, registration refuses (exit 1) if any entry is still missing one
of `isin`/`cusip`/`cik`/`gvkey`/`dbga_secid` after enrichment (`sedol` stays
optional) -- pass `--allow-incomplete` to register anyway.

Registers into the datalake's `raw` layer (vintaged reference input data, not
a pipeline output). Per-source connectors (`wrds_client.linking`,
`deutsche_boerse.identifiers`) then resolve this universe to their own native
identifiers -- see `scripts/ingest_optionmetrics.py --universe-artifact`/
`--universe-name`.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from datalake import DatalakeIndex
from universe.enrich import enrich_universe
from universe.ingest import register_universe_entries
from universe.schema import entries_from_rows, find_incomplete, load_universe_rows

PIPELINE = "PhD-Narrative-Finance"
PIPELINE_REPO = "https://github.com/GabinTB/PhD-Narrative-Finance"
PIPELINE_VERSION = "v0.1.0"

log = logging.getLogger(__name__)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--file", required=True, type=Path, help="CSV or parquet with the universe schema"
    )
    ap.add_argument("--name", default=None, help="optional human-readable tag, e.g. sp500_2020")
    ap.add_argument("--resolve-wrds", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--resolve-sedol", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--dbga-mic", default=None)
    ap.add_argument("--dbga-market-segment-id", type=int, default=None)
    ap.add_argument("--allow-incomplete", action="store_true")
    ap.add_argument("--pipeline-version", default=PIPELINE_VERSION)
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

    rows = load_universe_rows(args.file)
    if not rows:
        log.error("%s contains no rows", args.file)
        return 1
    log.info("loaded %d rows from %s", len(rows), args.file)

    wrds_client_cm = None
    if args.resolve_wrds:
        from wrds_client import WRDSClient

        wrds_client_cm = WRDSClient.from_env()

    try:
        client = wrds_client_cm.__enter__() if wrds_client_cm else None

        rows = enrich_universe(
            rows,
            wrds_client=client,
            resolve_sedol=args.resolve_sedol,
            dbga_mic=args.dbga_mic,
            dbga_market_segment_id=args.dbga_market_segment_id,
        )

        entries = entries_from_rows(rows, source=str(args.file))
        if not entries:
            log.error("%s contains no usable universe entries", args.file)
            return 1
        log.info("%d/%d rows passed validation after enrichment", len(entries), len(rows))

        incomplete = find_incomplete(entries)
        if incomplete:
            log.warning("%d entr(y/ies) still incomplete after enrichment:", len(incomplete))
            for i, fields in incomplete:
                log.warning("  row %d (%s): missing %s", i, entries[i].ticker, fields)
            if not args.allow_incomplete:
                log.error(
                    "refusing to register incomplete entries -- pass --allow-incomplete "
                    "to register anyway."
                )
                return 1

        with DatalakeIndex(root) as index:
            artifact = register_universe_entries(
                index,
                entries,
                source_label=str(args.file),
                name=args.name,
                allow_incomplete=args.allow_incomplete,
                pipeline=PIPELINE,
                pipeline_version=args.pipeline_version,
                pipeline_repo=PIPELINE_REPO,
            )
    finally:
        if wrds_client_cm:
            wrds_client_cm.__exit__(*sys.exc_info())

    print(f"\nuniverse artifact: {artifact.artifact_id}")
    print(f"name:              {args.name or '(unnamed)'}")
    print(f"layer:             {artifact.layer}")
    print(f"entries:           {artifact.meta.hyperparams['n_entries']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
