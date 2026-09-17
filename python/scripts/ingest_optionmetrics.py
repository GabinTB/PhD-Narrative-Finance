#!/usr/bin/env python3
"""Ingest OptionMetrics option prices for a universe into datalake artifacts.

Usage:
    python scripts/ingest_optionmetrics.py \\
        --start-date 2015-01-01 --end-date 2023-12-31 \\
        [--universe-name sp500_2020 | --universe-artifact <id>]   # default: latest universe \\
        [--columns secid date cp_flag strike_price impl_volatility] \\
        [--library optionm] \\
        [--skip-existing / --no-skip-existing] \\
        [--allow-ambiguous]

The universe (ISIN/CIK/ticker/name) must already be registered via
`scripts/ingest_universe.py` -- this script reads it from the datalake, it
does not take a CSV/parquet path directly, so every ingestion run's
provenance ties back to one immutable, versioned universe snapshot.
`--universe-name`/`--universe-artifact` are mutually exclusive; with neither,
the most recently registered universe (of any name) is used.

Identifiers are resolved to OptionMetrics secid(s) first (isin/cik -> gvkey ->
permno -> secid, each hop time-bounded -- see `wrds_client.linking`), and that
resolution is itself persisted as a `wrds_secid_resolution` artifact (sourced
from the universe artifact) so it's independently inspectable and so the
option-price artifacts it produces carry `sources` lineage back through it.
By default the run refuses to proceed (exit 1) if any input resolved
ambiguously (matched >1 gvkey/permno/secid for the same period) -- pass
--allow-ambiguous to proceed anyway using every match found.

Artifacts are immutable; one per contiguous (date range, secid set) batch
computed by `wrds_client.optionmetrics.universe.plan_secid_batches`, not one
per universe or one per security.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date

import pandas as pd
from dotenv import find_dotenv, load_dotenv

from datalake import DatalakeError, DatalakeIndex
from universe import find_universe_by_name, load_universe_entries
from wrds_client import WRDSClient
from wrds_client.linking import ingest_secid_resolution, resolve_universe_secids
from wrds_client.optionmetrics import ingest_universe_option_prices

PIPELINE = "PhD-Narrative-Finance"
PIPELINE_REPO = "https://github.com/GabinTB/PhD-Narrative-Finance"
PIPELINE_VERSION = "v0.1.0"

#: Flags that mean resolution could not determine a single answer for some
#: part of the universe -- refuse to silently guess which one is right.
_FATAL_FLAGS = frozenset(
    {"ambiguous_gvkey", "multiple_permno_for_window", "multiple_secid_for_window"}
)
_WARN_FLAGS = frozenset({"unresolved", "non_preferred_link"})

log = logging.getLogger(__name__)


def _report_flagged_rows(resolved: pd.DataFrame, flags: frozenset[str], title: str) -> int:
    matching = resolved[resolved["flags"].apply(lambda fs: any(f in flags for f in fs))]
    if len(matching):
        log.warning("%s (%d rows):", title, len(matching))
        cols = ["ticker", "name", "isin", "cik", "gvkey", "permno", "secid", "flags"]
        log.warning("\n%s", matching[cols].to_string(index=False))
    return len(matching)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    universe_selector = ap.add_mutually_exclusive_group()
    universe_selector.add_argument("--universe-artifact", default=None)
    universe_selector.add_argument(
        "--universe-name", default=None, help="e.g. sp500_2020 (see scripts/ingest_universe.py)"
    )
    ap.add_argument("--start-date", required=True, type=date.fromisoformat)
    ap.add_argument("--end-date", required=True, type=date.fromisoformat)
    ap.add_argument("--columns", nargs="+", default=None, help="default: all option_price columns")
    ap.add_argument("--library", default="optionm")
    ap.add_argument("--pipeline-version", default=PIPELINE_VERSION)
    ap.add_argument("--env", default=".env")
    ap.add_argument("--datalake-root", help="overrides $DATALAKE_ROOT")
    ap.add_argument("--allow-ambiguous", action="store_true")
    ap.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
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

    with WRDSClient.from_env() as client, DatalakeIndex(root) as index:
        try:
            if args.universe_name:
                universe_artifact = find_universe_by_name(index, args.universe_name)
            elif args.universe_artifact:
                universe_artifact = index.get(args.universe_artifact)
            else:
                universe_artifact = index.latest("universe")
        except (DatalakeError, ValueError) as exc:
            log.error(
                "%s -- register one first with scripts/ingest_universe.py --file <your_universe>",
                exc,
            )
            return 1

        universe = load_universe_entries(index, universe_artifact.artifact_id)
        if not universe:
            log.error("universe artifact %s has no usable entries", universe_artifact.artifact_id)
            return 1
        log.info(
            "loaded %d universe entries from %s", len(universe), universe_artifact.artifact_id
        )

        resolved = resolve_universe_secids(client, universe, args.start_date, args.end_date)

        n_fatal = _report_flagged_rows(resolved, _FATAL_FLAGS, "Ambiguous/conflicting resolutions")
        _report_flagged_rows(resolved, _WARN_FLAGS, "Unresolved or non-preferred-link rows")

        if n_fatal and not args.allow_ambiguous:
            log.error(
                "%d ambiguous/conflicting row(s) -- refusing to guess. "
                "Re-run with --allow-ambiguous to proceed using every match found.",
                n_fatal,
            )
            return 1

        resolution_artifact = ingest_secid_resolution(
            index,
            client,
            universe,
            args.start_date,
            args.end_date,
            pipeline=PIPELINE,
            pipeline_version=args.pipeline_version,
            pipeline_repo=PIPELINE_REPO,
            sources=[universe_artifact],
        )
        log.info("resolution artifact: %s", resolution_artifact.artifact_id)

        artifacts = ingest_universe_option_prices(
            index,
            client,
            resolved,
            args.start_date,
            args.end_date,
            pipeline=PIPELINE,
            pipeline_version=args.pipeline_version,
            columns=args.columns,
            library=args.library,
            sources=[resolution_artifact],
            pipeline_repo=PIPELINE_REPO,
            skip_existing=args.skip_existing,
        )

    print(f"\nuniverse artifact: {universe_artifact.artifact_id}")
    print(f"resolution artifact: {resolution_artifact.artifact_id}")
    print(f"option price artifacts run: {len(artifacts)}")
    for artifact in artifacts:
        print(f"  {artifact.artifact_id}  ({len(artifact.file_hashes)} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
