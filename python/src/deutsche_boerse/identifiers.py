"""Identifier enrichment for Deutsche Boerse data: security_id -> ISIN -> universe join.

Deutsche Boerse's own markets (XEUR/XETR, via the T7 RDI API) support only
security_id -> ISIN (`dbg_cdm.a7_utils.get_isin`), never the reverse -- there
is no way to query "which security_id is ISIN X" directly (confirmed against
the installed `a7` SDK source: its ISIN-searchable `sd` resource is for CME
markets, not XEUR/XETR). This module therefore enriches data collection's
OWN already-chosen `(mic, market_segment_id, security_id, date)` after the
fact: resolve that instrument-day's ISIN (a targeted, supported lookup), then
join against the shared universe master (`universe.schema.UniverseEntry`,
loaded from its registered datalake artifact) to attach CIK/ticker/name.

Raw cache data (orderbook/microprice, written by `data_collection/*.py`) is
never modified -- see `schema.py`'s "Raw ... never modified" contract. This
module reads that raw cache and registers a *separate*, identifier-enriched
`derived` datalake artifact, mirroring `analytics.hft_event_detection.
run_hft_analytics`'s exact `index.run(..., layer="derived")` pattern.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd

    from datalake import Artifact, DatalakeIndex

log = logging.getLogger(__name__)

#: Reasons an enriched row might have null isin/cik/ticker/name, surfaced
#: rather than silently dropped or guessed.
_FLAG_OK = "ok"
_FLAG_NO_ISIN = "no_isin"
_FLAG_ISIN_NOT_IN_UNIVERSE = "isin_not_in_universe"


def resolve_isin(mic: str, ccyymmdd: int, market_segment_id: int, security_id: int) -> str | None:
    """The ISIN for one (mic, market_segment_id, security_id) on one day, if any.

    Thin wrapper over `dbg_cdm.a7_utils.get_isin`, imported lazily so this
    module stays importable without `dbg_cdm`/API credentials, matching
    `analytics.hft_event_detection`'s convention for I/O-bound functions.
    """
    from dbg_cdm.a7_utils import get_isin

    return get_isin(mic, ccyymmdd, market_segment_id, security_id)


def enrich_with_universe(
    df: pl.DataFrame,
    isin: str | None,
    universe: pd.DataFrame,
    *,
    mic: str,
    ccyymmdd: int,
    market_segment_id: int,
    security_id: int,
) -> pl.DataFrame:
    """Attach isin/cik/ticker/name and the DBAG native key to every row of `df`.

    If `isin` is `None` (DBAG has no ISIN for this instrument on this day) or
    doesn't match any row in `universe`, the identifier columns are filled
    with `None` and `enrichment_flag` records why -- rows are never dropped
    or given a guessed identifier.
    """
    if isin is None:
        flag, cik, ticker, name = _FLAG_NO_ISIN, None, None, None
    else:
        match = universe.loc[universe["isin"] == isin]
        if match.empty:
            flag, cik, ticker, name = _FLAG_ISIN_NOT_IN_UNIVERSE, None, None, None
        else:
            row = match.iloc[0]
            flag = _FLAG_OK
            cik, ticker, name = row["cik"], row["ticker"], row["name"]

    return df.with_columns(
        pl.lit(mic).alias("mic"),
        pl.lit(market_segment_id).alias("market_segment_id"),
        pl.lit(security_id).alias("security_id"),
        pl.lit(ccyymmdd).alias("ccyymmdd"),
        pl.lit(isin).alias("isin"),
        pl.lit(cik).alias("cik"),
        pl.lit(ticker).alias("ticker"),
        pl.lit(name).alias("name"),
        pl.lit(flag).alias("enrichment_flag"),
    )


def scan_market_segment_isins(mic: str, ccyymmdd: int, market_segment_id: int) -> dict[str, int]:
    """Every ISIN -> security_id in one market segment on one day, from a
    single bulk RDI call.

    Deutsche Boerse supports only security_id -> ISIN, never the reverse --
    there is no way to query "which security_id is ISIN X" directly (confirmed
    against the installed `a7` SDK source: its ISIN-searchable `sd` resource
    is for CME markets, not XEUR/XETR). `get_market_segment_details` returns
    every RDI snapshot for the segment/day in ONE API call, including one
    `InstrumentSnapshot` per instrument with its `SecurityID` and a
    `SecurityAlt` list containing the ISIN (`SecurityAltIDSource == '4'`), so
    this builds the full reverse map from one API call rather than one call
    per security -- the "full market-day scan" approach, made considerably
    cheaper than its naive per-instrument form.
    """
    from dbg_cdm.a7_utils import get_market_segment_details

    snapshots = get_market_segment_details(mic, ccyymmdd, market_segment_id)
    isin_to_security_id: dict[str, int] = {}
    for snapshot in snapshots:
        if snapshot.get("Template") != "InstrumentSnapshot":
            continue
        security_id = snapshot.get("SecurityID")
        if security_id is None:
            continue
        for sec_alt in snapshot.get("SecurityAlt", []):
            if sec_alt.get("SecurityAltIDSource") == "4":
                isin_to_security_id[sec_alt["SecurityAltID"].strip().upper()] = int(security_id)
                break
    return isin_to_security_id


def backfill_dbga_secid(
    rows: Sequence[dict[str, Any]], mic: str, market_segment_id: int
) -> list[dict[str, Any]]:
    """Backfill `dbga_secid` for rows missing it, by scanning `mic`/
    `market_segment_id` once per distinct `snapshot_date` present in `rows`
    (not once per row) and matching by ISIN. Rows whose ISIN isn't found in
    that day's scan (delisted, wrong segment, no ISIN, ...) stay unresolved --
    surfaced by `universe.schema.find_incomplete`, not guessed. Never
    overwrites a value already present in a row. Returns new dicts -- does
    not mutate the input rows.
    """
    rows_out = [dict(r) for r in rows]
    candidates = [
        i for i, r in enumerate(rows_out) if r.get("dbga_secid") is None and r.get("isin")
    ]
    if not candidates:
        return rows_out

    by_ccyymmdd: dict[int, list[int]] = {}
    for i in candidates:
        ccyymmdd = int(rows_out[i]["snapshot_date"].strftime("%Y%m%d"))
        by_ccyymmdd.setdefault(ccyymmdd, []).append(i)

    for ccyymmdd, indices in by_ccyymmdd.items():
        isin_to_security_id = scan_market_segment_isins(mic, ccyymmdd, market_segment_id)
        for i in indices:
            isin_key = rows_out[i]["isin"].strip().upper()
            if isin_key in isin_to_security_id:
                rows_out[i]["dbga_secid"] = str(isin_to_security_id[isin_key])

    return rows_out


def run_identifier_enrichment(
    index: DatalakeIndex,
    mic: str,
    product: str,
    from_ccyymmdd: int,
    to_ccyymmdd: int,
    *,
    source: Literal["orderbook", "microprice"],
    universe: pd.DataFrame,
    pipeline: str,
    pipeline_version: str,
    pipeline_repo: str | None = None,
    skip_missing: bool = True,
    orderbook_from_time: str = "00:00:00",
    orderbook_to_time: str = "23:59:59",
) -> Artifact:
    """Enrich `source` data day-by-day for the most-liquid contract of `product`
    and register the result as a derived datalake artifact.

    Reuses `resolve_most_liquid`'s day-iteration pattern (the same one
    `analytics.hft_event_detection.iter_jobs` uses) rather than `HFTJob`
    itself, since that dataclass carries HFT-specific fields (`markout_period`)
    that don't apply here.
    """
    from dbg_cdm.time_utils import SKIP_DATES, daterange, today_ccyymmdd

    from deutsche_boerse.analytics.hft_event_detection import resolve_most_liquid
    from deutsche_boerse.data_collection.microprice import get_microprice
    from deutsche_boerse.data_collection.orderbook import get_orderbook

    hyperparams = {
        "mic": mic,
        "product": product,
        "from_ccyymmdd": from_ccyymmdd,
        "to_ccyymmdd": to_ccyymmdd,
        "source": source,
    }

    with index.run(
        kind=f"{source}_enriched",
        pipeline=pipeline,
        pipeline_version=pipeline_version,
        pipeline_repo=pipeline_repo,
        hyperparams=hyperparams,
        layer="derived",
    ) as run:
        n_days = 0
        for ccyymmdd in daterange(from_ccyymmdd, to_ccyymmdd):
            if ccyymmdd in SKIP_DATES or ccyymmdd >= today_ccyymmdd():
                continue
            resolved = resolve_most_liquid(mic, ccyymmdd, product)
            if resolved is None:
                continue
            market_segment_id, security_id = resolved

            try:
                if source == "microprice":
                    raw = get_microprice(mic, ccyymmdd, market_segment_id, security_id)
                else:
                    raw = get_orderbook(
                        mic, ccyymmdd, market_segment_id, security_id,
                        orderbook_from_time, orderbook_to_time,
                    )
            except FileNotFoundError as exc:
                if skip_missing:
                    log.warning("skip %s %d: %s", mic, ccyymmdd, exc)
                    continue
                raise

            if raw.is_empty():
                log.warning("skip %s %d: empty %s data", mic, ccyymmdd, source)
                continue

            isin = resolve_isin(mic, ccyymmdd, market_segment_id, security_id)
            enriched = enrich_with_universe(
                raw,
                isin,
                universe,
                mic=mic,
                ccyymmdd=ccyymmdd,
                market_segment_id=market_segment_id,
                security_id=security_id,
            )
            out_path = run.out_dir / f"{market_segment_id}_{security_id}_{ccyymmdd}.parquet"
            enriched.write_parquet(out_path, compression="zstd")
            n_days += 1

        if n_days == 0:
            raise RuntimeError(
                f"no {source} data enriched for {mic} {product} {from_ccyymmdd}-{to_ccyymmdd}"
            )
        run.note(f"{n_days} instrument-day(s) enriched")

    return index.get(run.artifact_id)
