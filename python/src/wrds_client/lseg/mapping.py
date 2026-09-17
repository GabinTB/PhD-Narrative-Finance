"""LSEG (formerly Refinitiv/Thomson Reuters) SEDOL backfill (WRDS `tr_common`).

Given a universe row's isin-or-cusip, matched against its own `snapshot_date`,
fills in `sedol` if missing. Never overwrites a value the caller already
supplied.

Chain confirmed live against WRDS (`describe_table`, not inferred):
    tr_common.permisindata / permcusipdata  -> instrpermid (time-bounded)
    tr_common.perminstrref.valquotepermid   -> the primary quote for that instrument
    tr_common.permsedoldata                 -> SEDOL, keyed by quotepermid (time-bounded)

`instrpermid` (instrument-level) and `quotepermid` (quote/listing-level) are
different LSEG PermID namespaces -- SEDOL is a per-listing identifier, so it's
keyed by quotepermid, not instrpermid directly; `perminstrref.valquotepermid`
is the documented "main primary Quote" link between the two.

As with `wrds_client.capitaliq`, every query here is scoped to the
identifiers actually present in the rows being enriched (`tr_common.
permisindata` alone is ~231.5M rows, confirmed live -- not something to pull
whole), not batched per distinct `snapshot_date` either: one query per
identifier column for the whole call, no server-side date filter, fetching
every validity window and matching each row against the right one **locally**
(`wrds_client.linking._shared.match_as_of_windows`).

Rows are plain dicts (not `UniverseEntry`) -- see
`wrds_client.capitaliq.mapping` and `universe.enrich.enrich_universe` for why.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from wrds_client.client import WRDSClient

from wrds_client.linking._shared import match_as_of_windows, normalise_isin


def build_isin_instrpermid_query(isins: Sequence[str]) -> tuple[str, dict[str, Any]]:
    if not isins:
        raise ValueError("isins must be non-empty")
    sql = (
        "SELECT isin, instrpermid, startdate, enddate FROM tr_common.permisindata "
        "WHERE isin IN %(isins)s"
    )
    return sql, {"isins": tuple(normalise_isin(i) for i in isins)}


def build_cusip_instrpermid_query(cusips: Sequence[str]) -> tuple[str, dict[str, Any]]:
    if not cusips:
        raise ValueError("cusips must be non-empty")
    sql = (
        "SELECT cusip, instrpermid, startdate, enddate FROM tr_common.permcusipdata "
        "WHERE cusip IN %(cusips)s"
    )
    return sql, {"cusips": tuple(c.strip().upper() for c in cusips)}


def resolve_instrpermids(client: WRDSClient, rows: Sequence[dict[str, Any]]) -> dict[int, int]:
    """Row index -> LSEG instrpermid, matched against each row's own
    snapshot_date. ISIN first, CUSIP fallback. Queries are scoped to the
    isins/cusips actually present across `rows` -- one query per identifier
    column for the whole call, not per distinct date.
    """
    result: dict[int, int] = {}

    isins = sorted({normalise_isin(r["isin"]) for r in rows if r.get("isin")})
    if isins:
        sql, params = build_isin_instrpermid_query(isins)
        df = client.raw_sql(sql, params=params, date_cols=["startdate", "enddate"])
        requests = pd.DataFrame(
            {
                "isin": [normalise_isin(r["isin"]) if r.get("isin") else None for r in rows],
                "snapshot_date": [r["snapshot_date"] for r in rows],
            }
        )
        requests = requests[requests["isin"].notna()]
        matched = match_as_of_windows(
            df, requests, key_col="isin", start_col="startdate", end_col="enddate"
        )
        for i, matched_row in matched.iterrows():
            result[i] = int(matched_row["instrpermid"])

    remaining = [i for i in range(len(rows)) if i not in result]
    cusips = sorted({rows[i]["cusip"].strip().upper() for i in remaining if rows[i].get("cusip")})
    if cusips:
        sql, params = build_cusip_instrpermid_query(cusips)
        df = client.raw_sql(sql, params=params, date_cols=["startdate", "enddate"])
        requests = pd.DataFrame(
            {
                "cusip": [
                    rows[i]["cusip"].strip().upper() if rows[i].get("cusip") else None
                    for i in remaining
                ],
                "snapshot_date": [rows[i]["snapshot_date"] for i in remaining],
            },
            index=remaining,
        )
        requests = requests[requests["cusip"].notna()]
        matched = match_as_of_windows(
            df, requests, key_col="cusip", start_col="startdate", end_col="enddate"
        )
        for i, matched_row in matched.iterrows():
            result[i] = int(matched_row["instrpermid"])

    return result


def fetch_primary_quotepermid(client: WRDSClient, instrpermids: Sequence[int]) -> pd.DataFrame:
    """instrpermid -> its primary quotepermid (`perminstrref.valquotepermid`)."""
    sql = (
        "SELECT instrpermid, valquotepermid FROM tr_common.perminstrref "
        "WHERE instrpermid IN %(instrpermids)s"
    )
    df = client.raw_sql(sql, params={"instrpermids": tuple(instrpermids)})
    return df.dropna(subset=["valquotepermid"]).drop_duplicates("instrpermid")


def fetch_sedol(
    client: WRDSClient, rows: Sequence[dict[str, Any]], quotepermid_by_index: dict[int, int]
) -> pd.DataFrame:
    """Row index -> SEDOL, matched against that row's own snapshot_date."""
    quotepermids = sorted(set(quotepermid_by_index.values()))
    sql = (
        "SELECT quotepermid, sedol, startdate, enddate FROM tr_common.permsedoldata "
        "WHERE quotepermid IN %(quotepermids)s"
    )
    df = client.raw_sql(
        sql, params={"quotepermids": tuple(quotepermids)}, date_cols=["startdate", "enddate"]
    )
    requests = pd.DataFrame(
        {
            "quotepermid": list(quotepermid_by_index.values()),
            "snapshot_date": [rows[i]["snapshot_date"] for i in quotepermid_by_index],
        },
        index=list(quotepermid_by_index.keys()),
    )
    matched = match_as_of_windows(
        df, requests, key_col="quotepermid", start_col="startdate", end_col="enddate"
    )
    if matched.empty or "sedol" not in matched.columns:
        return pd.DataFrame(columns=["sedol"])
    return matched[["sedol"]]


def backfill_sedol(
    client: WRDSClient, rows: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Backfill `sedol` for rows missing it, matched against each row's own
    snapshot_date. Never overwrites a value already present. Returns new
    dicts -- does not mutate the input rows.
    """
    rows_out = [dict(r) for r in rows]
    candidates = [i for i, r in enumerate(rows_out) if r.get("sedol") is None]
    if not candidates:
        return rows_out

    instrpermid_by_index = resolve_instrpermids(client, [rows_out[i] for i in candidates])
    # resolve_instrpermids was called on a filtered sub-list; remap back to
    # rows_out's own indices.
    instrpermid_by_index = {candidates[k]: v for k, v in instrpermid_by_index.items()}
    if not instrpermid_by_index:
        return rows_out

    quote_df = fetch_primary_quotepermid(
        client, sorted(set(instrpermid_by_index.values()))
    ).set_index("instrpermid")["valquotepermid"]

    quotepermid_by_index: dict[int, int] = {}
    for i, instrpermid in instrpermid_by_index.items():
        if instrpermid in quote_df.index:
            quotepermid_by_index[i] = int(quote_df.loc[instrpermid])
    if not quotepermid_by_index:
        return rows_out

    sedol_df = fetch_sedol(client, rows_out, quotepermid_by_index)
    for i in quotepermid_by_index:
        if i not in sedol_df.index:
            continue
        value = sedol_df.loc[i, "sedol"]
        if pd.notna(value):
            rows_out[i]["sedol"] = value

    return rows_out
