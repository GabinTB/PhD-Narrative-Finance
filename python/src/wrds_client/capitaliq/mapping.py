"""CapitalIQ identifier + GICS classification backfill (WRDS `ciq_common`/`comp`).

Given a universe row's isin-or-cusip, matched against its own `snapshot_date`
(not a date range -- a universe row is a point-in-time snapshot), fills in
whichever of isin/cusip/cik/gvkey/ticker/ciq_secid/country_name/country_iso/
region/gics_* the row doesn't already have. Never overwrites a value the
caller already supplied.

Every WRDS query here is scoped to the identifiers actually present in the
rows being enriched (`WHERE isin/cusip/companyid IN (...)`), not the whole
reference table (`ciq_common.wrds_isin` alone is tens of millions of rows) and
not batched per distinct `snapshot_date` either -- one query per identifier
column for the whole call, with no server-side date filter, fetching every
validity window for those identifiers. Each row is then matched against the
right window **locally** (`wrds_client.linking._shared.match_as_of_windows`),
against its own `snapshot_date`.

Chain confirmed live against WRDS (`describe_table`, not inferred):
    ciq_common.wrds_isin / wrds_cusip   -> companyid (time-bounded, primaryflag)
    companyid -> wrds_cik / wrds_gvkey / wrds_cusip / wrds_isin / wrds_ticker (time-bounded)
    companyid -> ciqsecurity.securityid (ciq_secid, time-bounded)
    companyid -> ciqcompany.countryid -> ciqcountrygeo (country name/ISO/region)
    gvkey -> comp.co_hgic (time-bounded GICS codes) -> comp.r_giccd (code -> name)

Ambiguity handling is deliberately simpler here than in `wrds_client.linking`:
this is best-effort enrichment of optional metadata, not the audited secid
resolution chain. Where more than one companyid matches an isin/cusip as of a
row's date, the `primaryflag = 1` row is preferred; if none is flagged
primary, the first match is used rather than left unresolved.

Rows are plain dicts (not `UniverseEntry`), so enrichment can run *before*
validation -- a row missing `ticker` can't even construct a `UniverseEntry`
(it's mandatory), so it must be backfilled while still a dict. See
`universe.enrich.enrich_universe` for where this fits in the pipeline.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from wrds_client.client import WRDSClient

from wrds_client.linking._shared import match_as_of_windows, normalise_isin

#: Backfillable columns sourced from ciq_common.wrds_{isin,cusip,cik,gvkey,ticker}.
_IDENTIFIER_COLUMNS = ("isin", "cusip", "cik", "gvkey", "ticker")
#: Backfillable columns sourced from ciqcompany/ciqcountrygeo.
_GEO_COLUMNS = ("country_name", "country_iso", "region")
#: Backfillable columns sourced from comp.co_hgic/comp.r_giccd, in (code, name) pairs.
_GICS_COLUMNS = (
    "gics_sector_code",
    "gics_sector",
    "gics_industry_group_code",
    "gics_industry_group",
    "gics_industry_code",
    "gics_industry",
    "gics_subindustry_code",
    "gics_subindustry",
)

_IDENTIFIER_TABLE_BY_COLUMN = {
    "isin": "wrds_isin",
    "cusip": "wrds_cusip",
    "cik": "wrds_cik",
    "gvkey": "wrds_gvkey",
    "ticker": "wrds_ticker",
}


def build_isin_companyid_query(isins: Sequence[str]) -> tuple[str, dict[str, Any]]:
    if not isins:
        raise ValueError("isins must be non-empty")
    sql = (
        "SELECT isin, companyid, primaryflag, startdate, enddate FROM ciq_common.wrds_isin "
        "WHERE isin IN %(isins)s"
    )
    return sql, {"isins": tuple(normalise_isin(i) for i in isins)}


def build_cusip_companyid_query(cusips: Sequence[str]) -> tuple[str, dict[str, Any]]:
    if not cusips:
        raise ValueError("cusips must be non-empty")
    sql = (
        "SELECT cusip, companyid, primaryflag, startdate, enddate FROM ciq_common.wrds_cusip "
        "WHERE cusip IN %(cusips)s"
    )
    return sql, {"cusips": tuple(c.strip().upper() for c in cusips)}


def resolve_companyids(client: WRDSClient, rows: Sequence[dict[str, Any]]) -> dict[int, int]:
    """Row index -> CapitalIQ companyid, matched against each row's own
    snapshot_date. ISIN first, CUSIP fallback (mirrors the ISIN-first/
    CIK-fallback convention used elsewhere in `wrds_client.linking`). Queries
    are scoped to the isins/cusips actually present across `rows` -- one
    query per identifier column for the whole call, not per distinct date.
    """
    result: dict[int, int] = {}

    isins = sorted({normalise_isin(r["isin"]) for r in rows if r.get("isin")})
    if isins:
        sql, params = build_isin_companyid_query(isins)
        df = client.raw_sql(sql, params=params, date_cols=["startdate", "enddate"])
        requests = pd.DataFrame(
            {
                "isin": [normalise_isin(r["isin"]) if r.get("isin") else None for r in rows],
                "snapshot_date": [r["snapshot_date"] for r in rows],
            }
        )
        requests = requests[requests["isin"].notna()]
        matched = match_as_of_windows(
            df, requests, key_col="isin", start_col="startdate", end_col="enddate",
            prefer_col="primaryflag",
        )
        for i, matched_row in matched.iterrows():
            result[i] = int(matched_row["companyid"])

    remaining = [i for i in range(len(rows)) if i not in result]
    cusips = sorted({rows[i]["cusip"].strip().upper() for i in remaining if rows[i].get("cusip")})
    if cusips:
        sql, params = build_cusip_companyid_query(cusips)
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
            df, requests, key_col="cusip", start_col="startdate", end_col="enddate",
            prefer_col="primaryflag",
        )
        for i, matched_row in matched.iterrows():
            result[i] = int(matched_row["companyid"])

    return result


def fetch_identifier_backfill(
    client: WRDSClient, rows: Sequence[dict[str, Any]], companyid_by_index: dict[int, int]
) -> pd.DataFrame:
    """Row index -> isin/cusip/cik/gvkey/ticker, each matched against that
    row's own snapshot_date. Queries `WHERE companyid IN (...)` for the
    companyids actually resolved (no date filter), one query per column.
    """
    companyids = sorted(set(companyid_by_index.values()))
    requests = pd.DataFrame(
        {
            "companyid": list(companyid_by_index.values()),
            "snapshot_date": [rows[i]["snapshot_date"] for i in companyid_by_index],
        },
        index=list(companyid_by_index.keys()),
    )

    out = pd.DataFrame(index=requests.index)
    for column, table in _IDENTIFIER_TABLE_BY_COLUMN.items():
        sql = (
            f"SELECT companyid, {column}, primaryflag, startdate, enddate FROM ciq_common.{table} "
            "WHERE companyid IN %(companyids)s"
        )
        df = client.raw_sql(
            sql, params={"companyids": tuple(companyids)}, date_cols=["startdate", "enddate"]
        )
        matched = match_as_of_windows(
            df, requests, key_col="companyid", start_col="startdate", end_col="enddate",
            prefer_col="primaryflag",
        )
        out[column] = matched[column] if column in matched.columns else pd.NA
    return out


def fetch_ciq_secid(
    client: WRDSClient, rows: Sequence[dict[str, Any]], companyid_by_index: dict[int, int]
) -> pd.DataFrame:
    """Row index -> CapitalIQ's own security-level ID, matched against that
    row's own snapshot_date."""
    companyids = sorted(set(companyid_by_index.values()))
    sql = (
        "SELECT companyid, securityid, primaryflag, securitystartdate, securityenddate "
        "FROM ciq_common.ciqsecurity WHERE companyid IN %(companyids)s"
    )
    df = client.raw_sql(
        sql,
        params={"companyids": tuple(companyids)},
        date_cols=["securitystartdate", "securityenddate"],
    )
    requests = pd.DataFrame(
        {
            "companyid": list(companyid_by_index.values()),
            "snapshot_date": [rows[i]["snapshot_date"] for i in companyid_by_index],
        },
        index=list(companyid_by_index.keys()),
    )
    matched = match_as_of_windows(
        df, requests, key_col="companyid", start_col="securitystartdate",
        end_col="securityenddate", prefer_col="primaryflag",
    )
    if matched.empty or "securityid" not in matched.columns:
        return pd.DataFrame(columns=["ciq_secid"])
    return matched.rename(columns={"securityid": "ciq_secid"})[["ciq_secid"]]


def fetch_country_region(client: WRDSClient, companyids: Sequence[int]) -> pd.DataFrame:
    """companyid -> country name/ISO2/region. Not time-bounded at this grain."""
    sql = (
        "SELECT co.companyid, geo.country AS country_name, "
        "geo.isocountry2 AS country_iso, geo.region "
        "FROM ciq_common.ciqcompany co "
        "JOIN ciq_common.ciqcountrygeo geo ON co.countryid = geo.countryid "
        "WHERE co.companyid IN %(companyids)s"
    )
    df = client.raw_sql(sql, params={"companyids": tuple(companyids)})
    return df.drop_duplicates("companyid")


def fetch_gics(
    client: WRDSClient, rows: Sequence[dict[str, Any]], gvkey_by_index: dict[int, str]
) -> pd.DataFrame:
    """Row index -> GICS sector/industry_group/industry/subindustry code +
    name, matched against that row's own snapshot_date."""
    gvkeys = sorted(set(gvkey_by_index.values()))
    sql = (
        "SELECT gvkey, gsector, ggroup, gind, gsubind, indfrom, indthru FROM comp.co_hgic "
        "WHERE gvkey IN %(gvkeys)s"
    )
    df = client.raw_sql(sql, params={"gvkeys": tuple(gvkeys)}, date_cols=["indfrom", "indthru"])
    if df.empty:
        return pd.DataFrame(columns=_GICS_COLUMNS)

    ref = client.raw_sql("SELECT giccd, gicdesc, gictype FROM comp.r_giccd")
    ref_by_type = {
        gictype: dict(zip(sub["giccd"], sub["gicdesc"], strict=True))
        for gictype, sub in ref.groupby("gictype")
    }

    requests = pd.DataFrame(
        {
            "gvkey": list(gvkey_by_index.values()),
            "snapshot_date": [rows[i]["snapshot_date"] for i in gvkey_by_index],
        },
        index=list(gvkey_by_index.keys()),
    )
    matched = match_as_of_windows(
        df, requests, key_col="gvkey", start_col="indfrom", end_col="indthru"
    )
    if matched.empty:
        return pd.DataFrame(columns=_GICS_COLUMNS)

    out = pd.DataFrame(index=matched.index)
    level_columns = [
        ("gsector", "GSECTOR", "gics_sector_code", "gics_sector"),
        ("ggroup", "GGROUP", "gics_industry_group_code", "gics_industry_group"),
        ("gind", "GIND", "gics_industry_code", "gics_industry"),
        ("gsubind", "GSUBIND", "gics_subindustry_code", "gics_subindustry"),
    ]
    for raw_col, gictype, code_col, name_col in level_columns:
        out[code_col] = matched[raw_col]
        out[name_col] = matched[raw_col].map(ref_by_type.get(gictype, {}))
    return out


def backfill_from_capitaliq(
    client: WRDSClient, rows: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Backfill isin/cusip/cik/gvkey/ticker/ciq_secid/country_*/region/gics_*
    for rows missing them, matched against each row's own snapshot_date.
    Never overwrites a value already present in a row. Returns new dicts --
    does not mutate the input rows.
    """
    rows_out = [dict(r) for r in rows]

    companyid_by_index = resolve_companyids(client, rows_out)
    if not companyid_by_index:
        return rows_out

    id_df = fetch_identifier_backfill(client, rows_out, companyid_by_index)
    secid_df = fetch_ciq_secid(client, rows_out, companyid_by_index)

    companyids = sorted(set(companyid_by_index.values()))
    geo_df = fetch_country_region(client, companyids).set_index("companyid")

    for i, companyid in companyid_by_index.items():
        row = rows_out[i]

        if i in id_df.index:
            for column in _IDENTIFIER_COLUMNS:
                if row.get(column) is None:
                    value = id_df.loc[i, column]
                    if pd.notna(value):
                        row[column] = value

        if row.get("ciq_secid") is None and i in secid_df.index:
            value = secid_df.loc[i, "ciq_secid"]
            if pd.notna(value):
                row["ciq_secid"] = str(int(value))

        if companyid in geo_df.index:
            for column in _GEO_COLUMNS:
                if row.get(column) is None:
                    value = geo_df.loc[companyid, column]
                    if pd.notna(value):
                        row[column] = value

    gvkey_by_index = {
        i: rows_out[i]["gvkey"] for i in companyid_by_index if rows_out[i].get("gvkey")
    }
    if gvkey_by_index:
        gics_df = fetch_gics(client, rows_out, gvkey_by_index)
        for i in gvkey_by_index:
            if i not in gics_df.index:
                continue
            row = rows_out[i]
            for column in _GICS_COLUMNS:
                if row.get(column) is None:
                    value = gics_df.loc[i, column]
                    if pd.notna(value):
                        row[column] = value

    return rows_out
