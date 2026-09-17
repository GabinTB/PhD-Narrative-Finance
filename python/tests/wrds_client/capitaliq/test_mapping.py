"""Tests for wrds_client.capitaliq.mapping: CapitalIQ identifier + GICS backfill.

No real WRDS server involved: query builders are pure; resolve/fetch/backfill
functions are tested against a stub client with canned responses. Rows are
plain dicts (see wrds_client.capitaliq.mapping's module docstring for why);
canned reference frames include startdate/enddate (None = open-ended) since
matching now happens locally via `match_as_of_windows`, not a server-side
as_of filter.
"""
from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
import pytest

from wrds_client.capitaliq.mapping import (
    backfill_from_capitaliq,
    build_cusip_companyid_query,
    build_isin_companyid_query,
    fetch_ciq_secid,
    fetch_country_region,
    fetch_gics,
    fetch_identifier_backfill,
    resolve_companyids,
)

_DATE = date(2023, 1, 1)


def _row(**overrides: Any) -> dict[str, Any]:
    row = {
        "snapshot_date": _DATE,
        "name": "Apple Inc",
        "ticker": "AAPL",
        "isin": "US0378331005",
        "cusip": None,
        "cik": None,
        "gvkey": None,
        "ciq_secid": None,
        "country_name": None,
        "country_iso": None,
        "region": None,
        "gics_sector": None,
        "gics_sector_code": None,
        "gics_industry_group": None,
        "gics_industry_group_code": None,
        "gics_industry": None,
        "gics_industry_code": None,
        "gics_subindustry": None,
        "gics_subindustry_code": None,
    }
    row.update(overrides)
    return row


class ScriptedWRDSClient:
    """Returns a canned DataFrame based on a substring match against the SQL text."""

    def __init__(self, scripts: list[tuple[str, pd.DataFrame]]) -> None:
        self._scripts = list(scripts)
        self.calls: list[dict[str, Any]] = []

    def raw_sql(self, sql: str, **kwargs: Any) -> pd.DataFrame:
        self.calls.append({"sql": sql, **kwargs})
        for i, (match, df) in enumerate(self._scripts):
            if match in sql:
                del self._scripts[i]
                return df
        raise AssertionError(f"no scripted response matches SQL: {sql}")


def _companyid_df(isin: str = "US0378331005", companyid: int = 21835) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "isin": [isin],
            "companyid": [companyid],
            "primaryflag": [1],
            "startdate": [None],
            "enddate": [None],
        }
    )


def _identifier_frames(companyid: int = 21835) -> list[tuple[str, pd.DataFrame]]:
    return [
        (
            "wrds_isin",
            pd.DataFrame(
                {
                    "companyid": [companyid],
                    "isin": ["US0378331005"],
                    "primaryflag": [1],
                    "startdate": [None],
                    "enddate": [None],
                }
            ),
        ),
        (
            "wrds_cusip",
            pd.DataFrame(
                {
                    "companyid": [companyid],
                    "cusip": ["037833100"],
                    "primaryflag": [1],
                    "startdate": [None],
                    "enddate": [None],
                }
            ),
        ),
        (
            "wrds_cik",
            pd.DataFrame(
                {
                    "companyid": [companyid],
                    "cik": ["0000320193"],
                    "primaryflag": [1],
                    "startdate": [None],
                    "enddate": [None],
                }
            ),
        ),
        (
            "wrds_gvkey",
            pd.DataFrame(
                {
                    "companyid": [companyid],
                    "gvkey": ["001690"],
                    "primaryflag": [1],
                    "startdate": [None],
                    "enddate": [None],
                }
            ),
        ),
        (
            "wrds_ticker",
            pd.DataFrame(
                {
                    "companyid": [companyid],
                    "ticker": ["AAPL"],
                    "primaryflag": [1],
                    "startdate": [None],
                    "enddate": [None],
                }
            ),
        ),
    ]


def _gics_frames() -> list[tuple[str, pd.DataFrame]]:
    return [
        (
            "co_hgic",
            pd.DataFrame(
                {
                    "gvkey": ["001690"],
                    "gsector": ["45"],
                    "ggroup": ["4520"],
                    "gind": ["452020"],
                    "gsubind": ["45202030"],
                    "indfrom": [None],
                    "indthru": [None],
                }
            ),
        ),
        (
            "r_giccd",
            pd.DataFrame(
                {
                    "giccd": ["45", "4520", "452020", "45202030"],
                    "gicdesc": ["Tech", "Hardware", "Hardware Storage", "Hardware Storage Sub"],
                    "gictype": ["GSECTOR", "GGROUP", "GIND", "GSUBIND"],
                }
            ),
        ),
    ]


# ---------------------------------------------------------------------------
# query builders
# ---------------------------------------------------------------------------


def test_build_isin_companyid_query_binds_params() -> None:
    sql, params = build_isin_companyid_query(["us0378331005"])
    assert "%(isins)s" in sql
    assert params == {"isins": ("US0378331005",)}


def test_build_isin_companyid_query_empty_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        build_isin_companyid_query([])


def test_build_cusip_companyid_query_binds_params() -> None:
    sql, params = build_cusip_companyid_query([" 037833100 "])
    assert params == {"cusips": ("037833100",)}


# ---------------------------------------------------------------------------
# resolve_companyids
# ---------------------------------------------------------------------------


def test_resolve_companyids_isin_hit() -> None:
    client = ScriptedWRDSClient([("wrds_isin", _companyid_df())])
    result = resolve_companyids(client, [_row()])
    assert result == {0: 21835}


def test_resolve_companyids_falls_back_to_cusip_when_isin_misses() -> None:
    empty = pd.DataFrame(
        {"isin": [], "companyid": [], "primaryflag": [], "startdate": [], "enddate": []}
    )
    cusip_hit = pd.DataFrame(
        {
            "cusip": ["037833100"],
            "companyid": [21835],
            "primaryflag": [1],
            "startdate": [None],
            "enddate": [None],
        }
    )
    client = ScriptedWRDSClient([("wrds_isin", empty), ("wrds_cusip", cusip_hit)])
    row = _row(isin="XX0000000000", cusip="037833100")
    result = resolve_companyids(client, [row])
    assert result == {0: 21835}


def test_resolve_companyids_prefers_primaryflag() -> None:
    df = pd.DataFrame(
        {
            "isin": ["US0378331005", "US0378331005"],
            "companyid": [999, 21835],
            "primaryflag": [0, 1],
            "startdate": [None, None],
            "enddate": [None, None],
        }
    )
    client = ScriptedWRDSClient([("wrds_isin", df)])
    result = resolve_companyids(client, [_row()])
    assert result == {0: 21835}


def test_resolve_companyids_picks_window_covering_row_snapshot_date() -> None:
    """A companyid can have several validity windows across its history --
    the whole point of no-date-filter + local matching is to pick the one
    covering each row's own snapshot_date, not just take the newest/first."""
    df = pd.DataFrame(
        {
            "isin": ["US0378331005", "US0378331005"],
            "companyid": [111, 222],
            "primaryflag": [1, 1],
            "startdate": [date(2000, 1, 1), date(2020, 1, 1)],
            "enddate": [date(2019, 12, 31), None],
        }
    )
    client = ScriptedWRDSClient([("wrds_isin", df)])
    result = resolve_companyids(client, [_row(snapshot_date=date(2010, 1, 1))])
    assert result == {0: 111}


def test_resolve_companyids_no_match_produces_empty_dict() -> None:
    empty = pd.DataFrame(
        {"isin": [], "companyid": [], "primaryflag": [], "startdate": [], "enddate": []}
    )
    client = ScriptedWRDSClient([("wrds_isin", empty)])
    row = _row(isin="XX0000000000", cusip=None)
    result = resolve_companyids(client, [row])
    assert result == {}


def test_resolve_companyids_multiple_snapshot_dates_issue_one_query() -> None:
    """Rows spanning many distinct snapshot_dates still only cost one isin
    query for the whole call -- the query is scoped by identifier, not
    batched per date."""
    df = pd.DataFrame(
        {
            "isin": ["A", "B"],
            "companyid": [1, 2],
            "primaryflag": [1, 1],
            "startdate": [None, None],
            "enddate": [None, None],
        }
    )
    client = ScriptedWRDSClient([("wrds_isin", df)])
    rows = [
        _row(isin="A", snapshot_date=date(2023, 1, 1)),
        _row(isin="B", snapshot_date=date(2023, 6, 1)),
    ]
    result = resolve_companyids(client, rows)
    assert result == {0: 1, 1: 2}
    assert len(client.calls) == 1


# ---------------------------------------------------------------------------
# fetch_* helpers
# ---------------------------------------------------------------------------


def test_fetch_identifier_backfill_combines_five_tables() -> None:
    client = ScriptedWRDSClient(_identifier_frames())
    rows = [_row()]
    df = fetch_identifier_backfill(client, rows, {0: 21835})
    row = df.loc[0]
    assert row["isin"] == "US0378331005"
    assert row["cusip"] == "037833100"
    assert row["cik"] == "0000320193"
    assert row["gvkey"] == "001690"
    assert row["ticker"] == "AAPL"


def test_fetch_ciq_secid() -> None:
    df = pd.DataFrame(
        {
            "companyid": [21835],
            "securityid": [12345],
            "primaryflag": [1],
            "securitystartdate": [None],
            "securityenddate": [None],
        }
    )
    client = ScriptedWRDSClient([("ciqsecurity", df)])
    rows = [_row()]
    result = fetch_ciq_secid(client, rows, {0: 21835})
    assert result.loc[0, "ciq_secid"] == 12345


def test_fetch_country_region() -> None:
    df = pd.DataFrame(
        {
            "companyid": [21835],
            "country_name": ["United States"],
            "country_iso": ["US"],
            "region": ["North America"],
        }
    )
    client = ScriptedWRDSClient([("ciqcountrygeo", df)])
    result = fetch_country_region(client, [21835])
    row = result.iloc[0]
    assert row["country_name"] == "United States"
    assert row["country_iso"] == "US"
    assert row["region"] == "North America"


def test_fetch_gics_joins_reference_names() -> None:
    client = ScriptedWRDSClient(_gics_frames())
    rows = [_row(gvkey="001690")]
    df = fetch_gics(client, rows, {0: "001690"})
    row = df.loc[0]
    assert row["gics_sector_code"] == "45"
    assert row["gics_sector"] == "Tech"
    assert row["gics_subindustry_code"] == "45202030"


def test_fetch_gics_empty_gvkeys_returns_empty_frame() -> None:
    empty = pd.DataFrame(
        {
            "gvkey": [],
            "gsector": [],
            "ggroup": [],
            "gind": [],
            "gsubind": [],
            "indfrom": [],
            "indthru": [],
        }
    )
    client = ScriptedWRDSClient([("co_hgic", empty)])
    rows = [_row(gvkey="001690")]
    df = fetch_gics(client, rows, {0: "001690"})
    assert df.empty
    assert "gics_sector" in df.columns


# ---------------------------------------------------------------------------
# backfill_from_capitaliq
# ---------------------------------------------------------------------------


def _full_stub_client(companyid: int = 21835) -> ScriptedWRDSClient:
    secid_df = pd.DataFrame(
        {
            "companyid": [companyid],
            "securityid": [999],
            "primaryflag": [1],
            "securitystartdate": [None],
            "securityenddate": [None],
        }
    )
    geo_df = pd.DataFrame(
        {
            "companyid": [companyid],
            "country_name": ["United States"],
            "country_iso": ["US"],
            "region": ["North America"],
        }
    )
    return ScriptedWRDSClient(
        [
            ("wrds_isin", _companyid_df(companyid=companyid)),
            *_identifier_frames(companyid=companyid),
            ("ciqsecurity", secid_df),
            ("ciqcountrygeo", geo_df),
            *_gics_frames(),
        ]
    )


def test_backfill_from_capitaliq_fills_missing_fields() -> None:
    client = _full_stub_client()
    rows = [_row()]  # only snapshot_date/name/ticker/isin set
    backfilled = backfill_from_capitaliq(client, rows)

    assert len(backfilled) == 1
    row = backfilled[0]
    assert row["cusip"] == "037833100"
    assert row["cik"] == "0000320193"
    assert row["gvkey"] == "001690"
    assert row["ciq_secid"] == "999"
    assert row["country_name"] == "United States"
    assert row["gics_sector"] == "Tech"
    assert row["gics_subindustry_code"] == "45202030"


def test_backfill_from_capitaliq_fills_missing_ticker() -> None:
    client = _full_stub_client()
    rows = [_row(ticker=None)]
    backfilled = backfill_from_capitaliq(client, rows)
    assert backfilled[0]["ticker"] == "AAPL"


def test_backfill_from_capitaliq_never_overwrites_existing_values() -> None:
    client = _full_stub_client()
    rows = [_row(cik="9999999999", gvkey="OVERRIDE")]
    backfilled = backfill_from_capitaliq(client, rows)

    row = backfilled[0]
    assert row["cik"] == "9999999999"
    assert row["gvkey"] == "OVERRIDE"
    # gvkey wasn't backfilled, but the row's own gvkey ("OVERRIDE") doesn't
    # exist in the stubbed GICS table, so GICS columns should stay None.
    assert row["gics_sector"] is None


def test_backfill_from_capitaliq_unresolved_row_is_unchanged() -> None:
    empty = pd.DataFrame(
        {"isin": [], "companyid": [], "primaryflag": [], "startdate": [], "enddate": []}
    )
    client = ScriptedWRDSClient([("wrds_isin", empty)])
    row = _row(isin="XX0000000000", cusip=None)
    backfilled = backfill_from_capitaliq(client, [row])
    assert backfilled == [row]


def test_backfill_from_capitaliq_does_not_mutate_input_rows() -> None:
    client = _full_stub_client()
    row = _row()
    backfill_from_capitaliq(client, [row])
    assert row["cusip"] is None
