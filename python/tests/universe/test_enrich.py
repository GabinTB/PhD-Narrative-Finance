"""Tests for universe.enrich.enrich_universe: orchestrates CapitalIQ/LSEG/DBGA backfill.

Uses stub clients throughout -- no real WRDS server, no real dbg_cdm (stubbed
via sys.modules injection for the Deutsche Boerse step, same technique as
tests/Deutesche_Boerse/test_identifiers.py). Rows are plain dicts (see
universe.enrich's module docstring for why); canned reference frames include
startdate/enddate (None = open-ended) since matching now happens locally, not
via a server-side as_of filter.
"""
from __future__ import annotations

import sys
import types
from datetime import date
from typing import Any

import pandas as pd
import pytest

from universe.enrich import enrich_universe

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
        "sedol": None,
        "dbga_secid": None,
    }
    row.update(overrides)
    return row


class StubWRDSClient:
    """Answers CapitalIQ + LSEG queries needed for one full enrichment pass."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def raw_sql(self, sql: str, **kwargs: Any) -> pd.DataFrame:
        self.calls.append(sql)
        if "wrds_isin" in sql:
            return pd.DataFrame(
                {
                    "isin": ["US0378331005"],
                    "companyid": [21835],
                    "primaryflag": [1],
                    "startdate": [None],
                    "enddate": [None],
                }
            )
        if "wrds_cusip" in sql:
            return pd.DataFrame(
                {
                    "companyid": [21835],
                    "cusip": ["037833100"],
                    "primaryflag": [1],
                    "startdate": [None],
                    "enddate": [None],
                }
            )
        if "wrds_cik" in sql:
            return pd.DataFrame(
                {
                    "companyid": [21835],
                    "cik": ["0000320193"],
                    "primaryflag": [1],
                    "startdate": [None],
                    "enddate": [None],
                }
            )
        if "wrds_gvkey" in sql:
            return pd.DataFrame(
                {
                    "companyid": [21835],
                    "gvkey": ["001690"],
                    "primaryflag": [1],
                    "startdate": [None],
                    "enddate": [None],
                }
            )
        if "wrds_ticker" in sql:
            return pd.DataFrame(
                {
                    "companyid": [21835],
                    "ticker": ["AAPL"],
                    "primaryflag": [1],
                    "startdate": [None],
                    "enddate": [None],
                }
            )
        if "ciqsecurity" in sql:
            return pd.DataFrame(
                {
                    "companyid": [21835],
                    "securityid": [999],
                    "primaryflag": [1],
                    "securitystartdate": [None],
                    "securityenddate": [None],
                }
            )
        if "ciqcountrygeo" in sql:
            return pd.DataFrame(
                {
                    "companyid": [21835],
                    "country_name": ["United States"],
                    "country_iso": ["US"],
                    "region": ["North America"],
                }
            )
        if "co_hgic" in sql:
            return pd.DataFrame(
                {
                    "gvkey": ["001690"],
                    "gsector": ["45"],
                    "ggroup": ["4520"],
                    "gind": ["452020"],
                    "gsubind": ["45202030"],
                    "indfrom": [None],
                    "indthru": [None],
                }
            )
        if "r_giccd" in sql:
            return pd.DataFrame(
                {
                    "giccd": ["45", "4520", "452020", "45202030"],
                    "gicdesc": ["Tech", "Hardware", "Hardware Storage", "Hardware Storage Sub"],
                    "gictype": ["GSECTOR", "GGROUP", "GIND", "GSUBIND"],
                }
            )
        if "permisindata" in sql:
            return pd.DataFrame(
                {
                    "isin": ["US0378331005"],
                    "instrpermid": [555],
                    "startdate": [None],
                    "enddate": [None],
                }
            )
        if "perminstrref" in sql:
            return pd.DataFrame({"instrpermid": [555], "valquotepermid": [777]})
        if "permsedoldata" in sql:
            return pd.DataFrame(
                {"quotepermid": [777], "sedol": ["2046251"], "startdate": [None], "enddate": [None]}
            )
        raise AssertionError(f"unexpected SQL: {sql}")


def _install_fake_dbg_cdm(monkeypatch: pytest.MonkeyPatch, security_id: int = 204934) -> None:
    def get_market_segment_details(mic: str, ccyymmdd: int, market_segment_id: int) -> list[dict]:
        return [
            {
                "Template": "InstrumentSnapshot",
                "SecurityID": security_id,
                "SecurityAlt": [{"SecurityAltIDSource": "4", "SecurityAltID": "US0378331005"}],
            }
        ]

    fake_a7_utils = types.ModuleType("dbg_cdm.a7_utils")
    fake_a7_utils.get_market_segment_details = get_market_segment_details
    fake_dbg_cdm = types.ModuleType("dbg_cdm")
    fake_dbg_cdm.a7_utils = fake_a7_utils
    monkeypatch.setitem(sys.modules, "dbg_cdm", fake_dbg_cdm)
    monkeypatch.setitem(sys.modules, "dbg_cdm.a7_utils", fake_a7_utils)


def test_enrich_universe_no_client_no_dbga_is_noop() -> None:
    rows = [_row()]
    result = enrich_universe(rows)
    assert result == rows


def test_enrich_universe_wrds_client_runs_capitaliq_and_lseg() -> None:
    client = StubWRDSClient()
    result = enrich_universe([_row()], wrds_client=client)

    row = result[0]
    assert row["cik"] == "0000320193"
    assert row["gvkey"] == "001690"
    assert row["gics_sector"] == "Tech"
    assert row["sedol"] == "2046251"


def test_enrich_universe_fills_missing_ticker() -> None:
    client = StubWRDSClient()
    result = enrich_universe([_row(ticker=None)], wrds_client=client)
    assert result[0]["ticker"] == "AAPL"


def test_enrich_universe_resolve_sedol_false_skips_lseg() -> None:
    client = StubWRDSClient()
    result = enrich_universe([_row()], wrds_client=client, resolve_sedol=False)

    row = result[0]
    assert row["cik"] == "0000320193"  # CapitalIQ still ran
    assert row["sedol"] is None  # LSEG skipped
    assert not any("permisindata" in c for c in client.calls)


def test_enrich_universe_dbga_backfill_needs_both_mic_and_segment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_dbg_cdm(monkeypatch)
    rows = [_row()]

    # Only mic given -> DBGA step skipped, no dbg_cdm call attempted.
    result = enrich_universe(rows, dbga_mic="XETR")
    assert result[0]["dbga_secid"] is None


def test_enrich_universe_dbga_backfill_runs_when_both_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_dbg_cdm(monkeypatch, security_id=204934)
    result = enrich_universe([_row()], dbga_mic="XETR", dbga_market_segment_id=688)
    assert result[0]["dbga_secid"] == "204934"


def test_enrich_universe_full_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_dbg_cdm(monkeypatch, security_id=204934)
    client = StubWRDSClient()
    result = enrich_universe(
        [_row()], wrds_client=client, dbga_mic="XETR", dbga_market_segment_id=688
    )

    row = result[0]
    assert row["cik"] == "0000320193"
    assert row["gvkey"] == "001690"
    assert row["sedol"] == "2046251"
    assert row["dbga_secid"] == "204934"


def test_enrich_universe_never_overwrites_existing_values(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_dbg_cdm(monkeypatch, security_id=204934)
    client = StubWRDSClient()
    rows = [_row(cik="PRESET", sedol="PRESET", dbga_secid="PRESET")]
    result = enrich_universe(
        rows, wrds_client=client, dbga_mic="XETR", dbga_market_segment_id=688
    )

    row = result[0]
    assert row["cik"] == "PRESET"
    assert row["sedol"] == "PRESET"
    assert row["dbga_secid"] == "PRESET"
