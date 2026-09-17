"""Tests for deutsche_boerse.identifiers: security_id -> ISIN -> universe enrichment.

`enrich_with_universe` is pure (no dbg_cdm/API dependency) and tested
directly. `resolve_isin` is I/O (wraps `dbg_cdm.a7_utils.get_isin`) and not
unit-tested here, matching this module's existing convention of not testing
dbg_cdm-dependent I/O functions directly (see
`analytics/hft_event_detection.py`'s module docstring).
"""
from __future__ import annotations

import sys
import types
from datetime import date
from typing import Any

import pandas as pd
import polars as pl
import pytest

from deutsche_boerse.identifiers import (
    backfill_dbga_secid,
    enrich_with_universe,
    scan_market_segment_isins,
)


def _universe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "isin": ["DE0007164600", "US0378331005"],
            "cik": [None, "0000320193"],
            "ticker": ["SAP", "AAPL"],
            "name": ["SAP SE", "Apple Inc"],
        }
    )


def _raw() -> pl.DataFrame:
    return pl.DataFrame({"timestamp": [1, 2], "price": [10.0, 11.0]})


def test_enrich_with_universe_golden_path() -> None:
    out = enrich_with_universe(
        _raw(),
        "DE0007164600",
        _universe(),
        mic="XETR",
        ccyymmdd=20230103,
        market_segment_id=688,
        security_id=204934,
    )
    assert out.height == 2
    assert out["ticker"].to_list() == ["SAP", "SAP"]
    assert out["name"].to_list() == ["SAP SE", "SAP SE"]
    assert out["enrichment_flag"].to_list() == ["ok", "ok"]
    assert out["mic"].to_list() == ["XETR", "XETR"]
    assert out["market_segment_id"].to_list() == [688, 688]
    assert out["security_id"].to_list() == [204934, 204934]
    assert out["ccyymmdd"].to_list() == [20230103, 20230103]


def test_enrich_with_universe_no_isin_flagged_not_dropped() -> None:
    out = enrich_with_universe(
        _raw(), None, _universe(), mic="XETR", ccyymmdd=20230103, market_segment_id=1, security_id=1
    )
    assert out.height == 2
    assert out["enrichment_flag"].to_list() == ["no_isin", "no_isin"]
    assert out["isin"].to_list() == [None, None]
    assert out["ticker"].to_list() == [None, None]


def test_enrich_with_universe_isin_not_in_universe_flagged_not_dropped() -> None:
    out = enrich_with_universe(
        _raw(),
        "XX0000000000",
        _universe(),
        mic="XETR",
        ccyymmdd=20230103,
        market_segment_id=1,
        security_id=1,
    )
    assert out.height == 2
    assert out["enrichment_flag"].to_list() == ["isin_not_in_universe", "isin_not_in_universe"]
    assert out["isin"].to_list() == ["XX0000000000", "XX0000000000"]
    assert out["ticker"].to_list() == [None, None]


def test_enrich_with_universe_empty_frame_stays_empty() -> None:
    empty = pl.DataFrame({"timestamp": [], "price": []})
    out = enrich_with_universe(
        empty, "DE0007164600", _universe(), mic="XETR", ccyymmdd=20230103,
        market_segment_id=688, security_id=204934,
    )
    assert out.height == 0
    assert "ticker" in out.columns


def test_enrich_with_universe_second_isin_in_universe() -> None:
    out = enrich_with_universe(
        _raw(),
        "US0378331005",
        _universe(),
        mic="XETR",
        ccyymmdd=20230103,
        market_segment_id=1,
        security_id=1,
    )
    assert out["cik"].to_list() == ["0000320193", "0000320193"]
    assert out["ticker"].to_list() == ["AAPL", "AAPL"]


# ---------------------------------------------------------------------------
# scan_market_segment_isins / backfill_dbga_secid
#
# dbg_cdm isn't installed in this environment (no a7/API credentials here),
# so these stub it via sys.modules injection rather than skipping coverage
# entirely -- unlike run_identifier_enrichment, which also needs `a7` itself
# (imported at module level by data_collection/microprice.py and
# orderbook.py) and stays untested here, matching the pre-existing
# run_hft_analytics precedent.
# ---------------------------------------------------------------------------


def _install_fake_dbg_cdm(monkeypatch: pytest.MonkeyPatch, get_market_segment_details) -> None:
    fake_a7_utils = types.ModuleType("dbg_cdm.a7_utils")
    fake_a7_utils.get_market_segment_details = get_market_segment_details
    fake_dbg_cdm = types.ModuleType("dbg_cdm")
    fake_dbg_cdm.a7_utils = fake_a7_utils
    monkeypatch.setitem(sys.modules, "dbg_cdm", fake_dbg_cdm)
    monkeypatch.setitem(sys.modules, "dbg_cdm.a7_utils", fake_a7_utils)


def _instrument_snapshot(security_id: int, isin: str | None) -> dict:
    security_alt = [{"SecurityAltIDSource": "4", "SecurityAltID": isin}] if isin else []
    return {
        "Template": "InstrumentSnapshot",
        "SecurityID": security_id,
        "SecurityAlt": security_alt,
    }


def _row(**overrides: Any) -> dict[str, Any]:
    row = {
        "snapshot_date": date(2023, 1, 3),
        "name": "SAP SE",
        "ticker": "SAP",
        "isin": "DE0007164600",
        "cusip": None,
        "dbga_secid": None,
    }
    row.update(overrides)
    return row


def test_scan_market_segment_isins_builds_reverse_map(monkeypatch: pytest.MonkeyPatch) -> None:
    snapshots = [
        {"Template": "ProductSnapshot"},  # non-instrument snapshots are ignored
        _instrument_snapshot(204934, "DE0007164600"),
        _instrument_snapshot(204935, "US0378331005"),
        _instrument_snapshot(204936, None),  # no SecurityAlt at all -- skipped
    ]
    _install_fake_dbg_cdm(monkeypatch, lambda mic, ccyymmdd, market_segment_id: snapshots)

    result = scan_market_segment_isins("XETR", 20230103, 688)
    assert result == {"DE0007164600": 204934, "US0378331005": 204935}


def test_backfill_dbga_secid_fills_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    snapshots = [_instrument_snapshot(204934, "DE0007164600")]
    _install_fake_dbg_cdm(monkeypatch, lambda mic, ccyymmdd, market_segment_id: snapshots)

    rows = [_row()]
    backfilled = backfill_dbga_secid(rows, "XETR", 688)
    assert backfilled[0]["dbga_secid"] == "204934"


def test_backfill_dbga_secid_never_overwrites_existing_value() -> None:
    rows = [_row(dbga_secid="ALREADY_SET")]
    # No fake dbg_cdm installed -- if this tried to scan, it would raise
    # ModuleNotFoundError, proving the no-candidates short-circuit works.
    backfilled = backfill_dbga_secid(rows, "XETR", 688)
    assert backfilled[0]["dbga_secid"] == "ALREADY_SET"


def test_backfill_dbga_secid_isin_not_in_scan_stays_none(monkeypatch: pytest.MonkeyPatch) -> None:
    snapshots = [_instrument_snapshot(204934, "US0378331005")]  # different ISIN
    _install_fake_dbg_cdm(monkeypatch, lambda mic, ccyymmdd, market_segment_id: snapshots)

    rows = [_row()]
    backfilled = backfill_dbga_secid(rows, "XETR", 688)
    assert backfilled[0]["dbga_secid"] is None


def test_backfill_dbga_secid_no_isin_rows_skips_without_scanning() -> None:
    rows = [_row(isin=None, cusip="717081103")]
    # No fake dbg_cdm installed -- would raise if a scan were attempted.
    backfilled = backfill_dbga_secid(rows, "XETR", 688)
    assert backfilled == rows


def test_backfill_dbga_secid_groups_by_snapshot_date(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def fake_get_market_segment_details(mic, ccyymmdd, market_segment_id):
        calls.append(ccyymmdd)
        return [_instrument_snapshot(204934, "DE0007164600")]

    _install_fake_dbg_cdm(monkeypatch, fake_get_market_segment_details)

    rows = [
        _row(snapshot_date=date(2023, 1, 3)),
        _row(snapshot_date=date(2023, 1, 3), ticker="SAP2"),
        _row(snapshot_date=date(2023, 2, 1)),
    ]
    backfilled = backfill_dbga_secid(rows, "XETR", 688)
    assert all(r["dbga_secid"] == "204934" for r in backfilled)
    assert calls == [20230103, 20230201]
