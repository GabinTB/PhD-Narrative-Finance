"""Tests for deutsche_boerse.identifiers: security_id -> ISIN -> universe enrichment.

`enrich_with_universe` is pure (no dbg_cdm/API dependency) and tested
directly. `resolve_isin` is I/O (wraps `dbg_cdm.a7_utils.get_isin`) and not
unit-tested here, matching this module's existing convention of not testing
dbg_cdm-dependent I/O functions directly (see
`analytics/hft_event_detection.py`'s module docstring).
"""
from __future__ import annotations

import pandas as pd
import polars as pl

from deutsche_boerse.identifiers import enrich_with_universe


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
