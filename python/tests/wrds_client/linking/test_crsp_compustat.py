"""Tests for wrds_client.linking.crsp_compustat: isin/cik -> gvkey -> permno.

No real WRDS server involved: query builders are pure, resolve_gvkeys/
fetch_permno_windows are tested against a stub client with canned responses.
"""
from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
import pytest

from universe.schema import UniverseEntry
from wrds_client.linking.crsp_compustat import (
    build_ccm_link_query,
    build_cik_gvkey_query,
    build_isin_gvkey_query,
    fetch_permno_windows,
    resolve_gvkeys,
)

_SNAPSHOT_DATE = date(2023, 1, 1)


def _entry(**overrides: Any) -> UniverseEntry:
    """A UniverseEntry with mandatory fields defaulted, overridable per test."""
    kwargs = {"snapshot_date": _SNAPSHOT_DATE, "name": "Test Co", "ticker": "TST"}
    kwargs.update(overrides)
    return UniverseEntry(**kwargs)


class ScriptedWRDSClient:
    """Returns a canned DataFrame based on a substring match against the SQL text.

    `scripts` is checked in order; the first entry whose `match` substring is
    in the query is used and popped, so repeated calls with different SQL can
    be scripted precisely (e.g. Pass A returning empty, Pass B non-empty).
    """

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


# ---------------------------------------------------------------------------
# query builders
# ---------------------------------------------------------------------------


def test_build_isin_gvkey_query_binds_params() -> None:
    sql, params = build_isin_gvkey_query(["us0378331005", " CA03785Y1007 "])
    assert "%(isins)s" in sql
    assert params == {"isins": ("US0378331005", "CA03785Y1007")}


def test_build_isin_gvkey_query_empty_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        build_isin_gvkey_query([])


def test_build_cik_gvkey_query_zero_pads() -> None:
    sql, params = build_cik_gvkey_query(["320193"])
    assert params == {"ciks": ("0000320193",)}


def test_build_ccm_link_query_preferred_only_adds_filters() -> None:
    sql, params = build_ccm_link_query(
        ["001690"], date(2020, 1, 1), date(2020, 12, 31), preferred_only=True
    )
    assert "linkprim IN ('P', 'C')" in sql
    assert "usedflag = 1" in sql
    assert params["gvkeys"] == ("001690",)


def test_build_ccm_link_query_unfiltered_omits_preferred_filters() -> None:
    sql, _ = build_ccm_link_query(
        ["001690"], date(2020, 1, 1), date(2020, 12, 31), preferred_only=False
    )
    assert "usedflag = 1" not in sql
    assert "linkprim IN" not in sql


def test_build_ccm_link_query_empty_gvkeys_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        build_ccm_link_query([], date(2020, 1, 1), date(2020, 12, 31), preferred_only=True)


def test_build_ccm_link_query_end_before_start_raises() -> None:
    with pytest.raises(ValueError, match="before start_date"):
        build_ccm_link_query(["001690"], date(2020, 6, 1), date(2020, 1, 1), preferred_only=True)


# ---------------------------------------------------------------------------
# resolve_gvkeys
# ---------------------------------------------------------------------------


def test_resolve_gvkeys_isin_hit() -> None:
    client = ScriptedWRDSClient(
        [("comp.security", pd.DataFrame({"isin": ["US0378331005"], "gvkey": ["001690"]}))]
    )
    universe = [_entry(isin="US0378331005", cik="0000320193", ticker="AAPL")]
    results = resolve_gvkeys(client, universe)
    assert len(results) == 1
    r = results[0]
    assert r.input_index == 0
    assert r.gvkey == "001690"
    assert r.resolution_method == "isin"
    assert r.ambiguous is False


def test_resolve_gvkeys_falls_back_to_cik_when_isin_misses() -> None:
    client = ScriptedWRDSClient(
        [
            ("comp.security", pd.DataFrame({"isin": [], "gvkey": []})),
            ("comp.company", pd.DataFrame({"cik": ["0000320193"], "gvkey": ["001690"]})),
        ]
    )
    universe = [_entry(isin="XX0000000000", cik="320193", ticker="AAPL")]
    results = resolve_gvkeys(client, universe)
    assert len(results) == 1
    assert results[0].resolution_method == "cik"
    assert results[0].gvkey == "001690"


def test_resolve_gvkeys_isin_ambiguous_match_flags_all_rows() -> None:
    client = ScriptedWRDSClient(
        [
            (
                "comp.security",
                pd.DataFrame(
                    {"isin": ["US0378331005", "US0378331005"], "gvkey": ["001690", "999999"]}
                ),
            )
        ]
    )
    universe = [_entry(isin="US0378331005")]
    results = resolve_gvkeys(client, universe)
    assert len(results) == 2
    assert {r.gvkey for r in results} == {"001690", "999999"}
    assert all(r.ambiguous for r in results)
    assert all(r.input_index == 0 for r in results)


def test_resolve_gvkeys_no_match_produces_no_rows() -> None:
    client = ScriptedWRDSClient(
        [
            ("comp.security", pd.DataFrame({"isin": [], "gvkey": []})),
            ("comp.company", pd.DataFrame({"cik": [], "gvkey": []})),
        ]
    )
    universe = [_entry(isin="XX0000000000", cik="0000000000")]
    results = resolve_gvkeys(client, universe)
    assert results == []


def test_resolve_gvkeys_no_identifiers_produces_no_rows_and_no_queries() -> None:
    client = ScriptedWRDSClient([])
    results = resolve_gvkeys(client, [_entry(isin=None, cusip="000000000", name="mystery")])
    assert results == []
    assert client.calls == []


def test_resolve_gvkeys_preserves_input_index_across_multiple_entries() -> None:
    client = ScriptedWRDSClient(
        [("comp.security", pd.DataFrame({"isin": ["ISIN_B"], "gvkey": ["222"]}))]
    )
    universe = [_entry(isin="ISIN_A"), _entry(isin="ISIN_B")]
    results = resolve_gvkeys(client, universe)
    assert len(results) == 1
    assert results[0].input_index == 1


# ---------------------------------------------------------------------------
# fetch_permno_windows
# ---------------------------------------------------------------------------


def test_fetch_permno_windows_pass_a_only() -> None:
    client = ScriptedWRDSClient(
        [
            (
                "usedflag = 1",
                pd.DataFrame(
                    {
                        "gvkey": ["001690"],
                        "lpermno": [14593.0],
                        "lpermco": [7.0],
                        "linkprim": ["P"],
                        "linktype": ["LU"],
                        "usedflag": [1],
                        "linkdt": [date(1980, 12, 12)],
                        "linkenddt": [None],
                    }
                ),
            )
        ]
    )
    windows = fetch_permno_windows(client, ["001690"], date(2020, 1, 1), date(2020, 12, 31))
    assert len(windows) == 1
    w = windows[0]
    assert w.permno == 14593
    assert w.preferred is True
    assert w.valid_start == date(2020, 1, 1)
    assert w.valid_end == date(2020, 12, 31)  # open-ended, clipped to query end
    assert len(client.calls) == 1  # no Pass B needed


def test_fetch_permno_windows_handles_real_timestamp_and_nat_columns() -> None:
    """Regression: raw_sql(date_cols=...) returns pandas Timestamp/NaT, not
    plain date/None -- a stub DataFrame built with plain date objects (as the
    other tests here do, for readability) doesn't exercise that. This builds
    the frame the way pandas.read_sql_query(parse_dates=...) actually would.
    """
    df = pd.DataFrame(
        {
            "gvkey": ["001690"],
            "lpermno": [14593.0],
            "lpermco": [7.0],
            "linkprim": ["P"],
            "linktype": ["LU"],
            "usedflag": [1],
            "linkdt": pd.to_datetime(["1980-12-12"]),
            "linkenddt": pd.to_datetime([None]),
        }
    )
    client = ScriptedWRDSClient([("usedflag = 1", df)])
    windows = fetch_permno_windows(client, ["001690"], date(2020, 1, 1), date(2020, 12, 31))
    assert len(windows) == 1
    assert windows[0].valid_start == date(2020, 1, 1)
    assert windows[0].valid_end == date(2020, 12, 31)


def test_fetch_permno_windows_falls_back_to_pass_b_for_uncovered_gvkeys() -> None:
    client = ScriptedWRDSClient(
        [
            (
                "usedflag = 1",
                pd.DataFrame(
                    {
                        "gvkey": [],
                        "lpermno": [],
                        "lpermco": [],
                        "linkprim": [],
                        "linktype": [],
                        "usedflag": [],
                        "linkdt": [],
                        "linkenddt": [],
                    }
                ),
            ),
            (
                "crsp_a_ccm.ccmxpf_linktable",
                pd.DataFrame(
                    {
                        "gvkey": ["001690"],
                        "lpermno": [14593.0],
                        "lpermco": [7.0],
                        "linkprim": ["C"],
                        "linktype": ["NU"],
                        "usedflag": [-1],
                        "linkdt": [date(1979, 10, 1)],
                        "linkenddt": [date(1980, 12, 11)],
                    }
                ),
            ),
        ]
    )
    windows = fetch_permno_windows(client, ["001690"], date(1980, 1, 1), date(1980, 12, 31))
    assert len(windows) == 1
    assert windows[0].preferred is False
    assert len(client.calls) == 2


def test_fetch_permno_windows_empty_gvkeys_returns_empty_without_querying() -> None:
    client = ScriptedWRDSClient([])
    assert fetch_permno_windows(client, [], date(2020, 1, 1), date(2020, 12, 31)) == []
    assert client.calls == []
