"""Tests for wrds_client.lseg.mapping: LSEG SEDOL backfill.

No real WRDS server involved: query builders are pure; resolve/fetch/backfill
functions are tested against a stub client with canned responses. Rows are
plain dicts; canned reference frames include startdate/enddate (None = open-
ended) since matching now happens locally via `match_as_of_windows`, not a
server-side as_of filter.
"""
from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
import pytest

from wrds_client.lseg.mapping import (
    backfill_sedol,
    build_cusip_instrpermid_query,
    build_isin_instrpermid_query,
    fetch_primary_quotepermid,
    fetch_sedol,
    resolve_instrpermids,
)

_DATE = date(2023, 1, 1)


def _row(**overrides: Any) -> dict[str, Any]:
    row = {
        "snapshot_date": _DATE,
        "name": "Apple Inc",
        "ticker": "AAPL",
        "isin": "US0378331005",
        "cusip": None,
        "sedol": None,
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


# ---------------------------------------------------------------------------
# query builders
# ---------------------------------------------------------------------------


def test_build_isin_instrpermid_query_binds_params() -> None:
    sql, params = build_isin_instrpermid_query(["us0378331005"])
    assert "%(isins)s" in sql
    assert params == {"isins": ("US0378331005",)}


def test_build_isin_instrpermid_query_empty_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        build_isin_instrpermid_query([])


def test_build_cusip_instrpermid_query_binds_params() -> None:
    sql, params = build_cusip_instrpermid_query([" 037833100 "])
    assert params == {"cusips": ("037833100",)}


# ---------------------------------------------------------------------------
# resolve_instrpermids
# ---------------------------------------------------------------------------


def test_resolve_instrpermids_isin_hit() -> None:
    df = pd.DataFrame(
        {"isin": ["US0378331005"], "instrpermid": [555], "startdate": [None], "enddate": [None]}
    )
    client = ScriptedWRDSClient([("permisindata", df)])
    result = resolve_instrpermids(client, [_row()])
    assert result == {0: 555}


def test_resolve_instrpermids_falls_back_to_cusip() -> None:
    empty = pd.DataFrame({"isin": [], "instrpermid": [], "startdate": [], "enddate": []})
    cusip_hit = pd.DataFrame(
        {"cusip": ["037833100"], "instrpermid": [555], "startdate": [None], "enddate": [None]}
    )
    client = ScriptedWRDSClient([("permisindata", empty), ("permcusipdata", cusip_hit)])
    row = _row(isin="XX0000000000", cusip="037833100")
    result = resolve_instrpermids(client, [row])
    assert result == {0: 555}


def test_resolve_instrpermids_picks_window_covering_row_snapshot_date() -> None:
    df = pd.DataFrame(
        {
            "isin": ["US0378331005", "US0378331005"],
            "instrpermid": [111, 222],
            "startdate": [date(2000, 1, 1), date(2020, 1, 1)],
            "enddate": [date(2019, 12, 31), None],
        }
    )
    client = ScriptedWRDSClient([("permisindata", df)])
    result = resolve_instrpermids(client, [_row(snapshot_date=date(2010, 1, 1))])
    assert result == {0: 111}


def test_resolve_instrpermids_no_match_empty_dict() -> None:
    empty = pd.DataFrame({"isin": [], "instrpermid": [], "startdate": [], "enddate": []})
    client = ScriptedWRDSClient([("permisindata", empty)])
    row = _row(isin="XX0000000000", cusip=None)
    assert resolve_instrpermids(client, [row]) == {}


def test_resolve_instrpermids_multiple_snapshot_dates_issue_one_query() -> None:
    df = pd.DataFrame(
        {
            "isin": ["A", "B"],
            "instrpermid": [1, 2],
            "startdate": [None, None],
            "enddate": [None, None],
        }
    )
    client = ScriptedWRDSClient([("permisindata", df)])
    rows = [
        _row(isin="A", snapshot_date=date(2023, 1, 1)),
        _row(isin="B", snapshot_date=date(2023, 6, 1)),
    ]
    result = resolve_instrpermids(client, rows)
    assert result == {0: 1, 1: 2}
    assert len(client.calls) == 1


# ---------------------------------------------------------------------------
# fetch_* helpers
# ---------------------------------------------------------------------------


def test_fetch_primary_quotepermid() -> None:
    df = pd.DataFrame({"instrpermid": [555], "valquotepermid": [777]})
    client = ScriptedWRDSClient([("perminstrref", df)])
    result = fetch_primary_quotepermid(client, [555])
    assert result.iloc[0]["valquotepermid"] == 777


def test_fetch_primary_quotepermid_drops_null_quotepermid() -> None:
    df = pd.DataFrame({"instrpermid": [555], "valquotepermid": [None]})
    client = ScriptedWRDSClient([("perminstrref", df)])
    result = fetch_primary_quotepermid(client, [555])
    assert result.empty


def test_fetch_sedol() -> None:
    df = pd.DataFrame(
        {"quotepermid": [777], "sedol": ["2046251"], "startdate": [None], "enddate": [None]}
    )
    client = ScriptedWRDSClient([("permsedoldata", df)])
    rows = [_row()]
    result = fetch_sedol(client, rows, {0: 777})
    assert result.loc[0, "sedol"] == "2046251"


# ---------------------------------------------------------------------------
# backfill_sedol
# ---------------------------------------------------------------------------


def _full_stub_client() -> ScriptedWRDSClient:
    return ScriptedWRDSClient(
        [
            (
                "permisindata",
                pd.DataFrame(
                    {
                        "isin": ["US0378331005"],
                        "instrpermid": [555],
                        "startdate": [None],
                        "enddate": [None],
                    }
                ),
            ),
            ("perminstrref", pd.DataFrame({"instrpermid": [555], "valquotepermid": [777]})),
            (
                "permsedoldata",
                pd.DataFrame(
                    {
                        "quotepermid": [777],
                        "sedol": ["2046251"],
                        "startdate": [None],
                        "enddate": [None],
                    }
                ),
            ),
        ]
    )


def test_backfill_sedol_fills_missing_sedol() -> None:
    client = _full_stub_client()
    rows = [_row()]
    backfilled = backfill_sedol(client, rows)
    assert backfilled[0]["sedol"] == "2046251"


def test_backfill_sedol_never_overwrites_existing_value() -> None:
    client = ScriptedWRDSClient([])  # no queries expected
    rows = [_row(sedol="ALREADY_SET")]
    backfilled = backfill_sedol(client, rows)
    assert backfilled[0]["sedol"] == "ALREADY_SET"
    assert client.calls == []


def test_backfill_sedol_unresolved_row_is_unchanged() -> None:
    empty = pd.DataFrame({"isin": [], "instrpermid": [], "startdate": [], "enddate": []})
    client = ScriptedWRDSClient([("permisindata", empty)])
    row = _row(isin="XX0000000000", cusip=None)
    backfilled = backfill_sedol(client, [row])
    assert backfilled == [row]


def test_backfill_sedol_mixed_rows_only_queries_for_missing() -> None:
    client = _full_stub_client()
    rows = [_row(sedol="PRESET"), _row(isin="US0378331005", ticker="AAPL2")]
    backfilled = backfill_sedol(client, rows)
    assert backfilled[0]["sedol"] == "PRESET"
    assert backfilled[1]["sedol"] == "2046251"


def test_backfill_sedol_does_not_mutate_input_rows() -> None:
    client = _full_stub_client()
    row = _row()
    backfill_sedol(client, [row])
    assert row["sedol"] is None
