"""Tests for wrds_client.linking.optionmetrics: permno->secid + the
resolve_universe_secids orchestrator.

No real WRDS server involved: query builders are pure; resolve_universe_secids
is tested end-to-end against a fully-stubbed client (canned frames for every
hop of comp.security/comp.company/ccmxpf_linktable/opcrsphist).
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from datalake import DatalakeIndex
from universe.schema import UniverseEntry
from wrds_client.linking.optionmetrics import (
    build_opcrsphist_query,
    ingest_secid_resolution,
    resolve_universe_secids,
)

PIPELINE = "PhD-Narrative-Finance"
VERSION = "v0.1.0"
_SNAPSHOT_DATE = date(2023, 1, 1)


def _entry(**overrides: Any) -> UniverseEntry:
    """A UniverseEntry with mandatory fields defaulted, overridable per test."""
    kwargs = {"snapshot_date": _SNAPSHOT_DATE, "name": "Test Co", "ticker": "TST"}
    kwargs.update(overrides)
    return UniverseEntry(**kwargs)


@pytest.fixture
def index(tmp_path: Path) -> DatalakeIndex:
    with DatalakeIndex(tmp_path / "datalake") as idx:
        yield idx


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


_EMPTY_LINK = pd.DataFrame(
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
)


def _link_row(gvkey: str, permno: float, start: date, end: date | None) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gvkey": [gvkey],
            "lpermno": [permno],
            "lpermco": [1.0],
            "linkprim": ["P"],
            "linktype": ["LU"],
            "usedflag": [1],
            "linkdt": [start],
            "linkenddt": [end],
        }
    )


def _opcrsphist_row(
    permno: int, secid: float, start: date, end: date | None, score: float = 1.0
) -> pd.DataFrame:
    return pd.DataFrame(
        {"secid": [secid], "permno": [permno], "sdate": [start], "edate": [end], "score": [score]}
    )


# ---------------------------------------------------------------------------
# build_opcrsphist_query
# ---------------------------------------------------------------------------


def test_build_opcrsphist_query_binds_params() -> None:
    sql, params = build_opcrsphist_query([14593], date(2020, 1, 1), date(2020, 12, 31))
    assert "%(permnos)s" in sql
    assert params["permnos"] == (14593,)


def test_build_opcrsphist_query_empty_permnos_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        build_opcrsphist_query([], date(2020, 1, 1), date(2020, 12, 31))


# ---------------------------------------------------------------------------
# resolve_universe_secids
# ---------------------------------------------------------------------------


def test_resolve_universe_secids_golden_path() -> None:
    client = ScriptedWRDSClient(
        [
            ("comp.security", pd.DataFrame({"isin": ["US0378331005"], "gvkey": ["001690"]})),
            ("usedflag = 1", _link_row("001690", 14593.0, date(1980, 12, 12), None)),
            ("opcrsphist", _opcrsphist_row(14593, 101594.0, date(1996, 1, 2), None)),
        ]
    )
    universe = [_entry(isin="US0378331005", cik="0000320193", ticker="AAPL")]
    resolved = resolve_universe_secids(client, universe, date(2020, 1, 1), date(2020, 12, 31))

    assert len(resolved) == 1
    row = resolved.iloc[0]
    assert row["secid"] == 101594
    assert row["gvkey"] == "001690"
    assert row["permno"] == 14593
    assert row["valid_start"] == date(2020, 1, 1)
    assert row["valid_end"] == date(2020, 12, 31)
    assert row["flags"] == []


def test_resolve_universe_secids_unresolved_entry_is_surfaced_not_dropped() -> None:
    client = ScriptedWRDSClient(
        [
            ("comp.security", pd.DataFrame({"isin": [], "gvkey": []})),
            ("comp.company", pd.DataFrame({"cik": [], "gvkey": []})),
        ]
    )
    universe = [_entry(isin="XX0000000000", cik="0000000000", name="ghost")]
    resolved = resolve_universe_secids(client, universe, date(2020, 1, 1), date(2020, 12, 31))

    assert len(resolved) == 1
    row = resolved.iloc[0]
    assert row["flags"] == ["unresolved"]
    assert row["secid"] is None


def test_resolve_universe_secids_link_change_mid_range_produces_two_rows() -> None:
    """One input, two distinct secid windows covering adjacent, non-overlapping
    sub-ranges of the query range -- the point-in-time correctness case."""
    client = ScriptedWRDSClient(
        [
            ("comp.security", pd.DataFrame({"isin": ["US0378331005"], "gvkey": ["001690"]})),
            ("usedflag = 1", _link_row("001690", 14593.0, date(1980, 12, 12), None)),
            (
                "opcrsphist",
                pd.concat(
                    [
                        _opcrsphist_row(14593, 101594.0, date(1996, 1, 2), date(2020, 6, 30)),
                        pd.DataFrame(
                            {
                                "secid": [999999.0],
                                "permno": [14593],
                                "sdate": [date(2020, 7, 1)],
                                "edate": [None],
                                "score": [1.0],
                            }
                        ),
                    ],
                    ignore_index=True,
                ),
            ),
        ]
    )
    universe = [_entry(isin="US0378331005", ticker="AAPL")]
    resolved = resolve_universe_secids(client, universe, date(2020, 1, 1), date(2020, 12, 31))

    assert len(resolved) == 2
    resolved_sorted = resolved.sort_values("valid_start").reset_index(drop=True)
    assert resolved_sorted.loc[0, "secid"] == 101594
    assert resolved_sorted.loc[0, "valid_start"] == date(2020, 1, 1)
    assert resolved_sorted.loc[0, "valid_end"] == date(2020, 6, 30)
    assert resolved_sorted.loc[1, "secid"] == 999999
    assert resolved_sorted.loc[1, "valid_start"] == date(2020, 7, 1)
    assert resolved_sorted.loc[1, "valid_end"] == date(2020, 12, 31)
    # Adjacent, non-overlapping, covering the full input range.
    assert resolved_sorted.loc[1, "valid_start"] > resolved_sorted.loc[0, "valid_end"]


def test_resolve_universe_secids_ambiguous_gvkey_flagged() -> None:
    client = ScriptedWRDSClient(
        [
            (
                "comp.security",
                pd.DataFrame(
                    {"isin": ["US0378331005", "US0378331005"], "gvkey": ["001690", "999999"]}
                ),
            ),
            ("usedflag = 1", _link_row("001690", 14593.0, date(1980, 12, 12), None)),
            ("crsp_a_ccm.ccmxpf_linktable", _link_row("999999", 55555.0, date(1980, 1, 1), None)),
            (
                "opcrsphist",
                pd.concat(
                    [
                        _opcrsphist_row(14593, 101594.0, date(1996, 1, 2), None),
                        _opcrsphist_row(55555, 202020.0, date(1996, 1, 2), None),
                    ],
                    ignore_index=True,
                ),
            ),
        ]
    )
    universe = [_entry(isin="US0378331005")]
    resolved = resolve_universe_secids(client, universe, date(2020, 1, 1), date(2020, 12, 31))

    assert len(resolved) == 2
    assert all("ambiguous_gvkey" in flags for flags in resolved["flags"])


def test_resolve_universe_secids_end_before_start_raises() -> None:
    client = ScriptedWRDSClient([])
    universe = [_entry(isin="X")]
    with pytest.raises(ValueError, match="before start_date"):
        resolve_universe_secids(client, universe, date(2020, 6, 1), date(2020, 1, 1))


# ---------------------------------------------------------------------------
# ingest_secid_resolution
# ---------------------------------------------------------------------------


def test_ingest_secid_resolution_registers_artifact(index: DatalakeIndex) -> None:
    client = ScriptedWRDSClient(
        [
            ("comp.security", pd.DataFrame({"isin": ["US0378331005"], "gvkey": ["001690"]})),
            ("usedflag = 1", _link_row("001690", 14593.0, date(1980, 12, 12), None)),
            ("opcrsphist", _opcrsphist_row(14593, 101594.0, date(1996, 1, 2), None)),
        ]
    )
    universe = [_entry(isin="US0378331005", cik="0000320193", ticker="AAPL")]

    artifact = ingest_secid_resolution(
        index,
        client,
        universe,
        date(2020, 1, 1),
        date(2020, 12, 31),
        pipeline=PIPELINE,
        pipeline_version=VERSION,
    )

    assert not artifact.partial
    assert artifact.meta.hyperparams["isins"] == ["US0378331005"]
    assert artifact.meta.hyperparams["ciks"] == ["0000320193"]
    files = artifact.files("*.parquet")
    written = pd.read_parquet(files[0])
    assert written.loc[0, "secid"] == 101594
