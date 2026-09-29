"""wrds_client.indices: index constituents from WRDS, with a scripted stub client (no
server)."""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from universe.index_universe import index_universe_rows
from wrds_client.indices import (
    IndexNotAvailable,
    build_constituents,
    membership_spells,
    search_indices,
    snapshot_dates,
    snapshots,
    write_constituents,
)
from wrds_client.indices.constituents import cusip_check_digit


class Stub:
    """Canned frames by SQL substring; ``deny`` substrings raise a permission error."""

    def __init__(self, scripts: dict[str, pd.DataFrame], deny: tuple[str, ...] = ()):
        self.scripts, self.deny, self.calls = scripts, deny, []

    def raw_sql(self, sql: str, **kw: Any) -> pd.DataFrame:
        self.calls.append((sql, kw.get("params")))
        if any(d in sql for d in self.deny):
            raise RuntimeError("(psycopg2.errors.InsufficientPrivilege) permission denied "
                               "for schema crsp")
        for key, df in self.scripts.items():
            if key in sql:
                return df.copy()
        return pd.DataFrame()


def _catalogue(rows):
    return pd.DataFrame(rows, columns=["gvkeyx", "conm", "idxcstflg"])


NA_EMPTY = _catalogue([])
STOXX = _catalogue([("150376", "STOXX 600 Price Index", "Y")])
MSCI = _catalogue([("150066", "MSCI - World Index (12/31/69)", "N")])


def test_search_matches_every_word_and_reports_unlicensed_sources():
    client = Stub({"FROM comp.idx_index": NA_EMPTY, "FROM comp.g_idx_index": MSCI},
                  deny=("crsp.",))
    found = search_indices(client, "MSCI World")
    assert found[["source", "index_id", "has_constituents"]].values.tolist()[0] == [
        "compustat", "150066", False]
    crsp = found[found["source"] == "crsp"].iloc[0]
    assert pd.isna(crsp["index_id"]) and "not licensed" in crsp["name"]
    sql, params = next(c for c in client.calls if "g_idx_index" in c[0])
    assert "conm ILIKE %(w0)s AND conm ILIKE %(w1)s" in sql
    assert params == {"w0": "%MSCI%", "w1": "%World%"}


def test_index_without_constituents_says_why():
    client = Stub({"FROM comp.idx_index": NA_EMPTY, "FROM comp.g_idx_index": MSCI})
    with pytest.raises(IndexNotAvailable, match="no WRDS licence changes that"):
        membership_spells(client, "compustat", "150066")
    with pytest.raises(IndexNotAvailable, match="no index with gvkeyx"):
        membership_spells(Stub({}), "compustat", "999999")


def test_unlicensed_spell_table_raises_not_licensed():
    with pytest.raises(IndexNotAvailable, match="not licensed"):
        membership_spells(Stub({}, deny=("crsp.",)), "crsp", "1000502")


SPELLS = pd.DataFrame({
    "gvkey": ["001", "002", "003", "004"], "iid": ["01W", "01W", "01W", "02W"],
    "start": [date(2000, 1, 1), date(2020, 2, 3), date(2000, 1, 1), date(2000, 1, 1)],
    "end": [None, None, date(2020, 1, 31), date(2020, 2, 3)],
})


def test_spells_to_monthly_snapshots_boundaries_inclusive():
    dates = snapshot_dates(date(2020, 1, 1), date(2020, 3, 31))
    assert dates == [date(2020, 1, 1), date(2020, 2, 3), date(2020, 3, 2)]    # business days
    got = snapshots(SPELLS, dates)
    members = got.groupby("snapshot_date")["gvkey"].apply(list).to_dict()
    assert members == {date(2020, 1, 1): ["001", "003", "004"],
                       date(2020, 2, 3): ["001", "002", "004"],     # start and end inclusive
                       date(2020, 3, 2): ["001", "002"]}


def test_current_members_only_is_warned(caplog):
    spells = pd.DataFrame({"gvkey": ["001"], "iid": ["01"], "start": [date(1990, 1, 1)],
                           "end": [None]})
    client = Stub({"FROM comp.idx_index": _catalogue([("000003", "S&P 500 Comp-Ltd", "Y")]),
                   "FROM comp.idxcst_his": spells})
    with caplog.at_level(logging.WARNING):
        membership_spells(client, "compustat", "000003")
    assert "survivorship" in caplog.text


def _compustat_client():
    return Stub({
        "FROM comp.idx_index": NA_EMPTY, "FROM comp.g_idx_index": STOXX,
        "FROM comp.g_idxcst_his": SPELLS,          # the SQL aliases from/thru to start/end
        "FROM comp.security ": pd.DataFrame(columns=["gvkey", "iid", "isin", "cusip", "sedol",
                                                     "tic"]),
        "FROM comp.g_security": pd.DataFrame({
            "gvkey": ["001", "002", "003", "004"], "iid": ["01W"] * 3 + ["02W"],
            "isin": ["GB0000000001", None, "FR0000000003", "DE0000000004"],
            "cusip": [None] * 4, "sedol": ["B000001", "B000002", None, None],
            "tic": [None] * 4}),
        "FROM comp.company ": pd.DataFrame(columns=["gvkey", "conm"]),
        "FROM comp.g_company": pd.DataFrame({"gvkey": ["001", "002", "003", "004"],
                                             "conm": ["A PLC", "B SA", "C SA", "D AG"]}),
    })


def test_compustat_constituents_and_loader_round_trip(tmp_path: Path):
    frame = build_constituents(_compustat_client(), "compustat", "150376",
                               date(2020, 1, 1), date(2020, 3, 31))
    assert list(frame.columns) == ["snapshot_date", "name", "ticker", "isin", "cusip", "sedol",
                                   "gvkey", "iid"]
    assert len(frame) == 8
    row = frame[(frame["gvkey"] == "004")].iloc[0]
    assert (row["name"], row["isin"], row["iid"]) == ("D AG", "DE0000000004", "02W")
    path = write_constituents(frame, tmp_path, "STOXX600")
    assert path.name == "STOXX600_constituents-20200101_to_20200302.parquet"
    with pytest.raises(FileExistsError):
        write_constituents(frame, tmp_path, "STOXX600")
    rows = index_universe_rows(tmp_path, "STOXX600")
    assert len(rows) == 8 and {r["gvkey"] for r in rows} == {"001", "002", "003", "004"}
    with pytest.raises(ValueError, match="index name"):
        write_constituents(frame, tmp_path, "STOXX 600")


def test_crsp_names_are_point_in_time():
    spells = pd.DataFrame({"permno": [14593.0], "start": [date(1982, 11, 30)],
                           "end": [date(9999, 12, 31)]})                  # CRSP open end
    names = pd.DataFrame({
        "permno": [14593.0, 14593.0], "namedt": [date(1980, 12, 12), date(2007, 1, 10)],
        "nameenddt": [date(2007, 1, 9), date(2024, 12, 31)],
        "ncusip": ["03783310", "03783310"], "ticker": ["AAPL", "AAPL"],
        "comnam": ["APPLE COMPUTER INC", "APPLE INC"]})
    client = Stub({"crsp.stkindmembership_ind": spells, "crsp.stocknames": names})
    frame = build_constituents(client, "crsp", "1000502", date(2006, 12, 1), date(2007, 2, 28))
    assert frame["name"].tolist() == ["APPLE COMPUTER INC", "APPLE COMPUTER INC", "APPLE INC"]
    assert set(frame["cusip"]) == {"037833100"} and frame["isin"].isna().all()
    assert frame["permno"].tolist() == [14593] * 3


@pytest.mark.parametrize("cusip8,check", [("03783310", "0"), ("59491810", "4"),
                                          ("38259P50", "8"), ("G5960L10", "3")])
def test_cusip_check_digit(cusip8, check):
    assert cusip_check_digit(cusip8) == check
