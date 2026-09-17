"""Tests for universe.schema: UniverseEntry validation and CSV/DataFrame/parquet round-trips."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from universe.schema import (
    UniverseEntry,
    from_frame,
    load_universe_csv,
    load_universe_file,
    load_universe_parquet,
    to_frame,
)

_DATE = date(2023, 1, 1)


def _entry(**overrides: object) -> UniverseEntry:
    kwargs = {
        "snapshot_date": _DATE,
        "name": "Apple Inc",
        "ticker": "AAPL",
        "isin": "US0378331005",
    }
    kwargs.update(overrides)
    return UniverseEntry(**kwargs)


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def test_requires_snapshot_date() -> None:
    with pytest.raises(ValueError, match="snapshot_date"):
        _entry(snapshot_date=None)


def test_requires_name() -> None:
    with pytest.raises(ValueError, match="name"):
        _entry(name=None)


def test_requires_ticker() -> None:
    with pytest.raises(ValueError, match="ticker"):
        _entry(ticker=None)


def test_requires_isin_or_cusip() -> None:
    with pytest.raises(ValueError, match="isin/cusip"):
        _entry(isin=None)


def test_cusip_alone_satisfies_isin_or_cusip() -> None:
    e = _entry(isin=None, cusip="037833100")
    assert e.isin is None
    assert e.cusip == "037833100"


def test_all_optional_columns_default_to_none() -> None:
    e = _entry()
    assert e.cik is None
    assert e.gvkey is None
    assert e.dbga_secid is None
    assert e.gics_sector is None


# ---------------------------------------------------------------------------
# load_universe_csv
# ---------------------------------------------------------------------------


def test_load_universe_csv_mandatory_columns_only(tmp_path: Path) -> None:
    csv = tmp_path / "universe.csv"
    csv.write_text("snapshot_date,name,ticker,isin\n2023-01-01,Apple Inc,AAPL,US0378331005\n")
    entries = load_universe_csv(csv)
    assert entries == [_entry()]


def test_load_universe_csv_with_optional_columns(tmp_path: Path) -> None:
    csv = tmp_path / "universe.csv"
    csv.write_text(
        "snapshot_date,name,ticker,isin,cik,gvkey,dbga_secid,gics_sector\n"
        "2023-01-01,Apple Inc,AAPL,US0378331005,0000320193,001690,12345,Technology\n"
    )
    entries = load_universe_csv(csv)
    assert entries == [
        _entry(cik="0000320193", gvkey="001690", dbga_secid="12345", gics_sector="Technology")
    ]


def test_load_universe_csv_blank_optional_cells_become_none(tmp_path: Path) -> None:
    csv = tmp_path / "universe.csv"
    csv.write_text("snapshot_date,name,ticker,isin,cik\n2023-01-01,Apple Inc,AAPL,US0378331005,\n")
    entries = load_universe_csv(csv)
    assert entries == [_entry(cik=None)]


def test_load_universe_csv_missing_mandatory_column_raises(tmp_path: Path) -> None:
    csv = tmp_path / "universe.csv"
    csv.write_text("name,ticker,isin\nApple Inc,AAPL,US0378331005\n")
    with pytest.raises(ValueError, match="missing mandatory column"):
        load_universe_csv(csv)


def test_load_universe_csv_missing_ticker_value_is_skipped_not_raised(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A row missing a mandatory field is skipped (and logged), not allowed
    to abort the whole load -- real universe data has a real fraction of
    such rows (e.g. LGT's MSCI World metadata: ~3.75% missing ticker)."""
    csv = tmp_path / "universe.csv"
    csv.write_text(
        "snapshot_date,name,ticker,isin\n"
        "2023-01-01,Apple Inc,,US0378331005\n"
        "2023-01-01,Alphabet Inc,GOOGL,US02079K3059\n"
    )
    with caplog.at_level("WARNING"):
        entries = load_universe_csv(csv)
    assert entries == [_entry(name="Alphabet Inc", ticker="GOOGL", isin="US02079K3059")]
    assert "skipped 1/2 rows" in caplog.text
    assert "requires ticker" in caplog.text


def test_load_universe_csv_missing_isin_and_cusip_is_skipped_not_raised(tmp_path: Path) -> None:
    csv = tmp_path / "universe.csv"
    csv.write_text("snapshot_date,name,ticker\n2023-01-01,Apple Inc,AAPL\n")
    assert load_universe_csv(csv) == []


def test_load_universe_csv_multiple_snapshot_dates_same_isin(tmp_path: Path) -> None:
    """The actual "one row = one day" use case: the same constituent observed
    on different snapshot dates, e.g. with a different dbga_secid each time."""
    csv = tmp_path / "universe.csv"
    csv.write_text(
        "snapshot_date,name,ticker,isin,dbga_secid\n"
        "2023-01-01,Apple Inc,AAPL,US0378331005,111\n"
        "2023-02-01,Apple Inc,AAPL,US0378331005,222\n"
    )
    entries = load_universe_csv(csv)
    assert entries == [
        _entry(dbga_secid="111"),
        _entry(snapshot_date=date(2023, 2, 1), dbga_secid="222"),
    ]


# ---------------------------------------------------------------------------
# to_frame / from_frame / parquet
# ---------------------------------------------------------------------------


def test_to_frame_and_from_frame_roundtrip() -> None:
    entries = [
        _entry(cik="0000320193", gvkey="001690"),
        _entry(isin=None, cusip="037833100", name="Alphabet Inc", ticker="GOOGL"),
    ]
    df = to_frame(entries)
    assert "snapshot_date" in df.columns
    assert "gics_subindustry_code" in df.columns
    assert from_frame(df) == entries


def test_from_frame_treats_nan_as_none() -> None:
    df = pd.DataFrame(
        {
            "snapshot_date": [_DATE],
            "name": ["Apple Inc"],
            "ticker": ["AAPL"],
            "isin": ["US0378331005"],
            "cusip": [None],
            "sedol": [None],
            "cik": [None],
            "figi": [None],
            "gvkey": [None],
            "ciq_secid": [None],
            "dbga_secid": [None],
            "country_name": [None],
            "country_iso": [None],
            "region": [None],
            "gics_sector": [None],
            "gics_sector_code": [None],
            "gics_industry_group": [None],
            "gics_industry_group_code": [None],
            "gics_industry": [None],
            "gics_industry_code": [None],
            "gics_subindustry": [None],
            "gics_subindustry_code": [None],
        }
    )
    assert from_frame(df) == [_entry()]


def test_load_universe_parquet_roundtrip(tmp_path: Path) -> None:
    entries = [_entry(gvkey="001690")]
    path = tmp_path / "universe.parquet"
    to_frame(entries).to_parquet(path, index=False)
    assert load_universe_parquet(path) == entries


def test_load_universe_parquet_snapshot_date_survives_timestamp_roundtrip(tmp_path: Path) -> None:
    """Regression: parquet round-trips snapshot_date through pandas Timestamp/NaT,
    not plain date/None (same class of issue hit in wrds_client.linking earlier)."""
    entries = [_entry(), _entry(snapshot_date=date(2023, 6, 15))]
    path = tmp_path / "universe.parquet"
    to_frame(entries).to_parquet(path, index=False)
    loaded = load_universe_parquet(path)
    assert [e.snapshot_date for e in loaded] == [date(2023, 1, 1), date(2023, 6, 15)]


def test_load_universe_parquet_tolerates_partial_optional_columns(tmp_path: Path) -> None:
    """A parquet with only some of _OPTIONAL_COLUMNS (e.g. a freshly-converted,
    not-yet-enriched universe) loads fine -- missing columns default to None
    rather than raising."""
    df = pd.DataFrame(
        {
            "snapshot_date": [_DATE],
            "name": ["Apple Inc"],
            "ticker": ["AAPL"],
            "isin": ["US0378331005"],
        }
    )
    path = tmp_path / "universe.parquet"
    df.to_parquet(path, index=False)
    assert load_universe_parquet(path) == [_entry()]


def test_from_frame_tolerates_partial_optional_columns() -> None:
    df = pd.DataFrame(
        {
            "snapshot_date": [_DATE],
            "name": ["Apple Inc"],
            "ticker": ["AAPL"],
            "isin": ["US0378331005"],
            "gvkey": ["001690"],
        }
    )
    assert from_frame(df) == [_entry(gvkey="001690")]


def test_from_frame_skips_and_logs_rows_missing_mandatory_fields(
    caplog: pytest.LogCaptureFixture,
) -> None:
    df = pd.DataFrame(
        {
            "snapshot_date": [_DATE, _DATE],
            "name": ["Apple Inc", "Alphabet Inc"],
            "ticker": [None, "GOOGL"],
            "isin": ["US0378331005", "US02079K3059"],
        }
    )
    with caplog.at_level("WARNING"):
        entries = from_frame(df, source="test.parquet")
    assert entries == [_entry(name="Alphabet Inc", ticker="GOOGL", isin="US02079K3059")]
    assert "test.parquet: skipped 1/2 rows" in caplog.text


# ---------------------------------------------------------------------------
# load_universe_file dispatch
# ---------------------------------------------------------------------------


def test_load_universe_file_dispatches_csv(tmp_path: Path) -> None:
    csv = tmp_path / "universe.csv"
    csv.write_text("snapshot_date,name,ticker,isin\n2023-01-01,Apple Inc,AAPL,US0378331005\n")
    assert load_universe_file(csv) == [_entry()]


def test_load_universe_file_dispatches_parquet(tmp_path: Path) -> None:
    entries = [_entry()]
    path = tmp_path / "universe.parquet"
    to_frame(entries).to_parquet(path, index=False)
    assert load_universe_file(path) == entries


def test_load_universe_file_dispatches_pq_suffix(tmp_path: Path) -> None:
    entries = [_entry()]
    path = tmp_path / "universe.pq"
    to_frame(entries).to_parquet(path, index=False)
    assert load_universe_file(path) == entries


def test_load_universe_file_unsupported_suffix_raises(tmp_path: Path) -> None:
    path = tmp_path / "universe.txt"
    path.write_text("not a universe file")
    with pytest.raises(ValueError, match="unsupported universe file suffix"):
        load_universe_file(path)
