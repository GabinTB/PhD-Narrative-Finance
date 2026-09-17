"""Tests for wrds_client.optionmetrics.option_prices.

No real WRDS server involved: build_query is pure, and fetch/ingest are
tested against a stub client / a real DatalakeIndex on tmp_path.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from datalake import DatalakeIndex
from wrds_client.optionmetrics.option_prices import (
    DATE_COLUMNS,
    OPTION_PRICE_COLUMNS,
    build_query,
    fetch_option_prices,
    ingest_option_prices,
)

PIPELINE = "PhD-Narrative-Finance"
VERSION = "v0.1.0"


@pytest.fixture
def index(tmp_path: Path) -> DatalakeIndex:
    with DatalakeIndex(tmp_path / "datalake") as idx:
        yield idx


class StubWRDSClient:
    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df
        self.raw_sql_calls: list[dict[str, Any]] = []

    def raw_sql(self, sql: str, **kwargs: Any) -> pd.DataFrame:
        self.raw_sql_calls.append({"sql": sql, **kwargs})
        return self._df


def _df() -> pd.DataFrame:
    return pd.DataFrame({"secid": [101], "date": [date(2023, 1, 3)], "close": [10.5]})


# ---------------------------------------------------------------------------
# build_query
# ---------------------------------------------------------------------------


def test_build_query_single_year_single_secid() -> None:
    sql, params = build_query(101, date(2023, 1, 1), date(2023, 6, 30))
    assert sql.count("UNION ALL") == 0
    assert "optionm.opprcd2023" in sql
    assert params == {
        "secids": (101,),
        "start_date": date(2023, 1, 1),
        "end_date": date(2023, 6, 30),
    }


def test_build_query_spans_years_with_union_all() -> None:
    sql, _ = build_query(101, date(2022, 11, 1), date(2023, 2, 1))
    assert "optionm.opprcd2022" in sql
    assert "optionm.opprcd2023" in sql
    assert sql.count("UNION ALL") == 1


def test_build_query_multiple_secids() -> None:
    _, params = build_query([101, 102, 103], date(2023, 1, 1), date(2023, 1, 31))
    assert params["secids"] == (101, 102, 103)


def test_build_query_default_columns_is_all() -> None:
    sql, _ = build_query(101, date(2023, 1, 1), date(2023, 1, 31))
    for col in OPTION_PRICE_COLUMNS:
        assert col in sql


def test_build_query_custom_columns_subset() -> None:
    sql, _ = build_query(
        101, date(2023, 1, 1), date(2023, 1, 31), columns=["secid", "date", "impl_volatility"]
    )
    assert "SELECT secid, date, impl_volatility FROM" in sql
    assert "delta" not in sql


def test_build_query_unknown_column_raises() -> None:
    with pytest.raises(ValueError, match="unknown option_price column"):
        build_query(101, date(2023, 1, 1), date(2023, 1, 31), columns=["not_a_column"])


def test_build_query_empty_secid_sequence_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        build_query([], date(2023, 1, 1), date(2023, 1, 31))


def test_build_query_end_before_start_raises() -> None:
    with pytest.raises(ValueError, match="before start_date"):
        build_query(101, date(2023, 6, 1), date(2023, 1, 1))


def test_build_query_custom_library() -> None:
    sql, _ = build_query(101, date(2023, 1, 1), date(2023, 1, 31), library="optionm_all")
    assert "optionm_all.opprcd2023" in sql


# ---------------------------------------------------------------------------
# fetch_option_prices
# ---------------------------------------------------------------------------


def test_fetch_passes_only_selected_date_cols() -> None:
    client = StubWRDSClient(_df())
    fetch_option_prices(
        client, 101, date(2023, 1, 1), date(2023, 1, 31), columns=["secid", "date", "cp_flag"]
    )
    call = client.raw_sql_calls[0]
    assert call["date_cols"] == ["date"]


def test_fetch_default_columns_uses_all_date_cols() -> None:
    client = StubWRDSClient(_df())
    fetch_option_prices(client, 101, date(2023, 1, 1), date(2023, 1, 31))
    call = client.raw_sql_calls[0]
    assert set(call["date_cols"]) == set(DATE_COLUMNS)


def test_fetch_returns_client_result() -> None:
    client = StubWRDSClient(_df())
    df = fetch_option_prices(client, 101, date(2023, 1, 1), date(2023, 1, 31))
    pd.testing.assert_frame_equal(df, _df())


# ---------------------------------------------------------------------------
# ingest_option_prices
# ---------------------------------------------------------------------------


def test_ingest_registers_artifact_with_hyperparams(index: DatalakeIndex) -> None:
    client = StubWRDSClient(_df())
    artifact = ingest_option_prices(
        index,
        client,
        [101, 102],
        date(2023, 1, 1),
        date(2023, 1, 31),
        pipeline=PIPELINE,
        pipeline_version=VERSION,
        columns=["secid", "date", "impl_volatility"],
    )
    assert not artifact.partial
    assert artifact.meta.hyperparams["secid"] == [101, 102]
    assert artifact.meta.hyperparams["start_date"] == "2023-01-01"
    assert artifact.meta.hyperparams["end_date"] == "2023-01-31"
    assert artifact.meta.hyperparams["columns"] == ["secid", "date", "impl_volatility"]
    assert artifact.meta.hyperparams["library"] == "optionm"

    files = artifact.files("*.parquet")
    written = pd.read_parquet(files[0])
    pd.testing.assert_frame_equal(written, _df())
