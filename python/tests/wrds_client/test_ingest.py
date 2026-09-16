"""Tests for wrds_client.ingest: fetch-to-datalake-artifact helpers.

Uses a real DatalakeIndex against tmp_path and a stub WRDSClient (no real
WRDS server or .env credentials involved).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from datalake import DatalakeIndex
from wrds_client.ingest import fetch_query_artifact, fetch_table_artifact

PIPELINE = "PhD-Narrative-Finance"
VERSION = "v0.1.0"


@pytest.fixture
def index(tmp_path: Path) -> DatalakeIndex:
    with DatalakeIndex(tmp_path / "datalake") as idx:
        yield idx


class StubWRDSClient:
    """Stands in for WRDSClient: records calls, returns a canned DataFrame."""

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df
        self.raw_sql_calls: list[dict[str, Any]] = []
        self.get_table_calls: list[dict[str, Any]] = []

    def raw_sql(self, sql: str, **kwargs: Any) -> pd.DataFrame:
        self.raw_sql_calls.append({"sql": sql, **kwargs})
        return self._df

    def get_table(self, library: str, table: str, **kwargs: Any) -> pd.DataFrame:
        self.get_table_calls.append({"library": library, "table": table, **kwargs})
        return self._df


def _df() -> pd.DataFrame:
    return pd.DataFrame({"secid": [101, 102], "close": [10.5, 11.25]})


def test_fetch_query_artifact_registers_and_writes_parquet(index: DatalakeIndex) -> None:
    client = StubWRDSClient(_df())
    sql = "select secid, close from optionm.secprd where date = '2024-01-02'"

    artifact = fetch_query_artifact(
        index,
        client,
        sql,
        kind="optionm_secprd",
        pipeline=PIPELINE,
        pipeline_version=VERSION,
    )

    assert not artifact.partial
    files = artifact.files("*.parquet")
    assert len(files) == 1
    written = pd.read_parquet(files[0])
    pd.testing.assert_frame_equal(written, _df())
    assert artifact.meta.hyperparams["sql"] == sql
    assert client.raw_sql_calls == [{"sql": sql, "date_cols": None}]


def test_fetch_query_artifact_merges_extra_hyperparams(index: DatalakeIndex) -> None:
    client = StubWRDSClient(_df())
    artifact = fetch_query_artifact(
        index,
        client,
        "select 1",
        kind="optionm_secprd",
        pipeline=PIPELINE,
        pipeline_version=VERSION,
        hyperparams={"as_of": "2024-01-02"},
    )
    assert artifact.meta.hyperparams["as_of"] == "2024-01-02"
    assert artifact.meta.hyperparams["sql"] == "select 1"


def test_fetch_table_artifact_registers_and_writes_parquet(index: DatalakeIndex) -> None:
    client = StubWRDSClient(_df())

    artifact = fetch_table_artifact(
        index,
        client,
        "optionm",
        "securd",
        kind="optionm_securd",
        pipeline=PIPELINE,
        pipeline_version=VERSION,
        columns=["secid", "close"],
        obs=100,
    )

    assert not artifact.partial
    files = artifact.files("*.parquet")
    written = pd.read_parquet(files[0])
    pd.testing.assert_frame_equal(written, _df())
    assert artifact.meta.hyperparams["library"] == "optionm"
    assert artifact.meta.hyperparams["table"] == "securd"
    assert artifact.meta.hyperparams["columns"] == ["secid", "close"]
    assert artifact.meta.hyperparams["obs"] == 100
    assert client.get_table_calls == [
        {
            "library": "optionm",
            "table": "securd",
            "columns": ["secid", "close"],
            "obs": 100,
            "offset": 0,
            "date_cols": None,
        }
    ]


def test_fetch_query_artifact_left_partial_on_query_failure(index: DatalakeIndex) -> None:
    class FailingClient:
        def raw_sql(self, sql: str, **kwargs: Any) -> pd.DataFrame:
            raise RuntimeError("connection reset")

    with pytest.raises(RuntimeError, match="connection reset"):
        fetch_query_artifact(
            index,
            FailingClient(),
            "select 1",
            kind="optionm_secprd",
            pipeline=PIPELINE,
            pipeline_version=VERSION,
        )

    [artifact] = index.list("optionm_secprd", include_partial=True)
    assert artifact.partial
