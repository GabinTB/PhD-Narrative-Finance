"""Tests for wrds_client.optionmetrics.universe: secid-batch planning and
universe-level ingestion.

plan_secid_batches is pure and tested table-driven; ingest_universe_option_prices
is tested against a stub client + a real DatalakeIndex on tmp_path.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from datalake import DatalakeIndex
from wrds_client.optionmetrics.option_prices import KIND
from wrds_client.optionmetrics.universe import (
    SecidBatch,
    _already_ingested,
    ingest_universe_option_prices,
    plan_secid_batches,
)

PIPELINE = "PhD-Narrative-Finance"
VERSION = "v0.1.0"


@pytest.fixture
def index(tmp_path: Path) -> DatalakeIndex:
    with DatalakeIndex(tmp_path / "datalake") as idx:
        yield idx


def _resolved(rows: list[tuple[int, date, date]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "secid": [r[0] for r in rows],
            "valid_start": [r[1] for r in rows],
            "valid_end": [r[2] for r in rows],
        }
    )


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
# plan_secid_batches
# ---------------------------------------------------------------------------


def test_single_secid_whole_range_is_one_batch() -> None:
    resolved = _resolved([(101, date(2020, 1, 1), date(2020, 12, 31))])
    batches = plan_secid_batches(resolved, date(2020, 1, 1), date(2020, 12, 31))
    assert batches == [SecidBatch(date(2020, 1, 1), date(2020, 12, 31), (101,))]


def test_mid_range_handoff_produces_two_contiguous_batches() -> None:
    resolved = _resolved(
        [
            (101, date(2020, 1, 1), date(2020, 6, 30)),
            (202, date(2020, 7, 1), date(2020, 12, 31)),
        ]
    )
    batches = plan_secid_batches(resolved, date(2020, 1, 1), date(2020, 12, 31))
    assert batches == [
        SecidBatch(date(2020, 1, 1), date(2020, 6, 30), (101,)),
        SecidBatch(date(2020, 7, 1), date(2020, 12, 31), (202,)),
    ]


def test_identical_windows_merge_into_one_batch_not_per_input() -> None:
    """Two different inputs whose secids are both valid for the whole range
    should collapse into ONE query, not one per input row."""
    resolved = _resolved(
        [
            (101, date(2020, 1, 1), date(2020, 12, 31)),
            (202, date(2020, 1, 1), date(2020, 12, 31)),
        ]
    )
    batches = plan_secid_batches(resolved, date(2020, 1, 1), date(2020, 12, 31))
    assert len(batches) == 1
    assert batches[0].secids == (101, 202)


def test_partial_overlap_splits_only_where_set_changes() -> None:
    resolved = _resolved(
        [
            (101, date(2020, 1, 1), date(2020, 12, 31)),
            (202, date(2020, 7, 1), date(2020, 12, 31)),
        ]
    )
    batches = plan_secid_batches(resolved, date(2020, 1, 1), date(2020, 12, 31))
    assert batches == [
        SecidBatch(date(2020, 1, 1), date(2020, 6, 30), (101,)),
        SecidBatch(date(2020, 7, 1), date(2020, 12, 31), (101, 202)),
    ]


def test_gap_is_absent_not_an_empty_batch() -> None:
    resolved = _resolved(
        [
            (101, date(2020, 1, 1), date(2020, 3, 31)),
            (101, date(2020, 9, 1), date(2020, 12, 31)),
        ]
    )
    batches = plan_secid_batches(resolved, date(2020, 1, 1), date(2020, 12, 31))
    assert len(batches) == 2
    assert all(b.secids for b in batches)
    assert batches[0].end == date(2020, 3, 31)
    assert batches[1].start == date(2020, 9, 1)


def test_unresolved_rows_ignored() -> None:
    resolved = pd.DataFrame(
        {
            "secid": [None, 101],
            "valid_start": [None, date(2020, 1, 1)],
            "valid_end": [None, date(2020, 12, 31)],
        }
    )
    batches = plan_secid_batches(resolved, date(2020, 1, 1), date(2020, 12, 31))
    assert batches == [SecidBatch(date(2020, 1, 1), date(2020, 12, 31), (101,))]


def test_no_resolved_rows_returns_empty_list() -> None:
    resolved = pd.DataFrame({"secid": [], "valid_start": [], "valid_end": []})
    assert plan_secid_batches(resolved, date(2020, 1, 1), date(2020, 12, 31)) == []


def test_end_before_start_raises() -> None:
    resolved = _resolved([(101, date(2020, 1, 1), date(2020, 12, 31))])
    with pytest.raises(ValueError, match="before start_date"):
        plan_secid_batches(resolved, date(2020, 6, 1), date(2020, 1, 1))


# ---------------------------------------------------------------------------
# ingest_universe_option_prices
# ---------------------------------------------------------------------------


def test_ingest_universe_option_prices_one_artifact_per_batch(index: DatalakeIndex) -> None:
    client = StubWRDSClient(_df())
    resolved = _resolved(
        [
            (101, date(2020, 1, 1), date(2020, 6, 30)),
            (202, date(2020, 7, 1), date(2020, 12, 31)),
        ]
    )
    artifacts = ingest_universe_option_prices(
        index, client, resolved, date(2020, 1, 1), date(2020, 12, 31),
        pipeline=PIPELINE, pipeline_version=VERSION,
    )
    assert len(artifacts) == 2
    assert artifacts[0].meta.hyperparams["secid"] == [101]
    assert artifacts[0].meta.hyperparams["start_date"] == "2020-01-01"
    assert artifacts[1].meta.hyperparams["secid"] == [202]


def test_ingest_universe_option_prices_threads_sources(index: DatalakeIndex) -> None:
    client = StubWRDSClient(_df())
    resolved = _resolved([(101, date(2020, 1, 1), date(2020, 12, 31))])
    artifacts = ingest_universe_option_prices(
        index, client, resolved, date(2020, 1, 1), date(2020, 12, 31),
        pipeline=PIPELINE, pipeline_version=VERSION,
        sources=["wrds_secid_resolution__some_artifact_id"],
    )
    assert artifacts[0].meta.sources == ["wrds_secid_resolution__some_artifact_id"]


# ---------------------------------------------------------------------------
# _already_ingested / skip_existing
# ---------------------------------------------------------------------------


def test_already_ingested_true_for_matching_complete_artifact(index: DatalakeIndex) -> None:
    client = StubWRDSClient(_df())
    resolved = _resolved([(101, date(2020, 1, 1), date(2020, 12, 31))])
    ingest_universe_option_prices(
        index, client, resolved, date(2020, 1, 1), date(2020, 12, 31),
        pipeline=PIPELINE, pipeline_version=VERSION,
    )
    batch = SecidBatch(date(2020, 1, 1), date(2020, 12, 31), (101,))
    assert _already_ingested(index, batch, None, "optionm") is True


def test_already_ingested_false_for_different_batch(index: DatalakeIndex) -> None:
    client = StubWRDSClient(_df())
    resolved = _resolved([(101, date(2020, 1, 1), date(2020, 12, 31))])
    ingest_universe_option_prices(
        index, client, resolved, date(2020, 1, 1), date(2020, 12, 31),
        pipeline=PIPELINE, pipeline_version=VERSION,
    )
    other_batch = SecidBatch(date(2021, 1, 1), date(2021, 12, 31), (101,))
    assert _already_ingested(index, other_batch, None, "optionm") is False


def test_skip_existing_avoids_duplicate_ingest(index: DatalakeIndex) -> None:
    client = StubWRDSClient(_df())
    resolved = _resolved([(101, date(2020, 1, 1), date(2020, 12, 31))])
    ingest_universe_option_prices(
        index, client, resolved, date(2020, 1, 1), date(2020, 12, 31),
        pipeline=PIPELINE, pipeline_version=VERSION,
    )
    assert len(client.raw_sql_calls) == 1

    artifacts = ingest_universe_option_prices(
        index, client, resolved, date(2020, 1, 1), date(2020, 12, 31),
        pipeline=PIPELINE, pipeline_version=VERSION, skip_existing=True,
    )
    assert artifacts == []
    assert len(client.raw_sql_calls) == 1  # no new query issued

    assert len(index.list(kind=KIND, include_partial=False)) == 1
