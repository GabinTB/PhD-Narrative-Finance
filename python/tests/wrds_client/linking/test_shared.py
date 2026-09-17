"""Tests for wrds_client.linking._shared, especially `to_date`.

`to_date` exists because `raw_sql(..., date_cols=[...])` returns pandas
`Timestamp`/`NaT`, not plain `datetime.date`/`None` -- comparing a bare
`Timestamp` against a `date` raises `TypeError` (caught via a live WRDS run,
not by the original stub-based tests, which used plain `date` objects and
so didn't exercise this).
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from wrds_client.linking._shared import clip_window, overlap_clause, to_date


def test_to_date_converts_timestamp() -> None:
    assert to_date(pd.Timestamp("2020-06-15")) == date(2020, 6, 15)


def test_to_date_converts_nat_to_none() -> None:
    assert to_date(pd.NaT) is None


def test_to_date_passes_through_plain_date() -> None:
    assert to_date(date(2020, 6, 15)) == date(2020, 6, 15)


def test_to_date_passes_through_none() -> None:
    assert to_date(None) is None


def test_clip_window_accepts_timestamp_inputs_via_to_date() -> None:
    """Regression: clip_window must work on to_date()'s output when the
    underlying values came from a real raw_sql(date_cols=...) call."""
    row_start = to_date(pd.Timestamp("2020-01-01"))
    row_end = to_date(pd.NaT)
    clipped_start, clipped_end = clip_window(
        row_start, row_end, date(2020, 3, 1), date(2020, 6, 30)
    )
    assert clipped_start == date(2020, 3, 1)
    assert clipped_end == date(2020, 6, 30)


def test_overlap_clause_shape() -> None:
    clause = overlap_clause("linkdt", "linkenddt", "start_date", "end_date")
    assert "linkdt <= %(end_date)s" in clause
    assert "linkenddt IS NULL OR linkenddt >= %(start_date)s" in clause


@pytest.mark.parametrize(
    ("row_start", "row_end", "query_start", "query_end", "expected"),
    [
        (
            date(2020, 1, 1),
            date(2020, 12, 31),
            date(2020, 3, 1),
            date(2020, 6, 1),
            (date(2020, 3, 1), date(2020, 6, 1)),
        ),
        (
            date(2019, 1, 1),
            None,
            date(2020, 1, 1),
            date(2020, 12, 31),
            (date(2020, 1, 1), date(2020, 12, 31)),
        ),
    ],
)
def test_clip_window_cases(row_start, row_end, query_start, query_end, expected) -> None:
    assert clip_window(row_start, row_end, query_start, query_end) == expected
