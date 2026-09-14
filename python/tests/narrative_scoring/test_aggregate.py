"""Tests for narrative_scoring.aggregate."""
from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from narrative_scoring.aggregate import AggMode, aggregate_level
from narrative_scoring.schema import PRIMITIVE_SCORES_SCHEMA


def _taxonomy_frame() -> pl.DataFrame:
    # p0, p1 -> narrative n0 -> dimension d0 -> reservoir r0
    # p2      -> narrative n1 -> dimension d0 -> reservoir r0
    return pl.DataFrame(
        {
            "reservoir": ["r0", "r0", "r0"],
            "dimension": ["d0", "d0", "d0"],
            "narrative": ["n0", "n0", "n1"],
            "primitive": ["p0", "p1", "p2"],
            "description": ["d0", "d1", "d2"],
        }
    )


def _daily_frame(d: date) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "DATE": [d, d, d],
            "PRIMITIVE": ["p0", "p1", "p2"],
            "INTENSITY": [0.8, 0.4, 0.0],
            "SUPPORT": [5, 3, 0],
            "PEAK": [0.9, 0.5, 0.0],
        },
        schema=PRIMITIVE_SCORES_SCHEMA,
    )


class TestAggregateLevel:
    def test_narrative_rollup_includes_zero_intensity_children(self):
        d = date(2008, 1, 1)
        daily = _daily_frame(d)
        tax = _taxonomy_frame()

        out = aggregate_level(daily, tax, "narrative", mode=AggMode.MEAN)
        rows = {r["NODE"]: r for r in out.iter_rows(named=True)}

        # n0 = mean(p0=0.8, p1=0.4) = 0.6
        assert rows["n0"]["INTENSITY"] == pytest.approx(0.6)
        assert rows["n0"]["SUPPORT"] == 8
        assert rows["n0"]["PEAK"] == pytest.approx(0.9)

        # n1 = mean(p2=0.0) -- the zero-intensity primitive is NOT dropped
        assert rows["n1"]["INTENSITY"] == pytest.approx(0.0)
        assert rows["n1"]["SUPPORT"] == 0
        assert rows["n1"]["PEAK"] == pytest.approx(0.0)

    def test_dimension_rollup_averages_across_narratives(self):
        d = date(2008, 1, 1)
        daily = _daily_frame(d)
        tax = _taxonomy_frame()

        out = aggregate_level(daily, tax, "dimension", mode=AggMode.MEAN)
        row = out.filter(pl.col("NODE") == "d0").row(0, named=True)
        # d0 = mean(p0=0.8, p1=0.4, p2=0.0) -- zero-intensity primitive included
        assert row["INTENSITY"] == pytest.approx((0.8 + 0.4 + 0.0) / 3)
        assert row["SUPPORT"] == 8

    def test_reservoir_rollup(self):
        d = date(2008, 1, 1)
        daily = _daily_frame(d)
        tax = _taxonomy_frame()

        out = aggregate_level(daily, tax, "reservoir", mode=AggMode.MEAN)
        assert out.height == 1
        row = out.row(0, named=True)
        assert row["NODE"] == "r0"
        assert row["INTENSITY"] == pytest.approx((0.8 + 0.4 + 0.0) / 3)

    def test_median_mode(self):
        d = date(2008, 1, 1)
        daily = pl.DataFrame(
            {
                "DATE": [d, d, d],
                "PRIMITIVE": ["p0", "p1", "p2"],
                "INTENSITY": [0.1, 0.5, 0.9],
                "SUPPORT": [1, 1, 1],
                "PEAK": [0.2, 0.6, 1.0],
            },
            schema=PRIMITIVE_SCORES_SCHEMA,
        )
        tax = _taxonomy_frame()
        out = aggregate_level(daily, tax, "dimension", mode=AggMode.MEDIAN)
        row = out.row(0, named=True)
        assert row["INTENSITY"] == pytest.approx(0.5)

    def test_unknown_level_raises(self):
        d = date(2008, 1, 1)
        daily = _daily_frame(d)
        tax = _taxonomy_frame()
        with pytest.raises(ValueError, match="unknown level"):
            aggregate_level(daily, tax, "galaxy")
