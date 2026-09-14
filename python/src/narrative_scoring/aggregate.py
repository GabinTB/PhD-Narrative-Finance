"""Roll primitive-level daily scores up the taxonomy hierarchy.

primitive -> narrative -> dimension -> reservoir, using the canonical mapping
from taxonomy.py.  INTENSITY is aggregated with ``mode`` (mean/median),
SUPPORT is summed, PEAK takes the max.

Aggregation runs over ALL child primitives, including zero-intensity ones: a
narrative whose primitives were mostly silent should show low intensity, and
dropping the zeros would bias it upward.  This assumes ``daily`` already
carries one row per (date, primitive) for every primitive in the taxonomy
(scoring.py's contract: zero support/intensity, never a missing row).
"""
from __future__ import annotations

from enum import Enum

import polars as pl

from narrative_scoring.schema import AGGREGATED_SCORES_SCHEMA

_LEVEL_COLUMN: dict[str, str] = {
    "narrative": "narrative",
    "dimension": "dimension",
    "reservoir": "reservoir",
}


class AggMode(str, Enum):
    MEAN = "mean"
    MEDIAN = "median"


def aggregate_level(
    daily: pl.DataFrame,     # DATE, PRIMITIVE, INTENSITY, SUPPORT, PEAK
    taxonomy: pl.DataFrame,  # canonical hierarchy frame (taxonomy.py primitives())
    to_level: str,           # "narrative" | "dimension" | "reservoir"
    mode: AggMode = AggMode.MEAN,
) -> pl.DataFrame:
    if to_level not in _LEVEL_COLUMN:
        raise ValueError(f"unknown level {to_level!r}, expected one of {list(_LEVEL_COLUMN)}")

    node_col = _LEVEL_COLUMN[to_level]
    if node_col not in taxonomy.columns:
        raise ValueError(f"taxonomy frame missing column {node_col!r} for level {to_level!r}")
    if "primitive" not in taxonomy.columns:
        raise ValueError("taxonomy frame missing 'primitive' column")

    joined = daily.join(
        taxonomy.select(["primitive", node_col]),
        left_on="PRIMITIVE", right_on="primitive", how="inner",
    )

    intensity_expr = (
        pl.col("INTENSITY").mean() if mode is AggMode.MEAN else pl.col("INTENSITY").median()
    )

    out = (
        joined.group_by(["DATE", node_col])
        .agg(
            [
                intensity_expr.alias("INTENSITY"),
                pl.col("SUPPORT").sum().alias("SUPPORT"),
                pl.col("PEAK").max().alias("PEAK"),
            ]
        )
        .rename({node_col: "NODE"})
        .with_columns(pl.lit(to_level).alias("LEVEL"))
        .select(["DATE", "LEVEL", "NODE", "INTENSITY", "SUPPORT", "PEAK"])
        .sort(["DATE", "NODE"])
    )
    return out.cast(dict(AGGREGATED_SCORES_SCHEMA))
