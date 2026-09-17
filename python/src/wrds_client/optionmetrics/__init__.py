"""OptionMetrics IvyDB US (WRDS `optionm` library): option prices and
realized (historical) volatility.

Table/column names confirmed live against WRDS (`describe_table`), cross
-checked against IvyDB US Reference Manual v7.0 (python/doc/wrds/). Both
tables are year-partitioned (`opprcd{YYYY}`, `hvold{YYYY}`); the fetch/ingest
functions here span a date range by UNION-ing the relevant per-year tables.
"""
from __future__ import annotations

from wrds_client.optionmetrics.historical_volatility import (
    HISTORICAL_VOLATILITY_COLUMNS,
    fetch_historical_volatility,
    ingest_historical_volatility,
)
from wrds_client.optionmetrics.option_prices import (
    OPTION_PRICE_COLUMNS,
    fetch_option_prices,
    ingest_option_prices,
)
from wrds_client.optionmetrics.universe import (
    SecidBatch,
    ingest_universe_option_prices,
    plan_secid_batches,
)

__all__ = [
    "HISTORICAL_VOLATILITY_COLUMNS",
    "OPTION_PRICE_COLUMNS",
    "SecidBatch",
    "fetch_historical_volatility",
    "fetch_option_prices",
    "ingest_historical_volatility",
    "ingest_option_prices",
    "ingest_universe_option_prices",
    "plan_secid_batches",
]
