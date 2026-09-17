"""OptionMetrics realized (historical) volatility (WRDS `optionm.hvold{YYYY}`).

Schema confirmed via `describe_table("optionm", "hvold2023")` against a live
WRDS connection, matching the IvyDB US Reference Manual v7.0
"Historical_Volatility File" section exactly (python/doc/wrds/
IvyDB_US_v7.0_Reference_Manual.pdf, p.30): `secid`, `date`, `days`,
`volatility`. `days` is the lookback window (10, 14, 30, 60, 91, 122, 152,
182, 273, 365, 547, 730, or 1825 calendar days) -- each (secid, date) has one
row per window, all returned by default.

Note `optionm.historical_volatility` (no year suffix) is a *different*,
differently-keyed table (`securityid`/`currency` instead of `secid`) and is
not used here.

Data is year-partitioned like option prices; `secid` and a date range are
required for the same reason (unfiltered pulls are not realistic: ~1.9*10^7
rows/year).
"""
from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from datalake import Artifact, DatalakeIndex
    from wrds_client.client import WRDSClient

from wrds_client.ingest import fetch_query_artifact
from wrds_client.optionmetrics._shared import (
    build_secid_date_range_query,
    normalise_secid,
    validate_columns,
)

KIND = "optionm_historical_volatility"

#: All 4 columns of optionm.hvold{YYYY}, in table order.
HISTORICAL_VOLATILITY_COLUMNS: tuple[str, ...] = ("secid", "date", "days", "volatility")

#: Date-typed columns, for `date_cols` when reading results into pandas.
DATE_COLUMNS: tuple[str, ...] = ("date",)

_LABEL = "historical_volatility"
_TABLE_PREFIX = "hvold"


def build_query(
    secid: int | Sequence[int],
    start_date: date,
    end_date: date,
    *,
    columns: Sequence[str] | None = None,
    library: str = "optionm",
) -> tuple[str, dict]:
    """Build the (SQL, params) pair to fetch realized volatility for `secid`.

    See `wrds_client.optionmetrics._shared.build_secid_date_range_query`.
    """
    return build_secid_date_range_query(
        secid,
        start_date,
        end_date,
        columns=columns,
        all_columns=HISTORICAL_VOLATILITY_COLUMNS,
        label=_LABEL,
        library=library,
        table_prefix=_TABLE_PREFIX,
    )


def fetch_historical_volatility(
    client: WRDSClient,
    secid: int | Sequence[int],
    start_date: date,
    end_date: date,
    *,
    columns: Sequence[str] | None = None,
    library: str = "optionm",
) -> pd.DataFrame:
    """Fetch realized volatility for `secid` over `start_date`..`end_date`.

    `columns` selects which of `HISTORICAL_VOLATILITY_COLUMNS` to return;
    `None` (the default) returns all of them.
    """
    sql, params = build_query(secid, start_date, end_date, columns=columns, library=library)
    resolved_columns = validate_columns(
        columns, all_columns=HISTORICAL_VOLATILITY_COLUMNS, label=_LABEL
    )
    date_cols = [c for c in resolved_columns if c in DATE_COLUMNS]
    return client.raw_sql(sql, date_cols=date_cols or None, params=params)


def ingest_historical_volatility(
    index: DatalakeIndex,
    client: WRDSClient,
    secid: int | Sequence[int],
    start_date: date,
    end_date: date,
    *,
    pipeline: str,
    pipeline_version: str,
    columns: Sequence[str] | None = None,
    library: str = "optionm",
    sources: Sequence[Artifact | str] | None = None,
    notes: str = "",
    pipeline_repo: str | None = None,
) -> Artifact:
    """Fetch realized volatility for `secid` and register the result as a datalake artifact."""
    sql, params = build_query(secid, start_date, end_date, columns=columns, library=library)
    resolved_columns = validate_columns(
        columns, all_columns=HISTORICAL_VOLATILITY_COLUMNS, label=_LABEL
    )
    date_cols = [c for c in resolved_columns if c in DATE_COLUMNS]
    hyperparams = {
        "secid": list(normalise_secid(secid)),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "columns": list(resolved_columns),
        "library": library,
    }
    return fetch_query_artifact(
        index,
        client,
        sql,
        kind=KIND,
        pipeline=pipeline,
        pipeline_version=pipeline_version,
        hyperparams=hyperparams,
        sources=sources,
        notes=notes,
        pipeline_repo=pipeline_repo,
        date_cols=date_cols or None,
        params=params,
    )
