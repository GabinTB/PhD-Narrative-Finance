"""OptionMetrics option prices (WRDS `optionm.opprcd{YYYY}`).

Schema confirmed via `describe_table("optionm", "opprcd2023")` against a live
WRDS connection (26 columns), matching the IvyDB US Reference Manual v7.0
"Option_Price File" section (python/doc/wrds/IvyDB_US_v7.0_Reference_Manual.pdf,
p.18) plus two columns (`root`, `suffix`) present in the live table but not
individually documented there.

Data is year-partitioned: one physical table per year. `secid` and a date
range are required (rather than optional filters) because a single year's
table holds on the order of 3-4*10^8 rows -- an unfiltered pull is not a
realistic query.
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

KIND = "optionm_option_prices"

#: All 26 columns of optionm.opprcd{YYYY}, in table order.
OPTION_PRICE_COLUMNS: tuple[str, ...] = (
    "secid",
    "date",
    "symbol",
    "symbol_flag",
    "exdate",
    "last_date",
    "cp_flag",
    "strike_price",
    "best_bid",
    "best_offer",
    "volume",
    "open_interest",
    "impl_volatility",
    "delta",
    "gamma",
    "vega",
    "theta",
    "optionid",
    "cfadj",
    "am_settlement",
    "contract_size",
    "ss_flag",
    "forward_price",
    "expiry_indicator",
    "root",
    "suffix",
)

#: Date-typed columns, for `date_cols` when reading results into pandas.
DATE_COLUMNS: tuple[str, ...] = ("date", "exdate", "last_date")

_LABEL = "option_price"
_TABLE_PREFIX = "opprcd"


def build_query(
    secid: int | Sequence[int],
    start_date: date,
    end_date: date,
    *,
    columns: Sequence[str] | None = None,
    library: str = "optionm",
) -> tuple[str, dict]:
    """Build the (SQL, params) pair to fetch option prices for `secid`.

    See `wrds_client.optionmetrics._shared.build_secid_date_range_query`.
    """
    return build_secid_date_range_query(
        secid,
        start_date,
        end_date,
        columns=columns,
        all_columns=OPTION_PRICE_COLUMNS,
        label=_LABEL,
        library=library,
        table_prefix=_TABLE_PREFIX,
    )


def fetch_option_prices(
    client: WRDSClient,
    secid: int | Sequence[int],
    start_date: date,
    end_date: date,
    *,
    columns: Sequence[str] | None = None,
    library: str = "optionm",
) -> pd.DataFrame:
    """Fetch option prices for `secid` over `start_date`..`end_date`.

    `columns` selects which of `OPTION_PRICE_COLUMNS` to return; `None`
    (the default) returns all of them.
    """
    sql, params = build_query(secid, start_date, end_date, columns=columns, library=library)
    resolved_columns = validate_columns(columns, all_columns=OPTION_PRICE_COLUMNS, label=_LABEL)
    date_cols = [c for c in resolved_columns if c in DATE_COLUMNS]
    return client.raw_sql(sql, date_cols=date_cols or None, params=params)


def ingest_option_prices(
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
    """Fetch option prices for `secid` and register the result as a datalake artifact."""
    sql, params = build_query(secid, start_date, end_date, columns=columns, library=library)
    resolved_columns = validate_columns(columns, all_columns=OPTION_PRICE_COLUMNS, label=_LABEL)
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
