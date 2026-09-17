"""Shared query-building helpers for year-partitioned OptionMetrics tables.

Every optionm table used here (opprcd{YYYY}, hvold{YYYY}, ...) follows the
same shape: one physical table per year, keyed by secid, queried over a date
range via a UNION ALL across the years spanned. Factored out of the
per-table modules (option_prices.py, historical_volatility.py) so each
doesn't re-derive it.
"""
from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Any


def validate_columns(
    columns: Sequence[str] | None, *, all_columns: tuple[str, ...], label: str
) -> tuple[str, ...]:
    """Resolve a caller's `columns` against `all_columns`, defaulting to all of them."""
    if columns is None:
        return all_columns
    resolved = tuple(columns)
    if not resolved:
        raise ValueError("columns must be non-empty if provided")
    unknown = [c for c in resolved if c not in all_columns]
    if unknown:
        raise ValueError(f"unknown {label} column(s): {unknown}; valid columns are {all_columns}")
    return resolved


def normalise_secid(secid: int | Sequence[int]) -> tuple[int, ...]:
    secids = (secid,) if isinstance(secid, int) else tuple(secid)
    if not secids:
        raise ValueError("secid must be a non-empty int or sequence of int")
    return secids


def build_secid_date_range_query(
    secid: int | Sequence[int],
    start_date: date,
    end_date: date,
    *,
    columns: Sequence[str] | None,
    all_columns: tuple[str, ...],
    label: str,
    library: str,
    table_prefix: str,
) -> tuple[str, dict[str, Any]]:
    """Build (SQL, params) selecting `columns` for `secid` over a date range.

    UNION ALLs one `SELECT ... FROM {library}.{table_prefix}{year}` per year
    spanned by `start_date`..`end_date` (inclusive). `secid`(s) and the date
    range are bound as query parameters (`%(secids)s`, `%(start_date)s`,
    `%(end_date)s`), never interpolated into the SQL string; only `columns`
    (checked against `all_columns`), `library`, and `table_prefix` (developer
    -supplied, not user input) are interpolated, since SQL parameter binding
    cannot target identifiers.
    """
    if end_date < start_date:
        raise ValueError(f"end_date {end_date} is before start_date {start_date}")
    secids = normalise_secid(secid)
    cols = validate_columns(columns, all_columns=all_columns, label=label)

    col_sql = ", ".join(cols)
    year_queries = [
        f"SELECT {col_sql} FROM {library}.{table_prefix}{year} "
        "WHERE secid IN %(secids)s AND date BETWEEN %(start_date)s AND %(end_date)s"
        for year in range(start_date.year, end_date.year + 1)
    ]
    sql = "\nUNION ALL\n".join(year_queries)
    params: dict[str, Any] = {"secids": secids, "start_date": start_date, "end_date": end_date}
    return sql, params
