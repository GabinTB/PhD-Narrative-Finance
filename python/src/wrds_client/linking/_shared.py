"""Shared helpers for the identifier-linking hop queries.

Both `crsp_compustat.py` (gvkey<->permno via `crsp_a_ccm.ccmxpf_linktable`) and
`optionmetrics.py` (permno<->secid via `wrdsapps_link_crsp_optionm.opcrsphist`)
query a time-bounded link table with the same shape: a start column that is
never null and an end column that is `NULL` while the link is still active.
Factored out so each hop doesn't re-derive the overlap/clipping logic.
"""
from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd


def to_date(value: Any) -> date | None:
    """Normalise a value read back from `raw_sql(..., date_cols=[...])`.

    pandas parses date_cols to `Timestamp` (missing -> `NaT`), not
    `datetime.date`/`None` -- comparing a bare `Timestamp` against a plain
    `date` raises `TypeError`, so every row value from a date_cols column
    must go through this before being compared or stored.
    """
    if value is None or pd.isna(value):
        return None
    if isinstance(value, date) and not isinstance(value, pd.Timestamp):
        return value
    return pd.Timestamp(value).date()


def overlap_clause(start_col: str, end_col: str, start_param: str, end_param: str) -> str:
    """SQL predicate: does [start_col, end_col-or-open] overlap [%(start_param)s, %(end_param)s]?

    `end_col IS NULL` means the link is still active (open-ended).
    """
    return (
        f"{start_col} <= %({end_param})s "
        f"AND ({end_col} IS NULL OR {end_col} >= %({start_param})s)"
    )


def as_of_clause(start_col: str, end_col: str, param: str) -> str:
    """SQL predicate: is %(param)s within [start_col-or-open, end_col-or-open]?

    The point-in-time counterpart to `overlap_clause` (a range vs a range) --
    used by `wrds_client.capitaliq`/`wrds_client.lseg`, which resolve "as of
    this row's own snapshot_date" rather than over a query date range.

    Both `start_col` and `end_col` may be `NULL` meaning open-ended --
    confirmed live: `ciq_common.wrds_cik`/`wrds_gvkey` can have a fully
    NULL/NULL (always-valid) row for a company's primary identifier, not just
    a NULL end like the CRSP/CCM/OptionMetrics link tables this pattern was
    first built for.
    """
    return (
        f"({start_col} IS NULL OR {start_col} <= %({param})s) "
        f"AND ({end_col} IS NULL OR {end_col} >= %({param})s)"
    )


def clip_window(
    row_start: date, row_end: date | None, query_start: date, query_end: date
) -> tuple[date, date]:
    """Intersect a link's validity window with the query's date range.

    `row_end is None` means the link is still open-ended; it clips to
    `query_end` rather than extending past it.
    """
    clipped_start = max(row_start, query_start)
    clipped_end = min(row_end or query_end, query_end)
    return clipped_start, clipped_end


def match_as_of_windows(
    candidates: pd.DataFrame,
    requests: pd.DataFrame,
    *,
    key_col: str,
    start_col: str,
    end_col: str,
    snapshot_col: str = "snapshot_date",
    prefer_col: str | None = None,
) -> pd.DataFrame:
    """Vectorised point-in-time match, local to the process (no per-row SQL).

    Used by `wrds_client.capitaliq`/`wrds_client.lseg`, which now fetch every
    validity window for the identifiers actually present in a universe (one
    query per identifier column, no server-side date filter -- see
    `as_of_clause`'s docstring for why that clause is superseded here) and
    match each row against the right window locally instead.

    For each row in `requests` (which must have `key_col` and `snapshot_col`),
    finds the `candidates` row(s) sharing `key_col` whose
    `[start_col-or-open, end_col-or-open]` window covers `requests[snapshot_col]`,
    preferring `prefer_col == 1` when more than one does, else the first match
    in `candidates`' original order. A request with no covering window is
    dropped, not raised or filled with NaN.

    Returns one row per matched request, indexed by `requests`'s own index
    (so callers can `.loc[]` back into their original rows), with
    `candidates`'s columns.
    """
    if requests.empty or candidates.empty:
        return candidates.iloc[0:0]

    requests = requests.copy()
    requests["_request_idx"] = requests.index
    requests = requests.reset_index(drop=True)

    candidates = candidates.reset_index(drop=True)
    candidates = candidates.assign(_candidate_order=candidates.index)

    merged = requests.merge(candidates, on=key_col, how="inner")
    if merged.empty:
        return candidates.iloc[0:0]

    start = merged[start_col]
    end = merged[end_col]
    as_of = merged[snapshot_col]
    covers = (start.isna() | (start <= as_of)) & (end.isna() | (end >= as_of))
    merged = merged[covers]
    if merged.empty:
        return candidates.iloc[0:0]

    sort_cols = ["_request_idx"]
    ascending = [True]
    if prefer_col is not None and prefer_col in merged.columns:
        merged = merged.assign(_pref=(merged[prefer_col] == 1).astype(int))
        sort_cols.append("_pref")
        ascending.append(False)
    sort_cols.append("_candidate_order")
    ascending.append(True)

    merged = merged.sort_values(sort_cols, ascending=ascending)
    result = merged.drop_duplicates("_request_idx", keep="first")
    return result.set_index("_request_idx")


def normalise_isin(isin: str) -> str:
    return isin.strip().upper()


def normalise_cik(cik: str) -> str:
    """Zero-pad to WRDS's stored 10-digit CIK format (e.g. "320193" -> "0000320193").

    Matches `edgar_tools.config._normalise_cik`'s convention -- confirmed live
    against `comp.company.cik`, which stores CIKs this way.
    """
    return str(cik).strip().zfill(10)
