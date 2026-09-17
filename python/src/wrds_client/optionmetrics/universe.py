"""Batch a `resolve_universe_secids` result into WRDS queries, and run
`ingest_option_prices` once per batch.

The resolved DataFrame has one row per (input, secid, valid_start, valid_end)
-- several rows for the same input when its secid link changed mid-range.
Naively calling `ingest_option_prices` once per row would fetch correctly but
issue far more WRDS queries than necessary; naively grouping all secids into
one query for the whole date range would be simpler but wrong (it would fetch
rows for a secid outside its true valid window, misattributing data to an
identifier it doesn't belong to for that period). `plan_secid_batches` computes
the exact partition: same query range only where the *set* of valid secids is
identical, so most universes collapse into one or two long-lived batches, with
extra batches only around actual link-change dates.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from datalake import Artifact, DatalakeIndex
    from wrds_client.client import WRDSClient

from wrds_client.optionmetrics._shared import validate_columns
from wrds_client.optionmetrics.option_prices import KIND, OPTION_PRICE_COLUMNS, ingest_option_prices

_ONE_DAY = timedelta(days=1)


@dataclass(frozen=True)
class SecidBatch:
    """One `ingest_option_prices` call: a set of secids valid for the whole
    [start, end] sub-range."""

    start: date
    end: date
    secids: tuple[int, ...]


def plan_secid_batches(
    resolved: pd.DataFrame, start_date: date, end_date: date
) -> list[SecidBatch]:
    """Partition [start_date, end_date] into batches of (range, secid set).

    Rows with `secid` null (unresolved inputs) are ignored. A sub-range where
    nothing resolved is simply absent from the output, not an empty-secid
    batch.
    """
    if end_date < start_date:
        raise ValueError(f"end_date {end_date} is before start_date {start_date}")

    rows = [
        (row.secid, max(row.valid_start, start_date), min(row.valid_end, end_date))
        for row in resolved.itertuples()
        if pd.notna(row.secid) and row.valid_start <= end_date and row.valid_end >= start_date
    ]
    if not rows:
        return []

    breakpoints: set[date] = {start_date, end_date + _ONE_DAY}
    for _secid, valid_start, valid_end in rows:
        breakpoints.add(valid_start)
        breakpoints.add(valid_end + _ONE_DAY)
    sorted_breakpoints = sorted(b for b in breakpoints if start_date <= b <= end_date + _ONE_DAY)

    elementary: list[tuple[date, date, frozenset[int]]] = []
    for i in range(len(sorted_breakpoints) - 1):
        elem_start = sorted_breakpoints[i]
        elem_end = sorted_breakpoints[i + 1] - _ONE_DAY
        active = frozenset(
            int(secid)
            for secid, valid_start, valid_end in rows
            if valid_start <= elem_start and valid_end >= elem_end
        )
        if active:
            elementary.append((elem_start, elem_end, active))

    batches: list[SecidBatch] = []
    for elem_start, elem_end, active in elementary:
        secids = tuple(sorted(active))
        contiguous = batches and batches[-1].end + _ONE_DAY == elem_start
        if batches and batches[-1].secids == secids and contiguous:
            batches[-1] = SecidBatch(batches[-1].start, elem_end, batches[-1].secids)
        else:
            batches.append(SecidBatch(elem_start, elem_end, secids))

    return batches


def _already_ingested(
    index: DatalakeIndex, batch: SecidBatch, columns: Sequence[str] | None, library: str
) -> bool:
    """True if a complete artifact already covers exactly this batch."""
    target_secids = sorted(batch.secids)
    resolved_columns = validate_columns(
        columns, all_columns=OPTION_PRICE_COLUMNS, label="option_price"
    )
    target_columns = sorted(resolved_columns)
    for artifact in index.list(kind=KIND, include_partial=False):
        hp = artifact.meta.hyperparams
        if (
            sorted(hp.get("secid", [])) == target_secids
            and hp.get("start_date") == batch.start.isoformat()
            and hp.get("end_date") == batch.end.isoformat()
            and sorted(hp.get("columns", [])) == target_columns
            and hp.get("library") == library
        ):
            return True
    return False


def ingest_universe_option_prices(
    index: DatalakeIndex,
    client: WRDSClient,
    resolved: pd.DataFrame,
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
    skip_existing: bool = False,
) -> list[Artifact]:
    """Run `ingest_option_prices` once per batch from `plan_secid_batches`.

    Returns one artifact per batch (not one per universe): each artifact's
    hyperparams stay meaningful and independently re-runnable/inspectable.
    `skip_existing=True` skips batches already covered by a complete artifact
    with matching hyperparams (see `_already_ingested`).
    """
    batches = plan_secid_batches(resolved, start_date, end_date)
    artifacts: list[Artifact] = []
    for batch in batches:
        if skip_existing and _already_ingested(index, batch, columns, library):
            continue
        artifacts.append(
            ingest_option_prices(
                index,
                client,
                list(batch.secids),
                batch.start,
                batch.end,
                pipeline=pipeline,
                pipeline_version=pipeline_version,
                columns=columns,
                library=library,
                sources=sources,
                notes=notes,
                pipeline_repo=pipeline_repo,
            )
        )
    return artifacts
