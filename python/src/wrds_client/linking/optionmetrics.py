"""permno -> secid resolution (WRDS `wrdsapps_link_crsp_optionm.opcrsphist`), plus
the end-to-end universe resolution orchestrator: (isin, cik) -> ... -> secid.

`resolve_universe_secids` is the single public entry point. It never silently
picks a winner among ambiguous matches or drops an unresolved input -- every
input in `universe` produces at least one output row, either a resolved
(gvkey, permno, secid, valid_start, valid_end) with `flags` describing any
ambiguity encountered along the way, or a single `flags=("unresolved",)` row
if it never resolved at all. Deciding what's fatal (e.g. refusing to proceed
on an `"ambiguous_gvkey"` row) is left to the caller -- this module always
returns everything found.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd

if TYPE_CHECKING:
    from datalake import Artifact, DatalakeIndex
    from wrds_client.client import WRDSClient

from universe.schema import UniverseEntry
from wrds_client.linking._shared import clip_window, overlap_clause, to_date
from wrds_client.linking.crsp_compustat import PermnoWindow, fetch_permno_windows, resolve_gvkeys

KIND = "wrds_secid_resolution"


@dataclass(frozen=True)
class SecidWindow:
    """One permno<->secid link, clipped to a query's date range."""

    permno: int
    secid: int
    valid_start: date
    valid_end: date
    score: float | None


@dataclass(frozen=True)
class SecidResolution:
    """One resolved (input, gvkey, permno, secid) row over a validity sub-range.

    A single input entry may produce several rows (one per non-overlapping
    validity window, or one per ambiguous branch), or exactly one row with
    `flags=("unresolved",)` and every other field `None` if it never
    resolved.
    """

    input_index: int
    isin: str | None
    cik: str | None
    ticker: str | None
    name: str | None
    gvkey: str | None
    permno: int | None
    secid: int | None
    valid_start: date | None
    valid_end: date | None
    resolution_method: Literal["isin", "cik"] | None
    score: float | None
    flags: tuple[str, ...]


def build_opcrsphist_query(
    permnos: Sequence[int], start_date: date, end_date: date
) -> tuple[str, dict[str, Any]]:
    if not permnos:
        raise ValueError("permnos must be non-empty")
    if end_date < start_date:
        raise ValueError(f"end_date {end_date} is before start_date {start_date}")
    sql = (
        "SELECT secid, permno, sdate, edate, score "
        "FROM wrdsapps_link_crsp_optionm.opcrsphist WHERE permno IN %(permnos)s AND "
        + overlap_clause("sdate", "edate", "start_date", "end_date")
    )
    params = {"permnos": tuple(permnos), "start_date": start_date, "end_date": end_date}
    return sql, params


def fetch_secid_windows(
    client: WRDSClient, permnos: Sequence[int], start_date: date, end_date: date
) -> list[SecidWindow]:
    """Fetch permno<->secid links overlapping [start_date, end_date].

    Returns every overlapping row (no score-based filtering): `score`'s sort
    convention for this table hasn't been confirmed, and this design never
    relies on it to silently break ties -- ties are flagged instead.
    """
    unique_permnos = sorted(set(permnos))
    if not unique_permnos:
        return []

    sql, params = build_opcrsphist_query(unique_permnos, start_date, end_date)
    df = client.raw_sql(sql, date_cols=["sdate", "edate"], params=params)

    windows: list[SecidWindow] = []
    for row in df.to_dict("records"):
        secid = row["secid"]
        if secid is None or secid != secid:  # NaN check without importing numpy
            continue
        sdate = to_date(row["sdate"])
        edate = to_date(row["edate"])
        valid_start, valid_end = clip_window(sdate, edate, start_date, end_date)
        score = row["score"]
        windows.append(
            SecidWindow(
                permno=int(row["permno"]),
                secid=int(round(secid)),
                valid_start=valid_start,
                valid_end=valid_end,
                score=score if score == score else None,
            )
        )
    return windows


def _overlapping_indices(windows: Sequence[Any]) -> set[int]:
    """Indices of windows in `windows` whose [valid_start, valid_end] overlaps
    another window's -- used to flag link-table inconsistencies rather than
    assume they can't happen."""
    flagged: set[int] = set()
    for i in range(len(windows)):
        for j in range(i + 1, len(windows)):
            a, b = windows[i], windows[j]
            if a.valid_start <= b.valid_end and b.valid_start <= a.valid_end:
                flagged.add(i)
                flagged.add(j)
    return flagged


def resolve_universe_secids(
    client: WRDSClient, universe: Sequence[UniverseEntry], start_date: date, end_date: date
) -> pd.DataFrame:
    """Resolve `universe` to OptionMetrics secid(s) valid over [start_date, end_date].

    See module docstring for the unresolved/ambiguity contract. Returns a
    flat DataFrame (columns match `SecidResolution`'s fields) rather than a
    list of dataclasses since this is what gets persisted as a datalake
    artifact and consumed row-wise by `wrds_client.optionmetrics.universe`.
    """
    if end_date < start_date:
        raise ValueError(f"end_date {end_date} is before start_date {start_date}")

    gvkey_resolutions = resolve_gvkeys(client, universe)

    gvkeys = sorted({r.gvkey for r in gvkey_resolutions})
    permno_windows = fetch_permno_windows(client, gvkeys, start_date, end_date) if gvkeys else []

    permnos = sorted({w.permno for w in permno_windows})
    secid_windows = fetch_secid_windows(client, permnos, start_date, end_date) if permnos else []

    permno_by_gvkey: dict[str, list[PermnoWindow]] = {}
    for w in permno_windows:
        permno_by_gvkey.setdefault(w.gvkey, []).append(w)

    secid_by_permno: dict[int, list[SecidWindow]] = {}
    for w in secid_windows:
        secid_by_permno.setdefault(w.permno, []).append(w)

    multi_permno_gvkeys = {
        gvkey for gvkey, ws in permno_by_gvkey.items() if len(ws) > 1 and _overlapping_indices(ws)
    }
    multi_secid_permnos = {
        permno for permno, ws in secid_by_permno.items() if len(ws) > 1 and _overlapping_indices(ws)
    }

    rows: list[SecidResolution] = []
    produced_input_indices: set[int] = set()

    for gres in gvkey_resolutions:
        for pw in permno_by_gvkey.get(gres.gvkey, []):
            for sw in secid_by_permno.get(pw.permno, []):
                valid_start = max(pw.valid_start, sw.valid_start)
                valid_end = min(pw.valid_end, sw.valid_end)
                if valid_start > valid_end:
                    continue

                flags: list[str] = []
                if gres.ambiguous:
                    flags.append("ambiguous_gvkey")
                if not pw.preferred:
                    flags.append("non_preferred_link")
                if gres.gvkey in multi_permno_gvkeys:
                    flags.append("multiple_permno_for_window")
                if pw.permno in multi_secid_permnos:
                    flags.append("multiple_secid_for_window")

                rows.append(
                    SecidResolution(
                        input_index=gres.input_index,
                        isin=gres.isin,
                        cik=gres.cik,
                        ticker=gres.ticker,
                        name=gres.name,
                        gvkey=gres.gvkey,
                        permno=pw.permno,
                        secid=sw.secid,
                        valid_start=valid_start,
                        valid_end=valid_end,
                        resolution_method=gres.resolution_method,
                        score=sw.score,
                        flags=tuple(flags),
                    )
                )
                produced_input_indices.add(gres.input_index)

    for input_index, entry in enumerate(universe):
        if input_index in produced_input_indices:
            continue
        rows.append(
            SecidResolution(
                input_index=input_index,
                isin=entry.isin,
                cik=entry.cik,
                ticker=entry.ticker,
                name=entry.name,
                gvkey=None,
                permno=None,
                secid=None,
                valid_start=None,
                valid_end=None,
                resolution_method=None,
                score=None,
                flags=("unresolved",),
            )
        )

    return pd.DataFrame(
        {
            "input_index": [r.input_index for r in rows],
            "isin": [r.isin for r in rows],
            "cik": [r.cik for r in rows],
            "ticker": [r.ticker for r in rows],
            "name": [r.name for r in rows],
            "gvkey": [r.gvkey for r in rows],
            "permno": [r.permno for r in rows],
            "secid": [r.secid for r in rows],
            "valid_start": [r.valid_start for r in rows],
            "valid_end": [r.valid_end for r in rows],
            "resolution_method": [r.resolution_method for r in rows],
            "score": [r.score for r in rows],
            "flags": [list(r.flags) for r in rows],
        }
    )


def ingest_secid_resolution(
    index: DatalakeIndex,
    client: WRDSClient,
    universe: Sequence[UniverseEntry],
    start_date: date,
    end_date: date,
    *,
    pipeline: str,
    pipeline_version: str,
    sources: Sequence[Artifact | str] | None = None,
    notes: str = "",
    pipeline_repo: str | None = None,
) -> Artifact:
    """Resolve `universe` and register the result as a datalake artifact.

    No single SQL query underlies this (it's several queries across the
    resolution chain), so this uses `index.run(...)` directly rather than
    `wrds_client.ingest.fetch_query_artifact`, mirroring
    `ravenpack.headlines.ingest.ingest_to_datalake`'s pattern.
    """
    resolved = resolve_universe_secids(client, universe, start_date, end_date)
    hyperparams = {
        "isins": sorted({e.isin for e in universe if e.isin}),
        "ciks": sorted({e.cik for e in universe if e.cik}),
        "tickers": sorted({e.ticker for e in universe if e.ticker}),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
    }

    with index.run(
        kind=KIND,
        pipeline=pipeline,
        pipeline_version=pipeline_version,
        pipeline_repo=pipeline_repo,
        hyperparams=hyperparams,
        sources=sources,
        notes=notes,
        hash_pattern="*.parquet",
    ) as run:
        resolved.to_parquet(run.out_dir / "data.parquet", index=False)
        run.note(f"{len(resolved)} resolution rows for {len(universe)} input entries")

    return index.get(run.artifact_id)
