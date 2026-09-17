"""ISIN/CIK -> gvkey -> permno resolution (WRDS `comp` and `crsp_a_ccm` libraries).

Chain confirmed live against WRDS:
    comp.security.isin  \\
                          -> gvkey -> (crsp_a_ccm.ccmxpf_linktable, time-bounded) -> permno
    comp.company.cik    /

`comp.security` is security/share-class level (one gvkey can have several ISINs
-- share classes, cross-listings); `comp.company` is company level. ISIN is
tried first (finer-grained); CIK is only used as a fallback for inputs whose
ISIN didn't match anything.

`crsp_a_ccm.ccmxpf_linktable.lpermno`/`lpermco` are stored as nullable
`DOUBLE PRECISION` (not `INTEGER` like `opcrsphist.permno`), and `usedflag`
can be -1 on superseded/junk rows, not just 0/1 -- confirmed live on Apple's
real link history. The standard CCM "preferred link" filter is
`linkprim IN ('P','C') AND linktype IN ('LU','LC') AND usedflag = 1`.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from wrds_client.client import WRDSClient

from universe.schema import UniverseEntry
from wrds_client.linking._shared import (
    clip_window,
    normalise_cik,
    normalise_isin,
    overlap_clause,
    to_date,
)


@dataclass(frozen=True)
class GvkeyResolution:
    """One (input, gvkey) match. Multiple rows per input when ambiguous.

    `input_index` is the position of the originating entry in the `universe`
    sequence passed to `resolve_gvkeys` -- used to join back to the exact
    input entry downstream without relying on (isin, cik, ticker, name)
    equality, which could be ambiguous if the universe contains duplicates.
    """

    input_index: int
    isin: str | None
    cik: str | None
    ticker: str | None
    name: str | None
    gvkey: str
    resolution_method: Literal["isin", "cik"]
    ambiguous: bool


@dataclass(frozen=True)
class PermnoWindow:
    """One gvkey<->permno link, clipped to a query's date range."""

    gvkey: str
    permno: int
    lpermco: int | None
    valid_start: date
    valid_end: date
    linkprim: str
    linktype: str
    usedflag: int
    preferred: bool


def build_isin_gvkey_query(isins: Sequence[str]) -> tuple[str, dict[str, Any]]:
    if not isins:
        raise ValueError("isins must be non-empty")
    normalised = tuple(normalise_isin(i) for i in isins)
    sql = "SELECT isin, gvkey FROM comp.security WHERE isin IN %(isins)s"
    return sql, {"isins": normalised}


def build_cik_gvkey_query(ciks: Sequence[str]) -> tuple[str, dict[str, Any]]:
    if not ciks:
        raise ValueError("ciks must be non-empty")
    normalised = tuple(normalise_cik(c) for c in ciks)
    sql = "SELECT cik, gvkey FROM comp.company WHERE cik IN %(ciks)s"
    return sql, {"ciks": normalised}


def resolve_gvkeys(client: WRDSClient, universe: Sequence[UniverseEntry]) -> list[GvkeyResolution]:
    """Resolve each universe entry to gvkey(s): ISIN first, CIK fallback.

    An entry matching >1 distinct gvkey emits one `GvkeyResolution` per match,
    all `ambiguous=True` -- never picked down to one silently. An entry
    matching nothing (via either identifier) emits no rows; the caller
    (`optionmetrics.resolve_universe_secids`) is responsible for surfacing
    that as `"unresolved"` rather than treating an empty result as success.
    """
    isins = sorted({normalise_isin(e.isin) for e in universe if e.isin})
    isin_to_gvkeys: dict[str, list[str]] = {}
    if isins:
        sql, params = build_isin_gvkey_query(isins)
        df = client.raw_sql(sql, params=params)
        for isin, gvkey in zip(df["isin"], df["gvkey"], strict=True):
            isin_to_gvkeys.setdefault(normalise_isin(isin), []).append(gvkey)

    results: list[GvkeyResolution] = []
    cik_fallback: list[tuple[int, UniverseEntry]] = []

    for input_index, entry in enumerate(universe):
        matched_gvkeys = isin_to_gvkeys.get(normalise_isin(entry.isin)) if entry.isin else None
        if matched_gvkeys:
            ambiguous = len(set(matched_gvkeys)) > 1
            for gvkey in matched_gvkeys:
                results.append(
                    GvkeyResolution(
                        input_index=input_index,
                        isin=entry.isin,
                        cik=entry.cik,
                        ticker=entry.ticker,
                        name=entry.name,
                        gvkey=gvkey,
                        resolution_method="isin",
                        ambiguous=ambiguous,
                    )
                )
        elif entry.cik:
            cik_fallback.append((input_index, entry))

    if cik_fallback:
        ciks = sorted({normalise_cik(entry.cik) for _, entry in cik_fallback if entry.cik})
        sql, params = build_cik_gvkey_query(ciks)
        df = client.raw_sql(sql, params=params)
        cik_to_gvkeys: dict[str, list[str]] = {}
        for cik, gvkey in zip(df["cik"], df["gvkey"], strict=True):
            cik_to_gvkeys.setdefault(normalise_cik(cik), []).append(gvkey)

        for input_index, entry in cik_fallback:
            matched_gvkeys = cik_to_gvkeys.get(normalise_cik(entry.cik))
            if not matched_gvkeys:
                continue
            ambiguous = len(set(matched_gvkeys)) > 1
            for gvkey in matched_gvkeys:
                results.append(
                    GvkeyResolution(
                        input_index=input_index,
                        isin=entry.isin,
                        cik=entry.cik,
                        ticker=entry.ticker,
                        name=entry.name,
                        gvkey=gvkey,
                        resolution_method="cik",
                        ambiguous=ambiguous,
                    )
                )

    return results


def build_ccm_link_query(
    gvkeys: Sequence[str], start_date: date, end_date: date, *, preferred_only: bool
) -> tuple[str, dict[str, Any]]:
    if not gvkeys:
        raise ValueError("gvkeys must be non-empty")
    if end_date < start_date:
        raise ValueError(f"end_date {end_date} is before start_date {start_date}")

    clauses = [
        "gvkey IN %(gvkeys)s",
        overlap_clause("linkdt", "linkenddt", "start_date", "end_date"),
    ]
    if preferred_only:
        clauses.append("linkprim IN ('P', 'C')")
        clauses.append("linktype IN ('LU', 'LC')")
        clauses.append("usedflag = 1")

    sql = (
        "SELECT gvkey, lpermno, lpermco, linkprim, linktype, usedflag, linkdt, linkenddt "
        "FROM crsp_a_ccm.ccmxpf_linktable WHERE " + " AND ".join(clauses)
    )
    params = {"gvkeys": tuple(gvkeys), "start_date": start_date, "end_date": end_date}
    return sql, params


def _rows_to_permno_windows(
    df: Any, start_date: date, end_date: date, *, preferred: bool
) -> list[PermnoWindow]:
    windows: list[PermnoWindow] = []
    for row in df.to_dict("records"):
        lpermno = row["lpermno"]
        if lpermno is None or (isinstance(lpermno, float) and lpermno != lpermno):
            continue
        linkdt = to_date(row["linkdt"])
        linkenddt = to_date(row["linkenddt"])
        valid_start, valid_end = clip_window(linkdt, linkenddt, start_date, end_date)
        lpermco = row["lpermco"]
        windows.append(
            PermnoWindow(
                gvkey=row["gvkey"],
                permno=int(round(lpermno)),
                lpermco=int(round(lpermco)) if lpermco == lpermco else None,
                valid_start=valid_start,
                valid_end=valid_end,
                linkprim=row["linkprim"],
                linktype=row["linktype"],
                usedflag=int(row["usedflag"]),
                preferred=preferred,
            )
        )
    return windows


def fetch_permno_windows(
    client: WRDSClient, gvkeys: Sequence[str], start_date: date, end_date: date
) -> list[PermnoWindow]:
    """Fetch gvkey<->permno links overlapping [start_date, end_date].

    Two-pass: preferred links first (standard CCM filter); for gvkeys with no
    preferred link overlapping the range, fall back to an unfiltered query so
    a non-preferred link is still surfaced (`preferred=False`) rather than the
    gvkey silently vanishing.
    """
    unique_gvkeys = sorted(set(gvkeys))
    if not unique_gvkeys:
        return []

    sql, params = build_ccm_link_query(unique_gvkeys, start_date, end_date, preferred_only=True)
    df = client.raw_sql(sql, date_cols=["linkdt", "linkenddt"], params=params)
    windows = _rows_to_permno_windows(df, start_date, end_date, preferred=True)

    covered = {w.gvkey for w in windows}
    missing = sorted(g for g in unique_gvkeys if g not in covered)
    if missing:
        sql2, params2 = build_ccm_link_query(missing, start_date, end_date, preferred_only=False)
        df2 = client.raw_sql(sql2, date_cols=["linkdt", "linkenddt"], params=params2)
        windows.extend(_rows_to_permno_windows(df2, start_date, end_date, preferred=False))

    return windows
