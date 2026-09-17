"""Canonical security-identifier record shared across data-source connectors.

One row = one constituent as observed on `snapshot_date`. `snapshot_date`,
`name`, and `ticker` are mandatory; at least one of `isin`/`cusip` is
mandatory; every other identifier/classification column is optional and, if
missing, is meant to be backfilled by `universe.enrich.enrich_universe` (WRDS
CapitalIQ, WRDS/LSEG, Deutsche Boerse -- see that module, not this one, for
resolution logic). This module defines the shared record and how a
user-supplied master list round-trips to/from a DataFrame -- not
source-specific resolution logic, which stays in each connector.

Loading is split into two stages so enrichment can run *before* validation
(a row missing only `ticker`, say, still needs to reach `enrich_universe` --
it can't if `UniverseEntry` construction, which requires `ticker`, happens
first): `load_universe_rows`/`_rows_from_*` produce plain, unvalidated
row dicts; `entries_from_rows` (used by `load_universe_csv`/
`load_universe_parquet`/`from_frame` for direct, non-enriched use) is where
validation actually happens, skipping+logging whatever's still invalid
rather than aborting the whole load on the first bad row.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)


def _to_date(value: Any) -> date | None:
    """Normalise a value read back from a parquet date column.

    pandas parses date columns to `Timestamp` (missing -> `NaT`), not
    `datetime.date`/`None`. Deliberately duplicated (not imported) from
    `wrds_client.linking._shared.to_date`, which does the same thing for WRDS
    query results: `universe` must stay dependency-free (no import of
    `wrds_client`), since `wrds_client.linking` already imports `UniverseEntry`
    from here -- importing the other way would be a real circular import
    (verified: it breaks at `wrds_client.linking.__init__`, which needs
    `UniverseEntry` from this module before this module has finished
    defining it).
    """
    if value is None or pd.isna(value):
        return None
    if isinstance(value, date) and not isinstance(value, pd.Timestamp):
        return value
    return pd.Timestamp(value).date()


#: Optional string identifier/classification columns (everything except the
#: mandatory snapshot_date/name/ticker and the date-typed snapshot_date).
_OPTIONAL_COLUMNS = (
    "isin",
    "cusip",
    "sedol",
    "cik",
    "figi",
    "gvkey",
    "ciq_secid",
    "dbga_secid",
    "country_name",
    "country_iso",
    "region",
    "gics_sector",
    "gics_sector_code",
    "gics_industry_group",
    "gics_industry_group_code",
    "gics_industry",
    "gics_industry_code",
    "gics_subindustry",
    "gics_subindustry_code",
)

#: All columns in file order: snapshot_date, mandatory strings, then optional strings.
_COLUMNS = ("snapshot_date", "name", "ticker", *_OPTIONAL_COLUMNS)

#: Rows logged verbatim in the skipped-row warning before it just gives a count.
_MAX_LOGGED_SKIP_REASONS = 5


@dataclass(frozen=True)
class UniverseEntry:
    """One constituent of a research universe, as observed on `snapshot_date`.

    `snapshot_date`, `name`, `ticker` are mandatory, and at least one of
    `isin`/`cusip` is mandatory -- an entry lacking any of these can't be
    backfilled or resolved by any connector, so it's rejected at construction
    rather than silently carried through as dead weight. Every other field is
    optional, filled in later by `universe.enrich.enrich_universe` where
    possible.
    """

    snapshot_date: date | None = None
    name: str | None = None
    ticker: str | None = None
    isin: str | None = None
    cusip: str | None = None
    sedol: str | None = None
    cik: str | None = None
    figi: str | None = None
    gvkey: str | None = None
    ciq_secid: str | None = None
    dbga_secid: str | None = None
    country_name: str | None = None
    country_iso: str | None = None
    region: str | None = None
    gics_sector: str | None = None
    gics_sector_code: str | None = None
    gics_industry_group: str | None = None
    gics_industry_group_code: str | None = None
    gics_industry: str | None = None
    gics_industry_code: str | None = None
    gics_subindustry: str | None = None
    gics_subindustry_code: str | None = None

    def __post_init__(self) -> None:
        if self.snapshot_date is None:
            raise ValueError("UniverseEntry requires snapshot_date")
        if not self.name:
            raise ValueError("UniverseEntry requires name")
        if not self.ticker:
            raise ValueError("UniverseEntry requires ticker")
        if not (self.isin or self.cusip):
            raise ValueError("UniverseEntry requires at least one of isin/cusip")


def _rows_from_csv(path: Path) -> list[dict[str, Any]]:
    """Unvalidated row dicts from a universe CSV (see `load_universe_csv` for
    the column/blank-cell conventions)."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing_mandatory = [c for c in ("snapshot_date", "name", "ticker") if c not in df.columns]
    if missing_mandatory:
        raise ValueError(f"{path} is missing mandatory column(s): {missing_mandatory}")

    present_optional = [c for c in _OPTIONAL_COLUMNS if c in df.columns]
    rows: list[dict[str, Any]] = []
    for row in df.itertuples(index=False):
        snapshot_date_str = row.snapshot_date or None
        parsed_date = date.fromisoformat(snapshot_date_str) if snapshot_date_str else None
        rows.append(
            {
                "snapshot_date": parsed_date,
                "name": row.name or None,
                "ticker": row.ticker or None,
                **{c: (getattr(row, c) or None) for c in present_optional},
            }
        )
    return rows


def _rows_from_frame(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Unvalidated row dicts from an already-loaded DataFrame (see `from_frame`)."""
    present_optional = [c for c in _OPTIONAL_COLUMNS if c in df.columns]
    rows: list[dict[str, Any]] = []
    for row in df.itertuples(index=False):
        rows.append(
            {
                "snapshot_date": _to_date(row.snapshot_date),
                "name": row.name if pd.notna(row.name) else None,
                "ticker": row.ticker if pd.notna(row.ticker) else None,
                **{
                    c: (getattr(row, c) if pd.notna(getattr(row, c)) else None)
                    for c in present_optional
                },
            }
        )
    return rows


def load_universe_rows(path: Path) -> list[dict[str, Any]]:
    """Load a universe file (`.csv` or `.parquet`/`.pq`) as unvalidated row
    dicts -- all `_COLUMNS`, `None` where absent/blank. No `UniverseEntry`
    validation happens here; this is the entry point for
    `universe.enrich.enrich_universe`, which needs to run *before*
    validation so it can fill in even mandatory fields like `ticker`. For
    direct, non-enriched use, prefer `load_universe_file`/`load_universe_csv`/
    `load_universe_parquet`, which validate.
    """
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _rows_from_csv(path)
    if suffix in (".parquet", ".pq"):
        return _rows_from_frame(pd.read_parquet(path))
    raise ValueError(
        f"unsupported universe file suffix {suffix!r} for {path}; use .csv or .parquet"
    )


def entries_from_rows(
    rows: Sequence[dict[str, Any]], *, source: str = "<rows>"
) -> list[UniverseEntry]:
    """Construct `UniverseEntry`s from row dicts (as produced by
    `load_universe_rows`/`_rows_from_*`), skipping (and logging) any row that
    fails `UniverseEntry.__post_init__` rather than aborting the whole load
    on the first bad row.

    Real universe data legitimately has some fraction of rows missing a
    mandatory field (e.g. LGT's MSCI World metadata: ~3.75% of rows have no
    ticker) -- that's a data-quality fact to surface, not something a single
    bad row should be allowed to block the other 99%+ of valid rows over.
    """
    entries: list[UniverseEntry] = []
    skipped: list[str] = []
    for kwargs in rows:
        try:
            entries.append(UniverseEntry(**kwargs))
        except ValueError as exc:
            skipped.append(str(exc))
    if skipped:
        examples = "; ".join(skipped[:_MAX_LOGGED_SKIP_REASONS])
        extra = len(skipped) - _MAX_LOGGED_SKIP_REASONS
        more = f" (+{extra} more)" if extra > 0 else ""
        log.warning(
            "%s: skipped %d/%d rows failing UniverseEntry validation -- %s%s",
            source, len(skipped), len(rows), examples, more,
        )
    return entries


def load_universe_csv(path: Path) -> list[UniverseEntry]:
    """Load and validate a universe CSV.

    Columns: `snapshot_date,name,ticker` (mandatory) plus any subset of the
    optional identifier/classification columns (see `UniverseEntry`).
    `snapshot_date` cells parse via `date.fromisoformat`; blank cells in any
    other column become `None`. Rows missing a mandatory field are skipped
    (and logged as a warning with a count + example reasons) rather than
    aborting the whole load -- see `entries_from_rows`. For enrichment
    *before* validation (so e.g. a missing ticker can still be backfilled),
    use `load_universe_rows` + `universe.enrich.enrich_universe` instead.
    """
    return entries_from_rows(_rows_from_csv(path), source=str(path))


def load_universe_parquet(path: Path) -> list[UniverseEntry]:
    """Load and validate a universe parquet file (the same schema `to_frame` writes)."""
    return entries_from_rows(_rows_from_frame(pd.read_parquet(path)), source=str(path))


def load_universe_file(path: Path) -> list[UniverseEntry]:
    """Load and validate a universe from `path`, dispatching on its suffix
    (`.csv` or `.parquet`/`.pq`)."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return load_universe_csv(path)
    if suffix in (".parquet", ".pq"):
        return load_universe_parquet(path)
    raise ValueError(
        f"unsupported universe file suffix {suffix!r} for {path}; use .csv or .parquet"
    )


def to_frame(entries: Sequence[UniverseEntry]) -> pd.DataFrame:
    return pd.DataFrame(
        {"snapshot_date": [e.snapshot_date for e in entries]}
        | {"name": [e.name for e in entries], "ticker": [e.ticker for e in entries]}
        | {c: [getattr(e, c) for e in entries] for c in _OPTIONAL_COLUMNS}
    )


def from_frame(df: pd.DataFrame, *, source: str = "<dataframe>") -> list[UniverseEntry]:
    """Build and validate `UniverseEntry`s from a DataFrame that has at least
    `snapshot_date`/`name`/`ticker`. Any `_OPTIONAL_COLUMNS` not present in
    `df` are left at their dataclass default (`None`) rather than raising --
    a freshly-converted, not-yet-enriched universe (see `universe.enrich`)
    legitimately only has the columns its source actually provided. Rows
    missing a mandatory field are skipped (and logged as a warning with a
    count + example reasons) rather than aborting the whole load -- see
    `entries_from_rows`.
    """
    return entries_from_rows(_rows_from_frame(df), source=source)


#: Fields that `universe.enrich.enrich_universe` is expected to fill in and
#: `universe.ingest.register_universe_entries` treats as required post-backfill
#: (`sedol` is deliberately excluded -- it stays optional even after an
#: attempted backfill).
MANDATORY_AFTER_ENRICHMENT = ("isin", "cusip", "cik", "gvkey", "dbga_secid")


def find_incomplete(entries: Sequence[UniverseEntry]) -> list[tuple[int, list[str]]]:
    """For each entry (by index) still missing any `MANDATORY_AFTER_ENRICHMENT`
    field, the list of which ones. Pure inspection of already-loaded entries --
    does not attempt to resolve anything itself (see `universe.enrich` for that).
    """
    incomplete: list[tuple[int, list[str]]] = []
    for i, entry in enumerate(entries):
        missing = [f for f in MANDATORY_AFTER_ENRICHMENT if getattr(entry, f) is None]
        if missing:
            incomplete.append((i, missing))
    return incomplete
