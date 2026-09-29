"""Index-constituent universes, loaded by index name from a directory of snapshot files.

A directory (read only, never modified) holds, per index, files named::

    {INDEX}_constituents-{from}_to_{to}.parquet    required
    {INDEX}_metadata-{from}_to_{to}.parquet        optional

``INDEX`` is the universe name (``MSCI_WORLD``, ``SP500``, ...); ``list_index_universes``
lists the names a directory offers. Exactly one file per index and kind is expected.

constituents
    snapshot_date, name, ticker, isin (+ optional figi / cusip / sedol / gvkey, and any vendor
    columns, which are ignored): one row per constituent and snapshot date. Duplicate
    (snapshot_date, isin) rows raise.
metadata
    the same securities with country_name / country_iso / region (+ GICS, not used).
    Metadata files can repeat a security once per classification it ever had, on every
    snapshot (such a file lists a stock's pre- and post-2018 GICS codes on every date), so
    their GICS is not point in time and is never taken: GICS is left empty for
    ``wrds_client.capitaliq.backfill_from_capitaliq`` to fill as of each snapshot_date
    (Compustat comp.co_hgic). Geography is taken only where it is unique for the
    (snapshot_date, isin): a re-domiciled company can show two countries on one date.

Constituents without an ISIN are kept as rows; they fail ``UniverseEntry`` validation
later unless enrichment finds a CUSIP, and are counted there.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import polars as pl

CONSTITUENTS = "constituents"
METADATA = "metadata"
_FILE = re.compile(r"^(?P<index>[A-Za-z0-9_]+?)_(?P<kind>constituents|metadata)-"
                   r"(?P<start>[^_]+)_to_(?P<end>[^_]+)\.parquet$")
_IDENTIFIERS = ("snapshot_date", "name", "ticker", "isin", "figi", "cusip", "sedol", "gvkey")
_GEO = ("country_name", "country_iso", "region")


def _files(directory: Path) -> dict[tuple[str, str], list[Path]]:
    found: dict[tuple[str, str], list[Path]] = {}
    for p in sorted(Path(directory).glob("*.parquet")):
        m = _FILE.match(p.name)
        if m:
            found.setdefault((m["index"], m["kind"]), []).append(p)
    return found


def list_index_universes(directory: Path) -> list[str]:
    """The index names with a constituents file in ``directory``."""
    return sorted({index for index, kind in _files(directory) if kind == CONSTITUENTS})


def _file(directory: Path, index: str, kind: str, *, required: bool) -> Path | None:
    paths = _files(directory).get((index, kind), [])
    if len(paths) > 1:
        raise ValueError(f"{len(paths)} {kind} files for {index} in {directory}: "
                         f"{[p.name for p in paths]}; keep exactly one")
    if not paths:
        if required:
            raise FileNotFoundError(
                f"no {index}_{kind}-*_to_*.parquet in {directory}; available: "
                f"{list_index_universes(directory)}")
        return None
    return paths[0]


def index_universe_rows(directory: Path, index: str) -> list[dict[str, Any]]:
    """Unvalidated universe row dicts (``universe.schema`` columns) for ``index``."""
    cons = pl.read_parquet(_file(directory, index, CONSTITUENTS, required=True))
    missing = [c for c in ("snapshot_date", "name", "ticker", "isin") if c not in cons.columns]
    if missing:
        raise ValueError(f"{index} constituents lack column(s) {missing}")
    base = cons.select([c for c in _IDENTIFIERS if c in cons.columns])
    keyed = base.filter(pl.col("isin").is_not_null()).select("snapshot_date", "isin")
    if keyed.is_duplicated().any():
        raise ValueError(f"{index} constituents have duplicate (snapshot_date, isin) rows")

    meta_path = _file(directory, index, METADATA, required=False)
    if meta_path is not None:
        meta = pl.read_parquet(meta_path)
        geo_cols = [c for c in _GEO if c in meta.columns]
        if geo_cols:
            geo = (meta.filter(pl.col("isin").is_not_null())
                   .group_by("snapshot_date", "isin")
                   .agg([pl.col(c).unique() for c in geo_cols])
                   .with_columns([pl.when(pl.col(c).list.len() == 1)
                                  .then(pl.col(c).list.first()).otherwise(None).alias(c)
                                  for c in geo_cols]))
            base = base.join(geo, on=["snapshot_date", "isin"], how="left")
    return base.sort("snapshot_date", "isin").to_dicts()
