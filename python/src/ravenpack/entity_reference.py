"""RP_ENTITY_ID backfill from a RavenPack entity reference file, point in time.

The company reference file (``EntitiesResource.download_reference_file(entity_type=COMP)``,
``company_<date>.csv``, ~2.6 GB) is long-format::

    RP_ENTITY_ID, ENTITY_TYPE, DATA_TYPE, DATA_VALUE, RANGE_START, RANGE_END

one row per (entity, identifier, validity window); RANGE_END null = still valid. Only the
identifier rows matter here (ISIN, CUSIP, SEDOL, CIK: ~1.8M rows), so
``write_identifier_extract`` keeps them in a small parquet file with the same columns, and
``load_identifier_windows`` reads either form.

A universe row (plain dict with ``snapshot_date`` and any of isin/cusip/sedol/cik) is
matched in this order, each step only for rows still unmatched, and the step is recorded in
``rp_entity_match``:

    isin, cusip, sedol, cik   the identifier's window contains snapshot_date, and exactly
                              one entity holds it on that date
    isin_undated              the ISIN was held by exactly one entity over all its windows,
                              none of which covers the date (e.g. Deutsche Boerse AG's ISIN
                              has gaps in 2011-12 and 2016-17 during the merger attempts).
                              Not point in time; flagged so it can be excluded
    api                       optional ``POST /entity-mapping`` (isin + name + date) for what
                              is left; kept only when the top score is unique and the entity
                              is a company

Ambiguous matches (two entities holding the identifier on the date) are never resolved by a
guess: the row stays null and is counted. A value already present in a row is never
overwritten.
"""
from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl

if TYPE_CHECKING:
    from ravenpack.edge_api.client import RavenPackClient

log = logging.getLogger(__name__)

#: universe column -> reference DATA_TYPE, in matching order
KEYS: tuple[tuple[str, str], ...] = (("isin", "ISIN"), ("cusip", "CUSIP"), ("sedol", "SEDOL"),
                                     ("cik", "CIK"))
_COLUMNS = ("RP_ENTITY_ID", "ENTITY_TYPE", "DATA_TYPE", "DATA_VALUE", "RANGE_START", "RANGE_END")
_API_BATCH = 100


def normalise(column: str, value: Any) -> str | None:
    if value is None:
        return None
    v = str(value).strip().upper()
    if column == "cik":
        v = v.lstrip("0")
    return v or None


def _normalised_value() -> pl.Expr:
    v = pl.col("DATA_VALUE").str.strip_chars().str.to_uppercase()
    return (pl.when(pl.col("DATA_TYPE") == "CIK").then(v.str.strip_chars_start("0"))
            .otherwise(v).alias("DATA_VALUE"))


def _scan(path: Path) -> pl.LazyFrame:
    path = Path(path)
    if path.suffix.lower() in (".parquet", ".pq"):
        return pl.scan_parquet(path)
    return pl.scan_csv(path, infer_schema=False)


def write_identifier_extract(csv_path: Path, out_path: Path) -> int:
    """The identifier rows (ISIN, CUSIP, SEDOL, CIK) of a reference file, as parquet with
    the reference file's own columns (values untouched). Returns the row count."""
    keep = [t for _, t in KEYS]
    df = (_scan(csv_path).filter(pl.col("DATA_TYPE").is_in(keep)).select(_COLUMNS)
          .collect(engine="streaming"))
    df.write_parquet(out_path, compression="zstd")
    return df.height


def load_identifier_windows(path: Path, values: dict[str, set[str]] | None = None
                            ) -> pl.DataFrame:
    """(RP_ENTITY_ID, DATA_TYPE, DATA_VALUE, START, END) identifier windows from a reference
    CSV or its extract, values normalised (upper case; CIK without leading zeros), dates
    parsed (START null = valid from the file's first date, END null = still valid).
    ``values`` ({DATA_TYPE: {normalised value}}) restricts the scan to those values."""
    lf = (_scan(path).filter(pl.col("DATA_TYPE").is_in([t for _, t in KEYS]))
          .with_columns(_normalised_value()))
    if values is not None:
        cond = pl.lit(False)
        for dtype, vals in values.items():
            if vals:
                cond = cond | ((pl.col("DATA_TYPE") == dtype)
                               & pl.col("DATA_VALUE").is_in(sorted(vals)))
        lf = lf.filter(cond)
    return lf.select(
        "RP_ENTITY_ID", "DATA_TYPE", "DATA_VALUE",
        pl.col("RANGE_START").cast(pl.String).str.to_date(strict=False).alias("START"),
        pl.col("RANGE_END").cast(pl.String).str.to_date(strict=False).alias("END"),
    ).collect(engine="streaming")


def _requests(rows: Sequence[dict[str, Any]], todo: list[int], column: str) -> pl.DataFrame:
    idx, day, val = [], [], []
    for i in todo:
        v = normalise(column, rows[i].get(column))
        if v is not None:
            idx.append(i)
            day.append(rows[i]["snapshot_date"])
            val.append(v)
    return pl.DataFrame({"idx": idx, "snapshot_date": day, "DATA_VALUE": val},
                        schema={"idx": pl.Int64, "snapshot_date": pl.Date,
                                "DATA_VALUE": pl.String})


def _unique_entity(matches: pl.DataFrame) -> tuple[dict[int, str], int]:
    """idx -> entity where exactly one entity matched; and the number of ambiguous idx."""
    per = matches.group_by("idx").agg(pl.col("RP_ENTITY_ID").unique())
    unique = per.filter(pl.col("RP_ENTITY_ID").list.len() == 1)
    return (dict(zip(unique["idx"].to_list(),
                     unique["RP_ENTITY_ID"].list.first().to_list(), strict=True)),
            per.height - unique.height)


def backfill_rp_entity_id(
    rows: Sequence[dict[str, Any]],
    windows: pl.DataFrame,
    *,
    client: RavenPackClient | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Fill ``rp_entity_id`` / ``rp_entity_match`` (see the module docstring). Returns new
    row dicts and the count per outcome (match steps, ``ambiguous_<key>``, ``unmatched``)."""
    out = [dict(r) for r in rows]
    stats: Counter[str] = Counter()
    todo = [i for i, r in enumerate(out) if not r.get("rp_entity_id")]

    def assign(found: dict[int, str], how: str) -> None:
        for i, ent in found.items():
            out[i]["rp_entity_id"] = ent
            out[i]["rp_entity_match"] = how
        stats[how] += len(found)

    for column, dtype in KEYS:
        req = _requests(out, todo, column)
        if req.is_empty():
            continue
        win = windows.filter(pl.col("DATA_TYPE") == dtype)
        m = req.join(win, on="DATA_VALUE").filter(
            (pl.col("START").is_null() | (pl.col("START") <= pl.col("snapshot_date")))
            & (pl.col("END").is_null() | (pl.col("snapshot_date") <= pl.col("END"))))
        found, ambiguous = _unique_entity(m)
        assign(found, column)
        stats[f"ambiguous_{column}"] += ambiguous
        todo = [i for i in todo if i not in found]

    req = _requests(out, todo, "isin")
    if not req.is_empty():
        m = req.join(windows.filter(pl.col("DATA_TYPE") == "ISIN"), on="DATA_VALUE")
        found, _ = _unique_entity(m)
        assign(found, "isin_undated")
        todo = [i for i in todo if i not in found]

    if client is not None and todo:
        found = _map_via_api(client, out, todo)
        assign(found, "api")
        todo = [i for i in todo if i not in found]

    stats["unmatched"] = len(todo)
    log.info("rp_entity_id: %s", dict(stats))
    return out, dict(stats)


def _map_via_api(client: RavenPackClient, rows: list[dict[str, Any]], todo: list[int]
                 ) -> dict[int, str]:
    """``POST /entity-mapping`` once per distinct ISIN still unmatched (at its latest
    snapshot date); a match is kept only when unique at the top score and a company."""
    latest: dict[str, tuple[date, str]] = {}
    for i in todo:
        isin = normalise("isin", rows[i].get("isin"))
        if isin is None:
            continue
        d = rows[i]["snapshot_date"]
        if isin not in latest or d > latest[isin][0]:
            latest[isin] = (d, rows[i].get("name") or "")
    by_isin: dict[str, str] = {}
    items = sorted(latest.items())
    for k in range(0, len(items), _API_BATCH):
        batch = items[k:k + _API_BATCH]
        resp = client.entities.map([
            {"client_id": isin, "isin": isin, "name": name or None, "date": d,
             "entity_type": "COMP"} for isin, (d, name) in batch])
        for mapped in resp.identifiers_mapped:
            req = mapped.requested_data
            ents = [e for e in mapped.rp_entities if e.rp_entity_type in (None, "COMP")]
            if req is None or req.client_id is None or not ents:
                continue
            top = max(e.score if e.score is not None else float("-inf") for e in ents)
            best = [e for e in ents if (e.score if e.score is not None else float("-inf")) == top]
            if len(best) == 1:
                by_isin[req.client_id] = best[0].rp_entity_id
    return {i: by_isin[isin] for i in todo
            if (isin := normalise("isin", rows[i].get("isin"))) in by_isin}
