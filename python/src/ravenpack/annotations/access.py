"""Reading the RavenPack Annotations tables: which artifacts, and which calendar day.

Consumers (embeddings, sentiment, mu_asof, the narrative scorer) read the headlines
table (``rp_headlines``: one row per story) and, when they need entities, its
sibling (``rp_headline_entities``: one row per story x entity). Both are produced by
``ravenpack.annotations.ingest`` in one pass; the entities artifact records the
headlines artifact as its only source.

Time: TIMESTAMP_UTC is the vendor's UTC publication time, stored as
``Datetime(us, "UTC")``: ``STORAGE_TZ`` states that once. A consumer bucketing stories
into calendar days does it in a chosen time zone (default UTC) with ``day_of``
(polars) or ``duckdb_day`` (SQL), converting from UTC explicitly; DuckDB sessions are
pinned to UTC (``pin_utc``) so no cast ever depends on the machine's zone.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl

from ravenpack.annotations.feeds import RPA1_ALL
from ravenpack.annotations.fields import TIMESTAMP

if TYPE_CHECKING:
    from datalake import Artifact, DatalakeIndex

HEADLINES_KIND = RPA1_ALL.headlines_kind          # "rp_headlines"
ENTITIES_KIND = RPA1_ALL.entities_kind            # "rp_headline_entities"
STORAGE_TZ = "UTC"
DEFAULT_DAY_TZ = "UTC"


class HeadlinesError(ValueError):
    """Not a usable headlines (or entities) artifact."""


def require_headlines(art: Artifact) -> Artifact:
    if art.kind != HEADLINES_KIND:
        raise HeadlinesError(f"{art.artifact_id} is a {art.kind}, not a {HEADLINES_KIND} "
                             "(the headline consumers read the RavenPack Annotations tables)")
    return art


def latest_headlines(index: DatalakeIndex) -> Artifact:
    """The newest complete, non-deprecated headlines artifact."""
    return index.latest(HEADLINES_KIND)


def entities_of(index: DatalakeIndex, headlines: Artifact) -> Artifact:
    """The entities sibling of ``headlines``: its unique complete, non-deprecated
    ``rp_headline_entities`` child."""
    require_headlines(headlines)
    kids = [index.get(c) for c in index.children(headlines.artifact_id)
            if c.split("__", 1)[0] == ENTITIES_KIND]
    usable = [k for k in kids if not k.partial and not k.deprecated]
    if len(usable) != 1:
        raise HeadlinesError(f"{headlines.artifact_id} has {len(usable)} complete "
                             f"{ENTITIES_KIND} sibling(s) ({[k.artifact_id for k in kids]})")
    return usable[0]


def check_tz(tz: str) -> str:
    """``tz`` if polars knows it (an IANA name such as "UTC" or "Europe/Paris")."""
    try:
        pl.Series([0], dtype=pl.Datetime("us", STORAGE_TZ)).dt.convert_time_zone(tz)
    except Exception as exc:  # noqa: BLE001 - polars raises several types here
        raise ValueError(f"unknown time zone {tz!r}") from exc
    return tz


def day_of(col: str | pl.Expr = TIMESTAMP, tz: str = DEFAULT_DAY_TZ) -> pl.Expr:
    """Calendar day (Date) in ``tz`` of a UTC timestamp column."""
    expr = pl.col(col) if isinstance(col, str) else col
    if tz != STORAGE_TZ:
        expr = expr.dt.convert_time_zone(tz)
    return expr.dt.date()


def duckdb_day(col: str = TIMESTAMP, tz: str = DEFAULT_DAY_TZ) -> str:
    """SQL for the calendar day in ``tz`` of a TIMESTAMPTZ column (session pinned to
    UTC by ``pin_utc``): ``timezone(tz, ts)`` gives the local wall time in ``tz``."""
    check_tz(tz)
    return f"CAST(timezone('{tz}', {col}) AS DATE)"


def pin_utc(conn: Any) -> Any:
    """Pin a DuckDB connection's session time zone to UTC (returns it)."""
    conn.execute("SET TimeZone='UTC'")
    return conn


def day_tz_params(tz: str) -> dict[str, Any]:
    """Hyperparams entry for a day time zone: none for the default (ids unchanged)."""
    check_tz(tz)
    return {} if tz == DEFAULT_DAY_TZ else {"day_tz": tz}


__all__ = ["HEADLINES_KIND", "ENTITIES_KIND", "STORAGE_TZ", "DEFAULT_DAY_TZ", "HeadlinesError",
           "require_headlines", "latest_headlines", "entities_of", "check_tz", "day_of",
           "duckdb_day", "pin_utc", "day_tz_params"]
