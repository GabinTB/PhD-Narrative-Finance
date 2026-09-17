"""Universe row enrichment: WRDS CapitalIQ, WRDS/LSEG, Deutsche Boerse backfill.

**Not re-exported from `universe/__init__.py`** -- import this module
explicitly (`from universe.enrich import enrich_universe`). `universe.schema`
and `universe.ingest` stay dependency-free (no import of `wrds_client`/
`deutsche_boerse`), since `wrds_client.linking` already imports `UniverseEntry`
from `universe.schema` -- if `universe`'s own `__init__.py` pulled in this
module, and this module pulled in `wrds_client`, that would invert the
intended dependency direction (the foundational `universe` package reaching
back into one specific connector). This module is the one place allowed to
depend on specific connectors, since enrichment inherently needs to reach
into each source's own tables/APIs.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from wrds_client.client import WRDSClient


def enrich_universe(
    rows: Sequence[dict[str, Any]],
    *,
    wrds_client: WRDSClient | None = None,
    resolve_sedol: bool = True,
    dbga_mic: str | None = None,
    dbga_market_segment_id: int | None = None,
) -> list[dict[str, Any]]:
    """Backfill missing isin/cusip/cik/gvkey/ticker/ciq_secid/country_*/
    region/gics_*/sedol/dbga_secid fields, in order: CapitalIQ, then LSEG
    (sedol), then Deutsche Boerse (dbga_secid). Each step only fills
    genuinely-missing fields (never overwrites a value already present) and
    is skipped entirely if its prerequisites aren't given:

    - CapitalIQ (isin/cusip/cik/gvkey/ticker/ciq_secid/country_*/region/
      gics_*): needs `wrds_client`.
    - LSEG (sedol): needs `wrds_client` and `resolve_sedol=True` (the default).
    - Deutsche Boerse (dbga_secid): needs both `dbga_mic` and
      `dbga_market_segment_id` -- there's no way to derive which market/
      segment to scan from an ISIN alone (see `deutsche_boerse.identifiers`).

    Operates on plain row dicts, not `UniverseEntry`, so a row missing even a
    mandatory field (e.g. `ticker`) can still be enriched -- `UniverseEntry`
    construction/validation happens afterwards, via
    `universe.schema.entries_from_rows`.
    """
    result = [dict(r) for r in rows]

    if wrds_client is not None:
        from wrds_client.capitaliq import backfill_from_capitaliq

        result = backfill_from_capitaliq(wrds_client, result)

        if resolve_sedol:
            from wrds_client.lseg import backfill_sedol

            result = backfill_sedol(wrds_client, result)

    if dbga_mic and dbga_market_segment_id is not None:
        from deutsche_boerse.identifiers import backfill_dbga_secid

        result = backfill_dbga_secid(result, dbga_mic, dbga_market_segment_id)

    return result
