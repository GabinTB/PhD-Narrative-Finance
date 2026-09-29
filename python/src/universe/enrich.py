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
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from deutsche_boerse.rdi import RdiClient
    from lseg_client.client import LsegClient
    from ravenpack.edge_api.client import RavenPackClient
    from wrds_client.client import WRDSClient


def enrich_universe(
    rows: Sequence[dict[str, Any]],
    *,
    wrds_client: WRDSClient | None = None,
    resolve_sedol: bool = True,
    rp_reference: Path | None = None,
    rp_client: RavenPackClient | None = None,
    lseg_client: LsegClient | None = None,
    dbg_client: RdiClient | None = None,
    dbg_cache_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Backfill missing isin/cusip/cik/gvkey/ticker/ciq_secid/country_*/
    region/gics_*/sedol/rp_entity_id/dbga_secid fields, in order: CapitalIQ, then LSEG
    (sedol), then the LSEG API (ric/lseg_permid/lei, sedol still missing), then RavenPack
    (rp_entity_id, which can use the cusip/sedol/cik filled before it), then Deutsche
    Boerse (dbga_secid). Each step only fills
    genuinely-missing fields (never overwrites a value already present) and
    is skipped entirely if its prerequisites aren't given:

    - CapitalIQ (isin/cusip/cik/gvkey/ticker/ciq_secid/country_*/region/
      gics_*): needs `wrds_client`.
    - LSEG (sedol): needs `wrds_client` and `resolve_sedol=True` (the default).
    - LSEG API (ric, lseg_permid, lei; sedol where WRDS left it empty): needs
      `lseg_client` (`lseg_client.LsegClient`); current values keyed by ISIN.
    - RavenPack (rp_entity_id, rp_entity_match): needs `rp_reference`, a company entity
      reference file or its identifier extract (`ravenpack.entity_reference`), matched
      point in time; `rp_client` adds the `/entity-mapping` fallback for what is left.
    - Deutsche Boerse (dbga_secid = Xetra SecurityID, by ISIN as of each snapshot date):
      needs `dbg_client` (`deutsche_boerse.rdi.RdiClient`); A7 reference tables are cached
      in `dbg_cache_dir` (default `$RAW_DATA_PATH/Deutsche_Boerse/rdi`). Eurex products are
      not stored in the universe: `deutsche_boerse.rdi.eurex_derivatives` joins them on need.

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

    if lseg_client is not None:
        from lseg_client.enrich import backfill_lseg_ids

        result, _ = backfill_lseg_ids(result, lseg_client)

    if rp_reference is not None:
        from ravenpack.entity_reference import (
            KEYS,
            backfill_rp_entity_id,
            load_identifier_windows,
            normalise,
        )

        values = {dtype: {v for r in result if (v := normalise(col, r.get(col)))}
                  for col, dtype in KEYS}
        windows = load_identifier_windows(Path(rp_reference), values)
        result, _ = backfill_rp_entity_id(result, windows, client=rp_client)

    if dbg_client is not None:
        from deutsche_boerse import RAW_ROOT
        from deutsche_boerse.rdi import backfill_dbga_secid

        cache = Path(dbg_cache_dir) if dbg_cache_dir else RAW_ROOT / "rdi"
        result, _ = backfill_dbga_secid(result, dbg_client, cache)

    return result
