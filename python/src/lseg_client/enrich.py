"""Universe enrichment with LSEG identifiers: ric, lseg_permid (organisation PermID), lei,
and sedol where still missing.

Rows are matched by ISIN (``LsegClient.rics_from_isins``, then ``identifiers`` on the RICs
found). These are CURRENT values keyed by ISIN, so a row's snapshot date does not enter:
what LSEG maps an ISIN to today is applied to every snapshot of that ISIN. SEDOL therefore
comes first from ``wrds_client.lseg.backfill_sedol`` (LSEG's tr_common through WRDS, point in
time), which ``universe.enrich.enrich_universe`` runs before this step; this one only fills
SEDOLs still missing, and adds what WRDS does not carry for the universe (RIC, PermID, LEI).
Never overwrites a value already present; unresolved ISINs stay null and are counted.
"""
from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from lseg_client.client import LsegClient

log = logging.getLogger(__name__)

#: universe column <- LsegClient.identifiers column
_COLUMNS = {"ric": "ric", "lseg_permid": "org_permid", "lei": "lei", "sedol": "sedol"}


def _isin(row: dict[str, Any]) -> str | None:
    v = row.get("isin")
    return str(v).strip().upper() if v else None


def backfill_lseg_ids(rows: Sequence[dict[str, Any]], client: LsegClient
                      ) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """New row dicts with ``_COLUMNS`` filled where missing, and counts: rows filled per
    column, ``unresolved`` rows whose ISIN LSEG does not convert, ``no_isin``."""
    out = [dict(r) for r in rows]
    need = [i for i, r in enumerate(out)
            if _isin(r) and any(not r.get(c) for c in _COLUMNS)]
    stats: Counter[str] = Counter(no_isin=sum(1 for r in out if not _isin(r)))
    if not need:
        return out, dict(stats)
    conv = client.rics_from_isins(sorted({_isin(out[i]) for i in need}))
    ids = client.identifiers(conv["ric"]) if not conv.empty else pd.DataFrame(columns=["ric"])
    info = conv.merge(ids.drop(columns=["isin"], errors="ignore"), on="ric", how="left",
                      suffixes=("_conv", ""))
    if "org_permid_conv" in info:
        info["org_permid"] = info["org_permid"].fillna(info["org_permid_conv"])
    by_isin = info.set_index("isin").to_dict("index")
    for i in need:
        hit = by_isin.get(_isin(out[i]))
        if hit is None:
            stats["unresolved"] += 1
            continue
        for col, src in _COLUMNS.items():
            v = hit.get(src)
            if not out[i].get(col) and v is not None and not pd.isna(v) and str(v):
                out[i][col] = str(int(v)) if isinstance(v, float) and v.is_integer() else str(v)
                stats[col] += 1
    log.info("lseg ids: %s", dict(stats))
    return out, dict(stats)
