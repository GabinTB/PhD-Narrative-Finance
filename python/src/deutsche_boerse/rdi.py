"""Xetra and Eurex identifiers from A7 reference data (RDI v2), point in time.

A7 serves the T7 reference data of every trading day (XETR from 2019-04-02, XEUR from
2020-11-02, probed 2026-09-29)::

    GET /v2/rdi/{mic}/                              trading dates
    GET /v2/rdi/{mic}/{date}/                       market segment ids
    GET /v2/rdi/{mic}/{date}/{segment}/             security ids of a segment
    GET /v2/rdi/{mic}/{date}/{segment}/{security}   ProductSnapshot + instrument messages

    XETR  one segment per listed instrument; ProductSnapshot.MarketSegment is its ISIN and
          the InstrumentSnapshot carries SecurityID (the ``dbga_secid``), SecurityAlt (ISIN
          "4", WKN "B", mnemonic "8"), SecurityType, SecurityDesc
    XEUR  one segment per product; ProductSnapshot.MarketSegment is the product symbol,
          DerivativesDescriptorGroup.ParentMktSegmID its family (FSTK single-stock futures,
          OSTK single-stock options, TRF total return futures, ...) and
          UnderlyingDescriptorGroup.UnderlyingSecurityID the underlying (an ISIN when
          UnderlyingSecurityIDSource = "4")

``reference_table`` turns one (mic, date) into one row per product (XEUR) or listed security
(XETR); a scan costs two calls per segment (~0.3 s each, run on a thread pool). XETR dates
reuse the rows of an earlier date for segments it already knows and only resolve new ones,
after re-fetching a random sample of the reused segments (any mismatch makes that date a full
scan): a Xetra segment is named by its ISIN, so it does not change (0 of 50 sampled segments
differed across 2019, 2022 and 2026). XEUR dates are always scanned in full: a product keeps
its segment but its underlying ISIN changes with corporate actions (5 of 50 sampled products
across 2020-2026, e.g. CAJ FR0000125585 -> FR001400OKR3), which a sample would miss.
Tables are cached as ``{cache_dir}/{mic}/{YYYYMMDD}.parquet``, written only when absent.

``backfill_dbga_secid`` fills a universe row's Xetra SecurityID by ISIN as of the last A7
trading date on or before its snapshot date; ``eurex_derivatives`` lists the Eurex products
on each (snapshot, ISIN) as underlying, joined when needed rather than stored (many Eurex
underlyings are not Xetra-listed, so the join is on the underlying ISIN, not the SecurityID).
"""
from __future__ import annotations

import bisect
import logging
import os
import random
import time
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)

XETR, XEUR = "XETR", "XEUR"
COLUMNS = ["date", "mic", "segment_id", "security_id", "product", "isin", "security_type",
           "description", "family", "underlying_isin"]
_RETRIES = 4
_RETRY_STATUS = {429, 500, 502, 503, 504}
_ISIN_SOURCE = "4"


class RdiClient:
    """The four RDI v2 calls above, over the ``a7`` SDK's authenticated HTTP client, with
    finite retries (backoff on 429 / 5xx) and a thread pool for per-segment calls."""

    def __init__(self, http: Any, workers: int = 8):
        self._http, self.workers = http, workers

    @classmethod
    def from_env(cls, workers: int = 8) -> RdiClient:
        """``A7_TOKEN`` (required), ``A7_BASE_URL`` (default: the SDK's), ``A7_VERIFY_SSL``."""
        from a7 import A7Client

        token = os.environ.get("A7_TOKEN")
        if not token:
            raise RuntimeError("A7_TOKEN is not set (see .env)")
        kw: dict[str, Any] = {"verify_ssl": os.environ.get("A7_VERIFY_SSL", "true").lower()
                              != "false", "timeout": 120.0}
        if os.environ.get("A7_BASE_URL"):
            kw["base_url"] = os.environ["A7_BASE_URL"]
        return cls(A7Client(token=token, **kw)._client, workers)

    def _get(self, path: str) -> Any:
        for attempt in range(_RETRIES):
            response = self._http.get(path)
            if response.status_code == 404:
                return None
            if response.status_code in _RETRY_STATUS and attempt + 1 < _RETRIES:
                time.sleep(2 ** attempt)
                continue
            response.raise_for_status()
            return response.json()
        raise RuntimeError(f"unreachable: {path}")

    def dates(self, mic: str) -> list[int]:
        return sorted(int(d) for d in (self._get(f"/v2/rdi/{mic}/") or []))

    def segments(self, mic: str, day: int) -> list[int]:
        return [int(s) for s in (self._get(f"/v2/rdi/{mic}/{day}/") or [])]

    def securities(self, mic: str, day: int, segment: int) -> list[str]:
        return [str(s) for s in (self._get(f"/v2/rdi/{mic}/{day}/{segment}/") or [])]

    def snapshots(self, mic: str, day: int, segment: int, security: str) -> list[dict]:
        return self._get(f"/v2/rdi/{mic}/{day}/{segment}/{security}") or []

    def map(self, fn, items: Sequence[Any]) -> list[Any]:
        with ThreadPoolExecutor(self.workers) as pool:
            return list(pool.map(fn, items))


# ---------------------------------------------------------------------------
# One date's reference table
# ---------------------------------------------------------------------------

def _alt(instrument: dict, source: str) -> str | None:
    for alt in instrument.get("SecurityAlt") or []:
        if alt.get("SecurityAltIDSource") == source:
            return alt.get("SecurityAltID")
    return None


def _rows_for_segment(client: RdiClient, mic: str, day: int, segment: int) -> list[dict]:
    securities = client.securities(mic, day, segment)
    if not securities:
        return []
    base = {"date": day, "mic": mic, "segment_id": segment}
    if mic == XEUR:          # a product: its snapshot is the same for every contract
        snaps = client.snapshots(mic, day, segment, securities[0])
        product = next((s for s in snaps if s.get("Template") == "ProductSnapshot"), None)
        if product is None:
            return []
        deriv = product.get("DerivativesDescriptorGroup") or {}
        under = product.get("UnderlyingDescriptorGroup") or {}
        return [{**base, "security_id": None, "product": product.get("MarketSegment"),
                 "isin": None, "security_type": None,
                 "description": deriv.get("MarketSegmentDesc"),
                 "family": deriv.get("ParentMktSegmID"),
                 "underlying_isin": (under.get("UnderlyingSecurityID")
                                     if under.get("UnderlyingSecurityIDSource") == _ISIN_SOURCE
                                     else None)}]
    rows = []
    for security in securities:
        snaps = client.snapshots(mic, day, segment, security)
        product = next((s for s in snaps if s.get("Template") == "ProductSnapshot"), {})
        inst = next((s for s in snaps if s.get("Template") == "InstrumentSnapshot"
                     and str(s.get("SecurityID")) == security), None)
        if inst is None:
            continue
        rows.append({**base, "security_id": security, "product": product.get("MarketSegment"),
                     "isin": _alt(inst, _ISIN_SOURCE) or product.get("MarketSegment"),
                     "security_type": inst.get("SecurityType"),
                     "description": inst.get("SecurityDesc"), "family": None,
                     "underlying_isin": None})
    return rows


_TEXT = ["mic", "security_id", "product", "isin", "security_type", "description", "family",
         "underlying_isin"]


def _frame(rows: list[dict]) -> pd.DataFrame:
    """The table's dtypes: int64 date and segment, text columns with None (never NaN)."""
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["date"] = df["date"].astype("int64")
    df["segment_id"] = df["segment_id"].astype("int64")
    df[_TEXT] = df[_TEXT].astype(object).where(df[_TEXT].notna(), None)
    return df


def reference_table(client: RdiClient, mic: str, day: int, known: pd.DataFrame | None = None,
                    *, check: int = 20, seed: int = 0) -> pd.DataFrame:
    """One row per XETR security / XEUR product listed on ``day``. ``known`` (an earlier
    date's table) is reused for segments still listed, after re-fetching ``check`` of them at
    random: one difference and the whole date is scanned afresh."""
    segments = client.segments(mic, day)
    reuse = pd.DataFrame(columns=COLUMNS)
    todo = segments
    if known is not None and not known.empty:
        listed = set(segments)
        reuse = known[known["segment_id"].isin(listed)]
        sample = random.Random(seed + day).sample(sorted(set(reuse["segment_id"])),
                                                  min(check, reuse["segment_id"].nunique()))
        fresh = _frame([r for rows in client.map(
            lambda s: _rows_for_segment(client, mic, day, s), sample) for r in rows])
        cols = ["segment_id", "security_id", "product", "isin", "underlying_isin"]
        old = reuse[reuse["segment_id"].isin(sample)][cols].sort_values(cols[:2]).reset_index(
            drop=True).astype(object).where(lambda x: x.notna(), None)
        new = fresh[cols].sort_values(cols[:2]).reset_index(drop=True).astype(object).where(
            lambda x: x.notna(), None)
        if not old.equals(new):
            log.warning("%s %d: reused segments changed since the earlier table; full scan",
                        mic, day)
            reuse, todo = pd.DataFrame(columns=COLUMNS), segments
        else:
            todo = sorted(listed - set(reuse["segment_id"]))
    new_rows = [r for rows in client.map(lambda s: _rows_for_segment(client, mic, day, s), todo)
                for r in rows]
    out = pd.concat([reuse.assign(date=day), _frame(new_rows)], ignore_index=True)
    log.info("%s %d: %d segments, %d resolved now, %d reused", mic, day, len(segments),
             len(todo), len(segments) - len(todo))
    return _frame(out.to_dict("records")).sort_values(["segment_id", "security_id"],
                                                        na_position="first").reset_index(drop=True)


def reference_tables(client: RdiClient, mic: str, days: Sequence[int], cache_dir: Path, *,
                     reuse: bool | None = None) -> dict[int, pd.DataFrame]:
    """``reference_table`` for each day (ascending), cached in ``cache_dir/{mic}/``. With
    ``reuse`` (default: XETR only, see the module docstring) an uncached day reuses the latest
    earlier table (cached or just built)."""
    reuse = (mic == XETR) if reuse is None else reuse
    folder = Path(cache_dir) / mic
    out: dict[int, pd.DataFrame] = {}
    previous: pd.DataFrame | None = None
    for day in sorted(set(days)):
        path = folder / f"{day}.parquet"
        if path.exists():
            table = _frame(pd.read_parquet(path).to_dict("records"))
        else:
            table = reference_table(client, mic, day, previous if reuse else None)
            folder.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".parquet.tmp")
            table.to_parquet(tmp, index=False)
            tmp.replace(path)
        out[day] = previous = table
    return out


def trading_day_for(snapshot: date, trading_days: Sequence[int]) -> int | None:
    """The last trading day on or before ``snapshot`` (None before the history starts)."""
    key = int(snapshot.strftime("%Y%m%d"))
    i = bisect.bisect_right(trading_days, key)
    return trading_days[i - 1] if i else None


def _days_for(rows: Sequence[dict[str, Any]], trading_days: list[int]) -> dict[date, int | None]:
    return {d: trading_day_for(d, trading_days) for d in {r["snapshot_date"] for r in rows}}


# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------

def backfill_dbga_secid(rows: Sequence[dict[str, Any]], client: RdiClient, cache_dir: Path
                        ) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Fill ``dbga_secid`` (Xetra SecurityID) by ISIN as of each row's snapshot date; never
    overwrites. Counts: filled, not_listed (ISIN absent from Xetra that day), before_history,
    no_isin."""
    out = [dict(r) for r in rows]
    stats: Counter[str] = Counter()
    need = [i for i, r in enumerate(out) if not r.get("dbga_secid")]
    if not need:
        return out, dict(stats)
    day_of = _days_for([out[i] for i in need], client.dates(XETR))
    tables = reference_tables(client, XETR, [d for d in day_of.values() if d], cache_dir)
    lookup = {day: dict(zip(t["isin"], t["security_id"], strict=True))
              for day, t in tables.items()}
    for i in need:
        isin = (out[i].get("isin") or "").strip().upper()
        day = day_of[out[i]["snapshot_date"]]
        if not isin:
            stats["no_isin"] += 1
        elif day is None:
            stats["before_history"] += 1
        elif isin in lookup[day]:
            out[i]["dbga_secid"] = str(lookup[day][isin])
            stats["filled"] += 1
        else:
            stats["not_listed"] += 1
    log.info("dbga_secid: %s", dict(stats))
    return out, dict(stats)


def eurex_derivatives(rows: Sequence[dict[str, Any]], client: RdiClient, cache_dir: Path
                      ) -> pd.DataFrame:
    """The Eurex products whose underlying is a row's ISIN on its snapshot date: one row per
    (snapshot_date, underlying_isin, product), 0..n per universe row. Snapshots before the
    XEUR history have none."""
    rows = [r for r in rows if r.get("isin")]
    cols = ["snapshot_date", "underlying_isin", "product_symbol", "family", "market_segment_id",
            "description", "rdi_date"]
    if not rows:
        return pd.DataFrame(columns=cols)
    day_of = _days_for(rows, client.dates(XEUR))
    tables = reference_tables(client, XEUR, [d for d in day_of.values() if d], cache_dir)
    wanted = pd.DataFrame({"snapshot_date": [r["snapshot_date"] for r in rows],
                           "underlying_isin": [r["isin"].strip().upper() for r in rows]})
    wanted["rdi_date"] = wanted["snapshot_date"].map(day_of)
    wanted = wanted.dropna(subset=["rdi_date"]).drop_duplicates()
    if wanted.empty or not tables:
        return pd.DataFrame(columns=cols)
    products = pd.concat(tables.values(), ignore_index=True).rename(columns={
        "date": "rdi_date", "product": "product_symbol", "segment_id": "market_segment_id"})
    out = wanted.astype({"rdi_date": "int64"}).merge(
        products.dropna(subset=["underlying_isin"]), on=["rdi_date", "underlying_isin"])
    return out[cols].sort_values(["snapshot_date", "underlying_isin", "product_symbol"]) \
        .reset_index(drop=True)


# ---------------------------------------------------------------------------
# Eurex contracts of one product on one day
# ---------------------------------------------------------------------------
#
# A Eurex product (one market segment) holds contracts of a single kind; the kind is read
# from each contract's InstrumentSnapshot, never from the product symbol. The FIX fields
# used (T7 RDI):
#
#     SecurityType   OPT option | FUT future | TRF total return future
#     ExerciseStyle  (options, FIX 1194) "0" European | "1" American
#     SettlMethod    (futures, FIX 1193) "C" cash | "P" physical delivery
#     PutOrCall      (options, FIX 201)  "0" put | "1" call
#
# Symbols follow Eurex conventions but are only examples, not rules. For Deutsche Boerse AG
# (underlying DE0005810055) on 2026-09-28:
#
#     DB1   segment 361      OPT, ExerciseStyle 1   American options      (665 contracts)
#     DB1E  segment 46225    OPT, ExerciseStyle 0   European options      (631 contracts)
#     DB1H  segment 363      FUT, SettlMethod C     cash-settled futures  (15, monthly expiries)
#     DB1P  segment 567612   FUT, SettlMethod P     physically settled futures (16, DAILY
#                                                   expiries: MaturityFrequencyUnit "D")
#     TDB1  segment 134594   TRF, SettlMethod C     total return futures  (8)
#
# i.e. the "E" suffix marks the European options and the plain symbol the American ones.

OPTION_EUROPEAN = "option_european"
OPTION_AMERICAN = "option_american"
FUTURE_CASH = "future_cash"
FUTURE_PHYSICAL = "future_physical"
TOTAL_RETURN_FUTURE = "total_return_future"
OTHER = "other"
CONTRACT_KINDS = (OPTION_EUROPEAN, OPTION_AMERICAN, FUTURE_CASH, FUTURE_PHYSICAL,
                  TOTAL_RETURN_FUTURE, OTHER)
CONTRACT_COLUMNS = ["date", "segment_id", "security_id", "product", "kind", "security_type",
                    "expiry", "contract_month", "maturity_frequency", "strike", "put_call",
                    "exercise_style", "settl_method", "multiplier", "display_name", "is_primary",
                    "description"]
_SIMPLE_INSTRUMENT = "1"          # ProductComplex: 1 simple; other values are strategies


def contract_kind(instrument: dict) -> str:
    """The kind of one contract from its InstrumentSnapshot (see the table above)."""
    kind = instrument.get("SecurityType")
    detail = ((instrument.get("DerivativesDescriptorGroup") or {})
              .get("SimpleInstrumentDescriptorGroup") or {})
    if kind == "OPT":
        style = detail.get("ExerciseStyle")
        return {"0": OPTION_EUROPEAN, "1": OPTION_AMERICAN}.get(style, OTHER)
    if kind == "FUT":
        return {"C": FUTURE_CASH, "P": FUTURE_PHYSICAL}.get(detail.get("SettlMethod"), OTHER)
    if kind == "TRF":
        return TOTAL_RETURN_FUTURE
    return OTHER


def _contract_row(day: int, segment: int, snaps: list[dict], security: str) -> dict | None:
    product = next((s for s in snaps if s.get("Template") == "ProductSnapshot"), {})
    inst = next((s for s in snaps if s.get("Template") == "InstrumentSnapshot"
                 and str(s.get("SecurityID")) == security), None)
    if inst is None or inst.get("ProductComplex", _SIMPLE_INSTRUMENT) != _SIMPLE_INSTRUMENT:
        return None
    deriv = inst.get("DerivativesDescriptorGroup") or {}
    detail = deriv.get("SimpleInstrumentDescriptorGroup") or {}
    put_call = detail.get("PutOrCall")
    return {
        "date": day, "segment_id": segment, "security_id": security,
        "product": product.get("MarketSegment"), "kind": contract_kind(inst),
        "security_type": inst.get("SecurityType"), "expiry": detail.get("ContractDate"),
        "contract_month": detail.get("ContractMonthYear"),
        "maturity_frequency": detail.get("MaturityFrequencyUnit"),
        "strike": detail.get("StrikePrice"),
        "put_call": {"0": "put", "1": "call"}.get(put_call) if put_call is not None else None,
        "exercise_style": detail.get("ExerciseStyle"), "settl_method": detail.get("SettlMethod"),
        "multiplier": detail.get("ContractMultiplier"), "display_name": deriv.get("DisplayName"),
        "is_primary": deriv.get("IsPrimary") == "Y", "description": inst.get("SecurityDesc"),
    }


def eurex_contracts(client: RdiClient, day: int, segment_id: int,
                    kinds: Sequence[str] | None = None) -> pd.DataFrame:
    """Every simple contract live on ``day`` in one Eurex product (``segment_id``, e.g. from
    ``eurex_derivatives``): one row per contract with ``CONTRACT_COLUMNS``. ``kinds`` keeps
    only those ``CONTRACT_KINDS``; a product has one kind, so its first contract decides and
    a product of another kind costs a single detail call. One call per contract otherwise
    (~1,300 for Deutsche Boerse AG's five products on one day, ~25 s with 16 workers): a
    contract's expiry and strike never change, so cache what you fetch."""
    unknown = set(kinds or ()) - set(CONTRACT_KINDS)
    if unknown:
        raise ValueError(f"unknown kind(s) {sorted(unknown)}; choose from {CONTRACT_KINDS}")
    securities = client.securities(XEUR, day, segment_id)
    if not securities:
        return pd.DataFrame(columns=CONTRACT_COLUMNS)

    def row(security: str) -> dict | None:
        return _contract_row(day, segment_id, client.snapshots(XEUR, day, segment_id, security),
                             security)

    first = row(securities[0])
    if kinds and (first is None or first["kind"] not in kinds):
        return pd.DataFrame(columns=CONTRACT_COLUMNS)
    rows = [first, *client.map(row, securities[1:])]
    out = pd.DataFrame([r for r in rows if r is not None], columns=CONTRACT_COLUMNS)
    if kinds:
        out = out[out["kind"].isin(kinds)]
    return out.sort_values(["kind", "expiry", "strike", "put_call"], na_position="first") \
        .reset_index(drop=True)
