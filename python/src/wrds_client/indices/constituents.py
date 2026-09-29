"""Index constituents from WRDS, as the snapshot files ``universe.index_universe`` loads.

Generic by source and index: nothing here names an index. ``search_indices`` finds an
index's id in each source's catalogue, ``build_constituents`` turns its membership spells
into one row per (snapshot date, member) with identifiers, and ``write_constituents``
writes ``{INDEX}_constituents-{first}_to_{last}.parquet`` for
``universe.index_universe.index_universe_rows``.

Sources (tables confirmed live with ``information_schema``, not inferred):

    compustat   catalogue comp.idx_index (North America) / comp.g_idx_index (global):
                gvkeyx, conm, idxcstflg ('Y' = constituents available)
                spells    comp.idxcst_his / comp.g_idxcst_his: gvkey, iid, gvkeyx, from, thru
                ids       comp.security / comp.g_security (isin, cusip, sedol, tic) and
                          comp.company / comp.g_company (conm): CURRENT values, not history
    crsp        catalogue crsp.indseriesinfohdr: indno, indnm
                spells    crsp.stkindmembership_ind: permno, indno, mbrstartdt, mbrenddt
                ids       crsp.stocknames: ncusip, ticker, comnam per [namedt, nameenddt],
                          matched point in time; ISIN is left empty (the CUSIP country prefix
                          cannot be told from the CUSIP) for the CapitalIQ enrichment to fill

Some indices are listed with index levels but no constituents in any source (Compustat
flags every MSCI index ``idxcstflg = 'N'``; that flag is a property of the data, not of the
subscription). ``IndexNotAvailable`` says so instead of returning an empty universe; MSCI
membership has to come from MSCI or LSEG / Datastream. A source the account is not licensed
for raises ``IndexNotAvailable`` too, and is reported (not fatal) by ``search_indices``.

A spell set with no ended spell is almost always a current-members-only extract (e.g. the
S&P 500 in comp.idxcst_his without the full North America subscription): it is logged as a
survivorship warning, because every earlier snapshot would then list today's members.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from wrds_client.client import WRDSClient

from wrds_client.linking._shared import match_as_of_windows, to_date

log = logging.getLogger(__name__)

SOURCES = ("compustat", "crsp")
#: Compustat catalogue and spell tables, per region
_COMPUSTAT = (("north_america", "comp.idx_index", "comp.idxcst_his"),
              ("global", "comp.g_idx_index", "comp.g_idxcst_his"))
_SECURITY_TABLES = (("comp.security", "comp.company"), ("comp.g_security", "comp.g_company"))
#: CRSP stores an open membership as a far-future end date
_OPEN_END = date(2099, 1, 1)
_INDEX_NAME = re.compile(r"^[A-Za-z0-9]+(_[A-Za-z0-9]+)*$")
#: loader columns first, then the source's own keys
COLUMNS = ("snapshot_date", "name", "ticker", "isin", "cusip", "sedol", "gvkey")


class IndexNotAvailable(RuntimeError):
    """The index has no constituents this account can read in this source."""


def _is_permission_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "permission denied" in text or "insufficientprivilege" in text


def _sql(client: WRDSClient, sql: str, params: dict[str, Any] | None = None,
         what: str = "") -> pd.DataFrame:
    try:
        return client.raw_sql(sql, params=params)
    except Exception as exc:  # noqa: BLE001 - re-raised unless it is a licence refusal
        if _is_permission_error(exc):
            raise IndexNotAvailable(f"{what}: this WRDS account is not licensed for it") \
                from exc
        raise


def _words(text: str, column: str) -> tuple[str, dict[str, str]]:
    """``column`` contains every word of ``text`` (case-insensitive), in any order."""
    words = text.split()
    if not words:
        raise ValueError("search text is empty")
    clause = " AND ".join(f"{column} ILIKE %(w{i})s" for i in range(len(words)))
    return clause, {f"w{i}": f"%{w}%" for i, w in enumerate(words)}


# ---------------------------------------------------------------------------
# Catalogue
# ---------------------------------------------------------------------------

def search_indices(client: WRDSClient, text: str) -> pd.DataFrame:
    """Every index whose name contains every word of ``text`` (case-insensitive, any order,
    so "MSCI World" finds "MSCI - World Index"), across sources:
    source, index_id, name, has_constituents, table. A source this account cannot read
    appears as one row with ``has_constituents`` null and the refusal in ``name``."""
    rows: list[dict[str, Any]] = []
    for region, catalogue, spells in _COMPUSTAT:
        try:
            clause, params = _words(text, "conm")
            df = _sql(client, f"SELECT gvkeyx, conm, idxcstflg FROM {catalogue} "
                              f"WHERE {clause} ORDER BY conm", params, catalogue)
        except IndexNotAvailable as exc:
            rows.append({"source": "compustat", "index_id": None, "name": str(exc),
                         "has_constituents": None, "table": spells})
            continue
        rows += [{"source": "compustat", "index_id": r.gvkeyx, "name": r.conm,
                  "has_constituents": r.idxcstflg == "Y", "table": spells}
                 for r in df.itertuples()]
    try:
        clause, params = _words(text, "h.indnm")
        df = _sql(client, "SELECT h.indno, h.indnm, EXISTS (SELECT 1 FROM "
                          "crsp.stkindmembership_ind m WHERE m.indno = h.indno) AS has "
                          f"FROM crsp.indseriesinfohdr h WHERE {clause} "
                          "ORDER BY h.indnm", params, "crsp.stkindmembership_ind")
        rows += [{"source": "crsp", "index_id": str(int(r.indno)), "name": r.indnm,
                  "has_constituents": bool(r.has), "table": "crsp.stkindmembership_ind"}
                 for r in df.itertuples()]
    except IndexNotAvailable as exc:
        rows.append({"source": "crsp", "index_id": None, "name": str(exc),
                     "has_constituents": None, "table": "crsp.stkindmembership_ind"})
    return pd.DataFrame(rows, columns=["source", "index_id", "name", "has_constituents",
                                       "table"])


# ---------------------------------------------------------------------------
# Membership spells
# ---------------------------------------------------------------------------

def _compustat_index(client: WRDSClient, gvkeyx: str) -> tuple[str, str, str]:
    """(name, idxcstflg, spell table) of a gvkeyx, from whichever catalogue lists it."""
    for _, catalogue, spells in _COMPUSTAT:
        df = _sql(client, f"SELECT conm, idxcstflg FROM {catalogue} WHERE gvkeyx = %(id)s",
                  {"id": gvkeyx}, catalogue)
        if not df.empty:
            return str(df.iloc[0]["conm"]), str(df.iloc[0]["idxcstflg"]), spells
    raise IndexNotAvailable(f"compustat: no index with gvkeyx {gvkeyx!r} "
                            "(search_indices lists them)")


def membership_spells(client: WRDSClient, source: str, index_id: str) -> pd.DataFrame:
    """Membership spells of one index: the security key columns (compustat: gvkey, iid;
    crsp: permno), ``start`` and ``end`` (dates, ``end`` None = still a member)."""
    if source == "compustat":
        name, flag, table = _compustat_index(client, index_id)
        if flag != "Y":
            raise IndexNotAvailable(
                f"compustat lists {name} ({index_id}) with index levels but no constituents "
                f"(idxcstflg={flag!r}); no WRDS licence changes that. MSCI and other such "
                "providers' memberships come from the provider itself or LSEG / Datastream")
        df = _sql(client, f'SELECT gvkey, iid, "from" AS start, thru AS "end" FROM {table} '
                          "WHERE gvkeyx = %(id)s", {"id": index_id}, table)
        keys = ["gvkey", "iid"]
    elif source == "crsp":
        df = _sql(client, 'SELECT permno, mbrstartdt AS start, mbrenddt AS "end" '
                          "FROM crsp.stkindmembership_ind WHERE indno = %(id)s",
                  {"id": int(index_id)}, "crsp.stkindmembership_ind")
        df["permno"] = df["permno"].astype("int64")
        keys = ["permno"]
        name = f"CRSP index {index_id}"
    else:
        raise ValueError(f"source must be one of {SOURCES}, got {source!r}")
    if df.empty:
        raise IndexNotAvailable(f"{source} {index_id}: no membership spells")
    df["start"] = df["start"].map(to_date)
    df["end"] = df["end"].map(to_date).map(lambda d: None if d is None or d >= _OPEN_END else d)
    if df["end"].isna().all():
        log.warning("%s %s (%s): none of its %d spells has ended: this looks like a "
                    "current-members-only extract, so every past snapshot would list today's "
                    "members (survivorship)", source, index_id, name, len(df))
    return df[keys + ["start", "end"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Snapshots and identifiers
# ---------------------------------------------------------------------------

def snapshot_dates(start: date, end: date) -> list[date]:
    """The first business day of every month in [start, end]."""
    return [d.date() for d in pd.bdate_range(start, end, freq="BMS")]


def snapshots(spells: pd.DataFrame, dates: Sequence[date]) -> pd.DataFrame:
    """Members on each date: start <= date <= end (end None = open), one row per
    (snapshot_date, security)."""
    keys = [c for c in spells.columns if c not in ("start", "end")]
    grid = spells.merge(pd.DataFrame({"snapshot_date": list(dates)}), how="cross")
    live = (grid["start"] <= grid["snapshot_date"]) & (
        grid["end"].isna() | (grid["snapshot_date"] <= grid["end"]))
    out = grid.loc[live, ["snapshot_date", *keys]].drop_duplicates()
    return out.sort_values(["snapshot_date", *keys]).reset_index(drop=True)


def cusip_check_digit(cusip8: str) -> str:
    """The ninth (check) character of a CUSIP from its first eight."""
    total = 0
    for i, ch in enumerate(cusip8.upper()):
        if ch.isdigit():
            v = int(ch)
        elif ch.isalpha():
            v = ord(ch) - ord("A") + 10
        else:
            v = {"*": 36, "@": 37, "#": 38}[ch]
        if i % 2:
            v *= 2
        total += v // 10 + v % 10
    return str((10 - total % 10) % 10)


def _compustat_identifiers(client: WRDSClient, members: pd.DataFrame) -> pd.DataFrame:
    gvkeys = tuple(sorted(members["gvkey"].unique()))
    sec, com = [], []
    for sec_table, com_table in _SECURITY_TABLES:
        sec.append(_sql(client, f"SELECT gvkey, iid, isin, cusip, sedol, tic FROM {sec_table} "
                                "WHERE gvkey IN %(g)s", {"g": gvkeys}, sec_table))
        com.append(_sql(client, f"SELECT gvkey, conm FROM {com_table} WHERE gvkey IN %(g)s",
                        {"g": gvkeys}, com_table))
    sec_df = pd.concat(sec).drop_duplicates(["gvkey", "iid"])
    com_df = pd.concat(com).drop_duplicates("gvkey")
    out = (members.merge(sec_df, on=["gvkey", "iid"], how="left")
           .merge(com_df, on="gvkey", how="left")
           .rename(columns={"conm": "name", "tic": "ticker"}))
    return out


def _crsp_identifiers(client: WRDSClient, members: pd.DataFrame) -> pd.DataFrame:
    permnos = tuple(int(p) for p in sorted(members["permno"].unique()))
    names = _sql(client, "SELECT permno, namedt, nameenddt, ncusip, ticker, comnam "
                         "FROM crsp.stocknames WHERE permno IN %(p)s", {"p": permnos},
                 "crsp.stocknames")
    names["permno"] = names["permno"].astype("int64")
    names["namedt"] = names["namedt"].map(to_date)
    names["nameenddt"] = names["nameenddt"].map(to_date)
    matched = match_as_of_windows(names, members, key_col="permno", start_col="namedt",
                                  end_col="nameenddt")
    out = members.copy()
    ncusip = matched["ncusip"].reindex(out.index)
    out["cusip"] = [c + cusip_check_digit(c) if isinstance(c, str) and len(c) == 8 else None
                    for c in ncusip]
    out["ticker"] = matched["ticker"].reindex(out.index)
    out["name"] = matched["comnam"].reindex(out.index)
    out["isin"] = None
    out["sedol"] = None
    out["gvkey"] = None
    return out


def build_constituents(client: WRDSClient, source: str, index_id: str, start: date,
                       end: date, dates: Sequence[date] | None = None) -> pd.DataFrame:
    """One row per (snapshot date, member) with ``COLUMNS`` then the source keys (iid /
    permno). ``dates`` defaults to ``snapshot_dates(start, end)``."""
    spells = membership_spells(client, source, index_id)
    members = snapshots(spells, dates if dates is not None else snapshot_dates(start, end))
    if members.empty:
        raise IndexNotAvailable(f"{source} {index_id}: no members between {start} and {end}")
    ids = (_compustat_identifiers if source == "compustat" else _crsp_identifiers)(
        client, members)
    extra = [c for c in ("iid", "permno") if c in ids.columns]
    out = ids[[*COLUMNS, *extra]].reset_index(drop=True)
    for c in ("name", "ticker", "isin", "cusip", "sedol", "gvkey"):
        out[c] = out[c].astype("string")
    log.info("%s %s: %d rows, %d snapshots, %.1f%% with an ISIN, %.1f%% with a CUSIP",
             source, index_id, len(out), out["snapshot_date"].nunique(),
             100 * out["isin"].notna().mean(), 100 * out["cusip"].notna().mean())
    return out


def write_constituents(frame: pd.DataFrame, out_dir: Path, index: str) -> Path:
    """``{index}_constituents-{first}_to_{last}.parquet`` (YYYYMMDD) in ``out_dir``;
    never overwrites."""
    if not _INDEX_NAME.match(index):
        raise ValueError(f"index name must be letters, digits and single underscores: {index!r}")
    first, last = min(frame["snapshot_date"]), max(frame["snapshot_date"])
    path = Path(out_dir) / (f"{index}_constituents-{first:%Y%m%d}_to_{last:%Y%m%d}.parquet")
    if path.exists():
        raise FileExistsError(f"{path} exists; move or delete it first")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path
