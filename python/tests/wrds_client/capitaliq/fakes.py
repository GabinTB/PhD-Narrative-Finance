"""An in-memory WRDS for ``ciq_keydev``: answers the job's queries from versioned frames
shaped like WRDS returns them (numerics as floats, like ``raw_sql``)."""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd

from wrds_client.capitaliq.keydev import (
    CHANGED_IDS_SQL,
    DIM_TABLES,
    EVENTS_BY_IDS_SQL,
    EVENTS_SQL,
    LINKS_BY_IDS_SQL,
    LINKS_SQL,
    KeyDevSource,
)

ROLES = {1: "Target", 2: "Buyer", 3: "Seller"}
EVENT_TYPES = {16: "Executive Changes", 28: "Announcements of Earnings", 80: "M&A Rumors"}
HISTORY_START = datetime(2018, 4, 14, 12, 8)      # WRDS's first speffectivedate
LINK_GROUP = ["keydevid", "companyid", "keydevtoobjectroletypeid", "keydeveventtypeid"]


def make_events(start: date, end: date, *, every: int = 5, per_day: int = 2,
                first_id: int = 1000) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(links, events), one current version each (valid since ``HISTORY_START``): every
    ``every`` days, ``per_day`` events; every third event has two companies."""
    links, events = [], []
    kid, d = first_id, start
    while d <= end:
        for j in range(per_day):
            kid += 1
            ts = datetime(d.year, d.month, d.day, 9 + j, 30)
            events.append({
                "keydevid": float(kid), "headline": f"Company {kid % 7} news {kid}",
                "situation": f"Situation paragraph of event {kid}.",
                "announceddate": ts, "announceddateutc": ts + timedelta(hours=5),
                "announceddatetimezoneid": 3.0, "entereddate": ts + timedelta(days=1),
                "entereddateutc": ts + timedelta(days=1, hours=5),
                "lastmodifieddate": ts + timedelta(days=2),
                "lastmodifieddateutc": ts + timedelta(days=2, hours=5),
                "mostimportantdateutc": ts + timedelta(hours=5),
                "speffectivedate": HISTORY_START, "sptodate": pd.NaT})
            companies = [(kid % 7 + 1, 1)] + ([(kid % 5 + 100, 2)] if kid % 3 == 0 else [])
            for cid, role in companies:
                etype = (16, 28, 80)[kid % 3]
                links.append({
                    "keydevid": float(kid), "companyid": float(cid),
                    "companyname": f"Company {cid}", "gvkey": f"{cid:06d}",
                    "keydeveventtypeid": float(etype), "eventtype": EVENT_TYPES[etype],
                    "keydevtoobjectroletypeid": float(role), "objectroletype": ROLES[role],
                    "sourcetypename": "Press Release", "announcedate": d,
                    "speffectivedate": HISTORY_START, "sptodate": pd.NaT})
        d += timedelta(days=every)
    return pd.DataFrame(links), pd.DataFrame(events)


class FakeWRDS:
    """``query(sql, params)`` for the ``KeyDevSource``; ``calls`` counts each query kind."""

    def __init__(self, links: pd.DataFrame, events: pd.DataFrame) -> None:
        self.links, self.events = links, events
        self.calls: dict[str, int] = {}

    def revise(self, keydevid: int, at: datetime, **changes: Any) -> None:
        """CIQ edits an event: its current versions (event and links) close at ``at``, new
        ones open at ``at`` with ``changes`` (``announceddate`` also re-dates the links)."""
        ev_cur = (self.events["keydevid"] == keydevid) & self.events["sptodate"].isna()
        lk_cur = (self.links["keydevid"] == keydevid) & self.links["sptodate"].isna()
        new_ev = self.events[ev_cur].assign(speffectivedate=at, sptodate=pd.NaT, **changes)
        new_lk = self.links[lk_cur].assign(speffectivedate=at, sptodate=pd.NaT)
        if "announceddate" in changes:
            new_lk = new_lk.assign(announcedate=changes["announceddate"].date())
        self.events.loc[ev_cur, "sptodate"] = at - timedelta(seconds=1)
        self.links.loc[lk_cur, "sptodate"] = at - timedelta(seconds=1)
        self.events = pd.concat([self.events, new_ev], ignore_index=True)
        self.links = pd.concat([self.links, new_lk], ignore_index=True)

    def add(self, links: pd.DataFrame, events: pd.DataFrame, at: datetime) -> None:
        """New events entering CIQ at ``at``."""
        self.links = pd.concat([self.links, links.assign(speffectivedate=at)],
                               ignore_index=True)
        self.events = pd.concat([self.events, events.assign(speffectivedate=at)],
                                ignore_index=True)

    # -- the queries ---------------------------------------------------------

    def _count(self, kind: str) -> None:
        self.calls[kind] = self.calls.get(kind, 0) + 1

    def _with_first(self, frame: pd.DataFrame, by: list[str]) -> pd.DataFrame:
        out = frame.copy()
        out["first_speffectivedate"] = out.groupby(by)["speffectivedate"].transform("min")
        return out.reset_index(drop=True)

    def _month_ids(self, params: dict[str, Any]) -> set[float]:
        start, stop = date.fromisoformat(params["start"]), date.fromisoformat(params["stop"])
        d = self.links["announcedate"]
        return set(self.links.loc[(d >= start) & (d < stop), "keydevid"])

    def query(self, sql: str, params: dict[str, Any]) -> pd.DataFrame:
        if sql == LINKS_SQL:
            self._count("links")
            start, stop = date.fromisoformat(params["start"]), date.fromisoformat(params["stop"])
            x = self._with_first(self.links[self.links["keydevid"].isin(self._month_ids(params))],
                                 LINK_GROUP)
            return x[(x["announcedate"] >= start) & (x["announcedate"] < stop)]
        if sql == EVENTS_SQL:
            self._count("events")
            start = datetime.fromisoformat(params["start"])
            stop = datetime.fromisoformat(params["stop"])
            x = self._with_first(
                self.events[self.events["keydevid"].isin(self._month_ids(params))], ["keydevid"])
            return x[(x["announceddate"] >= start) & (x["announceddate"] < stop)]
        if sql == CHANGED_IDS_SQL:
            self._count("changed")
            since = datetime.fromisoformat(params["since"])
            ids = self.links.loc[self.links["speffectivedate"] > since, "keydevid"].unique()
            return pd.DataFrame({"keydevid": ids})
        if sql in (LINKS_BY_IDS_SQL, EVENTS_BY_IDS_SQL):
            self._count("by_ids")
            ids = {float(i) for i in params["ids"]}
            if sql == LINKS_BY_IDS_SQL:
                return self._with_first(self.links[self.links["keydevid"].isin(ids)], LINK_GROUP)
            return self._with_first(self.events[self.events["keydevid"].isin(ids)], ["keydevid"])
        table = sql.rsplit(".", 1)[-1]
        assert table in DIM_TABLES, sql
        self._count("dim")
        if table == "ciqkeydevcategorytype":
            return pd.DataFrame({"keydeveventtypeid": list(EVENT_TYPES),
                                 "keydevcategoryid": [1.0, 2.0, 3.0],
                                 "keydevcategoryname": ["People", "Results", "M&A"],
                                 "keydeveventtypename": list(EVENT_TYPES.values())})
        if table == "ciqkeydevobjectroletype":
            return pd.DataFrame({"keydevtoobjectroletypeid": list(ROLES),
                                 "keydevtoobjectroletypename": list(ROLES.values())})
        return pd.DataFrame({"announceddatetimezoneid": [3],
                             "announceddatetimezonename": ["Eastern Standard Time"]})

    def source(self) -> KeyDevSource:
        return KeyDevSource(self.query)
