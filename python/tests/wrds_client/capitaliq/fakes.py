"""An in-memory WRDS for ``ciq_keydev``: answers the job's two monthly queries and lookups
from frames shaped like WRDS returns them (numerics as floats, like ``raw_sql``)."""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd

from wrds_client.capitaliq.keydev import DIM_TABLES, EVENTS_SQL, LINKS_SQL, KeyDevSource

ROLES = {1: "Target", 2: "Buyer", 3: "Seller"}
EVENT_TYPES = {16: "Executive Changes", 28: "Announcements of Earnings", 80: "M&A Rumors"}


def make_events(start: date, end: date, *, every: int = 5, per_day: int = 2,
                first_id: int = 1000) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(links, events): every ``every`` days, ``per_day`` events; every third event has two
    companies (target + buyer), the others one."""
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
                "mostimportantdateutc": ts + timedelta(hours=5)})
            companies = [(kid % 7 + 1, 1)] + ([(kid % 5 + 100, 2)] if kid % 3 == 0 else [])
            for cid, role in companies:
                etype = (16, 28, 80)[kid % 3]
                links.append({
                    "keydevid": float(kid), "companyid": float(cid),
                    "companyname": f"Company {cid}", "gvkey": f"{cid:06d}",
                    "keydeveventtypeid": float(etype), "eventtype": EVENT_TYPES[etype],
                    "keydevtoobjectroletypeid": float(role), "objectroletype": ROLES[role],
                    "sourcetypename": "Press Release", "announcedate": d})
        d += timedelta(days=every)
    return pd.DataFrame(links), pd.DataFrame(events)


class FakeWRDS:
    """``query(sql, params)`` for the ``KeyDevSource``; ``calls`` counts each query kind."""

    def __init__(self, links: pd.DataFrame, events: pd.DataFrame) -> None:
        self.links, self.events = links, events
        self.calls = {"links": 0, "events": 0, "dim": 0}

    def _month_links(self, params: dict[str, Any]) -> pd.DataFrame:
        start, stop = date.fromisoformat(params["start"]), date.fromisoformat(params["stop"])
        d = self.links["announcedate"]
        return self.links[(d >= start) & (d < stop)]

    def query(self, sql: str, params: dict[str, Any]) -> pd.DataFrame:
        if sql == LINKS_SQL:
            self.calls["links"] += 1
            return self._month_links(params).reset_index(drop=True)
        if sql == EVENTS_SQL:
            self.calls["events"] += 1
            ids = set(self._month_links(params)["keydevid"])
            return self.events[self.events["keydevid"].isin(ids)].reset_index(drop=True)
        table = sql.rsplit(".", 1)[-1]
        assert table in DIM_TABLES, sql
        self.calls["dim"] += 1
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
