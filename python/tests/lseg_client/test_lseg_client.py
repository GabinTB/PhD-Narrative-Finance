"""lseg_client with a fake ``lseg.data`` module: sessions and fallback, chunking, paging,
column shapes, the ISIN-keyed enrichment. No network."""
from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

import lseg_client.client as client_mod
from lseg_client import LsegClient, LsegConfig, SessionUnavailable, backfill_lseg_ids, open_session


class FakeSession:
    def __init__(self, opens: bool, record: list, kind: str, user: str | None = None):
        self.opens, self.record, self.kind, self.user = opens, record, kind, user
        self.open_state = "OpenState.Closed"

    def open(self):
        self.record.append((self.kind, self.user))
        if self.opens:
            self.open_state = "OpenState.Opened"

    def close(self):
        self.open_state = "OpenState.Closed"


def fake_ld(*, desktop: bool, platform_users: tuple[str, ...] = ()) -> SimpleNamespace:
    record: list = []
    config: dict = {}

    class Desktop:
        def __init__(self, app_key): self.app_key = app_key
        def get_session(self): return FakeSession(desktop, record, "desktop")

    class Platform:
        def __init__(self, app_key, grant, signon_control):
            assert signon_control is False           # never kicks the user's other session
            self.user = grant.username
        def get_session(self): return FakeSession(self.user in platform_users, record,
                                                  "platform", self.user)

    return SimpleNamespace(
        record=record, config=config, default=[],
        get_config=lambda: config,
        session=SimpleNamespace(
            desktop=SimpleNamespace(Definition=Desktop),
            platform=SimpleNamespace(Definition=Platform,
                                     GrantPassword=lambda username, password:
                                     SimpleNamespace(username=username)),
            set_default=lambda s: None))


CFG = LsegConfig(app_key="k" * 40, desktop_url="http://127.0.0.1:9006",
                 usernames=("a@x", "b@x"), password="p")


def test_desktop_first_with_the_tunnel_url():
    ld = fake_ld(desktop=True, platform_users=("a@x",))
    session, kind = open_session(CFG, ld=ld)
    assert kind == "desktop" and ld.record == [("desktop", None)]
    assert ld.config["sessions.desktop.workspace.base-url"] == "http://127.0.0.1:9006"


def test_platform_fallback_tries_each_user():
    ld = fake_ld(desktop=False, platform_users=("b@x",))
    _, kind = open_session(CFG, ld=ld)
    assert kind == "platform"
    assert ld.record == [("desktop", None), ("platform", "a@x"), ("platform", "b@x")]


def test_no_session_reports_every_attempt_without_secrets():
    ld = fake_ld(desktop=False)
    with pytest.raises(SessionUnavailable) as err:
        open_session(CFG, ld=ld)
    msg = str(err.value)
    assert "desktop: not opened" in msg and "quota" in msg
    assert "a@x" not in msg and "k" * 40 not in msg
    with pytest.raises(SessionUnavailable, match="no app key"):
        open_session(LsegConfig(app_key=None), ld=ld)


def test_config_from_env(monkeypatch):
    for k in ("LSEG_APP_KEY", "LSEG_USERNAME", "LSEG_DESKTOP_URL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("LSEG_CLIENT_SECRET", "s" * 40)
    monkeypatch.setenv("LSEG_USERNAME_1", "a@x")
    monkeypatch.setenv("LSEG_USERNAME_2", "a@x")
    monkeypatch.setenv("LSEG_PASSWORD", "p")
    c = LsegConfig.from_env()
    assert c.app_key == "s" * 40 and c.usernames == ("a@x",) and c.desktop_url is None
    assert "password" not in repr(c)                          # never shown in logs/tracebacks


# ---------------------------------------------------------------------------
# data calls
# ---------------------------------------------------------------------------

class DataLd:
    """get_data / get_history / news / discovery fakes, recording calls."""

    def __init__(self):
        self.calls: list[tuple[str, Any]] = []
        self.news = SimpleNamespace(get_headlines=self._headlines, get_story=lambda s: "body")
        self.discovery = SimpleNamespace(convert_symbols=self._convert)
        self.stories = pd.DataFrame({
            "versionCreated": pd.date_range("2020-01-01", periods=250, freq="h"),
            "headline": [f"h{i}" for i in range(250)],
            "storyId": [f"s{i}" for i in range(250)], "sourceCode": "NS:RTRS"})

    def get_data(self, universe, fields, parameters=None):
        self.calls.append(("get_data", (list(universe), list(fields), parameters)))
        rows = [[u, *[f"{u}|{f}" for f in fields]] for u in universe]
        return pd.DataFrame(rows, columns=["Instrument", *[f"T{f}" for f in fields]])

    def get_history(self, universe, fields=None, interval=None, start=None, end=None):
        self.calls.append(("get_history", (list(universe), start, end)))
        idx = pd.DatetimeIndex(pd.bdate_range(start, end), name="Date")
        cols = pd.MultiIndex.from_product([universe, ["CLOSE"]])
        return pd.DataFrame(1.0, index=idx, columns=cols)

    def _headlines(self, query, count, start, end):
        self.calls.append(("headlines", (count, end)))
        s = self.stories
        live = s.versionCreated.between(pd.Timestamp(start), pd.Timestamp(end))
        page = s[live]
        return page.sort_values("versionCreated", ascending=False).head(count).set_index(
            "versionCreated")

    def _convert(self, symbols, from_symbol_type):
        known = {"DE0005810055": ("DB1Gn.DE", 4298007872.0),
                 "US0378331005": ("AAPL.O", 4295905573.0)}
        rows = {s: {"RIC": known[s][0], "IssuerOAPermID": known[s][1]} for s in symbols
                if s in known}
        return pd.DataFrame.from_dict(rows, orient="index")


def test_members_use_the_chain_and_identifiers_are_chunked(monkeypatch):
    ld = DataLd()
    c = LsegClient(ld)
    m = c.index_members(".STOXX")
    assert ld.calls[0][1][0] == ["0#.STOXX"] and list(m.columns) == ["ric", "name", "isin"]
    monkeypatch.setattr(client_mod, "CHUNK", 2)
    ids = c.identifiers(["A", "B", "C", "A", None])
    assert [call[1][0] for call in ld.calls[1:]] == [["A", "B"], ["C"]]
    assert list(ids.columns) == ["ric", *client_mod.IDENTIFIER_FIELDS.values()]


def test_fundamentals_carry_the_period_date_and_parameters():
    ld = DataLd()
    df = LsegClient(ld).fundamentals(["X"], ["TR.Revenue"], "2018-01-01", "2020-12-31", "FQ")
    _, (universe, fields, params) = ld.calls[0]
    assert fields == ["TR.Revenue.date", "TR.Revenue"]
    assert params == {"SDate": "2018-01-01", "EDate": "2020-12-31", "Frq": "FQ"}
    assert list(df.columns) == ["ric", "date", "TR.Revenue"]


def test_prices_are_inclusive_of_start_and_long():
    ld = DataLd()
    df = LsegClient(ld).prices(["A", "B"], "2024-01-02", "2024-01-05")
    assert ld.calls[0][1][1] == "2024-01-01"                     # asked a day earlier
    assert sorted(df["date"].dt.strftime("%Y-%m-%d").unique()) == [
        "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    assert set(df["ric"]) == {"A", "B"} and "CLOSE" in df.columns


def test_headlines_page_backwards_without_duplicates():
    ld = DataLd()
    h = LsegClient(ld).headlines("R:X", "2020-01-01", "2020-12-31")
    assert len(h) == 250 and h["story_id"].is_unique
    assert [c[1][0] for c in ld.calls] == [100, 100, 100]
    assert list(h.columns) == ["version_created", "headline", "story_id", "source_code"]
    assert len(LsegClient(DataLd()).headlines("R:X", "2020-01-01", "2020-12-31",
                                              max_items=120)) == 120


def test_throttling_is_retried(monkeypatch):
    monkeypatch.setattr(client_mod.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("Error code 429 | Too many requests")
        return "ok"

    assert LsegClient(None)._call(flaky) == "ok" and calls["n"] == 3
    with pytest.raises(RuntimeError, match="bad field"):
        LsegClient(None)._call(lambda: (_ for _ in ()).throw(RuntimeError("bad field")))


def test_enrichment_fills_only_missing_and_counts_unresolved():
    rows = [{"snapshot_date": date(2020, 1, 2), "isin": "de0005810055", "sedol": "KEEP"},
            {"snapshot_date": date(2021, 1, 4), "isin": "DE0005810055"},
            {"snapshot_date": date(2020, 1, 2), "isin": "XX0000000000"},
            {"snapshot_date": date(2020, 1, 2), "isin": None, "cusip": "1"}]
    out, stats = backfill_lseg_ids(rows, LsegClient(DataLd()))
    assert out[0]["ric"] == "DB1Gn.DE" and out[0]["sedol"] == "KEEP"
    assert out[0]["lseg_permid"] == "DB1Gn.DE|TR.OrganizationID"    # identifiers win over conv
    assert out[1]["ric"] == "DB1Gn.DE" and out[1]["sedol"] == "DB1Gn.DE|TR.SEDOL"
    assert "ric" not in out[2] and "ric" not in out[3]
    assert stats["unresolved"] == 1 and stats["no_isin"] == 1 and stats["ric"] == 2
    assert rows[1].get("ric") is None                                 # inputs not mutated
