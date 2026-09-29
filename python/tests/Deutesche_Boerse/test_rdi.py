"""deutsche_boerse.rdi with a stubbed A7 RDI v2 API (no network)."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from deutsche_boerse.rdi import (
    RdiClient,
    backfill_dbga_secid,
    contract_kind,
    eurex_contracts,
    eurex_derivatives,
    reference_table,
    reference_tables,
    trading_day_for,
)


def _xetr_snaps(isin: str, security: str, desc: str) -> list[dict]:
    return [{"Template": "ProductSnapshot", "MarketSegment": isin},
            {"Template": "InstrumentSnapshot", "SecurityID": security, "SecurityType": "CS",
             "SecurityDesc": desc, "SecurityAlt": [{"SecurityAltIDSource": "4",
                                                    "SecurityAltID": isin}]}]


def _xeur_snaps(symbol: str, family: str, underlying: str | None, source: str = "4") -> list:
    under = {"UnderlyingSecurityID": underlying, "UnderlyingSecurityIDSource": source} \
        if underlying else {}
    return [{"Template": "ProductSnapshot", "MarketSegment": symbol,
             "DerivativesDescriptorGroup": {"ParentMktSegmID": family,
                                            "MarketSegmentDesc": f"{family} ON {symbol}"},
             "UnderlyingDescriptorGroup": under},
            {"Template": "FuturesStatus", "SecurityID": "999"}]


class StubHttp:
    """Serves /v2/rdi paths from a dict {mic: {day: {segment: {security: snapshots}}}}."""

    def __init__(self, data: dict, fail_first: int = 0):
        self.data, self.calls, self.fail_first = data, [], fail_first

    def get(self, path: str):
        self.calls.append(path)
        if self.fail_first:
            self.fail_first -= 1
            return _Resp(503, None)
        parts = [p for p in path.split("/") if p][2:]      # after v2/rdi
        node: object = self.data
        for p in parts:
            if not isinstance(node, dict):
                return _Resp(404, None)
            node = node.get(int(p)) if p.isdigit() and int(p) in node else node.get(p)
            if node is None:
                return _Resp(404, None)
        if isinstance(node, dict):
            return _Resp(200, sorted(node))
        return _Resp(200, node)


class _Resp:
    def __init__(self, status, payload):
        self.status_code, self._payload = status, payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


D1, D2, D3 = 20240102, 20240201, 20240301
DATA = {
    "XETR": {
        D1: {100: {"501": _xetr_snaps("DE0005810055", "501", "DT BOERSE")},
             101: {"502": _xetr_snaps("DE0007164600", "502", "SAP")}},
        D2: {100: {"501": _xetr_snaps("DE0005810055", "501", "DT BOERSE")},
             101: {"502": _xetr_snaps("DE0007164600", "502", "SAP")},
             102: {"503": _xetr_snaps("NL0000235190", "503", "AIRBUS")}},
        D3: {100: {"501": _xetr_snaps("DE0005810055", "501", "DT BOERSE")}},
    },
    "XEUR": {
        D1: {3: {"11": _xeur_snaps("DB1", "OSTK", "DE0005810055"),
                 "12": _xeur_snaps("DB1", "OSTK", "DE0005810055")},
             4: {"21": _xeur_snaps("DB1H", "FSTK", "DE0005810055")},
             5: {"31": _xeur_snaps("A2EN", "FSTK", "XC000A1DKC51")},
             6: {"41": _xeur_snaps("FDAX", "FINX", "DE0008469008", source="M")}},
        D2: {3: {"11": _xeur_snaps("DB1", "OSTK", "DE000NEWISIN")}},     # corporate action
    },
}


def _client(**kw) -> tuple[RdiClient, StubHttp]:
    http = StubHttp(DATA, **kw)
    return RdiClient(http, workers=2), http


def test_xetra_table_one_row_per_security():
    client, _ = _client()
    t = reference_table(client, "XETR", D1)
    assert t[["segment_id", "security_id", "isin", "security_type"]].values.tolist() == [
        [100, "501", "DE0005810055", "CS"], [101, "502", "DE0007164600", "CS"]]


def test_eurex_table_one_row_per_product_with_isin_underlyings_only():
    client, _ = _client()
    t = reference_table(client, "XEUR", D1)
    assert t[["product", "family", "underlying_isin"]].values.tolist() == [
        ["DB1", "OSTK", "DE0005810055"], ["DB1H", "FSTK", "DE0005810055"],
        ["A2EN", "FSTK", "XC000A1DKC51"], ["FDAX", "FINX", None]]     # source "M": not an ISIN


def test_xetra_dates_reuse_known_segments_and_only_resolve_new_ones(tmp_path: Path):
    client, http = _client()
    tables = reference_tables(client, "XETR", [D1, D2], tmp_path)
    d2_detail_calls = [c for c in http.calls if f"/{D2}/" in c and c.count("/") >= 6]
    assert sorted(tables[D2]["isin"]) == ["DE0005810055", "DE0007164600", "NL0000235190"]
    assert (tmp_path / "XETR" / f"{D2}.parquet").exists()
    # D2 re-checks the reused segments (2, all sampled) and resolves the new one only
    assert any("/102/" in c for c in d2_detail_calls)
    http.calls.clear()
    again = reference_tables(client, "XETR", [D1, D2], tmp_path)            # from the cache
    assert http.calls == [] and again[D2].equals(tables[D2])


def test_a_changed_reused_segment_forces_a_full_scan(tmp_path: Path):
    client, _ = _client()
    known = reference_table(client, "XETR", D1)
    known.loc[known["segment_id"] == 100, "isin"] = "XX0000000000"          # stale
    t = reference_table(client, "XETR", D2, known)
    assert "XX0000000000" not in set(t["isin"]) and len(t) == 3


def test_eurex_dates_are_always_scanned_in_full(tmp_path: Path):
    client, _ = _client()
    tables = reference_tables(client, "XEUR", [D1, D2], tmp_path)
    assert tables[D2]["underlying_isin"].tolist() == ["DE000NEWISIN"]


def test_trading_day_mapping():
    days = [D1, D2, D3]
    assert trading_day_for(date(2024, 2, 15), days) == D2
    assert trading_day_for(date(2024, 2, 1), days) == D2
    assert trading_day_for(date(2023, 12, 29), days) is None


def test_backfill_dbga_secid(tmp_path: Path):
    client, _ = _client()
    rows = [{"snapshot_date": date(2024, 2, 5), "isin": "de0005810055"},
            {"snapshot_date": date(2024, 3, 4), "isin": "DE0007164600"},        # delisted by D3
            {"snapshot_date": date(2023, 6, 1), "isin": "DE0005810055"},        # before history
            {"snapshot_date": date(2024, 2, 5), "isin": None},
            {"snapshot_date": date(2024, 2, 5), "isin": "NL0000235190", "dbga_secid": "KEEP"}]
    out, stats = backfill_dbga_secid(rows, client, tmp_path)
    assert [r.get("dbga_secid") for r in out] == ["501", None, None, None, "KEEP"]
    assert stats == {"filled": 1, "not_listed": 1, "before_history": 1, "no_isin": 1}
    assert rows[0].get("dbga_secid") is None                                 # not mutated


def test_eurex_derivatives_join_on_underlying_isin_and_date(tmp_path: Path):
    client, _ = _client()
    rows = [{"snapshot_date": date(2024, 1, 15), "isin": "DE0005810055"},
            {"snapshot_date": date(2024, 2, 15), "isin": "DE0005810055"},       # ISIN changed
            {"snapshot_date": date(2024, 1, 15), "isin": "US0378331005"}]       # no product
    ed = eurex_derivatives(rows, client, tmp_path)
    assert ed[["snapshot_date", "product_symbol", "family"]].values.tolist() == [
        [date(2024, 1, 15), "DB1", "OSTK"], [date(2024, 1, 15), "DB1H", "FSTK"]]
    assert set(ed["rdi_date"]) == {D1}


def test_transient_errors_are_retried(monkeypatch):
    import deutsche_boerse.rdi as rdi
    monkeypatch.setattr(rdi.time, "sleep", lambda s: None)
    client, http = _client(fail_first=2)
    assert client.dates("XETR") == [D1, D2, D3] and len(http.calls) == 3
    client, _ = _client(fail_first=9)
    with pytest.raises(RuntimeError, match="HTTP 503"):
        client.dates("XETR")


# ---------------------------------------------------------------------------
# eurex_contracts
# ---------------------------------------------------------------------------

def _contract(symbol: str, security: str, sec_type: str, *, style: str | None = None,
              settl: str | None = None, put_call: str | None = None,
              strike: float | None = None, expiry: int = 20261016,
              complex_: str = "1") -> list[dict]:
    detail = {"ContractDate": expiry, "ContractMonthYear": expiry // 100,
              "MaturityFrequencyUnit": "Mo", "ContractMultiplier": 100}
    for key, value in (("ExerciseStyle", style), ("SettlMethod", settl),
                       ("PutOrCall", put_call), ("StrikePrice", strike)):
        if value is not None:
            detail[key] = value
    return [{"Template": "ProductSnapshot", "MarketSegment": symbol},
            {"Template": "InstrumentSnapshot", "SecurityID": security, "SecurityType": sec_type,
             "ProductComplex": complex_, "SecurityDesc": f"{symbol} {security}",
             "DerivativesDescriptorGroup": {"DisplayName": f"{symbol} OCT26", "IsPrimary": "Y",
                                            "SimpleInstrumentDescriptorGroup": detail}}]


DAY = 20260928
CONTRACTS = {"XEUR": {DAY: {
    361: {"1": _contract("DB1", "1", "OPT", style="1", put_call="1", strike=280.0),
          "2": _contract("DB1", "2", "OPT", style="1", put_call="0", strike=280.0),
          "3": _contract("DB1", "3", "OPT", style="1", put_call="1", complex_="5")},  # strategy
    46225: {"4": _contract("DB1E", "4", "OPT", style="0", put_call="1", strike=300.0)},
    363: {"5": _contract("DB1H", "5", "FUT", settl="C")},
    567612: {"6": _contract("DB1P", "6", "FUT", settl="P", expiry=20260928)},
    134594: {"7": _contract("TDB1", "7", "TRF", settl="C")},
}}}


def test_eurex_contracts_classify_each_product_from_its_fields():
    client = RdiClient(StubHttp(CONTRACTS), workers=2)
    kinds = {seg: eurex_contracts(client, DAY, seg)["kind"].unique().tolist()
             for seg in (361, 46225, 363, 567612, 134594)}
    assert kinds == {361: ["option_american"], 46225: ["option_european"],
                     363: ["future_cash"], 567612: ["future_physical"],
                     134594: ["total_return_future"]}
    opts = eurex_contracts(client, DAY, 361)
    assert opts[["security_id", "put_call", "strike", "expiry"]].values.tolist() == [
        ["1", "call", 280.0, 20261016], ["2", "put", 280.0, 20261016]]    # strategy dropped
    assert set(opts["product"]) == {"DB1"} and opts["is_primary"].all()


def test_eurex_contracts_kind_filter_skips_other_products_after_one_call():
    http = StubHttp(CONTRACTS)
    client = RdiClient(http, workers=2)
    assert eurex_contracts(client, DAY, 361, kinds=["future_cash"]).empty
    assert sum(1 for c in http.calls if not c.endswith("/")) == 1         # one detail call
    fut = eurex_contracts(client, DAY, 363, kinds=["future_cash", "future_physical"])
    assert fut["security_id"].tolist() == ["5"]
    with pytest.raises(ValueError, match="unknown kind"):
        eurex_contracts(client, DAY, 363, kinds=["future"])
    assert eurex_contracts(client, DAY, 999).empty                        # unknown segment


def test_contract_kind_unknown_values_are_other():
    assert contract_kind({"SecurityType": "OPT", "DerivativesDescriptorGroup": {
        "SimpleInstrumentDescriptorGroup": {"ExerciseStyle": "2"}}}) == "other"   # Bermudan
    assert contract_kind({"SecurityType": "MLEG"}) == "other"
