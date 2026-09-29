"""ravenpack.entity_reference: point-in-time RP_ENTITY_ID backfill from a synthetic
entity reference file (long format, as RavenPack publishes it)."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import polars as pl
import pytest

from ravenpack.edge_api.entities import EntityMappingResponse
from ravenpack.entity_reference import (
    backfill_rp_entity_id,
    load_identifier_windows,
    write_identifier_extract,
)

REF = [
    # id, type, value, start, end
    ("AAA111", "ISIN", "US0000000001", "2000-01-01", ""),
    ("AAA111", "NAME", "Alpha Inc", "2000-01-01", ""),
    # ISIN re-used: company B until 2015, company C from 2016 (a reverse merger)
    ("BBB222", "ISIN", "US0000000002", "2000-01-01", "2015-12-31"),
    ("CCC333", "ISIN", "US0000000002", "2016-01-01", ""),
    # gap in the ISIN history (2011-05-06 .. 2012-02-06): only the undated fallback covers it
    ("DDD444", "ISIN", "DE0000000004", "2001-02-05", "2011-05-05"),
    ("DDD444", "ISIN", "DE0000000004", "2012-02-07", ""),
    # two entities holding one ISIN at the same time: ambiguous, never guessed
    ("EEE555", "ISIN", "GB0000000005", "2000-01-01", ""),
    ("FFF666", "ISIN", "GB0000000005", "2000-01-01", ""),
    ("GGG777", "CUSIP", "123456789", "2000-01-01", ""),
    ("HHH888", "SEDOL", "B0YBKJ7", "2000-01-01", ""),
    ("III999", "CIK", "320193", "2000-01-01", ""),
    ("JJJ000", "CORP_DEBT_ISIN", "US0000000009", "2000-01-01", ""),
]


@pytest.fixture
def ref_csv(tmp_path: Path) -> Path:
    path = tmp_path / "company_2026-09-27.csv"
    pl.DataFrame(REF, schema=["RP_ENTITY_ID", "DATA_TYPE", "DATA_VALUE", "RANGE_START",
                              "RANGE_END"], orient="row") \
        .with_columns(pl.lit("COMP").alias("ENTITY_TYPE")) \
        .select("RP_ENTITY_ID", "ENTITY_TYPE", "DATA_TYPE", "DATA_VALUE", "RANGE_START",
                "RANGE_END") \
        .with_columns(pl.col("RANGE_END").replace("", None)) \
        .write_csv(path)
    return path


def _row(day: date, **ids: Any) -> dict[str, Any]:
    return {"snapshot_date": day, "name": "x", "ticker": "X", "isin": None, "cusip": None,
            "sedol": None, "cik": None, **ids}


def _run(ref: Path, rows, **kw):
    return backfill_rp_entity_id(rows, load_identifier_windows(ref), **kw)


def test_point_in_time_isin_follows_the_holder(ref_csv):
    rows = [_row(date(2014, 6, 2), isin="US0000000002"),
            _row(date(2020, 6, 1), isin="us0000000002 "),          # normalised
            _row(date(2020, 6, 1), isin="US0000000001")]
    out, stats = _run(ref_csv, rows)
    assert [r["rp_entity_id"] for r in out] == ["BBB222", "CCC333", "AAA111"]
    assert {r["rp_entity_match"] for r in out} == {"isin"}
    assert stats["isin"] == 3 and stats["unmatched"] == 0


def test_gap_uses_the_flagged_undated_fallback(ref_csv):
    out, stats = _run(ref_csv, [_row(date(2011, 9, 1), isin="DE0000000004"),
                                _row(date(2013, 1, 2), isin="DE0000000004")])
    assert [(r["rp_entity_id"], r["rp_entity_match"]) for r in out] == [
        ("DDD444", "isin_undated"), ("DDD444", "isin")]
    # an ISIN held by two entities over its life is never matched outside its windows
    out, _ = _run(ref_csv, [_row(date(1999, 1, 1), isin="US0000000002")])
    assert out[0].get("rp_entity_id") is None


def test_ambiguous_is_left_null_and_counted(ref_csv):
    out, stats = _run(ref_csv, [_row(date(2020, 1, 1), isin="GB0000000005")])
    assert out[0].get("rp_entity_id") is None
    assert stats["ambiguous_isin"] == 1 and stats["unmatched"] == 1


def test_other_identifiers_in_order_and_cik_padding(ref_csv):
    rows = [_row(date(2020, 1, 1), isin="XX0000000000", cusip="123456789"),
            _row(date(2020, 1, 1), sedol="b0ybkj7"),
            _row(date(2020, 1, 1), cik="0000320193"),               # universe CIKs are padded
            _row(date(2020, 1, 1), isin="US0000000009")]            # a bond ISIN: not used
    out, stats = _run(ref_csv, rows)
    assert [(r.get("rp_entity_id"), r.get("rp_entity_match")) for r in out] == [
        ("GGG777", "cusip"), ("HHH888", "sedol"), ("III999", "cik"), (None, None)]


def test_existing_value_is_never_overwritten(ref_csv):
    out, stats = _run(ref_csv, [_row(date(2020, 1, 1), isin="US0000000001",
                                     rp_entity_id="KEEP00", rp_entity_match="manual")])
    assert (out[0]["rp_entity_id"], out[0]["rp_entity_match"]) == ("KEEP00", "manual")
    assert stats == {"unmatched": 0}


def test_extract_round_trip_and_value_filter(ref_csv, tmp_path):
    ext = tmp_path / "ids.parquet"
    assert write_identifier_extract(ref_csv, ext) == len(REF) - 2      # NAME, CORP_DEBT_ISIN
    rows = [_row(date(2020, 6, 1), isin="US0000000002")]
    a, _ = _run(ref_csv, rows)
    b, _ = backfill_rp_entity_id(rows, load_identifier_windows(ext))
    assert a == b
    w = load_identifier_windows(ext, {"ISIN": {"US0000000001"}})
    assert w["RP_ENTITY_ID"].to_list() == ["AAA111"]


class _Entities:
    def __init__(self, response: dict[str, Any]):
        self.response, self.calls = response, []

    def map(self, identifiers):
        self.calls.append(identifiers)
        return EntityMappingResponse.model_validate(self.response)


def test_api_fallback_keeps_only_a_unique_top_company(ref_csv):
    response = {"identifiers_mapped": [
        {"requested_data": {"client_id": "ZZ0000000001"},
         "rp_entities": [{"rp_entity_id": "API001", "rp_entity_type": "COMP", "score": 90},
                         {"rp_entity_id": "API002", "rp_entity_type": "COMP", "score": 40}]},
        {"requested_data": {"client_id": "ZZ0000000002"},
         "rp_entities": [{"rp_entity_id": "API003", "rp_entity_type": "COMP", "score": 70},
                         {"rp_entity_id": "API004", "rp_entity_type": "COMP", "score": 70}]},
    ]}
    client = SimpleNamespace(entities=_Entities(response))
    rows = [_row(date(2019, 1, 1), isin="ZZ0000000001"),
            _row(date(2020, 1, 1), isin="ZZ0000000001"),
            _row(date(2020, 1, 1), isin="ZZ0000000002"),
            _row(date(2020, 1, 1), isin="US0000000001")]                # matched before the API
    out, stats = _run(ref_csv, rows, client=client)
    assert [(r.get("rp_entity_id"), r.get("rp_entity_match")) for r in out] == [
        ("API001", "api"), ("API001", "api"), (None, None), ("AAA111", "isin")]
    sent = client.entities.calls[0]
    assert sorted(i["isin"] for i in sent) == ["ZZ0000000001", "ZZ0000000002"]  # once per ISIN
    assert next(i for i in sent if i["isin"] == "ZZ0000000001")["date"] == date(2020, 1, 1)
    assert stats["api"] == 2 and stats["unmatched"] == 1
