"""universe.index_universe: index universes loaded by name from snapshot files."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from universe.index_universe import index_universe_rows, list_index_universes

D1, D2 = date(2014, 3, 3), date(2019, 1, 2)


def _c(day, isin, name="Meta Platforms", ticker="META"):
    return {"snapshot_date": day, "name": name, "ticker": ticker, "isin": isin,
            "figi": "BBG0", "vendor_id": 1}


def _m(day, isin, country="US", sub="45101010"):
    return {"snapshot_date": day, "name": "n", "ticker": "t", "isin": isin,
            "country_name": {"US": "United States", "NL": "Netherlands", "ES": "Spain"}[country],
            "country_iso": country, "region": "Americas" if country == "US" else "Europe",
            "gics_subindustry_code": sub}


def _write(root: Path, index: str, cons: list[dict], meta: list[dict] | None = None,
           span: str = "20060703_to_20260901") -> Path:
    root.mkdir(exist_ok=True)
    pl.DataFrame(cons).write_parquet(root / f"{index}_constituents-{span}.parquet")
    if meta is not None:
        pl.DataFrame(meta).write_parquet(root / f"{index}_metadata-{span}.parquet")
    return root


def test_rows_by_index_name_without_the_gics_history(tmp_path):
    cons = [_c(D1, "US30303M1027"), _c(D2, "US30303M1027"),
            _c(D2, "ES0118900010", "Ferrovial", "FER"), _c(D2, None, "No Isin", None)]
    meta = [_m(D1, "US30303M1027", sub="45101010"), _m(D1, "US30303M1027", sub="50203010"),
            _m(D1, "US30303M1027", sub="50203010"),
            _m(D2, "US30303M1027", sub="45101010"), _m(D2, "US30303M1027", sub="50203010"),
            _m(D2, "ES0118900010", "ES"), _m(D2, "ES0118900010", "NL")]
    root = _write(tmp_path / "idx", "MSCI_WORLD", cons, meta)
    _write(root, "SP500", [_c(D2, "US0378331005", "Apple", "AAPL")])      # no metadata file
    (root / "DEPRECIATED-MSCI_WORLD_constituents-20060703_to_20260901.parquet").write_bytes(b"")
    assert list_index_universes(root) == ["MSCI_WORLD", "SP500"]

    rows = index_universe_rows(root, "MSCI_WORLD")
    assert len(rows) == 4
    meta_rows = [r for r in rows if r["isin"] == "US30303M1027"]
    assert [r["snapshot_date"] for r in meta_rows] == [D1, D2]
    assert all(r["country_iso"] == "US" and r["region"] == "Americas" for r in meta_rows)
    assert not any(k.startswith("gics") or k == "vendor_id" for k in rows[0])
    fer = next(r for r in rows if r["isin"] == "ES0118900010")
    assert fer["country_iso"] is None and fer["region"] == "Europe"      # 2 countries: unset
    assert any(r["isin"] is None for r in rows)

    sp = index_universe_rows(root, "SP500")
    assert [r["ticker"] for r in sp] == ["AAPL"] and "country_iso" not in sp[0]


def test_unknown_index_duplicates_and_two_files_raise(tmp_path):
    root = _write(tmp_path / "a", "MSCI_WORLD", [_c(D1, "X1")])
    with pytest.raises(FileNotFoundError, match=r"available: \['MSCI_WORLD'\]"):
        index_universe_rows(root, "SP500")
    root = _write(tmp_path / "b", "SP500", [_c(D1, "X1"), _c(D1, "X1")])
    with pytest.raises(ValueError, match="duplicate"):
        index_universe_rows(root, "SP500")
    root = _write(tmp_path / "c", "SP500", [_c(D1, "X1")])
    _write(root, "SP500", [_c(D1, "X1")], span="20000101_to_20100101")
    with pytest.raises(ValueError, match="keep exactly one"):
        index_universe_rows(root, "SP500")
