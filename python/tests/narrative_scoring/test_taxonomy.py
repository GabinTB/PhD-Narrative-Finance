"""Tests for narrative_scoring.taxonomy.

No real data: each test builds a minimal six-file taxonomy version directory
on tmp_path (CSV + 4 JSONLs for taxonomy/garbage x pure/headlined).
"""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from narrative_scoring.taxonomy import (
    CANONICAL_COLUMNS,
    TaxonomyError,
    load_taxonomy,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


def _write_taxonomy_version(
    root: Path,
    version: str,
    *,
    family: str = "evergreen",
    primitives: list[str] | None = None,
    garbage_primitives: list[str] | None = None,
    k: int = 2,
    gc_k: int | None = None,
    with_role: bool = False,
    use_type_subtype: bool = False,
    no_garbage: bool = False,
) -> Path:
    primitives = primitives or ["prim_a", "prim_b", "prim_c"]
    garbage_primitives = garbage_primitives or ["gc_a", "gc_b"]
    gc_k = gc_k if gc_k is not None else k

    vdir = root / version
    vdir.mkdir(parents=True)

    def _csv_df(names: list[str], role: bool) -> pl.DataFrame:
        data = {
            "TOPIC": ["topic1"] * len(names),
            "GROUP": ["group1"] * len(names),
            "DISPLAY_NAME": names,
            "DESCRIPTION": [f"description of {n}" for n in names],
        }
        if use_type_subtype:
            data["TYPE"] = ["type1"] * len(names)
            data["SUB_TYPE"] = ["sub1"] * len(names)
        else:
            data["CATEGORY"] = ["cat1"] * len(names)
        if role:
            data["ROLE"] = ["subject"] * len(names)
        return pl.DataFrame(data)

    _csv_df(primitives, role=with_role).write_csv(vdir / f"{family}_taxonomy.csv")
    if not no_garbage:
        _csv_df(garbage_primitives, role=with_role).write_csv(
            vdir / "garbage-catching_taxonomy.csv"
        )

    for style in ("pure", "headlined"):
        suffix = "" if style == "pure" else "-headlined"
        _write_jsonl(
            vdir / f"{family}_taxonomy_primitive_paraphrases{suffix}.jsonl",
            [
                {
                    "id": p,
                    "master": f"master {style} {p}",
                    "paraphrases": [f"{p} para {i}" for i in range(k)],
                }
                for p in primitives
            ],
        )
        if not no_garbage:
            _write_jsonl(
                vdir / f"garbage-catching_taxonomy_primitive_paraphrases{suffix}.jsonl",
                [
                    {
                        "id": p,
                        "master": f"master {style} {p}",
                        "paraphrases": [f"{p} para {i}" for i in range(gc_k)],
                    }
                    for p in garbage_primitives
                ],
            )

    return vdir


class TestCanonicalMapping:
    def test_evergreen_category_column(self, tmp_path: Path):
        _write_taxonomy_version(tmp_path, "v1", family="evergreen")
        tv = load_taxonomy(tmp_path, "v1", family="evergreen")
        df = tv.primitives()
        expected_cols = {"reservoir", "dimension", "narrative", "primitive", "description"}
        assert set(df.columns) >= expected_cols
        assert df["primitive"].to_list() == ["prim_a", "prim_b", "prim_c"]
        assert df["narrative"].to_list() == ["cat1", "cat1", "cat1"]

    def test_ravenpack_type_subtype_and_role(self, tmp_path: Path):
        _write_taxonomy_version(
            tmp_path, "v1", family="ravenpack", with_role=True, use_type_subtype=True
        )
        tv = load_taxonomy(tmp_path, "v1", family="ravenpack")
        df = tv.primitives()
        # CATEGORY constructed from TYPE + "-" + SUB_TYPE
        assert df["narrative"].to_list() == ["type1-sub1", "type1-sub1", "type1-sub1"]
        # ROLE carried through as metadata, not part of the hierarchy
        assert "ROLE" in df.columns
        assert CANONICAL_COLUMNS["narrative"] == "CATEGORY"

    def test_garbage_primitives_canonical_frame(self, tmp_path: Path):
        _write_taxonomy_version(tmp_path, "v1")
        tv = load_taxonomy(tmp_path, "v1")
        gc = tv.garbage_primitives()
        assert gc["primitive"].to_list() == ["gc_a", "gc_b"]

    def test_masters_and_paraphrases_and_k(self, tmp_path: Path):
        _write_taxonomy_version(tmp_path, "v1", k=3)
        tv = load_taxonomy(tmp_path, "v1", paraphrase_style="pure")
        masters = tv.masters()
        assert masters["prim_a"] == "master pure prim_a"
        paraphrases = tv.paraphrases()
        assert len(paraphrases["prim_a"]) == 3
        assert tv.k == 3

        tv_head = load_taxonomy(tmp_path, "v1", paraphrase_style="headlined")
        assert tv_head.masters()["prim_a"] == "master headlined prim_a"


class TestValidation:
    def test_valid_taxonomy_passes(self, tmp_path: Path):
        _write_taxonomy_version(tmp_path, "v1")
        tv = load_taxonomy(tmp_path, "v1")  # must not raise
        assert tv.validate() == []

    def test_duplicate_primitive_in_csv(self, tmp_path: Path):
        vdir = _write_taxonomy_version(tmp_path, "v1")
        df = pl.read_csv(vdir / "evergreen_taxonomy.csv")
        df = pl.concat([df, df.head(1)])
        df.write_csv(vdir / "evergreen_taxonomy.csv")
        with pytest.raises(TaxonomyError, match="duplicate DISPLAY_NAME"):
            load_taxonomy(tmp_path, "v1")

    def test_empty_description(self, tmp_path: Path):
        vdir = _write_taxonomy_version(tmp_path, "v1")
        df = pl.read_csv(vdir / "evergreen_taxonomy.csv")
        df = df.with_columns(
            pl.when(pl.col("DISPLAY_NAME") == "prim_a")
            .then(pl.lit(""))
            .otherwise(pl.col("DESCRIPTION"))
            .alias("DESCRIPTION")
        )
        df.write_csv(vdir / "evergreen_taxonomy.csv")
        with pytest.raises(TaxonomyError, match="empty DESCRIPTION"):
            load_taxonomy(tmp_path, "v1")

    def test_broken_bijection_missing_from_jsonl(self, tmp_path: Path):
        vdir = _write_taxonomy_version(tmp_path, "v1")
        jsonl_path = vdir / "evergreen_taxonomy_primitive_paraphrases.jsonl"
        records = [json.loads(line) for line in jsonl_path.read_text().splitlines()]
        records = [r for r in records if r["id"] != "prim_a"]
        _write_jsonl(jsonl_path, records)
        with pytest.raises(TaxonomyError, match="mismatch with CSV"):
            load_taxonomy(tmp_path, "v1", paraphrase_style="pure")

    def test_inconsistent_k_within_file(self, tmp_path: Path):
        vdir = _write_taxonomy_version(tmp_path, "v1", k=2)
        jsonl_path = vdir / "evergreen_taxonomy_primitive_paraphrases.jsonl"
        records = [json.loads(line) for line in jsonl_path.read_text().splitlines()]
        records[0]["paraphrases"] = records[0]["paraphrases"][:1]  # now K=1 for this one
        _write_jsonl(jsonl_path, records)
        with pytest.raises(TaxonomyError, match="inconsistent paraphrase counts"):
            load_taxonomy(tmp_path, "v1", paraphrase_style="pure")

    def test_taxonomy_garbage_name_overlap(self, tmp_path: Path):
        _write_taxonomy_version(
            tmp_path, "v1", primitives=["prim_a", "shared"], garbage_primitives=["gc_a", "shared"]
        )
        with pytest.raises(TaxonomyError, match="appear in both taxonomy and garbage"):
            load_taxonomy(tmp_path, "v1")

    def test_missing_master(self, tmp_path: Path):
        vdir = _write_taxonomy_version(tmp_path, "v1")
        jsonl_path = vdir / "evergreen_taxonomy_primitive_paraphrases.jsonl"
        records = [json.loads(line) for line in jsonl_path.read_text().splitlines()]
        records[0]["master"] = ""
        _write_jsonl(jsonl_path, records)
        with pytest.raises(TaxonomyError, match="missing 'master'"):
            load_taxonomy(tmp_path, "v1", paraphrase_style="pure")

    def test_differing_k_between_taxonomy_and_garbage_is_legal(self, tmp_path: Path):
        _write_taxonomy_version(tmp_path, "v1", k=2, gc_k=5)
        tv = load_taxonomy(tmp_path, "v1")  # must not raise
        assert tv.validate() == []


class TestNoGarbage:
    """Vendor taxonomies (e.g. RavenPack's own) may ship without a garbage catcher."""

    def test_has_garbage_false_when_csv_absent(self, tmp_path: Path):
        _write_taxonomy_version(tmp_path, "v1", family="vendor", no_garbage=True)
        tv = load_taxonomy(tmp_path, "v1", family="vendor")  # must not raise
        assert tv.has_garbage is False
        assert tv.validate() == []

    def test_has_garbage_true_when_csv_present(self, tmp_path: Path):
        _write_taxonomy_version(tmp_path, "v1")
        tv = load_taxonomy(tmp_path, "v1")
        assert tv.has_garbage is True

    def test_garbage_accessors_raise_clearly_when_absent(self, tmp_path: Path):
        _write_taxonomy_version(tmp_path, "v1", family="vendor", no_garbage=True)
        tv = load_taxonomy(tmp_path, "v1", family="vendor")
        with pytest.raises(TaxonomyError, match="has no garbage-catching taxonomy"):
            tv.garbage_primitives()
        with pytest.raises(TaxonomyError, match="has no garbage-catching taxonomy"):
            tv.garbage_paraphrases()
        with pytest.raises(TaxonomyError, match="has no garbage-catching taxonomy"):
            tv.garbage_masters()

    def test_duplicate_and_empty_description_still_checked_without_garbage(self, tmp_path: Path):
        vdir = _write_taxonomy_version(tmp_path, "v1", family="vendor", no_garbage=True)
        df = pl.read_csv(vdir / "vendor_taxonomy.csv")
        df = df.with_columns(
            pl.when(pl.col("DISPLAY_NAME") == "prim_a")
            .then(pl.lit(""))
            .otherwise(pl.col("DESCRIPTION"))
            .alias("DESCRIPTION")
        )
        df.write_csv(vdir / "vendor_taxonomy.csv")
        with pytest.raises(TaxonomyError, match="empty DESCRIPTION"):
            load_taxonomy(tmp_path, "v1", family="vendor")

    def test_partial_garbage_files_still_fail_validation(self, tmp_path: Path):
        # CSV present but a JSONL missing -> has_garbage=True, so it's a real error,
        # not silently treated as "no garbage catcher".
        vdir = _write_taxonomy_version(tmp_path, "v1")
        (vdir / "garbage-catching_taxonomy_primitive_paraphrases.jsonl").unlink()
        with pytest.raises(TaxonomyError, match="garbage pure JSONL file missing"):
            load_taxonomy(tmp_path, "v1")
