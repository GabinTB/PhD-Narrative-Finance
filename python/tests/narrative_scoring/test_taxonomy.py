"""Tests for narrative_scoring.taxonomy (Narrative_Taxonomy monorepo layout).

No real data: each test builds a minimal three-file named taxonomy
(`{name}_taxonomy.authored.csv` + semantic/headline paraphrase JSONLs) flat
in a tmp_path acting as the Narrative_Taxonomy root.
"""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from narrative_scoring.taxonomy import (
    CANONICAL_COLUMNS,
    RAVENPACK_BASE_SCHEMA,
    TaxonomyError,
    list_taxonomies,
    load_taxonomy,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


def _write_taxonomy(
    root: Path,
    name: str,
    *,
    primitives: list[str] | None = None,
    k: int = 2,
    with_extra_col: bool = False,
    use_type_subtype: bool = False,
    extra_csv_cols: dict[str, list[str]] | None = None,
) -> Path:
    """Write a complete {name} taxonomy (authored CSV + both JSONLs) into root."""
    primitives = primitives or ["prim_a", "prim_b", "prim_c"]
    root.mkdir(parents=True, exist_ok=True)

    data = {
        "TOPIC": ["topic1"] * len(primitives),
        "GROUP": ["group1"] * len(primitives),
        "DISPLAY_NAME": primitives,
        "DESCRIPTION": [f"description of {n}" for n in primitives],
    }
    if use_type_subtype:
        data["TYPE"] = ["type1"] * len(primitives)
        data["SUB_TYPE"] = ["sub1"] * len(primitives)
    else:
        data["CATEGORY"] = ["cat1"] * len(primitives)
    if with_extra_col:
        data["OBSERVABILITY_CHANNEL"] = ["official-release"] * len(primitives)
    if extra_csv_cols:
        data.update(extra_csv_cols)

    pl.DataFrame(data).write_csv(root / f"{name}_taxonomy.authored.csv")
    # the non-authored skeleton sibling, DISPLAY_NAME/DESCRIPTION blank -- must never be read
    skeleton = dict(data)
    skeleton["DISPLAY_NAME"] = [""] * len(primitives)
    skeleton["DESCRIPTION"] = [""] * len(primitives)
    pl.DataFrame(skeleton).write_csv(root / f"{name}_taxonomy.csv")

    for style in ("semantic", "headline"):
        _write_jsonl(
            root / f"{name}-primitive_{style}_paraphrases.jsonl",
            [
                {
                    "id": f"hash-{style}-{i}",
                    "display_name": p,
                    "master": f"master {style} {p}",
                    "paraphrases": [f"{p} para {i}" for i in range(k)],
                }
                for i, p in enumerate(primitives)
            ],
        )

    return root


class TestCanonicalMapping:
    def test_category_column_direct(self, tmp_path: Path):
        _write_taxonomy(tmp_path, "Evergreen_v5")
        tv = load_taxonomy(tmp_path, "Evergreen_v5")
        df = tv.primitives()
        expected_cols = {"reservoir", "dimension", "narrative", "primitive", "description"}
        assert set(df.columns) >= expected_cols
        assert df["primitive"].to_list() == ["prim_a", "prim_b", "prim_c"]
        assert df["narrative"].to_list() == ["cat1", "cat1", "cat1"]

    def test_category_built_from_type_and_subtype(self, tmp_path: Path):
        _write_taxonomy(tmp_path, "Vendor_RP", use_type_subtype=True)
        tv = load_taxonomy(tmp_path, "Vendor_RP")
        df = tv.primitives()
        assert df["narrative"].to_list() == ["type1-sub1", "type1-sub1", "type1-sub1"]
        assert CANONICAL_COLUMNS["narrative"] == "CATEGORY"

    def test_ravenpack_baseline_null_filled_when_absent(self, tmp_path: Path):
        _write_taxonomy(tmp_path, "Evergreen_v5")
        tv = load_taxonomy(tmp_path, "Evergreen_v5")
        df = tv.primitives()
        # ROLE/TYPE/SUB_TYPE/SCHEDULED/VALID_ENTITY_TYPES/TAGS never provided -> null, not missing
        for col in ("ROLE", "TYPE", "SUB_TYPE", "SCHEDULED", "VALID_ENTITY_TYPES", "TAGS"):
            assert col in df.columns
            assert df[col].null_count() == df.height

    def test_extra_columns_beyond_baseline_carried_through(self, tmp_path: Path):
        _write_taxonomy(tmp_path, "Evergreen_v5", with_extra_col=True)
        tv = load_taxonomy(tmp_path, "Evergreen_v5")
        df = tv.primitives()
        assert df["OBSERVABILITY_CHANNEL"].to_list() == ["official-release"] * 3

    def test_custom_base_schema_import_and_extend(self, tmp_path: Path):
        _write_taxonomy(
            tmp_path, "Evergreen_v5",
            extra_csv_cols={"MY_EXTRA_COL": ["x", "y", "z"]},
        )
        custom_schema = [*RAVENPACK_BASE_SCHEMA, "MY_EXTRA_COL"]
        tv = load_taxonomy(tmp_path, "Evergreen_v5", base_schema=custom_schema)
        df = tv.primitives()
        assert df["MY_EXTRA_COL"].to_list() == ["x", "y", "z"]

    def test_never_reads_non_authored_skeleton(self, tmp_path: Path):
        # the skeleton sibling has blank DISPLAY_NAME/DESCRIPTION; if it were
        # ever read instead of the authored one, primitives() would be empty/blank.
        _write_taxonomy(tmp_path, "Evergreen_v5")
        tv = load_taxonomy(tmp_path, "Evergreen_v5")
        df = tv.primitives()
        assert "" not in df["primitive"].to_list()
        assert "" not in df["description"].to_list()

    def test_masters_and_paraphrases_and_k(self, tmp_path: Path):
        _write_taxonomy(tmp_path, "Evergreen_v5", k=3)
        tv = load_taxonomy(tmp_path, "Evergreen_v5", paraphrase_style="semantic")
        masters = tv.masters()
        assert masters["prim_a"] == "master semantic prim_a"
        paraphrases = tv.paraphrases()
        assert len(paraphrases["prim_a"]) == 3
        assert tv.k == 3

        tv_head = load_taxonomy(tmp_path, "Evergreen_v5", paraphrase_style="headline")
        assert tv_head.masters()["prim_a"] == "master headline prim_a"


class TestListTaxonomies:
    def test_finds_names_in_root(self, tmp_path: Path):
        _write_taxonomy(tmp_path, "Evergreen_v5")
        _write_taxonomy(tmp_path, "Other_Taxonomy")
        assert list_taxonomies(tmp_path) == ["Evergreen_v5", "Other_Taxonomy"]

    def test_ignores_legacy_subdirectory(self, tmp_path: Path):
        _write_taxonomy(tmp_path, "Evergreen_v5")
        _write_taxonomy(tmp_path / "Legacy" / "v4", "v4.3")
        assert list_taxonomies(tmp_path) == ["Evergreen_v5"]


class TestValidation:
    def test_valid_taxonomy_passes(self, tmp_path: Path):
        _write_taxonomy(tmp_path, "Evergreen_v5")
        tv = load_taxonomy(tmp_path, "Evergreen_v5")  # must not raise
        assert tv.validate() == []

    def test_missing_authored_csv(self, tmp_path: Path):
        tmp_path.mkdir(parents=True, exist_ok=True)
        with pytest.raises(TaxonomyError, match="authored taxonomy CSV missing"):
            load_taxonomy(tmp_path, "Nonexistent")

    def test_duplicate_display_name(self, tmp_path: Path):
        _write_taxonomy(tmp_path, "Evergreen_v5")
        csv_path = tmp_path / "Evergreen_v5_taxonomy.authored.csv"
        df = pl.read_csv(csv_path)
        df = pl.concat([df, df.head(1)])
        df.write_csv(csv_path)
        with pytest.raises(TaxonomyError, match="duplicate DISPLAY_NAME"):
            load_taxonomy(tmp_path, "Evergreen_v5")

    def test_empty_display_name(self, tmp_path: Path):
        _write_taxonomy(tmp_path, "Evergreen_v5")
        csv_path = tmp_path / "Evergreen_v5_taxonomy.authored.csv"
        df = pl.read_csv(csv_path)
        df = df.with_columns(
            pl.when(pl.col("DISPLAY_NAME") == "prim_a")
            .then(pl.lit(""))
            .otherwise(pl.col("DISPLAY_NAME"))
            .alias("DISPLAY_NAME")
        )
        df.write_csv(csv_path)
        with pytest.raises(TaxonomyError, match="empty DISPLAY_NAME"):
            load_taxonomy(tmp_path, "Evergreen_v5")

    def test_empty_description(self, tmp_path: Path):
        _write_taxonomy(tmp_path, "Evergreen_v5")
        csv_path = tmp_path / "Evergreen_v5_taxonomy.authored.csv"
        df = pl.read_csv(csv_path)
        df = df.with_columns(
            pl.when(pl.col("DISPLAY_NAME") == "prim_a")
            .then(pl.lit(""))
            .otherwise(pl.col("DESCRIPTION"))
            .alias("DESCRIPTION")
        )
        df.write_csv(csv_path)
        with pytest.raises(TaxonomyError, match="empty DESCRIPTION"):
            load_taxonomy(tmp_path, "Evergreen_v5")

    def test_broken_bijection_via_display_name(self, tmp_path: Path):
        _write_taxonomy(tmp_path, "Evergreen_v5")
        jsonl_path = tmp_path / "Evergreen_v5-primitive_semantic_paraphrases.jsonl"
        records = [json.loads(line) for line in jsonl_path.read_text().splitlines()]
        records = [r for r in records if r["display_name"] != "prim_a"]
        _write_jsonl(jsonl_path, records)
        with pytest.raises(TaxonomyError, match="mismatch with CSV"):
            load_taxonomy(tmp_path, "Evergreen_v5", paraphrase_style="semantic")

    def test_jsonl_id_is_not_the_join_key(self, tmp_path: Path):
        # a record's "id" is an opaque hash; only display_name matters for the join.
        _write_taxonomy(tmp_path, "Evergreen_v5")
        jsonl_path = tmp_path / "Evergreen_v5-primitive_headline_paraphrases.jsonl"
        records = [json.loads(line) for line in jsonl_path.read_text().splitlines()]
        for r in records:
            r["id"] = "some-opaque-hash-unrelated-to-display-name"
        _write_jsonl(jsonl_path, records)
        tv = load_taxonomy(tmp_path, "Evergreen_v5", paraphrase_style="headline")
        assert tv.validate() == []
        assert set(tv.paraphrases().keys()) == {"prim_a", "prim_b", "prim_c"}

    def test_inconsistent_k_within_file(self, tmp_path: Path):
        _write_taxonomy(tmp_path, "Evergreen_v5", k=2)
        jsonl_path = tmp_path / "Evergreen_v5-primitive_semantic_paraphrases.jsonl"
        records = [json.loads(line) for line in jsonl_path.read_text().splitlines()]
        records[0]["paraphrases"] = records[0]["paraphrases"][:1]
        _write_jsonl(jsonl_path, records)
        with pytest.raises(TaxonomyError, match="inconsistent paraphrase counts"):
            load_taxonomy(tmp_path, "Evergreen_v5", paraphrase_style="semantic")

    def test_missing_master(self, tmp_path: Path):
        _write_taxonomy(tmp_path, "Evergreen_v5")
        jsonl_path = tmp_path / "Evergreen_v5-primitive_semantic_paraphrases.jsonl"
        records = [json.loads(line) for line in jsonl_path.read_text().splitlines()]
        records[0]["master"] = ""
        _write_jsonl(jsonl_path, records)
        with pytest.raises(TaxonomyError, match="missing 'master'"):
            load_taxonomy(tmp_path, "Evergreen_v5", paraphrase_style="semantic")

    def test_missing_semantic_jsonl(self, tmp_path: Path):
        _write_taxonomy(tmp_path, "Evergreen_v5")
        (tmp_path / "Evergreen_v5-primitive_semantic_paraphrases.jsonl").unlink()
        with pytest.raises(TaxonomyError, match="semantic paraphrases JSONL file missing"):
            load_taxonomy(tmp_path, "Evergreen_v5")

    def test_missing_headline_jsonl(self, tmp_path: Path):
        _write_taxonomy(tmp_path, "Evergreen_v5")
        (tmp_path / "Evergreen_v5-primitive_headline_paraphrases.jsonl").unlink()
        with pytest.raises(TaxonomyError, match="headline paraphrases JSONL file missing"):
            load_taxonomy(tmp_path, "Evergreen_v5")
