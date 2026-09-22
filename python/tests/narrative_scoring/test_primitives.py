"""Primitive table: sha1-path join, key uniqueness, K, polarity, collisions, scoring matrix."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from narrative_scoring.config import PoolRule
from narrative_scoring.corrections import Correction, apply_mode
from narrative_scoring.primitives import (
    load_primitive_table,
    orphan_pole_candidates,
    primitive_scores,
    sanity_checks,
    scoring_matrix,
    to_text_major,
)
from narrative_scoring.schema import EMBEDDING_DIM

from .conftest import TAX_NAME, toy_rows, unit_rows, write_toy_taxonomy


class TestLoad:
    def test_bijection_and_shape(self, toy_table):
        assert toy_table.n_primitives == 8
        assert toy_table.k_paraphrases == 2 and toy_table.n_texts == 3
        assert len(toy_table.texts) == 24
        assert toy_table.texts[:3] == [
            f"master {toy_table.frame['primitive'][0]}",
            f"{toy_table.frame['primitive'][0]} para 0",
            f"{toy_table.frame['primitive'][0]} para 1",
        ]

    def test_category_collision_across_reservoirs_not_merged(self, toy_table):
        # 'liquidity-stress' exists under macro|funding and firm|balance-sheet
        keys = toy_table.narrative_frame["narrative_key"].to_list()
        assert "macro|funding|liquidity-stress" in keys
        assert "firm|balance-sheet|liquidity-stress" in keys
        assert toy_table.n_narratives == 5
        n_prim = dict(zip(keys, toy_table.narrative_frame["n_primitives"].to_list()))
        assert n_prim["macro|funding|liquidity-stress"] == 3
        assert n_prim["firm|balance-sheet|liquidity-stress"] == 1

    def test_primitives_contiguous_per_narrative(self, toy_table):
        nid = toy_table.primitive_to_narrative
        assert nid.dtype == np.int32
        assert (np.diff(nid) >= 0).all()

    def test_unsigned_sub_type_is_empty_string_in_path(self, tmp_path: Path):
        # The JSONL path keeps an empty SUB_TYPE segment; reading it as null would
        # break the join for every unsigned primitive.
        root = write_toy_taxonomy(tmp_path / "t")
        table = load_primitive_table(root, TAX_NAME, "headline")
        unsigned = table.frame.filter(pl.col("pole") == "")
        assert unsigned.height == 3
        assert table.frame["pole"].null_count() == 0

    def test_broken_bijection_raises(self, tmp_path: Path):
        root = write_toy_taxonomy(tmp_path / "t")
        p = root / f"{TAX_NAME}-primitive_headline_paraphrases.jsonl"
        recs = [json.loads(line) for line in p.read_text().splitlines()]
        recs[0]["id"] = "0" * 40
        p.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
        with pytest.raises(ValueError, match="bijection"):
            load_primitive_table(root, TAX_NAME, "headline")

    def test_display_name_is_not_the_join_key(self, tmp_path: Path):
        rows = toy_rows()
        rows[1]["DISPLAY_NAME"] = rows[0]["DISPLAY_NAME"]        # duplicate display name
        root = write_toy_taxonomy(tmp_path / "t", rows=rows)
        table = load_primitive_table(root, TAX_NAME, "headline")  # joins fine on the path
        assert table.n_primitives == 8

    def test_duplicate_primitive_key_raises(self, tmp_path: Path):
        rows = toy_rows()
        rows.append(dict(rows[0]))
        rows[-1]["TYPE"] = "other"                                # different path, same key
        root = write_toy_taxonomy(tmp_path / "t", rows=rows)
        with pytest.raises(ValueError, match="duplicate primitive key"):
            load_primitive_table(root, TAX_NAME, "headline")

    def test_non_uniform_k_raises(self, tmp_path: Path):
        root = write_toy_taxonomy(tmp_path / "t")
        p = root / f"{TAX_NAME}-primitive_headline_paraphrases.jsonl"
        recs = [json.loads(line) for line in p.read_text().splitlines()]
        recs[0]["paraphrases"] = recs[0]["paraphrases"][:1]
        p.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
        with pytest.raises(ValueError, match="non-uniform K"):
            load_primitive_table(root, TAX_NAME, "headline")

    def test_exclude_master(self, toy_root):
        table = load_primitive_table(toy_root, TAX_NAME, "headline", include_master=False)
        assert table.n_texts == 2 and len(table.texts) == 16


class TestChecks:
    def test_sanity_checks_pass(self, toy_table, toy_root):
        checks = sanity_checks(toy_table, toy_root)
        res = dict(zip(checks["check"].to_list(), checks["result"].to_list()))
        assert res["1. primitive key uniqueness"] == "PASS"
        assert res["5. polarity coherence"] == "PASS"
        assert res["6. bipolar mirror presence"] == "REVIEW"

    def test_polarity_incoherence_detected(self, tmp_path: Path, ):
        rows = toy_rows()
        rows[0]["POLARITY"] = "easing"          # SUB_TYPE stays 'stress'
        root = write_toy_taxonomy(tmp_path / "t", rows=rows)
        table = load_primitive_table(root, TAX_NAME, "headline")
        checks = sanity_checks(table, root)
        res = dict(zip(checks["check"].to_list(), checks["result"].to_list()))
        assert res["5. polarity coherence"] == "FAIL"

    def test_orphan_pole_candidates(self, toy_table):
        orphans = orphan_pole_candidates(toy_table)
        # macro|funding has stress + easing (a pair); firm|balance-sheet has only stress
        assert orphans.select(["reservoir", "dimension"]).rows() == [("firm", "balance-sheet")]


class TestScoringMatrix:
    @pytest.mark.parametrize("pooling", [PoolRule.MAX, PoolRule.MEAN, PoolRule.MEDIAN])
    @pytest.mark.parametrize("mode", [Correction.RAW, Correction.R1, Correction.R2])
    def test_pooled_score_equals_pool_of_cosines(self, toy_table, toy_embeddings, pooling, mode):
        rng = np.random.default_rng(3)
        mu = rng.normal(size=EMBEDDING_DIM).astype(np.float32) * 0.3
        mu_hat = mu / np.linalg.norm(mu)
        P_scoring = scoring_matrix(toy_embeddings, toy_table, mode, pooling, mu, mu_hat)
        H = apply_mode(unit_rows(rng, 7), mode, mu, mu_hat)
        S = primitive_scores(H, P_scoring, toy_table.n_primitives, toy_table.n_texts, pooling)
        assert S.shape == (7, 8) and S.dtype == np.float32 and S.flags.c_contiguous

        # same correction on the primitive texts, primitive-major, pooled by hand
        T = apply_mode(toy_embeddings, mode, mu, mu_hat).reshape(8, 3, EMBEDDING_DIM)
        C = np.einsum("hd,pkd->hpk", H, T)
        ref = {PoolRule.MAX: C.max(-1), PoolRule.MEAN: C.mean(-1),
               PoolRule.MEDIAN: np.median(C, -1)}[pooling]
        np.testing.assert_allclose(S, ref, atol=2e-6)

    def test_mean_matrix_is_not_renormalised(self, toy_table, toy_embeddings):
        P_scoring = scoring_matrix(toy_embeddings, toy_table, Correction.RAW, PoolRule.MEAN,
                                   None, None)
        assert P_scoring.shape == (8, EMBEDDING_DIM)
        assert (np.linalg.norm(P_scoring, axis=1) < 1.0).all()

    def test_text_major_permutation(self):
        v = np.arange(12, dtype=np.float32).reshape(6, 2)       # 3 primitives x 2 texts
        tm = to_text_major(v, 2)
        np.testing.assert_array_equal(tm[:3], v[[0, 2, 4]])
        np.testing.assert_array_equal(tm[3:], v[[1, 3, 5]])

    def test_shape_mismatch_raises(self, toy_table, toy_embeddings):
        with pytest.raises(ValueError, match="shape"):
            scoring_matrix(toy_embeddings[:-1], toy_table, Correction.RAW, PoolRule.MAX, None, None)
