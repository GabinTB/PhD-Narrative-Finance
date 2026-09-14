"""Tests for narrative_scoring.scoring.

No real data or model: exercises the pure-arithmetic pieces (score_chunk,
daily_stats, gate_with_source_tau) on hand-built matrices.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from narrative_scoring.descriptions import PoolingMode
from narrative_scoring.scoring import (
    daily_stats,
    gate_with_source_tau,
    score_chunk,
)


class TestScoreChunk:
    def test_centroid_is_matmul(self):
        X = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        D = np.array([[1.0, 0.0], [0.0, 1.0], [0.70710678, 0.70710678]], dtype=np.float32)
        S = score_chunk(X, D, PoolingMode.CENTROID)
        expected = X @ D.T
        np.testing.assert_allclose(S, expected, atol=1e-6)

    def test_max_pooling(self):
        X = np.array([[1.0, 0.0]], dtype=np.float32)
        # primitive 0 has two paraphrases: one aligned with X, one orthogonal
        D = np.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float32)  # (n_prim=1, K=2, dim=2)
        S = score_chunk(X, D, PoolingMode.MAX)
        assert S.shape == (1, 1)
        assert S[0, 0] == pytest.approx(1.0)

    def test_median_pooling(self):
        X = np.array([[1.0, 0.0]], dtype=np.float32)
        D = np.array([[[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]]], dtype=np.float32)  # K=3
        S = score_chunk(X, D, PoolingMode.MEDIAN)
        # dot products: 1.0, 0.0, 0.6 -> median = 0.6
        assert S[0, 0] == pytest.approx(0.6)


class TestDailyStats:
    def test_hand_built_matrix(self):
        primitives = ["p0", "p1", "p2"]
        # 3 headlines x 3 primitives; p2 is never retained (all zero)
        S = np.array(
            [
                [0.9, 0.0, 0.0],
                [0.8, 0.5, 0.0],
                [0.0, 0.7, 0.0],
            ]
        )
        d = date(2008, 1, 15)
        df = daily_stats([S], primitives, d)

        row = {r["PRIMITIVE"]: r for r in df.iter_rows(named=True)}
        assert row["p0"]["SUPPORT"] == 2
        assert row["p0"]["INTENSITY"] == pytest.approx((0.9 + 0.8) / 2)
        assert row["p0"]["PEAK"] == pytest.approx(0.9)

        assert row["p1"]["SUPPORT"] == 2
        assert row["p1"]["INTENSITY"] == pytest.approx((0.5 + 0.7) / 2)
        assert row["p1"]["PEAK"] == pytest.approx(0.7)

        # p2: zero support -> INTENSITY=0.0, SUPPORT=0, PEAK=0.0, never null
        assert row["p2"]["SUPPORT"] == 0
        assert row["p2"]["INTENSITY"] == pytest.approx(0.0)
        assert row["p2"]["PEAK"] == pytest.approx(0.0)
        assert df["INTENSITY"].null_count() == 0
        assert df["SUPPORT"].null_count() == 0
        assert df["PEAK"].null_count() == 0

    def test_fully_gated_day_yields_zeros_not_nulls(self):
        primitives = ["p0", "p1"]
        S = np.zeros((5, 2))  # every headline was gated to zero
        df = daily_stats([S], primitives, date(2008, 2, 1))
        assert (df["SUPPORT"] == 0).all()
        assert (df["INTENSITY"] == 0.0).all()
        assert (df["PEAK"] == 0.0).all()
        assert df["INTENSITY"].null_count() == 0
        assert df["SUPPORT"].null_count() == 0
        assert df["PEAK"].null_count() == 0

    def test_chunking_matches_single_pass(self):
        rng = np.random.default_rng(0)
        n_head, n_prim = 97, 11
        S = rng.integers(0, 2, size=(n_head, n_prim)).astype(np.float64) * rng.integers(
            1, 5, size=(n_head, n_prim)
        )
        primitives = [f"p{i}" for i in range(n_prim)]
        d = date(2008, 3, 1)

        single = daily_stats([S], primitives, d)

        chunk_size = 13
        chunks = [S[i : i + chunk_size] for i in range(0, n_head, chunk_size)]
        chunked = daily_stats(chunks, primitives, d)

        np.testing.assert_array_equal(single["SUPPORT"].to_numpy(), chunked["SUPPORT"].to_numpy())
        np.testing.assert_allclose(
            single["INTENSITY"].to_numpy(), chunked["INTENSITY"].to_numpy(), atol=1e-6
        )
        np.testing.assert_allclose(single["PEAK"].to_numpy(), chunked["PEAK"].to_numpy(), atol=1e-6)


class TestGateWithSourceTau:
    def test_per_source_tau_applied(self):
        S = np.array([[0.9, 0.5], [0.9, 0.5]])
        sources = ["reuters", "blog"]
        tau_by_source = {"reuters": 0.5, "blog": 0.95}
        out = gate_with_source_tau(S, sources, tau_by_source, fallback_tau=0.0, rel_floor=0.65)
        # reuters row clears its tau (0.5) -> rel_floor gating applies
        np.testing.assert_array_equal(out[0], [0.9, 0.0])
        # blog row's max (0.9) is below its tau (0.95) -> fully gated
        np.testing.assert_array_equal(out[1], [0.0, 0.0])

    def test_unseen_source_uses_fallback(self):
        S = np.array([[0.9, 0.5]])
        out = gate_with_source_tau(S, ["unknown"], {}, fallback_tau=0.4, rel_floor=0.65)
        np.testing.assert_array_equal(out[0], [0.9, 0.0])
