"""Tests for narrative_scoring.null_model."""
from __future__ import annotations

from math import ceil

import numpy as np
import pytest

from narrative_scoring.null_model import (
    apply_gate,
    build_null,
    compute_n_eff,
    compute_tau,
    trim_null_draws,
    trim_null_draws_batch,
)


class TestComputeNEff:
    def test_orthonormal_gives_n_prim(self):
        n_prim = 20
        D = np.eye(n_prim, dtype=np.float64)
        n_eff = compute_n_eff(D)
        assert n_eff == pytest.approx(n_prim, rel=1e-6)

    def test_all_identical_gives_approximately_one(self):
        n_prim = 20
        v = np.zeros(8)
        v[0] = 1.0
        D = np.tile(v, (n_prim, 1))
        n_eff = compute_n_eff(D)
        assert n_eff == pytest.approx(1.0, rel=1e-6)


class TestTrim:
    def test_keeps_exact_count(self):
        n_prim = 37
        trim_frac = 0.10
        s = np.random.default_rng(0).normal(size=n_prim)
        kept = trim_null_draws(s, trim_frac=trim_frac)
        expected = n_prim - ceil(trim_frac * n_prim)
        assert kept.shape[0] == expected

    def test_drops_the_top_values(self):
        s = np.arange(10, dtype=np.float64)  # 0..9, top 10% (ceil(1)) = value 9
        kept = trim_null_draws(s, trim_frac=0.10)
        assert 9.0 not in kept
        assert kept.shape[0] == 9

    def test_batch_matches_single_row(self):
        rng = np.random.default_rng(1)
        S = rng.normal(size=(5, 41))
        batch = trim_null_draws_batch(S, trim_frac=0.10)
        for i in range(5):
            single = trim_null_draws(S[i], trim_frac=0.10)
            np.testing.assert_allclose(np.sort(batch[i]), np.sort(single))


class TestBuildNull:
    def test_pools_trimmed_draws(self):
        rng = np.random.default_rng(2)
        headlines = [rng.normal(size=30) for _ in range(100)]
        F0 = build_null(headlines, trim_frac=0.10, cap=10_000, seed=0)
        n_keep = 30 - ceil(0.10 * 30)
        assert F0.shape[0] == 100 * n_keep

    def test_respects_cap(self):
        rng = np.random.default_rng(3)
        headlines = [rng.normal(size=30) for _ in range(1000)]
        F0 = build_null(headlines, trim_frac=0.10, cap=500, seed=0)
        assert F0.shape[0] == 500


class TestTauAndGate:
    def test_tau_rises_with_n_eff(self):
        rng = np.random.default_rng(4)
        F0 = rng.normal(loc=0.0, scale=1.0, size=100_000)
        tau_low = compute_tau(F0, n_eff=1.0, alpha=0.01)
        tau_high = compute_tau(F0, n_eff=1000.0, alpha=0.01)
        assert tau_high > tau_low

    def test_gate_zeros_everything_below_tau(self):
        s = np.array([0.1, 0.2, 0.3, 0.4])
        out = apply_gate(s, tau=0.5, rel_floor=0.65)
        np.testing.assert_array_equal(out, np.zeros(4))

    def test_gate_applies_rel_floor_above_tau(self):
        s = np.array([0.5, 0.9, 1.0, 0.3])
        out = apply_gate(s, tau=0.8, rel_floor=0.65)
        # s_max = 1.0, rel_floor threshold = 0.65
        expected = np.array([0.0, 0.9, 1.0, 0.0])
        np.testing.assert_array_equal(out, expected)

    def test_gate_matrix_rowwise(self):
        S = np.array([[0.1, 0.2], [0.9, 1.0]])
        out = apply_gate(S, tau=0.5, rel_floor=0.65)
        np.testing.assert_array_equal(out[0], [0.0, 0.0])
        np.testing.assert_array_equal(out[1], [0.9, 1.0])
