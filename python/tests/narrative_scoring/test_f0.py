"""F0 machinery: trim, subsampling, spectrum / N_eff, tau formulas, t-digest, Welford."""
from __future__ import annotations

from math import ceil

import numpy as np
import pytest

from narrative_scoring.f0 import (
    TDigest,
    Welford,
    compute_gaussian_tau,
    compute_n_eff,
    compute_tau,
    gaussian_tau,
    gram_spectrum,
    n_keep,
    sample_null_draws,
    tail_probability,
    trim_null_draws,
    trim_null_draws_batch,
    trim_threshold,
)


class TestTrim:
    def test_keeps_exact_count(self):
        s = np.random.default_rng(0).normal(size=37)
        kept = trim_null_draws(s, trim_frac=0.10)
        assert kept.shape[0] == 37 - ceil(0.10 * 37) == n_keep(37, 0.10)

    def test_drops_the_top_values(self):
        s = np.arange(10, dtype=np.float64)
        kept = trim_null_draws(s, trim_frac=0.10)
        assert 9.0 not in kept and kept.shape[0] == 9

    def test_batch_matches_single_row(self):
        S = np.random.default_rng(1).normal(size=(5, 41))
        batch = trim_null_draws_batch(S, trim_frac=0.10)
        for i in range(5):
            np.testing.assert_allclose(np.sort(batch[i]), np.sort(trim_null_draws(S[i], 0.10)))

    def test_threshold_matches_batch_semantics(self):
        S = np.random.default_rng(2).normal(size=(50, 1508)).astype(np.float32)
        thr = trim_threshold(S, 0.10)
        kept = trim_null_draws_batch(S, 0.10)
        # everything strictly below the threshold is in the kept set, and the kept set
        # never contains anything above it
        for i in range(50):
            assert kept[i].max() <= thr[i]
            assert (S[i] < thr[i]).sum() <= kept.shape[1]

    def test_sample_null_draws_is_uniform_over_kept_and_seeded(self):
        rng = np.random.default_rng(3)
        S = rng.normal(size=(200, 1508)).astype(np.float32)
        thr = trim_threshold(S, 0.10)
        d1 = sample_null_draws(S, thr, 64, np.random.default_rng(7))
        d2 = sample_null_draws(S, thr, 64, np.random.default_rng(7))
        np.testing.assert_array_equal(d1, d2)
        assert d1.dtype == np.float64
        assert 0.85 * 200 * 64 < d1.size < 0.95 * 200 * 64          # ~90% acceptance
        kept = trim_null_draws_batch(S, 0.10)
        assert d1.max() <= kept.max() and (d1 < np.repeat(thr, 64).max()).all()


class TestSpectrum:
    def test_orthonormal_gives_n_prim(self):
        assert compute_n_eff(np.eye(20)) == pytest.approx(20.0, rel=1e-6)

    def test_identical_rows_give_one(self):
        v = np.zeros(8)
        v[0] = 1.0
        spec = gram_spectrum(np.tile(v, (20, 1)))
        assert spec.n_eff == pytest.approx(1.0, rel=1e-6)
        assert spec.lambda1_share == pytest.approx(1.0)
        assert spec.effective_rank == pytest.approx(1.0, abs=1e-6)
        assert spec.eigenvalues.shape == (20,)

    def test_effective_rank_bounds(self):
        rng = np.random.default_rng(4)
        D = rng.normal(size=(60, 30))
        D /= np.linalg.norm(D, axis=1, keepdims=True)
        spec = gram_spectrum(D)
        assert 1.0 <= spec.n_eff <= spec.effective_rank <= 30.0 + 1e-9


class TestTau:
    def test_tail_probability_and_tau_rise_with_n_eff(self):
        F0 = np.random.default_rng(5).normal(size=100_000)
        assert tail_probability(0.01, 1.0) < tail_probability(0.01, 1000.0)
        assert compute_tau(F0, 1.0) < compute_tau(F0, 1000.0)
        with pytest.raises(ValueError):
            tail_probability(0.01, 0.0)

    def test_gaussian_tau_formula(self):
        from scipy.stats import norm

        assert gaussian_tau(0.2, 0.05, 14.0, 0.01) == pytest.approx(
            0.2 + 0.05 * norm.isf(0.01 / 14.0))
        F0 = np.random.default_rng(6).normal(0.2, 0.05, size=50_000)
        assert compute_gaussian_tau(F0, 14.0) == pytest.approx(
            gaussian_tau(F0.mean(), F0.std(), 14.0), rel=1e-9)


class TestWelford:
    def test_streaming_equals_batch(self):
        x = np.random.default_rng(8).normal(0.3, 0.1, size=100_003)
        w = Welford()
        for i in range(0, x.size, 999):
            w.add(x[i:i + 999])
        assert w.count == x.size
        assert w.mean == pytest.approx(x.mean(), rel=1e-12)
        assert w.std == pytest.approx(x.std(), rel=1e-10)
        parts = [Welford(), Welford()]
        parts[0].add(x[:50_000])
        parts[1].add(x[50_000:])
        m = Welford.merged(parts)
        assert m.count == x.size and m.mean == pytest.approx(x.mean(), rel=1e-12)
        assert m.std == pytest.approx(x.std(), rel=1e-10)

    def test_empty(self):
        w = Welford()
        assert w.count == 0 and np.isnan(w.std)
        w.add(np.empty(0))
        assert w.count == 0


class TestTDigest:
    def test_tail_quantiles_close_to_exact(self):
        x = np.random.default_rng(9).normal(0.25, 0.08, size=2_000_000)
        d = TDigest(2000)
        for i in range(0, x.size, 300_000):
            d.add(x[i:i + 300_000])
        for q in (0.5, 0.99, 0.999, 0.9993, 0.9999):
            exact = np.quantile(x, q)
            assert abs(d.cdf(exact) - q) < 3e-5, q
            assert abs(d.quantile(q) - exact) < 2e-3, q
        assert d.count == x.size and d.n_centroids < 2 * 2000
        assert d.min == x.min() and d.max == x.max()

    def test_deterministic_and_merge_identical_bytes(self):
        x = np.random.default_rng(10).normal(size=500_000)
        a, b = TDigest(500), TDigest(500)
        for i in range(0, x.size, 70_000):
            a.add(x[i:i + 70_000])
            b.add(x[i:i + 70_000])
        a.flush(), b.flush()
        assert a.means.tobytes() == b.means.tobytes()
        assert a.weights.tobytes() == b.weights.tobytes()
        parts = []
        for i in range(5):
            p = TDigest(500)
            p.add(x[i * 100_000:(i + 1) * 100_000])
            parts.append(p)
        m1, m2 = TDigest.merge(parts), TDigest.merge(parts)
        assert m1.means.tobytes() == m2.means.tobytes()
        assert m1.count == x.size
        assert abs(m1.quantile(0.999) - np.quantile(x, 0.999)) < 5e-3

    def test_merge_order_matters_and_is_recorded_by_caller(self):
        x = np.random.default_rng(11).normal(size=200_000)
        p1, p2 = TDigest(200), TDigest(200)
        p1.add(x[:100_000])
        p2.add(x[100_000:])
        ab, ba = TDigest.merge([p1, p2]), TDigest.merge([p2, p1])
        assert ab.count == ba.count
        # both are valid sketches of the same data
        assert abs(ab.quantile(0.99) - ba.quantile(0.99)) < 1e-2

    def test_roundtrip_arrays(self):
        d = TDigest(100)
        d.add(np.arange(1000, dtype=np.float64))
        d.flush()
        e = TDigest.from_arrays(d.means, d.weights, d.count, d.min, d.max, d.compression)
        assert e.quantile(0.5) == d.quantile(0.5) and e.count == 1000

    def test_empty_and_single(self):
        d = TDigest(100)
        assert np.isnan(d.quantile(0.5))
        d.add(np.array([0.7]))
        assert d.quantile(0.1) == 0.7 == d.quantile(0.99)
        with pytest.raises(ValueError):
            TDigest(100).add(np.array([np.nan]))
