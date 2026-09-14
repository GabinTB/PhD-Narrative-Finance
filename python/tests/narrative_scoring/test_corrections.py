"""Tests for narrative_scoring.corrections."""
from __future__ import annotations

import numpy as np
import pytest

from narrative_scoring.corrections import Correction, apply_correction


def _random_unit_rows(n: int, dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, dim)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


class TestRaw:
    def test_identity(self):
        X = _random_unit_rows(10, 384)
        out, n_degenerate = apply_correction(X, Correction.RAW)
        np.testing.assert_array_equal(out, X)
        assert n_degenerate == 0

    def test_raw_does_not_require_mu(self):
        X = _random_unit_rows(5, 384)
        out, _ = apply_correction(X, Correction.RAW, mu=None, mu_hat=None)
        np.testing.assert_array_equal(out, X)


class TestR1:
    def test_output_is_unit_norm(self):
        X = _random_unit_rows(50, 384, seed=1)
        mu = X.mean(axis=0) * 0.3  # plausible pooled mean, not unit norm
        out, n_degenerate = apply_correction(X, Correction.R1, mu=mu)
        assert n_degenerate == 0
        norms = np.linalg.norm(out, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_requires_mu(self):
        X = _random_unit_rows(5, 384)
        with pytest.raises(ValueError, match="requires mu"):
            apply_correction(X, Correction.R1, mu=None)

    def test_degenerate_rows_zeroed_not_nan(self):
        dim = 8
        rng = np.random.default_rng(2)
        mu = rng.normal(size=dim).astype(np.float32)
        mu /= np.linalg.norm(mu)
        mu *= 0.5
        # A row exactly equal to mu after correction collapses to (near) zero.
        X = np.stack([mu.copy(), _random_unit_rows(1, dim, seed=3)[0]])
        out, n_degenerate = apply_correction(X, Correction.R1, mu=mu)
        assert n_degenerate == 1
        assert not np.any(np.isnan(out))
        np.testing.assert_array_equal(out[0], np.zeros(dim))
        assert np.linalg.norm(out[1]) == pytest.approx(1.0, abs=1e-5)


class TestR2:
    def test_output_is_unit_norm(self):
        X = _random_unit_rows(50, 384, seed=4)
        mu_hat = _random_unit_rows(1, 384, seed=5)[0]
        out, n_degenerate = apply_correction(X, Correction.R2, mu_hat=mu_hat)
        assert n_degenerate == 0
        norms = np.linalg.norm(out, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_output_orthogonal_to_mu_hat(self):
        X = _random_unit_rows(50, 384, seed=6)
        mu_hat = _random_unit_rows(1, 384, seed=7)[0]
        out, n_degenerate = apply_correction(X, Correction.R2, mu_hat=mu_hat)
        assert n_degenerate == 0
        proj = out @ mu_hat
        assert np.all(np.abs(proj) < 1e-5)

    def test_requires_mu_hat(self):
        X = _random_unit_rows(5, 384)
        with pytest.raises(ValueError, match="requires mu_hat"):
            apply_correction(X, Correction.R2, mu_hat=None)

    def test_degenerate_row_when_parallel_to_mu_hat(self):
        dim = 8
        mu_hat = _random_unit_rows(1, dim, seed=8)[0]
        # A row parallel to mu_hat has nothing left after projecting it out.
        X = np.stack([mu_hat.copy(), _random_unit_rows(1, dim, seed=9)[0]])
        out, n_degenerate = apply_correction(X, Correction.R2, mu_hat=mu_hat)
        assert n_degenerate == 1
        assert not np.any(np.isnan(out))
        np.testing.assert_array_equal(out[0], np.zeros(dim))
