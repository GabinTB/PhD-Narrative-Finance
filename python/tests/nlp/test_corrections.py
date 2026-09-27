"""Tests for nlp.corrections."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pytest

from nlp.corrections import (
    Correction,
    LookaheadError,
    apply_correction,
    apply_mode,
    count_degenerate,
    l2_normalise,
)


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


class TestApplyMode:
    """The hot-path variant must match apply_correction and never mutate its input."""

    @pytest.mark.parametrize("mode", [Correction.RAW, Correction.R1, Correction.R2])
    def test_matches_apply_correction_on_unit_and_non_unit_rows(self, mode):
        rng = np.random.default_rng(10)
        X = rng.normal(size=(30, 384)).astype(np.float32) * 3.0      # not unit norm
        mu = rng.normal(size=384).astype(np.float32) * 0.3
        mu_hat = mu / np.linalg.norm(mu)
        before = X.copy()
        got = apply_mode(X, mode, mu, mu_hat)
        ref, _ = apply_correction(l2_normalise(X), mode, mu=mu, mu_hat=mu_hat)
        np.testing.assert_allclose(got, ref, atol=1e-6)
        np.testing.assert_array_equal(X, before)
        np.testing.assert_allclose(np.linalg.norm(got, axis=1), 1.0, atol=1e-5)

    def test_r2_orthogonal_to_u_and_same_transform_both_sides(self):
        rng = np.random.default_rng(11)
        mu = rng.normal(size=384).astype(np.float32)
        u = mu / np.linalg.norm(mu)
        H = apply_mode(rng.normal(size=(20, 384)).astype(np.float32), Correction.R2, mu, u)
        P = apply_mode(rng.normal(size=(50, 384)).astype(np.float32), Correction.R2, mu, u)
        assert np.abs(H @ u).max() < 1e-5 and np.abs(P @ u).max() < 1e-5
        # the cosine between corrected sides is invariant to any shared u-component
        S = H @ P.T
        H2 = apply_mode(H + 0.4 * u[None, :], Correction.R2, mu, u)
        np.testing.assert_allclose(H2 @ P.T, S, atol=1e-5)

    def test_raw_does_not_mutate_and_zero_row_stays_zero(self):
        X = np.zeros((2, 8), dtype=np.float32)
        X[1, 0] = 2.0
        before = X.copy()
        out = apply_mode(X, Correction.RAW, None, None)
        np.testing.assert_array_equal(X, before)
        assert out is not X
        np.testing.assert_array_equal(out[0], np.zeros(8))
        assert out[1, 0] == 1.0

    def test_missing_mu_raises(self):
        X = np.ones((1, 8), dtype=np.float32)
        with pytest.raises(ValueError, match="requires mu"):
            apply_mode(X, Correction.R1, None, None)
        with pytest.raises(ValueError, match="requires mu_hat"):
            apply_mode(X, Correction.R2, None, None)


@dataclass
class _DatedRef:
    date: date
    mu: np.ndarray
    mu_hat: np.ndarray


class TestCorrectionObject:
    """``Correction.<mode>.correct`` is apply_mode behind a reference-aware interface."""

    def _inputs(self):
        rng = np.random.default_rng(12)
        X = rng.normal(size=(40, 384)).astype(np.float32)
        mu = rng.normal(size=384).astype(np.float32) * 0.3
        return X, mu, mu / np.linalg.norm(mu)

    @pytest.mark.parametrize("mode", list(Correction))
    def test_bitwise_equal_to_apply_mode(self, mode):
        X, mu, mu_hat = self._inputs()
        want = apply_mode(X, mode, mu, mu_hat)
        np.testing.assert_array_equal(mode.correct(X, (mu, mu_hat)), want)
        np.testing.assert_array_equal(
            mode.correct(X, _DatedRef(date(2020, 1, 1), mu, mu_hat)), want)

    def test_undated_returns_array_dated_returns_tuple(self):
        X, mu, mu_hat = self._inputs()
        out = Correction.R2.correct(X, (mu, mu_hat))
        assert isinstance(out, np.ndarray)
        day = date(2020, 3, 2)
        rows, got_day = Correction.R2.correct(X, (mu, mu_hat), as_of=day)
        assert got_day == day
        np.testing.assert_array_equal(rows, out)

    def test_future_dated_reference_raises(self):
        X, mu, mu_hat = self._inputs()
        ref = _DatedRef(date(2020, 3, 3), mu, mu_hat)
        with pytest.raises(LookaheadError):
            Correction.R1.correct(X, ref, as_of=date(2020, 3, 2))
        # same-day and earlier references are allowed
        Correction.R1.correct(X, ref, as_of=date(2020, 3, 3))

    def test_raw_needs_no_reference_and_ignores_it(self):
        X, mu, mu_hat = self._inputs()
        np.testing.assert_array_equal(Correction.RAW.correct(X), l2_normalise(X))

    def test_missing_reference_raises_for_r1_r2(self):
        X, _, _ = self._inputs()
        with pytest.raises(ValueError, match="requires mu"):
            Correction.R1.correct(X)
        with pytest.raises(TypeError):
            Correction.R2.correct(X, reference=object())

    def test_degenerate_rows_counted(self):
        dim = 8
        mu_hat = np.zeros(dim, dtype=np.float32)
        mu_hat[0] = 1.0
        X = np.stack([mu_hat, _random_unit_rows(1, dim, seed=13)[0]])
        out = Correction.R2.correct(X, (mu_hat, mu_hat))
        assert count_degenerate(out) == 1
        np.testing.assert_array_equal(out[0], np.zeros(dim))
