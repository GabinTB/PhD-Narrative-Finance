"""Embedding corrections: RAW, R1 (mean centering), R2 (mean-direction removal).

X is (n, 384) unit-norm rows throughout.

    RAW:  X_out = X

    R1:   X_tilde = X - mu[None, :]                    # mu is the raw mean
                                                         # vector, NOT unit norm
          X_out   = X_tilde / ||X_tilde||_row

    R2:   proj    = X @ mu_hat                          # (n,) scalar projection
          X_tilde = X - proj[:, None] * mu_hat[None, :]
          X_out   = X_tilde / ||X_tilde||_row

Degenerate-row guard (R1/R2 only): a row whose ||X_tilde|| < 1e-6 is almost
entirely the mean component being removed; renormalizing it would amplify
numerical noise into an arbitrary unit vector. Such rows are zeroed, counted,
and returned to the caller -- never silently divided.

Critical consistency rule (see narrative_scoring/__init__.py and pipeline.py):
descriptions are static but the correction is time-varying (mu_t from
mu_asof).  At each scoring date t, the SAME mu_t/mu_hat_t must be applied to
the description matrix as to the headlines -- never a fixed reference mean
for descriptions alongside a time-varying one for headlines.
"""
from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Any, overload

import numpy as np

_DEGENERATE_NORM_TOL = 1e-6


class LookaheadError(ValueError):
    """A point-in-time value dated after the date it would be used for."""


class Correction(str, Enum):
    """RAW / R1 / R2. Each member is also the correction object itself.

    ``Correction.R2.correct(X, reference)`` applies the correction; the enum
    value ("raw", "r1", "r2") stays the serialised form used in configs.
    """

    RAW = "raw"
    R1 = "r1"
    R2 = "r2"

    @overload
    def correct(self, X: np.ndarray, reference: Any = ..., as_of: None = ...) -> np.ndarray: ...

    @overload
    def correct(self, X: np.ndarray, reference: Any = ...,
                as_of: date = ...) -> tuple[np.ndarray, date]: ...

    def correct(self, X: np.ndarray, reference: Any = None,
                as_of: date | None = None) -> np.ndarray | tuple[np.ndarray, date]:
        """Corrected unit rows of ``X`` (the :func:`apply_mode` transform).

        Args:
            X:         (n, d) embeddings, any float dtype; never mutated.
            reference: the reference vector. Either an object exposing ``mu``
                       (raw pooled vector, not unit norm) and ``mu_hat`` (unit
                       direction), optionally dated through ``date``
                       (e.g. ``ReferenceVector`` values, ``MuRecord``);
                       or a ``(mu, mu_hat)`` pair. Ignored by RAW, required by
                       R1 (uses ``mu``) and R2 (uses ``mu_hat``).
            as_of:     None for an undated correction. A date makes the call
                       point-in-time: a reference dated AFTER ``as_of`` raises
                       ``LookaheadError``, and the date is returned with the array.

        Returns:
            float32 (n, d) corrected rows (degenerate rows zeroed, see
            :func:`apply_mode`), or ``(rows, as_of)`` when ``as_of`` is given.
        """
        mu, mu_hat, ref_date = _unpack_reference(reference)
        if as_of is not None and ref_date is not None and ref_date > as_of:
            raise LookaheadError(f"reference dated {ref_date} used for {as_of}")
        out = apply_mode(X, self, mu, mu_hat)
        return out if as_of is None else (out, as_of)


def _unpack_reference(reference: Any) -> tuple[np.ndarray | None, np.ndarray | None,
                                                date | None]:
    """(mu, mu_hat, date) from a reference object, a (mu, mu_hat) pair, or None."""
    if reference is None:
        return None, None, None
    if isinstance(reference, tuple):
        if len(reference) != 2:
            raise ValueError("a reference pair must be (mu, mu_hat)")
        return reference[0], reference[1], None
    if not (hasattr(reference, "mu") and hasattr(reference, "mu_hat")):
        raise TypeError(f"reference must expose mu and mu_hat, got {type(reference).__name__}")
    return reference.mu, reference.mu_hat, getattr(reference, "date", None)


def count_degenerate(X_corrected: np.ndarray) -> int:
    """All-zero rows in a corrected output: the rows a correction zeroed.

    On :func:`apply_mode` / :meth:`Correction.correct` output every surviving
    row has unit norm, so an all-zero row is a degenerate one (corrected norm
    below 1e-6) -- or a row that was already all-zero in the input, which no
    correction can turn into a direction either.
    """
    return int((~np.any(X_corrected, axis=1)).sum())


def _renormalize_rows(X_tilde: np.ndarray) -> tuple[np.ndarray, int]:
    """Row-normalize, zeroing (and counting) rows whose norm is ~0."""
    norms = np.linalg.norm(X_tilde, axis=1)
    degenerate = norms < _DEGENERATE_NORM_TOL
    n_degenerate = int(degenerate.sum())

    out = np.zeros_like(X_tilde)
    ok = ~degenerate
    if ok.any():
        out[ok] = X_tilde[ok] / norms[ok, None]
    return out, n_degenerate


def apply_correction(
    X: np.ndarray,
    correction: Correction,
    mu: np.ndarray | None = None,
    mu_hat: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    """Returns (corrected, n_degenerate_rows_zeroed)."""
    X = np.asarray(X, dtype=np.float32)

    if correction is Correction.RAW:
        return X.copy(), 0

    if correction is Correction.R1:
        if mu is None:
            raise ValueError("R1 correction requires mu")
        X_tilde = X - np.asarray(mu, dtype=np.float32)[None, :]
        return _renormalize_rows(X_tilde)

    if correction is Correction.R2:
        if mu_hat is None:
            raise ValueError("R2 correction requires mu_hat")
        mu_hat = np.asarray(mu_hat, dtype=np.float32)
        proj = X @ mu_hat
        X_tilde = X - proj[:, None] * mu_hat[None, :]
        return _renormalize_rows(X_tilde)

    raise ValueError(f"unknown correction: {correction!r}")


def l2_normalise(X: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation to float32. Never mutates its input.

    A zero row stays zero (its norm is clamped to 1e-12 rather than divided
    through), so no NaN can be produced here.
    """
    X = np.asarray(X, dtype=np.float32)
    norms = np.sqrt(np.einsum("ij,ij->i", X, X, dtype=np.float32))
    np.maximum(norms, 1e-12, out=norms)
    return X / norms[:, None]


def apply_mode(
    X: np.ndarray,
    mode: Correction,
    mu: np.ndarray | None,
    mu_hat: np.ndarray | None,
) -> np.ndarray:
    """Step 1 in full: L2-normalise, apply raw / R1 / R2, renormalise.

    This is the scoring hot path's version of :func:`apply_correction`: same
    arithmetic and the same degenerate-row rule (a row whose corrected norm is
    below 1e-6 is zeroed, so it can never clear the F0 floor), but with two
    norm passes instead of three and no intermediate copies. ``X`` is never
    mutated. Callers apply it to headlines and to primitive texts with the
    SAME ``mu``/``mu_hat`` (the consistency rule in this module's docstring).
    """
    X = l2_normalise(X)
    if mode is Correction.RAW:
        return X
    if mode is Correction.R1:
        if mu is None:
            raise ValueError("R1 correction requires mu")
        X = X - np.asarray(mu, dtype=np.float32)[None, :]
    elif mode is Correction.R2:
        if mu_hat is None:
            raise ValueError("R2 correction requires mu_hat")
        u = np.asarray(mu_hat, dtype=np.float32)
        X = X - (X @ u)[:, None] * u[None, :]
    else:
        raise ValueError(f"unknown correction: {mode!r}")
    norms = np.sqrt(np.einsum("ij,ij->i", X, X, dtype=np.float32))
    degenerate = norms < _DEGENERATE_NORM_TOL
    np.maximum(norms, 1e-12, out=norms)
    X /= norms[:, None]
    if degenerate.any():
        X[degenerate] = 0.0
    return X
