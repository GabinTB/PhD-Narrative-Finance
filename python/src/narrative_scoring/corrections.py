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

Critical consistency rule (see narrative_scoring/__init__.py and scoring.py):
descriptions are static but the correction is time-varying (mu_t from
mu_asof).  At each scoring date t, the SAME mu_t/mu_hat_t must be applied to
the description matrix as to the headlines -- never a fixed reference mean
for descriptions alongside a time-varying one for headlines.
"""
from __future__ import annotations

from enum import Enum

import numpy as np

_DEGENERATE_NORM_TOL = 1e-6


class Correction(str, Enum):
    RAW = "raw"
    R1 = "r1"
    R2 = "r2"


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
