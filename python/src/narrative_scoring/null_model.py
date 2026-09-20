"""The F0 null-model rejection floor.

Per-headline trim
------------------
For a headline with score vector s over n_prim aspects, discard the top
trim_frac (default 0.10) and keep the rest as null draws::

    n_keep = n_prim - ceil(trim_frac * n_prim)
    null_draws_for_this_headline = np.partition(s, n_keep)[:n_keep]

The per-headline trim is what makes contamination impossible: polluting the
null would require a single headline to be genuinely relevant to more than
10% of the taxonomy, which does not occur.

Pooling
-------
Null draws accumulate in a lagged expanding window keyed by source.  "Lagged"
means the null used to score date t is built only from headlines strictly
before t - delay, mirroring mu_asof.  Pooled draws per source are kept as a
reservoir-sampled array capped at ``cap`` values so memory stays bounded;
reservoir sampling keeps the sample unbiased regardless of stream length.

Effective number of independent draws
--------------------------------------
A headline matching nothing still has a best-aspect score equal to the max
over many *correlated* null draws.  The correlation between the null scores
against aspects i and j equals cos(d_i, d_j), so the relevant correlation
matrix is the description Gram matrix::

    G       = D @ D.T                      # (n_prim, n_prim), D rows unit-norm
    lambdas = np.linalg.eigvalsh(G)        # symmetric, use eigvalsh not eig
    lambdas = lambdas[lambdas > 0]
    N_eff   = (lambdas.sum() ** 2) / (lambdas ** 2).sum()

N_eff is a property of the taxonomy AND the correction, because the
correction changes D.  It must be recomputed per (taxonomy version, pooling
mode, correction) combination -- and, since the correction is time-varying,
per scoring date too.  Never reuse a value measured under one correction (or
one day's mu_t) when scoring under another.

The floor
---------
::

    tau = np.quantile(F0, (1 - alpha) ** (1 / N_eff))     # alpha = 0.01 default

The Gaussian cross-check ``mu_F0 + z * sigma_F0`` (z = norm.ppf((1-alpha)**(1/N_eff)))
is computed for diagnostics only -- the empirical quantile is always tau.

The gate
--------
::

    s_max = s.max()
    if s_max < tau:
        retained = zeros_like(s)                              # unassigned
    else:
        retained = np.where(s >= rel_floor * s_max, s, 0.0)    # rel_floor = 0.65
"""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import numpy as np
from scipy.stats import norm


@dataclass
class NullModel:
    tau: float
    n_eff: float
    alpha: float
    trim_frac: float
    rel_floor: float
    n_draws: int
    source: str | None
    gaussian_tau: float     # diagnostic cross-check only


# ---------------------------------------------------------------------------
# Effective N
# ---------------------------------------------------------------------------

def compute_n_eff(D: np.ndarray) -> float:
    """D: (n_prim, dim) unit-norm rows. Returns the effective independent-draw count."""
    G = D @ D.T
    lambdas = np.linalg.eigvalsh(G)
    lambdas = lambdas[lambdas > 0]
    if lambdas.size == 0:
        return 0.0
    return float((lambdas.sum() ** 2) / (lambdas ** 2).sum())


# ---------------------------------------------------------------------------
# Per-headline trim
# ---------------------------------------------------------------------------

def _n_keep(n_prim: int, trim_frac: float) -> int:
    return n_prim - ceil(trim_frac * n_prim)


def trim_null_draws(s: np.ndarray, trim_frac: float = 0.10) -> np.ndarray:
    """s: (n_prim,) score vector for one headline. Returns the kept null draws."""
    s = np.asarray(s)
    n_prim = s.shape[0]
    n_keep = _n_keep(n_prim, trim_frac)
    if n_keep <= 0:
        return np.empty(0, dtype=s.dtype)
    if n_keep >= n_prim:
        return s.copy()
    return np.partition(s, n_keep)[:n_keep]


def trim_null_draws_batch(S: np.ndarray, trim_frac: float = 0.10) -> np.ndarray:
    """Vectorized per-row version of trim_null_draws.

    S: (n_head, n_prim). Returns (n_head, n_keep): np.partition's per-row
    (last-axis) behavior applies the same per-headline trim as
    trim_null_draws, without a Python-level loop over headlines.
    """
    S = np.asarray(S)
    n_prim = S.shape[1]
    n_keep = _n_keep(n_prim, trim_frac)
    if n_keep <= 0:
        return np.empty((S.shape[0], 0), dtype=S.dtype)
    if n_keep >= n_prim:
        return S.copy()
    return np.partition(S, n_keep, axis=1)[:, :n_keep]


# ---------------------------------------------------------------------------
# Reservoir-sampled pooling
# ---------------------------------------------------------------------------

def build_null(
    scores_iter,
    trim_frac: float = 0.10,
    cap: int = 5_000_000,
    seed: int = 0,
) -> np.ndarray:
    """Pool trimmed null draws from an iterable of per-headline score vectors.

    ``scores_iter`` yields (n_prim,) score vectors, one per headline. Draws
    are reservoir-sampled into a capped array so memory stays bounded
    regardless of stream length, while remaining an unbiased sample of the
    full stream.
    """
    pool = ReservoirPool(cap=cap, seed=seed)
    for s in scores_iter:
        pool.add(trim_null_draws(np.asarray(s), trim_frac=trim_frac))
    return pool.draws


class ReservoirPool:
    """Incremental reservoir sampler over a capped float64 buffer.

    Used by scoring.py to maintain one pool per source across the run,
    fed batches of trimmed draws day by day (rather than a single
    ``scores_iter`` pass), while remaining an unbiased sample of everything
    ever added.

    Keyed (priority) reservoir sampling: every element ever added draws an iid
    Uniform(0,1) key, and the reservoir is always the ``cap`` elements with the
    largest keys.  Because the keys are iid, the retained set is a uniformly
    random ``cap``-subset of the whole stream -- the same guarantee as the
    classic sequential Algorithm R, but a batch merges with vectorized numpy
    instead of one ``rng.integers()`` call per scalar.  That per-scalar loop was
    the dominant cost of the whole pipeline at realistic scale: one month of
    headlines yields ~3.6e8 trimmed draws, ~15s per day of Python-level RNG
    calls versus ~0.2s for the batch merge below.

    Two facts make the batched form exact rather than an approximation:

      - ``top_cap(A | B) == top_cap(top_cap(A) | B)`` when ``|A| >= cap``, so
        merging batch-by-batch matches one pass over the concatenated stream.
      - Once the reservoir is full, an incoming key below its minimum key is
        dominated by ``cap`` elements already held, so it cannot enter the
        top ``cap`` and is discarded without materializing it.
    """

    def __init__(self, cap: int = 5_000_000, seed: int = 0):
        self.cap = cap
        self._rng = np.random.default_rng(seed)
        self._reservoir = np.empty(0, dtype=np.float64)
        self._keys = np.empty(0, dtype=np.float64)
        self.n_filled = 0
        self.n_seen = 0

    def add(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64).ravel()
        n_new = values.shape[0]
        if n_new == 0:
            return
        self.n_seen += n_new

        keys = self._rng.random(n_new)
        if self.n_filled >= self.cap:
            survived = keys > self._keys.min()
            values, keys = values[survived], keys[survived]

        values = np.concatenate([self._reservoir, values])
        keys = np.concatenate([self._keys, keys])
        if values.shape[0] > self.cap:
            top = np.argpartition(keys, -self.cap)[-self.cap:]
            values, keys = values[top], keys[top]

        self._reservoir, self._keys = values, keys
        self.n_filled = values.shape[0]

    @property
    def draws(self) -> np.ndarray:
        return self._reservoir


# ---------------------------------------------------------------------------
# tau / gate
# ---------------------------------------------------------------------------

def compute_tau(F0: np.ndarray, n_eff: float, alpha: float = 0.01) -> float:
    if n_eff <= 0:
        raise ValueError(f"n_eff must be positive, got {n_eff}")
    q = (1 - alpha) ** (1.0 / n_eff)
    return float(np.quantile(F0, q))


def compute_gaussian_tau(F0: np.ndarray, n_eff: float, alpha: float = 0.01) -> float:
    """Diagnostic-only Gaussian cross-check for tau; never used as the gate."""
    if n_eff <= 0:
        raise ValueError(f"n_eff must be positive, got {n_eff}")
    z = norm.ppf((1 - alpha) ** (1.0 / n_eff))
    return float(np.mean(F0) + z * np.std(F0))


def fit_null_model(
    F0: np.ndarray,
    D: np.ndarray,
    *,
    alpha: float = 0.01,
    trim_frac: float = 0.10,
    rel_floor: float = 0.65,
    source: str | None = None,
) -> NullModel:
    """Convenience: fit tau/N_eff/gaussian cross-check from pooled draws + D."""
    n_eff = compute_n_eff(D)
    return NullModel(
        tau=compute_tau(F0, n_eff, alpha=alpha),
        n_eff=n_eff,
        alpha=alpha,
        trim_frac=trim_frac,
        rel_floor=rel_floor,
        n_draws=int(F0.shape[0]),
        source=source,
        gaussian_tau=compute_gaussian_tau(F0, n_eff, alpha=alpha),
    )


def apply_gate(S: np.ndarray, tau: float, rel_floor: float = 0.65) -> np.ndarray:
    """Gate a score vector (n_prim,) or matrix (n_head, n_prim) against tau.

        s_max = s.max()
        if s_max < tau: retained = zeros
        else:           retained = where(s >= rel_floor * s_max, s, 0)

    Applied row-wise when S is 2D.
    """
    S = np.asarray(S)
    s_max = S.max(axis=-1, keepdims=True)
    above = s_max >= tau
    kept = np.where(S >= rel_floor * s_max, S, 0.0)
    return np.where(above, kept, 0.0)
