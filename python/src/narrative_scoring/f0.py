"""The F0 null model: trim, pooled null distribution, N_eff, and the floor tau.

Per-headline trim (unchanged from the validated behaviour)
----------------------------------------------------------
For a headline with primitive-score vector s over P primitives (post-correction,
post-pooling), the top ``trim_frac`` (default 10%) are discarded and the rest
are null draws::

    n_keep = P - ceil(trim_frac * P)
    null_draws = np.partition(s, n_keep)[:n_keep]

Contamination would require one headline to be genuinely relevant to more
than 10% of the taxonomy, which does not occur.

Pooling
-------
The draws of each calendar month go into a bounded t-digest (``TDigest``)
plus Welford moments; the month partitions are merged, in chronological
order, by the tau_asof job (partitions.py, tau_asof.py). Draws are
subsampled per headline (``sample_null_draws``) before entering the digest:
uniformly over the kept draws, so the sample is unbiased for the month's
null distribution; ``n_draws_available`` vs ``n_draws_sampled`` are recorded
with every partition.

Effective number of independent draws
--------------------------------------
G = D D^T for the unit-norm representative primitive vectors under the
correction in use; N_eff = (sum lambda)^2 / sum lambda^2 (participation ratio
of the spectrum). It depends on (taxonomy, correction, pooling) and on the
mu row the correction uses, so the monthly tau job recomputes it at every
cutoff from the mu_asof row available then (warmup.py, tau_asof.py).

The floor
---------
::

    p_tail        = (1 - alpha) ** (1 / N_eff)
    tau_empirical = quantile(F0, p_tail)                     # the mechanism of record
    tau_gauss     = mean + std * norm.isf(alpha / N_eff)     # cross-check ONLY

Why a t-digest and not a histogram: p_tail is ~1 - 7e-4 for N_eff ~ 14 and
the recorded pool quantiles go to 0.9999. A fixed-width bucket histogram with
0.01 or 0.1 wide buckets puts the whole upper tail of a cosine distribution
(scores in ~[0.2, 0.6]) into a handful of buckets and cannot resolve a
quantile that sits 1e-4 from the top; the t-digest spends its centroids at
the tails (k1 scale function) and resolves it to ~1e-6 in q.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Iterable, Sequence

import numpy as np
from numba import njit
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Per-headline trim
# ---------------------------------------------------------------------------


def n_trim(n_prim: int, trim_frac: float) -> int:
    """Number of top scores dropped per headline: ceil(trim_frac * P)."""
    return int(ceil(trim_frac * n_prim))


def n_keep(n_prim: int, trim_frac: float) -> int:
    return n_prim - n_trim(n_prim, trim_frac)


def trim_null_draws(s: np.ndarray, trim_frac: float = 0.10) -> np.ndarray:
    """s: (P,) score vector for one headline. Returns the kept null draws."""
    s = np.asarray(s)
    keep = n_keep(s.shape[0], trim_frac)
    if keep <= 0:
        return np.empty(0, dtype=s.dtype)
    if keep >= s.shape[0]:
        return s.copy()
    return np.partition(s, keep)[:keep]


def trim_null_draws_batch(S: np.ndarray, trim_frac: float = 0.10) -> np.ndarray:
    """Vectorised per-row trim: (n_head, P) -> (n_head, n_keep)."""
    S = np.asarray(S)
    keep = n_keep(S.shape[1], trim_frac)
    if keep <= 0:
        return np.empty((S.shape[0], 0), dtype=S.dtype)
    if keep >= S.shape[1]:
        return S.copy()
    return np.partition(S, keep, axis=1)[:, :keep]


def trim_threshold(S: np.ndarray, trim_frac: float = 0.10) -> np.ndarray:
    """Per-row value of the n_trim-th largest score: draws strictly below it are kept.

    ``sample_null_draws`` uses this instead of materialising the (n_head,
    n_keep) trimmed matrix. Exact ties at the threshold are dropped rather
    than kept, which differs from ``trim_null_draws_batch`` only on tied
    float32 scores at the cut.
    """
    S = np.asarray(S)
    k = n_trim(S.shape[1], trim_frac)
    if k <= 0:
        return np.full(S.shape[0], np.inf, dtype=S.dtype)
    return np.partition(S, S.shape[1] - k, axis=1)[:, S.shape[1] - k]


def sample_null_draws(
    S: np.ndarray, thresholds: np.ndarray, draws_per_headline: int, rng: np.random.Generator,
) -> np.ndarray:
    """Uniform subsample of each headline's kept null draws.

    ``draws_per_headline`` columns are drawn uniformly per row (with
    replacement) and those at or above the row's trim threshold are rejected,
    which leaves a uniform sample of the kept set. Returns the accepted draws
    as float64, row-major order (deterministic for a given ``rng`` state).
    """
    n, p = S.shape
    cols = rng.integers(0, p, size=(n, draws_per_headline))
    vals = np.take_along_axis(S, cols, axis=1)
    keep = vals < thresholds[:, None]
    return vals[keep].astype(np.float64)


# ---------------------------------------------------------------------------
# Spectrum, N_eff, tau
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Spectrum:
    """Eigen-spectrum of the primitive Gram matrix and the derived scalars."""

    eigenvalues: np.ndarray       # descending, all of them (including ~0)
    n_eff: float                  # participation ratio (sum l)^2 / sum l^2
    effective_rank: float         # exp(entropy of the normalised spectrum)
    lambda1_share: float          # l_1 / sum l


def gram_spectrum(D: np.ndarray) -> Spectrum:
    """D: (n_prim, dim) unit-norm rows. Eigendecomposition of G = D D^T."""
    G = D.astype(np.float64) @ D.astype(np.float64).T
    lam = np.linalg.eigvalsh(G)[::-1]
    pos = lam[lam > 0]
    if pos.size == 0:
        return Spectrum(lam, 0.0, 0.0, float("nan"))
    p = pos / pos.sum()
    return Spectrum(
        eigenvalues=lam,
        n_eff=float(pos.sum() ** 2 / (pos ** 2).sum()),
        effective_rank=float(np.exp(-(p * np.log(p)).sum())),
        lambda1_share=float(pos[0] / pos.sum()),
    )


def compute_n_eff(D: np.ndarray) -> float:
    """D: (n_prim, dim) unit-norm rows. Effective independent-draw count."""
    return gram_spectrum(D).n_eff


def tail_probability(alpha: float, n_eff: float) -> float:
    if n_eff <= 0:
        raise ValueError(f"n_eff must be positive, got {n_eff}")
    return (1.0 - alpha) ** (1.0 / n_eff)


def compute_tau(F0: np.ndarray, n_eff: float, alpha: float = 0.01) -> float:
    """Empirical tau from an array of null draws (reference / tests)."""
    return float(np.quantile(F0, tail_probability(alpha, n_eff)))


def gaussian_tau(mean: float, std: float, n_eff: float, alpha: float = 0.01) -> float:
    """mean + std * norm.isf(alpha / N_eff): a robustness cross-check, never the gate."""
    if n_eff <= 0:
        raise ValueError(f"n_eff must be positive, got {n_eff}")
    return float(mean + std * norm.isf(alpha / n_eff))


def compute_gaussian_tau(F0: np.ndarray, n_eff: float, alpha: float = 0.01) -> float:
    return gaussian_tau(float(np.mean(F0)), float(np.std(F0)), n_eff, alpha)


# ---------------------------------------------------------------------------
# Welford moments (mergeable)
# ---------------------------------------------------------------------------


@dataclass
class Welford:
    """Streaming count / mean / M2, mergeable with Chan's formula."""

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def add(self, values: np.ndarray) -> None:
        v = np.asarray(values, dtype=np.float64).ravel()
        n = int(v.size)
        if n == 0:
            return
        self.merge(Welford(n, float(v.mean()), float(((v - v.mean()) ** 2).sum())))

    def merge(self, other: "Welford") -> None:
        if other.count == 0:
            return
        if self.count == 0:
            self.count, self.mean, self.m2 = other.count, other.mean, other.m2
            return
        n = self.count + other.count
        delta = other.mean - self.mean
        self.m2 = self.m2 + other.m2 + delta * delta * self.count * other.count / n
        self.mean = self.mean + delta * other.count / n
        self.count = n

    @property
    def variance(self) -> float:
        return self.m2 / self.count if self.count else float("nan")

    @property
    def std(self) -> float:
        return float(np.sqrt(self.variance)) if self.count else float("nan")

    @staticmethod
    def merged(parts: Iterable["Welford"]) -> "Welford":
        out = Welford()
        for p in parts:
            out.merge(p)
        return out


# ---------------------------------------------------------------------------
# t-digest (Dunning & Ertl's merging digest, k1 scale function), deterministic
# ---------------------------------------------------------------------------


@njit(cache=True)
def _k_scale(q: float, compression: float) -> float:
    return compression / (2.0 * np.pi) * np.arcsin(2.0 * q - 1.0)


@njit(cache=True)
def _k_inverse(k: float, compression: float) -> float:
    x = np.sin(k * 2.0 * np.pi / compression)
    q = (x + 1.0) / 2.0
    if q < 0.0:
        return 0.0
    if q > 1.0:
        return 1.0
    return q


@njit(cache=True)
def _compress(means: np.ndarray, weights: np.ndarray, compression: float):
    """Greedy merge of SORTED (means, weights) under the k1 size bound. Deterministic."""
    n = means.shape[0]
    out_m = np.empty(n, dtype=np.float64)
    out_w = np.empty(n, dtype=np.float64)
    total = 0.0
    for i in range(n):
        total += weights[i]
    k = 0
    w_done = 0.0
    cur_sum = means[0] * weights[0]
    cur_w = weights[0]
    q_limit = _k_inverse(_k_scale(0.0, compression) + 1.0, compression)
    for i in range(1, n):
        q_right = (w_done + cur_w + weights[i]) / total
        if q_right <= q_limit:
            cur_sum += means[i] * weights[i]
            cur_w += weights[i]
        else:
            out_m[k] = cur_sum / cur_w
            out_w[k] = cur_w
            k += 1
            w_done += cur_w
            q_limit = _k_inverse(_k_scale(w_done / total, compression) + 1.0, compression)
            cur_sum = means[i] * weights[i]
            cur_w = weights[i]
    out_m[k] = cur_sum / cur_w
    out_w[k] = cur_w
    k += 1
    return out_m[:k], out_w[:k]


class TDigest:
    """Bounded-memory quantile sketch with tail-focused resolution.

    Values are buffered and folded into the centroid set in sorted order;
    nothing here uses randomness, so two digests built from the same values
    in the same order are identical, and ``merge`` of the same partitions in
    the same order is identical too. Memory is O(compression).
    """

    __slots__ = ("compression", "means", "weights", "count", "min", "max",
                 "_buffer", "_buffered", "buffer_size")

    def __init__(self, compression: float = 500.0, *, buffer_size: int = 1 << 20):
        if compression < 20:
            raise ValueError("compression must be >= 20")
        self.compression = float(compression)
        self.means = np.empty(0, dtype=np.float64)
        self.weights = np.empty(0, dtype=np.float64)
        self.count = 0
        self.min = np.inf
        self.max = -np.inf
        self._buffer: list[np.ndarray] = []
        self._buffered = 0
        self.buffer_size = buffer_size

    # -- construction --------------------------------------------------------

    def add(self, values: np.ndarray) -> None:
        v = np.asarray(values, dtype=np.float64).ravel()
        if v.size == 0:
            return
        if not np.isfinite(v).all():
            raise ValueError("non-finite null draw")
        self._buffer.append(v)
        self._buffered += v.size
        if self._buffered >= self.buffer_size:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        v = np.sort(np.concatenate(self._buffer))
        self._buffer, self._buffered = [], 0
        self.count += int(v.size)
        self.min = min(self.min, float(v[0]))
        self.max = max(self.max, float(v[-1]))
        self._fold(v, np.ones(v.size, dtype=np.float64))

    def _fold(self, means: np.ndarray, weights: np.ndarray) -> None:
        m = np.concatenate([self.means, means])
        w = np.concatenate([self.weights, weights])
        order = np.argsort(m, kind="stable")
        self.means, self.weights = _compress(m[order], w[order], self.compression)

    @classmethod
    def from_arrays(cls, means: np.ndarray, weights: np.ndarray, count: int,
                    vmin: float, vmax: float, compression: float) -> "TDigest":
        d = cls(compression)
        d.means = np.asarray(means, dtype=np.float64).copy()
        d.weights = np.asarray(weights, dtype=np.float64).copy()
        d.count, d.min, d.max = int(count), float(vmin), float(vmax)
        return d

    @classmethod
    def merge(cls, digests: Sequence["TDigest"], compression: float | None = None) -> "TDigest":
        """Merge in the order given (callers pass chronological order)."""
        if not digests:
            raise ValueError("nothing to merge")
        comp = compression or digests[0].compression
        out = cls(comp)
        for d in digests:
            d.flush()
            if d.count == 0:
                continue
            out._fold(d.means, d.weights)
            out.count += d.count
            out.min, out.max = min(out.min, d.min), max(out.max, d.max)
        return out

    # -- queries -------------------------------------------------------------

    def quantile(self, q: float) -> float:
        return float(self.quantiles(np.asarray([q]))[0])

    def quantiles(self, qs: np.ndarray) -> np.ndarray:
        self.flush()
        qs = np.asarray(qs, dtype=np.float64)
        if self.count == 0:
            return np.full(qs.shape, np.nan)
        if self.means.size == 1:
            return np.full(qs.shape, self.means[0])
        centers = np.cumsum(self.weights) - self.weights / 2.0
        xs = np.concatenate([[0.0], centers, [float(self.count)]])
        ys = np.concatenate([[self.min], self.means, [self.max]])
        return np.interp(np.clip(qs, 0.0, 1.0) * self.count, xs, ys)

    def cdf(self, x: float) -> float:
        self.flush()
        if self.count == 0:
            return float("nan")
        centers = np.cumsum(self.weights) - self.weights / 2.0
        xs = np.concatenate([[self.min], self.means, [self.max]])
        ys = np.concatenate([[0.0], centers, [float(self.count)]])
        return float(np.interp(x, xs, ys) / self.count)

    @property
    def n_centroids(self) -> int:
        self.flush()
        return int(self.means.size)
