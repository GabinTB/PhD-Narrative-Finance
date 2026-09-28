"""Sentimeter: text -> a sentiment distribution on the 41-point grid s = linspace(-1, 1, 41).

Every model family is stored the same way, as its output spread on one ordinal grid
(the RavenBERT grid), so scores, confidences and buckets are computed later, at load
time, with one set of rules for every model (``nlp.sentiment.ordinal_sql``):

    preprocess   ravenbert ``clean_text`` (the RavenPack demo cleaning); None -> ""
    backend      ``classify`` -> softmax probabilities (float64) in label order
    canonical    those probabilities permuted into the subclass's canonical order
    grid         the canonical classes spread on the grid (``to_grid``); rows of an
                 empty cleaned text are all NaN ("no output", stored as null)

The remap is owned by this class. A subclass declares only ``band_edges``, the
class-to-band edges on [-1, 1] (k + 1 increasing values from -1 to 1 for k classes,
in canonical order). Grid point s_j stands for its bin [s_j - 0.025, s_j + 0.025]
clipped to [-1, 1]; class c, whose band is [e_c, e_{c+1}], gives point j the mass

    p_c * overlap(bin_j, band_c) / (e_{c+1} - e_c)

i.e. its probability at a uniform density over its band. Mass is conserved, density
is equal within a band, and a point whose bin straddles an edge splits its mass
between the two bands (no point-membership rule, so no band is favoured by holding
more grid points). RavenBERT's 41 classes are the 41 bins (identity); FinBERT's
three bands are [-1, -1/3], [-1/3, 1/3], [1/3, 1].

At construction the subclass checks that the backend's classifier head is compatible
(label structure only, never a model name or weights): any 41-label head is
RavenBERT-compatible, any {negative, neutral, positive} head FinBERT-compatible.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from functools import cache
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from nlp.backends.base import Backend

if TYPE_CHECKING:
    from datalake import ModelCard

N_GRID = 41
GRID = np.linspace(-1.0, 1.0, N_GRID)                  # s_j, the ordinal grid
GRID_STEP = 2.0 / (N_GRID - 1)                          # 0.05
GRID_COLUMNS = tuple(f"P_{j:02d}" for j in range(N_GRID))
# bin of s_j = [GRID_EDGES[j], GRID_EDGES[j + 1]]: midpoints between grid values, clipped to
# [-1, 1] (so the end bins have half width); shared edges make adjacent bins meet exactly
GRID_EDGES = np.concatenate([[-1.0], (GRID[:-1] + GRID[1:]) / 2, [1.0]])
GRID_BIN_LO, GRID_BIN_HI = GRID_EDGES[:-1], GRID_EDGES[1:]


def clean_texts(texts: Sequence[str | None]) -> list[str]:
    """ravenbert's ``clean_text`` (RavenPack demo preprocessing); None -> ""."""
    from ravenbert.sentiment.model import clean_text

    return [clean_text(t) if t is not None else "" for t in texts]


def _grid_units(x: np.ndarray) -> np.ndarray:
    """Positions on [-1, 1] in grid units (s_j -> j), snapped to exact half-units when
    within 1e-9 of one, so bin and band edges that coincide compare exactly."""
    u = (np.asarray(x, dtype=np.float64) + 1.0) * ((N_GRID - 1) / 2.0)
    half = np.round(u * 2.0) / 2.0
    return np.where(np.abs(u - half) < 1e-9, half, u)


@cache
def band_matrix(edges: tuple[float, ...]) -> np.ndarray:
    """(k, 41) remap: row c spreads class c uniformly over its band [e_c, e_{c+1}].

    Computed in grid units, where every bin fully inside a band has overlap exactly 1:
    such points receive bit-identical masses (exact argmax ties within a band). Raises
    unless the edges run strictly increasing from -1 to 1. Each row sums to 1.
    """
    e = np.asarray(edges, dtype=np.float64)
    if e.ndim != 1 or e.size < 2 or e[0] != -1.0 or e[-1] != 1.0 or not np.all(np.diff(e) > 0):
        raise ValueError(f"band edges must increase strictly from -1 to 1, got {edges}")
    u = _grid_units(e)
    lo, hi = u[:-1, None], u[1:, None]
    j = np.arange(N_GRID, dtype=np.float64)[None, :]
    bin_lo, bin_hi = np.maximum(j - 0.5, 0.0), np.minimum(j + 0.5, N_GRID - 1.0)
    overlap = np.clip(np.minimum(hi, bin_hi) - np.maximum(lo, bin_lo), 0.0, None)
    M = overlap / (hi - lo)
    if not np.allclose(M.sum(axis=1), 1.0, rtol=0, atol=1e-12):
        raise ValueError(f"band edges {edges} do not tile [-1, 1]")
    return M


def to_grid(P: np.ndarray, edges: Sequence[float]) -> np.ndarray:
    """(N, k) class probabilities -> (N, 41) grid distribution (``band_matrix``)."""
    return np.asarray(P, dtype=np.float64) @ band_matrix(tuple(float(x) for x in edges))


class Sentimeter(ABC):
    """Parent class: preprocessing, inference, the grid remap, shapes, provenance.

    Args:
        backend: a classifier backend (local or TEI).
        clean:   apply ``clean_text`` before inference (the default, as in the
                 RavenPack demos).
    """

    name: ClassVar[str]                    # model family, e.g. "ravenbert"
    version: ClassVar[str]                 # version of this output spec
    repo: ClassVar[str]
    weights_public: ClassVar[bool]
    canonical_names: ClassVar[tuple[str, ...]]
    band_edges: ClassVar[tuple[float, ...]]   # k + 1 edges on [-1, 1], canonical order

    def __init__(self, backend: Backend, *, clean: bool = True) -> None:
        self.backend = backend
        self.clean = clean
        self.labels = backend.labels()
        self.order = np.asarray(self.canonical_order(self.labels), dtype=np.intp)
        if len(self.band_edges) != len(self.canonical_names) + 1:
            raise ValueError(f"{self.name}: {len(self.band_edges)} band edges for "
                             f"{len(self.canonical_names)} classes")
        band_matrix(tuple(self.band_edges))                  # validates the edges

    # -- subclass hook --------------------------------------------------------

    @classmethod
    @abstractmethod
    def canonical_order(cls, labels: list[str]) -> list[int]:
        """Label indices in canonical order; raise IncompatibleModelError if the
        head cannot be this model family."""

    # -- inference ------------------------------------------------------------

    def _prepare(self, texts: Sequence[str | None]) -> list[str]:
        return clean_texts(texts) if self.clean else ["" if t is None else t for t in texts]

    def canonical_batch(self, texts: Sequence[str | None]) -> np.ndarray:
        """(N, k) float64 class probabilities in canonical order (no empty-text rule)."""
        P = self.backend.classify(self._prepare(texts))
        return np.ascontiguousarray(P[:, self.order])

    def grid_batch(self, texts: Sequence[str | None]) -> np.ndarray:
        """(N, 41) float64 grid distributions; all-NaN rows where the cleaned text is empty."""
        prepared = self._prepare(texts)
        out = np.full((len(prepared), N_GRID), np.nan)
        keep = np.array([bool(t.strip()) for t in prepared], dtype=bool)
        if keep.any():
            P = self.backend.classify([t for t, k in zip(prepared, keep) if k])
            out[keep] = to_grid(np.asarray(P, dtype=np.float64)[:, self.order], self.band_edges)
        return out

    def grid(self, texts: str | Sequence[str | None]) -> np.ndarray:
        """One text -> (41,); N texts -> (N, 41)."""
        single = isinstance(texts, str)
        G = self.grid_batch([texts] if single else texts)
        return G[0] if single else G

    def canonical(self, texts: str | Sequence[str | None]) -> np.ndarray:
        single = isinstance(texts, str)
        P = self.canonical_batch([texts] if single else texts)
        return P[0] if single else P

    # -- table / provenance ---------------------------------------------------

    def columns(self) -> list[str]:
        """The headline_sentiment columns this sentimeter writes (Float16 P_00..P_40)."""
        return list(GRID_COLUMNS)

    def describe(self) -> dict[str, Any]:
        return {"model": self.name, "version": self.version, "clean_text": self.clean,
                "n_labels": len(self.labels), "band_edges": [float(e) for e in self.band_edges],
                "grid": f"linspace(-1, 1, {N_GRID})"}

    def model_card(self) -> ModelCard:
        from datalake import ModelCard

        return ModelCard(
            model_id=f"{self.name}-sentiment", version=self.version, repo=self.repo,
            weights_public=self.weights_public, weights_sha256=self.backend.weights_sha256(),
            architecture="BertForSequenceClassification",
            notes=f"{type(self).__name__}: {self.__doc__.strip().splitlines()[0]}",
            backend=self.backend.name,
            serving={**self.backend.info(), "sentimeter": self.describe(),
                     "identity": self.backend.identity(),
                     "checks": {"labels": self.labels, "canonical_order": self.order.tolist(),
                                "dtype": self.backend.dtype, "passed": True}},
        )
