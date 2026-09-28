"""RavenbertSentimeter: 41 ordinal classes on s = linspace(-1, 1, 41).

Canonical output: the 41 class probabilities, class i <-> s_i (logit index order, as
RavenBERT was trained). They ARE the grid distribution: the band of class i is the bin
of s_i, so the remap is the identity. Scores (mean / median / argmax) and their
confidences are computed at load time by ``nlp.sentiment.ordinal_sql``.
"""
from __future__ import annotations

from typing import ClassVar

from nlp.backends.base import IncompatibleModelError
from nlp.sentiment.base import GRID_EDGES, N_GRID, Sentimeter

N_CLASSES = N_GRID


class RavenbertSentimeter(Sentimeter):
    """41-class ordinal RavenBERT sentiment, stored as-is on the 41-point grid."""

    name = "ravenbert"
    version = "2.1"          # 2.1: grid storage (P_00..P_40), scores computed at load time
    repo = "https://github.com/GabinTB/RavenBERT"
    weights_public = False
    canonical_names: ClassVar[tuple[str, ...]] = tuple(f"{i:02d}" for i in range(N_CLASSES))
    band_edges: ClassVar[tuple[float, ...]] = tuple(float(e) for e in GRID_EDGES)

    @classmethod
    def canonical_order(cls, labels: list[str]) -> list[int]:
        if len(labels) != N_CLASSES:
            raise IncompatibleModelError(
                f"RavenbertSentimeter needs a {N_CLASSES}-label head, got {len(labels)}")
        return list(range(N_CLASSES))        # logit index i <-> s_i
