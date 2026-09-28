"""FinbertSentimeter: negative / neutral / positive (ProsusAI/finbert-compatible).

Canonical output: [p_neg, p_neu, p_pos], the columns located through the head's
``id2label`` (lower-cased), never assumed.

Grid: the three classes on the bands [-1, -1/3], [-1/3, 1/3], [1/3, 1], each spread
at a uniform density over its band (``base.to_grid``). The bands have equal width, so
the densest band is the most probable class. Consequences on the grid rules of
``nlp.sentiment.ordinal_sql``:

    mean     (2/3)(p_pos - p_neg), up to the discretisation of the bins straddling
             +-1/3 and the half-width end bins (about 1e-3)
    argmax   -0.675 / 0 / +0.675: the 12 / 13 / 12 grid points whose whole bin lies
             inside the winning band; confidence 0.90 p_neg, 0.975 p_neu, 0.90 p_pos
"""
from __future__ import annotations

from typing import ClassVar

from nlp.backends.base import IncompatibleModelError
from nlp.sentiment.base import Sentimeter

THIRD = 1.0 / 3.0
CANONICAL = ("negative", "neutral", "positive")


class FinbertSentimeter(Sentimeter):
    """3-class FinBERT sentiment, stored as its classes spread on the 41-point grid."""

    name = "finbert"
    version = "2.1"          # 2.1: grid storage (P_00..P_40), scores computed at load time
    repo = "https://huggingface.co/ProsusAI/finbert"
    weights_public = True
    canonical_names: ClassVar[tuple[str, ...]] = ("NEG", "NEU", "POS")
    band_edges: ClassVar[tuple[float, ...]] = (-1.0, -THIRD, THIRD, 1.0)

    @classmethod
    def canonical_order(cls, labels: list[str]) -> list[int]:
        by_name = {label.lower(): i for i, label in enumerate(labels)}
        if len(labels) != 3 or set(by_name) != set(CANONICAL):
            raise IncompatibleModelError(
                f"FinbertSentimeter needs a negative/neutral/positive head, got {labels}")
        return [by_name[c] for c in CANONICAL]
