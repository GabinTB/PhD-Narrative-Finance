"""FinbertSentimeter: negative / neutral / positive (ProsusAI/finbert-compatible).

Canonical output: [p_neg, p_neu, p_pos], the columns located through the
head's ``id2label`` (lower-cased), never assumed.

Score: the band-consistent score, whose sign is the argmax class, with m the
second-largest probability:

    positive (p_pos >= p_neu, p_pos > p_neg):  s =  1/3 + (2/3)(p_pos - m)
    negative (p_neg >= p_neu, p_neg > p_pos):  s = -1/3 - (2/3)(p_neg - m)
    neutral  (p_neu > p_pos, p_neu > p_neg):
                            s = (1/3)(p_pos - p_neg) / (p_neu - min(p_pos, p_neg))
    p_pos == p_neg > p_neu, or all equal:      s = 0

[-1, -1/3] is negative, (-1/3, 1/3) neutral, [1/3, 1] positive, matching the
argmax class. The neutral formula is strictly inside (-1/3, 1/3) because
p_neu > max(p_pos, p_neg) >= |p_pos - p_neg| + min(p_pos, p_neg); it reaches
+-1/3 exactly on the neutral/positive (negative) tie, where the polar formula
also gives +-1/3, so the score is continuous across both neutral/polar
boundaries. The only jump is the positive/negative tie, unavoidable because
those bands are not adjacent.

Confidence: the margin p_(1) - p_(2) between the two most probable classes,
in [0, 1]; 0 at any tie at the top (e.g. all three at 1/3).
"""
from __future__ import annotations

from typing import ClassVar

import numpy as np

from nlp.backends.base import IncompatibleModelError
from nlp.sentiment.base import Sentimeter

THIRD = 1.0 / 3.0
CANONICAL = ("negative", "neutral", "positive")


def finbert_band(p_neg: np.ndarray, p_neu: np.ndarray,
                 p_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The band-consistent score (module docstring) and its class (+1 / 0 / -1), float64.

    The class comes from the probability comparisons, not from the float score, so it
    cannot flip on rounding next to a band edge."""
    p_neg, p_neu, p_pos = (np.asarray(a, dtype=np.float64) for a in (p_neg, p_neu, p_pos))
    pos = (p_pos >= p_neu) & (p_pos > p_neg)
    neg = (p_neg >= p_neu) & (p_neg > p_pos)
    neu = (p_neu > p_pos) & (p_neu > p_neg)
    out = np.zeros_like(p_pos)                     # pos/neg tie and triple point
    m_pos = np.maximum(p_neu, p_neg)
    m_neg = np.maximum(p_neu, p_pos)
    out[pos] = THIRD + 2 * THIRD * (p_pos[pos] - m_pos[pos])
    out[neg] = -THIRD - 2 * THIRD * (p_neg[neg] - m_neg[neg])
    denom = p_neu[neu] - np.minimum(p_pos[neu], p_neg[neu])      # > 0 on neu
    out[neu] = THIRD * (p_pos[neu] - p_neg[neu]) / denom
    label = pos.astype(np.float64) - neg.astype(np.float64)
    return out, label


def top_margin(P: np.ndarray) -> np.ndarray:
    """p_(1) - p_(2) per row (vectorised partial sort)."""
    top2 = np.partition(P, -2, axis=1)[:, -2:]
    return top2[:, 1] - top2[:, 0]


class FinbertSentimeter(Sentimeter):
    """3-class FinBERT sentiment, score = band-consistent score in [-1, 1]."""

    name = "finbert"
    version = "2.0"          # 2.0: nlp Sentimeter (SENT_SCORE / SENT_CONF table)
    repo = "https://huggingface.co/ProsusAI/finbert"
    weights_public = True
    score_rules: ClassVar[tuple[str, ...]] = ("band",)
    canonical_names: ClassVar[tuple[str, ...]] = ("NEG", "NEU", "POS")

    @classmethod
    def canonical_order(cls, labels: list[str]) -> list[int]:
        by_name = {label.lower(): i for i, label in enumerate(labels)}
        if len(labels) != 3 or set(by_name) != set(CANONICAL):
            raise IncompatibleModelError(
                f"FinbertSentimeter needs a negative/neutral/positive head, got {labels}")
        return [by_name[c] for c in CANONICAL]

    def score_from(self, P: np.ndarray) -> np.ndarray:
        return finbert_band(P[:, 0], P[:, 1], P[:, 2])[0]

    def confidence_from(self, P: np.ndarray) -> np.ndarray:
        return top_margin(P)
