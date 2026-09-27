"""RavenbertSentimeter: 41 ordinal classes on s = linspace(-1, 1, 41).

Canonical output: the 41 class probabilities, class i <-> s_i (logit index
order, as RavenBERT was trained).

Score rules:
    mean    EV = sum_i p_i s_i  (default; equals ``ravenbert.SentimentModel.predict``)
    median  the median of the ordinal distribution with class i spread
            uniformly over its bin [s_i - h, s_i + h], h = step / 2:
            j = first class with F_j >= 1/2,
            median = s_j - h + (1/2 - F_{j-1}) / p_j * step   (F_{-1} = 0).
            A point mass at s_j gives exactly s_j; the result stays in [-1, 1]
            and is continuous in p.

Confidence: 1 - sigma, sigma = sqrt(sum_i p_i (s_i - EV)^2) the ordinal
dispersion around the mean. sigma <= 1 on [-1, 1] (half the mass at each
end), so confidence is in [0, 1]. Mass spread over ADJACENT classes (a
confident "moderately positive") keeps sigma small; mass on OPPOSITE poles
(a bimodal output whose mean ~0 would read "neutral") gives sigma ~1 and
confidence ~0 -- the case the mean score misrepresents. Entropy would
confuse the two, as it ignores the class order.
"""
from __future__ import annotations

from typing import ClassVar

import numpy as np

from nlp.backends.base import IncompatibleModelError
from nlp.sentiment.base import Sentimeter

N_CLASSES = 41
CLASS_VALUES = np.linspace(-1.0, 1.0, N_CLASSES)


def ordinal_mean(P: np.ndarray) -> np.ndarray:
    return P @ CLASS_VALUES


def ordinal_median(P: np.ndarray) -> np.ndarray:
    """Median of (N, 41) ordinal distributions, each class uniform on its bin (vectorised)."""
    cdf = np.cumsum(P, axis=1)
    j = np.argmax(cdf >= 0.5, axis=1)
    rows = np.arange(len(P))
    prev = np.where(j > 0, cdf[rows, np.maximum(j - 1, 0)], 0.0)
    step = CLASS_VALUES[1] - CLASS_VALUES[0]
    p_j = P[rows, j]
    frac = np.divide(0.5 - prev, p_j, out=np.full_like(p_j, 0.5), where=p_j > 0)
    return CLASS_VALUES[j] - step / 2 + frac * step


def ordinal_dispersion(P: np.ndarray) -> np.ndarray:
    """sigma = sqrt(sum p (s - EV)^2), in [0, 1] on the [-1, 1] grid."""
    ev = ordinal_mean(P)
    var = P @ (CLASS_VALUES ** 2) - ev ** 2
    return np.sqrt(np.clip(var, 0.0, None))


class RavenbertSentimeter(Sentimeter):
    """41-class ordinal RavenBERT sentiment, score = expected value (or median)."""

    name = "ravenbert"
    version = "2.0"          # 2.0: nlp Sentimeter (SENT_SCORE / SENT_CONF table)
    repo = "https://github.com/GabinTB/RavenBERT"
    weights_public = False
    score_rules: ClassVar[tuple[str, ...]] = ("mean", "median")
    canonical_names: ClassVar[tuple[str, ...]] = tuple(f"{i:02d}" for i in range(N_CLASSES))

    @classmethod
    def canonical_order(cls, labels: list[str]) -> list[int]:
        if len(labels) != N_CLASSES:
            raise IncompatibleModelError(
                f"RavenbertSentimeter needs a {N_CLASSES}-label head, got {len(labels)}")
        return list(range(N_CLASSES))        # logit index i <-> s_i

    def score_from(self, P: np.ndarray) -> np.ndarray:
        return ordinal_median(P) if self.score_rule == "median" else ordinal_mean(P)

    def confidence_from(self, P: np.ndarray) -> np.ndarray:
        return 1.0 - ordinal_dispersion(P)
