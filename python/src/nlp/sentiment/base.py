"""Sentimeter: text -> sentiment score in [-1, 1] on any classifier backend.

The parent class holds the default behaviour; each subclass is one model
family with its own canonical output and the map from it to the score:

    preprocess   ravenbert ``clean_text`` (the RavenPack demo cleaning); None -> ""
    backend      ``classify`` -> softmax probabilities (float64) in label order
    canonical    those probabilities permuted into the subclass's canonical order
    score        the subclass map canonical -> [-1, 1]
    confidence   the subclass measure in [0, 1] of how well one sentiment dominates;
                 rows below ``min_confidence`` score NaN ("no score"), never 0

Output shapes follow the input: one text gives a ``float`` score (canonical
``(k,)``), N texts give ``(N,)`` scores (canonical ``(N, k)``).

At construction the subclass checks that the backend's classifier head is
compatible (label structure only, never a model name or weights): any
41-label head is RavenBERT-compatible, any {negative, neutral, positive} head
FinBERT-compatible. ``evaluate`` returns score, confidence and canonical from
ONE inference pass, which is what corpus jobs use.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from nlp.backends.base import Backend

if TYPE_CHECKING:
    from datalake import ModelCard

SCORE_COLUMN = "SENT_SCORE"
CONFIDENCE_COLUMN = "SENT_CONF"


def clean_texts(texts: Sequence[str | None]) -> list[str]:
    """ravenbert's ``clean_text`` (RavenPack demo preprocessing); None -> ""."""
    from ravenbert.sentiment.model import clean_text

    return [clean_text(t) if t is not None else "" for t in texts]


class Sentimeter(ABC):
    """Parent class: preprocessing, inference, NaN rule, shapes, provenance.

    Args:
        backend:        a classifier backend (local or TEI).
        score_rule:     subclass-specific score map; None = the subclass default.
        min_confidence: rows whose confidence is below it score NaN; None = keep all.
        clean:          apply ``clean_text`` before inference (the default, as in
                        the RavenPack demos).
    """

    name: ClassVar[str]                    # model family, e.g. "ravenbert"
    version: ClassVar[str]                 # version of this scoring spec
    repo: ClassVar[str]
    weights_public: ClassVar[bool]
    score_rules: ClassVar[tuple[str, ...]]
    canonical_names: ClassVar[tuple[str, ...]]   # P_* column suffixes, canonical order

    def __init__(self, backend: Backend, *, score_rule: str | None = None,
                 min_confidence: float | None = None, clean: bool = True) -> None:
        self.backend = backend
        self.score_rule = score_rule or self.score_rules[0]
        if self.score_rule not in self.score_rules:
            raise ValueError(f"{self.name}: score_rule must be one of {self.score_rules}, "
                             f"got {self.score_rule!r}")
        if min_confidence is not None and not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")
        self.min_confidence = min_confidence
        self.clean = clean
        self.labels = backend.labels()
        self.order = np.asarray(self.canonical_order(self.labels), dtype=np.intp)

    # -- subclass hooks ------------------------------------------------------

    @classmethod
    @abstractmethod
    def canonical_order(cls, labels: list[str]) -> list[int]:
        """Label indices in canonical order; raise IncompatibleModelError if the
        head cannot be this model family."""

    @abstractmethod
    def score_from(self, P: np.ndarray) -> np.ndarray:
        """(N, k) canonical probabilities -> (N,) float64 scores in [-1, 1]."""

    @abstractmethod
    def confidence_from(self, P: np.ndarray) -> np.ndarray:
        """(N, k) canonical probabilities -> (N,) float64 confidence in [0, 1]."""

    # -- inference ------------------------------------------------------------

    def canonical_batch(self, texts: Sequence[str | None]) -> np.ndarray:
        prepared = clean_texts(texts) if self.clean else ["" if t is None else t for t in texts]
        P = self.backend.classify(prepared)
        return np.ascontiguousarray(P[:, self.order])

    def evaluate(self, texts: Sequence[str | None]) -> dict[str, np.ndarray]:
        """score (float32, NaN below min_confidence), confidence (float32) and
        canonical probabilities (float64) of N texts, from one inference pass."""
        P = self.canonical_batch(texts)
        score = self.score_from(P)
        conf = self.confidence_from(P)
        if self.min_confidence is not None:
            score = np.where(conf < self.min_confidence, np.nan, score)
        return {"score": score.astype(np.float32), "confidence": conf.astype(np.float32),
                "canonical": P}

    def score(self, texts: str | Sequence[str | None]) -> float | np.ndarray:
        single = isinstance(texts, str)
        out = self.evaluate([texts] if single else texts)["score"]
        return float(out[0]) if single else out

    def confidence(self, texts: str | Sequence[str | None]) -> float | np.ndarray:
        single = isinstance(texts, str)
        out = self.evaluate([texts] if single else texts)["confidence"]
        return float(out[0]) if single else out

    def canonical(self, texts: str | Sequence[str | None]) -> np.ndarray:
        single = isinstance(texts, str)
        P = self.canonical_batch([texts] if single else texts)
        return P[0] if single else P

    # -- table / provenance ---------------------------------------------------

    def canonical_columns(self) -> list[str]:
        return [f"P_{n}" for n in self.canonical_names]

    def columns(self, *, canonical: bool = False) -> list[str]:
        """The headline_sentiment columns this sentimeter writes."""
        return [SCORE_COLUMN, CONFIDENCE_COLUMN] + (self.canonical_columns() if canonical else [])

    def describe(self) -> dict[str, Any]:
        return {"model": self.name, "version": self.version, "score_rule": self.score_rule,
                "min_confidence": self.min_confidence, "clean_text": self.clean,
                "n_labels": len(self.labels)}

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
