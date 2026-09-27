"""Sentimeters: one parent class, one subclass per model family.

    RavenbertSentimeter   41 ordinal classes -> EV (or median) in [-1, 1]
    FinbertSentimeter     negative / neutral / positive -> band score in [-1, 1]

``load_sentimeter("ravenbert")`` builds one on the default backend (TEI, fp16);
the local backend reads the weights directory from the model's env var.
"""
from __future__ import annotations

import os
from typing import Any

from nlp.sentiment.base import CONFIDENCE_COLUMN, SCORE_COLUMN, Sentimeter
from nlp.sentiment.finbert import FinbertSentimeter
from nlp.sentiment.ravenbert import RavenbertSentimeter

SENTIMETERS: dict[str, type[Sentimeter]] = {
    RavenbertSentimeter.name: RavenbertSentimeter,
    FinbertSentimeter.name: FinbertSentimeter,
}

# local backend weights directory, per model family
MODEL_PATH_ENV = {"ravenbert": "RAVENBERT_SENTIMENT_MODEL_PATH", "finbert": "FINBERT_MODEL_PATH"}


def load_sentimeter(model: str, backend: str = "tei", dtype: str = "float16", *,
                    model_path: str | None = None, score_rule: str | None = None,
                    min_confidence: float | None = None, **backend_kwargs: Any) -> Sentimeter:
    """A ``Sentimeter`` of family ``model`` on a named backend ("tei" | "local")."""
    from nlp.backends import make_backend

    if model not in SENTIMETERS:
        raise ValueError(f"model must be one of {sorted(SENTIMETERS)}, got {model!r}")
    if backend == "embedx":
        raise ValueError("embedx serves embeddings only; use tei or local for sentiment")
    if backend == "local":
        model_path = model_path or os.environ.get(MODEL_PATH_ENV[model])
    b = make_backend(backend, task="classification", dtype=dtype, model_path=model_path,
                     **backend_kwargs)
    return SENTIMETERS[model](b, score_rule=score_rule, min_confidence=min_confidence)


def sentimeter_from_card(card: Any, *, model_path: str | None = None,
                         **backend_kwargs: Any) -> Sentimeter:
    """The Sentimeter an artifact was produced with, rebuilt from its model card:
    family, score rule and confidence threshold from the card, backend from the
    recorded identity (must serve the same model; remote servers queried first)."""
    from nlp.backends import backend_from_identity

    serving = card.serving or {}
    settings = serving.get("sentimeter") or {}
    model = settings.get("model") or card.model_id.removesuffix("-sentiment")
    if model not in SENTIMETERS:
        raise ValueError(f"unknown sentimeter family in card: {model!r}")
    identity = serving.get("identity") or {}
    if identity.get("backend") == "local":
        model_path = model_path or os.environ.get(MODEL_PATH_ENV[model])
    backend = backend_from_identity(identity, task="classification", model_path=model_path,
                                    **backend_kwargs)
    return SENTIMETERS[model](backend, score_rule=settings.get("score_rule"),
                              min_confidence=settings.get("min_confidence"),
                              clean=settings.get("clean_text", True))


__all__ = ["CONFIDENCE_COLUMN", "MODEL_PATH_ENV", "SCORE_COLUMN", "SENTIMETERS",
           "FinbertSentimeter", "RavenbertSentimeter", "Sentimeter", "load_sentimeter",
           "sentimeter_from_card"]
