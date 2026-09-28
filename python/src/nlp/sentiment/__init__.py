"""Sentimeters: one parent class, one subclass per model family.

    RavenbertSentimeter   41 ordinal classes, stored as-is on the 41-point grid
    FinbertSentimeter     negative / neutral / positive, spread on the grid by band

Every family writes the same table (P_00..P_40, a distribution on s = linspace(-1, 1,
41)); scores and confidences are computed at load time (``nlp.sentiment.ordinal_sql``).
``load_sentimeter("ravenbert")`` builds one on the default backend (TEI, fp16); the local
backend reads the weights directory from the model's env var.
"""
from __future__ import annotations

import os
from typing import Any

from nlp.sentiment.base import GRID, GRID_COLUMNS, N_GRID, Sentimeter, to_grid
from nlp.sentiment.finbert import FinbertSentimeter
from nlp.sentiment.ravenbert import RavenbertSentimeter

SENTIMETERS: dict[str, type[Sentimeter]] = {
    RavenbertSentimeter.name: RavenbertSentimeter,
    FinbertSentimeter.name: FinbertSentimeter,
}

# local backend weights directory, per model family
MODEL_PATH_ENV = {"ravenbert": "RAVENBERT_SENTIMENT_MODEL_PATH", "finbert": "FINBERT_MODEL_PATH"}


def load_sentimeter(model: str, backend: str = "tei", dtype: str = "float16", *,
                    model_path: str | None = None, **backend_kwargs: Any) -> Sentimeter:
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
    return SENTIMETERS[model](b)


def sentimeter_from_card(card: Any, *, model_path: str | None = None,
                         **backend_kwargs: Any) -> Sentimeter:
    """The Sentimeter an artifact was produced with, rebuilt from its model card: family
    and text cleaning from the card, backend from the recorded identity (must serve the
    same model; remote servers queried first)."""
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
    return SENTIMETERS[model](backend, clean=settings.get("clean_text", True))


__all__ = ["GRID", "GRID_COLUMNS", "MODEL_PATH_ENV", "N_GRID", "SENTIMETERS",
           "FinbertSentimeter", "RavenbertSentimeter", "Sentimeter", "load_sentimeter",
           "sentimeter_from_card", "to_grid"]
