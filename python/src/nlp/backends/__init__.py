"""Inference backends (local, TEI, embedx): canonical model operations only.

See ``nlp.backends.base`` for the contract. Backends import their heavy
dependencies (torch, transformers, httpx clients) only when constructed.
"""
from typing import Any

from nlp.backends.base import (
    DTYPES,
    Backend,
    EmbeddingConfig,
    IncompatibleModelError,
    canonical_dtype,
)

__all__ = ["BACKENDS", "DEFAULT_BACKEND", "DEFAULT_DTYPE", "DTYPES", "Backend",
           "EmbeddingConfig", "IncompatibleModelError", "assert_same_model",
           "backend_from_identity", "canonical_dtype", "make_backend"]

BACKENDS: tuple[str, ...] = ("tei", "local", "embedx")
DEFAULT_BACKEND = "tei"
DEFAULT_DTYPE = "float16"


def assert_same_model(recorded: dict[str, Any], backend: Backend) -> None:
    """Refuse to continue a job on a backend that is not serving the recorded model.

    ``recorded`` is the ``identity`` stored in the artifact's model card when the
    job started (TEI: model_id, dtype, model_type, max_input_length, version;
    embedx: model, pooling, dim, dtype, version; local: weights sha256, dir name,
    structure). The backend's CURRENT identity -- for remote servers read live
    from their metadata endpoint -- must be identical.
    """
    current = backend.identity()
    diff = {k: (recorded.get(k), current.get(k))
            for k in sorted(set(recorded) | set(current)) if recorded.get(k) != current.get(k)}
    if diff:
        raise IncompatibleModelError(
            "the backend is not serving the model this artifact was produced with; "
            + "; ".join(f"{k}: recorded {a!r}, now {b!r}" for k, (a, b) in diff.items()))


def backend_from_identity(identity: dict[str, Any], *, task: str,
                          model_path: str | None = None, **kwargs: Any) -> Backend:
    """Rebuild the backend recorded in an artifact and check it serves the same model."""
    if not identity or "backend" not in identity:
        raise ValueError("the artifact records no backend identity (written before resume "
                         "support); it cannot be resumed safely")
    backend = make_backend(identity["backend"], task=task, dtype=identity["dtype"],
                           model_path=model_path, **kwargs)
    assert_same_model(identity, backend)
    return backend


def make_backend(name: str = DEFAULT_BACKEND, *, task: str, dtype: str = DEFAULT_DTYPE,
                 model_path: str | None = None, **kwargs: Any) -> Backend:
    """Build a backend by name.

    ``task`` is "embedding" or "classification". ``model_path`` is used by the
    local backend only (required there); remote backends read their endpoint
    from the environment (TEI_BASE_URL / EMBEDX_BASE_URL + EMBEDX_MODEL).
    """
    if name == "local":
        from nlp.backends.local import LocalBackend

        if not model_path:
            raise ValueError("the local backend needs a model_path")
        return LocalBackend(model_path, task=task, dtype=dtype, **kwargs)
    if name == "tei":
        from nlp.backends.tei import TEIBackend

        return TEIBackend(dtype=dtype, **kwargs)
    if name == "embedx":
        if task != "embedding":
            raise ValueError("embedx serves embeddings only")
        from nlp.backends.embedx import EmbedxBackend

        return EmbedxBackend(dtype=dtype, **kwargs)
    raise ValueError(f"backend must be one of {BACKENDS}, got {name!r}")
