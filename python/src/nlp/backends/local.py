"""In-process backend: the model runs on this machine (torch).

One ``LocalBackend`` loads one model directory for one task:

  * ``task="embedding"``: a sentence-transformers directory, through
    ``ravenbert.embedding.model.EmbeddingModel`` (``optimized_inference=True``:
    length-sorted batches, original order restored). Pooling and dim are read
    from the directory's ``1_Pooling/config.json``.
  * ``task="classification"``: a HuggingFace sequence classifier through
    transformers; logits in length-sorted batches, softmax in float64 here
    (the same map every backend uses). Labels come from ``config.json``
    ``id2label``, in index order.

torch / transformers / ravenbert are imported only when the backend is built,
so importing ``nlp`` never requires the GPU stack.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np

from nlp.backends.base import (
    Backend,
    EmbeddingConfig,
    run_in_length_order,
    softmax64,
    unit_rows,
)

Task = Literal["embedding", "classification"]

# sentence-transformers 1_Pooling/config.json flag -> TEI pooling name
_ST_POOLING = {
    "pooling_mode_cls_token": "cls",
    "pooling_mode_mean_tokens": "mean",
    "pooling_mode_max_tokens": "max",
    "pooling_mode_lasttoken": "last-token",
    "pooling_mode_mean_sqrt_len_tokens": "mean-sqrt-len",
    "pooling_mode_weightedmean_tokens": "weighted-mean",
}


def weights_sha256(model_path: Path | str) -> str:
    """sha256 over a model directory (relative path + bytes, sorted order).

    Hidden entries (e.g. the ``.cache/`` a HF ``--local-dir`` download leaves)
    are skipped, so the same weights hash identically on every machine.
    """
    model_path = Path(model_path)
    files = sorted(p for p in model_path.rglob("*") if p.is_file()
                   and not any(part.startswith(".") for part in p.relative_to(model_path).parts))
    if not files:
        raise FileNotFoundError(f"no files under model directory {model_path}")
    h = hashlib.sha256()
    for path in files:
        h.update(path.relative_to(model_path).as_posix().encode() + b"\0")
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(8 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def st_embedding_config(model_path: Path) -> EmbeddingConfig:
    """Pooling + dim declared by a sentence-transformers directory."""
    cfg_path = Path(model_path) / "1_Pooling" / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"{model_path} has no 1_Pooling/config.json "
                                "(not a sentence-transformers embedding model)")
    cfg = json.loads(cfg_path.read_text())
    if "pooling_mode" in cfg:                 # sentence-transformers >= 5 format
        pooling = str(cfg["pooling_mode"]).replace("lasttoken", "last-token")
        return EmbeddingConfig(dim=int(cfg["embedding_dimension"]), pooling=pooling)
    modes = [name for flag, name in _ST_POOLING.items() if cfg.get(flag)]   # legacy flags
    pooling = modes[0] if len(modes) == 1 else ("+".join(modes) or None)
    return EmbeddingConfig(dim=int(cfg["word_embedding_dimension"]), pooling=pooling)


def config_labels(model_path: Path) -> list[str]:
    """Classifier labels from config.json ``id2label``, in logit index order."""
    cfg = json.loads((Path(model_path) / "config.json").read_text())
    id2label = cfg.get("id2label") or {}
    if not id2label:
        raise ValueError(f"{model_path}/config.json declares no id2label")
    labels = {int(k): str(v) for k, v in id2label.items()}
    if sorted(labels) != list(range(len(labels))):
        raise ValueError(f"id2label indices are not 0..{len(labels) - 1}: {sorted(labels)}")
    return [labels[i] for i in range(len(labels))]


class LocalBackend(Backend):
    """One model directory run in-process.

    Args:
        model_path: local model directory.
        task:       "embedding" or "classification".
        dtype:      "float16" or "float32" compute dtype.
        device:     torch device; None picks cuda -> mps -> cpu.
        batch_size: texts per forward pass.
        max_length: classifier truncation length in tokens.
    """

    name = "local"

    def __init__(self, model_path: Path | str, *, task: Task, dtype: str = "float16",
                 device: str | None = None, batch_size: int = 128,
                 max_length: int = 512) -> None:
        super().__init__(dtype)
        if task not in ("embedding", "classification"):
            raise ValueError(f"task must be 'embedding' or 'classification', got {task!r}")
        self.model_path = Path(model_path)
        if not self.model_path.is_dir():
            raise FileNotFoundError(f"model directory does not exist: {self.model_path}")
        self.task: str = task
        self.batch_size, self.max_length = batch_size, max_length
        self._sha: str | None = None
        if task == "embedding":
            from ravenbert.embedding.model import EmbeddingModel

            self._config = st_embedding_config(self.model_path)
            self._model = EmbeddingModel.from_path(str(self.model_path), device=device,
                                                   precision=self.dtype)
            self.device = self._model.device
        else:
            import torch
            from ravenbert._device import resolve_device
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._labels = config_labels(self.model_path)
            self.device = resolve_device(device)
            self._torch = torch
            self._tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            self._model = AutoModelForSequenceClassification.from_pretrained(
                str(self.model_path), dtype=getattr(torch, self.dtype)).to(self.device).eval()

    # -- embedding ----------------------------------------------------------

    def embedding_config(self) -> EmbeddingConfig:
        if self.task != "embedding":
            return super().embedding_config()
        return self._config

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if self.task != "embedding":
            return super().embed(texts)
        texts = list(texts)
        if not texts:
            return np.empty((0, self._config.dim), dtype=np.float32)
        vectors = self._model.encode(texts, batch_size=self.batch_size, normalize=True,
                                     optimized_inference=True, show_progress_bar=False)
        return unit_rows(np.asarray(vectors, dtype=np.float32))

    # -- classification -----------------------------------------------------

    def labels(self) -> list[str]:
        if self.task != "classification":
            return super().labels()
        return list(self._labels)

    def classify(self, texts: Sequence[str]) -> np.ndarray:
        if self.task != "classification":
            return super().classify(texts)
        torch = self._torch

        def logits(batch: list[str]) -> np.ndarray:
            enc = self._tokenizer(batch, padding=True, truncation=True,
                                  max_length=self.max_length, return_tensors="pt")
            with torch.inference_mode():
                return self._model(**enc.to(self.device)).logits.float().cpu().numpy()

        raw = run_in_length_order(list(texts), self.batch_size, len(self._labels), logits,
                                  dtype=np.float64)
        return softmax64(raw)

    # -- metadata -----------------------------------------------------------

    def weights_sha256(self) -> str:
        if self._sha is None:
            self._sha = weights_sha256(self.model_path)
        return self._sha

    def identity(self) -> dict[str, Any]:
        ident = {**super().identity(), "task": self.task,
                 "model": self.model_path.name, "weights_sha256": self.weights_sha256()}
        if self.task == "embedding":
            ident.update(dim=self._config.dim, pooling=self._config.pooling)
        else:
            ident.update(n_labels=len(self._labels))
        return ident

    def info(self) -> dict[str, Any]:
        import torch

        info: dict[str, Any] = {
            "engine": "local", "task": self.task, "model_path": str(self.model_path),
            "device": str(self.device), "dtype": self.dtype,
            "weights_sha256": self.weights_sha256(), "torch_version": torch.__version__,
        }
        if self.task == "embedding":
            info.update(dim=self._config.dim, pooling=self._config.pooling,
                        batching="length-sorted (optimized_inference)")
        else:
            import transformers

            info.update(id2label=dict(enumerate(self._labels)), max_length=self.max_length,
                        transformers_version=transformers.__version__,
                        batching="length-sorted")
        return info
