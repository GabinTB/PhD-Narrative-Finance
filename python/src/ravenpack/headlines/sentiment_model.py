"""Model sentiment (RavenBERT, FinBERT) -> a ``headline_sentiment`` artifact.

Every headline of a ``ravenpack_headlines`` month is cleaned (``ravenbert``'s
``clean_text``; a null headline becomes "") and classified by a
BertForSequenceClassification in fp32; the softmax probabilities, in a canonical
class order, are turned into ``SENT_*`` columns by pure numpy functions:

RavenBERT (41 ordinal classes, s_i = linspace(-1, 1, 41)_i, source ``ravenbert``)
    SENT_EV      sum_i p_i s_i                      (== SentimentModel.predict)
    SENT_ARGMAX  s at argmax_i p_i                  (ties: the lowest class)
    SENT_TOP3    sum_{top3} p_i s_i / sum_{top3} p_i  (renormalised over the 3 most
                                                      probable classes)

FinBERT (ProsusAI/finbert, classes negative / neutral / positive, source ``finbert``;
the class order is read from the model's ``id2label``, never assumed)
    SENT_BAND    band-consistent score (below): [-1, -1/3] negative, (-1/3, 1/3)
                 neutral, [1/3, 1] positive, matching the argmax class
    SENT_EV      p_pos - p_neg
    SENT_ARGMAX  +1 / 0 / -1 for the band's class
    P_NEG, P_NEU, P_POS   the probabilities (Float16), kept so the mapping can be
                 changed later without re-running inference; not selectable

The band score, with m the second-largest probability:

    positive (p_pos >= p_neu, p_pos > p_neg):  s =  1/3 + (2/3)(p_pos - m)
    negative (p_neg >= p_neu, p_neg > p_pos):  s = -1/3 - (2/3)(p_neg - m)
    neutral  (p_neu > p_pos, p_neu > p_neg):
                            s = (1/3)(p_pos - p_neg) / (p_neu - min(p_pos, p_neg))
    p_pos == p_neg > p_neu, or all equal:      s = 0

The neutral formula is strictly inside (-1/3, 1/3) because p_neu > max(p_pos,
p_neg) >= |p_pos - p_neg| + min(p_pos, p_neg); it reaches +-1/3 exactly on the
neutral/positive (negative) tie, where the polar formula also gives +-1/3, so the
score is continuous across both neutral/polar boundaries. The only jump is the
positive/negative tie, unavoidable because those bands are not adjacent. A tie
with neutral lands on the closed polar band edge. The scorer labels s > eps
positive (open), so with --neutral-eps 1/3 a score of exactly 1/3 (a tie, or
float32 rounding within ~3e-8 of it) reads as neutral: measure-zero.

Backends: ``LocalBackend`` runs the model in-process (``--device``);
``RayBackend`` dispatches ordered text batches to an actor pool on a Ray cluster
(``RAY_ADDRESS``; the address is never logged). Each actor loads the model from
its own machine's path (the same env var as locally), hashes its weights and the
driver refuses a pool whose weights differ from the local model card. Both
backends run the same ``Classifier`` code.

    uv run python -m ravenpack.headlines.sentiment_model run --model finbert \
        --start-year 2000 --end-year 2025 [--backend ray --actors 2] [--temp]
    uv run python -m ravenpack.headlines.sentiment_model resume <partial artifact id> ...
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from ravenpack.headlines.sentiment import (
    ID_COL,
    SOURCE_KIND,
    MonthProducer,
    ingest_to_datalake,
    resume_partial,
)

if TYPE_CHECKING:
    from datalake import ModelCard

log = logging.getLogger(__name__)

RAVENBERT_CLASSES = np.linspace(-1.0, 1.0, 41)
THIRD = 1.0 / 3.0


# ---------------------------------------------------------------------------
# Probabilities -> SENT_* columns (pure, vectorised)
# ---------------------------------------------------------------------------

def _check_probs(probs: np.ndarray, n_classes: int) -> np.ndarray:
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != n_classes:
        raise ValueError(f"expected (n, {n_classes}) probabilities, got {p.shape}")
    if not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("probabilities must be finite and non-negative")
    return p


def ravenbert_columns(probs: np.ndarray) -> dict[str, np.ndarray]:
    """SENT_EV / SENT_ARGMAX / SENT_TOP3 (float32) from (n, 41) softmax probabilities."""
    p = _check_probs(probs, RAVENBERT_CLASSES.size)
    s = RAVENBERT_CLASSES
    top3 = np.argsort(-p, axis=1, kind="stable")[:, :3]
    p3 = np.take_along_axis(p, top3, axis=1)
    return {
        "SENT_EV": (p @ s).astype(np.float32),
        "SENT_ARGMAX": s[np.argmax(p, axis=1)].astype(np.float32),
        "SENT_TOP3": ((p3 * s[top3]).sum(axis=1) / p3.sum(axis=1)).astype(np.float32),
    }


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


def finbert_columns(probs: np.ndarray) -> dict[str, np.ndarray]:
    """SENT_BAND / SENT_EV / SENT_ARGMAX (float32) + P_* (float16) from (n, 3)
    probabilities ordered [neg, neu, pos]."""
    p = _check_probs(probs, 3)
    p_neg, p_neu, p_pos = p[:, 0], p[:, 1], p[:, 2]
    band, label = finbert_band(p_neg, p_neu, p_pos)
    return {
        "SENT_BAND": band.astype(np.float32),
        "SENT_EV": (p_pos - p_neg).astype(np.float32),
        "SENT_ARGMAX": label.astype(np.float32),
        "P_NEG": p_neg.astype(np.float16),
        "P_NEU": p_neu.astype(np.float16),
        "P_POS": p_pos.astype(np.float16),
    }


# ---------------------------------------------------------------------------
# Model specs
# ---------------------------------------------------------------------------

def _ravenbert_order(config: dict[str, Any]) -> list[int]:
    from ravenbert.sentiment.model import validate_sentiment_config

    validate_sentiment_config(config)             # BERT classifier with exactly 41 labels
    return list(range(RAVENBERT_CLASSES.size))


def _finbert_order(config: dict[str, Any]) -> list[int]:
    """Column indices of negative, neutral, positive in the model's logits."""
    id2label = {int(k): str(v).lower() for k, v in (config.get("id2label") or {}).items()}
    by_name = {v: k for k, v in id2label.items()}
    missing = {"negative", "neutral", "positive"} - set(by_name)
    if missing or len(id2label) != 3:
        raise ValueError(f"FinBERT id2label must be negative/neutral/positive, got {id2label}")
    return [by_name["negative"], by_name["neutral"], by_name["positive"]]


@dataclass(frozen=True)
class ModelSpec:
    name: str                                     # the artifact's ``source``
    env_var: str                                  # weights directory, on every machine
    version: str
    repo: str
    weights_public: bool
    columns: tuple[str, ...]                      # the SENT_* columns, in file order
    class_order: Callable[[dict[str, Any]], list[int]]
    derive: Callable[[np.ndarray], dict[str, np.ndarray]]
    notes: str


MODELS: dict[str, ModelSpec] = {
    "ravenbert": ModelSpec(
        name="ravenbert", env_var="RAVENBERT_SENTIMENT_MODEL_PATH", version="1.0",
        repo="https://github.com/GabinTB/RavenBERT", weights_public=False,
        columns=("SENT_EV", "SENT_ARGMAX", "SENT_TOP3"), class_order=_ravenbert_order,
        derive=ravenbert_columns,
        notes="41-class ordinal BERT, s=linspace(-1,1,41); SENT_EV=sum p s, SENT_ARGMAX=s[argmax],"
              " SENT_TOP3=top-3 renormalised EV. fp32, clean_text, max_length 512."),
    "finbert": ModelSpec(
        name="finbert", env_var="FINBERT_MODEL_PATH", version="prosusai",
        repo="https://huggingface.co/ProsusAI/finbert", weights_public=True,
        columns=("SENT_BAND", "SENT_EV", "SENT_ARGMAX"), class_order=_finbert_order,
        derive=finbert_columns,
        notes="ProsusAI/finbert (Araci 2019). SENT_BAND=band-consistent score (+-1/3 bands match "
              "argmax), SENT_EV=p_pos-p_neg, SENT_ARGMAX in {-1,0,1}; P_NEG/P_NEU/P_POS kept "
              "(float16). fp32, ravenbert clean_text, max_length 512."),
}


def weights_sha256(model_path: Path) -> str:
    """sha256 over the model directory's files (relative path + bytes, sorted), skipping
    hidden entries (e.g. the ``.cache/`` a HF ``--local-dir`` download leaves), so the
    same weights hash identically on every machine."""
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


def model_card(spec: ModelSpec, sha: str, backend_note: str = "") -> ModelCard:
    from datalake import ModelCard

    return ModelCard(model_id=f"{spec.name}-sentiment", version=spec.version, repo=spec.repo,
                     weights_public=spec.weights_public, weights_sha256=sha,
                     architecture="BertForSequenceClassification",
                     notes=f"{spec.notes} {backend_note}".strip())


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def prepare_texts(values: Sequence[str | None]) -> list[str]:
    """ravenbert's clean_text; a null headline becomes "" (the row is never dropped)."""
    from ravenbert.sentiment.model import clean_text

    return [clean_text(v) if v is not None else "" for v in values]


class Classifier:
    """fp32 sequence classifier returning softmax probabilities in the spec's class order."""

    def __init__(self, spec: ModelSpec, model_path: Path | str, device: str | None = None,
                 batch_size: int = 256, max_length: int = 512):
        import json

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.model_path = Path(model_path)
        config = json.loads((self.model_path / "config.json").read_text())
        self.order = spec.class_order(config)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size, self.max_length = batch_size, max_length
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(self.model_path), dtype=torch.float32).to(self.device).eval()
        self._torch = torch

    def probs(self, texts: Sequence[str]) -> np.ndarray:
        """(n, n_classes) float32, rows aligned with ``texts``. Batches are formed in
        descending-length order (less padding), results are restored to input order."""
        torch = self._torch
        n = len(texts)
        out = np.empty((n, len(self.order)), dtype=np.float32)
        order = np.argsort([-len(t) for t in texts], kind="stable")
        with torch.inference_mode():
            for start in range(0, n, self.batch_size):
                idx = order[start:start + self.batch_size]
                enc = self.tokenizer([texts[i] for i in idx], padding=True, truncation=True,
                                     max_length=self.max_length, return_tensors="pt")
                logits = self.model(**enc.to(self.device)).logits.float()
                p = torch.softmax(logits, dim=-1).cpu().numpy()
                out[idx] = p[:, self.order]
        return out


class Backend(Protocol):
    weights_sha256: str

    def probs(self, texts: Sequence[str]) -> np.ndarray: ...

    def describe(self) -> str: ...


class LocalBackend:
    def __init__(self, spec: ModelSpec, model_path: Path, device: str | None = None,
                 batch_size: int = 256, max_length: int = 512):
        self.classifier = Classifier(spec, model_path, device, batch_size, max_length)
        self.weights_sha256 = weights_sha256(model_path)

    def probs(self, texts: Sequence[str]) -> np.ndarray:
        return self.classifier.probs(texts)

    def describe(self) -> str:
        return f"local:{self.classifier.device}"


class _ClassifierActor:
    """Ray actor body: the model path comes from the WORKER's environment unless given."""

    def __init__(self, spec_name: str, model_path: str | None, batch_size: int,
                 max_length: int):
        spec = MODELS[spec_name]
        path = Path(model_path or os.environ[spec.env_var])
        self.classifier = Classifier(spec, path, None, batch_size, max_length)
        self.sha = weights_sha256(path)

    def info(self) -> tuple[str, str]:
        return self.sha, self.classifier.device

    def probs(self, texts: list[str]) -> np.ndarray:
        return self.classifier.probs(texts)


class RayBackend:
    """Ordered dispatch of text slices to an actor pool; bitwise the same Classifier code."""

    def __init__(self, spec: ModelSpec, expected_sha: str, *, address: str | None,
                 n_actors: int = 1, gpus_per_actor: float = 1.0, model_path: str | None = None,
                 batch_size: int = 256, max_length: int = 512, slice_rows: int = 8_192):
        import ray

        if not ray.is_initialized():
            ray.init(address=address, log_to_driver=False)     # address is never logged
        actor_cls = ray.remote(num_gpus=gpus_per_actor)(_ClassifierActor)
        self._ray = ray
        self.actors = [actor_cls.remote(spec.name, model_path, batch_size, max_length)
                       for _ in range(n_actors)]
        infos = ray.get([a.info.remote() for a in self.actors])
        bad = [sha for sha, _ in infos if sha != expected_sha]
        if bad:
            raise RuntimeError(f"{len(bad)} Ray actor(s) hold different {spec.name} weights "
                               f"(sha {bad[0][:12]} != local {expected_sha[:12]})")
        self.devices = [dev for _, dev in infos]
        self.weights_sha256 = expected_sha
        self.slice_rows = slice_rows
        log.info("Ray pool: %d actor(s) on %s", len(self.actors), sorted(set(self.devices)))

    def probs(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        slices = [texts[i:i + self.slice_rows] for i in range(0, len(texts), self.slice_rows)]
        refs = [self.actors[i % len(self.actors)].probs.remote(s) for i, s in enumerate(slices)]
        parts = self._ray.get(refs)                             # ordered like ``slices``
        return np.concatenate(parts) if parts else np.empty((0, 0), dtype=np.float32)

    def describe(self) -> str:
        return f"ray:{len(self.actors)}x{sorted(set(self.devices))}"


# ---------------------------------------------------------------------------
# Month producer and CLI
# ---------------------------------------------------------------------------

def producer(spec: ModelSpec, backend: Backend, read_rows: int = 200_000) -> MonthProducer:
    """headlines ``YYYY-MM.parquet`` -> RP_STORY_ID + the spec's columns, streamed."""
    def produce(headlines_month: Path, tag: str) -> pl.DataFrame:
        frames: list[pl.DataFrame] = []
        t0, n = time.monotonic(), 0
        for batch in pq.ParquetFile(headlines_month).iter_batches(
                batch_size=read_rows, columns=[ID_COL, "HEADLINE"]):
            texts = prepare_texts(batch.column("HEADLINE").to_pylist())
            cols = spec.derive(backend.probs(texts))
            frames.append(pl.DataFrame({ID_COL: batch.column(ID_COL).to_pylist(), **cols}))
            n += batch.num_rows
            log.info("%s  %d headlines scored | %.0f/s", tag, n, n / max(time.monotonic() - t0,
                                                                          1e-9))
        if not frames:
            return pl.DataFrame(schema={ID_COL: pl.String,
                                        **{c: pl.Float32 for c in spec.columns}})
        return pl.concat(frames)
    return produce


def _backend(args: argparse.Namespace, spec: ModelSpec, model_path: Path) -> Backend:
    if args.backend == "local":
        return LocalBackend(spec, model_path, args.device, args.batch_size)
    address = os.environ.get("RAY_ADDRESS")
    if not address:
        raise SystemExit("--backend ray needs RAY_ADDRESS (in .env)")
    return RayBackend(spec, weights_sha256(model_path), address=address, n_actors=args.actors,
                      gpus_per_actor=args.gpus_per_actor, batch_size=args.batch_size)


def main(argv: list[str] | None = None) -> int:
    from dotenv import find_dotenv, load_dotenv

    from datalake import DatalakeIndex

    load_dotenv(find_dotenv(usecwd=True))
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--backend", default="local", choices=["local", "ray"])
    ap.add_argument("--device", default=None, help="local backend: cuda|mps|cpu (default auto)")
    ap.add_argument("--actors", type=int, default=1, help="ray backend: actor count")
    ap.add_argument("--gpus-per-actor", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=256)
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--start-year", type=int, required=True)
    run.add_argument("--end-year", type=int, required=True)
    run.add_argument("--headlines-artifact", default=None)
    run.add_argument("--temp", action="store_true", help="mark the artifact agent-created")
    res = sub.add_parser("resume")
    res.add_argument("artifact_id")
    args = ap.parse_args(argv)

    spec = MODELS[args.model]
    if not os.environ.get(spec.env_var):
        raise SystemExit(f"{spec.env_var} is not set (see .env)")
    model_path = Path(os.environ[spec.env_var])
    backend = _backend(args, spec, model_path)
    produce = producer(spec, backend)
    repo_dir = Path(__file__).resolve().parents[4]
    with DatalakeIndex(os.environ["DATALAKE_ROOT"]) as dl:
        if args.cmd == "resume":
            art = dl.get(args.artifact_id)
            card = art.meta.model_card
            if art.meta.hyperparams.get("source") != spec.name or card is None \
                    or card.weights_sha256 != backend.weights_sha256:
                raise SystemExit(f"{args.artifact_id} was not produced by these {spec.name} "
                                 "weights; refusing to resume")
            art = resume_partial(dl, args.artifact_id, produce, repo_dir=repo_dir)
        else:
            hl = (dl.get(args.headlines_artifact) if args.headlines_artifact
                  else dl.latest(SOURCE_KIND))
            art = ingest_to_datalake(
                dl, source=spec.name, columns=list(spec.columns), produce=produce, headlines=hl,
                start_year=args.start_year, end_year=args.end_year,
                extra_hyperparams={"model_version": spec.version,
                                   "weights_sha": backend.weights_sha256[:12]},
                model_card=model_card(spec, backend.weights_sha256,
                                      f"backend={backend.describe()}"),
                temp=args.temp, repo_dir=repo_dir)
    print(art.artifact_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
