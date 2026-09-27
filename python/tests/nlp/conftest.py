"""Tiny random-weight models (no real weights) shared by the nlp tests."""
from __future__ import annotations

from pathlib import Path

import pytest

WORDS = ["stocks", "rally", "fall", "bank", "profit", "loss", "fed", "rates", "oil", "surge"]


def _tokenizer_and_config(root: Path, **extra):
    from transformers import BertConfig, BertTokenizerFast

    root.mkdir(parents=True)
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *WORDS]
    (root / "vocab.txt").write_text("\n".join(vocab))
    BertTokenizerFast(vocab_file=str(root / "vocab.txt")).save_pretrained(str(root))
    return BertConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1,
                      num_attention_heads=2, intermediate_size=32,
                      max_position_embeddings=64, **extra)


def tiny_classifier(root: Path, id2label: dict[int, str]) -> Path:
    import torch
    from transformers import BertForSequenceClassification

    cfg = _tokenizer_and_config(root, num_labels=len(id2label), id2label=id2label,
                                label2id={v: k for k, v in id2label.items()})
    torch.manual_seed(0)
    BertForSequenceClassification(cfg).save_pretrained(str(root))
    (root / ".cache").mkdir()                                 # like a HF --local-dir download
    (root / ".cache" / "meta").write_text("machine-specific")
    return root


def tiny_embedder(root: Path, pooling: str = "cls") -> Path:
    """A sentence-transformers directory: tiny BERT + a Pooling module."""
    import torch
    from sentence_transformers import SentenceTransformer, models
    from transformers import BertModel

    bert_dir = root.parent / f"{root.name}_bert"
    cfg = _tokenizer_and_config(bert_dir)
    torch.manual_seed(0)
    BertModel(cfg).save_pretrained(str(bert_dir))
    word = models.Transformer(str(bert_dir), max_seq_length=64)
    pool = models.Pooling(word.get_word_embedding_dimension(), pooling_mode=pooling)
    SentenceTransformer(modules=[word, pool], device="cpu").save(str(root))
    return root


@pytest.fixture(scope="session")
def ravenbert_like_dir(tmp_path_factory) -> Path:
    return tiny_classifier(tmp_path_factory.mktemp("rb") / "m",
                           {i: f"LABEL_{i}" for i in range(41)})


@pytest.fixture(scope="session")
def finbert_like_dir(tmp_path_factory) -> Path:
    # deliberately not in [neg, neu, pos] order
    return tiny_classifier(tmp_path_factory.mktemp("fb") / "m",
                           {0: "positive", 1: "negative", 2: "neutral"})


@pytest.fixture(scope="session")
def embedder_dir(tmp_path_factory) -> Path:
    return tiny_embedder(tmp_path_factory.mktemp("emb") / "m", pooling="cls")


@pytest.fixture(scope="session")
def mean_embedder_dir(tmp_path_factory) -> Path:
    return tiny_embedder(tmp_path_factory.mktemp("embm") / "m", pooling="mean")


TEXTS = ["stocks rally", "bank loss fall fall", "fed rates", "oil surge profit profit profit",
         "", "stocks"]
