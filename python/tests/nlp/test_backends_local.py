"""LocalBackend and the shared backend helpers, on tiny random-weight models."""
from __future__ import annotations

import numpy as np
import pytest

from nlp.backends.base import (
    canonical_dtype,
    length_sorted_batches,
    run_in_length_order,
    softmax64,
)
from nlp.backends.local import LocalBackend, weights_sha256

from .conftest import TEXTS

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def test_canonical_dtype():
    assert canonical_dtype("fp16") == canonical_dtype("float16") == "float16"
    assert canonical_dtype("FP32") == "float32"
    with pytest.raises(ValueError):
        canonical_dtype("bfloat16")


def test_length_sorted_batches_cover_every_index_once_longest_first():
    texts = ["a", "abcd", "ab", "abcdef", "abc"]
    batches = list(length_sorted_batches(texts, 2))
    flat = np.concatenate(batches)
    assert sorted(flat.tolist()) == list(range(5))
    lengths = [len(texts[i]) for i in flat]
    assert lengths == sorted(lengths, reverse=True)


def test_run_in_length_order_restores_input_order():
    texts = ["aaa", "b", "cc", "dddd"]
    out = run_in_length_order(texts, 3, 1, lambda batch: np.array([[len(t)] for t in batch]))
    assert out[:, 0].tolist() == [3, 1, 2, 4]
    with pytest.raises(ValueError, match="shape"):
        run_in_length_order(texts, 2, 2, lambda batch: np.zeros((len(batch), 1)))


def test_softmax64_rows_sum_to_one_and_is_shift_invariant():
    z = np.array([[1.0, 2.0, 3.0], [1000.0, 1001.0, 1002.0]])
    p = softmax64(z)
    assert p.dtype == np.float64
    np.testing.assert_allclose(p.sum(1), 1.0, atol=1e-15)
    np.testing.assert_allclose(p[0], p[1], atol=1e-15)


def test_weights_hash_ignores_hidden_cache(finbert_like_dir):
    h = weights_sha256(finbert_like_dir)
    (finbert_like_dir / ".cache" / "meta").write_text("other machine")
    assert weights_sha256(finbert_like_dir) == h


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------


def _direct_probs(model_dir, texts):
    """Reference: one text at a time, no batching, fp32 softmax."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).eval()
    rows = []
    with torch.inference_mode():
        for t in texts:
            logits = model(**tok([t], return_tensors="pt")).logits.double()
            rows.append(torch.softmax(logits, -1).numpy()[0])
    return np.stack(rows)


def test_classify_matches_unbatched_forward_in_input_order(finbert_like_dir):
    b = LocalBackend(finbert_like_dir, task="classification", dtype="float32", device="cpu",
                     batch_size=4)
    assert b.labels() == ["positive", "negative", "neutral"]      # id2label index order
    p = b.classify(TEXTS)
    assert p.shape == (len(TEXTS), 3) and p.dtype == np.float64
    np.testing.assert_allclose(p.sum(1), 1.0, atol=1e-12)
    np.testing.assert_allclose(p, _direct_probs(finbert_like_dir, TEXTS), atol=1e-5)


def test_classifier_backend_refuses_embedding(ravenbert_like_dir):
    b = LocalBackend(ravenbert_like_dir, task="classification", dtype="float32", device="cpu")
    assert len(b.labels()) == 41
    with pytest.raises(NotImplementedError):
        b.embed(["x"])
    with pytest.raises(NotImplementedError):
        b.embedding_config()


def test_classifier_info_and_identity(finbert_like_dir):
    b = LocalBackend(finbert_like_dir, task="classification", dtype="float32", device="cpu")
    info = b.info()
    assert info["engine"] == "local" and info["dtype"] == "float32"
    assert info["id2label"] == {0: "positive", 1: "negative", 2: "neutral"}
    assert info["weights_sha256"] == weights_sha256(finbert_like_dir)
    assert b.identity()["n_labels"] == 3


# ---------------------------------------------------------------------------
# embedding
# ---------------------------------------------------------------------------


def test_embed_unit_rows_match_sentence_transformers(embedder_dir):
    from sentence_transformers import SentenceTransformer

    b = LocalBackend(embedder_dir, task="embedding", dtype="float32", device="cpu", batch_size=2)
    cfg = b.embedding_config()
    assert (cfg.dim, cfg.pooling) == (16, "cls")
    X = b.embed(TEXTS)
    assert X.shape == (len(TEXTS), 16) and X.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(X, axis=1), 1.0, atol=1e-6)
    ref = SentenceTransformer(str(embedder_dir), device="cpu").encode(
        TEXTS, normalize_embeddings=True)
    np.testing.assert_allclose(X, ref, atol=1e-5)
    assert b.embed([]).shape == (0, 16)


def test_mean_pooled_directory_reports_mean(mean_embedder_dir):
    b = LocalBackend(mean_embedder_dir, task="embedding", dtype="float32", device="cpu")
    assert b.embedding_config().pooling == "mean"


def test_embedder_backend_refuses_classification(embedder_dir):
    b = LocalBackend(embedder_dir, task="embedding", dtype="float32", device="cpu")
    with pytest.raises(NotImplementedError):
        b.classify(["x"])
    assert b.info()["pooling"] == "cls"


def test_bad_task_and_missing_dir(tmp_path, embedder_dir):
    with pytest.raises(ValueError, match="task"):
        LocalBackend(embedder_dir, task="rerank", dtype="float32")
    with pytest.raises(FileNotFoundError):
        LocalBackend(tmp_path / "nope", task="embedding")


def test_space_and_empty_text_embed_identically(embedder_dir):
    """Gate for TEI's "" -> " " substitution: BERT tokenizers strip whitespace,
    so both are [CLS] [SEP] and the embedding is bit-identical."""
    b = LocalBackend(embedder_dir, task="embedding", dtype="float32", device="cpu")
    X = b.embed(["", " "])
    np.testing.assert_array_equal(X[0], X[1])
