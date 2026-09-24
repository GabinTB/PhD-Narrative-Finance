"""Model sentiment: the probability -> SENT_* mappings (hand-checked), the classifier on
tiny random-weight BERTs built here (no real weights), local vs Ray, end to end."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from ravenpack.headlines.sentiment import validate_sentiment_frame, verify_artifact
from ravenpack.headlines.sentiment_model import (
    MODELS,
    RAVENBERT_CLASSES,
    THIRD,
    Classifier,
    LocalBackend,
    RayBackend,
    finbert_band,
    finbert_columns,
    ingest_to_datalake,
    producer,
    ravenbert_columns,
    weights_sha256,
)

# ---------------------------------------------------------------------------
# FinBERT band mapping
# ---------------------------------------------------------------------------


def _band(neg, neu, pos) -> tuple[float, float]:
    s, lab = finbert_band(np.array([neg]), np.array([neu]), np.array([pos]))
    return float(s[0]), float(lab[0])


class TestFinbertBand:
    def test_hand_checked_values(self):
        s, lab = _band(0.05, 0.50, 0.45)          # argmax neutral: (0.40 / 0.45) / 3
        assert s == pytest.approx(0.4 / 0.45 / 3) and s == pytest.approx(0.2963, abs=1e-4)
        assert lab == 0.0
        s, lab = _band(0.10, 0.20, 0.70)          # positive: 1/3 + (2/3)(0.70 - 0.20)
        assert s == pytest.approx(2 / 3) and lab == 1.0
        s, lab = _band(0.70, 0.20, 0.10)
        assert s == pytest.approx(-2 / 3) and lab == -1.0
        assert _band(0.0, 0.0, 1.0) == (pytest.approx(1.0), 1.0)
        assert _band(1.0, 0.0, 0.0) == (pytest.approx(-1.0), -1.0)
        assert _band(0.0, 1.0, 0.0) == (0.0, 0.0)
        s, _ = _band(0.30, 0.40, 0.30)             # symmetric neutral
        assert s == 0.0

    def test_ties(self):
        assert _band(0.2, 0.4, 0.4) == (pytest.approx(THIRD), 1.0)     # neu/pos tie -> +1/3
        assert _band(0.4, 0.4, 0.2) == (pytest.approx(-THIRD), -1.0)
        assert _band(0.45, 0.10, 0.45) == (0.0, 0.0)                   # pos/neg tie
        assert _band(1 / 3, 1 / 3, 1 / 3) == (0.0, 0.0)                # triple point

    @pytest.mark.parametrize("p_neg", [0.0, 0.1, 0.25])
    def test_continuous_across_neutral_polar_boundaries(self, p_neg):
        rest = 1.0 - p_neg
        for sign, flip in ((1, False), (-1, True)):
            # approach the neu == pole tie from both sides
            for e in (1e-7, 1e-9):
                a, b = (rest / 2 + e, rest / 2 - e), (rest / 2 - e, rest / 2 + e)
                for pole, neu in (a, b):
                    probs = (p_neg, neu, pole) if not flip else (pole, neu, p_neg)
                    s, _ = _band(*probs)
                    assert s == pytest.approx(sign * THIRD, abs=1e-6)

    def test_band_matches_argmax_on_random_simplex(self):
        rng = np.random.default_rng(0)
        p = rng.dirichlet([0.7, 0.7, 0.7], size=200_000)
        s, lab = finbert_band(p[:, 0], p[:, 1], p[:, 2])
        arg = np.array([-1.0, 0.0, 1.0])[np.argmax(p, axis=1)]
        np.testing.assert_array_equal(lab, arg)
        assert (s[lab == 1] >= THIRD).all() and (s[lab == -1] <= -THIRD).all()
        assert (np.abs(s[lab == 0]) < THIRD).all()
        assert (np.abs(s) <= 1.0).all()

    def test_columns_and_dtypes(self):
        cols = finbert_columns(np.array([[0.1, 0.2, 0.7], [0.3, 0.4, 0.3]]))
        assert list(cols) == ["SENT_BAND", "SENT_EV", "SENT_ARGMAX", "P_NEG", "P_NEU", "P_POS"]
        assert cols["SENT_EV"].tolist() == pytest.approx([0.6, 0.0])
        assert cols["SENT_ARGMAX"].tolist() == [1.0, 0.0]
        assert cols["P_POS"].dtype == np.float16 and cols["SENT_BAND"].dtype == np.float32
        with pytest.raises(ValueError, match=r"\(n, 3\)"):
            finbert_columns(np.ones((2, 4)) / 4)
        with pytest.raises(ValueError, match="finite"):
            finbert_columns(np.array([[np.nan, 0.5, 0.5]]))


class TestRavenbertColumns:
    def test_hand_checked(self):
        p = np.zeros((2, 41))
        p[0, [40, 39, 20, 0]] = [0.5, 0.2, 0.2, 0.1]      # s = 1, 0.95, 0, -1
        p[1, 20] = 1.0
        cols = ravenbert_columns(p)
        assert cols["SENT_EV"][0] == pytest.approx(0.5 + 0.2 * 0.95 - 0.1)
        assert cols["SENT_ARGMAX"].tolist() == [1.0, 0.0]
        # top 3 = {1.0: .5, 0.95: .2, 0.0: .2} (stable: class 20 before class 0 at equal p? no,
        # p=.2 ties between classes 39 and 20; both are in the top 3) -> renormalised by .9
        assert cols["SENT_TOP3"][0] == pytest.approx((0.5 + 0.2 * 0.95) / 0.9, rel=1e-6)
        assert cols["SENT_EV"][1] == 0.0 and cols["SENT_TOP3"][1] == 0.0
        rng = np.random.default_rng(1)
        q = rng.dirichlet(np.ones(41), size=1000)
        c = ravenbert_columns(q)
        np.testing.assert_allclose(c["SENT_EV"], q @ RAVENBERT_CLASSES, rtol=1e-6, atol=1e-7)
        assert all((np.abs(v) <= 1).all() and v.dtype == np.float32 for v in c.values())


# ---------------------------------------------------------------------------
# Tiny random-weight classifiers
# ---------------------------------------------------------------------------

WORDS = ["stocks", "rally", "plunge", "bank", "profit", "loss", "fed", "rates", "up", "down"]


def _tiny_model(root: Path, id2label: dict[int, str]) -> Path:
    import torch
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

    root.mkdir(parents=True)
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *WORDS]
    (root / "vocab.txt").write_text("\n".join(vocab) + "\n")
    BertTokenizerFast(vocab_file=str(root / "vocab.txt")).save_pretrained(str(root))
    torch.manual_seed(0)
    cfg = BertConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1,
                     num_attention_heads=2, intermediate_size=32, max_position_embeddings=64,
                     num_labels=len(id2label), id2label=id2label,
                     label2id={v: k for k, v in id2label.items()})
    BertForSequenceClassification(cfg).save_pretrained(str(root))
    (root / ".cache").mkdir()                                  # like a HF --local-dir download
    (root / ".cache" / "meta").write_text("machine-specific")
    return root


@pytest.fixture(scope="module")
def ravenbert_dir(tmp_path_factory) -> Path:
    return _tiny_model(tmp_path_factory.mktemp("rb") / "m",
                       {i: f"LABEL_{i}" for i in range(41)})


@pytest.fixture(scope="module")
def finbert_dir(tmp_path_factory) -> Path:
    # deliberately not in [neg, neu, pos] order
    return _tiny_model(tmp_path_factory.mktemp("fb") / "m",
                       {0: "positive", 1: "negative", 2: "neutral"})


TEXTS = ["stocks rally", "bank loss down >AAPL", "", "fed rates up up up rally profit",
         "plunge", "profit\nloss -- Reuters"]


def test_ravenbert_ev_equals_sentiment_model_predict(ravenbert_dir):
    from ravenbert.sentiment.model import SentimentModel, clean_text

    ref = SentimentModel.from_path(ravenbert_dir, device="cpu", precision="fp32")
    want = ref.predict(TEXTS, batch_size=64, optimized_inference=False)
    clf = Classifier(MODELS["ravenbert"], ravenbert_dir, device="cpu", batch_size=2)
    got = ravenbert_columns(clf.probs([clean_text(t) for t in TEXTS]))["SENT_EV"]
    np.testing.assert_allclose(got, want, atol=1e-6)


def test_finbert_class_order_read_from_id2label(finbert_dir):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    clf = Classifier(MODELS["finbert"], finbert_dir, device="cpu")
    got = clf.probs(TEXTS)
    tok = AutoTokenizer.from_pretrained(str(finbert_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(finbert_dir)).eval()
    with torch.inference_mode():
        raw = torch.softmax(model(**tok(TEXTS, padding=True, return_tensors="pt")).logits,
                            -1).numpy()
    np.testing.assert_allclose(got, raw[:, [1, 2, 0]], atol=1e-6)     # neg, neu, pos
    np.testing.assert_allclose(got.sum(axis=1), 1.0, atol=1e-6)


def test_bad_label_sets_are_refused(tmp_path):
    bad = _tiny_model(tmp_path / "bad", {0: "bullish", 1: "bearish", 2: "neutral"})
    with pytest.raises(ValueError, match="negative/neutral/positive"):
        Classifier(MODELS["finbert"], bad, device="cpu")
    three = _tiny_model(tmp_path / "three", {0: "negative", 1: "neutral", 2: "positive"})
    with pytest.raises(ValueError, match="41"):
        Classifier(MODELS["ravenbert"], three, device="cpu")


def test_weights_hash_ignores_hidden_cache(ravenbert_dir, tmp_path):
    import shutil

    copy = tmp_path / "copy"
    shutil.copytree(ravenbert_dir, copy)
    (copy / ".cache" / "meta").write_text("other machine")
    assert weights_sha256(copy) == weights_sha256(ravenbert_dir)
    (copy / "vocab.txt").write_text("changed")
    assert weights_sha256(copy) != weights_sha256(ravenbert_dir)


@pytest.fixture(scope="module")
def local_ray():
    ray = pytest.importorskip("ray")
    from ray._private import ray_constants

    # Under `uv run`, Ray packages the whole project as the workers' working_dir; a local
    # test cluster shares this venv and needs none of it.
    saved = ray_constants.RAY_ENABLE_UV_RUN_RUNTIME_ENV
    ray_constants.RAY_ENABLE_UV_RUN_RUNTIME_ENV = False
    ray.init(num_cpus=2, include_dashboard=False, log_to_driver=False)
    yield ray
    ray.shutdown()
    ray_constants.RAY_ENABLE_UV_RUN_RUNTIME_ENV = saved


def test_ray_backend_matches_local_and_checks_weights(finbert_dir, local_ray):
    spec = MODELS["finbert"]
    local = LocalBackend(spec, finbert_dir, device="cpu", batch_size=4)
    texts = TEXTS * 7
    remote = RayBackend(spec, local.weights_sha256, address=None, n_actors=2,
                        gpus_per_actor=0, model_path=str(finbert_dir), batch_size=4,
                        slice_rows=5)
    np.testing.assert_allclose(remote.probs(texts), local.probs(texts), atol=1e-6)
    assert remote.describe().startswith("ray:2x")
    with pytest.raises(RuntimeError, match="different finbert weights"):
        RayBackend(spec, "0" * 64, address=None, n_actors=1, gpus_per_actor=0,
                   model_path=str(finbert_dir))


def test_end_to_end_finbert_artifact(finbert_dir, tmp_path):
    from datalake import DatalakeIndex

    dl = DatalakeIndex(tmp_path / "lake")
    with dl.run(kind="ravenpack_headlines", pipeline="t", pipeline_version="v0") as r:
        pl.DataFrame({"RP_STORY_ID": [f"s{i}" for i in range(len(TEXTS))],
                      "HEADLINE": [t or None for t in TEXTS]}) \
            .write_parquet(r.out_dir / "2008-01.parquet")
    spec = MODELS["finbert"]
    backend = LocalBackend(spec, finbert_dir, device="cpu", batch_size=4)
    art = ingest_to_datalake(dl, source=spec.name, columns=list(spec.columns),
                             produce=producer(spec, backend, read_rows=4),
                             headlines=dl.latest("ravenpack_headlines"), start_year=2008,
                             end_year=2008, extra_hyperparams={"weights_sha": "x"}, temp=True)
    out = pl.read_parquet(art.path / "2008-01.parquet")
    assert validate_sentiment_frame(out) == list(spec.columns)
    assert out.schema["P_NEU"] == pl.Float16 and out.height == len(TEXTS)
    assert not out["SENT_BAND"].is_nan().any()                  # a null headline still scores
    assert verify_artifact(art) == []
    dl.close()
