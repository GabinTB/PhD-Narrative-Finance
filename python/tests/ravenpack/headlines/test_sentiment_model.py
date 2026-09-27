"""The model-sentiment job: Sentimeter -> headline_sentiment artifact, end to end.

Score maps, confidence and compatibility checks are tested in tests/nlp; here
the RavenPack side: the month table obeys the headline_sentiment contract,
canonical columns are optional, the card and hyperparams record the engine,
and a resume refuses a sentimeter that scores differently.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from nlp.backends.local import LocalBackend
from nlp.sentiment import FinbertSentimeter, RavenbertSentimeter
from ravenpack.headlines.sentiment import validate_sentiment_frame, verify_artifact
from ravenpack.headlines.sentiment_model import (
    check_resume_compatible,
    ingest_to_datalake,
    producer,
    run_hyperparams,
)

WORDS = ["stocks", "rally", "plunge", "bank", "profit", "loss", "fed", "rates", "up", "down"]
TEXTS = ["stocks rally", "bank loss plunge down", "fed rates up", "profit profit", None,
         "loss >AAPL -- Reuters"]


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
    return root


@pytest.fixture(scope="module")
def finbert_dir(tmp_path_factory) -> Path:
    return _tiny_model(tmp_path_factory.mktemp("fb") / "m",
                       {0: "positive", 1: "negative", 2: "neutral"})


@pytest.fixture(scope="module")
def ravenbert_dir(tmp_path_factory) -> Path:
    return _tiny_model(tmp_path_factory.mktemp("rb") / "m",
                       {i: f"LABEL_{i}" for i in range(41)})


def _local(model_dir: Path) -> LocalBackend:
    return LocalBackend(model_dir, task="classification", dtype="float32", device="cpu",
                        batch_size=4)


def _lake_with_headlines(tmp_path: Path):
    from datalake import DatalakeIndex

    dl = DatalakeIndex(tmp_path / "lake")
    with dl.run(kind="ravenpack_headlines", pipeline="t", pipeline_version="v0") as r:
        pl.DataFrame({"RP_STORY_ID": [f"s{i}" for i in range(len(TEXTS))],
                      "HEADLINE": TEXTS}).write_parquet(r.out_dir / "2008-01.parquet")
    return dl


@pytest.mark.parametrize("cls,fixture", [(FinbertSentimeter, "finbert_dir"),
                                         (RavenbertSentimeter, "ravenbert_dir")])
def test_end_to_end_artifact(cls, fixture, request, tmp_path):
    sentimeter = cls(_local(request.getfixturevalue(fixture)))
    dl = _lake_with_headlines(tmp_path)
    art = ingest_to_datalake(dl, source=sentimeter.name, columns=sentimeter.columns(),
                             produce=producer(sentimeter, read_rows=4),
                             headlines=dl.latest("ravenpack_headlines"), start_year=2008,
                             end_year=2008, extra_hyperparams=run_hyperparams(sentimeter),
                             model_card=sentimeter.model_card(), temp=True)
    out = pl.read_parquet(art.path / "2008-01.parquet")
    assert validate_sentiment_frame(out) == ["SENT_SCORE", "SENT_CONF"]
    assert out.height == len(TEXTS) and not out["SENT_SCORE"].is_nan().any()
    conf = out["SENT_CONF"].to_numpy()
    assert ((conf >= 0) & (conf <= 1)).all()
    card = art.meta.model_card
    assert card.backend == "local" and card.model_id == f"{sentimeter.name}-sentiment"
    assert card.weights_sha256 and card.serving["sentimeter"]["score_rule"]
    hp = art.meta.hyperparams
    assert hp["backend"] == "local" and hp["dtype"] == "float32" and hp["model_version"] == "2.0"
    assert verify_artifact(art) == []
    dl.close()


def test_canonical_columns_are_optional_float16(finbert_dir, tmp_path):
    sentimeter = FinbertSentimeter(_local(finbert_dir))
    month = tmp_path / "2008-01.parquet"
    pl.DataFrame({"RP_STORY_ID": ["a", "b"], "HEADLINE": ["stocks rally", "loss"]}) \
        .write_parquet(month)
    plain = producer(sentimeter)(month, "t")
    full = producer(sentimeter, canonical_columns=True)(month, "t")
    assert plain.columns == ["RP_STORY_ID", "SENT_SCORE", "SENT_CONF"]
    assert full.columns[-3:] == ["P_NEG", "P_NEU", "P_POS"]
    assert full.schema["P_NEU"] == pl.Float16
    np.testing.assert_allclose(full.select(["P_NEG", "P_NEU", "P_POS"]).to_numpy().sum(1),
                               1.0, atol=2e-3)
    validate_sentiment_frame(full)


def test_min_confidence_nan_passes_the_contract(ravenbert_dir, tmp_path):
    sentimeter = RavenbertSentimeter(_local(ravenbert_dir), min_confidence=1.0)
    month = tmp_path / "2008-01.parquet"
    pl.DataFrame({"RP_STORY_ID": ["a", "b"], "HEADLINE": ["stocks", "loss"]}) \
        .write_parquet(month)
    frame = producer(sentimeter)(month, "t")
    assert frame["SENT_SCORE"].is_nan().all()           # nothing is fully certain
    validate_sentiment_frame(frame)
    assert run_hyperparams(sentimeter)["min_confidence"] == 1.0


def test_resume_refuses_a_different_sentimeter(finbert_dir):
    a = FinbertSentimeter(_local(finbert_dir))
    hp = {"source": "finbert", **run_hyperparams(a)}
    check_resume_compatible(a.model_card(), hp, a)
    b = FinbertSentimeter(_local(finbert_dir), min_confidence=0.2)
    with pytest.raises(ValueError, match="refusing to mix"):
        check_resume_compatible(a.model_card(), hp, b)
    with pytest.raises(ValueError, match="no model card"):
        check_resume_compatible(None, hp, a)


def test_killed_run_resumes_from_its_card_alone(finbert_dir, tmp_path):
    """Everything comes back from the artifact: Sentimeter family, settings, backend
    (same weights), canonical-column choice; only the missing month is scored."""
    from datalake import DatalakeIndex
    from nlp.sentiment import sentimeter_from_card
    from ravenpack.headlines.sentiment import resume_partial

    dl = DatalakeIndex(tmp_path / "lake")
    with dl.run(kind="ravenpack_headlines", pipeline="t", pipeline_version="v0") as r:
        for month, ids in (("2008-01", ["a", "b"]), ("2008-02", ["x1", "x2"])):
            pl.DataFrame({"RP_STORY_ID": ids, "HEADLINE": ["stocks up", "loss"]}) \
                .write_parquet(r.out_dir / f"{month}.parquet")
    hl = dl.latest("ravenpack_headlines")
    sentimeter = FinbertSentimeter(_local(finbert_dir), min_confidence=0.05)
    good = producer(sentimeter, canonical_columns=True)

    def dies_on_february(month, tag):
        if month.name == "2008-02.parquet":
            raise RuntimeError("killed")
        return good(month, tag)

    with pytest.raises(RuntimeError, match="killed"):
        ingest_to_datalake(dl, source=sentimeter.name, columns=sentimeter.columns(),
                           produce=dies_on_february, headlines=hl, start_year=2008,
                           end_year=2008, extra_hyperparams=run_hyperparams(sentimeter, True),
                           model_card=sentimeter.model_card(), temp=True)
    part = dl.list("headline_sentiment", include_partial=True)[0]
    assert part.partial and [p.name for p in part.path.glob("*.parquet")] == ["2008-01.parquet"]

    again = sentimeter_from_card(part.meta.model_card, model_path=str(finbert_dir), device="cpu")
    assert again.min_confidence == 0.05
    check_resume_compatible(part.meta.model_card, part.meta.hyperparams, again)
    done = resume_partial(dl, part.artifact_id,
                          producer(again, canonical_columns=part.meta.hyperparams[
                              "canonical_columns"]))
    assert not done.partial and sorted(done.file_hashes) == ["2008-01.parquet", "2008-02.parquet"]
    assert pl.read_parquet(done.path / "2008-02.parquet").columns[-3:] == ["P_NEG", "P_NEU",
                                                                          "P_POS"]
    dl.close()
