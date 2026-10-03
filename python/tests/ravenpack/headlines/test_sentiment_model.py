"""The model-sentiment job: Sentimeter -> headline_sentiment artifact, end to end.

The grid remap, the load-time rules and compatibility checks are tested in tests/nlp;
here the RavenPack side: the table is the Float16 41-grid of the headline_sentiment
contract (all null for an empty headline), the card and hyperparams record the
engine, finalize writes the per-year summary, and a resume refuses a sentimeter that
produces differently.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from nlp.backends.local import LocalBackend
from nlp.sentiment import GRID_COLUMNS, FinbertSentimeter, RavenbertSentimeter, to_grid
from ravenpack.headlines.sentiment import (
    SUMMARY_FILE,
    ingest_to_datalake,
    validate_sentiment_frame,
    verify_artifact,
)
from ravenpack.headlines.sentiment_model import (
    check_resume_compatible,
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
    with dl.run(kind="rp_headlines", pipeline="t", pipeline_version="v0") as r:
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
                             headlines=dl.latest("rp_headlines"), start_year=2008,
                             end_year=2008, extra_hyperparams=run_hyperparams(sentimeter),
                             model_card=sentimeter.model_card(), temp=True)
    out = pl.read_parquet(art.path / "2008-01.parquet")
    assert validate_sentiment_frame(out) == list(GRID_COLUMNS)
    assert out.columns == ["RP_STORY_ID", *GRID_COLUMNS]
    assert all(out.schema[c] == pl.Float16 for c in GRID_COLUMNS)
    empty = out["RP_STORY_ID"] == "s4"                            # the None headline
    assert out.filter(empty).select(pl.all_horizontal(pl.col(GRID_COLUMNS).is_null())).item()
    G = out.filter(~empty).select(GRID_COLUMNS).to_numpy().astype(np.float64)
    np.testing.assert_allclose(G.sum(axis=1), 1.0, atol=1e-2)
    card = art.meta.model_card
    assert card.backend == "local" and card.model_id == f"{sentimeter.name}-sentiment"
    assert card.weights_sha256 and card.serving["sentimeter"]["band_edges"]
    hp = art.meta.hyperparams
    assert hp["backend"] == "local" and hp["dtype"] == "float32" and hp["model_version"] == "2.1"
    assert "score_rule" not in hp and "min_confidence" not in hp
    summary = (art.path / SUMMARY_FILE).read_text()
    assert '"2008"' in summary and '"median"' in summary and '"null_share"' in summary
    assert verify_artifact(art) == []
    dl.close()


def test_grid_rows_are_the_remapped_classes(finbert_dir, tmp_path):
    sentimeter = FinbertSentimeter(_local(finbert_dir))
    month = tmp_path / "2008-01.parquet"
    pl.DataFrame({"RP_STORY_ID": ["a", "b", "c"], "HEADLINE": ["stocks rally", "", "loss"]}) \
        .write_parquet(month)
    frame = producer(sentimeter)(month, "t")
    validate_sentiment_frame(frame)
    G = frame.select(GRID_COLUMNS).to_numpy().astype(np.float64)
    assert np.isnan(G[1]).all()                                  # empty headline: all null
    want = to_grid(sentimeter.canonical(["stocks rally", "loss"]), sentimeter.band_edges)
    np.testing.assert_allclose(G[[0, 2]], want, rtol=2 ** -10, atol=1e-6)   # Float16


def test_resume_refuses_a_different_sentimeter(finbert_dir):
    a = FinbertSentimeter(_local(finbert_dir))
    hp = {"source": "finbert", **run_hyperparams(a)}
    check_resume_compatible(a.model_card(), hp, a)
    for other in ({**hp, "dtype": "float16"}, {**hp, "model_version": "2.0"},
                  {**hp, "source": "ravenbert"}):
        with pytest.raises(ValueError, match="refusing to mix"):
            check_resume_compatible(a.model_card(), other, a)
    with pytest.raises(ValueError, match="no model card"):
        check_resume_compatible(None, hp, a)


def test_killed_run_resumes_from_its_card_alone(finbert_dir, tmp_path):
    """Everything comes back from the artifact: Sentimeter family, settings, backend
    (same weights); only the missing month is scored."""
    from datalake import DatalakeIndex
    from nlp.sentiment import sentimeter_from_card
    from ravenpack.headlines.sentiment import resume_partial

    dl = DatalakeIndex(tmp_path / "lake")
    with dl.run(kind="rp_headlines", pipeline="t", pipeline_version="v0") as r:
        for month, ids in (("2008-01", ["a", "b"]), ("2008-02", ["x1", "x2"])):
            pl.DataFrame({"RP_STORY_ID": ids, "HEADLINE": ["stocks up", "loss"]}) \
                .write_parquet(r.out_dir / f"{month}.parquet")
    hl = dl.latest("rp_headlines")
    sentimeter = FinbertSentimeter(_local(finbert_dir))
    good = producer(sentimeter)

    def dies_on_february(month, tag):
        if month.name == "2008-02.parquet":
            raise RuntimeError("killed")
        return good(month, tag)

    with pytest.raises(RuntimeError, match="killed"):
        ingest_to_datalake(dl, source=sentimeter.name, columns=sentimeter.columns(),
                           produce=dies_on_february, headlines=hl, start_year=2008,
                           end_year=2008, extra_hyperparams=run_hyperparams(sentimeter),
                           model_card=sentimeter.model_card(), temp=True)
    part = dl.list("headline_sentiment", include_partial=True)[0]
    assert part.partial and [p.name for p in part.path.glob("*.parquet")] == ["2008-01.parquet"]

    again = sentimeter_from_card(part.meta.model_card, model_path=str(finbert_dir), device="cpu")
    assert again.band_edges == sentimeter.band_edges
    check_resume_compatible(part.meta.model_card, part.meta.hyperparams, again)
    done = resume_partial(dl, part.artifact_id, producer(again))
    assert not done.partial
    assert sorted(p for p in done.file_hashes if p.endswith(".parquet")) == [
        "2008-01.parquet", "2008-02.parquet"]
    assert pl.read_parquet(done.path / "2008-02.parquet").columns == ["RP_STORY_ID",
                                                                      *GRID_COLUMNS]
    dl.close()


def test_jobs_resume_rebuilds_the_sentimeter_and_stops_on_a_model_switch(finbert_dir,
                                                                       tmp_path):
    """``JobRunner.resume`` alone: the Sentimeter comes from the card; a backend whose
    served model changes before a partition fails the job with that reason."""
    from datalake import DatalakeIndex
    from datalake.jobs import JobRunner, JobState
    from nlp.backends.base import IncompatibleModelError
    from ravenpack.headlines import sentiment_model as sm
    from ravenpack.headlines.sentiment import sentiment_layout

    dl = DatalakeIndex(tmp_path / "lake")
    with dl.run(kind="rp_headlines", pipeline="t", pipeline_version="v0",
                hyperparams={"start_year": 2008, "end_year": 2008}) as r:
        for month, ids in (("2008-01", ["a", "b"]), ("2008-02", ["x1", "x2"])):
            pl.DataFrame({"RP_STORY_ID": ids, "HEADLINE": ["stocks up", "loss"]}) \
                .write_parquet(r.out_dir / f"{month}.parquet")
    hl = dl.latest("rp_headlines")
    backend = _local(finbert_dir)
    switched = {"after": "2008-01"}
    real_check = type(backend).check_unchanged

    def check_unchanged(self):
        if switched["after"] and (dl.list("headline_sentiment", include_partial=True)[0].path
                                  / "2008-01.parquet").exists():
            raise IncompatibleModelError("served model changed: finbert -> other")
        return real_check(self)

    backend.check_unchanged = check_unchanged.__get__(backend)
    job = sm.job(hl, sentiment_layout(hl), FinbertSentimeter(backend), temp=True)
    runner = JobRunner(dl, allow_dirty=True, handle_signals=False)
    with pytest.raises(IncompatibleModelError, match="served model changed"):
        runner.start(job)
    part = dl.list("headline_sentiment", include_partial=True)[0]
    state = JobState.read(part.path)
    assert state.status == "failed" and state.units_done == 1
    assert "served model changed" in state.last_error
    done = runner.resume(part.artifact_id, model_path=finbert_dir, device="cpu")
    assert not done.partial
    assert sorted(p for p in done.file_hashes if p.endswith(".parquet")) == [
        "2008-01.parquet", "2008-02.parquet"]
    dl.close()
