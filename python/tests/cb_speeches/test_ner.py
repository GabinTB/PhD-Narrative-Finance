"""cb_speeches.ner: NER table keyed by speech_id, updates, resume, verifier."""
from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from cb_speeches import ner
from cb_speeches.ner import NerJob, read_ner, speeches_with_ner, verify_artifact
from cb_speeches.speeches import read_speeches
from datalake import DatalakeIndex
from datalake.jobs import JobError, JobRunner
from tests.cb_speeches.fakes import FakeBIS, FakeChat, chat_backend, every_third_day, speech
from tests.cb_speeches.test_speeches import T0, Clock, _job


def _runner(index):
    return JobRunner(index, allow_dirty=True, handle_signals=False, heartbeat_s=0.05)


@pytest.fixture
def env(tmp_path):
    fake = FakeBIS({2008: every_third_day(date(2008, 1, 1), date(2008, 3, 31))})
    with DatalakeIndex(tmp_path / "lake") as index:
        speeches = _runner(index).start(_job(fake, end=date(2008, 3, 31)))
        yield index, fake, speeches


def _ner(index, speeches, chat, **kw):
    job = NerJob.new(speeches, chat_backend(chat, **kw), temp=True,
                     clock=Clock(T0 + timedelta(hours=1)))
    return _runner(index).start(job)


def test_one_row_per_speech_keyed_by_speech_id(env):
    index, _, speeches = env
    chat = FakeChat()
    art = _ner(index, speeches, chat)
    df = read_ner(art)
    assert df["speech_id"].to_list() == read_speeches(speeches)["speech_id"].to_list()
    assert df["error"].null_count() == df.height and chat.calls == df.height
    assert df["response_model"].unique().to_list() == [chat.model]
    assert sorted(p.name for p in art.path.glob("*.parquet")) == [
        "2008-01.parquet", "2008-02.parquet", "2008-03.parquet"]
    hp = art.meta.hyperparams
    assert hp["speeches_id"] == speeches.artifact_id and hp["provider"] == "openai"
    assert hp["prompt_sha256"] == ner.prompt_sha256()
    assert art.meta.sources == [speeches.artifact_id]
    assert art.meta.model_card.backend == "openai"
    assert "sk-test-secret" not in (art.path / "meta.json").read_text()
    joined = speeches_with_ner(speeches, art)
    assert joined["organization"].unique().to_list() == ["Bank of Testland"]
    assert verify_artifact(art) == []


def test_failed_extractions_are_error_rows(env):
    index, _, speeches = env
    art = _ner(index, speeches, FakeChat(bad_replies=3), max_attempts=3, workers=1)
    df = read_ner(art)
    assert df.filter(pl.col("error").is_not_null()).height == 1
    findings = verify_artifact(art)
    assert [f.severity.value for f in findings] == ["warning"]


def test_update_only_runs_new_or_changed_metadata(env):
    index, fake, speeches = env
    chat = FakeChat()
    art = _ner(index, speeches, chat)
    first = chat.calls
    rows = fake.rows[2008]
    rows[0] = dict(rows[0], text="new text only")                 # not a prompt input
    rows[1] = dict(rows[1], title="A retitled speech")            # a prompt input
    rows.append(speech(date(2008, 3, 31), 6))
    _runner(index).update(speeches.artifact_id, transport=fake.transport(),
                          clock=Clock(T0 + timedelta(days=1)))
    done = _runner(index).update(art.artifact_id, http_client=chat.client(),
                                 base_url="http://llm.test/v1", api_key="k", max_retries=0,
                                 clock=Clock(T0 + timedelta(days=2)))
    assert chat.calls - first == 2
    assert len(done.meta.runs) == 2
    df = read_ner(done)
    assert df.height == read_speeches(index.get(speeches.artifact_id)).height
    assert verify_artifact(done) == []
    again = _runner(index).update(art.artifact_id, http_client=chat.client(),
                                  base_url="http://llm.test/v1", api_key="k", max_retries=0,
                                  clock=Clock(T0 + timedelta(days=3)))
    assert chat.calls - first == 2 and len(again.meta.runs) == 3


def test_resume_is_pinned_to_the_recorded_snapshot(env):
    index, _, speeches = env
    chat = FakeChat()
    job = NerJob.new(speeches, chat_backend(chat), temp=True, clock=Clock(T0))
    real, calls = job.run_unit, [0]

    def run_unit(unit, ctx):
        calls[0] += 1
        if calls[0] == 2:
            raise RuntimeError("killed")
        real(unit, ctx)
    job.run_unit = run_unit
    with pytest.raises(RuntimeError):
        _runner(index).start(job)
    aid = index.list("cb_speech_ner", include_partial=True)[0].artifact_id
    chat.model = "fake-llm-2026-06-01"
    with pytest.raises(Exception, match="model"):
        _runner(index).resume(aid, http_client=chat.client(), base_url="http://llm.test/v1",
                              api_key="k", max_retries=0)
    chat.model = "fake-llm-2026-01-01"
    done = _runner(index).resume(aid, http_client=chat.client(),
                                 base_url="http://llm.test/v1", api_key="k", max_retries=0)
    assert not done.partial and read_ner(done).height == read_speeches(speeches).height


def test_a_changed_prompt_cannot_resume_or_update(env, monkeypatch):
    index, _, speeches = env
    chat = FakeChat()
    art = _ner(index, speeches, chat)
    monkeypatch.setattr(ner, "SYSTEM_PROMPT", ner.SYSTEM_PROMPT + " Be terse.")
    with pytest.raises(JobError, match="prompt"):
        _runner(index).update(art.artifact_id, http_client=chat.client(),
                              base_url="http://llm.test/v1", api_key="k")
