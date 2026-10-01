"""nlp.batched_job: each vendor's exact request payload and its own quirks."""
from __future__ import annotations

import pytest

from nlp.batched_job import ChatRequest
from nlp.batched_job.base import BatchStatus, deep_merge
from nlp.batched_job.google import _state
from tests.nlp.batched_job.fakes import (
    SCHEMA,
    FakeAnthropic,
    FakeGoogle,
    FakeMistral,
    FakeOpenAI,
    make_job,
)

REQ = [ChatRequest("a", "SYS", "Author: 'A'")]


def test_deep_merge_keeps_sibling_keys():
    assert deep_merge({"o": {"format": 1}, "t": 0}, {"o": {"effort": "low"}, "t": 1}) == {
        "o": {"format": 1, "effort": "low"}, "t": 1}


def test_openai_payload():
    fake = FakeOpenAI()
    make_job("openai", fake, temperature=0.0, seed=7, max_tokens=64,
             extra={"reasoning_effort": "low"}).run(REQ)
    line = fake.payloads["a"]
    assert line["method"] == "POST" and line["url"] == "/v1/chat/completions"
    body = line["body"]
    assert body["model"] == fake.model
    assert body["messages"] == [{"role": "system", "content": "SYS"},
                                {"role": "user", "content": "Author: 'A'"}]
    assert body["response_format"] == {"type": "json_schema", "json_schema": {
        "name": "ner", "schema": SCHEMA, "strict": True}}
    assert (body["temperature"], body["seed"], body["max_completion_tokens"]) == (0.0, 7, 64)
    assert body["reasoning_effort"] == "low"


def test_mistral_payload_puts_the_model_on_the_job():
    fake = FakeMistral()
    make_job("mistral", fake, seed=3).run(REQ)
    body = fake.payloads["a"]["body"]
    assert "model" not in body and body["random_seed"] == 3 and "seed" not in body
    assert body["response_format"]["json_schema"]["schema"] == SCHEMA
    assert fake.job == {"input_files": ["f-in"], "model": fake.model,
                        "endpoint": "/v1/chat/completions", "metadata": {"name": "ner"}}


def test_anthropic_payload_uses_native_structured_output():
    fake = FakeAnthropic()
    make_job("anthropic", fake, temperature=None,
             extra={"output_config": {"effort": "low"}}).run(REQ)
    params = fake.payloads["a"]
    assert params["system"] == "SYS"
    assert params["messages"] == [{"role": "user", "content": "Author: 'A'"}]
    assert params["output_config"] == {"format": {"type": "json_schema", "schema": SCHEMA},
                                       "effort": "low"}
    assert "temperature" not in params            # Claude 5: non-default temperature is a 400
    assert params["max_tokens"] == 4096


def test_anthropic_errored_lines_do_not_fail_the_batch():
    fake = FakeAnthropic(errored={"a"})
    job = make_job("anthropic", fake)
    handle = job.submit(REQ)
    assert job.wait(handle).status is BatchStatus.SUCCEEDED


def test_google_payload_and_native_root():
    fake = FakeGoogle()
    job = make_job("google", fake, temperature=0.0, seed=1, max_tokens=99,
                   extra={"generationConfig": {"thinkingConfig": {"thinkingLevel": "low"}}})
    out = job.run(REQ)
    assert fake.model_path == f"/v1beta/models/{fake.model}:batchGenerateContent"
    req = fake.payloads["a"]
    assert req["systemInstruction"] == {"parts": [{"text": "SYS"}]}
    gen = req["generationConfig"]
    assert gen["responseMimeType"] == "application/json" and gen["responseJsonSchema"] == SCHEMA
    assert (gen["temperature"], gen["seed"], gen["maxOutputTokens"]) == (0.0, 1, 99)
    assert gen["thinkingConfig"] == {"thinkingLevel": "low"}
    assert out[0].completion_tokens == 12                 # thoughts are billed output
    assert out[0].data["author"] == "A"                   # thought parts are not the reply


def test_google_states_in_both_spellings():
    assert _state("BATCH_STATE_SUCCEEDED") is BatchStatus.SUCCEEDED
    assert _state("JOB_STATE_PENDING") is BatchStatus.QUEUED
    assert _state("BATCH_STATE_EXPIRED") is BatchStatus.EXPIRED


def test_google_inline_size_cap(monkeypatch):
    import nlp.batched_job.google as g

    monkeypatch.setattr(g, "INLINE_LIMIT_BYTES", 100)
    with pytest.raises(ValueError, match="MB"):
        make_job("google", FakeGoogle()).submit(REQ)


@pytest.mark.parametrize("fake_cls, vendor", [(FakeGoogle, "google"), (FakeMistral, "mistral")])
def test_rest_vendors_retry_server_errors(fake_cls, vendor):
    assert make_job(vendor, fake_cls(fail_first=2)).run(REQ)[0].error is None
    with pytest.raises(RuntimeError, match="HTTP 500") as exc:
        make_job(vendor, fake_cls(fail_first=100)).run(REQ)
    assert "test" not in str(exc.value)                  # no URL in the error


def test_anthropic_rejected_temperature_is_resubmitted_without_it(caplog):
    fake = FakeAnthropic(reject_temperature=True)
    job = make_job("anthropic", fake, temperature=0.0)
    out = job.run([ChatRequest("a", "SYS", "Author: 'A'"), ChatRequest("b", "SYS",
                                                                        "Author: 'B'")])
    assert [r.data["author"] for r in out] == ["A", "B"]
    assert fake.submits == 2 and "temperature" not in fake.payloads["a"]
    assert job.temperature is None
    assert job.identity()["temperature_dropped"] is True
    assert "rejects temperature=0.0" in caplog.text


def test_temperature_none_is_never_sent_and_needs_no_fallback():
    fake = FakeAnthropic(reject_temperature=True)
    job = make_job("anthropic", fake, temperature=None)
    assert job.run(REQ)[0].error is None
    assert fake.submits == 1 and "temperature_dropped" not in job.identity()
