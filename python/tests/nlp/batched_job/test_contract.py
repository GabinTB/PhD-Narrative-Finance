"""nlp.batched_job: the lifecycle every vendor child must honour, run on each vendor's fake."""
from __future__ import annotations

import json
import logging

import pytest

from nlp.backends.base import IncompatibleModelError
from nlp.batched_job import SUPPORTED, BatchHandle, BatchStatus, ChatRequest, batch_job
from tests.cb_speeches.fakes import FakeChat, chat_backend
from tests.nlp.batched_job.fakes import FAKES, KEY, answer, make_job

VENDORS = sorted(FAKES)


def reqs(n: int = 4) -> list[ChatRequest]:
    return [ChatRequest(f"s{i}", "Extract.", f"Author: 'Speaker {i}'") for i in range(n)]


@pytest.fixture(params=VENDORS)
def vendor(request):
    return request.param


def test_every_supported_vendor_has_a_fake():
    assert sorted(SUPPORTED) == VENDORS


def test_run_returns_validated_replies_in_input_order(vendor):
    fake = FAKES[vendor]()
    job = make_job(vendor, fake)
    requests = reqs()
    out = job.run(list(reversed(requests)))
    assert [r.data for r in out] == [answer(r.user) for r in reversed(requests)]
    assert all(r.error is None and r.attempts == 1 for r in out)
    assert {r.response_model for r in out} == {fake.model}
    assert job.served == fake.model
    assert fake.submits == 1


def test_invalid_and_errored_replies_go_to_the_fallback_only(vendor):
    fake = FAKES[vendor](invalid={"s1"}, errored={"s2"})
    live = FakeChat()
    out = make_job(vendor, fake).run(reqs(), fallback=chat_backend(live))
    assert live.calls == 2
    assert [r.data for r in out] == [answer(r.user) for r in reqs()]
    assert [r.attempts for r in out] == [1, 2, 2, 1]


def test_without_fallback_failures_stay_explicit_error_rows(vendor):
    fake = FAKES[vendor](invalid={"s1"}, errored={"s2"})
    out = make_job(vendor, fake).run(reqs())
    assert out[0].error is None and out[3].error is None
    assert out[1].data is None and out[1].error.startswith("unparseable reply")
    assert out[2].data is None and "boom" in out[2].error


def test_expired_batch_keeps_its_replies_and_redoes_the_rest_live(vendor):
    fake = FAKES[vendor](final="expired", missing={"s0", "s3"})
    live = FakeChat()
    out = make_job(vendor, fake).run(reqs(), fallback=chat_backend(live))
    assert live.calls == 2
    assert [r.data for r in out] == [answer(r.user) for r in reqs()]
    with pytest.raises(RuntimeError, match="expired"):
        make_job(vendor, FAKES[vendor](final="expired", missing={"s0"})).run(reqs())


def test_failed_batch_goes_entirely_live(vendor):
    if vendor == "anthropic":
        pytest.skip("Anthropic batches fail per request, never as a whole")
    live = FakeChat()
    out = make_job(vendor, FAKES[vendor](final="failed")).run(reqs(), fallback=chat_backend(live))
    assert live.calls == 4 and all(r.error is None for r in out)
    with pytest.raises(RuntimeError, match="failed"):
        make_job(vendor, FAKES[vendor](final="failed")).run(reqs())


def test_served_model_drift_raises(vendor):
    with pytest.raises(IncompatibleModelError, match="-b"):
        make_job(vendor, FAKES[vendor](drift=True)).run(reqs())


def test_handle_round_trips_and_reattaches(vendor):
    fake = FAKES[vendor]()
    handle = make_job(vendor, fake).submit(reqs())
    saved = json.loads(json.dumps(handle.to_dict()))
    again = make_job(vendor, fake)                       # e.g. after a crash
    back = BatchHandle.from_dict(saved)
    assert again.wait(back).status is BatchStatus.SUCCEEDED
    got = again.results(back)
    assert sorted(got) == ["s0", "s1", "s2", "s3"] and fake.submits == 1
    other = BatchHandle.from_dict({**saved, "model": "another"})
    with pytest.raises(ValueError, match="another"):
        again.progress(other)


def test_wait_times_out(vendor):
    job = make_job(vendor, FAKES[vendor](polls=10**6))
    handle = job.submit(reqs())
    with pytest.raises(TimeoutError):
        job.wait(handle, timeout=0.0)


def test_request_checks(vendor, monkeypatch):
    job = make_job(vendor, FAKES[vendor]())
    with pytest.raises(ValueError, match="empty"):
        job.submit([])
    with pytest.raises(ValueError, match="unique"):
        job.submit([ChatRequest("a", "s", "u"), ChatRequest("a", "s", "v")])
    monkeypatch.setattr(type(job), "max_requests", 2)
    with pytest.raises(ValueError, match="limit"):
        job.submit(reqs(3))


def test_key_never_in_repr_info_identity_or_logs(vendor, caplog):
    caplog.set_level(logging.DEBUG)
    job = make_job(vendor, FAKES[vendor]())
    job.run(reqs())
    text = " ".join([repr(job), str(job.info()), str(job.identity()), caplog.text])
    assert KEY not in text
    assert job.info()["mode"] == "batch" and "/" not in job.info()["endpoint"]


def test_identity_records_sampling_and_extra(vendor):
    job = make_job(vendor, FAKES[vendor](), temperature=None, extra={"x": 1})
    ident = job.identity()
    assert ident["provider"] == vendor and ident["temperature"] is None
    assert ident["extra"] == {"x": 1}


def test_configuration_comes_from_the_environment(vendor, monkeypatch):
    from nlp.llm import PROVIDERS

    p = PROVIDERS[vendor]
    monkeypatch.delenv(p.base_url_env, raising=False)
    with pytest.raises(RuntimeError, match=p.base_url_env):
        batch_job(vendor, "m", schema={})
    monkeypatch.setenv(p.base_url_env, "http://x.test/v1")
    monkeypatch.delenv(p.api_key_env, raising=False)
    with pytest.raises(RuntimeError, match=p.api_key_env):
        batch_job(vendor, "m", schema={})
    with pytest.raises(ValueError, match="model"):
        batch_job(vendor, "", schema={}, api_key="k")


def test_providers_without_batch_api():
    assert batch_job("perplexity", "sonar", schema={}) is None
    assert batch_job("ollama", "llama", schema={}) is None
