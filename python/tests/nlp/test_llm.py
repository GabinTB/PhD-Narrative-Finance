"""nlp.llm: provider resolution, structured replies, identity, secrecy."""
from __future__ import annotations

import logging

import pytest

from nlp.backends.base import IncompatibleModelError
from nlp.llm import PROVIDERS, ChatBackend, parse_json, validate
from tests.cb_speeches.fakes import CHAT_URL, FakeChat, chat_backend

SCHEMA = {"type": "object", "properties": {
    "name": {"type": "string"}, "mood": {"type": "string", "enum": ["up", "down"]},
    "note": {"type": ["string", "null"]}},
    "required": ["name", "mood"], "additionalProperties": False}
NER = {"type": "object", "properties": {
    k: {"type": "string"} for k in ("author", "organization", "country_code", "sentiment")},
    "required": ["author", "organization", "country_code", "sentiment"],
    "additionalProperties": False}


def test_env_names_follow_the_dotenv_pattern():
    assert {p: (v.base_url_env, v.api_key_env) for p, v in PROVIDERS.items()} == {
        "ollama": ("VERTEX_OLLAMA_URL", "VERTEX_OLLAMA_SECRET"),
        "openai": ("OPENAI_ENDPOINT", "OPENAI_API_KEY"),
        "anthropic": ("ANTHROPIC_ENDPOINT", "ANTHROPIC_API_KEY"),
        "perplexity": ("PERPLEXITY_ENDPOINT", "PERPLEXITY_API_KEY"),
        "google": ("GEMINI_ENDPOINT", "GEMINI_API_KEY"),
        "mistral": ("OPENMISTRAL_ENDPOINT", "MISTRAL_API_KEY")}


def test_endpoint_and_key_come_from_the_environment(monkeypatch):
    fake = FakeChat()
    monkeypatch.setenv("MISTRAL_API_KEY", "sk-env-secret")
    monkeypatch.setenv("OPENMISTRAL_ENDPOINT", CHAT_URL)
    backend = ChatBackend("mistral", fake.model, http_client=fake.client(), max_retries=0)
    assert backend.base_url == CHAT_URL
    backend.complete_json("s", "Author: 'A'", NER)
    assert fake.calls == 1


def test_missing_configuration_names_the_variable(monkeypatch):
    monkeypatch.delenv("PERPLEXITY_ENDPOINT", raising=False)
    with pytest.raises(RuntimeError, match="PERPLEXITY_ENDPOINT"):
        ChatBackend("perplexity", "sonar")
    monkeypatch.setenv("PERPLEXITY_ENDPOINT", CHAT_URL)
    monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="PERPLEXITY_API_KEY"):
        ChatBackend("perplexity", "sonar")
    with pytest.raises(ValueError, match="model"):
        ChatBackend("openai", "", base_url=CHAT_URL, api_key="k")
    with pytest.raises(ValueError, match="provider"):
        ChatBackend("nope", "m", base_url=CHAT_URL, api_key="k")


def test_key_never_in_repr_info_identity_or_logs(caplog):
    caplog.set_level(logging.DEBUG)
    backend = chat_backend(FakeChat())
    backend.complete_json("s", "Author: 'A'", NER)
    text = " ".join([repr(backend), str(backend.info()), str(backend.identity()),
                     caplog.text])
    assert "sk-test-secret" not in text


def test_schema_sent_where_honoured_else_in_the_prompt():
    fake = FakeChat()
    chat_backend(fake, "openai").complete_json("sys", "Author: 'A'", NER)
    chat_backend(fake, "anthropic").complete_json("sys", "Author: 'A'", NER)
    openai_body, anthropic_body = fake.bodies
    assert openai_body["response_format"]["json_schema"]["schema"] == NER
    assert openai_body["seed"] == 0 and openai_body["temperature"] == 0
    assert "response_format" not in anthropic_body and "seed" not in anthropic_body
    assert '"additionalProperties"' in anthropic_body["messages"][0]["content"]


def test_invalid_replies_are_retried_then_an_explicit_error():
    fake = FakeChat(bad_replies=1)
    ok = chat_backend(fake).complete_json("s", "Author: 'A'", NER)
    assert ok.error is None and ok.attempts == 2 and ok.data["author"] == "A"
    fake = FakeChat(bad_replies=5)
    bad = chat_backend(fake, max_attempts=3).complete_json("s", "Author: 'A'", NER)
    assert bad.data is None and "unparseable" in bad.error and bad.attempts == 3


def test_hosted_snapshot_change_is_refused():
    fake = FakeChat()
    backend = chat_backend(fake)
    backend.complete_json("s", "Author: 'A'", NER)
    assert backend.served == fake.model
    fake.model = "fake-llm-2026-06-01"
    with pytest.raises(IncompatibleModelError, match="answered with model"):
        backend.complete_json("s", "Author: 'A'", NER)


def test_ollama_identity_is_the_weights_digest():
    fake = FakeChat(model="qwen3:14b")
    backend = chat_backend(fake, "ollama")
    assert backend.identity()["digest"] == fake.digest
    backend.check_unchanged()
    fake.digest = "sha256:" + "b" * 64
    with pytest.raises(IncompatibleModelError, match="changed mid-run"):
        backend.check_unchanged()
    with pytest.raises(IncompatibleModelError, match="recorded"):
        chat_backend(fake, "ollama").expect_served("sha256:" + "a" * 64)


def test_complete_many_keeps_order():
    backend = chat_backend(FakeChat())
    out = backend.complete_many([("s", f"Author: 'A{i}'") for i in range(7)], NER)
    assert [r.data["author"] for r in out] == [f"A{i}" for i in range(7)]


def test_validate_and_parse():
    assert validate({"name": "x", "mood": "up", "note": None}, SCHEMA) == []
    assert validate({"name": 1, "mood": "sideways", "x": 0}, SCHEMA) == [
        "unexpected 'x'", "'name' is int, expected string", "'mood'='sideways' not in "
        "['up', 'down']"]
    assert validate({"mood": "up"}, SCHEMA) == ["missing 'name'"]
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    with pytest.raises(ValueError):
        parse_json("no")


def test_rejected_temperature_falls_back_to_the_provider_default(caplog):
    import json as _json

    import httpx

    fake = FakeChat()

    def handler(request: httpx.Request) -> httpx.Response:
        if "temperature" in _json.loads(request.content):
            return httpx.Response(400, json={"error": {
                "type": "invalid_request_error",
                "message": "`temperature` is deprecated for this model."}})
        return fake.handler(request)

    backend = ChatBackend("anthropic", fake.model, base_url=CHAT_URL, api_key="k",
                          http_client=httpx.Client(transport=httpx.MockTransport(handler)),
                          max_retries=0, temperature=0.0)
    assert backend.complete_json("s", "Author: 'A'", NER).data["author"] == "A"
    assert backend.complete_json("s", "Author: 'B'", NER).data["author"] == "B"
    assert backend.temperature is None and fake.calls == 2    # dropped once, then never sent
    assert backend.identity()["temperature_dropped"] is True
    assert "rejects temperature=0.0" in caplog.text


def test_other_bad_requests_still_raise():
    import httpx
    from openai import BadRequestError

    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(
        400, json={"error": {"message": "unknown model"}})))
    backend = ChatBackend("openai", "m", base_url=CHAT_URL, api_key="k", http_client=client,
                          max_retries=0)
    with pytest.raises(BadRequestError):
        backend.complete_json("s", "u", NER)
    assert backend.temperature == 0.0


@pytest.mark.parametrize("message, rejected", [
    ("`temperature` is deprecated for this model.", True),
    ("Unsupported value: 'temperature' does not support 0 with this model. Only the default "
     "(1) value is supported.", True),
    ("Unsupported value: 'top_p' does not support 0.5 with this model.", False),
    ("unknown model", False)])
def test_temperature_rejection_messages(message, rejected):
    from nlp.llm import TEMPERATURE_REJECTED

    assert bool(TEMPERATURE_REJECTED.search(message)) is rejected
