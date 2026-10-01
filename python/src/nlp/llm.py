"""OpenAI-compatible chat backend: one generative model behind one provider endpoint.

Every provider is reached through the OpenAI SDK (``/chat/completions``); the
endpoint and the key come from the environment (``.env``), by provider name:

    provider     base URL env           key env                capabilities
    ollama       VERTEX_OLLAMA_URL      VERTEX_OLLAMA_SECRET   json_schema, seed, digest
    openai       OPENAI_ENDPOINT        OPENAI_API_KEY         json_schema, seed
    anthropic    ANTHROPIC_ENDPOINT     ANTHROPIC_API_KEY      prompt-only JSON
    perplexity   PERPLEXITY_ENDPOINT    PERPLEXITY_API_KEY     json_schema
    google       GEMINI_ENDPOINT        GEMINI_API_KEY         json_schema
    mistral      OPENMISTRAL_ENDPOINT   MISTRAL_API_KEY        json_schema

The model name is always the caller's (no default model).

* Structured output: the JSON schema is sent as ``response_format`` where the
  provider honours it; elsewhere it is appended to the system prompt. The reply
  is ALWAYS parsed and validated locally (``validate``), retried up to
  ``max_attempts`` times, then returned as an explicit error -- never a silent
  partial record.
* Temperature: a model that rejects a non-default temperature (Claude 5 models)
  is retried without it, with a warning; ``identity()`` then records
  ``temperature=None`` and ``temperature_dropped=True``.
* API failures (auth, unknown model, bad request, connection after the SDK's
  own 429/5xx retries) raise: a job stops with the reason instead of writing
  thousands of error rows.
* Identity: ``identity()`` is what the outputs depend on (provider, model,
  sampling, prompt capability; for Ollama the served weights digest).
  ``check_unchanged()`` raises when the served model differs from the recorded
  one: the Ollama digest is re-read from ``/api/tags``; hosted providers are
  checked on the dated snapshot every reply reports (``expect_served`` sets
  the snapshot an artifact already recorded).
* Keys live only in the SDK client: never logged, never in ``info()`` / repr.
"""
from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import urlsplit

from nlp.backends.base import IncompatibleModelError
from nlp.backends.http import map_ordered

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Provider:
    """An OpenAI-compatible endpoint family and how far it honours the API."""

    name: str
    base_url_env: str
    api_key_env: str
    json_schema: bool          # honours response_format={"type": "json_schema", ...}
    seed: bool                 # accepts the ``seed`` sampling parameter


PROVIDERS: dict[str, Provider] = {p.name: p for p in (
    Provider("ollama", "VERTEX_OLLAMA_URL", "VERTEX_OLLAMA_SECRET", json_schema=True, seed=True),
    Provider("openai", "OPENAI_ENDPOINT", "OPENAI_API_KEY", json_schema=True, seed=True),
    Provider("anthropic", "ANTHROPIC_ENDPOINT", "ANTHROPIC_API_KEY", json_schema=False,
             seed=False),
    Provider("perplexity", "PERPLEXITY_ENDPOINT", "PERPLEXITY_API_KEY", json_schema=True,
             seed=False),
    Provider("google", "GEMINI_ENDPOINT", "GEMINI_API_KEY", json_schema=True, seed=False),
    Provider("mistral", "OPENMISTRAL_ENDPOINT", "MISTRAL_API_KEY", json_schema=True,
             seed=False),
)}


def provider(name: str) -> Provider:
    if name not in PROVIDERS:
        raise ValueError(f"provider must be one of {sorted(PROVIDERS)}, got {name!r}")
    return PROVIDERS[name]


# ---------------------------------------------------------------------------
# Schema validation (the subset structured outputs use: a flat object)
# ---------------------------------------------------------------------------

_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,), "integer": (int,), "number": (int, float), "boolean": (bool,),
}


def validate(value: Any, schema: dict[str, Any]) -> list[str]:
    """Problems of ``value`` against a flat object schema (properties of scalar
    types, ``enum``, ``required``, ``additionalProperties: false``, nullable
    ``["string", "null"]`` types). Nested schemas are refused, not half-checked."""
    if schema.get("type") != "object":
        raise ValueError("validate() supports object schemas only")
    if not isinstance(value, dict):
        return [f"expected an object, got {type(value).__name__}"]
    props: dict[str, Any] = schema.get("properties", {})
    problems = [f"missing {k!r}" for k in schema.get("required", []) if k not in value]
    if schema.get("additionalProperties") is False:
        problems += [f"unexpected {k!r}" for k in value if k not in props]
    for key, spec in props.items():
        if key not in value:
            continue
        types = spec.get("type", [])
        types = [types] if isinstance(types, str) else list(types)
        if any(t in ("object", "array") for t in types):
            raise ValueError(f"validate() does not support nested schemas ({key!r})")
        v = value[key]
        if v is None:
            if "null" not in types:
                problems.append(f"{key!r} is null")
            continue
        allowed = tuple(c for t in types if t != "null" for c in _JSON_TYPES[t])
        if types and (not isinstance(v, allowed) or (isinstance(v, bool)
                                                      and "boolean" not in types)):
            problems.append(f"{key!r} is {type(v).__name__}, expected {'/'.join(types)}")
        elif "enum" in spec and v not in spec["enum"]:
            problems.append(f"{key!r}={v!r} not in {spec['enum']}")
    return problems


# Models that refuse a non-default temperature answer with a 400, matched on the error (not on
# a model list) so the request is retried without it:
#   Claude 5 / Opus 4.7+:    "`temperature` is deprecated for this model."
#   OpenAI reasoning models: "Unsupported value: 'temperature' does not support 0 with this
#                             model. Only the default (1) value is supported."
TEMPERATURE_REJECTED = re.compile(
    r"temperature[`'\"]? (is deprecated|does not support)", re.I)


def warn_temperature_dropped(provider_name: str, model: str, temperature: float) -> None:
    log.warning("%s %s rejects temperature=%s: falling back to "
                "the provider default, recorded as temperature=None", provider_name, model,
                temperature)


_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.S)


def parse_json(text: str | None) -> Any:
    """The JSON object in a reply (a ```json fenced block is unwrapped)."""
    if text is None:
        raise ValueError("empty reply")
    m = _FENCE.match(text)
    return json.loads(m.group(1) if m else text)


def check_reply(text: str | None, schema: dict[str, Any]) -> tuple[dict[str, Any] | None,
                                                                  str | None]:
    """(data, None) for a reply that parses and validates, else (None, error)."""
    try:
        data = parse_json(text)
    except ValueError as exc:
        return None, f"unparseable reply: {exc}"
    problems = validate(data, schema)
    if problems:
        return None, "schema: " + "; ".join(problems)
    return data, None


def schema_system(system: str, schema: dict[str, Any]) -> str:
    """The system prompt carrying the schema, for providers without structured output."""
    return (f"{system}\n\nReply with one JSON object only, no prose, matching this "
            f"JSON schema:\n{json.dumps(schema, sort_keys=True)}")


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------

@dataclass
class ChatResult:
    """One structured completion: ``data`` is None exactly when ``error`` is set."""

    data: dict[str, Any] | None
    response_model: str | None = None
    system_fingerprint: str | None = None
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    attempts: int = 0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ChatBackend:
    """A chat model behind an OpenAI-compatible endpoint.

    Args:
        provider:     one of ``PROVIDERS``.
        model:        the served model name (required, no default).
        base_url / api_key: override the provider's environment variables.
        temperature:  sampling temperature (default 0; None = not sent). A model that rejects
                      it (Claude 5: "deprecated") falls back to the provider default with a
                      warning, recorded as ``temperature=None, temperature_dropped=True``.
        seed:         sampling seed, sent where the provider accepts one.
        max_attempts: tries per request when the reply fails parsing/validation.
        max_retries:  the SDK's own retries on 429 / 5xx / connection errors.
        timeout:      per-request timeout (s).
        workers:      concurrent requests in ``complete_many``.
        http_client:  injected ``httpx.Client`` (tests); also used for Ollama's
                      native ``/api/tags``.
    """

    name = "llm"

    def __init__(self, provider_name: str, model: str, *, base_url: str | None = None,
                 api_key: str | None = None, temperature: float | None = 0.0,
                 seed: int | None = 0,
                 max_attempts: int = 3, max_retries: int = 6, timeout: float = 120.0,
                 workers: int = 8, http_client: Any = None) -> None:
        from openai import OpenAI

        if not model:
            raise ValueError("a model name is required")
        self.provider = provider(provider_name)
        self.model = model
        base_url = base_url or os.environ.get(self.provider.base_url_env)
        if not base_url:
            raise RuntimeError(f"{self.provider.base_url_env} must be set (in .env or "
                               f"environment) to use the {self.provider.name} provider")
        api_key = api_key if api_key is not None else os.environ.get(self.provider.api_key_env)
        if not api_key and self.provider.name != "ollama":
            raise RuntimeError(f"{self.provider.api_key_env} must be set to use the "
                               f"{self.provider.name} provider")
        self.base_url = base_url.rstrip("/")
        self.temperature = None if temperature is None else float(temperature)
        self.temperature_dropped = False          # set when the model rejected it
        self.seed = seed if self.provider.seed else None
        self.max_attempts, self.workers = max_attempts, workers
        self._client = OpenAI(base_url=self.base_url, api_key=api_key or "ollama",
                              max_retries=max_retries, timeout=timeout, http_client=http_client)
        self._native = http_client
        self._api_key = api_key                   # only for Ollama's native endpoint
        self._served: str | None = None           # recorded snapshot / digest
        if self.provider.name == "ollama":
            self._served = self._ollama_digest()
        log.info("chat backend %s: %s (json_schema=%s, seed=%s)", self.provider.name, model,
                 self.provider.json_schema, self.seed)

    def __repr__(self) -> str:
        return f"ChatBackend({self.provider.name!r}, {self.model!r})"

    # -- served model -------------------------------------------------------

    def _ollama_digest(self) -> str:
        """Digest of the pulled model, from Ollama's native ``/api/tags``."""
        import httpx

        root = re.sub(r"/v1$", "", self.base_url)
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        client = self._native or httpx.Client(timeout=30.0)
        try:
            resp = client.get(f"{root}/api/tags", headers=headers)
        finally:
            if self._native is None:
                client.close()
        if resp.status_code >= 400:
            raise RuntimeError(f"Ollama /api/tags: HTTP {resp.status_code}")
        for m in resp.json().get("models", []):
            if self.model in (m.get("name"), m.get("model")):
                return str(m["digest"])
        raise IncompatibleModelError(f"Ollama does not serve {self.model!r}")

    def expect_served(self, served: str) -> None:
        """Pin the snapshot / digest an artifact recorded (resume, update)."""
        if self.provider.name == "ollama" and self._served != served:
            raise IncompatibleModelError(
                f"Ollama serves {self.model} digest {self._served}, the artifact recorded "
                f"{served}")
        self._served = served

    @property
    def served(self) -> str | None:
        """The served snapshot (hosted: set by the first reply) or weights digest."""
        return self._served

    def _observe(self, response_model: str | None) -> None:
        if self.provider.name == "ollama" or not response_model:
            return
        if self._served is None:
            self._served = response_model
        elif response_model != self._served:
            raise IncompatibleModelError(f"{self.provider.name} answered with model "
                                         f"{response_model}, recorded {self._served}")

    def check_unchanged(self) -> None:
        if self.provider.name == "ollama":
            now = self._ollama_digest()
            if now != self._served:
                raise IncompatibleModelError(
                    f"Ollama model {self.model} changed mid-run: {self._served} -> {now}")

    # -- completion ---------------------------------------------------------

    def _request(self, system: str, user: str, schema: dict[str, Any],
                 name: str) -> dict[str, Any]:
        if not self.provider.json_schema:
            system = schema_system(system, schema)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
        }
        if self.provider.json_schema:
            kwargs["response_format"] = {"type": "json_schema", "json_schema": {
                "name": name, "schema": schema, "strict": True}}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.seed is not None:
            kwargs["seed"] = self.seed
        return kwargs

    def _create(self, kwargs: dict[str, Any]) -> Any:
        """One API call; a model that rejects ``temperature`` is retried without it, once
        for the backend (warning, ``temperature`` becomes None in info / identity)."""
        from openai import BadRequestError

        try:
            return self._client.chat.completions.create(**kwargs)
        except BadRequestError as exc:
            if "temperature" not in kwargs or not TEMPERATURE_REJECTED.search(str(exc)):
                raise
        if self.temperature is not None:
            warn_temperature_dropped(self.provider.name, self.model, self.temperature)
            self.temperature, self.temperature_dropped = None, True
        kwargs.pop("temperature")
        return self._client.chat.completions.create(**kwargs)

    def complete_json(self, system: str, user: str, schema: dict[str, Any], *,
                      name: str = "extraction") -> ChatResult:
        """One structured completion, validated against ``schema``."""
        kwargs = self._request(system, user, schema, name)
        result = ChatResult(None)
        for attempt in range(1, self.max_attempts + 1):
            resp = self._create(kwargs)
            self._observe(resp.model)
            choice = resp.choices[0]
            usage = resp.usage
            result = ChatResult(
                None, response_model=resp.model,
                system_fingerprint=getattr(resp, "system_fingerprint", None),
                finish_reason=choice.finish_reason,
                prompt_tokens=usage.prompt_tokens if usage else None,
                completion_tokens=usage.completion_tokens if usage else None,
                attempts=attempt)
            result.data, result.error = check_reply(choice.message.content, schema)
            if result.error is None:
                return result
        return result

    def complete_many(self, requests: Sequence[tuple[str, str]], schema: dict[str, Any], *,
                      name: str = "extraction") -> list[ChatResult]:
        """``complete_json`` over (system, user) pairs, ``workers`` at a time, in order."""
        return map_ordered(lambda r: self.complete_json(r[0], r[1], schema, name=name),
                           list(requests), self.workers)

    # -- metadata -----------------------------------------------------------

    def info(self) -> dict[str, Any]:
        """Serving metadata (no key): provider, endpoint host, capabilities, snapshot."""
        return {"engine": self.provider.name, "endpoint": urlsplit(self.base_url).netloc,
                "model": self.model, "served": self._served,
                "capabilities": {"json_schema": self.provider.json_schema,
                                 "seed": self.provider.seed},
                "sampling": {"temperature": self.temperature, "seed": self.seed},
                "client": {"max_attempts": self.max_attempts, "workers": self.workers}}

    def identity(self) -> dict[str, Any]:
        """What the replies depend on (hosted snapshots are checked per reply)."""
        out = {"backend": self.name, "provider": self.provider.name, "model": self.model,
               "temperature": self.temperature, "seed": self.seed,
               "json_schema": self.provider.json_schema}
        if self.temperature_dropped:
            out["temperature_dropped"] = True
        if self.provider.name == "ollama":
            out["digest"] = self._served
        return out

    def close(self) -> None:
        self._client.close()


__all__ = ["PROVIDERS", "ChatBackend", "ChatResult", "Provider", "check_reply", "parse_json",
           "provider", "schema_system", "validate"]
