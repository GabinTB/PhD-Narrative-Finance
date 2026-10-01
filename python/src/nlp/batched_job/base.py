"""The provider-neutral batch contract: submit many structured chat completions, wait, collect.

A batch is a list of ``ChatRequest`` (custom_id, system, user) under one JSON schema and one
set of sampling parameters. A vendor child only translates: the unified request into its
payload (``_submit``), its status into ``BatchProgress`` (``_progress``) and its result lines
into ``RawReply`` (``_replies``). Everything else is shared here: request checks, polling,
reply validation (``nlp.llm.check_reply``), the served-model check, the live fallback, and the
identity an artifact records.

``extra`` is merged (recursively) into every request body, so vendor-only parameters (thinking,
effort, reasoning) pass through untouched; it is part of ``identity()`` since it changes the
replies. ``temperature=None`` / ``seed=None`` / ``max_tokens=None`` are not sent at all
(provider default). Claude 5 models reject any non-default temperature: ``run`` resubmits
those requests without it and records the fallback (see ``run``).
"""
from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar

from nlp.backends.base import IncompatibleModelError
from nlp.llm import (
    TEMPERATURE_REJECTED,
    ChatResult,
    check_reply,
    provider,
    warn_temperature_dropped,
)

if TYPE_CHECKING:
    from nlp.llm import ChatBackend

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChatRequest:
    """One (system, user) completion; ``custom_id`` matches the reply back."""

    custom_id: str
    system: str
    user: str


class BatchStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        return self not in (BatchStatus.QUEUED, BatchStatus.RUNNING)


@dataclass(frozen=True)
class BatchProgress:
    status: BatchStatus
    n_total: int
    n_succeeded: int = 0
    n_failed: int = 0


@dataclass(frozen=True)
class BatchHandle:
    """What re-attaches to a submitted batch (persist ``to_dict()`` as JSON)."""

    provider: str
    model: str
    batch_id: str
    n_requests: int
    submitted_at: str
    input_ref: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BatchHandle:
        return cls(**d)


@dataclass(frozen=True)
class RawReply:
    """One vendor result line, before validation: ``error`` set when no reply came back."""

    custom_id: str
    content: str | None = None
    model: str | None = None
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    error: str | None = None


def deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    """``base`` updated by ``extra``, nested dicts merged rather than replaced."""
    out = dict(base)
    for k, v in extra.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(
            out.get(k), dict) else v
    return out


def reply_from_chat_body(custom_id: str, body: dict[str, Any]) -> RawReply:
    """A ``/chat/completions`` response body (OpenAI and Mistral result lines)."""
    choices = body.get("choices") or []
    if not choices:
        return RawReply(custom_id, model=body.get("model"), error="reply has no choices")
    usage = body.get("usage") or {}
    return RawReply(custom_id, content=(choices[0].get("message") or {}).get("content"),
                    model=body.get("model"), finish_reason=choices[0].get("finish_reason"),
                    prompt_tokens=usage.get("prompt_tokens"),
                    completion_tokens=usage.get("completion_tokens"))


def reply_from_batch_line(line: dict[str, Any]) -> RawReply:
    """One OpenAI-format batch result line: ``{custom_id, response: {status_code, body},
    error}`` (OpenAI and Mistral output / error files)."""
    cid = str(line.get("custom_id"))
    err = line.get("error")
    resp = line.get("response") or {}
    if err:
        msg = err.get("message") if isinstance(err, dict) else str(err)
        return RawReply(cid, error=f"batch error: {msg}")
    status = resp.get("status_code", 200)
    if status >= 400:
        body = resp.get("body") or {}
        msg = (body.get("error") or {}).get("message") if isinstance(body.get("error"),
                                                                       dict) else body
        return RawReply(cid, error=f"batch error: HTTP {status}: {msg}")
    return reply_from_chat_body(cid, resp.get("body") or {})


def openai_status(status: str) -> BatchStatus:
    """OpenAI batch statuses (Mistral's are mapped by its child)."""
    return {"validating": BatchStatus.QUEUED, "in_progress": BatchStatus.RUNNING,
            "finalizing": BatchStatus.RUNNING, "cancelling": BatchStatus.RUNNING,
            "completed": BatchStatus.SUCCEEDED, "failed": BatchStatus.FAILED,
            "expired": BatchStatus.EXPIRED, "cancelled": BatchStatus.CANCELLED}[status]


def utc_stamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class BatchChatJob(ABC):
    """Structured chat completions through a vendor's asynchronous batch API.

    Args:
        model:         the served model name (required, no default).
        schema / name: the JSON schema every reply must match, and its name.
        temperature / seed / max_tokens: sampling; None = not sent (provider default).
        extra:         vendor-only request parameters, merged into each request body.
        base_url / api_key: override the provider's environment variables.
        http_client:   injected ``httpx.Client`` (tests).
        poll_interval: seconds between status polls in ``wait``.
        sleep:         injected sleep (tests).
    """

    provider_name: ClassVar[str]
    max_requests: ClassVar[int]
    sends_seed: ClassVar[bool] = False

    def __init__(self, model: str, *, schema: dict[str, Any], name: str = "extraction",
                 temperature: float | None = 0.0, seed: int | None = 0,
                 max_tokens: int | None = None, extra: dict[str, Any] | None = None,
                 base_url: str | None = None, api_key: str | None = None,
                 http_client: Any = None, poll_interval: float = 60.0,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        if not model:
            raise ValueError("a model name is required")
        self.provider = provider(self.provider_name)
        self.model = model
        self.schema, self.name = schema, name
        self.temperature = None if temperature is None else float(temperature)
        self.temperature_dropped = False          # set when the model rejected it
        self.seed = seed if self.sends_seed else None
        self.max_tokens = max_tokens
        self.extra = dict(extra or {})
        self.poll_interval, self._sleep = poll_interval, sleep
        base_url = base_url or os.environ.get(self.provider.base_url_env)
        if not base_url:
            raise RuntimeError(f"{self.provider.base_url_env} must be set (in .env or "
                               f"environment) to use {self.provider.name} batches")
        api_key = api_key if api_key is not None else os.environ.get(self.provider.api_key_env)
        if not api_key:
            raise RuntimeError(f"{self.provider.api_key_env} must be set to use "
                               f"{self.provider.name} batches")
        self.base_url = base_url.rstrip("/")
        self._served: str | None = None
        self._connect(api_key, http_client)
        log.info("batch job %s: %s", self.provider.name, model)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.model!r})"

    # -- vendor hooks ---------------------------------------------------------

    @abstractmethod
    def _connect(self, api_key: str, http_client: Any) -> None:
        """Build the vendor client; the key must live only there."""

    @abstractmethod
    def _submit(self, requests: Sequence[ChatRequest]) -> BatchHandle: ...

    @abstractmethod
    def _progress(self, handle: BatchHandle) -> BatchProgress: ...

    @abstractmethod
    def _replies(self, handle: BatchHandle) -> Iterator[RawReply]: ...

    @abstractmethod
    def cancel(self, handle: BatchHandle) -> None: ...

    def close(self) -> None:
        """Release the vendor client."""

    # -- lifecycle ------------------------------------------------------------

    def _handle(self, batch_id: str, n: int, input_ref: str | None = None) -> BatchHandle:
        return BatchHandle(self.provider.name, self.model, batch_id, n, utc_stamp(), input_ref)

    def _check_handle(self, handle: BatchHandle) -> None:
        if (handle.provider, handle.model) != (self.provider.name, self.model):
            raise ValueError(f"handle is a {handle.provider}/{handle.model} batch, this job "
                             f"is {self.provider.name}/{self.model}")

    def submit(self, requests: Sequence[ChatRequest]) -> BatchHandle:
        """Send the batch; returns at once with the handle to wait on."""
        if not requests:
            raise ValueError("an empty batch")
        ids = [r.custom_id for r in requests]
        if len(set(ids)) != len(ids):
            raise ValueError("custom_id values must be unique within a batch")
        if len(requests) > self.max_requests:
            raise ValueError(f"{len(requests)} requests exceed the {self.provider.name} batch "
                             f"limit of {self.max_requests}: split them")
        handle = self._submit(requests)
        log.info("%s batch %s submitted: %d request(s)", self.provider.name, handle.batch_id,
                 len(requests))
        return handle

    def progress(self, handle: BatchHandle) -> BatchProgress:
        self._check_handle(handle)
        return self._progress(handle)

    def wait(self, handle: BatchHandle, *, timeout: float | None = None) -> BatchProgress:
        """Poll until the batch is terminal; ``TimeoutError`` after ``timeout`` seconds."""
        start = time.monotonic()
        while True:
            prog = self.progress(handle)
            if prog.status.terminal:
                log.info("%s batch %s %s: %d/%d succeeded", self.provider.name,
                         handle.batch_id, prog.status.value, prog.n_succeeded, prog.n_total)
                return prog
            if timeout is not None and time.monotonic() - start >= timeout:
                raise TimeoutError(f"{self.provider.name} batch {handle.batch_id} still "
                                   f"{prog.status.value} after {timeout:.0f}s")
            self._sleep(self.poll_interval)

    def _observe(self, model: str | None) -> None:
        if not model:
            return
        if self._served is None:
            self._served = model
        elif model != self._served:
            raise IncompatibleModelError(f"{self.provider.name} batch answered with model "
                                         f"{model}, recorded {self._served}")

    def expect_served(self, served: str) -> None:
        """Pin the snapshot an artifact already recorded."""
        self._served = served

    @property
    def served(self) -> str | None:
        return self._served

    def results(self, handle: BatchHandle) -> dict[str, ChatResult]:
        """Every reply of a finished batch, validated; a failed line is an error result."""
        self._check_handle(handle)
        out: dict[str, ChatResult] = {}
        for raw in self._replies(handle):
            self._observe(raw.model)
            res = ChatResult(None, response_model=raw.model, finish_reason=raw.finish_reason,
                             prompt_tokens=raw.prompt_tokens,
                             completion_tokens=raw.completion_tokens, attempts=1,
                             error=raw.error)
            if raw.error is None:
                res.data, res.error = check_reply(raw.content, self.schema)
            out[raw.custom_id] = res
        return out

    def run(self, requests: Sequence[ChatRequest], *, fallback: ChatBackend | None = None,
            timeout: float | None = None) -> list[ChatResult]:
        """Submit, wait and collect, in input order. Missing, errored or invalid replies are
        redone live by ``fallback`` when given. A batch that did not succeed raises without a
        fallback; with one, the replies an expired / cancelled batch still returned are
        kept and the rest go live.

        Requests rejected because the model refuses ``temperature`` (Claude 5 models: a
        per-request error, unbilled, only known once the batch ended) are resubmitted once
        as a new batch without it: warning, ``temperature=None`` and ``temperature_dropped``
        in ``identity()``."""
        handle = self.submit(requests)
        prog = self.wait(handle, timeout=timeout)
        if prog.status is not BatchStatus.SUCCEEDED:
            if fallback is None:
                raise RuntimeError(f"{self.provider.name} batch {handle.batch_id} ended "
                                   f"{prog.status.value}")
            log.warning("%s batch %s ended %s: missing replies go live", self.provider.name,
                        handle.batch_id, prog.status.value)
        got = self.results(handle) if prog.status is not BatchStatus.FAILED else {}
        rejected = [r for r in requests if (res := got.get(r.custom_id)) is not None
                    and res.error and TEMPERATURE_REJECTED.search(res.error)]
        if rejected and self.temperature is not None:
            warn_temperature_dropped(self.provider.name, self.model, self.temperature)
            self.temperature, self.temperature_dropped = None, True
            got.update(zip([r.custom_id for r in rejected],
                           self.run(rejected, fallback=fallback, timeout=timeout)))
        out: list[ChatResult] = []
        n_live = 0
        for r in requests:
            res = got.get(r.custom_id) or ChatResult(None, attempts=0,
                                                     error="no reply in the batch output")
            if res.error is not None and fallback is not None:
                live = fallback.complete_json(r.system, r.user, self.schema, name=self.name)
                live.attempts += res.attempts
                res, n_live = live, n_live + 1
            out.append(res)
        if n_live:
            log.info("%s batch %s: %d request(s) redone live", self.provider.name,
                     handle.batch_id, n_live)
        return out

    # -- metadata -------------------------------------------------------------

    def info(self) -> dict[str, Any]:
        """Serving metadata (no key, no endpoint path)."""
        from urllib.parse import urlsplit

        return {"engine": self.provider.name, "mode": "batch",
                "endpoint": urlsplit(self.base_url).netloc, "model": self.model,
                "served": self._served, "sampling": self._sampling(), "extra": self.extra}

    def _sampling(self) -> dict[str, Any]:
        return {"temperature": self.temperature, "seed": self.seed,
                "max_tokens": self.max_tokens}

    def identity(self) -> dict[str, Any]:
        """What the replies depend on."""
        out = {"backend": "llm-batch", "provider": self.provider.name, "model": self.model,
               **self._sampling(), "extra": self.extra, "json_schema": True}
        if self.temperature_dropped:
            out["temperature_dropped"] = True
        return out


class RestBatchJob(BatchChatJob):
    """A vendor reached with plain ``httpx`` (no SDK): auth headers, retries on 429 / 5xx /
    connection errors, errors that never carry the URL or the key."""

    max_http_retries: ClassVar[int] = 6

    @abstractmethod
    def _auth(self, api_key: str) -> dict[str, str]: ...

    def _connect(self, api_key: str, http_client: Any) -> None:
        import httpx

        self._headers = self._auth(api_key)
        self._owns_client = http_client is None
        self._http = http_client or httpx.Client(timeout=300.0)

    def _call(self, method: str, path: str, *, raw: bool = False, **kwargs: Any) -> Any:
        """``method`` on ``base_url + path``; the decoded JSON, or the text when ``raw`` or
        not JSON. Raises ``RuntimeError`` with the HTTP status and the vendor message."""
        import httpx

        headers = {**self._headers, **kwargs.pop("headers", {})}
        for attempt in range(self.max_http_retries + 1):
            try:
                resp = self._http.request(method, f"{self.base_url}{path}", headers=headers,
                                          **kwargs)
            except httpx.TransportError as exc:
                if attempt == self.max_http_retries:
                    raise RuntimeError(f"{self.provider.name}: {type(exc).__name__}") from None
                self._sleep(min(2.0 ** attempt, 60.0))
                continue
            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt < self.max_http_retries:
                    self._sleep(min(2.0 ** attempt, 60.0))
                    continue
            if resp.status_code >= 400:
                raise RuntimeError(f"{self.provider.name} {method} {path.split('?')[0]}: "
                                   f"HTTP {resp.status_code}: {resp.text[:300]}")
            if not raw and "json" in resp.headers.get("content-type", ""):
                return resp.json()
            return resp.text
        raise AssertionError("unreachable")

    def close(self) -> None:
        if self._owns_client:
            self._http.close()


__all__ = ["BatchChatJob", "BatchHandle", "BatchProgress", "BatchStatus", "ChatRequest",
           "RawReply", "RestBatchJob", "deep_merge", "openai_status", "reply_from_batch_line",
           "reply_from_chat_body"]
