"""Gemini Batch API, native REST (``models/{model}:batchGenerateContent``) with inline
requests. The OpenAI-compatible layer can create batches but not upload or download their
files, so the native API is used end to end; ``GEMINI_ENDPOINT`` (the ``.../v1beta/openai``
compat URL) is mapped to its native ``.../v1beta`` root.

Inline batches are capped at 20 MB of requests (checked before sending). The schema is
enforced by ``generationConfig.responseJsonSchema``. Gemini 3.x Pro always thinks; tune it
through ``extra={"generationConfig": {"thinkingConfig": {...}}}``.
"""
from __future__ import annotations

import json
import re
from collections.abc import Iterator, Sequence
from typing import Any

from nlp.batched_job.base import (
    BatchHandle,
    BatchProgress,
    BatchStatus,
    ChatRequest,
    RawReply,
    RestBatchJob,
    deep_merge,
)

INLINE_LIMIT_BYTES = 20_000_000

_STATES = {"PENDING": BatchStatus.QUEUED, "UNSPECIFIED": BatchStatus.QUEUED,
           "RUNNING": BatchStatus.RUNNING, "SUCCEEDED": BatchStatus.SUCCEEDED,
           "FAILED": BatchStatus.FAILED, "CANCELLED": BatchStatus.CANCELLED,
           "EXPIRED": BatchStatus.EXPIRED}


def _state(name: str) -> BatchStatus:
    """``BATCH_STATE_*`` (REST) and ``JOB_STATE_*`` (SDK docs) spellings alike."""
    return _STATES[name.rsplit("_", 1)[-1]]


def _inlined(d: dict[str, Any]) -> list[dict[str, Any]]:
    """The inline responses of a finished batch, wherever this API version puts them."""
    for holder in (d.get("response"), (d.get("metadata") or {}).get("output"), d.get("output"),
                   d.get("dest")):
        if not isinstance(holder, dict):
            continue
        got = holder.get("inlinedResponses")
        if isinstance(got, dict):
            got = got.get("inlinedResponses")
        if isinstance(got, list):
            return got
    return []


class GoogleBatchJob(RestBatchJob):
    provider_name = "google"
    max_requests = 1_000_000            # the binding limit is INLINE_LIMIT_BYTES
    sends_seed = True

    def _auth(self, api_key: str) -> dict[str, str]:
        self.base_url = re.sub(r"/openai$", "", self.base_url)
        return {"x-goog-api-key": api_key}

    @property
    def _model_path(self) -> str:
        return self.model if self.model.startswith("models/") else f"models/{self.model}"

    def request(self, r: ChatRequest) -> dict[str, Any]:
        gen: dict[str, Any] = {"responseMimeType": "application/json",
                               "responseJsonSchema": self.schema}
        if self.temperature is not None:
            gen["temperature"] = self.temperature
        if self.seed is not None:
            gen["seed"] = self.seed
        if self.max_tokens is not None:
            gen["maxOutputTokens"] = self.max_tokens
        req = {"systemInstruction": {"parts": [{"text": r.system}]},
               "contents": [{"role": "user", "parts": [{"text": r.user}]}],
               "generationConfig": gen}
        return deep_merge(req, self.extra)

    def _submit(self, requests: Sequence[ChatRequest]) -> BatchHandle:
        body = {"batch": {"display_name": f"{self.name}-{len(requests)}", "input_config": {
            "requests": {"requests": [{"request": self.request(r),
                                       "metadata": {"key": r.custom_id}}
                                      for r in requests]}}}}
        size = len(json.dumps(body).encode())
        if size > INLINE_LIMIT_BYTES:
            raise ValueError(f"{size / 1e6:.1f} MB of inline requests exceed Gemini's "
                             f"{INLINE_LIMIT_BYTES / 1e6:.0f} MB: split them")
        op = self._call("POST", f"/{self._model_path}:batchGenerateContent", json=body)
        name = op.get("name") or (op.get("metadata") or {}).get("name")
        if not name:
            raise RuntimeError("google batchGenerateContent returned no batch name")
        return self._handle(name, len(requests))

    def _get(self, handle: BatchHandle) -> dict[str, Any]:
        return self._call("GET", f"/{handle.batch_id}")

    def _progress(self, handle: BatchHandle) -> BatchProgress:
        d = self._get(handle)
        meta = d.get("metadata") or d
        stats = meta.get("batchStats") or {}
        return BatchProgress(_state(meta.get("state", "STATE_UNSPECIFIED")),
                             int(stats.get("requestCount", handle.n_requests)),
                             int(stats.get("successfulRequestCount", 0)),
                             int(stats.get("failedRequestCount", 0)))

    def _replies(self, handle: BatchHandle) -> Iterator[RawReply]:
        for item in _inlined(self._get(handle)):
            cid = str((item.get("metadata") or {}).get("key"))
            if item.get("error"):
                yield RawReply(cid, error=f"batch error: {item['error'].get('message')}")
                continue
            resp = item.get("response") or {}
            cands = resp.get("candidates") or []
            if not cands:
                reason = (resp.get("promptFeedback") or {}).get("blockReason", "no candidates")
                yield RawReply(cid, model=resp.get("modelVersion"), error=f"reply: {reason}")
                continue
            parts = (cands[0].get("content") or {}).get("parts") or []
            text = "".join(p.get("text", "") for p in parts if not p.get("thought")) or None
            usage = resp.get("usageMetadata") or {}
            out_tokens = (usage.get("candidatesTokenCount") or 0) + (
                usage.get("thoughtsTokenCount") or 0)
            yield RawReply(cid, content=text, model=resp.get("modelVersion"),
                           finish_reason=cands[0].get("finishReason"),
                           prompt_tokens=usage.get("promptTokenCount"),
                           completion_tokens=out_tokens or None)

    def cancel(self, handle: BatchHandle) -> None:
        self._call("POST", f"/{handle.batch_id}:cancel")
