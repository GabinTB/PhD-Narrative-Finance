"""Anthropic Message Batches through the native SDK (not the OpenAI-compatible layer, which
has no batches and ignores ``response_format``). Requests go inline; the schema is enforced
by native structured outputs (``output_config.format``).

Claude 5 models: a non-default ``temperature`` is a per-request error ("`temperature` is
deprecated for this model"). Pass ``temperature=None``; otherwise ``run`` resubmits the
rejected requests without it, with a warning, and records ``temperature_dropped`` (see
``BatchChatJob.run``). Opus 5.5 always thinks (lower ``output_config.effort`` through
``extra`` to cut the thinking tokens). ``max_tokens`` is required by the API and covers the
thinking: it defaults to 4096 here.
"""
from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from typing import Any

from nlp.batched_job.base import (
    BatchChatJob,
    BatchHandle,
    BatchProgress,
    BatchStatus,
    ChatRequest,
    RawReply,
    deep_merge,
)

DEFAULT_MAX_TOKENS = 4096


class AnthropicBatchJob(BatchChatJob):
    provider_name = "anthropic"
    max_requests = 100_000

    def _connect(self, api_key: str, http_client: Any) -> None:
        from anthropic import Anthropic

        # ANTHROPIC_ENDPOINT points at the OpenAI-compatible ``.../v1``; the SDK adds it.
        root = re.sub(r"/v1$", "", self.base_url)
        self._client = Anthropic(base_url=root, api_key=api_key, max_retries=6,
                                 timeout=300.0, http_client=http_client)

    def params(self, r: ChatRequest) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self.model, "max_tokens": self.max_tokens or DEFAULT_MAX_TOKENS,
            "system": r.system, "messages": [{"role": "user", "content": r.user}],
            "output_config": {"format": {"type": "json_schema", "schema": self.schema}}}
        if self.temperature is not None:
            params["temperature"] = self.temperature
        return deep_merge(params, self.extra)

    def _sampling(self) -> dict[str, Any]:
        return {**super()._sampling(), "max_tokens": self.max_tokens or DEFAULT_MAX_TOKENS}

    def _submit(self, requests: Sequence[ChatRequest]) -> BatchHandle:
        batch = self._client.messages.batches.create(requests=[
            {"custom_id": r.custom_id, "params": self.params(r)} for r in requests])
        return self._handle(batch.id, len(requests))

    def _progress(self, handle: BatchHandle) -> BatchProgress:
        b = self._client.messages.batches.retrieve(handle.batch_id)
        c = b.request_counts
        failed = c.errored + c.canceled + c.expired
        if b.processing_status != "ended":
            status = BatchStatus.RUNNING
        elif c.succeeded + c.errored == handle.n_requests:
            status = BatchStatus.SUCCEEDED          # errored lines are per-request results
        elif c.expired:
            status = BatchStatus.EXPIRED
        else:
            status = BatchStatus.CANCELLED
        return BatchProgress(status, handle.n_requests, c.succeeded, failed)

    def _replies(self, handle: BatchHandle) -> Iterator[RawReply]:
        for line in self._client.messages.batches.results(handle.batch_id):
            res = line.result
            if res.type != "succeeded":
                err = getattr(res, "error", None)
                detail = getattr(getattr(err, "error", None), "message", None) or res.type
                yield RawReply(line.custom_id, error=f"batch {res.type}: {detail}")
                continue
            msg = res.message
            text = "".join(b.text for b in msg.content if b.type == "text") or None
            yield RawReply(line.custom_id, content=text, model=msg.model,
                           finish_reason=msg.stop_reason,
                           prompt_tokens=msg.usage.input_tokens,
                           completion_tokens=msg.usage.output_tokens)

    def cancel(self, handle: BatchHandle) -> None:
        self._client.messages.batches.cancel(handle.batch_id)

    def close(self) -> None:
        self._client.close()
