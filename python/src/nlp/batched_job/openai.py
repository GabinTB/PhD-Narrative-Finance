"""OpenAI Batch API through the OpenAI SDK: JSONL of ``/v1/chat/completions`` requests
uploaded as a ``batch`` file, one batch, output and error files read back."""
from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from typing import Any

from nlp.batched_job.base import (
    BatchChatJob,
    BatchHandle,
    BatchProgress,
    ChatRequest,
    RawReply,
    deep_merge,
    openai_status,
    reply_from_batch_line,
)

ENDPOINT = "/v1/chat/completions"


class OpenAIBatchJob(BatchChatJob):
    provider_name = "openai"
    max_requests = 50_000
    sends_seed = True

    def _connect(self, api_key: str, http_client: Any) -> None:
        from openai import OpenAI

        self._client = OpenAI(base_url=self.base_url, api_key=api_key, max_retries=6,
                              timeout=300.0, http_client=http_client)

    def body(self, r: ChatRequest) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": r.system},
                         {"role": "user", "content": r.user}],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": self.name, "schema": self.schema, "strict": True}}}
        if self.temperature is not None:
            body["temperature"] = self.temperature
        if self.seed is not None:
            body["seed"] = self.seed
        if self.max_tokens is not None:
            body["max_completion_tokens"] = self.max_tokens
        return deep_merge(body, self.extra)

    def _submit(self, requests: Sequence[ChatRequest]) -> BatchHandle:
        jsonl = "".join(json.dumps({"custom_id": r.custom_id, "method": "POST",
                                    "url": ENDPOINT, "body": self.body(r)}) + "\n"
                        for r in requests)
        f = self._client.files.create(file=("batch.jsonl", jsonl.encode()), purpose="batch")
        batch = self._client.batches.create(input_file_id=f.id, endpoint=ENDPOINT,
                                            completion_window="24h")
        return self._handle(batch.id, len(requests), f.id)

    def _progress(self, handle: BatchHandle) -> BatchProgress:
        b = self._client.batches.retrieve(handle.batch_id)
        c = b.request_counts
        return BatchProgress(openai_status(b.status), c.total if c else handle.n_requests,
                             c.completed if c else 0, c.failed if c else 0)

    def _replies(self, handle: BatchHandle) -> Iterator[RawReply]:
        b = self._client.batches.retrieve(handle.batch_id)
        for file_id in (b.output_file_id, b.error_file_id):
            if not file_id:
                continue
            for line in self._client.files.content(file_id).text.splitlines():
                if line.strip():
                    yield reply_from_batch_line(json.loads(line))

    def cancel(self, handle: BatchHandle) -> None:
        self._client.batches.cancel(handle.batch_id)

    def close(self) -> None:
        self._client.close()
