"""Mistral Batch API, native REST: JSONL of ``/v1/chat/completions`` bodies uploaded as a
``batch`` file, one ``/v1/batch/jobs`` job, output and error files read back (OpenAI-format
result lines). The schema is enforced by ``response_format`` json_schema."""
from __future__ import annotations

import json
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
    reply_from_batch_line,
)

ENDPOINT = "/v1/chat/completions"

_STATUS = {"QUEUED": BatchStatus.QUEUED, "RUNNING": BatchStatus.RUNNING,
           "CANCELLATION_REQUESTED": BatchStatus.RUNNING, "SUCCESS": BatchStatus.SUCCEEDED,
           "FAILED": BatchStatus.FAILED, "TIMEOUT_EXCEEDED": BatchStatus.EXPIRED,
           "CANCELLED": BatchStatus.CANCELLED}


class MistralBatchJob(RestBatchJob):
    provider_name = "mistral"
    max_requests = 1_000_000
    sends_seed = True

    def _auth(self, api_key: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {api_key}"}

    def body(self, r: ChatRequest) -> dict[str, Any]:
        body: dict[str, Any] = {
            "messages": [{"role": "system", "content": r.system},
                         {"role": "user", "content": r.user}],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": self.name, "schema": self.schema, "strict": True}}}
        if self.temperature is not None:
            body["temperature"] = self.temperature
        if self.seed is not None:
            body["random_seed"] = self.seed
        if self.max_tokens is not None:
            body["max_tokens"] = self.max_tokens
        return deep_merge(body, self.extra)

    def _submit(self, requests: Sequence[ChatRequest]) -> BatchHandle:
        jsonl = "".join(json.dumps({"custom_id": r.custom_id, "body": self.body(r)}) + "\n"
                        for r in requests)
        f = self._call("POST", "/files", data={"purpose": "batch"},
                       files={"file": ("batch.jsonl", jsonl.encode(), "application/jsonl")})
        job = self._call("POST", "/batch/jobs", json={
            "input_files": [f["id"]], "model": self.model, "endpoint": ENDPOINT,
            "metadata": {"name": self.name}})
        return self._handle(job["id"], len(requests), f["id"])

    def _job(self, handle: BatchHandle) -> dict[str, Any]:
        return self._call("GET", f"/batch/jobs/{handle.batch_id}")

    def _progress(self, handle: BatchHandle) -> BatchProgress:
        j = self._job(handle)
        return BatchProgress(_STATUS[j["status"]], int(j.get("total_requests") or
                                                      handle.n_requests),
                             int(j.get("succeeded_requests") or 0),
                             int(j.get("failed_requests") or 0))

    def _replies(self, handle: BatchHandle) -> Iterator[RawReply]:
        j = self._job(handle)
        for file_id in (j.get("output_file"), j.get("error_file")):
            if not file_id:
                continue
            for line in self._call("GET", f"/files/{file_id}/content", raw=True).splitlines():
                if line.strip():
                    yield reply_from_batch_line(json.loads(line))

    def cancel(self, handle: BatchHandle) -> None:
        self._call("POST", f"/batch/jobs/{handle.batch_id}/cancel")
