"""In-memory fakes of the four vendor batch APIs, on ``httpx.MockTransport``.

Every fake answers the NER-style schema from the user prompt (``Author: 'X'``), walks
queued -> running -> ``final`` over ``polls`` status reads, and has switches for an invalid
reply (``invalid``), a vendor error line (``errored``), replies missing from an expired batch
(``missing``) and a served-model change after the first reply (``drift``).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, ClassVar

import httpx

KEY = "sk-batch-secret"
SCHEMA = {"type": "object", "properties": {
    k: {"type": "string"} for k in ("author", "organization", "country_code", "sentiment")},
    "required": ["author", "organization", "country_code", "sentiment"],
    "additionalProperties": False}


def answer(user: str) -> dict[str, str]:
    author = re.search(r"Author: '(.*)'", user).group(1)
    return {"author": author, "organization": "Bank of Testland", "country_code": "TL",
            "sentiment": "neutral"}


def jsonl_lines(content: bytes) -> list[dict[str, Any]]:
    """The JSONL request lines inside a multipart upload."""
    return [json.loads(line) for line in content.decode().splitlines()
            if line.startswith('{"custom_id"')]


@dataclass
class FakeBatchAPI:
    model: str = "fake-model-2026-09-01"
    polls: int = 1                       # status reads that still say "running"
    final: str = "succeeded"             # succeeded | expired | failed
    invalid: set[str] = field(default_factory=set)
    errored: set[str] = field(default_factory=set)
    missing: set[str] = field(default_factory=set)
    drift: bool = False
    fail_first: int = 0                  # leading HTTP 500s (REST retry tests)
    reject_temperature: bool = False     # Claude 5: temperature is a per-request error
    payloads: dict[str, dict[str, Any]] = field(default_factory=dict)
    users: dict[str, str] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)
    status_reads: int = 0
    submits: int = 0

    base_url: ClassVar[str] = ""

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self._dispatch))

    def _dispatch(self, request: httpx.Request) -> httpx.Response:
        if self.fail_first > 0:
            self.fail_first -= 1
            return httpx.Response(500, json={"error": {"message": "try again"}})
        return self.handle(request)

    def handle(self, request: httpx.Request) -> httpx.Response:
        raise NotImplementedError

    # -- shared state machine ------------------------------------------------

    def record(self, cid: str, payload: dict[str, Any], user: str) -> None:
        self.payloads[cid], self.users[cid] = payload, user
        self.order.append(cid)

    def done(self) -> bool:
        self.status_reads += 1
        return self.status_reads > self.polls

    def served_model(self, i: int) -> str:
        return f"{self.model}-b" if self.drift and i > 0 else self.model

    def outcomes(self) -> list[tuple[int, str, str]]:
        """(index, custom_id, kind) with kind ok | invalid | errored, missing ones dropped
        when the batch expired."""
        out = []
        for i, cid in enumerate(self.order):
            if self.final == "expired" and cid in self.missing:
                continue
            kind = "errored" if cid in self.errored else "invalid" if cid in self.invalid else "ok"
            out.append((i, cid, kind))
        return out

    def text(self, cid: str, kind: str) -> str:
        return "not json at all" if kind == "invalid" else json.dumps(answer(self.users[cid]))


class FakeOpenAI(FakeBatchAPI):
    base_url = "http://openai.test/v1"

    def handle(self, request: httpx.Request) -> httpx.Response:
        path, method = request.url.path, request.method
        if method == "POST" and path == "/v1/files":
            for line in jsonl_lines(request.content):
                body = line["body"]
                self.record(line["custom_id"], line, body["messages"][1]["content"])
            return httpx.Response(200, json={"id": "file-in", "object": "file", "bytes": 1,
                                             "created_at": 0, "filename": "batch.jsonl",
                                             "purpose": "batch", "status": "processed"})
        if method == "POST" and path == "/v1/batches":
            self.submits += 1
            return httpx.Response(200, json=self._batch("validating"))
        if method == "GET" and path == "/v1/batches/batch_1":
            status = ({"succeeded": "completed"}.get(self.final, self.final) if self.done()
                      else "in_progress")
            return httpx.Response(200, json=self._batch(status))
        if method == "GET" and path == "/v1/files/file-out/content":
            return httpx.Response(200, text=self._lines(ok=True))
        if method == "GET" and path == "/v1/files/file-err/content":
            return httpx.Response(200, text=self._lines(ok=False))
        return httpx.Response(404, json={"error": {"message": f"no route {method} {path}"}})

    def _batch(self, status: str) -> dict[str, Any]:
        ended = status in ("completed", "expired")
        return {"id": "batch_1", "object": "batch", "endpoint": "/v1/chat/completions",
                "input_file_id": "file-in", "completion_window": "24h", "created_at": 0,
                "status": status,
                "output_file_id": "file-out" if ended else None,
                "error_file_id": "file-err" if ended and self.errored else None,
                "request_counts": {"total": len(self.order), "completed": 0, "failed": 0}}

    def _lines(self, *, ok: bool) -> str:
        out = []
        for i, cid, kind in self.outcomes():
            if (kind == "errored") == ok:
                continue
            if kind == "errored":
                out.append({"id": f"r{i}", "custom_id": cid, "response": None,
                            "error": {"code": "server_error", "message": "boom"}})
                continue
            out.append({"id": f"r{i}", "custom_id": cid, "error": None, "response": {
                "status_code": 200, "body": {
                    "id": f"c{i}", "object": "chat.completion", "model": self.served_model(i),
                    "choices": [{"index": 0, "finish_reason": "stop", "message": {
                        "role": "assistant", "content": self.text(cid, kind)}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5,
                              "total_tokens": 15}}}})
        return "".join(json.dumps(x) + "\n" for x in out)


class FakeMistral(FakeOpenAI):
    base_url = "http://mistral.test/v1"

    def handle(self, request: httpx.Request) -> httpx.Response:
        path, method = request.url.path, request.method
        if method == "POST" and path == "/v1/files":
            for line in jsonl_lines(request.content):
                self.record(line["custom_id"], line, line["body"]["messages"][1]["content"])
            return httpx.Response(200, json={"id": "f-in", "purpose": "batch"})
        if method == "POST" and path == "/v1/batch/jobs":
            self.submits += 1
            self.job = json.loads(request.content)
            return httpx.Response(200, json={"id": "job-1", "status": "QUEUED"})
        if method == "GET" and path == "/v1/batch/jobs/job-1":
            status = ({"succeeded": "SUCCESS", "expired": "TIMEOUT_EXCEEDED",
                       "failed": "FAILED"}[self.final] if self.done() else "RUNNING")
            ended = status in ("SUCCESS", "TIMEOUT_EXCEEDED")
            return httpx.Response(200, json={
                "id": "job-1", "status": status, "total_requests": len(self.order),
                "succeeded_requests": 0, "failed_requests": 0,
                "output_file": "out" if ended else None,
                "error_file": "err" if ended and self.errored else None})
        if method == "GET" and path == "/v1/files/out/content":
            return httpx.Response(200, text=self._lines(ok=True),
                                  headers={"content-type": "application/json"})
        if method == "GET" and path == "/v1/files/err/content":
            return httpx.Response(200, text=self._lines(ok=False))
        return httpx.Response(404, json={"message": f"no route {method} {path}"})


class FakeAnthropic(FakeBatchAPI):
    base_url = "http://anthropic.test/v1"

    def handle(self, request: httpx.Request) -> httpx.Response:
        path, method = request.url.path, request.method
        if method == "POST" and path == "/v1/messages/batches":
            self.submits += 1
            self.order, self.status_reads = [], 0        # a new batch replaces the last
            for r in json.loads(request.content)["requests"]:
                self.record(r["custom_id"], r["params"], r["params"]["messages"][0]["content"])
            return httpx.Response(200, json=self._batch(False))
        if method == "GET" and path == "/v1/messages/batches/msgbatch_1":
            return httpx.Response(200, json=self._batch(self.done()))
        if method == "GET" and path == "/v1/messages/batches/msgbatch_1/results":
            return httpx.Response(200, text=self._results(),
                                  headers={"content-type": "application/x-jsonl"})
        return httpx.Response(404, json={"type": "error", "error": {
            "type": "not_found_error", "message": f"no route {method} {path}"}})

    def _batch(self, ended: bool) -> dict[str, Any]:
        n = len(self.order)
        expired = len(self.missing) if ended and self.final == "expired" else 0
        errored = (len(self.errored) + sum("temperature" in self.payloads[c] for c in self.order)
                   * self.reject_temperature) if ended else 0
        counts = {"processing": 0 if ended else n, "succeeded": n - expired - errored
                  if ended else 0, "errored": errored, "canceled": 0, "expired": expired}
        return {"id": "msgbatch_1", "type": "message_batch",
                "processing_status": "ended" if ended else "in_progress",
                "request_counts": counts, "created_at": "2026-09-30T00:00:00Z",
                "expires_at": "2026-10-01T00:00:00Z", "ended_at": None,
                "cancel_initiated_at": None, "archived_at": None,
                "results_url": ("http://anthropic.test/v1/messages/batches/msgbatch_1/results"
                                if ended else None)}

    def _results(self) -> str:
        out = []
        for i, cid in enumerate(self.order):
            if self.final == "expired" and cid in self.missing:
                out.append({"custom_id": cid, "result": {"type": "expired"}})
                continue
            if cid in self.errored:
                out.append({"custom_id": cid, "result": {"type": "errored", "error": {
                    "type": "error", "error": {"type": "api_error", "message": "boom"}}}})
                continue
            if self.reject_temperature and "temperature" in self.payloads[cid]:
                out.append({"custom_id": cid, "result": {"type": "errored", "error": {
                    "type": "error", "error": {"type": "invalid_request_error",
                                               "message": "`temperature` is deprecated for "
                                                          "this model."}}}})
                continue
            kind = "invalid" if cid in self.invalid else "ok"
            out.append({"custom_id": cid, "result": {"type": "succeeded", "message": {
                "id": f"msg_{i}", "type": "message", "role": "assistant",
                "model": self.served_model(i),
                "content": [{"type": "text", "text": self.text(cid, kind)}],
                "stop_reason": "end_turn", "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 5}}}})
        return "".join(json.dumps(x) + "\n" for x in out)


class FakeGoogle(FakeBatchAPI):
    base_url = "http://google.test/v1beta/openai"

    def handle(self, request: httpx.Request) -> httpx.Response:
        path, method = request.url.path, request.method
        if method == "POST" and path.endswith(":batchGenerateContent"):
            self.submits += 1
            self.model_path = path
            body = json.loads(request.content)
            for r in body["batch"]["input_config"]["requests"]["requests"]:
                req = r["request"]
                self.record(r["metadata"]["key"], req, req["contents"][0]["parts"][0]["text"])
            return httpx.Response(200, json={"name": "batches/b1", "metadata": {
                "state": "BATCH_STATE_PENDING"}})
        if method == "GET" and path == "/v1beta/batches/b1":
            if not self.done():
                return httpx.Response(200, json={"name": "batches/b1", "metadata": {
                    "state": "BATCH_STATE_RUNNING"}})
            state = {"succeeded": "BATCH_STATE_SUCCEEDED", "expired": "BATCH_STATE_EXPIRED",
                     "failed": "BATCH_STATE_FAILED"}[self.final]
            return httpx.Response(200, json={
                "name": "batches/b1", "done": True,
                "metadata": {"state": state, "batchStats": {
                    "requestCount": str(len(self.order)),
                    "successfulRequestCount": str(len(self.order) - len(self.errored))}},
                "response": {"inlinedResponses": {"inlinedResponses": self._inlined()}}})
        return httpx.Response(404, json={"error": {"message": f"no route {method} {path}"}})

    def _inlined(self) -> list[dict[str, Any]]:
        out = []
        for i, cid, kind in self.outcomes():
            if kind == "errored":
                out.append({"metadata": {"key": cid}, "error": {"code": 13, "message": "boom"}})
                continue
            out.append({"metadata": {"key": cid}, "response": {
                "modelVersion": self.served_model(i),
                "candidates": [{"finishReason": "STOP", "content": {"role": "model", "parts": [
                    {"text": "thinking...", "thought": True},
                    {"text": self.text(cid, kind)}]}}],
                "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5,
                                  "thoughtsTokenCount": 7}}})
        return out


FAKES: dict[str, type[FakeBatchAPI]] = {"openai": FakeOpenAI, "anthropic": FakeAnthropic,
                                        "google": FakeGoogle, "mistral": FakeMistral}


def make_job(provider: str, fake: FakeBatchAPI, **kwargs: Any):
    from nlp.batched_job import job_class

    kwargs.setdefault("schema", SCHEMA)
    kwargs.setdefault("name", "ner")
    return job_class(provider)(fake.model, base_url=fake.base_url, api_key=KEY,
                               http_client=fake.client(), poll_interval=0.0,
                               sleep=lambda s: None, **kwargs)
