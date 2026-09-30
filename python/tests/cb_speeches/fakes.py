"""In-process fakes: a BIS download server and an OpenAI-compatible chat endpoint."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import zipfile
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import httpx

BULK_URL = "https://bis.test/pages/download/speeches.zip"
CHAT_URL = "http://llm.test/v1"


def speech(day: date, n: int, *, text: str | None = None, fmt: str = "htm") -> dict[str, str]:
    """One BIS CSV row; its review code encodes the day and n."""
    code = f"r{day:%y%m%d}{'abcdefghij'[n]}"
    return {"url": f"https://www.bis.org/review/{code}.{fmt}",
            "title": f"Speaker {n} on policy ({day})",
            "description": f"Speech by Mr Speaker {n}, Governor of the Bank of Testland, "
                           f"on {day}.",
            "date": f"{day} 00:00:00", "text": text or f"text of {code}",
            "author": f"Speaker {n}"}


def every_third_day(start: date, end: date, per_day: int = 2) -> list[dict[str, str]]:
    out, d = [], start
    while d <= end:
        out += [speech(d, i) for i in range(per_day)]
        d += timedelta(days=3)
    return out


def make_zip(rows: list[dict[str, str]], member: str = "speeches.csv") -> bytes:
    """A deterministic BIS-shaped zip (fixed member timestamp)."""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=["url", "title", "description", "date", "text",
                                        "author"])
    w.writeheader()
    w.writerows(rows)
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        info = zipfile.ZipInfo(member, date_time=(2020, 1, 1, 0, 0, 0))
        zf.writestr(info, buf.getvalue())
    return out.getvalue()


@dataclass
class FakeBIS:
    """Serves every year of ``rows`` as one bulk speeches.zip (ETag = content hash).

    Honours ``Range`` / ``If-Range`` like the real server (which does not advertise
    ranges). ``cut`` cuts that many GETs short, half way through what they would send,
    like the real server's dropped connections."""

    rows: dict[int, list[dict[str, str]]]
    cut: int = 0
    requests: list[tuple[str, str]] = field(default_factory=list)

    def zip(self) -> bytes:
        return make_zip([r for year in sorted(self.rows) for r in self.rows[year]])

    def etag(self) -> str:
        return f'"{hashlib.md5(self.zip()).hexdigest()}"'

    def handler(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.requests.append((request.method, url))
        if url != BULK_URL:
            return httpx.Response(404, text="not found")
        content = self.zip()
        headers = {"etag": self.etag(), "last-modified": "Sun, 23 Aug 2026 14:05:42 GMT",
                   "content-type": "application/zip", "content-length": str(len(content))}
        if request.method == "HEAD":
            return httpx.Response(200, headers=headers)
        status, body = 200, content
        rng = request.headers.get("range")
        if rng and request.headers.get("if-range", headers["etag"]) == headers["etag"]:
            start = int(rng.split("=")[1].rstrip("-"))
            status, body = 206, content[start:]
            headers["content-length"] = str(len(body))
        if self.cut > 0:
            self.cut -= 1
            return httpx.Response(status, headers=headers,
                                  stream=_CutStream(body[:len(body) // 2]))
        return httpx.Response(status, content=body, headers=headers)

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(lambda request: self.handler(request))   # patchable

    def downloads(self) -> int:
        return sum(1 for method, _ in self.requests if method == "GET")


class _CutStream(httpx.SyncByteStream):
    """Sends ``data`` then drops the connection."""

    def __init__(self, data: bytes) -> None:
        self.data = data

    def __iter__(self):
        yield self.data
        raise httpx.RemoteProtocolError("peer closed connection without sending complete "
                                        "message body")


@dataclass
class FakeChat:
    """``/chat/completions`` answering the NER schema from the prompt; ``model`` is the
    reported snapshot; ``bad_replies`` invalid replies are served first."""

    model: str = "fake-llm-2026-01-01"
    bad_replies: int = 0
    digest: str = "sha256:" + "a" * 64
    calls: int = 0
    bodies: list[dict[str, Any]] = field(default_factory=list)

    def answer(self, prompt: str) -> dict[str, Any]:
        author = re.search(r"Author: '(.*)'", prompt).group(1)
        return {"author": author, "organization": "Bank of Testland", "country_code": "TL",
                "sentiment": "neutral"}

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/api/tags"):
            return httpx.Response(200, json={"models": [
                {"name": self.model, "model": self.model, "digest": self.digest}]})
        body = json.loads(request.content)
        self.bodies.append(body)
        self.calls += 1
        if self.bad_replies > 0:
            self.bad_replies -= 1
            content = "not json at all"
        else:
            content = json.dumps(self.answer(body["messages"][1]["content"]))
        return httpx.Response(200, json={
            "id": f"c{self.calls}", "object": "chat.completion", "created": 0,
            "model": self.model, "system_fingerprint": "fp_test",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}})

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self.handler))


def chat_backend(fake: FakeChat, provider: str = "openai", **kwargs: Any):
    from nlp.llm import ChatBackend

    kwargs.setdefault("workers", 2)
    return ChatBackend(provider, fake.model, base_url=CHAT_URL, api_key="sk-test-secret",
                       http_client=fake.client(), max_retries=0, **kwargs)
