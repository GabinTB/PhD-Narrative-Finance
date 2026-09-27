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

BASE_URL = "https://bis.test/pages/download/"
INDEX_URL = "https://bis.test/cbspeeches/download.htm"
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


def make_zip(year: int, rows: list[dict[str, str]]) -> bytes:
    """A deterministic BIS-shaped zip (fixed member timestamp)."""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=["url", "title", "description", "date", "text",
                                        "author"])
    w.writeheader()
    w.writerows(rows)
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        info = zipfile.ZipInfo(f"speeches_{year}.csv", date_time=(2020, 1, 1, 0, 0, 0))
        zf.writestr(info, buf.getvalue())
    return out.getvalue()


@dataclass
class FakeBIS:
    """``rows[year]`` served as speeches-YYYY.zip; ETag = content hash."""

    rows: dict[int, list[dict[str, str]]]
    requests: list[tuple[str, str]] = field(default_factory=list)

    def zip(self, year: int) -> bytes:
        return make_zip(year, self.rows[year])

    def handler(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.requests.append((request.method, url))
        if url == INDEX_URL:
            links = "".join(f'<a href="/pages/download/speeches-{y}.zip">{y}</a>'
                            for y in sorted(self.rows))
            return httpx.Response(200, text=f"<html>{links}</html>")
        m = re.fullmatch(re.escape(BASE_URL) + r"speeches-(\d{4})\.zip", url)
        if not m or int(m.group(1)) not in self.rows:
            return httpx.Response(404, text="not found")
        content = self.zip(int(m.group(1)))
        headers = {"etag": f'"{hashlib.md5(content).hexdigest()}"',
                   "last-modified": "Sun, 23 Aug 2026 14:05:42 GMT",
                   "content-type": "application/zip"}
        if request.method == "HEAD":
            return httpx.Response(200, headers=headers)
        return httpx.Response(200, content=content, headers=headers)

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)

    def downloads(self) -> int:
        return sum(1 for method, url in self.requests if method == "GET" and url != INDEX_URL)


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
