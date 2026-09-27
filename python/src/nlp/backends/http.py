"""Shared HTTP plumbing for the remote backends (TEI, embedx).

* Bearer auth. The key lives only in the request headers of the private
  httpx client: it is never logged, never part of ``info()`` / ``repr``, and
  error messages carry the status and the server's error body only.
* 429 (server queue full) is retried with capped exponential backoff; every
  other non-2xx raises immediately. No fallback to another engine: a remote
  failure stops the job with a clear message.
* ``map_ordered`` runs batch requests on a bounded thread pool and returns
  results in submission order.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeVar

log = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


class RemoteError(RuntimeError):
    """A remote inference server answered with an error (message excludes secrets)."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"HTTP {status}: {message}")
        self.status = status


class HTTPClient:
    """Thin JSON client over httpx with Bearer auth and 429 backoff.

    Args:
        base_url:    server root, e.g. ``http://10.10.10.2:8081`` (TEI) or
                     ``http://10.10.10.2:8477/v1`` (embedx).
        api_key:     Bearer token, or None for an open server.
        timeout:     per-request timeout in seconds.
        max_retries: attempts on 429 before giving up.
        max_connections: connection-pool size (>= the caller's concurrency).
        transport:   injected httpx transport (tests).
    """

    def __init__(self, base_url: str, api_key: str | None = None, *, timeout: float = 120.0,
                 max_retries: int = 12, max_connections: int = 64,
                 transport: Any = None) -> None:
        import httpx

        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.Client(
            base_url=self.base_url, headers=headers, timeout=timeout, transport=transport,
            limits=httpx.Limits(max_connections=max_connections,
                                max_keepalive_connections=max_connections),
        )
        self.n_retries_429 = 0

    def __repr__(self) -> str:
        return f"HTTPClient({self.base_url!r})"

    def request(self, method: str, path: str, payload: Any = None) -> Any:
        for attempt in range(self.max_retries + 1):
            resp = self._client.request(method, path, json=payload)
            if resp.status_code == 429 and attempt < self.max_retries:
                self.n_retries_429 += 1
                time.sleep(min(0.05 * 2 ** attempt, 5.0))
                continue
            if resp.status_code >= 400:
                try:
                    body = resp.json()
                    detail = body.get("error", body) if isinstance(body, dict) else body
                except ValueError:
                    detail = resp.text[:500]
                raise RemoteError(resp.status_code, f"{method} {path}: {detail}")
            return resp.json()
        raise RemoteError(429, f"{method} {path}: still overloaded after "
                               f"{self.max_retries} retries")

    def get(self, path: str) -> Any:
        return self.request("GET", path)

    def post(self, path: str, payload: Any) -> Any:
        return self.request("POST", path, payload)

    def close(self) -> None:
        self._client.close()


def map_ordered(fn: Callable[[T], R], items: Sequence[T], workers: int) -> list[R]:
    """``[fn(x) for x in items]`` on a bounded thread pool, results in input order."""
    if workers <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, items))
