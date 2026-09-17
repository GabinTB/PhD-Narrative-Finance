"""Low level HTTP transport: authentication, retries, error mapping and downloads."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any, Literal

import httpx
from pydantic import ValidationError

from .errors import ValidationErrorResponse
from .exceptions import (
    APIError,
    AuthenticationError,
    BadRequestError,
    ChecksumMismatchError,
    GoneError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    ServerError,
)

__all__ = ["ApiBase", "HttpTransport", "raise_for_response"]

ApiBase = Literal["streaming", "data", "annotations"]

_STATUS_TO_ERROR: dict[int, type[APIError]] = {
    400: BadRequestError,
    401: AuthenticationError,
    403: PermissionDeniedError,
    404: NotFoundError,
    410: GoneError,
    429: RateLimitError,
}
_RETRYABLE_STATUS = frozenset({429, 502, 503, 504})
_MIN_RATE_LIMIT_WAIT = 5.0
_MAX_BACKOFF = 60.0
_CHUNK_SIZE = 1 << 20


def _clean_params(params: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if params is None:
        return None
    cleaned: dict[str, Any] = {}
    for key, value in params.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple, set, frozenset)):
            if not value:
                continue
            cleaned[key] = [str(v) for v in value]
        else:
            cleaned[key] = str(value)
    return cleaned


def raise_for_response(response: httpx.Response) -> None:
    """Raise the ``APIError`` subclass matching ``response``'s status, if it is an error.

    The body must already be read (``response.read()`` for streamed responses).
    """
    if response.status_code < 400:
        return

    payload: Any
    try:
        payload = response.json()
    except ValueError:
        payload = response.text

    validation_errors: ValidationErrorResponse | None = None
    message = response.reason_phrase or "HTTP error"
    if isinstance(payload, dict):
        if isinstance(payload.get("errors"), list):
            try:
                validation_errors = ValidationErrorResponse.model_validate(payload)
            except ValidationError:
                validation_errors = None
            reasons = (
                [e.reason for e in validation_errors.errors if e.reason]
                if validation_errors
                else []
            )
            if reasons:
                message = "; ".join(reasons)
        if isinstance(payload.get("message"), str):
            message = payload["message"]
    elif isinstance(payload, str) and payload.strip():
        message = payload.strip()[:500]

    status = response.status_code
    error_cls = _STATUS_TO_ERROR.get(status, ServerError if status >= 500 else APIError)
    raise error_cls(
        status,
        message,
        payload=payload,
        validation_errors=validation_errors,
        url=str(response.request.url),
    )


class HttpTransport:
    """Shared ``httpx.Client`` bound to the three RavenPack base URLs.

    Args:
        api_key: RavenPack API key, sent as the ``api_key`` header.
        streaming_url: Base URL of the streaming API.
        data_url: Base URL of the data query / reference API.
        annotations_url: Base URL of the Annotations API.
        timeout: Default timeout in seconds.
        max_retries: Retries on HTTP 429/502/503/504 and transport errors.
        client: Optional preconfigured ``httpx.Client`` (proxies, certificates...).
    """

    def __init__(
        self,
        api_key: str,
        *,
        streaming_url: str,
        data_url: str,
        annotations_url: str,
        timeout: float,
        max_retries: int,
        client: httpx.Client | None = None,
    ) -> None:
        self._bases: dict[ApiBase, str] = {
            "streaming": streaming_url.rstrip("/"),
            "data": data_url.rstrip("/"),
            "annotations": annotations_url.rstrip("/"),
        }
        self._auth_headers = {"api_key": api_key}
        self._max_retries = max(0, max_retries)
        self._owns_client = client is None
        self._client = client if client is not None else httpx.Client(timeout=timeout)

    # -- URLs -------------------------------------------------------------

    def url(self, base: ApiBase, path: str) -> str:
        """Join ``path`` to the base URL of ``base``."""
        return f"{self._bases[base]}/{path.lstrip('/')}"

    def absolute_url(self, base: ApiBase, url: str) -> str:
        """Resolve a possibly host-relative URL (e.g. ``/1.0/history/...``) against ``base``'s host."""
        if url.startswith(("http://", "https://")):
            return url
        parsed = httpx.URL(self._bases[base])
        origin = f"{parsed.scheme}://{parsed.netloc.decode()}"
        return f"{origin}/{url.lstrip('/')}"

    # -- Requests ---------------------------------------------------------

    def _sleep_before_retry(self, attempt: int, response: httpx.Response | None) -> None:
        delay = min(2.0**attempt, _MAX_BACKOFF)
        if response is not None:
            retry_after = response.headers.get("Retry-After")
            if retry_after is not None:
                try:
                    delay = float(retry_after)
                except ValueError:
                    pass
            if response.status_code == 429:
                delay = max(delay, _MIN_RATE_LIMIT_WAIT)
        time.sleep(delay)

    def request(
        self,
        method: str,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        json: Any = None,
        headers: Mapping[str, str] | None = None,
        follow_redirects: bool = False,
        authenticated: bool = True,
        raise_on_error: bool = True,
    ) -> httpx.Response:
        """Send a request with retries and map error statuses to exceptions.

        Args:
            method: HTTP method.
            url: Absolute URL (use :meth:`url` to build it).
            params: Query parameters. ``None`` values are dropped, sequences are repeated.
            json: JSON body.
            headers: Extra headers.
            follow_redirects: Follow HTTP redirects.
            authenticated: Send the ``api_key`` header.
            raise_on_error: Raise an ``APIError`` on status >= 400.

        Returns:
            The fully read response.
        """
        merged_headers = {**(self._auth_headers if authenticated else {}), **(headers or {})}
        attempt = 0
        while True:
            try:
                response = self._client.request(
                    method,
                    url,
                    params=_clean_params(params),
                    json=json,
                    headers=merged_headers,
                    follow_redirects=follow_redirects,
                )
            except (
                httpx.ConnectError,
                httpx.ReadError,
                httpx.RemoteProtocolError,
                httpx.TimeoutException,
            ):
                if attempt >= self._max_retries:
                    raise
                self._sleep_before_retry(attempt, None)
                attempt += 1
                continue

            if response.status_code in _RETRYABLE_STATUS and attempt < self._max_retries:
                self._sleep_before_retry(attempt, response)
                attempt += 1
                continue

            if raise_on_error:
                raise_for_response(response)
            return response

    def request_raw_put(
        self,
        url: str,
        *,
        content: IO[bytes] | bytes,
        headers: Mapping[str, str] | None = None,
    ) -> httpx.Response:
        """``PUT`` raw bytes to an external (e.g. presigned S3) URL without the ``api_key`` header."""
        response = self._client.put(url, content=content, headers=dict(headers or {}))
        raise_for_response(response)
        return response

    @contextmanager
    def stream(
        self,
        method: str,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: httpx.Timeout | float | None = None,
        follow_redirects: bool = False,
        authenticated: bool = True,
    ) -> Iterator[httpx.Response]:
        """Open a streamed response. Error statuses raise after reading the body. No retries."""
        merged_headers = {**(self._auth_headers if authenticated else {}), **(headers or {})}
        extra: dict[str, Any] = {} if timeout is None else {"timeout": timeout}
        with self._client.stream(
            method,
            url,
            params=_clean_params(params),
            headers=merged_headers,
            follow_redirects=follow_redirects,
            **extra,
        ) as response:
            if response.status_code >= 400:
                response.read()
                raise_for_response(response)
            yield response

    def download(
        self,
        url: str,
        destination: str | Path,
        *,
        expected_md5: str | None = None,
        authenticated: bool = True,
        follow_redirects: bool = True,
    ) -> Path:
        """Stream ``url`` to ``destination``.

        Writes to a ``.part`` file first and renames on success. If
        ``destination`` is an existing directory, the file name is taken from the URL.

        Args:
            url: Absolute URL.
            destination: Target file path or directory.
            expected_md5: If given, verify the MD5 of the downloaded bytes.
            authenticated: Send the ``api_key`` header.
            follow_redirects: Follow HTTP redirects.

        Returns:
            Path of the written file.

        Raises:
            ChecksumMismatchError: If ``expected_md5`` does not match.
        """
        target = Path(destination)
        if target.is_dir():
            name = httpx.URL(url).path.rstrip("/").rsplit("/", 1)[-1] or "download"
            target = target / name
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".part")

        digest = hashlib.md5(usedforsecurity=False)
        with self.stream(
            "GET", url, follow_redirects=follow_redirects, authenticated=authenticated
        ) as response:
            with partial.open("wb") as fh:
                for chunk in response.iter_bytes(_CHUNK_SIZE):
                    fh.write(chunk)
                    digest.update(chunk)

        if expected_md5 and digest.hexdigest().lower() != expected_md5.lower():
            partial.unlink(missing_ok=True)
            raise ChecksumMismatchError(
                f"MD5 mismatch for {url}: expected {expected_md5}, got {digest.hexdigest()}"
            )
        partial.replace(target)
        return target

    def close(self) -> None:
        """Close the underlying client if this transport created it."""
        if self._owns_client:
            self._client.close()
