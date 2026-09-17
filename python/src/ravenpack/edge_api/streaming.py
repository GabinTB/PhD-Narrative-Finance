"""Real-time streaming API (``feed-edge.ravenpack.com``)."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from typing import Any

import httpx

from ._base import Resource, validated
from .exceptions import StreamDisconnectedError

__all__ = ["StreamingResource"]

logger = logging.getLogger(__name__)

_TRANSIENT_ERRORS = (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError)


class StreamingResource(Resource):
    """Subscribe to real-time Edge data for a dataset."""

    @validated
    def stream(
        self,
        dataset_uuid: str,
        *,
        time_zone: str | None = None,
        keep_alive: bool = True,
        silence_timeout: float = 60.0,
        reconnect: bool = True,
        max_reconnects: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield records from ``GET /json/{dataset_uuid}`` as they arrive.

        The server keeps the HTTP 200 response open and sends one JSON object per
        line. With ``keep_alive`` a newline is sent after 30s of silence; the
        connection is considered dead after ``silence_timeout`` seconds without
        any byte, as recommended by RavenPack.

        Records published while disconnected are not replayed.

        Args:
            dataset_uuid: Dataset to subscribe to.
            time_zone: Time zone for ``TIMESTAMP_TZ``.
            keep_alive: Ask the server for keep-alive newlines.
            silence_timeout: Read timeout in seconds before resetting the connection.
            reconnect: Reconnect automatically on silence or network errors.
            max_reconnects: Consecutive reconnect attempts allowed. ``None`` for unlimited.

        Yields:
            Decoded records. Keys depend on the dataset fields.

        Raises:
            AuthenticationError: 401, not authorized for the dataset.
            NotFoundError: 404, no such dataset.
            RateLimitError: 429, too many open real-time connections.
            StreamDisconnectedError: Connection lost and reconnection disabled or exhausted.
        """
        url = self._http.url("streaming", f"json/{dataset_uuid}")
        params = {"time_zone": time_zone, "keep_alive": "t" if keep_alive else None}
        timeout = httpx.Timeout(30.0, read=silence_timeout)
        failures = 0

        while True:
            try:
                with self._http.stream("GET", url, params=params, timeout=timeout) as response:
                    failures = 0
                    for line in response.iter_lines():
                        if line.strip():
                            yield json.loads(line)
                reason: Exception | None = None
            except _TRANSIENT_ERRORS as exc:
                reason = exc

            if not reconnect:
                if reason is None:
                    return
                raise StreamDisconnectedError(
                    f"Stream for {dataset_uuid} disconnected: {reason!r}"
                ) from reason
            if max_reconnects is not None and failures >= max_reconnects:
                raise StreamDisconnectedError(
                    f"Stream for {dataset_uuid} disconnected after {failures} reconnect attempts"
                ) from reason

            failures += 1
            delay = min(2.0**failures, 30.0)
            logger.warning(
                "RavenPack stream %s dropped (%r), reconnecting in %.0fs",
                dataset_uuid,
                reason,
                delay,
            )
            time.sleep(delay)
