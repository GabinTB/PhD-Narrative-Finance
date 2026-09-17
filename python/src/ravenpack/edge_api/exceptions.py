"""Exception hierarchy for the RavenPack Edge API client."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .schemas.errors import ValidationErrorResponse

__all__ = [
    "APIError",
    "AuthenticationError",
    "BadRequestError",
    "ChecksumMismatchError",
    "ConfigurationError",
    "FileProcessingError",
    "GoneError",
    "JobFailedError",
    "NotFoundError",
    "PermissionDeniedError",
    "RateLimitError",
    "RavenPackError",
    "ServerError",
    "StreamDisconnectedError",
    "WaitTimeoutError",
]


class RavenPackError(Exception):
    """Base class for every error raised by this package."""


class ConfigurationError(RavenPackError):
    """Raised when the client cannot be configured (e.g. missing API key)."""


class APIError(RavenPackError):
    """An HTTP error returned by one of the RavenPack APIs.

    Attributes:
        status_code: HTTP status code of the response.
        message: Human readable message extracted from the response body.
        payload: Decoded JSON body, or raw text when the body is not JSON.
        validation_errors: Parsed ``ValidationError`` body when the API returned one.
        url: The requested URL.
    """

    def __init__(
        self,
        status_code: int,
        message: str,
        *,
        payload: Any = None,
        validation_errors: ValidationErrorResponse | None = None,
        url: str | None = None,
    ) -> None:
        super().__init__(f"[{status_code}] {message}")
        self.status_code = status_code
        self.message = message
        self.payload = payload
        self.validation_errors = validation_errors
        self.url = url


class BadRequestError(APIError):
    """HTTP 400: the request was not valid."""


class AuthenticationError(APIError):
    """HTTP 401: the API key is not authorized for this resource."""


class PermissionDeniedError(APIError):
    """HTTP 403: access denied to the requested resource."""


class NotFoundError(APIError):
    """HTTP 404: the dataset, job, entity, file or folder does not exist."""


class GoneError(APIError):
    """HTTP 410: the annotation results are obsolete and the content must be re-uploaded."""


class RateLimitError(APIError):
    """HTTP 429: too many requests, connections or concurrent datafile jobs."""


class ServerError(APIError):
    """HTTP 5xx: server side failure."""


class StreamDisconnectedError(RavenPackError):
    """The real-time stream dropped and reconnection is disabled or exhausted."""


class WaitTimeoutError(RavenPackError):
    """A polling helper exceeded its timeout."""


class JobFailedError(RavenPackError):
    """A datafile generation job finished with status ``error``."""


class FileProcessingError(RavenPackError):
    """An uploaded Annotations file finished with status ``FAILED``."""


class ChecksumMismatchError(RavenPackError):
    """The MD5 of a downloaded file does not match the one reported by the API."""
