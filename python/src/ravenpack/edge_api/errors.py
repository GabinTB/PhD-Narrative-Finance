"""Error bodies and generic response envelopes."""

from __future__ import annotations

from typing import Any

from ._base import RavenPackModel

__all__ = ["MessageResponse", "ServerStatus", "ValidationErrorDetail", "ValidationErrorResponse"]


class ValidationErrorDetail(RavenPackModel):
    """One validation error returned by the Edge data API."""

    parameter_name: str | None = None
    """Offending request parameter, e.g. ``fields``."""
    reason: str | None = None
    """Why the value was rejected."""
    type: str | None = None
    """Error type, e.g. ``ValidationError``."""
    value: Any = None
    """The rejected value."""


class ValidationErrorResponse(RavenPackModel):
    """Body returned by the Edge data API on HTTP 400."""

    endpoint: str | None = None
    """Endpoint on which the error occurred."""
    errors: list[ValidationErrorDetail] = []


class MessageResponse(RavenPackModel):
    """``{"message": ...}`` body used by the Annotations API (``RequestError``)."""

    message: str | None = None


class ServerStatus(RavenPackModel):
    """Response of ``GET /status``."""

    status: str
    """``"OK"`` when the server is healthy."""
