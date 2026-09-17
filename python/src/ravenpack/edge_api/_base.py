"""Base models, shared annotated types, and the base for endpoint-group resources."""

from __future__ import annotations

from datetime import date, datetime
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import BaseModel, ConfigDict, PlainSerializer, validate_call

if TYPE_CHECKING:
    from ._http import HttpTransport

__all__ = [
    "API_DATETIME_FORMAT",
    "ApiDate",
    "ApiDateTime",
    "JsonObject",
    "RavenPackModel",
    "RequestModel",
    "Resource",
    "format_api_date",
    "format_api_datetime",
    "validated",
]

API_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"

JsonObject = dict[str, Any]
"""Free-form JSON object (filters, conditions, custom fields, records...)."""


def format_api_datetime(value: datetime | date | str) -> str:
    """Format a datetime-like value as ``YYYY-MM-DD HH:MM:SS``.

    ``date`` values are expanded to midnight. Strings are passed through untouched.
    """
    if isinstance(value, datetime):
        return value.strftime(API_DATETIME_FORMAT)
    if isinstance(value, date):
        return value.strftime("%Y-%m-%d 00:00:00")
    return value


def format_api_date(value: datetime | date | str) -> str:
    """Format a date-like value as ``YYYY-MM-DD``. Strings are passed through untouched."""
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return value


ApiDateTime = Annotated[
    datetime | date | str, PlainSerializer(format_api_datetime, return_type=str)
]
"""Request datetime: accepts ``datetime``, ``date`` or a preformatted string."""

ApiDate = Annotated[date | str, PlainSerializer(format_api_date, return_type=str)]
"""Request date: accepts ``date``/``datetime`` or a preformatted ``YYYY-MM-DD`` string."""


class RavenPackModel(BaseModel):
    """Base for all schemas.

    Unknown fields are kept (``extra="allow"``) so that fields added by
    RavenPack after this client was written are not silently dropped.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)


class RequestModel(RavenPackModel):
    """Base for request bodies."""

    def to_payload(self) -> JsonObject:
        """Serialize to the JSON body sent to the API (``None`` values omitted)."""
        return self.model_dump(mode="json", by_alias=True, exclude_none=True)


validated = validate_call(config=ConfigDict(arbitrary_types_allowed=True))
"""Runtime validation of public method arguments against their type hints."""


class Resource:
    """A group of related endpoints sharing one :class:`HttpTransport`."""

    def __init__(self, http: HttpTransport) -> None:
        self._http = http

    def _json(self, *args: Any, **kwargs: Any) -> Any:
        return self._http.request(*args, **kwargs).json()
