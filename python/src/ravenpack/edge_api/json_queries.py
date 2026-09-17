"""Schemas and endpoints for the synchronous ``/json`` query API."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ._base import ApiDateTime, JsonObject, RavenPackModel, RequestModel, Resource, validated
from .enums import Frequency

__all__ = ["AdHocQuery", "DatasetQuery", "DateRange", "JsonQueriesResource", "JsonQueryResult"]


class DateRange(RequestModel):
    """Body of ``POST /json/{dataset_uuid}/preview``."""

    start_date: ApiDateTime
    end_date: ApiDateTime


class AdHocQuery(RequestModel):
    """Body of ``POST /json``: query Edge data without a stored dataset.

    Granular: max 10,000 records. Daily: max 500 entities, one year range,
    one year lookback for functions.
    """

    start_date: ApiDateTime
    end_date: ApiDateTime
    fields: list[str | JsonObject]
    filters: JsonObject | None = None
    frequency: Frequency = Frequency.GRANULAR
    time_zone: str | None = None
    """IANA time zone, e.g. ``Europe/Madrid``."""
    conditions: JsonObject | None = None
    custom_fields: list[JsonObject] | JsonObject | None = None
    having: list[JsonObject] | JsonObject | None = None
    product: str = "EDGE"
    product_version: str = "1.0"


class DatasetQuery(RequestModel):
    """Body of ``POST /json/{dataset_uuid}``. Optional fields override the stored dataset."""

    start_date: ApiDateTime
    end_date: ApiDateTime
    fields: list[str | JsonObject] | None = None
    frequency: Frequency | None = None
    having: list[JsonObject] | JsonObject | None = None
    time_zone: str | None = None


class JsonQueryResult(RavenPackModel):
    """Result of a JSON query. Record keys depend on the requested fields."""

    records: list[dict[str, Any]] = []

    @classmethod
    def from_payload(cls, payload: Any) -> JsonQueryResult:
        """Build from a raw body, accepting either ``{"records": [...]}`` or a bare list."""
        if isinstance(payload, list):
            return cls(records=payload)
        return cls.model_validate(payload)

    def __len__(self) -> int:
        return len(self.records)


def _as_list(
    value: Sequence[JsonObject] | JsonObject | None,
) -> list[JsonObject] | JsonObject | None:
    return list(value) if isinstance(value, Sequence) else value


class JsonQueriesResource(Resource):
    """Query Edge data synchronously in JSON.

    Granular: max 10,000 records. Daily: max 500 entities, one year range,
    one year lookback for functions.
    """

    @validated
    def query(
        self,
        *,
        fields: Sequence[str | JsonObject],
        start_date: ApiDateTime,
        end_date: ApiDateTime,
        filters: JsonObject | None = None,
        frequency: Frequency = Frequency.GRANULAR,
        time_zone: str | None = None,
        conditions: JsonObject | None = None,
        custom_fields: Sequence[JsonObject] | JsonObject | None = None,
        having: Sequence[JsonObject] | JsonObject | None = None,
        product: str = "EDGE",
        product_version: str = "1.0",
    ) -> JsonQueryResult:
        """Ad hoc query without a stored dataset (``POST /json``).

        Args:
            fields: Output fields.
            start_date: Range start.
            end_date: Range end.
            filters: Filter expression.
            frequency: ``granular`` or ``daily``.
            time_zone: IANA time zone for ``TIMESTAMP_TZ``.
            conditions: Condition expression.
            custom_fields: Custom field definitions.
            having: ``having`` clause (daily).
            product: Underlying product.
            product_version: Product version.

        Raises:
            BadRequestError: Invalid query. Details in ``validation_errors``.
            AuthenticationError: Not authorized.
        """
        body = AdHocQuery(
            fields=list(fields),
            start_date=start_date,
            end_date=end_date,
            filters=filters,
            frequency=frequency,
            time_zone=time_zone,
            conditions=conditions,
            custom_fields=_as_list(custom_fields),
            having=_as_list(having),
            product=product,
            product_version=product_version,
        )
        return self.query_from(body)

    @validated
    def query_from(self, query: AdHocQuery) -> JsonQueryResult:
        """Run a prebuilt :class:`AdHocQuery` (``POST /json``)."""
        data = self._json("POST", self._http.url("data", "json"), json=query.to_payload())
        return JsonQueryResult.from_payload(data)

    @validated
    def query_dataset(
        self,
        dataset_uuid: str,
        *,
        start_date: ApiDateTime,
        end_date: ApiDateTime,
        fields: Sequence[str | JsonObject] | None = None,
        frequency: Frequency | None = None,
        having: Sequence[JsonObject] | JsonObject | None = None,
        time_zone: str | None = None,
    ) -> JsonQueryResult:
        """Query a stored dataset (``POST /json/{dataset_uuid}``).

        ``fields``, ``frequency``, ``having`` and ``time_zone`` override the stored definition when given.

        Raises:
            AuthenticationError: Not authorized for this dataset.
            NotFoundError: No such dataset.
        """
        body = DatasetQuery(
            start_date=start_date,
            end_date=end_date,
            fields=list(fields) if fields is not None else None,
            frequency=frequency,
            having=_as_list(having),
            time_zone=time_zone,
        )
        data = self._json(
            "POST", self._http.url("data", f"json/{dataset_uuid}"), json=body.to_payload()
        )
        return JsonQueryResult.from_payload(data)

    @validated
    def preview(
        self, dataset_uuid: str, *, start_date: ApiDateTime, end_date: ApiDateTime
    ) -> JsonQueryResult:
        """Small sample of a dataset (``POST /json/{dataset_uuid}/preview``).

        Daily datasets: up to five entities and 10 days per entity.
        """
        body = DateRange(start_date=start_date, end_date=end_date)
        data = self._json(
            "POST", self._http.url("data", f"json/{dataset_uuid}/preview"), json=body.to_payload()
        )
        return JsonQueryResult.from_payload(data)
