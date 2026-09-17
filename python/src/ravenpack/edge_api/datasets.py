"""Dataset management (``/datasets``)."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any

from ._base import JsonObject, RavenPackModel, RequestModel, Resource, validated
from .enums import DatasetScope, Frequency

__all__ = [
    "Dataset",
    "DatasetCreate",
    "DatasetList",
    "DatasetReference",
    "DatasetSummary",
    "DatasetUpdate",
    "DatasetsResource",
]


class DatasetSummary(RavenPackModel):
    """Entry of ``GET /datasets``."""

    dataset_uuid: str
    name: str | None = None
    creation_time: datetime | None = None
    frequency: Frequency | str | None = None
    product: str | None = None
    tags: list[str] = []


class DatasetList(RavenPackModel):
    """Response of ``GET /datasets``."""

    count: int
    datasets: list[DatasetSummary] = []


class DatasetCreate(RequestModel):
    """Body of ``POST /datasets``.

    See the RavenPack Edge User Guide for the field list and the filter syntax
    (``{"and": [{"RP_ENTITY_ID": {"in": [...]}}, {"EVENT_RELEVANCE": {"gte": 90}}]}``).
    """

    name: str
    """Name of the dataset."""
    description: str | None = None
    """Short textual description."""
    tags: list[str] | None = None
    product: str = "EDGE"
    """Underlying data product."""
    product_version: str = "1.0"
    frequency: Frequency = Frequency.GRANULAR
    """``granular`` for raw records, ``daily`` for daily aggregates."""
    fields: list[str | JsonObject] | None = None
    """Output fields. Aggregated datasets may use function objects instead of names."""
    filters: JsonObject | None = None
    """Row filter expression."""
    conditions: JsonObject | None = None
    """Post-aggregation condition expression."""
    custom_fields: list[JsonObject] | JsonObject | None = None
    """Custom field definitions."""
    having: list[JsonObject] | JsonObject | None = None
    """``having`` clause for daily datasets."""


class DatasetUpdate(RequestModel):
    """Body of ``PUT /datasets/{dataset_uuid}``. Only provided fields are modified."""

    name: str | None = None
    description: str | None = None
    tags: list[str] | None = None
    product: str | None = None
    product_version: str | None = None
    frequency: Frequency | None = None
    fields: list[str | JsonObject] | None = None
    filters: JsonObject | None = None
    conditions: JsonObject | None = None
    custom_fields: list[JsonObject] | JsonObject | None = None
    having: list[JsonObject] | JsonObject | None = None


class Dataset(RavenPackModel):
    """Full dataset specification (``GET /datasets/{dataset_uuid}``)."""

    dataset_uuid: str
    """Unique identifier (read-only)."""
    name: str | None = None
    description: str | None = None
    tags: list[str] = []
    product: str | None = None
    product_version: str | None = None
    frequency: Frequency | str | None = None
    fields: list[Any] = []
    available_fields: list[str] = []
    filters: JsonObject | None = None
    conditions: JsonObject | None = None
    custom_fields: Any = None
    creation_time: datetime | None = None
    """UTC creation time (read-only)."""
    last_modified: datetime | None = None
    """UTC last modification time (read-only)."""


class DatasetReference(RavenPackModel):
    """Response of ``POST /datasets`` and ``PUT /datasets/{dataset_uuid}``."""

    dataset_uuid: str


class DatasetsResource(Resource):
    """Create, read, update, delete and list dataset definitions."""

    @validated
    def list(
        self,
        *,
        tags: Sequence[str] | None = None,
        scope: Sequence[DatasetScope] | None = None,
        product: Sequence[str] | None = None,
    ) -> DatasetList:
        """List datasets you can access (``GET /datasets``).

        Args:
            tags: Only datasets carrying these tags.
            scope: ``public``, ``private`` and/or ``shared``. Server default is ``private``.
            product: Only datasets built on these products.

        Returns:
            Dataset UUIDs and names.
        """
        params = {"tags": tags, "scope": scope, "product": product}
        return DatasetList.model_validate(
            self._json("GET", self._http.url("data", "datasets"), params=params)
        )

    @validated
    def create(
        self,
        *,
        name: str,
        fields: Sequence[str | JsonObject],
        filters: JsonObject | None = None,
        frequency: Frequency = Frequency.GRANULAR,
        description: str | None = None,
        tags: Sequence[str] | None = None,
        conditions: JsonObject | None = None,
        custom_fields: Sequence[JsonObject] | JsonObject | None = None,
        having: Sequence[JsonObject] | JsonObject | None = None,
        product: str = "EDGE",
        product_version: str = "1.0",
    ) -> DatasetReference:
        """Create a dataset definition (``POST /datasets``).

        Args:
            name: Dataset name.
            fields: Output fields, e.g. ``["TIMESTAMP_UTC", "RP_ENTITY_ID", "ENTITY_NAME"]``.
            filters: Filter expression, e.g. ``{"and": [{"EVENT_RELEVANCE": {"gte": 90}}]}``.
            frequency: ``granular`` or ``daily``.
            description: Free text description.
            tags: Tags for later filtering.
            conditions: Condition expression.
            custom_fields: Custom field definitions.
            having: ``having`` clause (daily datasets).
            product: Underlying product.
            product_version: Product version.

        Returns:
            The new ``dataset_uuid``.

        Raises:
            BadRequestError: Invalid definition. Details in ``validation_errors``.
        """
        body = DatasetCreate(
            name=name,
            fields=list(fields),
            filters=filters,
            frequency=frequency,
            description=description,
            tags=list(tags) if tags is not None else None,
            conditions=conditions,
            custom_fields=list(custom_fields)
            if isinstance(custom_fields, Sequence)
            else custom_fields,
            having=list(having) if isinstance(having, Sequence) else having,
            product=product,
            product_version=product_version,
        )
        return self.create_from(body)

    @validated
    def create_from(self, dataset: DatasetCreate) -> DatasetReference:
        """Create a dataset from a prebuilt :class:`DatasetCreate` (``POST /datasets``)."""
        data = self._json("POST", self._http.url("data", "datasets"), json=dataset.to_payload())
        return DatasetReference.model_validate(data)

    @validated
    def get(self, dataset_uuid: str) -> Dataset:
        """Get the full specification of a dataset (``GET /datasets/{dataset_uuid}``).

        Raises:
            AuthenticationError: Not authorized for this dataset.
            NotFoundError: No such dataset.
        """
        return Dataset.model_validate(
            self._json("GET", self._http.url("data", f"datasets/{dataset_uuid}"))
        )

    @validated
    def update(
        self,
        dataset_uuid: str,
        *,
        name: str | None = None,
        fields: Sequence[str | JsonObject] | None = None,
        filters: JsonObject | None = None,
        frequency: Frequency | None = None,
        description: str | None = None,
        tags: Sequence[str] | None = None,
        conditions: JsonObject | None = None,
        custom_fields: Sequence[JsonObject] | JsonObject | None = None,
        having: Sequence[JsonObject] | JsonObject | None = None,
        product: str | None = None,
        product_version: str | None = None,
    ) -> DatasetReference:
        """Modify a dataset (``PUT /datasets/{dataset_uuid}``).

        Only arguments that are not ``None`` are sent; everything else keeps its value.

        Raises:
            AuthenticationError: Not authorized for this dataset.
            NotFoundError: No such dataset.
        """
        body = DatasetUpdate(
            name=name,
            fields=list(fields) if fields is not None else None,
            filters=filters,
            frequency=frequency,
            description=description,
            tags=list(tags) if tags is not None else None,
            conditions=conditions,
            custom_fields=list(custom_fields)
            if isinstance(custom_fields, Sequence)
            else custom_fields,
            having=list(having) if isinstance(having, Sequence) else having,
            product=product,
            product_version=product_version,
        )
        data = self._json(
            "PUT", self._http.url("data", f"datasets/{dataset_uuid}"), json=body.to_payload()
        )
        return DatasetReference.model_validate(data)

    @validated
    def delete(self, dataset_uuid: str) -> None:
        """Delete a dataset (``DELETE /datasets/{dataset_uuid}``).

        Raises:
            AuthenticationError: Not authorized for this dataset.
            NotFoundError: No such dataset.
        """
        self._http.request("DELETE", self._http.url("data", f"datasets/{dataset_uuid}"))
