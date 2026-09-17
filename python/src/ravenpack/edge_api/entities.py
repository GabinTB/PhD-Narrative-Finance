"""Entity reference service (``/entity-mapping``, ``/entity-reference``)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any

from pydantic import RootModel

from ._base import ApiDate, RavenPackModel, RequestModel, Resource, format_api_date, validated
from .enums import ReferenceEntityType, ReferenceFileType
from .exceptions import APIError

__all__ = [
    "EntitiesResource",
    "EntityData",
    "EntityIdentifier",
    "EntityMappingRequest",
    "EntityMappingResponse",
    "EntityReference",
    "IdentifierMapping",
    "MappedEntity",
]


class EntityIdentifier(RequestModel):
    """Identifying information for an entity to map (``EntityMappingSearch``)."""

    client_id: str | None = None
    """Your own reference, echoed back and not used for matching."""
    name: str | None = None
    """Common name, e.g. ``Amazon Inc.``."""
    entity_type: str | None = None
    """RavenPack entity type code, e.g. ``COMP``."""
    date: ApiDate | None = None
    """Date at which the identifiers are valid."""
    isin: str | None = None
    cusip: str | None = None
    sedol: str | None = None
    figi: str | None = None
    lei: str | None = None
    ticker: str | None = None
    listing: str | None = None
    """``MIC:TICKER``, e.g. ``XNAS:AMZN``."""
    url: str | None = None
    """Web address of the entity."""


class EntityMappingRequest(RequestModel):
    """Body of ``POST /entity-mapping``."""

    identifiers: list[EntityIdentifier]
    product: str = "edge"
    product_version: str = "1.0"


class MappedEntity(RavenPackModel):
    """A candidate match in RavenPack's entity universe, ranked by ``score``."""

    rp_entity_id: str
    rp_entity_name: str | None = None
    rp_entity_type: str | None = None
    score: float | None = None


class IdentifierMapping(RavenPackModel):
    """Mapping result for one submitted identifier."""

    requested_data: EntityIdentifier | None = None
    rp_entities: list[MappedEntity] = []
    """Matches sorted by relevance. Empty when unmatched."""
    errors: list[dict[str, Any]] = []

    @property
    def best_match(self) -> MappedEntity | None:
        """Highest scoring candidate, if any."""
        if not self.rp_entities:
            return None
        return max(
            self.rp_entities, key=lambda e: e.score if e.score is not None else float("-inf")
        )


class EntityMappingResponse(RavenPackModel):
    """Response of ``POST /entity-mapping``."""

    identifiers_mapped: list[IdentifierMapping] = []
    identifiers_submitted: int | None = None


class EntityData(RavenPackModel):
    """A value valid over ``[range_start, range_end]`` (``None`` means unbounded)."""

    data_value: Any = None
    range_start: str | None = None
    """Valid from this date. ``None``: valid from or before 2000."""
    range_end: str | None = None
    """Valid until this date. ``None``: still valid."""


class EntityReference(RootModel[list[dict[str, list[EntityData]]]]):
    """Response of ``GET /entity-reference/{entity_id}``.

    The raw body is a list of ``{data_type: [EntityData, ...]}`` objects.
    """

    def to_dict(self) -> dict[str, list[EntityData]]:
        """Merge all objects into a single ``{data_type: [EntityData, ...]}`` mapping."""
        merged: dict[str, list[EntityData]] = {}
        for item in self.root:
            for key, values in item.items():
                merged.setdefault(key, []).extend(values)
        return merged


class EntitiesResource(Resource):
    """Map identifiers to RP_ENTITY_IDs and fetch entity reference data."""

    @validated
    def map(
        self,
        identifiers: Sequence[EntityIdentifier | Mapping[str, Any]],
        *,
        product: str = "edge",
        product_version: str = "1.0",
    ) -> EntityMappingResponse:
        """Map names, listings, ISINs, CUSIPs... into RavenPack's universe (``POST /entity-mapping``).

        Unmatched identifiers come back with no ``rp_entities`` and populated
        ``errors``. Multiple matches are ranked by ``score``.

        Args:
            identifiers: :class:`EntityIdentifier` objects or dicts with the same keys.
            product: ``edge`` or ``rpa``.
            product_version: Product version.
        """
        body = EntityMappingRequest(
            identifiers=[EntityIdentifier.model_validate(i) for i in identifiers],
            product=product,
            product_version=product_version,
        )
        data = self._json("POST", self._http.url("data", "entity-mapping"), json=body.to_payload())
        return EntityMappingResponse.model_validate(data)

    @staticmethod
    def _reference_params(
        entity_type: ReferenceEntityType | None,
        file_type: ReferenceFileType,
        on_date: date | str | None,
        product: str,
        product_version: str,
    ) -> dict[str, Any]:
        return {
            "entity_type": entity_type,
            "type": file_type,
            "date": format_api_date(on_date) if on_date is not None else None,
            "product": product,
            "product_version": product_version,
        }

    @validated
    def reference_file_url(
        self,
        *,
        entity_type: ReferenceEntityType | None = None,
        file_type: ReferenceFileType = ReferenceFileType.FULL,
        on_date: date | str | None = None,
        product: str = "edge",
        product_version: str = "1.0",
    ) -> str:
        """Resolve the location of an entity reference file (``GET /entity-reference``, HTTP 303).

        The ``api_key`` header is required to fetch the returned URL.

        Args:
            entity_type: Restrict to one entity type or taxonomy. ``None`` for all.
            file_type: ``full`` or ``delta`` (no delta for taxonomies).
            on_date: File date; only the past week is available.
            product: ``edge`` or ``rpa``.
            product_version: Product version.

        Returns:
            Absolute URL of the CSV file.
        """
        response = self._http.request(
            "GET",
            self._http.url("data", "entity-reference"),
            params=self._reference_params(
                entity_type, file_type, on_date, product, product_version
            ),
            follow_redirects=False,
        )
        location = response.headers.get("Location")
        if not location:
            raise APIError(
                response.status_code,
                "entity-reference did not return a Location header",
                url=str(response.request.url),
            )
        return self._http.absolute_url("data", location)

    @validated
    def download_reference_file(
        self,
        destination: str | Path,
        *,
        entity_type: ReferenceEntityType | None = None,
        file_type: ReferenceFileType = ReferenceFileType.FULL,
        on_date: date | str | None = None,
        product: str = "edge",
        product_version: str = "1.0",
    ) -> Path:
        """Download an entity reference file to ``destination`` (file or existing directory).

        Files are refreshed daily after midnight UTC and published after 07:30 UTC.
        """
        url = self.reference_file_url(
            entity_type=entity_type,
            file_type=file_type,
            on_date=on_date,
            product=product,
            product_version=product_version,
        )
        return self._http.download(url, destination)

    @validated
    def get_reference(self, entity_id: str) -> EntityReference:
        """Reference data for one entity (``GET /entity-reference/{entity_id}``).

        Several values may exist per data type, with overlapping or distinct validity ranges.

        Raises:
            NotFoundError: No such entity.
        """
        data = self._json("GET", self._http.url("data", f"entity-reference/{entity_id}"))
        return EntityReference.model_validate(data)
