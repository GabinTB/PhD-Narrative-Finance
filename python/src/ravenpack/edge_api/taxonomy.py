"""Event taxonomy (``/taxonomy``)."""

from __future__ import annotations

from collections.abc import Sequence

from ._base import RavenPackModel, RequestModel, Resource, validated

__all__ = ["TaxonomyCategory", "TaxonomyQuery", "TaxonomyResource", "TaxonomyResponse"]


class TaxonomyQuery(RequestModel):
    """Body of ``POST /taxonomy``. Each list filters one level of the taxonomy."""

    topics: list[str] = []
    groups: list[str] = []
    types: list[str] = []
    sub_types: list[str] = []
    roles: list[str] = []
    categories: list[str] = []
    product: str = "edge"


class TaxonomyCategory(RavenPackModel):
    """One event category of the RavenPack taxonomy."""

    category: str
    description: str | None = None
    topic: str | None = None
    group: str | None = None
    type: str | None = None
    sub_type: str | None = None
    property: str | None = None
    fact_level: str | None = None
    scheduled: bool | None = None
    valid_entity_types: list[str] = []


class TaxonomyResponse(RavenPackModel):
    """Response of ``POST /taxonomy``."""

    categories: list[TaxonomyCategory] = []


class TaxonomyResource(Resource):
    """Browse the RavenPack event taxonomy."""

    @validated
    def query(
        self,
        *,
        topics: Sequence[str] = (),
        groups: Sequence[str] = (),
        types: Sequence[str] = (),
        sub_types: Sequence[str] = (),
        roles: Sequence[str] = (),
        categories: Sequence[str] = (),
        product: str = "edge",
    ) -> TaxonomyResponse:
        """Query the taxonomy (``POST /taxonomy``). Empty filters match everything at that level.

        Example:
            ``client.taxonomy.query(categories=["earnings-above-expectations", "product-recall"])``
        """
        body = TaxonomyQuery(
            topics=list(topics),
            groups=list(groups),
            types=list(types),
            sub_types=list(sub_types),
            roles=list(roles),
            categories=list(categories),
            product=product,
        )
        return TaxonomyResponse.model_validate(
            self._json("POST", self._http.url("data", "taxonomy"), json=body.to_payload())
        )
