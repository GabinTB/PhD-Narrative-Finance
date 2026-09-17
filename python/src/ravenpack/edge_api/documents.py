"""Document properties (``/document``)."""

from __future__ import annotations

from ._base import RavenPackModel, Resource, validated

__all__ = ["DocumentUrl", "DocumentsResource"]


class DocumentUrl(RavenPackModel):
    """Response of ``GET /document/{rp_document_id}/url``."""

    url: str
    """Original URL (MoreOver) or RavenPack secure platform URL (premium providers)."""


class DocumentsResource(Resource):
    """Access story content."""

    @validated
    def get_url(self, rp_document_id: str) -> DocumentUrl:
        """URL of a story's content (``GET /document/{rp_document_id}/url``).

        Rate limit: 10 requests per minute, 1,000 per day.

        Raises:
            NotFoundError: Document not found.
            RateLimitError: Rate limit hit.
        """
        data = self._json("GET", self._http.url("data", f"document/{rp_document_id}/url"))
        return DocumentUrl.model_validate(data)
