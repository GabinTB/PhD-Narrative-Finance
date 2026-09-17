"""High level client for the RavenPack Edge APIs."""

from __future__ import annotations

import os
from types import TracebackType

import httpx

from ._config import ANNOTATIONS_URL, API_KEY_ENV_VAR, DATA_URL, STREAMING_URL, resolve_api_key
from ._http import HttpTransport
from .annotations import AnnotationsResource
from .datafiles import DatafilesResource, JobsResource
from .datasets import DatasetsResource
from .documents import DocumentsResource
from .entities import EntitiesResource
from .errors import ServerStatus
from .history import HistoryResource
from .json_queries import JsonQueriesResource
from .streaming import StreamingResource
from .taxonomy import TaxonomyResource

__all__ = ["RavenPackClient"]


class RavenPackClient:
    """Synchronous client covering every RavenPack Edge REST endpoint.

    The API key is read from ``api_key``, else from ``RAVENPACK_API_KEY`` in the
    environment or the nearest ``.env``.

    Attributes:
        streaming: Real-time feed (``feed-edge``).
        datasets: Dataset definitions.
        datafiles: Asynchronous CSV/Excel generation.
        jobs: Datafile job tracking, polling and download.
        json: Synchronous JSON queries.
        entities: Entity mapping and reference data.
        taxonomy: Event taxonomy.
        history: Historical flat files.
        documents: Story URLs.
        annotations: Annotations API (``files``, ``folders``, ``quota()``).

    Example:
        >>> with RavenPackClient() as rp:
        ...     ds = rp.datasets.create(name="test", fields=["TIMESTAMP_UTC", "RP_ENTITY_ID"])
        ...     rows = rp.json.query_dataset(ds.dataset_uuid, start_date="2021-01-01", end_date="2021-01-02")
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        env_file: str | os.PathLike[str] | None = None,
        env_var: str = API_KEY_ENV_VAR,
        timeout: float = 60.0,
        max_retries: int = 3,
        streaming_url: str = STREAMING_URL,
        data_url: str = DATA_URL,
        annotations_url: str = ANNOTATIONS_URL,
        http_client: httpx.Client | None = None,
    ) -> None:
        """Create a client.

        Args:
            api_key: Explicit API key.
            env_file: ``.env`` path. Defaults to the nearest ``.env`` from the working directory.
            env_var: Environment variable holding the key.
            timeout: Default request timeout in seconds.
            max_retries: Retries on 429/502/503/504 and transport errors.
            streaming_url: Streaming base URL.
            data_url: Data query and reference base URL.
            annotations_url: Annotations base URL.
            http_client: Preconfigured ``httpx.Client`` (proxy, CA bundle). Not closed by :meth:`close`.

        Raises:
            ConfigurationError: No API key found.
        """
        self._http = HttpTransport(
            resolve_api_key(api_key, env_file=env_file, env_var=env_var),
            streaming_url=streaming_url,
            data_url=data_url,
            annotations_url=annotations_url,
            timeout=timeout,
            max_retries=max_retries,
            client=http_client,
        )
        self.streaming = StreamingResource(self._http)
        self.datasets = DatasetsResource(self._http)
        self.jobs = JobsResource(self._http)
        self.datafiles = DatafilesResource(self._http, self.jobs)
        self.json = JsonQueriesResource(self._http)
        self.entities = EntitiesResource(self._http)
        self.taxonomy = TaxonomyResource(self._http)
        self.history = HistoryResource(self._http)
        self.documents = DocumentsResource(self._http)
        self.annotations = AnnotationsResource(self._http)

    def status(self) -> ServerStatus:
        """Check connectivity and server health (``GET /status``)."""
        data = self._http.request("GET", self._http.url("data", "status")).json()
        return ServerStatus.model_validate(data)

    def close(self) -> None:
        """Release network resources."""
        self._http.close()

    def __enter__(self) -> RavenPackClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
