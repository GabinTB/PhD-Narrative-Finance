"""Typed Python client for the RavenPack Edge REST APIs.

Covers the streaming API, data query API (datasets, datafiles, jobs, JSON
queries), reference API (entity mapping and reference, taxonomy, historical
flat files, documents) and the RavenPack Annotations API.

The API key is read from ``RAVENPACK_API_KEY`` (environment or ``.env``).

Example:
    >>> from ravenpack.edge_api import RavenPackClient
    >>> rp = RavenPackClient()
    >>> rp.status().status
    'OK'
"""

from ._base import (
    API_DATETIME_FORMAT,
    ApiDate,
    ApiDateTime,
    JsonObject,
    RavenPackModel,
    RequestModel,
    format_api_date,
    format_api_datetime,
)
from ._config import ANNOTATIONS_URL, API_KEY_ENV_VAR, DATA_URL, STREAMING_URL, resolve_api_key
from .annotations import (
    FileCountSummary,
    FileEntry,
    FileList,
    FileMetadataUpdate,
    Folder,
    FolderCreate,
    FolderList,
    FolderUpdate,
    MonthlyUsage,
    Quota,
    QuotaDetails,
    SubscriptionUsage,
    TextExtraction,
    UploadRequest,
    UploadResponse,
)
from .client import RavenPackClient
from .datafiles import (
    DatafileCount,
    DatafileCountRequest,
    DatafileJob,
    DatafileRequest,
    DatafileToken,
    JobList,
)
from .datasets import (
    Dataset,
    DatasetCreate,
    DatasetList,
    DatasetReference,
    DatasetSummary,
    DatasetUpdate,
)
from .documents import DocumentUrl
from .entities import (
    EntityData,
    EntityIdentifier,
    EntityMappingRequest,
    EntityMappingResponse,
    EntityReference,
    IdentifierMapping,
    MappedEntity,
)
from .enums import (
    DatasetScope,
    FileStatus,
    Frequency,
    JobStatus,
    ReferenceEntityType,
    ReferenceFileType,
)
from .errors import MessageResponse, ServerStatus, ValidationErrorDetail, ValidationErrorResponse
from .exceptions import (
    APIError,
    AuthenticationError,
    BadRequestError,
    ChecksumMismatchError,
    ConfigurationError,
    FileProcessingError,
    GoneError,
    JobFailedError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    RavenPackError,
    ServerError,
    StreamDisconnectedError,
    WaitTimeoutError,
)
from .history import FlatFile
from .json_queries import AdHocQuery, DatasetQuery, DateRange, JsonQueryResult
from .taxonomy import TaxonomyCategory, TaxonomyQuery, TaxonomyResponse

__all__ = [
    "ANNOTATIONS_URL",
    "API_DATETIME_FORMAT",
    "API_KEY_ENV_VAR",
    "DATA_URL",
    "STREAMING_URL",
    "APIError",
    "AdHocQuery",
    "ApiDate",
    "ApiDateTime",
    "AuthenticationError",
    "BadRequestError",
    "ChecksumMismatchError",
    "ConfigurationError",
    "DatafileCount",
    "DatafileCountRequest",
    "DatafileJob",
    "DatafileRequest",
    "DatafileToken",
    "Dataset",
    "DatasetCreate",
    "DatasetList",
    "DatasetQuery",
    "DatasetReference",
    "DatasetScope",
    "DatasetSummary",
    "DatasetUpdate",
    "DateRange",
    "DocumentUrl",
    "EntityData",
    "EntityIdentifier",
    "EntityMappingRequest",
    "EntityMappingResponse",
    "EntityReference",
    "FileCountSummary",
    "FileEntry",
    "FileList",
    "FileMetadataUpdate",
    "FileProcessingError",
    "FileStatus",
    "FlatFile",
    "Folder",
    "FolderCreate",
    "FolderList",
    "FolderUpdate",
    "Frequency",
    "GoneError",
    "IdentifierMapping",
    "JobFailedError",
    "JobList",
    "JobStatus",
    "JsonObject",
    "JsonQueryResult",
    "MappedEntity",
    "MessageResponse",
    "MonthlyUsage",
    "NotFoundError",
    "PermissionDeniedError",
    "Quota",
    "QuotaDetails",
    "RateLimitError",
    "RavenPackClient",
    "RavenPackError",
    "RavenPackModel",
    "ReferenceEntityType",
    "ReferenceFileType",
    "RequestModel",
    "ServerError",
    "ServerStatus",
    "StreamDisconnectedError",
    "SubscriptionUsage",
    "TaxonomyCategory",
    "TaxonomyQuery",
    "TaxonomyResponse",
    "TextExtraction",
    "UploadRequest",
    "UploadResponse",
    "ValidationErrorDetail",
    "ValidationErrorResponse",
    "WaitTimeoutError",
    "format_api_date",
    "format_api_datetime",
    "resolve_api_key",
]
