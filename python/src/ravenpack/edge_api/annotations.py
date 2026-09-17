"""RavenPack Annotations API (``upload.ravenpack.com``): upload your own content and get it annotated."""

from __future__ import annotations

import builtins
import json
import mimetypes
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import Field

from ._base import (
    ApiDateTime,
    JsonObject,
    RavenPackModel,
    RequestModel,
    Resource,
    format_api_datetime,
    validated,
)
from ._http import HttpTransport
from .enums import FileStatus
from .errors import MessageResponse
from .exceptions import FileProcessingError, WaitTimeoutError

__all__ = [
    "AnnotationsResource",
    "FileCountSummary",
    "FileEntry",
    "FileList",
    "FileMetadataUpdate",
    "FilesResource",
    "Folder",
    "FolderCreate",
    "FolderList",
    "FolderUpdate",
    "FoldersResource",
    "MonthlyUsage",
    "Quota",
    "QuotaDetails",
    "SubscriptionUsage",
    "TextExtraction",
    "UploadRequest",
    "UploadResponse",
]


class FileEntry(RavenPackModel):
    """An uploaded file and its metadata (``FullFileEntry``)."""

    file_id: str
    file_name: str | None = None
    status: FileStatus | str | None = None
    upload_ts: datetime | None = None
    raw_size: int | None = None
    """Size of the original document in bytes."""
    folder_id: str | None = None
    tags: list[str] | None = None
    starred: bool | None = None
    trashed: bool | None = None
    error: str | None = None
    """Error code if uploaded but not processed, ``None`` otherwise."""

    @property
    def is_finished(self) -> bool:
        """Whether processing reached a terminal status."""
        return self.status in (FileStatus.COMPLETED, FileStatus.FAILED)


class FileList(RavenPackModel):
    """Response of ``GET /files``."""

    results: list[FileEntry] = []
    offset: int | None = None
    page_size: int | None = None


class UploadRequest(RequestModel):
    """Body of ``POST /files``."""

    file_name: str
    """File name registered for the content."""
    properties: JsonObject | None = None
    """Properties and metadata controlling processing."""
    source_url: str | None = None
    """URL of a file to fetch and analyze instead of uploading bytes."""


class UploadResponse(RavenPackModel):
    """Response of ``POST /files``."""

    file_id: str
    location: str | None = Field(default=None, alias="Location")
    """Presigned S3 URL to ``PUT`` the file content to."""


class FileMetadataUpdate(RequestModel):
    """Body of ``PATCH /files/{file_id}/metadata``. Only provided fields are modified."""

    file_name: str | None = None
    folder_id: str | None = None
    starred: bool | None = None
    trashed: bool | None = None
    tags: list[str] | None = None


class TextExtraction(RavenPackModel):
    """Response of ``GET /files/{file_id}/text-extraction``."""

    uuid: str | None = None
    title: str | None = None
    src: str | None = None
    body: list[Any] = []


class Folder(RavenPackModel):
    """An Annotations folder."""

    folder_id: str
    folder_name: str | None = None
    parent_folder_id: str | None = None
    create_ts: datetime | None = None
    starred: bool | None = None
    trashed: bool | None = None


class FolderList(RavenPackModel):
    """Response of ``GET /folders``."""

    results: list[Folder] = []


class FolderCreate(RequestModel):
    """Body of ``POST /folders``."""

    folder_name: str
    parent_folder_id: str | None = None
    starred: bool | None = None
    trashed: bool | None = None


class FolderUpdate(RequestModel):
    """Body of ``PATCH /folders/{folder_id}``. Only provided fields are modified."""

    folder_name: str | None = None
    parent_folder_id: str | None = None
    starred: bool | None = None
    trashed: bool | None = None


class FileCountSummary(RavenPackModel):
    """File counters in ``GET /quota``."""

    total: int | None = None
    error: int | None = None


class MonthlyUsage(RavenPackModel):
    """Current month consumption."""

    storage_bytes_used: int | None = None
    units_used: int | None = None


class SubscriptionUsage(RavenPackModel):
    """Subscription level consumption and limits."""

    max_units_allowed: int | None = None
    units_used: int | None = None
    units_remaining: int | None = None
    storage_bytes_used: int | None = None


class QuotaDetails(RavenPackModel):
    """``quota`` block of ``GET /quota``."""

    current_month: MonthlyUsage | None = None
    subscription: SubscriptionUsage | None = None


class Quota(RavenPackModel):
    """Response of ``GET /quota``."""

    files: FileCountSummary | None = None
    quota: QuotaDetails | None = None


class FilesResource(Resource):
    """Uploaded files and their annotation results."""

    @validated
    def list(
        self,
        *,
        start_date: ApiDateTime | None = None,
        end_date: ApiDateTime | None = None,
        tags: str | None = None,
        status: Sequence[FileStatus] | None = None,
        file_name: str | None = None,
        offset: int | None = None,
        page_size: int | None = None,
    ) -> FileList:
        """List uploaded files (``GET /files``).

        Args:
            start_date: Upload time lower bound.
            end_date: Upload time upper bound.
            tags: Tag filter.
            status: ``PROCESSING``, ``COMPLETED`` and/or ``FAILED``.
            file_name: File name filter.
            offset: Offset from the start of the results.
            page_size: Max items per page.
        """
        params = {
            "start_date": format_api_datetime(start_date) if start_date is not None else None,
            "end_date": format_api_datetime(end_date) if end_date is not None else None,
            "tags": tags,
            "status": status,
            "file_name": file_name,
            "offset": offset,
            "page_size": page_size,
        }
        return FileList.model_validate(
            self._json("GET", self._http.url("annotations", "files"), params=params)
        )

    @validated
    def create_upload(
        self,
        file_name: str,
        *,
        properties: JsonObject | None = None,
        source_url: str | None = None,
    ) -> UploadResponse:
        """Register an upload (``POST /files``), step 1 of 2.

        With ``source_url`` RavenPack fetches the content itself and no ``PUT`` is needed.
        Otherwise ``PUT`` the bytes to ``location`` (see :meth:`upload`).

        Raises:
            BadRequestError: Body missing or badly formatted.
            RateLimitError: Server under load; retry after 5 seconds.
        """
        body = UploadRequest(file_name=file_name, properties=properties, source_url=source_url)
        data = self._json("POST", self._http.url("annotations", "files"), json=body.to_payload())
        return UploadResponse.model_validate(data)

    @validated
    def upload(
        self,
        path: str | Path,
        *,
        file_name: str | None = None,
        properties: JsonObject | None = None,
    ) -> UploadResponse:
        """Upload a local file for annotation (``POST /files`` then ``PUT`` to the presigned URL).

        Args:
            path: Local file.
            file_name: Registered name. Defaults to the local file name.
            properties: Processing properties and metadata.

        Returns:
            The upload registration, including ``file_id``.
        """
        source = Path(path)
        registration = self.create_upload(file_name or source.name, properties=properties)
        if not registration.location:
            raise FileProcessingError(f"No upload location returned for {registration.file_id}")
        content_type = mimetypes.guess_type(source.name)[0]
        headers = {"x-amz-server-side-encryption": "AES256"}
        if content_type:
            headers["Content-Type"] = content_type
        with source.open("rb") as fh:
            self._http.request_raw_put(registration.location, content=fh, headers=headers)
        return registration

    @validated
    def get(self, file_id: str) -> FileEntry:
        """Retrieve an uploaded file record (``GET /files/{file_id}``).

        Raises:
            NotFoundError: File not found.
        """
        return FileEntry.model_validate(
            self._json("GET", self._http.url("annotations", f"files/{file_id}"))
        )

    @validated
    def delete(self, file_id: str) -> MessageResponse:
        """Delete a file and all its results (``DELETE /files/{file_id}``).

        Raises:
            NotFoundError: File not found.
        """
        data = self._json("DELETE", self._http.url("annotations", f"files/{file_id}"))
        return MessageResponse.model_validate(data)

    @validated
    def get_analytics(
        self, file_id: str, *, jsonlines: bool = False
    ) -> builtins.list[dict[str, Any]]:
        """Tabular entity and event detections (``GET /files/{file_id}/analytics``).

        Args:
            file_id: File identifier.
            jsonlines: Request ``application/x-jsonlines`` instead of a JSON array.

        Returns:
            One dict per detection.

        Raises:
            PermissionDeniedError: File not associated with your user.
            NotFoundError: File not found.
            GoneError: Annotations system upgraded since upload; re-upload the content.
        """
        accept = "application/x-jsonlines" if jsonlines else "application/json"
        response = self._http.request(
            "GET",
            self._http.url("annotations", f"files/{file_id}/analytics"),
            headers={"accept": accept},
            follow_redirects=True,
        )
        if jsonlines:
            return [json.loads(line) for line in response.text.splitlines() if line.strip()]
        data = response.json()
        return data if isinstance(data, builtins.list) else [data]

    @validated
    def get_annotated(self, file_id: str) -> str:
        """Annotated RPXML document (``GET /files/{file_id}/annotated``).

        Raises:
            NotFoundError: File not found.
            GoneError: Annotations system upgraded since upload; re-upload the content.
        """
        response = self._http.request(
            "GET",
            self._http.url("annotations", f"files/{file_id}/annotated"),
            headers={"accept": "application/xml"},
            follow_redirects=True,
        )
        return response.text

    @validated
    def get_metadata(self, file_id: str) -> FileEntry:
        """File metadata: size, tags, starred, trashed, folder (``GET /files/{file_id}/metadata``)."""
        data = self._json("GET", self._http.url("annotations", f"files/{file_id}/metadata"))
        return FileEntry.model_validate(data)

    @validated
    def update_metadata(
        self,
        file_id: str,
        *,
        file_name: str | None = None,
        folder_id: str | None = None,
        starred: bool | None = None,
        trashed: bool | None = None,
        tags: Sequence[str] | None = None,
    ) -> FileEntry:
        """Modify file metadata (``PATCH /files/{file_id}/metadata``). ``None`` arguments are left untouched."""
        body = FileMetadataUpdate(
            file_name=file_name,
            folder_id=folder_id,
            starred=starred,
            trashed=trashed,
            tags=builtins.list(tags) if tags is not None else None,
        )
        data = self._json(
            "PATCH",
            self._http.url("annotations", f"files/{file_id}/metadata"),
            json=body.to_payload(),
        )
        return FileEntry.model_validate(data)

    @validated
    def get_text_extraction(self, file_id: str) -> TextExtraction:
        """Extracted text of the original file (``GET /files/{file_id}/text-extraction``)."""
        data = self._json("GET", self._http.url("annotations", f"files/{file_id}/text-extraction"))
        return TextExtraction.model_validate(data)

    @validated
    def wait(
        self, file_id: str, *, poll_interval: float = 5.0, timeout: float | None = None
    ) -> FileEntry:
        """Poll until processing is ``COMPLETED``.

        Raises:
            FileProcessingError: Processing ended with ``FAILED``.
            WaitTimeoutError: ``timeout`` elapsed.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            entry = self.get_metadata(file_id)
            if entry.status == FileStatus.COMPLETED:
                return entry
            if entry.status == FileStatus.FAILED:
                raise FileProcessingError(f"File {file_id} failed processing: {entry.error}")
            if deadline is not None and time.monotonic() + poll_interval > deadline:
                raise WaitTimeoutError(f"File {file_id} still {entry.status} after {timeout}s")
            time.sleep(poll_interval)


class FoldersResource(Resource):
    """Folders organising uploaded files."""

    @validated
    def list(self) -> FolderList:
        """List all folders (``GET /folders``)."""
        return FolderList.model_validate(
            self._json("GET", self._http.url("annotations", "folders"))
        )

    @validated
    def create(
        self,
        folder_name: str,
        *,
        parent_folder_id: str | None = None,
        starred: bool | None = None,
        trashed: bool | None = None,
    ) -> Folder:
        """Create a folder (``POST /folders``).

        Raises:
            BadRequestError: Invalid request.
        """
        body = FolderCreate(
            folder_name=folder_name,
            parent_folder_id=parent_folder_id,
            starred=starred,
            trashed=trashed,
        )
        return Folder.model_validate(
            self._json("POST", self._http.url("annotations", "folders"), json=body.to_payload())
        )

    @validated
    def update(
        self,
        folder_id: str,
        *,
        folder_name: str | None = None,
        parent_folder_id: str | None = None,
        starred: bool | None = None,
        trashed: bool | None = None,
    ) -> Folder:
        """Modify a folder (``PATCH /folders/{folder_id}``). ``None`` arguments are left untouched."""
        body = FolderUpdate(
            folder_name=folder_name,
            parent_folder_id=parent_folder_id,
            starred=starred,
            trashed=trashed,
        )
        return Folder.model_validate(
            self._json(
                "PATCH",
                self._http.url("annotations", f"folders/{folder_id}"),
                json=body.to_payload(),
            )
        )

    @validated
    def delete(self, folder_id: str) -> Folder:
        """Delete a folder and all associated data and results (``DELETE /folders/{folder_id}``).

        Raises:
            NotFoundError: Folder not found, not owned or already marked for deletion.
        """
        return Folder.model_validate(
            self._json("DELETE", self._http.url("annotations", f"folders/{folder_id}"))
        )


class AnnotationsResource(Resource):
    """Entry point for the Annotations API: ``files``, ``folders`` and ``quota()``."""

    def __init__(self, http: HttpTransport) -> None:
        super().__init__(http)
        self.files = FilesResource(http)
        self.folders = FoldersResource(http)

    @validated
    def quota(self) -> Quota:
        """Storage and unit consumption (``GET /quota``)."""
        return Quota.model_validate(self._json("GET", self._http.url("annotations", "quota")))
