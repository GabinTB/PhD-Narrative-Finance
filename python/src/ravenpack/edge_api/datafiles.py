"""Schemas and endpoints for asynchronous datafile generation (``/datafile``) and job tracking (``/jobs``)."""

from __future__ import annotations

import time
from collections.abc import Sequence
from datetime import date, datetime
from pathlib import Path

from ._base import ApiDateTime, RavenPackModel, RequestModel, Resource, validated
from ._http import HttpTransport
from .enums import JobStatus
from .exceptions import JobFailedError, WaitTimeoutError

__all__ = [
    "DatafileCount",
    "DatafileCountRequest",
    "DatafileJob",
    "DatafileRequest",
    "DatafileToken",
    "DatafilesResource",
    "JobList",
    "JobsResource",
]


class DatafileRequest(RequestModel):
    """Body of ``POST /datafile/{dataset_uuid}``.

    Limits: 50M records (granular), 15M records (aggregated), 1M records for
    Excel, 5 years max range for daily datasets.
    """

    start_date: ApiDateTime
    end_date: ApiDateTime
    format: str = "csv"
    """Output format, ``csv`` or Excel."""
    compressed: bool = False
    """Compress the file for transmission."""
    notify: bool = False
    """Send an email notification on completion."""


class DatafileToken(RavenPackModel):
    """Response of ``POST /datafile/{dataset_uuid}``."""

    token: str
    """Job token, used with the ``/jobs`` endpoints."""
    estimated_completion: datetime | None = None


class DatafileCountRequest(RequestModel):
    """Body of ``POST /datafile/{dataset_uuid}/count``."""

    start_date: ApiDateTime
    end_date: ApiDateTime


class DatafileCount(RavenPackModel):
    """Response of ``POST /datafile/{dataset_uuid}/count``.

    Exact for granular datasets, estimate (<1% error) for daily ones,
    unavailable when functions and conditions are set.
    """

    count: int | None = None
    """Number of rows."""
    stories: int | None = None
    """Number of distinct stories."""
    entities: int | None = None
    """Number of distinct entities."""


class DatafileJob(RavenPackModel):
    """A datafile generation job (``DatafileJob`` schema)."""

    token: str
    status: JobStatus | str
    start_date: datetime | None = None
    end_date: datetime | None = None
    submitted: datetime | None = None
    time_zone: str | None = None
    tags: list[str] | None = None
    size: int | None = None
    """File size in bytes, ``None`` until generated."""
    checksum: str | None = None
    """MD5 of the generated file, ``None`` until generated."""
    url: str | None = None
    """Download URL, empty until generated. Requires the ``api_key`` header."""

    @property
    def is_finished(self) -> bool:
        """Whether the job reached a terminal status."""
        return self.status in (JobStatus.COMPLETED, JobStatus.ERROR)


class JobList(RavenPackModel):
    """Response of ``GET /jobs``."""

    count: int
    jobs: list[DatafileJob] = []


def _format_minutes(value: datetime | date | str | None) -> str | None:
    """``/jobs`` expects ``YYYY-MM-DD HH:mm``."""
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M")
    return value.strftime("%Y-%m-%d 00:00")


class JobsResource(Resource):
    """Datafile generation jobs."""

    @validated
    def list(
        self,
        *,
        start_date: datetime | date | str | None = None,
        end_date: datetime | date | str | None = None,
        status: Sequence[JobStatus] | None = None,
    ) -> JobList:
        """List your jobs (``GET /jobs``).

        Args:
            start_date: Inclusive lower bound (``YYYY-MM-DD HH:mm`` if a string).
            end_date: Inclusive upper bound (``YYYY-MM-DD HH:mm`` if a string).
            status: Filter by ``processing``, ``enqueued``, ``completed``, ``error``.
        """
        params = {
            "start_date": _format_minutes(start_date),
            "end_date": _format_minutes(end_date),
            "status": status,
        }
        return JobList.model_validate(
            self._json("GET", self._http.url("data", "jobs"), params=params)
        )

    @validated
    def get(self, token: str) -> DatafileJob:
        """Get the status of a job (``GET /jobs/{token}``).

        When ``status`` is ``completed``, ``size``, ``checksum`` and ``url`` are set.
        """
        return DatafileJob.model_validate(
            self._json("GET", self._http.url("data", f"jobs/{token}"))
        )

    @validated
    def cancel(self, token: str) -> None:
        """Cancel an ``enqueued`` job (``DELETE /jobs/{token}``).

        Raises:
            NotFoundError: Token not in queue, or job already complete.
        """
        self._http.request("DELETE", self._http.url("data", f"jobs/{token}"))

    @validated
    def wait(
        self, token: str, *, poll_interval: float = 10.0, timeout: float | None = None
    ) -> DatafileJob:
        """Poll a job until it reaches ``completed`` or ``error``.

        Args:
            token: Job token.
            poll_interval: Seconds between polls.
            timeout: Give up after this many seconds. ``None`` waits forever.

        Returns:
            The completed job.

        Raises:
            JobFailedError: The job ended with status ``error``.
            WaitTimeoutError: ``timeout`` elapsed.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            job = self.get(token)
            if job.status == JobStatus.COMPLETED:
                return job
            if job.status == JobStatus.ERROR:
                raise JobFailedError(f"Datafile job {token} failed")
            if deadline is not None and time.monotonic() + poll_interval > deadline:
                raise WaitTimeoutError(f"Datafile job {token} still {job.status} after {timeout}s")
            time.sleep(poll_interval)

    @validated
    def download(
        self, job: DatafileJob | str, destination: str | Path, *, verify_checksum: bool = True
    ) -> Path:
        """Download the file of a completed job.

        Args:
            job: Job object or token. A token triggers one ``GET /jobs/{token}``.
            destination: Target file, or existing directory (name taken from the URL).
            verify_checksum: Check the MD5 against ``job.checksum``.

        Returns:
            Path of the downloaded file.

        Raises:
            JobFailedError: Job is not completed or has no URL.
            ChecksumMismatchError: MD5 mismatch.
        """
        resolved = self.get(job) if isinstance(job, str) else job
        if resolved.status != JobStatus.COMPLETED or not resolved.url:
            raise JobFailedError(f"Job {resolved.token} is {resolved.status}, no file to download")
        return self._http.download(
            self._http.absolute_url("data", resolved.url),
            destination,
            expected_md5=resolved.checksum if verify_checksum else None,
        )


class DatafilesResource(Resource):
    """Generate CSV/Excel files for a dataset over an arbitrary date range (2000 onward)."""

    def __init__(self, http: HttpTransport, jobs: JobsResource) -> None:
        super().__init__(http)
        self._jobs = jobs

    @validated
    def generate(
        self,
        dataset_uuid: str,
        *,
        start_date: ApiDateTime,
        end_date: ApiDateTime,
        format: str = "csv",
        compressed: bool = False,
        notify: bool = False,
    ) -> DatafileToken:
        """Submit a datafile generation job (``POST /datafile/{dataset_uuid}``).

        Limits: 50M records granular, 15M aggregated, 1M in Excel, 5 years for daily datasets.

        Args:
            dataset_uuid: Dataset to export.
            start_date: Range start.
            end_date: Range end.
            format: ``csv`` or Excel.
            compressed: Compress the output.
            notify: Email notification on completion.

        Returns:
            Job token and estimated completion time.

        Raises:
            RateLimitError: Too many concurrent datafile jobs.
        """
        body = DatafileRequest(
            start_date=start_date,
            end_date=end_date,
            format=format,
            compressed=compressed,
            notify=notify,
        )
        data = self._json(
            "POST", self._http.url("data", f"datafile/{dataset_uuid}"), json=body.to_payload()
        )
        return DatafileToken.model_validate(data)

    @validated
    def count(
        self, dataset_uuid: str, *, start_date: ApiDateTime, end_date: ApiDateTime
    ) -> DatafileCount:
        """Count rows, stories and entities a datafile would contain (``POST /datafile/{dataset_uuid}/count``).

        Exact for granular datasets, an estimate for daily ones, unavailable with
        functions and conditions or daily ranges over 5 years.
        """
        body = DatafileCountRequest(start_date=start_date, end_date=end_date)
        data = self._json(
            "POST", self._http.url("data", f"datafile/{dataset_uuid}/count"), json=body.to_payload()
        )
        return DatafileCount.model_validate(data)

    @validated
    def generate_and_download(
        self,
        dataset_uuid: str,
        destination: str | Path,
        *,
        start_date: ApiDateTime,
        end_date: ApiDateTime,
        format: str = "csv",
        compressed: bool = False,
        poll_interval: float = 10.0,
        timeout: float | None = None,
        verify_checksum: bool = True,
    ) -> Path:
        """Generate a datafile, wait for completion and download it.

        Returns:
            Path of the downloaded file.

        Raises:
            JobFailedError: The job failed.
            WaitTimeoutError: ``timeout`` elapsed. The job keeps running server side.
        """
        token = self.generate(
            dataset_uuid,
            start_date=start_date,
            end_date=end_date,
            format=format,
            compressed=compressed,
        ).token
        job = self._jobs.wait(token, poll_interval=poll_interval, timeout=timeout)
        return self._jobs.download(job, destination, verify_checksum=verify_checksum)
