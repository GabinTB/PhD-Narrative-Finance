"""Enumerations of values documented by the RavenPack API."""

from __future__ import annotations

from enum import StrEnum

__all__ = [
    "DatasetScope",
    "FileStatus",
    "Frequency",
    "JobStatus",
    "ReferenceEntityType",
    "ReferenceFileType",
]


class Frequency(StrEnum):
    """Dataset frequency: raw granular records or daily aggregates."""

    GRANULAR = "granular"
    DAILY = "daily"


class DatasetScope(StrEnum):
    """Scope filter for listing datasets."""

    PUBLIC = "public"
    PRIVATE = "private"
    SHARED = "shared"


class JobStatus(StrEnum):
    """Status of a datafile generation job."""

    PROCESSING = "processing"
    ENQUEUED = "enqueued"
    COMPLETED = "completed"
    ERROR = "error"


class ReferenceEntityType(StrEnum):
    """Entity types (and taxonomies) available as entity reference files."""

    CMDT = "CMDT"
    COMP = "COMP"
    CURR = "CURR"
    NATL = "NATL"
    ORGA = "ORGA"
    ORGT = "ORGT"
    PEOP = "PEOP"
    PLCE = "PLCE"
    POSI = "POSI"
    PROD = "PROD"
    PRDT = "PRDT"
    SRCE = "SRCE"
    TEAM = "TEAM"
    SECT = "SECT"
    OTHR = "OTHR"
    JOBS_TAXONOMY = "jobs-taxonomy"
    OCCUPATIONS_TAXONOMY = "occupations-taxonomy"


class ReferenceFileType(StrEnum):
    """Full reference file or daily delta (delta unavailable for taxonomies)."""

    FULL = "full"
    DELTA = "delta"


class FileStatus(StrEnum):
    """Processing status of a file uploaded to RavenPack Annotations."""

    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
