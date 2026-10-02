"""RavenPack Annotations 1.0: every raw field, its type and its level.

A raw CSV row is one DETECTION: a (story, entity, event) triple, unique on
(RP_STORY_ID, RP_ENTITY_ID, RP_STORY_EVENT_INDEX). Each field is constant at one
of three levels (checked on 200k rows of January every three years, 2000-2025):

  story      constant within a story            -> one row per story
  entity     constant within a story x entity   -> one row per story x entity
  detection  varies by raw row (the event)      -> aligned lists on the story x entity row

The ingestion checks these levels on every row it reads and refuses a month that
breaks one (``ingest.LevelError``): a field changing level is an owner decision,
never a silent "first value". Any raw column not in this registry, or missing
from it, refuses the month too.

Types: integers are parsed as numbers and must be whole and within range;
timestamps are UTC, parsed to microseconds; empty strings and pandas' default NA
strings ("NA", "None", "nan", ...: ``ingest.NULL_STRINGS``) are nulls.
"""
from __future__ import annotations

from dataclasses import dataclass

import polars as pl

STORY, ENTITY, DETECTION = "story", "entity", "detection"
LEVELS = (STORY, ENTITY, DETECTION)

STORY_KEY = "RP_STORY_ID"
ENTITY_KEY = "RP_ENTITY_ID"
EVENT_INDEX = "RP_STORY_EVENT_INDEX"
TIMESTAMP = "TIMESTAMP_UTC"
DATETIME = pl.Datetime("us", "UTC")
RAW_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S%.f"


@dataclass(frozen=True)
class Field:
    name: str
    dtype: pl.DataType
    level: str

    def __post_init__(self) -> None:
        if self.level not in LEVELS and self.name not in (STORY_KEY, ENTITY_KEY):
            raise ValueError(f"{self.name}: unknown level {self.level!r}")


def _f(name: str, dtype: pl.DataType, level: str) -> Field:
    return Field(name, dtype, level)


S, E, D = STORY, ENTITY, DETECTION
STR, F32, I8, U8, I32 = pl.String(), pl.Float32(), pl.Int8(), pl.UInt8(), pl.Int32()

# Raw column order (the vendor's header, 50 columns).
RPA1_FIELDS: tuple[Field, ...] = (
    _f("TIMESTAMP_UTC", DATETIME, S),
    _f("RP_STORY_ID", STR, "key"),
    _f("RP_ENTITY_ID", STR, "key"),
    _f("ENTITY_TYPE", STR, E),
    _f("ENTITY_NAME", STR, E),
    _f("COUNTRY_CODE", STR, E),
    _f("RELEVANCE", U8, E),
    _f("EVENT_SENTIMENT_SCORE", F32, D),
    _f("EVENT_RELEVANCE", U8, D),
    _f("EVENT_SIMILARITY_KEY", STR, D),
    _f("EVENT_SIMILARITY_DAYS", F32, D),
    _f("TOPIC", STR, D),
    _f("GROUP", STR, D),
    _f("TYPE", STR, D),
    _f("SUB_TYPE", STR, D),
    _f("PROPERTY", STR, D),
    _f("FACT_LEVEL", STR, D),
    _f("RP_POSITION_ID", STR, D),
    _f("POSITION_NAME", STR, D),
    _f("EVALUATION_METHOD", STR, D),
    _f("MATURITY", STR, D),
    _f("EARNINGS_TYPE", STR, D),
    _f("EVENT_START_DATE_UTC", DATETIME, D),
    _f("EVENT_END_DATE_UTC", DATETIME, D),
    _f("REPORTING_PERIOD", STR, D),
    _f("REPORTING_START_DATE_UTC", DATETIME, D),
    _f("REPORTING_END_DATE_UTC", DATETIME, D),
    _f("RELATED_ENTITY", STR, D),
    _f("RELATIONSHIP", STR, D),
    _f("CATEGORY", STR, D),
    _f("EVENT_TEXT", STR, D),
    _f("NEWS_TYPE", STR, S),
    _f("RP_SOURCE_ID", STR, S),
    _f("SOURCE_NAME", STR, S),
    _f("CSS", F32, S),
    _f("NIP", F32, S),
    _f("PEQ", I8, S),
    _f("BEE", I8, S),
    _f("BMQ", I8, S),
    _f("BAM", I8, S),
    _f("BCA", I8, S),
    _f("BER", I8, S),
    _f("ANL_CHG", I8, E),
    _f("MCQ", I8, E),
    _f("RP_STORY_EVENT_INDEX", I32, D),
    _f("RP_STORY_EVENT_COUNT", I32, S),
    _f("PRODUCT_KEY", STR, S),
    _f("PROVIDER_ID", STR, S),
    _f("PROVIDER_STORY_ID", STR, S),
    _f("HEADLINE", STR, S),
)


@dataclass(frozen=True)
class FieldSet:
    """A product's raw fields and the two output schemas derived from them."""

    fields: tuple[Field, ...]

    def __post_init__(self) -> None:
        names = [f.name for f in self.fields]
        if len(set(names)) != len(names):
            raise ValueError("duplicate field names")
        for key in (STORY_KEY, ENTITY_KEY, EVENT_INDEX, TIMESTAMP):
            if key not in names:
                raise ValueError(f"field set lacks {key}")
        if self.by_name[TIMESTAMP].level != STORY:
            raise ValueError(f"{TIMESTAMP} must be a story field")

    @property
    def by_name(self) -> dict[str, Field]:
        return {f.name: f for f in self.fields}

    @property
    def raw_columns(self) -> list[str]:
        return [f.name for f in self.fields]

    def level(self, level: str) -> list[str]:
        return [f.name for f in self.fields if f.level == level]

    @property
    def story_fields(self) -> list[str]:
        return self.level(STORY)

    @property
    def entity_fields(self) -> list[str]:
        return self.level(ENTITY)

    @property
    def detection_fields(self) -> list[str]:
        return self.level(DETECTION)

    @property
    def headlines_schema(self) -> pl.Schema:
        """One row per story: the key, the story fields, the reconciliation counts."""
        by = self.by_name
        cols: dict[str, pl.DataType] = {STORY_KEY: pl.String()}
        cols.update({c: by[c].dtype for c in self.story_fields})
        cols.update({"N_ENTITIES": pl.Int32(), "N_DETECTIONS": pl.Int32()})
        return pl.Schema(cols)

    @property
    def entities_schema(self) -> pl.Schema:
        """One row per story x entity: keys, timestamp, entity fields, detection lists."""
        by = self.by_name
        cols: dict[str, pl.DataType] = {STORY_KEY: pl.String(), ENTITY_KEY: pl.String(),
                                        TIMESTAMP: by[TIMESTAMP].dtype}
        cols.update({c: by[c].dtype for c in self.entity_fields})
        cols.update({c: pl.List(by[c].dtype) for c in self.detection_fields})
        return pl.Schema(cols)


RPA1 = FieldSet(RPA1_FIELDS)

__all__ = ["Field", "FieldSet", "RPA1", "RPA1_FIELDS", "STORY", "ENTITY", "DETECTION",
           "STORY_KEY", "ENTITY_KEY", "EVENT_INDEX", "TIMESTAMP", "DATETIME",
           "RAW_DATETIME_FORMAT"]
