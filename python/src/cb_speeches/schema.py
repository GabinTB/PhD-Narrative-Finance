"""Column schemas of the cb_speeches artifacts (speeches, NER)."""
from __future__ import annotations

import pyarrow as pa

# BIS CSV columns (identical to gingado's ``load_CB_speeches`` frame).
RAW_COLUMNS: tuple[str, ...] = ("url", "title", "description", "date", "text", "author")

# Fields whose content defines a speech version (content_sha256), in this order.
CONTENT_FIELDS: tuple[str, ...] = ("url", "title", "description", "date", "author", "text")

CHANGES: tuple[str, ...] = ("base", "new", "revised", "removed")

SPEECH_SCHEMA = pa.schema([
    ("speech_id", pa.string()),            # BIS review code from the URL, e.g. r970211c
    ("date", pa.date32()),                 # speech date (partition key)
    ("url", pa.string()),
    ("title", pa.string()),
    ("description", pa.string()),
    ("author", pa.string()),
    ("text", pa.string()),
    ("content_sha256", pa.string()),
    ("source_zip", pa.string()),           # speeches-YYYY.zip the row was read from
    ("vintage", pa.timestamp("us", tz="UTC")),   # the execution (fetch) that recorded it
    ("change", pa.string()),               # base | new | revised | removed
])

# Columns enough to resolve the current state (no text).
STATE_COLUMNS: tuple[str, ...] = ("speech_id", "content_sha256", "source_zip", "vintage",
                                  "change")

NER_SCHEMA = pa.schema([
    ("speech_id", pa.string()),
    ("input_sha256", pa.string()),         # sha256 of the prompt inputs (date/author/title/desc)
    ("date", pa.date32()),
    ("author", pa.string()),
    ("organization", pa.string()),
    ("country_code", pa.string()),         # ISO 3166-1 alpha-2
    ("sentiment", pa.string()),            # hawkish | dovish | neutral
    ("provider", pa.string()),
    ("response_model", pa.string()),
    ("system_fingerprint", pa.string()),
    ("finish_reason", pa.string()),
    ("prompt_tokens", pa.int32()),
    ("completion_tokens", pa.int32()),
    ("attempts", pa.int8()),
    ("error", pa.string()),                # set exactly when the fields are null
    ("vintage", pa.timestamp("us", tz="UTC")),
])

# Parquet key-value metadata key holding the raw-zip manifest of a speeches file.
SOURCES_METADATA_KEY = b"cb_speeches.sources"
