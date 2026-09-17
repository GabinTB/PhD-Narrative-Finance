"""Canonical security-identifier universe, shared across data-source connectors.

    from universe import UniverseEntry, ingest_universe, load_universe_entries

    with DatalakeIndex(root) as index:
        artifact = ingest_universe(index, "universe.csv", pipeline=..., pipeline_version=...)
        entries = load_universe_entries(index)  # or (index, artifact.artifact_id)

Each connector (e.g. `wrds_client.linking`, `deutsche_boerse.identifiers`) resolves
`UniverseEntry`s to that source's own native identifier -- this package only defines
the shared record and its round-trip to/from a versioned datalake artifact.
"""
from __future__ import annotations

from universe.ingest import (
    find_universe_by_name,
    ingest_universe,
    load_universe,
    load_universe_by_name,
    load_universe_entries,
    load_universe_entries_by_name,
    register_universe_entries,
)
from universe.schema import (
    MANDATORY_AFTER_ENRICHMENT,
    UniverseEntry,
    find_incomplete,
    from_frame,
    load_universe_csv,
    load_universe_file,
    load_universe_parquet,
    to_frame,
)

__all__ = [
    "MANDATORY_AFTER_ENRICHMENT",
    "UniverseEntry",
    "find_incomplete",
    "find_universe_by_name",
    "from_frame",
    "ingest_universe",
    "load_universe",
    "load_universe_by_name",
    "load_universe_csv",
    "load_universe_entries",
    "load_universe_entries_by_name",
    "load_universe_file",
    "load_universe_parquet",
    "register_universe_entries",
    "to_frame",
]
