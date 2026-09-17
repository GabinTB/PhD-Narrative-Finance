"""Register a universe master list as a versioned datalake artifact.

Registered into the `raw` layer, not `derived`: this is vintaged reference
input data supplied by the user (an ISIN/CIK/ticker/name master list), not
something computed by a pipeline. `raw` is defined in `datalake.index.
VALID_LAYERS` but, before this, no module in this repo actually registered
anything into it.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence

    from datalake import Artifact, DatalakeIndex

from universe.schema import (
    UniverseEntry,
    find_incomplete,
    from_frame,
    load_universe_file,
    to_frame,
)

KIND = "universe"


def register_universe_entries(
    index: DatalakeIndex,
    entries: Sequence[UniverseEntry],
    *,
    source_label: str,
    name: str | None = None,
    allow_incomplete: bool = False,
    pipeline: str,
    pipeline_version: str,
    notes: str = "",
    pipeline_repo: str | None = None,
) -> Artifact:
    """Register already-loaded (and, typically, already-enriched) `entries` as
    a `layer="raw"` universe artifact.

    Unless `allow_incomplete=True`, refuses to register (raises `ValueError`
    listing the offending rows/fields) if any entry is still missing one of
    `universe.schema.MANDATORY_AFTER_ENRICHMENT` -- these are meant to be
    filled in by `universe.enrich.enrich_universe` before calling this, not
    silently registered as gaps.

    `name` is an optional human-readable tag for this universe list (e.g.
    "sp500_2020") -- unrelated to `UniverseEntry.name` (a single security's
    company name). Passing it makes this registration findable later via
    `load_universe_by_name`/`load_universe_entries_by_name`, and it becomes
    part of the artifact_id (hyperparams flow into `datalake`'s identity
    slug). Names are not enforced unique: registering the same name again
    creates a new, independent artifact, and lookups return the most recent
    one -- the same "latest wins" semantics `index.latest(kind)` already has,
    one level down.
    """
    if not entries:
        raise ValueError(f"{source_label} contains no usable universe entries")

    if not allow_incomplete:
        incomplete = find_incomplete(entries)
        if incomplete:
            detail = "; ".join(f"row {i}: missing {fields}" for i, fields in incomplete)
            raise ValueError(
                f"{len(incomplete)} entr{'y' if len(incomplete) == 1 else 'ies'} still "
                f"missing mandatory-after-enrichment fields -- {detail}. Pass "
                "allow_incomplete=True to register anyway."
            )

    hyperparams: dict[str, object] = {"source_path": source_label, "n_entries": len(entries)}
    if name:
        hyperparams["name"] = name

    with index.run(
        kind=KIND,
        pipeline=pipeline,
        pipeline_version=pipeline_version,
        pipeline_repo=pipeline_repo,
        hyperparams=hyperparams,
        notes=notes,
        layer="raw",
        hash_pattern="*.parquet",
    ) as run:
        to_frame(entries).to_parquet(run.out_dir / "data.parquet", index=False)
        run.note(f"{len(entries)} universe entries")

    return index.get(run.artifact_id)


def ingest_universe(
    index: DatalakeIndex,
    path: Path,
    *,
    name: str | None = None,
    allow_incomplete: bool = False,
    pipeline: str,
    pipeline_version: str,
    notes: str = "",
    pipeline_repo: str | None = None,
) -> Artifact:
    """Load `path` (`.csv` or `.parquet`) and register it, unenriched.

    Thin wrapper over `register_universe_entries`; `allow_incomplete` defaults
    to `False` here too, so a raw file missing e.g. `gvkey`/`dbga_secid` still
    needs `allow_incomplete=True` (or a prior `universe.enrich.enrich_universe`
    pass) to register.
    """
    entries = load_universe_file(Path(path))
    return register_universe_entries(
        index,
        entries,
        source_label=str(path),
        name=name,
        allow_incomplete=allow_incomplete,
        pipeline=pipeline,
        pipeline_version=pipeline_version,
        notes=notes,
        pipeline_repo=pipeline_repo,
    )


def load_universe(index: DatalakeIndex, artifact_id: str | None = None) -> pd.DataFrame:
    """A registered universe artifact's entries, as a DataFrame.

    `artifact_id=None` (the default) reads the most recently registered one.
    """
    artifact = index.get(artifact_id) if artifact_id else index.latest(KIND)
    files = artifact.files("*.parquet")
    return pd.read_parquet(files[0])


def load_universe_entries(
    index: DatalakeIndex, artifact_id: str | None = None
) -> list[UniverseEntry]:
    """A registered universe artifact's entries, as `UniverseEntry`s.

    `artifact_id=None` (the default) reads the most recently registered one.
    """
    return from_frame(load_universe(index, artifact_id))


def find_universe_by_name(index: DatalakeIndex, name: str) -> Artifact:
    """The most recently registered universe `Artifact` tagged with `name`.

    Raises `ValueError` naming `name` if no registered universe artifact has
    a matching `name` hyperparam. `index.list` already orders newest-first,
    so the first match found is the most recent.
    """
    for artifact in index.list(KIND, include_partial=False):
        if artifact.meta.hyperparams.get("name") == name:
            return artifact
    raise ValueError(f"no universe artifact registered with name={name!r}")


def load_universe_by_name(index: DatalakeIndex, name: str) -> pd.DataFrame:
    """The most recently registered universe artifact tagged with `name`, as a DataFrame."""
    artifact = find_universe_by_name(index, name)
    files = artifact.files("*.parquet")
    return pd.read_parquet(files[0])


def load_universe_entries_by_name(index: DatalakeIndex, name: str) -> list[UniverseEntry]:
    """The most recently registered universe artifact tagged with `name`, as `UniverseEntry`s."""
    return from_frame(load_universe_by_name(index, name))
