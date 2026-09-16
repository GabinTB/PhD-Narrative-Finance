"""Artifact-adapted WRDS fetch: run a query, register the result as a
versioned datalake artifact.

Mirrors `ravenpack.headlines.ingest`'s use of `index.run(...)`: the caller
supplies a connected `WRDSClient` and a query (raw SQL or library/table), the
result is written as a single parquet file, and `DatalakeIndex` handles
hashing, sidecars, and lineage.

No pagination/chunking here -- this is the generic layer.  Per-source
submodules that know a table's real size can chunk before calling
`fetch_query_artifact`, or write their own artifact directly.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from datalake import Artifact, DatalakeIndex
    from wrds_client.client import WRDSClient


def fetch_query_artifact(
    index: DatalakeIndex,
    client: WRDSClient,
    sql: str,
    *,
    kind: str,
    pipeline: str,
    pipeline_version: str,
    hyperparams: dict[str, Any] | None = None,
    sources: Sequence[Artifact | str] | None = None,
    notes: str = "",
    pipeline_repo: str | None = None,
    repo_dir: Path | None = None,
    date_cols: list[str] | None = None,
) -> Artifact:
    """Run `sql` against WRDS and register the result as one artifact.

    `sql` is recorded as a hyperparam (merged with any caller-supplied
    `hyperparams`) so the artifact's identity and provenance capture exactly
    what was queried.

    Args:
        index:             Datalake index to register the artifact in.
        client:             Connected WRDSClient to query.
        sql:                Query to run via `client.raw_sql`.
        kind:               Artifact kind, e.g. "optionm_securd".
        pipeline:           Producing repo name, recorded in provenance.
        pipeline_version:   Semantic version of this pipeline.
        hyperparams:        Extra identity params (merged with `sql`).
        sources:            Input Artifacts (or IDs) for lineage, if any.
        notes:              Free text recorded on this execution's record.
        pipeline_repo:      URL of the producing repo.
        repo_dir:           Directory to read the git SHA from (default cwd).
        date_cols:          Columns to parse as dates (passed to raw_sql).

    Returns:
        The completed Artifact.
    """
    run_hyperparams = {"sql": sql, **(hyperparams or {})}

    with index.run(
        kind=kind,
        pipeline=pipeline,
        pipeline_version=pipeline_version,
        pipeline_repo=pipeline_repo,
        repo_dir=repo_dir,
        hyperparams=run_hyperparams,
        sources=sources,
        notes=notes,
        hash_pattern="*.parquet",
    ) as run:
        df = client.raw_sql(sql, date_cols=date_cols)
        df.to_parquet(run.out_dir / "data.parquet", index=False)
        run.note(f"{len(df)} rows fetched")

    return index.get(run.artifact_id)


def fetch_table_artifact(
    index: DatalakeIndex,
    client: WRDSClient,
    library: str,
    table: str,
    *,
    kind: str,
    pipeline: str,
    pipeline_version: str,
    columns: list[str] | None = None,
    obs: int = -1,
    offset: int = 0,
    hyperparams: dict[str, Any] | None = None,
    sources: Sequence[Artifact | str] | None = None,
    notes: str = "",
    pipeline_repo: str | None = None,
    repo_dir: Path | None = None,
    date_cols: list[str] | None = None,
) -> Artifact:
    """Fetch a whole (or row-limited) WRDS table and register it as one artifact.

    Convenience wrapper over `fetch_query_artifact`'s pattern for the common
    library/table case (`client.get_table`) instead of hand-written SQL.
    `library`, `table`, `columns`, `obs`, and `offset` are recorded as
    hyperparams for reproducible artifact identity.

    Returns:
        The completed Artifact.
    """
    run_hyperparams = {
        "library": library,
        "table": table,
        "columns": columns,
        "obs": obs,
        "offset": offset,
        **(hyperparams or {}),
    }

    with index.run(
        kind=kind,
        pipeline=pipeline,
        pipeline_version=pipeline_version,
        pipeline_repo=pipeline_repo,
        repo_dir=repo_dir,
        hyperparams=run_hyperparams,
        sources=sources,
        notes=notes,
        hash_pattern="*.parquet",
    ) as run:
        df = client.get_table(
            library, table, columns=columns, obs=obs, offset=offset, date_cols=date_cols
        )
        df.to_parquet(run.out_dir / "data.parquet", index=False)
        run.note(f"{len(df)} rows fetched")

    return index.get(run.artifact_id)
