"""Generic WRDS connector: env-authenticated connection plus a datalake-artifact
fetch helper.

    from wrds_client import WRDSClient

    with WRDSClient.from_env() as db:
        df = db.raw_sql("select * from optionm.securd limit 10")

Credentials come from `.env` (WRDS_USERNAME / WRDS_PASSWORD, see
`wrds_client.config`) so `wrds.Connection`'s interactive prompt never fires.

Per-source schemas and fetch logic (OptionMetrics, TAQ, ...) live in
submodules that build on `WRDSClient`, not here.
"""
from __future__ import annotations

from wrds_client.client import WRDSClient
from wrds_client.config import WRDSCredentials, credentials_from_env
from wrds_client.ingest import fetch_query_artifact, fetch_table_artifact

__all__ = [
    "WRDSClient",
    "WRDSCredentials",
    "credentials_from_env",
    "fetch_query_artifact",
    "fetch_table_artifact",
]
