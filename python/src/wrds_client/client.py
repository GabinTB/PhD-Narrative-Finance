"""Thin, env-authenticated wrapper around `wrds.Connection`.

Generic connector only: this module knows nothing about any particular WRDS
data source's schema.  Per-source submodules (`wrds_client.option_metrics`,
etc.) build on `WRDSClient` to fetch and interpret specific tables.

Credentials always come from `WRDSClient.from_env()` (backed by
`wrds_client.config.credentials_from_env`) or are passed explicitly -- never
left for `wrds.Connection` to prompt for interactively, which would hang a
batch job waiting on stdin.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

from wrds_client.config import credentials_from_env


class WRDSClient:
    """Wraps a `wrds.Connection`, exposing the query surface we use.

    Construct via `from_env()` (reads `WRDS_USERNAME`/`WRDS_PASSWORD` from
    `.env`) or `from_credentials()` (explicit username/password). The
    `connection` param on `__init__` exists for tests to inject a fake
    connection without touching a real WRDS server or `.env`.
    """

    def __init__(self, *, connection: Any) -> None:
        self._conn = connection

    @classmethod
    def from_env(cls, *, autoconnect: bool = True) -> WRDSClient:
        """Build a client authenticated from `.env`'s WRDS_USERNAME/WRDS_PASSWORD."""
        creds = credentials_from_env()
        return cls.from_credentials(
            creds.username, creds.password, autoconnect=autoconnect
        )

    @classmethod
    def from_credentials(
        cls, username: str, password: str, *, autoconnect: bool = True
    ) -> WRDSClient:
        """Build a client from an explicit username/password."""
        import wrds

        connection = wrds.Connection(
            wrds_username=username,
            wrds_password=password,
            autoconnect=autoconnect,
        )
        return cls(connection=connection)

    @property
    def raw(self) -> Any:
        """The underlying `wrds.Connection`, for anything not wrapped here."""
        return self._conn

    def get_table(
        self,
        library: str,
        table: str,
        *,
        obs: int = -1,
        offset: int = 0,
        columns: list[str] | None = None,
        date_cols: list[str] | None = None,
    ) -> pd.DataFrame:
        """Fetch a whole (or row-limited) WRDS table as a DataFrame."""
        return self._conn.get_table(
            library=library,
            table=table,
            obs=obs,
            offset=offset,
            columns=columns,
            date_cols=date_cols,
        )

    def raw_sql(
        self,
        sql: str,
        *,
        date_cols: list[str] | None = None,
        params: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Run arbitrary SQL against WRDS's Postgres and return a DataFrame."""
        return self._conn.raw_sql(sql, date_cols=date_cols, params=params)

    def list_libraries(self) -> list[str]:
        return self._conn.list_libraries()

    def list_tables(self, library: str) -> list[str]:
        return self._conn.list_tables(library=library)

    def describe_table(self, library: str, table: str) -> pd.DataFrame:
        return self._conn.describe_table(library=library, table=table)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> WRDSClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
