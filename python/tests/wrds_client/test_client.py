"""Tests for wrds_client.client.WRDSClient.

Uses a fake `wrds.Connection` throughout -- these tests never touch a real
WRDS server, and assert that credentials are passed explicitly (never left
for wrds.Connection's interactive prompt).
"""
from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from wrds_client.client import WRDSClient


class FakeConnection:
    """Records constructor args and method calls; returns canned data."""

    def __init__(self, wrds_username: str, wrds_password: str, autoconnect: bool = True):
        self.wrds_username = wrds_username
        self.wrds_password = wrds_password
        self.autoconnect = autoconnect
        self.closed = False
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def get_table(self, **kwargs: Any) -> pd.DataFrame:
        self.calls.append(("get_table", (), kwargs))
        return pd.DataFrame({"a": [1, 2]})

    def raw_sql(self, sql: str, **kwargs: Any) -> pd.DataFrame:
        self.calls.append(("raw_sql", (sql,), kwargs))
        return pd.DataFrame({"b": [3, 4]})

    def list_libraries(self) -> list[str]:
        return ["optionm", "crsp"]

    def list_tables(self, library: str) -> list[str]:
        self.calls.append(("list_tables", (), {"library": library}))
        return ["securd"]

    def describe_table(self, library: str, table: str) -> pd.DataFrame:
        self.calls.append(("describe_table", (), {"library": library, "table": table}))
        return pd.DataFrame({"name": ["secid"]})

    def close(self) -> None:
        self.closed = True


def test_from_credentials_passes_username_password_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import wrds

    monkeypatch.setattr(wrds, "Connection", FakeConnection)
    client = WRDSClient.from_credentials("jdoe", "s3cret")
    assert client.raw.wrds_username == "jdoe"
    assert client.raw.wrds_password == "s3cret"
    assert client.raw.autoconnect is True


def test_from_env_reads_dotenv_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    import wrds

    monkeypatch.setattr(wrds, "Connection", FakeConnection)
    monkeypatch.setenv("WRDS_USERNAME", "envuser")
    monkeypatch.setenv("WRDS_PASSWORD", "envpass")
    client = WRDSClient.from_env()
    assert client.raw.wrds_username == "envuser"
    assert client.raw.wrds_password == "envpass"


def test_from_env_missing_credentials_raises_without_connecting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import wrds

    monkeypatch.setattr(wrds, "Connection", FakeConnection)
    monkeypatch.delenv("WRDS_USERNAME", raising=False)
    monkeypatch.delenv("WRDS_PASSWORD", raising=False)
    with pytest.raises(RuntimeError, match="WRDS_USERNAME"):
        WRDSClient.from_env()


def test_get_table_forwards_kwargs() -> None:
    client = WRDSClient(connection=FakeConnection("u", "p"))
    df = client.get_table("optionm", "securd", columns=["secid"], obs=10, offset=5)
    assert list(df["a"]) == [1, 2]
    call = client.raw.calls[0]
    assert call == (
        "get_table",
        (),
        {
            "library": "optionm",
            "table": "securd",
            "obs": 10,
            "offset": 5,
            "columns": ["secid"],
            "date_cols": None,
        },
    )


def test_raw_sql_forwards_sql_and_kwargs() -> None:
    client = WRDSClient(connection=FakeConnection("u", "p"))
    df = client.raw_sql("select * from optionm.securd", date_cols=["date"])
    assert list(df["b"]) == [3, 4]
    call = client.raw.calls[0]
    assert call == (
        "raw_sql",
        ("select * from optionm.securd",),
        {"date_cols": ["date"], "params": None},
    )


def test_list_libraries_and_tables() -> None:
    client = WRDSClient(connection=FakeConnection("u", "p"))
    assert client.list_libraries() == ["optionm", "crsp"]
    assert client.list_tables("optionm") == ["securd"]


def test_describe_table() -> None:
    client = WRDSClient(connection=FakeConnection("u", "p"))
    df = client.describe_table("optionm", "securd")
    assert list(df["name"]) == ["secid"]


def test_context_manager_closes_connection() -> None:
    fake = FakeConnection("u", "p")
    with WRDSClient(connection=fake) as client:
        assert client.raw is fake
    assert fake.closed is True


def test_close_is_idempotent_call() -> None:
    fake = FakeConnection("u", "p")
    client = WRDSClient(connection=fake)
    client.close()
    assert fake.closed is True
