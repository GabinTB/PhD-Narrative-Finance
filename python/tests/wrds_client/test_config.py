"""Tests for wrds_client.config: WRDS credential resolution from the environment."""
from __future__ import annotations

import pytest

from wrds_client.config import WRDSCredentials, credentials_from_env


def test_credentials_from_env_reads_both_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WRDS_USERNAME", "jdoe")
    monkeypatch.setenv("WRDS_PASSWORD", "s3cret")
    creds = credentials_from_env()
    assert creds == WRDSCredentials(username="jdoe", password="s3cret")


def test_credentials_from_env_missing_username_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WRDS_USERNAME", raising=False)
    monkeypatch.setenv("WRDS_PASSWORD", "s3cret")
    with pytest.raises(RuntimeError, match="WRDS_USERNAME"):
        credentials_from_env()


def test_credentials_from_env_missing_password_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WRDS_USERNAME", "jdoe")
    monkeypatch.delenv("WRDS_PASSWORD", raising=False)
    with pytest.raises(RuntimeError, match="WRDS_PASSWORD"):
        credentials_from_env()
