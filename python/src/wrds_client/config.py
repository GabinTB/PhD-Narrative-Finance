"""Environment-based WRDS credentials.

Loaded from `.env` (via python-dotenv) so `WRDSClient.from_env()` can
authenticate by passing credentials explicitly to `wrds.Connection`, which
never triggers that library's interactive username/password prompt.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

ENV_USERNAME = "WRDS_USERNAME"
ENV_PASSWORD = "WRDS_PASSWORD"


@dataclass(frozen=True)
class WRDSCredentials:
    """WRDS login, resolved from the environment."""

    username: str
    password: str


def credentials_from_env() -> WRDSCredentials:
    """Read `WRDS_USERNAME`/`WRDS_PASSWORD` from the environment.

    Returns:
        WRDSCredentials: the resolved username and password.

    Raises:
        RuntimeError: naming the specific missing variable, rather than
            letting `wrds.Connection` fall back to its interactive prompt
            deep inside a batch job.
    """
    username = os.environ.get(ENV_USERNAME)
    if not username:
        raise RuntimeError(
            f"{ENV_USERNAME} must be set (in .env or environment) to connect to WRDS."
        )
    password = os.environ.get(ENV_PASSWORD)
    if not password:
        raise RuntimeError(
            f"{ENV_PASSWORD} must be set (in .env or environment) to connect to WRDS."
        )
    return WRDSCredentials(username=username, password=password)
