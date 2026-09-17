"""Configuration: base URLs and API key resolution from the environment / ``.env``."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from .exceptions import ConfigurationError

__all__ = [
    "ANNOTATIONS_URL",
    "API_KEY_ENV_VAR",
    "DATA_URL",
    "STREAMING_URL",
    "resolve_api_key",
]

STREAMING_URL = "https://feed-edge.ravenpack.com/1.0"
DATA_URL = "https://api-edge.ravenpack.com/1.0"
ANNOTATIONS_URL = "https://upload.ravenpack.com/1.0"

API_KEY_ENV_VAR = "RAVENPACK_API_KEY"


def resolve_api_key(
    api_key: str | None = None,
    *,
    env_file: str | os.PathLike[str] | None = None,
    env_var: str = API_KEY_ENV_VAR,
) -> str:
    """Resolve the RavenPack API key.

    Resolution order: explicit ``api_key`` argument, then the ``env_var``
    environment variable. Before reading the environment, the ``.env`` file is
    loaded without overriding variables that are already set. If ``env_file``
    is not given, the nearest ``.env`` walking up from the current working
    directory is used.

    Args:
        api_key: Explicit key. Takes precedence over everything else.
        env_file: Path to a ``.env`` file.
        env_var: Name of the environment variable holding the key.

    Returns:
        The API key.

    Raises:
        ConfigurationError: If no key can be found, or ``env_file`` does not exist.
    """
    if api_key:
        return api_key

    if env_file is not None:
        path = Path(env_file)
        if not path.is_file():
            raise ConfigurationError(f".env file not found: {path}")
        load_dotenv(path, override=False)
    else:
        found = find_dotenv(usecwd=True)
        if found:
            load_dotenv(found, override=False)

    key = os.environ.get(env_var, "").strip()
    if not key:
        raise ConfigurationError(
            f"No RavenPack API key found. Pass api_key=... or set {env_var} in the environment or .env."
        )
    return key
