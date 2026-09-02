"""Deutsche Boerse data collection and analytics.

Paths are resolved lazily so importing this package never fails due to
missing environment variables.  Call `init()` explicitly in scripts before
using any dbg_cdm functionality.

Environment variables (loaded from .env automatically):
    RAW_DATA_PATH   root of raw data tree  (e.g. /mnt/storage/gdrive/Datalake/raw)
    CACHE_PATH      root of local cache    (e.g. /mnt/storage/cache)

Raw data lands at:
    $RAW_DATA_PATH/Deutsche_Boerse/hpt/
    $RAW_DATA_PATH/Deutsche_Boerse/hpt_all/
    $RAW_DATA_PATH/Deutsche_Boerse/microprice/
    $RAW_DATA_PATH/Deutsche_Boerse/orderbook/

Cache lands at:
    $CACHE_PATH/Deutsche_Boerse/trades/
    $CACHE_PATH/Deutsche_Boerse/events/
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

_RAW_ROOT: Path | None = None
_CACHE_ROOT: Path | None = None
_INITIALISED = False


def _get_raw_root() -> Path:
    global _RAW_ROOT
    if _RAW_ROOT is None:
        value = os.environ.get("RAW_DATA_PATH")
        if not value:
            raise RuntimeError(
                "RAW_DATA_PATH is not set. Add it to your .env file."
            )
        _RAW_ROOT = Path(value) / "Deutsche_Boerse"
    return _RAW_ROOT


def _get_cache_root() -> Path:
    global _CACHE_ROOT
    if _CACHE_ROOT is None:
        value = os.environ.get("CACHE_PATH")
        if not value:
            raise RuntimeError(
                "CACHE_PATH is not set. Add it to your .env file."
            )
        _CACHE_ROOT = Path(value) / "Deutsche_Boerse"
    return _CACHE_ROOT


class _LazyPath:
    """Path proxy resolved on first use."""

    def __init__(self, resolver):
        object.__setattr__(self, "_resolver", resolver)
        object.__setattr__(self, "_resolved", None)

    def _resolve(self) -> Path:
        if object.__getattribute__(self, "_resolved") is None:
            object.__setattr__(
                self, "_resolved",
                object.__getattribute__(self, "_resolver")()
            )
        return object.__getattribute__(self, "_resolved")

    def __truediv__(self, other):
        return self._resolve() / other

    def __str__(self):
        return str(self._resolve())

    def __repr__(self):
        return repr(self._resolve())

    def __fspath__(self):
        return os.fspath(self._resolve())

    def __getattr__(self, name):
        return getattr(self._resolve(), name)


RAW_ROOT: Path = _LazyPath(_get_raw_root)       # type: ignore[assignment]
CACHE_ROOT: Path = _LazyPath(_get_cache_root)   # type: ignore[assignment]


def init() -> None:
    """Initialise directories and dbg-cdm DATA_BASE_DIR.

    Must be called in scripts before using any dbg_cdm submodule.
    Safe to call multiple times.
    """
    global _INITIALISED
    if _INITIALISED:
        return

    raw_root = _get_raw_root()
    cache_root = _get_cache_root()

    from dbg_cdm.environment_utils import set_data_base_dir
    set_data_base_dir(raw_root)

    for d in ("hpt", "hpt_all", "microprice", "orderbook"):
        (raw_root / d).mkdir(parents=True, exist_ok=True)
    for d in ("trades", "events"):
        (cache_root / d).mkdir(parents=True, exist_ok=True)

    _INITIALISED = True


__all__ = ["RAW_ROOT", "CACHE_ROOT", "init"]
