"""Config schema, resolution, canonicalisation, and hashing.

The config is the sole specification of what `Store.get()` produces; its hash
is the dataset identity. There is no second source of truth -- callers pass a
config (path or dict), it gets resolved (concrete dates, expanded CIKs) and
canonicalised (non-semantic keys dropped, order made stable), and *that* is
what gets hashed and stored alongside every exported row.

See .claude/skills/sec-filing-db-coding/reference/config-and-hashing.md.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1"

#: A config in its plain-dict form, before or after resolution/canonicalisation.
ConfigDict = dict[str, Any]

#: Keys dropped before hashing. Changing a contact email or worker count must
#: not change the dataset identity.
_RUNTIME_KEY = "runtime"


class ConfigError(ValueError):
    """A config is malformed or ambiguous in a way that must not be guessed past."""


@dataclass(frozen=True)
class ResolvedConfig:
    """A resolved, canonicalisation-ready config plus its identity hash.

    `raw` is the resolved dict (concrete dates, expanded CIKs) -- this is what
    gets written into `extracted/{config_hash}/config.json`, not the user's
    original input.
    """

    raw: ConfigDict
    config_hash: str


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def load_config(config: str | Path | ConfigDict) -> ConfigDict:
    """Accept a config as a path or a dict; return a plain dict.

    A file is a dict with a location; canonicalise to the dict internally and
    hash that -- never accept loose kwargs alongside a config that already
    carries the same field, or the two will diverge.
    """
    if isinstance(config, dict):
        return deepcopy(config)
    path = Path(config)
    with path.open("r", encoding="utf-8") as f:
        loaded: Any = json.load(f)
    if not isinstance(loaded, dict):
        raise ConfigError(f"{path} does not contain a JSON object at the top level")
    return loaded


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def _validate_items_to_extract(items_to_extract: Any) -> None:
    """`items_to_extract` must be form-keyed. A flat list is ambiguous the
    moment more than one form type is requested: Item 7 is MD&A in a 10-K,
    Part I Item 2 in a 10-Q, and `["1A", "7"]` cannot express both.
    """
    if items_to_extract is None:
        return
    if not isinstance(items_to_extract, dict):
        raise ConfigError(
            "items_to_extract must be form-keyed, e.g. "
            '{"10-K": ["1", "1A", "7"], "10-Q": ["part_1__2"]}, not a flat list. '
            "A flat list cannot disambiguate an item that means different things "
            "in different forms."
        )
    for form, items in items_to_extract.items():
        if not isinstance(form, str):
            raise ConfigError(f"items_to_extract key {form!r} must be a form string")
        if items is not None and not isinstance(items, list):
            raise ConfigError(f"items_to_extract[{form!r}] must be a list or null")


def _normalise_cik(cik: str | int) -> str:
    return str(cik).strip().zfill(10)


# --------------------------------------------------------------------------
# Resolution
# --------------------------------------------------------------------------


def resolve_config(
    config: str | Path | ConfigDict,
    *,
    today: date | None = None,
    resolve_company_names: Callable[[list[str], date], list[str]] | None = None,
) -> ConfigDict:
    """Resolve a user config into the concrete form that gets hashed and stored.

    - `filings.end_date: null` (meaning "present") resolves to a concrete date
      at call time, so the same config run a month apart does not silently
      produce different corpora under the same hash.
    - `universe.company_names` resolve to CIKs as of `universe.resolve_as_of`
      (defaulting to `today`) via `resolve_company_names`, and the expanded
      CIK list is written into `universe.ciks`.
    - CIKs are normalised to zero-padded 10-character strings and sorted.

    `today` and `resolve_company_names` are injectable so resolution is
    deterministic in tests -- do not reach for `date.today()` or a real CIK
    lookup inside this function's callers without going through here.
    """
    cfg = load_config(config)
    cfg.setdefault("schema_version", SCHEMA_VERSION)

    universe = cfg.setdefault("universe", {})
    filings = cfg.setdefault("filings", {})
    _validate_items_to_extract(cfg.get("items_to_extract"))

    resolved_today = today if today is not None else date.today()

    if filings.get("end_date") is None:
        filings["end_date"] = resolved_today.isoformat()

    resolve_as_of_raw = universe.get("resolve_as_of")
    resolve_as_of = date.fromisoformat(resolve_as_of_raw) if resolve_as_of_raw else resolved_today
    universe["resolve_as_of"] = resolve_as_of.isoformat()

    ciks = [str(c) for c in (universe.get("ciks") or [])]
    company_names = universe.get("company_names") or []
    if company_names:
        if resolve_company_names is None:
            raise ConfigError(
                "config specifies universe.company_names but no CIK resolver was "
                "provided. Real company-name -> CIK resolution lands with the "
                "EDGAR state sync (task 04+); pass resolve_company_names explicitly "
                "(a stub is fine) until then -- do not silently skip the names."
            )
        ciks.extend(resolve_company_names(company_names, resolve_as_of))

    universe["ciks"] = sorted({_normalise_cik(c) for c in ciks})

    return cfg


# --------------------------------------------------------------------------
# Canonicalisation
# --------------------------------------------------------------------------


def canonicalise(resolved_config: ConfigDict) -> ConfigDict:
    """Drop non-semantic keys and sort for stable hashing.

    Steps, in order (reference/config-and-hashing.md):
    1. Drop `runtime.*` -- user_agent, workers, root, network settings never
       affect the dataset identity.
    2. Sort all object keys recursively.
    3. Sort arrays whose order is not semantic. Every array in this schema
       (CIK lists, form lists, item lists, tool lists) is an unordered set,
       so any list of sortable primitives is sorted; this is a property of
       the schema, not a per-key allowlist that has to be kept in sync.
    """
    cfg = deepcopy(resolved_config)
    cfg.pop(_RUNTIME_KEY, None)
    result = _sort_recursive(cfg)
    assert isinstance(result, dict)
    return result


def _sort_recursive(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _sort_recursive(obj[k]) for k in sorted(obj.keys())}
    if isinstance(obj, list):
        children = [_sort_recursive(v) for v in obj]
        if children and all(isinstance(v, (str, int, float)) for v in children):
            children = sorted(children)
        return children
    return obj


# --------------------------------------------------------------------------
# Hashing
# --------------------------------------------------------------------------


def _hash_payload(canonical: ConfigDict) -> str:
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def config_hash(resolved_config: ConfigDict) -> str:
    """The full dataset identity: SHA-256 of the canonicalised resolved config.

    Excludes `runtime`. Includes every `preproc` entry (including
    `lib_version` -- bumping inscriptis or edgartools changes the output
    text and therefore must change the hash) and `validation` thresholds
    (they determine which rows are marked valid, and therefore which
    extractor firm-level selection picks).
    """
    return _hash_payload(canonicalise(resolved_config))


def row_config_hash(resolved_config: ConfigDict, tool: str, *, code_version: str) -> str:
    """The narrower, per-row hash used to decide resume compatibility.

    A previously-extracted row is reusable only if it was produced under a
    compatible config. This hashes only the extraction-relevant subset for
    a given `tool`: that tool's `preproc` entry, `extract_options`,
    `validation`, `healing.cache_version`, and the extractor code version.
    It deliberately excludes `universe` and `filings.start_date`/`end_date`,
    which do not affect a given row's extracted text -- that's what makes
    incremental builds across overlapping configs correct.
    """
    preproc = (resolved_config.get("preproc") or {}).get(tool, {})
    healing = resolved_config.get("healing") or {}
    subset: ConfigDict = {
        "tool": tool,
        "preproc": preproc,
        "extract_options": resolved_config.get("extract_options") or {},
        "validation": resolved_config.get("validation") or {},
        "healing_cache_version": healing.get("cache_version"),
        "code_version": code_version,
    }
    return _hash_payload(canonicalise(subset))


# --------------------------------------------------------------------------
# Archive naming
# --------------------------------------------------------------------------

_SLUG_RE = re.compile(r"[^a-zA-Z0-9]+")


def slugify(text: str) -> str:
    return _SLUG_RE.sub("-", text).strip("-")


def archive_name(title: str, cfg_hash: str) -> str:
    """A human-readable prefix plus 8 hex characters of the hash.

    The hash is a lookup key, not a decoder -- the full resolved config lives
    inside the archive at `config.json`, readable without decoding the name.
    """
    if not cfg_hash:
        raise ConfigError("cfg_hash must be non-empty")
    return f"{slugify(title)}_{cfg_hash[:8]}.zip"
