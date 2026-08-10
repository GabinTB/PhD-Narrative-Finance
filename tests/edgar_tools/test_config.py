"""Tests for edgar_tools.config: resolution, canonicalisation, hashing.

Per .claude/tasks/02-config-schema-and-hashing.md.
"""

from datetime import date

import pytest

from edgar_tools.config import (
    ConfigError,
    archive_name,
    canonicalise,
    config_hash,
    resolve_config,
    row_config_hash,
)


def _base_config(**overrides: object) -> dict:
    cfg = {
        "schema_version": "1",
        "universe": {
            "ciks": ["320193", "789019"],
            "company_names": [],
            "resolve_as_of": "2026-07-29",
        },
        "filings": {
            "forms": ["10-K", "10-Q"],
            "start_date": "2004-01-01",
            "end_date": "2025-12-31",
        },
        "items_to_extract": {
            "10-K": ["1", "1A", "3", "7", "7A"],
            "10-Q": ["part_1__2", "part_2__1A"],
        },
        "tools": ["edgartools", "edgar-crawler", "itemseg-bert", "own-regex"],
        "preproc": {
            "edgartools": {"engine": "dom", "lib_version": "5.43.0"},
            "edgar-crawler": {
                "engine": "ec-strip",
                "lib_version": "1.2.0",
                "remove_tables": True,
            },
            "itemseg-bert": {"engine": "inscriptis", "lib_version": "2.5.0"},
            "own-regex": {
                "engine": "inscriptis",
                "lib_version": "2.5.0",
                "title_keyword": True,
                "comma_lookbehind": True,
            },
        },
        "extract_options": {"include_signature": False},
        "healing": {"enabled": False, "cache_version": None},
        "validation": {
            "min_words": 50,
            "max_doc_fraction": 0.6,
            "reject_if_terminator_present": True,
            "reject_boilerplate_xref": True,
        },
        "runtime": {"user_agent": "Test (test@example.com)", "workers": 8, "root": "/data/edgar"},
    }
    cfg.update(overrides)
    return cfg


# --------------------------------------------------------------------------
# items_to_extract validation
# --------------------------------------------------------------------------


def test_flat_items_to_extract_raises() -> None:
    cfg = _base_config(items_to_extract=["1A", "7"])
    with pytest.raises(ConfigError):
        resolve_config(cfg, today=date(2026, 8, 1))


def test_null_items_to_extract_is_allowed() -> None:
    cfg = _base_config(items_to_extract=None)
    resolved = resolve_config(cfg, today=date(2026, 8, 1))
    assert resolved["items_to_extract"] is None


# --------------------------------------------------------------------------
# Resolution
# --------------------------------------------------------------------------


def test_end_date_null_resolves_to_concrete_date() -> None:
    cfg = _base_config()
    cfg["filings"]["end_date"] = None
    resolved = resolve_config(cfg, today=date(2026, 8, 1))
    assert resolved["filings"]["end_date"] == "2026-08-01"


def test_end_date_hash_stable_across_calls_same_injected_day() -> None:
    cfg = _base_config()
    cfg["filings"]["end_date"] = None
    resolved_a = resolve_config(cfg, today=date(2026, 8, 1))
    resolved_b = resolve_config(cfg, today=date(2026, 8, 1))
    assert config_hash(resolved_a) == config_hash(resolved_b)


def test_end_date_hash_differs_across_injected_days() -> None:
    cfg = _base_config()
    cfg["filings"]["end_date"] = None
    resolved_a = resolve_config(cfg, today=date(2026, 8, 1))
    resolved_b = resolve_config(cfg, today=date(2026, 9, 1))
    assert config_hash(resolved_a) != config_hash(resolved_b)


def test_ciks_normalised_and_sorted() -> None:
    cfg = _base_config()
    cfg["universe"]["ciks"] = ["789019", 320193]
    resolved = resolve_config(cfg, today=date(2026, 8, 1))
    assert resolved["universe"]["ciks"] == ["0000320193", "0000789019"]


def test_company_names_require_resolver() -> None:
    cfg = _base_config()
    cfg["universe"]["company_names"] = ["Apple Inc"]
    with pytest.raises(ConfigError):
        resolve_config(cfg, today=date(2026, 8, 1))


def test_company_names_resolved_and_merged_into_ciks() -> None:
    cfg = _base_config()
    cfg["universe"]["ciks"] = []
    cfg["universe"]["company_names"] = ["Apple Inc"]

    def fake_resolver(names: list[str], as_of: date) -> list[str]:
        assert names == ["Apple Inc"]
        assert as_of == date(2026, 7, 29)
        return ["320193"]

    resolved = resolve_config(cfg, today=date(2026, 8, 1), resolve_company_names=fake_resolver)
    assert resolved["universe"]["ciks"] == ["0000320193"]


# --------------------------------------------------------------------------
# Canonicalisation
# --------------------------------------------------------------------------


def test_canonicalise_drops_runtime() -> None:
    cfg = _base_config()
    canon = canonicalise(cfg)
    assert "runtime" not in canon


def test_canonicalise_sorts_unordered_arrays() -> None:
    cfg = _base_config()
    cfg["universe"]["ciks"] = ["0000789019", "0000320193"]
    cfg["tools"] = ["own-regex", "edgartools"]
    canon = canonicalise(cfg)
    assert canon["universe"]["ciks"] == ["0000320193", "0000789019"]
    assert canon["tools"] == ["edgartools", "own-regex"]


# --------------------------------------------------------------------------
# config_hash
# --------------------------------------------------------------------------


def test_hash_stable_under_key_and_array_reordering() -> None:
    cfg_a = _base_config()
    cfg_b = {k: cfg_a[k] for k in reversed(list(cfg_a.keys()))}
    cfg_b["universe"] = dict(cfg_a["universe"])
    cfg_b["universe"]["ciks"] = list(reversed(cfg_a["universe"]["ciks"]))
    cfg_b["tools"] = list(reversed(cfg_a["tools"]))

    resolved_a = resolve_config(cfg_a, today=date(2026, 8, 1))
    resolved_b = resolve_config(cfg_b, today=date(2026, 8, 1))

    assert config_hash(resolved_a) == config_hash(resolved_b)


def test_hash_unaffected_by_runtime_changes() -> None:
    cfg_a = _base_config()
    cfg_b = _base_config()
    cfg_b["runtime"] = {
        "user_agent": "Someone Else (someone@else.com)",
        "workers": 64,
        "root": "abfss://different/path",
    }

    resolved_a = resolve_config(cfg_a, today=date(2026, 8, 1))
    resolved_b = resolve_config(cfg_b, today=date(2026, 8, 1))

    assert config_hash(resolved_a) == config_hash(resolved_b)


def test_hash_changes_with_preproc_lib_version() -> None:
    cfg_a = _base_config()
    cfg_b = _base_config()
    cfg_b["preproc"]["edgartools"]["lib_version"] = "5.44.0"

    resolved_a = resolve_config(cfg_a, today=date(2026, 8, 1))
    resolved_b = resolve_config(cfg_b, today=date(2026, 8, 1))

    assert config_hash(resolved_a) != config_hash(resolved_b)


# --------------------------------------------------------------------------
# row_config_hash
# --------------------------------------------------------------------------


def test_row_hash_differs_by_tool() -> None:
    resolved = resolve_config(_base_config(), today=date(2026, 8, 1))
    h1 = row_config_hash(resolved, "edgartools", code_version="git:abc123")
    h2 = row_config_hash(resolved, "own-regex", code_version="git:abc123")
    assert h1 != h2


def test_row_hash_differs_by_preproc() -> None:
    cfg_a = _base_config()
    cfg_b = _base_config()
    cfg_b["preproc"]["edgartools"]["lib_version"] = "5.44.0"

    resolved_a = resolve_config(cfg_a, today=date(2026, 8, 1))
    resolved_b = resolve_config(cfg_b, today=date(2026, 8, 1))

    h1 = row_config_hash(resolved_a, "edgartools", code_version="git:abc123")
    h2 = row_config_hash(resolved_b, "edgartools", code_version="git:abc123")
    assert h1 != h2


def test_row_hash_identical_when_only_universe_or_dates_differ() -> None:
    cfg_a = _base_config()
    cfg_b = _base_config()
    cfg_b["universe"]["ciks"] = ["0000012345"]
    cfg_b["filings"]["start_date"] = "2010-01-01"
    cfg_b["filings"]["end_date"] = "2020-01-01"

    resolved_a = resolve_config(cfg_a, today=date(2026, 8, 1))
    resolved_b = resolve_config(cfg_b, today=date(2026, 8, 1))

    h1 = row_config_hash(resolved_a, "edgartools", code_version="git:abc123")
    h2 = row_config_hash(resolved_b, "edgartools", code_version="git:abc123")
    assert h1 == h2

    # But the full dataset identity *does* differ under a universe/date change.
    assert config_hash(resolved_a) != config_hash(resolved_b)


def test_row_hash_changes_with_code_version() -> None:
    resolved = resolve_config(_base_config(), today=date(2026, 8, 1))
    h1 = row_config_hash(resolved, "edgartools", code_version="git:abc123")
    h2 = row_config_hash(resolved, "edgartools", code_version="git:def456")
    assert h1 != h2


# --------------------------------------------------------------------------
# archive_name
# --------------------------------------------------------------------------


def test_archive_name_format() -> None:
    name = archive_name("MSCI World 10-K/10-Q Item 1A+7, 2004-2025", "a3f9c1e2beef")
    assert name.endswith("_a3f9c1e2.zip")
    assert name.startswith("MSCI-World-10-K-10-Q-Item-1A-7-2004-2025")


def test_archive_name_rejects_empty_hash() -> None:
    with pytest.raises(ConfigError):
        archive_name("title", "")
