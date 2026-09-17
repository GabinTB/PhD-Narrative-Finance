"""Tests for universe.ingest: registering/reading the universe as a raw datalake artifact."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from datalake import DatalakeIndex
from universe.ingest import (
    KIND,
    ingest_universe,
    load_universe,
    load_universe_by_name,
    load_universe_entries,
    load_universe_entries_by_name,
    register_universe_entries,
)
from universe.schema import UniverseEntry

PIPELINE = "PhD-Narrative-Finance"
VERSION = "v0.1.0"
_DATE = date(2023, 1, 1)


@pytest.fixture
def index(tmp_path: Path) -> DatalakeIndex:
    with DatalakeIndex(tmp_path / "datalake") as idx:
        yield idx


def _entry(**overrides: object) -> UniverseEntry:
    kwargs = {
        "snapshot_date": _DATE,
        "name": "Apple Inc",
        "ticker": "AAPL",
        "isin": "US0378331005",
    }
    kwargs.update(overrides)
    return UniverseEntry(**kwargs)


def _complete_entries() -> list[UniverseEntry]:
    """Entries with all of MANDATORY_AFTER_ENRICHMENT populated (both isin AND
    cusip, per that field list) -- pass the default (allow_incomplete=False)
    completeness gate."""
    return [
        _entry(cusip="037833100", cik="0000320193", gvkey="001690", dbga_secid="111"),
        _entry(
            name="Alphabet Inc",
            ticker="GOOGL",
            isin="US02079K3059",
            cusip="02079K305",
            cik="0001652044",
            gvkey="160329",
            dbga_secid="222",
        ),
    ]


def _write_csv(tmp_path: Path, entries: list[UniverseEntry]) -> Path:
    from universe.schema import to_frame

    csv = tmp_path / "universe.csv"
    to_frame(entries).to_csv(csv, index=False)
    return csv


def test_ingest_universe_registers_raw_layer_artifact(index: DatalakeIndex, tmp_path: Path) -> None:
    csv = _write_csv(tmp_path, _complete_entries())
    artifact = ingest_universe(index, csv, pipeline=PIPELINE, pipeline_version=VERSION)

    assert not artifact.partial
    assert artifact.layer == "raw"
    assert artifact.meta.hyperparams["n_entries"] == 2
    assert artifact.meta.hyperparams["source_path"] == str(csv)


def test_ingest_universe_empty_file_raises(index: DatalakeIndex, tmp_path: Path) -> None:
    csv = tmp_path / "empty.csv"
    csv.write_text("snapshot_date,name,ticker,isin\n")
    with pytest.raises(ValueError, match="no usable universe entries"):
        ingest_universe(index, csv, pipeline=PIPELINE, pipeline_version=VERSION)


def test_load_universe_defaults_to_latest(index: DatalakeIndex, tmp_path: Path) -> None:
    csv = _write_csv(tmp_path, _complete_entries())
    ingest_universe(index, csv, pipeline=PIPELINE, pipeline_version=VERSION)

    df = load_universe(index)
    assert len(df) == 2
    assert set(df["isin"].dropna()) == {"US0378331005", "US02079K3059"}


def test_load_universe_by_explicit_artifact_id(index: DatalakeIndex, tmp_path: Path) -> None:
    csv = _write_csv(tmp_path, _complete_entries())
    artifact = ingest_universe(index, csv, pipeline=PIPELINE, pipeline_version=VERSION)

    df = load_universe(index, artifact.artifact_id)
    assert len(df) == 2


def test_load_universe_entries_roundtrips_to_dataclasses(
    index: DatalakeIndex, tmp_path: Path
) -> None:
    entries = _complete_entries()
    csv = _write_csv(tmp_path, entries)
    ingest_universe(index, csv, pipeline=PIPELINE, pipeline_version=VERSION)

    assert load_universe_entries(index) == entries


def test_ingest_universe_without_name_leaves_hyperparams_unchanged(
    index: DatalakeIndex, tmp_path: Path
) -> None:
    csv = _write_csv(tmp_path, _complete_entries())
    artifact = ingest_universe(index, csv, pipeline=PIPELINE, pipeline_version=VERSION)
    assert "name" not in artifact.meta.hyperparams
    assert "name" not in artifact.artifact_id


def test_ingest_universe_with_name_appears_in_hyperparams_and_id(
    index: DatalakeIndex, tmp_path: Path
) -> None:
    csv = _write_csv(tmp_path, _complete_entries())
    artifact = ingest_universe(
        index, csv, name="sp500_2020", pipeline=PIPELINE, pipeline_version=VERSION
    )
    assert artifact.meta.hyperparams["name"] == "sp500_2020"
    assert "sp500_2020" in artifact.artifact_id


def test_load_universe_by_name_finds_matching_artifact(
    index: DatalakeIndex, tmp_path: Path
) -> None:
    csv = _write_csv(tmp_path, _complete_entries())
    ingest_universe(index, csv, pipeline=PIPELINE, pipeline_version=VERSION)  # unnamed
    ingest_universe(index, csv, name="demo", pipeline=PIPELINE, pipeline_version=VERSION)

    df = load_universe_by_name(index, "demo")
    assert len(df) == 2


def test_load_universe_entries_by_name(index: DatalakeIndex, tmp_path: Path) -> None:
    entries = _complete_entries()
    csv = _write_csv(tmp_path, entries)
    ingest_universe(index, csv, name="demo", pipeline=PIPELINE, pipeline_version=VERSION)

    assert load_universe_entries_by_name(index, "demo") == entries


def test_load_universe_by_name_raises_when_name_not_found(
    index: DatalakeIndex, tmp_path: Path
) -> None:
    csv = _write_csv(tmp_path, _complete_entries())
    ingest_universe(index, csv, name="demo", pipeline=PIPELINE, pipeline_version=VERSION)

    with pytest.raises(ValueError, match="no universe artifact registered with name='ghost'"):
        load_universe_by_name(index, "ghost")


def test_kind_constant() -> None:
    assert KIND == "universe"


# ---------------------------------------------------------------------------
# completeness gate
# ---------------------------------------------------------------------------


def test_ingest_universe_incomplete_raises_by_default(index: DatalakeIndex, tmp_path: Path) -> None:
    """An entry missing e.g. gvkey/dbga_secid (not yet enriched) is refused,
    not silently registered with gaps."""
    csv = _write_csv(tmp_path, [_entry()])  # no cusip/cik/gvkey/dbga_secid
    with pytest.raises(ValueError, match="missing mandatory-after-enrichment"):
        ingest_universe(index, csv, pipeline=PIPELINE, pipeline_version=VERSION)


def test_ingest_universe_incomplete_error_names_missing_fields(
    index: DatalakeIndex, tmp_path: Path
) -> None:
    csv = _write_csv(tmp_path, [_entry()])
    expected = r"row 0: missing \['cusip', 'cik', 'gvkey', 'dbga_secid'\]"
    with pytest.raises(ValueError, match=expected):
        ingest_universe(index, csv, pipeline=PIPELINE, pipeline_version=VERSION)


def test_ingest_universe_allow_incomplete_registers_anyway(
    index: DatalakeIndex, tmp_path: Path
) -> None:
    csv = _write_csv(tmp_path, [_entry()])
    artifact = ingest_universe(
        index, csv, allow_incomplete=True, pipeline=PIPELINE, pipeline_version=VERSION
    )
    assert not artifact.partial
    assert artifact.meta.hyperparams["n_entries"] == 1


def test_register_universe_entries_complete_entries_pass_gate(index: DatalakeIndex) -> None:
    artifact = register_universe_entries(
        index,
        _complete_entries(),
        source_label="in-memory",
        pipeline=PIPELINE,
        pipeline_version=VERSION,
    )
    assert not artifact.partial


def test_register_universe_entries_empty_list_raises(index: DatalakeIndex) -> None:
    with pytest.raises(ValueError, match="no usable universe entries"):
        register_universe_entries(
            index, [], source_label="in-memory", pipeline=PIPELINE, pipeline_version=VERSION
        )
