"""Tests for ravenpack.schema and ravenpack.ingest.

These tests use synthetic in-memory data only -- no zip files or real parquets
needed.  They cover:
  - Schema consistency (RAW_SCHEMA, STRUCTURED_SCHEMA, EMBEDDING_SCHEMA)
  - _dedup_stories: entity columns become aligned lists, scalars are first-occurrence
  - _to_arrow_table: output matches STRUCTURED_SCHEMA field-by-field
  - ingest_range: end-to-end via a synthetic zip fixture
  - Edge cases: empty month, single-story month, story spanning chunk boundary,
    missing EVENT_SENTIMENT_SCORE values (NaN)
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from datalake import DatalakeError, DatalakeIndex
from ravenpack.headlines.ingest import (
    KIND,
    _dedup_stories,
    _to_arrow_table,
    ingest_range,
    ingest_to_datalake,
)
from ravenpack.headlines.schema import (
    EMBEDDING_DIM,
    EMBEDDING_SCHEMA,
    ENTITY_LIST_COLS,
    RAW_COLUMNS,
    RAW_SCHEMA,
    SCALAR_COLS,
    STRUCTURED_SCHEMA,
    STRUCTURED_SCHEMA_POLARS,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_raw_df(**overrides) -> pd.DataFrame:
    """Return a minimal valid raw DataFrame (two detections for one story)."""
    base = {
        "TIMESTAMP_UTC":         ["2010-01-05 08:00:00", "2010-01-05 08:00:00"],
        "RP_STORY_ID":           ["S1", "S1"],
        "RP_ENTITY_ID":          ["E1", "E2"],
        "ENTITY_TYPE":           ["COMP", "GOVT"],
        "ENTITY_NAME":           ["Acme Corp", "US Fed"],
        "COUNTRY_CODE":          ["US", "US"],
        "NEWS_TYPE":             ["FULL-ARTICLE", "FULL-ARTICLE"],
        "SOURCE_NAME":           ["Reuters", "Reuters"],
        "HEADLINE":              ["Acme raises rates", "Acme raises rates"],
        "EVENT_SENTIMENT_SCORE": [0.5, -0.3],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def _make_two_story_df() -> pd.DataFrame:
    """Two stories, two entity detections each."""
    return pd.DataFrame({
        "TIMESTAMP_UTC":         ["2010-01-05 08:00:00", "2010-01-05 08:00:00",
                                  "2010-01-05 09:00:00", "2010-01-05 09:00:00"],
        "RP_STORY_ID":           ["S1", "S1", "S2", "S2"],
        "RP_ENTITY_ID":          ["E1", "E2", "E3", "E4"],
        "ENTITY_TYPE":           ["COMP", "GOVT", "COMP", "COMP"],
        "ENTITY_NAME":           ["Acme", "US Fed", "Beta", "Gamma"],
        "COUNTRY_CODE":          ["US", "US", "GB", "GB"],
        "NEWS_TYPE":             ["FULL-ARTICLE"] * 4,
        "SOURCE_NAME":           ["Reuters"] * 4,
        "HEADLINE":              ["Headline A"] * 2 + ["Headline B"] * 2,
        "EVENT_SENTIMENT_SCORE": [0.5, -0.3, 0.1, 0.2],
    })


# ---------------------------------------------------------------------------
# Schema consistency tests
# ---------------------------------------------------------------------------

class TestSchemas:
    def test_raw_columns_match_schema_keys(self):
        assert RAW_COLUMNS == list(RAW_SCHEMA.keys())

    def test_entity_list_cols_subset_of_raw(self):
        assert all(c in RAW_SCHEMA for c in ENTITY_LIST_COLS)

    def test_scalar_cols_disjoint_from_entity_list(self):
        assert not set(SCALAR_COLS) & set(ENTITY_LIST_COLS)
        assert "RP_STORY_ID" not in SCALAR_COLS

    def test_structured_schema_has_list_fields_for_entity_cols(self):
        for col in ENTITY_LIST_COLS:
            field = STRUCTURED_SCHEMA.field(col)
            assert pa.types.is_list(field.type), (
                f"{col} should be list type in STRUCTURED_SCHEMA, got {field.type}"
            )

    def test_structured_schema_has_scalar_fields_for_scalar_cols(self):
        for col in SCALAR_COLS:
            field = STRUCTURED_SCHEMA.field(col)
            assert not pa.types.is_list(field.type), (
                f"{col} should be scalar in STRUCTURED_SCHEMA, got {field.type}"
            )

    def test_embedding_schema_is_two_columns(self):
        # The embedding parquet is a standalone file joined on RP_STORY_ID,
        # not the structured schema plus a column.
        assert list(EMBEDDING_SCHEMA.keys()) == ["RP_STORY_ID", "EMBEDDING"]
        assert EMBEDDING_SCHEMA["RP_STORY_ID"] == pl.String

    def test_embedding_field_is_float16_array(self):
        emb = EMBEDDING_SCHEMA["EMBEDDING"]
        assert isinstance(emb, pl.Array)
        assert emb.inner == pl.Float16
        assert emb.size == EMBEDDING_DIM

    def test_structured_polars_schema_names_match_arrow(self):
        assert list(STRUCTURED_SCHEMA_POLARS.keys()) == STRUCTURED_SCHEMA.names


# ---------------------------------------------------------------------------
# _dedup_stories tests
# ---------------------------------------------------------------------------

class TestDedupStories:
    def test_single_story_two_detections_produces_one_row(self):
        df = _make_raw_df()
        out = _dedup_stories(df)
        assert len(out) == 1

    def test_entity_columns_become_lists(self):
        df = _make_raw_df()
        out = _dedup_stories(df)
        for col in ENTITY_LIST_COLS:
            val = out.iloc[0][col]
            assert isinstance(val, list), f"{col} should be list, got {type(val)}"

    def test_entity_lists_are_aligned(self):
        df = _make_raw_df(
            RP_ENTITY_ID=["E1", "E2"],
            ENTITY_TYPE=["COMP", "GOVT"],
            ENTITY_NAME=["Acme", "Fed"],
            EVENT_SENTIMENT_SCORE=[0.5, -0.3],
        )
        out = _dedup_stories(df)
        row = out.iloc[0]
        assert len(row["RP_ENTITY_ID"]) == len(row["ENTITY_TYPE"])
        assert len(row["RP_ENTITY_ID"]) == len(row["EVENT_SENTIMENT_SCORE"])

    def test_scalar_cols_keep_first_occurrence(self):
        # Two detections: same COUNTRY_CODE (they always are for same story).
        df = _make_raw_df(COUNTRY_CODE=["US", "US"])
        out = _dedup_stories(df)
        assert out.iloc[0]["COUNTRY_CODE"] == "US"

    def test_two_stories_produce_two_rows(self):
        df = _make_two_story_df()
        out = _dedup_stories(df)
        assert len(out) == 2

    def test_empty_input_returns_empty(self):
        df = _make_raw_df().iloc[0:0]  # empty with correct columns
        out = _dedup_stories(df)
        assert out.empty

    def test_single_detection_story(self):
        """Story with only one entity detection: lists of length 1."""
        df = _make_raw_df().iloc[[0]]  # one row
        df = df.copy()
        out = _dedup_stories(df)
        assert len(out) == 1
        assert len(out.iloc[0]["RP_ENTITY_ID"]) == 1

    def test_nan_sentiment_preserved_in_list(self):
        """NaN EVENT_SENTIMENT_SCORE should appear in the list, not be dropped."""
        df = _make_raw_df(EVENT_SENTIMENT_SCORE=[float("nan"), 0.5])
        out = _dedup_stories(df)
        scores = out.iloc[0]["EVENT_SENTIMENT_SCORE"]
        assert len(scores) == 2
        # At least one value should be a float (nan or 0.5).
        assert any(isinstance(s, float) for s in scores)

    def test_timestamp_utc_has_no_nulls(self):
        """Regression: TIMESTAMP_UTC is the sort key, not a groupby key, and was
        previously missing from the agg dict entirely -- pandas' dict-form .agg()
        drops any column not listed, and the trailing .reindex(columns=df.columns)
        silently refilled it as all-null rather than raising. That made every row
        of the real ravenpack_headlines artifact carry a null TIMESTAMP_UTC.
        """
        df = _make_two_story_df()
        out = _dedup_stories(df)
        assert out["TIMESTAMP_UTC"].notna().all()
        assert out["TIMESTAMP_UTC"].tolist() == [
            "2010-01-05 08:00:00", "2010-01-05 09:00:00",
        ]

    def test_timestamp_utc_keeps_first_occurrence(self):
        """Same first-occurrence convention as the other scalar columns."""
        df = _make_raw_df(
            TIMESTAMP_UTC=["2010-01-05 08:00:00", "2010-01-05 08:00:05"],
        )
        out = _dedup_stories(df)
        assert out.iloc[0]["TIMESTAMP_UTC"] == "2010-01-05 08:00:00"


# ---------------------------------------------------------------------------
# _to_arrow_table tests
# ---------------------------------------------------------------------------

class TestToArrowTable:
    def test_output_schema_matches_structured_schema(self):
        df = _dedup_stories(_make_two_story_df())
        table = _to_arrow_table(df)
        assert table.schema.equals(STRUCTURED_SCHEMA)

    def test_row_count_preserved(self):
        df = _dedup_stories(_make_two_story_df())
        table = _to_arrow_table(df)
        assert len(table) == 2

    def test_entity_columns_are_list_arrays(self):
        df = _dedup_stories(_make_raw_df())
        table = _to_arrow_table(df)
        for col in ENTITY_LIST_COLS:
            col_type = table.schema.field(col).type
            assert pa.types.is_list(col_type), f"{col}: expected list, got {col_type}"

    def test_roundtrip_parquet(self, tmp_path):
        df = _dedup_stories(_make_two_story_df())
        table = _to_arrow_table(df)
        path = tmp_path / "test.parquet"
        pq.write_table(table, path)
        back = pq.read_table(path)
        assert back.schema.equals(STRUCTURED_SCHEMA)
        assert len(back) == 2


# ---------------------------------------------------------------------------
# ingest_range end-to-end tests
# ---------------------------------------------------------------------------

def _make_zip(raw_dir: Path, year: int, month: int, df: pd.DataFrame) -> None:
    """Write a synthetic zip file matching RavenPack's layout convention."""
    stem = f"RavenPackAnalytics_AllEntities_1.0_{year}"
    zip_path = raw_dir / f"{stem}.zip"
    member = f"{stem}/{year}-{month:02d}.csv"
    csv_buf = df.to_csv(index=False).encode()
    mode = "a" if zip_path.exists() else "w"
    with zipfile.ZipFile(zip_path, mode) as zf:
        zf.writestr(member, csv_buf)


class TestIngestRange:
    def test_basic_month_produces_parquet(self, tmp_path):
        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "out"
        raw_dir.mkdir()

        df = _make_two_story_df()
        _make_zip(raw_dir, 2010, 1, df)

        ingest_range(raw_dir, out_dir, start_year=2010, end_year=2010)

        out_path = out_dir / "2010-01.parquet"
        assert out_path.exists(), "expected output parquet to exist"

        table = pq.read_table(out_path)
        assert table.schema.equals(STRUCTURED_SCHEMA)
        assert len(table) == 2

    def test_resumable_skips_existing(self, tmp_path):
        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "out"
        raw_dir.mkdir()
        out_dir.mkdir()

        df = _make_two_story_df()
        _make_zip(raw_dir, 2010, 1, df)

        # Pre-create the output file.
        sentinel = out_dir / "2010-01.parquet"
        sentinel.write_bytes(b"placeholder")

        ingest_range(raw_dir, out_dir, start_year=2010, end_year=2010)

        # File should be the placeholder, not replaced.
        assert sentinel.read_bytes() == b"placeholder"

    def test_overwrite_replaces_existing(self, tmp_path):
        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "out"
        raw_dir.mkdir()
        out_dir.mkdir()

        df = _make_two_story_df()
        _make_zip(raw_dir, 2010, 1, df)

        sentinel = out_dir / "2010-01.parquet"
        sentinel.write_bytes(b"placeholder")

        ingest_range(raw_dir, out_dir, start_year=2010, end_year=2010, overwrite=True)

        table = pq.read_table(sentinel)
        assert len(table) == 2  # real data, not placeholder

    def test_missing_zip_skipped_gracefully(self, tmp_path):
        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "out"
        raw_dir.mkdir()
        # No zip written.

        ingest_range(raw_dir, out_dir, start_year=2010, end_year=2010)

        assert not list(out_dir.glob("*.parquet")), "no output expected for missing zip"

    def test_empty_month_produces_no_file(self, tmp_path):
        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "out"
        raw_dir.mkdir()

        # CSV with header only, no rows.
        empty_df = pd.DataFrame(columns=RAW_COLUMNS)
        _make_zip(raw_dir, 2010, 1, empty_df)

        ingest_range(raw_dir, out_dir, start_year=2010, end_year=2010)

        assert not (out_dir / "2010-01.parquet").exists()

    def test_no_tmp_file_left_on_success(self, tmp_path):
        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "out"
        raw_dir.mkdir()

        _make_zip(raw_dir, 2010, 1, _make_two_story_df())
        ingest_range(raw_dir, out_dir, start_year=2010, end_year=2010)

        tmp_files = list(out_dir.glob("*.tmp"))
        assert not tmp_files, f"unexpected tmp files: {tmp_files}"

    def test_chunk_boundary_story_not_lost(self, tmp_path):
        """Story spanning a chunk boundary must appear exactly once in output."""
        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "out"
        raw_dir.mkdir()

        # Build a 4-row frame: 2 detections for S1, 2 for S2.
        df = _make_two_story_df()
        _make_zip(raw_dir, 2010, 1, df)

        # raw_chunk_rows=2 forces a chunk boundary after S1's first detection
        # (row 0, 1 are S1; row 2, 3 are S2).  With carry logic, S1 should
        # still produce exactly one output row.
        ingest_range(
            raw_dir, out_dir,
            start_year=2010, end_year=2010,
            raw_chunk_rows=2,
        )

        table = pq.read_table(out_dir / "2010-01.parquet")
        story_ids = table.column("RP_STORY_ID").to_pylist()
        assert story_ids.count("S1") == 1, "S1 should appear exactly once"
        assert story_ids.count("S2") == 1, "S2 should appear exactly once"

    def test_multi_year_range(self, tmp_path):
        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "out"
        raw_dir.mkdir()

        for year in [2010, 2011]:
            _make_zip(raw_dir, year, 1, _make_two_story_df())

        ingest_range(raw_dir, out_dir, start_year=2010, end_year=2011)

        assert (out_dir / "2010-01.parquet").exists()
        assert (out_dir / "2011-01.parquet").exists()


# ---------------------------------------------------------------------------
# Datalake integration
# ---------------------------------------------------------------------------


class TestIngestToDatalake:
    def test_produces_registered_artifact(self, tmp_path):
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        _make_zip(raw_dir, 2010, 1, _make_two_story_df())

        with DatalakeIndex(tmp_path / "dl") as index:
            artifact = ingest_to_datalake(
                index, raw_dir, 2010, 2010, pipeline_version="v0.1.0",
            )
            assert artifact.kind == KIND
            assert not artifact.partial
            assert index.exists(artifact.artifact_id)

    def test_parquet_written_into_artifact_dir(self, tmp_path):
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        _make_zip(raw_dir, 2010, 1, _make_two_story_df())

        with DatalakeIndex(tmp_path / "dl") as index:
            artifact = ingest_to_datalake(
                index, raw_dir, 2010, 2010, pipeline_version="v0.1.0",
            )
            files = list(artifact.path.glob("*.parquet"))
            assert len(files) == 1
            assert pq.read_table(files[0]).schema.equals(STRUCTURED_SCHEMA)

    def test_files_are_hashed(self, tmp_path):
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        for month in (1, 2):
            _make_zip(raw_dir, 2010, month, _make_two_story_df())

        with DatalakeIndex(tmp_path / "dl") as index:
            artifact = ingest_to_datalake(
                index, raw_dir, 2010, 2010, pipeline_version="v0.1.0",
            )
            assert len(artifact.file_hashes) == 2

    def test_run_notes_record_column_selection(self, tmp_path):
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        _make_zip(raw_dir, 2010, 1, _make_two_story_df())

        with DatalakeIndex(tmp_path / "dl") as index:
            artifact = ingest_to_datalake(
                index, raw_dir, 2010, 2010, pipeline_version="v0.1.0",
            )
            notes = artifact.meta.runs[0].notes
            assert "EVENT_SENTIMENT_SCORE" in notes

    def test_empty_range_raises_and_leaves_partial(self, tmp_path):
        """No input data must fail loudly, not register an empty artifact."""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()   # no zips at all

        with DatalakeIndex(tmp_path / "dl") as index:
            with pytest.raises(RuntimeError, match="produced no output"):
                ingest_to_datalake(
                    index, raw_dir, 2010, 2010, pipeline_version="v0.1.0",
                )
            # Nothing consumable, but the failure is on record.
            with pytest.raises(DatalakeError):
                index.latest(KIND)
            assert len(index.list(KIND, include_partial=True)) == 1

    def test_downstream_can_resolve_via_latest(self, tmp_path):
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        _make_zip(raw_dir, 2010, 1, _make_two_story_df())

        with DatalakeIndex(tmp_path / "dl") as index:
            produced = ingest_to_datalake(
                index, raw_dir, 2010, 2010, pipeline_version="v0.1.0",
            )
            assert index.latest(KIND).artifact_id == produced.artifact_id


def test_script_resume_completes_a_killed_ingest(tmp_path, monkeypatch):
    """scripts/ingest_ravenpack.py resume: params from the artifact, completed in place."""
    import importlib.util

    import ravenpack.headlines.ingest as ingest_mod

    spec = importlib.util.spec_from_file_location(
        "ingest_script", Path(__file__).resolve().parents[3] / "scripts" / "ingest_ravenpack.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)

    raw_all, raw_jan = tmp_path / "raw", tmp_path / "raw_jan"
    raw_all.mkdir()
    raw_jan.mkdir()
    for month in (1, 2):
        _make_zip(raw_all, 2010, month, _make_two_story_df())
    _make_zip(raw_jan, 2010, 1, _make_two_story_df())
    real = ingest_mod.ingest_range

    def killed_after_january(**kw):
        real(**{**kw, "raw_dir": raw_jan})
        raise RuntimeError("killed")

    with DatalakeIndex(tmp_path / "dl") as index:
        monkeypatch.setattr(ingest_mod, "ingest_range", killed_after_january)
        with pytest.raises(RuntimeError, match="killed"):
            ingest_to_datalake(index, raw_all, 2010, 2010, pipeline_version="v0.1.0")
        monkeypatch.setattr(ingest_mod, "ingest_range", real)
        part = index.list(KIND, include_partial=True)[0]
        assert part.partial and [p.name for p in part.path.glob("*.parquet")] == \
            ["2010-01.parquet"]
        done = script._resume(index, part.artifact_id, raw_all, 2_000_000, 100_000)
        assert not done.partial and done.artifact_id == part.artifact_id
        assert {"2010-01.parquet", "2010-02.parquet"} <= set(done.file_hashes)
        assert len(done.meta.runs) == 2


# ---------------------------------------------------------------------------
# Partition frequencies (datalake.layout): same content, other partitioning
# ---------------------------------------------------------------------------

def _month_df(year: int, month: int, n: int = 6) -> pd.DataFrame:
    """n stories spread over the month (two detections each), one per ~5 days."""
    rows = []
    for i in range(n):
        day = min(1 + 5 * i, 28)
        ts = f"{year}-{month:02d}-{day:02d} 0{i % 9}:00:00"
        for e in (1, 2):
            rows.append({"TIMESTAMP_UTC": ts, "RP_STORY_ID": f"S{year}{month:02d}{i}",
                         "RP_ENTITY_ID": f"E{e}", "ENTITY_TYPE": "COMP",
                         "ENTITY_NAME": f"N{e}", "COUNTRY_CODE": "US",
                         "NEWS_TYPE": "FULL-ARTICLE", "SOURCE_NAME": "Reuters",
                         "HEADLINE": f"H{i}", "EVENT_SENTIMENT_SCORE": 0.1 * e})
    return pd.DataFrame(rows)


def _raw_2010(tmp_path: Path, months=(1, 2, 3, 4)) -> Path:
    raw = tmp_path / "raw"
    raw.mkdir()
    for m in months:
        _make_zip(raw, 2010, m, _month_df(2010, m))
    return raw


def _all_rows(out: Path) -> pd.DataFrame:
    import polars as pl

    frames = [pl.read_parquet(p) for p in sorted(out.glob("*.parquet"))]
    return pl.concat(frames).sort("RP_STORY_ID").to_pandas()


@pytest.mark.parametrize("freq", ["D", "W", "Q", "Y"])
def test_other_partitionings_hold_exactly_the_monthly_content(tmp_path, freq):
    from datetime import date

    from datalake.layout import Layout

    raw = _raw_2010(tmp_path)
    monthly, other = tmp_path / "M", tmp_path / freq
    ingest_range(raw, monthly, layout=Layout("M", date(2010, 1, 1), date(2010, 4, 30)))
    ingest_range(raw, other, layout=Layout(freq, date(2010, 1, 1), date(2010, 4, 30)))
    lay = Layout(freq)
    for key, path in lay.existing(other).items():                 # every row in its period
        import polars as pl

        days = pl.read_parquet(path)["TIMESTAMP_UTC"].str.slice(0, 10).to_list()
        from datalake.periods import parse_key

        period = parse_key(key)
        assert all(period.contains(date.fromisoformat(d)) for d in days)
    pd.testing.assert_frame_equal(_all_rows(monthly), _all_rows(other))


def test_week_straddling_two_months_is_complete(tmp_path):
    from datetime import date

    from datalake.layout import Layout

    raw = _raw_2010(tmp_path)
    out = tmp_path / "W"
    ingest_range(raw, out, layout=Layout("W", date(2010, 1, 25), date(2010, 2, 7)))
    # ISO week 2010-W04 = Jan 25..31, W05 = Feb 1..7; stories on Jan 26 and Feb 1/6
    assert set(Layout("W").existing(out)) == {"2010-W04", "2010-W05"}


def test_row_dated_outside_its_raw_month_raises_for_daily(tmp_path):
    from datetime import date

    from datalake.layout import Layout

    raw = tmp_path / "raw"
    raw.mkdir()
    bad = _month_df(2010, 1)
    bad.loc[0:1, "TIMESTAMP_UTC"] = "2010-02-01 00:00:00"
    _make_zip(raw, 2010, 1, bad)
    with pytest.raises(ValueError, match="outside 2010-01"):
        ingest_range(raw, tmp_path / "D", layout=Layout("D", date(2010, 1, 1), date(2010, 1, 31)))
    assert not list((tmp_path / "D").glob("*.parquet*"))           # nothing half-written


def test_datalake_records_the_layout(tmp_path):
    from datetime import date

    from datalake.layout import Layout, layout_of

    raw = _raw_2010(tmp_path, months=(1, 2))
    with DatalakeIndex(tmp_path / "dl") as index:
        art = ingest_to_datalake(index, raw, layout=Layout("Q", date(2010, 1, 15),
                                                           date(2010, 2, 20)),
                                 pipeline_version="v0.2.0")
        assert layout_of(art) == Layout("Q", date(2010, 1, 1), date(2010, 3, 31))
        assert list(layout_of(art).existing(art.path)) == ["2010Q1"]
        from ravenpack.headlines.ingest import verify_artifact

        assert verify_artifact(art) == []
