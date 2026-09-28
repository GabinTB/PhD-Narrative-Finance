"""headline_sentiment contract: validation, alignment, the shared driver (fresh + resume),
and the verifier. Sandbox datalake only."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from datalake import DatalakeIndex
from nlp.sentiment import GRID_COLUMNS
from ravenpack.headlines.sentiment import (
    KIND,
    SentimentContractError,
    align_to_stories,
    fill_months,
    ingest_to_datalake,
    resume_partial,
    validate_sentiment_frame,
    verify_artifact,
)

NAN = float("nan")


def _frame(ids, **cols) -> pl.DataFrame:
    return pl.DataFrame({"RP_STORY_ID": ids,
                         **{k: pl.Series(v, dtype=pl.Float32) for k, v in cols.items()}})


class TestContract:
    def test_valid_frame_returns_score_columns_in_order(self):
        df = _frame(["a", "b"], SENT_X=[0.5, NAN], SENT_A=[-1.0, 1.0]).with_columns(
            pl.lit(0.3, dtype=pl.Float16).alias("P_POS"))
        assert validate_sentiment_frame(df) == ["SENT_X", "SENT_A"]

    @pytest.mark.parametrize("df, match", [
        (_frame(["a"], P_POS=[0.1]), "no SENT_"),
        (_frame(["a"], SENT_X=[1.5]), "outside"),
        (_frame(["a"], SENT_X=[float("inf")]), "outside"),
        (_frame(["a", "a"], SENT_X=[0.1, 0.2]), "duplicate"),
        (pl.DataFrame({"RP_STORY_ID": ["a"], "SENT_X": [0.1]}), "Float32"),     # Float64
        (pl.DataFrame({"RP_STORY_ID": ["a"], "SENT_X": pl.Series([None], dtype=pl.Float32)}),
         "null"),
        (pl.DataFrame({"RP_STORY_ID": [1], "SENT_X": pl.Series([0.1], dtype=pl.Float32)}),
         "String"),
        (pl.DataFrame({"SENT_X": pl.Series([0.1], dtype=pl.Float32)}), "missing RP_STORY_ID"),
    ])
    def test_violations_raise(self, df, match):
        with pytest.raises(SentimentContractError, match=match):
            validate_sentiment_frame(df)


class TestAlign:
    def test_order_and_missing_story_is_nan(self):
        scores = _frame(["c", "a"], SENT_X=[0.3, -0.2]).with_columns(
            pl.lit("x").alias("NOTE"))
        out = align_to_stories(scores, pl.Series(["a", "b", "c"]))
        assert out["RP_STORY_ID"].to_list() == ["a", "b", "c"]
        np.testing.assert_array_equal(out["SENT_X"].to_numpy(),
                                      np.array([-0.2, NAN, 0.3], dtype=np.float32))
        assert out["SENT_X"].null_count() == 0 and out["NOTE"].to_list() == ["x", None, "x"]

    def test_extra_or_duplicate_ids_raise(self):
        with pytest.raises(SentimentContractError, match="absent from the headlines"):
            align_to_stories(_frame(["a", "z"], SENT_X=[0.1, 0.2]), pl.Series(["a"]))
        with pytest.raises(SentimentContractError, match="duplicate"):
            align_to_stories(_frame(["a"], SENT_X=[0.1]), pl.Series(["a", "a"]))
        with pytest.raises(SentimentContractError, match="duplicate"):
            align_to_stories(_frame(["a", "a"], SENT_X=[0.1, 0.1]), pl.Series(["a"]))


ID_COL = "RP_STORY_ID"


def _grid(ids, G) -> pl.DataFrame:
    """A grid table (Float16, all-NaN rows written as all null)."""
    import pyarrow as pa

    G16 = np.asarray(G, dtype=np.float16)
    empty = np.isnan(G16).all(axis=1)
    cols = {ID_COL: pa.array(ids, type=pa.string())}
    for j, c in enumerate(GRID_COLUMNS):
        cols[c] = pa.array(G16[:, j], type=pa.float16(), mask=empty)
    return pl.from_arrow(pa.table(cols))


class TestGridContract:
    def test_valid_grid_with_a_no_output_row(self):
        G = np.vstack([np.eye(41)[3], np.full(41, 1 / 41), np.full(41, np.nan)])
        assert validate_sentiment_frame(_grid(["a", "b", "c"], G)) == list(GRID_COLUMNS)

    def test_some_but_not_all_null_refused(self):
        df = _grid(["a"], np.full((1, 41), 1 / 41)).with_columns(
            pl.lit(None, dtype=pl.Float16).alias("P_07"))
        with pytest.raises(SentimentContractError, match="some but not all"):
            validate_sentiment_frame(df)

    @pytest.mark.parametrize("scale", [0.95, 1.05])
    def test_row_sum_outside_tolerance_refused(self, scale):
        with pytest.raises(SentimentContractError, match="row sum"):
            validate_sentiment_frame(_grid(["a"], np.full((1, 41), scale / 41)))

    def test_row_sum_within_float16_tolerance_accepted(self):
        validate_sentiment_frame(_grid(["a"], np.full((1, 41), 1.008 / 41)))

    def test_float32_grid_and_mixed_tables_refused(self):
        g = _grid(["a"], np.full((1, 41), 1 / 41))
        with pytest.raises(SentimentContractError, match="Float16"):
            validate_sentiment_frame(g.with_columns(pl.col("P_00").cast(pl.Float32)))
        with pytest.raises(SentimentContractError, match="SENT_"):
            validate_sentiment_frame(g.with_columns(pl.lit(0.1, dtype=pl.Float32)
                                                    .alias("SENT_CSS")))

    def test_missing_story_stays_all_null_after_alignment(self):
        g = _grid(["a"], np.full((1, 41), 1 / 41))
        out = align_to_stories(g, pl.Series(["a", "b"]))
        assert out.filter(pl.col(ID_COL) == "b").select(
            pl.all_horizontal(pl.col(GRID_COLUMNS).is_null())).item()
        validate_sentiment_frame(out)


# ---------------------------------------------------------------------------
# Driver + verifier on a sandbox lake
# ---------------------------------------------------------------------------

MONTHS = {"2008-01.parquet": ["s1", "s2", "s3"], "2008-02.parquet": ["t1", "t2"]}


@pytest.fixture
def lake(tmp_path: Path):
    dl = DatalakeIndex(tmp_path / "lake")
    with dl.run(kind="ravenpack_headlines", pipeline="t", pipeline_version="v0",
                hyperparams={"start_year": 2008, "end_year": 2008}) as r:
        for name, ids in MONTHS.items():
            pl.DataFrame({"RP_STORY_ID": ids, "HEADLINE": [f"h {i}" for i in ids]}) \
                .write_parquet(r.out_dir / name)
    yield dl
    dl.close()


def _producer(calls: list[str] | None = None, fail_on: str | None = None):
    def produce(path: Path, tag: str) -> pl.DataFrame:
        if calls is not None:
            calls.append(path.name)
        if fail_on and path.name == fail_on:
            raise RuntimeError("boom")
        ids = pl.read_parquet(path)["RP_STORY_ID"].to_list()
        scored = ids[:-1]                            # last story unscored -> NaN row
        return _frame(scored, SENT_A=[0.5] * len(scored), SENT_B=[-0.25] * len(scored))
    return produce


def test_fresh_run_registers_a_valid_temp_artifact(lake):
    hl = lake.latest("ravenpack_headlines")
    art = ingest_to_datalake(lake, source="toy", columns=["SENT_A", "SENT_B"],
                             produce=_producer(), headlines=hl, start_year=2008, end_year=2008,
                             extra_hyperparams={"rule": "x"}, temp=True)
    assert art.kind == KIND and "__TEMP" in art.artifact_id and not art.partial
    assert art.meta.hyperparams["agent_created"] is True
    assert art.meta.hyperparams["headlines_id"] == hl.artifact_id
    assert art.meta.sources == [hl.artifact_id]
    assert sorted(art.file_hashes) == sorted(MONTHS)
    jan = pl.read_parquet(art.path / "2008-01.parquet")
    assert jan["RP_STORY_ID"].to_list() == MONTHS["2008-01.parquet"]
    assert np.isnan(jan["SENT_A"][2]) and jan["SENT_A"][0] == 0.5
    assert verify_artifact(art) == []


def test_declared_columns_are_enforced(tmp_path: Path, lake):
    hl = lake.latest("ravenpack_headlines")
    with pytest.raises(SentimentContractError, match="declared"):
        fill_months(_producer(), hl.path, tmp_path, list(MONTHS), ["SENT_B", "SENT_A"])


def test_resume_writes_only_missing_months_and_completes(lake):
    hl = lake.latest("ravenpack_headlines")
    with pytest.raises(RuntimeError, match="boom"):
        ingest_to_datalake(lake, source="toy", columns=["SENT_A", "SENT_B"],
                           produce=_producer(fail_on="2008-02.parquet"), headlines=hl,
                           start_year=2008, end_year=2008, extra_hyperparams={}, temp=True)
    partial = lake.list(KIND, include_partial=True)[0]
    assert partial.partial and (partial.path / "2008-01.parquet").exists()
    calls: list[str] = []
    art = resume_partial(lake, partial.artifact_id, _producer(calls))
    assert calls == ["2008-02.parquet"] and not art.partial
    assert sorted(art.file_hashes) == sorted(MONTHS) and verify_artifact(art) == []
    with pytest.raises(ValueError, match="complete"):
        resume_partial(lake, art.artifact_id, _producer())


def test_verifier_flags_corruption(lake):
    hl = lake.latest("ravenpack_headlines")
    art = ingest_to_datalake(lake, source="toy", columns=["SENT_A", "SENT_B"],
                             produce=_producer(), headlines=hl, start_year=2008, end_year=2008,
                             extra_hyperparams={}, temp=True)
    # a dropped story breaks the bijection; an out-of-range value breaks the contract
    jan = pl.read_parquet(art.path / "2008-01.parquet")
    jan.head(2).write_parquet(art.path / "2008-01.parquet")
    feb = pl.read_parquet(art.path / "2008-02.parquet").with_columns(
        pl.lit(2.0, dtype=pl.Float32).alias("SENT_A"))
    feb.write_parquet(art.path / "2008-02.parquet")
    msgs = " | ".join(f.message for f in verify_artifact(art))
    assert "2008-01.parquet: RP_STORY_ID set differs" in msgs
    assert "outside [-1, 1]" in msgs
