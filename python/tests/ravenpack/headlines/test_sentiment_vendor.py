"""RavenPack vendor sentiment: CSS / ESS rules from a synthetic raw zip, end to end."""
from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from datalake import DatalakeIndex
from ravenpack.headlines.sentiment import verify_artifact
from ravenpack.headlines.sentiment_vendor import (
    COLUMNS,
    ingest_to_datalake,
    producer,
    raw_month_batches,
    story_scores,
)

HEADER = ["TIMESTAMP_UTC", "RP_STORY_ID", "RP_ENTITY_ID", "CSS", "EVENT_SENTIMENT_SCORE",
          "EVENT_RELEVANCE", "CATEGORY", "HEADLINE"]

# (story, CSS, ESS, EVENT_RELEVANCE); "" = empty field
ROWS = [
    ("A", "0.10", "-0.82", "100"), ("A", "0.10", "-0.66", "60"), ("A", "0.10", "", ""),
    ("B", "-0.24", "", ""), ("B", "-0.24", "", ""),                  # no event
    ("C", "0.00", "0.50", "0"), ("C", "0.00", "0.30", "0"),           # zero weights
    ("D", "0.04", "0.46", "80"), ("D", "0.04", "0.40", ""),           # unknown relevance
    ("E", "", "", ""),                                                # no CSS at all
]


def _csv(rows) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(HEADER)
    for sid, css, ess, rel in rows:
        w.writerow(["2008-01-02 10:00:00.000", sid, "E1", css, ess, rel,
                    "cat" if ess else "", f"headline {sid}"])
    return buf.getvalue().encode()


def _write_zip(raw_dir: Path, rows, year=2008, month=1) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    stem = f"RavenPackAnalytics_AllEntities_1.0_{year}"
    with zipfile.ZipFile(raw_dir / f"{stem}.zip", "w") as zf:
        zf.writestr(f"{stem}/{year}-{month:02d}.csv", _csv(rows))


def _scores(tmp_path: Path, rows, block_bytes=1 << 20) -> pl.DataFrame:
    _write_zip(tmp_path / "raw", rows)
    hl = tmp_path / "2008-01.parquet"
    pl.DataFrame({"RP_STORY_ID": ["x"]}).write_parquet(hl)
    return producer(tmp_path / "raw", block_bytes)(hl, "t").sort("RP_STORY_ID")


def test_rules_hand_checked(tmp_path: Path):
    out = _scores(tmp_path, ROWS)
    assert out.columns == ["RP_STORY_ID", *COLUMNS]
    assert all(out.schema[c] == pl.Float32 for c in COLUMNS)
    got = {r["RP_STORY_ID"]: r for r in out.iter_rows(named=True)}
    f32 = lambda x: float(np.float32(x))                                 # noqa: E731
    assert got["A"]["SENT_CSS"] == f32(0.10)
    assert got["A"]["SENT_ESS_MEAN"] == f32((-0.82 - 0.66) / 2)
    assert got["A"]["SENT_ESS_WMEAN"] == f32((-0.82 * 100 - 0.66 * 60) / 160)
    assert got["B"]["SENT_CSS"] == f32(-0.24)
    assert np.isnan(got["B"]["SENT_ESS_MEAN"]) and np.isnan(got["B"]["SENT_ESS_WMEAN"])
    assert got["C"]["SENT_ESS_MEAN"] == f32(0.4) and np.isnan(got["C"]["SENT_ESS_WMEAN"])
    assert got["D"]["SENT_ESS_MEAN"] == f32(0.43) and got["D"]["SENT_ESS_WMEAN"] == f32(0.46)
    assert np.isnan(got["E"]["SENT_CSS"]) and np.isnan(got["E"]["SENT_ESS_MEAN"])


def test_story_split_across_batches_and_unordered(tmp_path: Path):
    # tiny CSV blocks force many batches; interleave A's rows with others
    rows = [ROWS[0], ROWS[3], ROWS[1], ROWS[5], ROWS[2], ROWS[4], ROWS[6]]
    many = _scores(tmp_path / "a", rows, block_bytes=128)
    assert len(list(raw_month_batches(tmp_path / "a" / "raw", 2008, 1, 128))) > 2
    one = _scores(tmp_path / "b", sorted(rows), block_bytes=1 << 20)
    assert many.equals(one, null_equal=True)


def test_varying_css_raises(tmp_path: Path):
    with pytest.raises(ValueError, match="CSS varies within 1 stor"):
        _scores(tmp_path, [("A", "0.10", "", ""), ("A", "0.12", "", "")])


def test_empty_batches_give_an_empty_typed_frame():
    out = story_scores(iter([]))
    assert out.height == 0 and out.columns == ["RP_STORY_ID", *COLUMNS]


def test_end_to_end_story_set_from_headlines(tmp_path: Path):
    _write_zip(tmp_path / "raw", ROWS)
    dl = DatalakeIndex(tmp_path / "lake")
    with dl.run(kind="ravenpack_headlines", pipeline="t", pipeline_version="v0") as r:
        # the headlines month holds one story (F) that the raw sample does not score
        pl.DataFrame({"RP_STORY_ID": list("ABCDEF")}).write_parquet(r.out_dir / "2008-01.parquet")
    art = ingest_to_datalake(dl, source="ravenpack", columns=COLUMNS,
                             produce=producer(tmp_path / "raw"),
                             headlines=dl.latest("ravenpack_headlines"), start_year=2008,
                             end_year=2008, extra_hyperparams={}, temp=True)
    out = pl.read_parquet(art.path / "2008-01.parquet")
    assert out["RP_STORY_ID"].to_list() == list("ABCDEF")
    assert np.isnan(out.row(5)[1:]).all()
    assert verify_artifact(art) == []
    dl.close()
