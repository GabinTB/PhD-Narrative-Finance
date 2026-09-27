"""datalake.layout: partition resolution, legacy monthly mapping."""
from __future__ import annotations

from datetime import date

import pytest

from datalake.layout import Layout, layout_from_hyperparams
from datalake.periods import PeriodError


def test_legacy_year_hyperparams_are_the_monthly_layout_of_today():
    lay = layout_from_hyperparams({"start_year": 2000, "end_year": 2001})
    assert (lay.freq, lay.start, lay.end) == ("M", date(2000, 1, 1), date(2001, 12, 31))
    names = [p.key + ".parquet" for p in lay.expected()]
    assert names == [f"{y}-{m:02d}.parquet" for y in (2000, 2001) for m in range(1, 13)]


def test_new_hyperparams_roundtrip():
    lay = Layout("Q", date(2008, 2, 3), date(2008, 11, 30))
    again = layout_from_hyperparams(lay.hyperparams())
    assert again == lay
    assert [p.key for p in again.expected()] == ["2008Q1", "2008Q2", "2008Q3", "2008Q4"]


def test_no_range_declared():
    lay = layout_from_hyperparams({})
    assert lay.freq == "M" and lay.start is None
    with pytest.raises(PeriodError):
        lay.expected()


@pytest.mark.parametrize("freq,name", [("D", "2008-03-05.parquet"), ("W", "2008-W10.parquet"),
                                       ("M", "2008-03.parquet"), ("Q", "2008Q1.parquet"),
                                       ("Y", "2008.parquet")])
def test_file_for_a_day(tmp_path, freq, name):
    assert Layout(freq).file_for(tmp_path, date(2008, 3, 5)) == tmp_path / name


def test_existing_and_between(tmp_path):
    for name in ("2008-01.parquet", "2008-03.parquet", "2008-02.parquet", "run_metadata.json",
                 "embeddings_provenance.json"):
        (tmp_path / name).write_text("x")
    lay = Layout("M")
    assert list(lay.existing(tmp_path)) == ["2008-01", "2008-02", "2008-03"]
    assert [p.name for p in lay.between(tmp_path, date(2008, 2, 15), date(2008, 3, 1))] == [
        "2008-02.parquet", "2008-03.parquet"]
    (tmp_path / "2008Q2.parquet").write_text("x")
    with pytest.raises(PeriodError, match="Q partition in a M layout"):
        lay.existing(tmp_path)


def test_invalid_layouts():
    with pytest.raises(PeriodError):
        Layout("H")
    with pytest.raises(PeriodError):
        Layout("M", date(2009, 1, 1), date(2008, 1, 1))
