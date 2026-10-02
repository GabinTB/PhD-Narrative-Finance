"""wrds_client.capitaliq.keydev: monthly Key Developments parquets (fake WRDS, no network)."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pandas as pd
import polars as pl
import pyarrow.parquet as pq
import pytest

from datalake import DatalakeIndex
from datalake.jobs import JobError, JobRunner
from datalake.layout import Layout
from tests.wrds_client.capitaliq.fakes import FakeWRDS, make_events
from wrds_client.capitaliq.keydev import (
    KEYDEV_SCHEMA,
    KeyDevJob,
    build_month,
    read_keydev,
    verify_artifact,
)

T0 = datetime(2026, 1, 2, tzinfo=timezone.utc)
START, END = date(2008, 1, 1), date(2008, 4, 30)


class Clock:
    def __init__(self, t: datetime) -> None:
        self.t = t

    def __call__(self) -> datetime:
        return self.t


def runner(index: DatalakeIndex) -> JobRunner:
    return JobRunner(index, allow_dirty=True, handle_signals=False, heartbeat_s=0.05)


@pytest.fixture
def lake(tmp_path):
    links, events = make_events(START, date(2008, 8, 31))
    wrds = FakeWRDS(links, events)
    index = DatalakeIndex(tmp_path / "lake")
    art = runner(index).start(KeyDevJob.new(Layout("M", START, END), temp=True,
                                            source=wrds.source(), clock=Clock(T0)))
    return wrds, index, art


def test_one_file_per_month_one_row_per_event_in_the_raw_layer(lake):
    wrds, index, art = lake
    assert not art.partial and art.layer == "raw"
    assert art.path.parent.parent.name == "CapIQ"
    months = sorted(p.stem for p in art.path.glob("2008-*.parquet"))
    assert months == ["2008-01", "2008-02", "2008-03", "2008-04"]
    assert wrds.calls == {"links": 4, "events": 4, "dim": 3}
    for path in art.path.glob("2008-*.parquet"):
        assert pq.read_schema(path).remove_metadata().equals(KEYDEV_SCHEMA)
    df = read_keydev(art)
    in_range = wrds.events[wrds.events["announceddate"].dt.date <= END]
    assert df.height == len(in_range) and df["keydevid"].is_unique().all()
    assert verify_artifact(index.get(art.artifact_id)) == []


def test_companies_keep_role_and_event_type_together(lake):
    wrds, _, art = lake
    df = read_keydev(art)
    multi = df.filter(pl.col("companies").list.len() == 2).row(0, named=True)
    roles = [c["objectroletype"] for c in multi["companies"]]
    assert roles == ["Target", "Buyer"]                       # ordered by role, then company
    want = wrds.links[wrds.links["keydevid"] == multi["keydevid"]]
    assert sorted(c["companyid"] for c in multi["companies"]) == sorted(
        want["companyid"].astype(int))
    assert {c["eventtype"] for c in multi["companies"]} == set(want["eventtype"])
    exploded = read_keydev(art, explode=True)
    assert exploded.height == len(wrds.links[wrds.links["announcedate"] <= END])
    assert {"companyid", "gvkey", "objectroletype"} <= set(exploded.columns)


def test_reader_filters_dates_and_point_in_time(lake):
    _, _, art = lake
    feb = read_keydev(art, start="2008-02-01", end="2008-02-29")
    assert feb["announcedate"].min() >= date(2008, 2, 1)
    assert feb["announcedate"].max() <= date(2008, 2, 29)
    cut = datetime(2008, 3, 1)
    pit = read_keydev(art, as_of=cut)
    assert pit.height and (pit["entereddateutc"] <= cut).all()
    assert pit.height < read_keydev(art).height


def test_orphans_are_counted_not_written():
    links, events = make_events(date(2008, 1, 1), date(2008, 1, 31))
    extra = links.iloc[[0]].assign(keydevid=999999.0)          # a link without its event
    table, counts = build_month(pd.concat([links, extra]), events.iloc[1:],
                                datetime(2026, 1, 1, tzinfo=timezone.utc))
    assert counts["links_without_event"] == 2                  # 999999 + the dropped event
    assert counts["events"] == table.num_rows == len(events) - 1


def test_empty_month_is_an_empty_file(tmp_path):
    links, events = make_events(date(2008, 1, 1), date(2008, 1, 31))
    wrds = FakeWRDS(links, events)
    art = runner(DatalakeIndex(tmp_path / "lake")).start(
        KeyDevJob.new(Layout("M", date(2008, 1, 1), date(2008, 2, 29)), temp=True,
                      source=wrds.source(), clock=Clock(T0)))
    assert pq.read_metadata(art.path / "2008-02.parquet").num_rows == 0


def test_update_writes_only_new_or_changed_events(lake):
    wrds, index, art = lake
    before = {p.name: p.read_bytes() for p in art.path.glob("*.parquet")}
    edited = int(wrds.events.loc[wrds.events["announceddate"].dt.month == 3,
                                 "keydevid"].iloc[0])
    wrds.events.loc[wrds.events["keydevid"] == edited, "situation"] = "Revised paragraph."
    job = KeyDevJob.for_update(index.get(art.artifact_id), index, source=wrds.source(),
                               end=date(2008, 5, 31), clock=Clock(T0 + timedelta(days=1)))
    assert [u.key for u in job.units()] == ["2008-02", "2008-03", "2008-04", "2008-05"]
    done = runner(index).update_job(art.artifact_id, job)
    assert {p.name: p.read_bytes() for p in done.path.glob("*.parquet")
            if p.name in before} == before                      # base never rewritten
    deltas = {p.stem[-7:]: pl.read_parquet(p) for p in done.path.glob("update-*.parquet")}
    assert deltas["2008-02"].height == 0 and deltas["2008-04"].height == 0
    assert deltas["2008-03"]["keydevid"].to_list() == [edited]
    may = wrds.events[wrds.events["announceddate"].dt.month == 5]
    assert deltas["2008-05"].height == len(may)
    df = read_keydev(done)
    assert df.filter(pl.col("keydevid") == edited)["situation"].item() == "Revised paragraph."
    assert df.height == len(wrds.events[wrds.events["announceddate"].dt.date <= date(2008, 5, 31)])
    assert verify_artifact(index.get(art.artifact_id)) == []


def test_update_vintage_must_move_forward(lake):
    wrds, index, art = lake
    with pytest.raises(JobError, match="later"):
        KeyDevJob.for_update(index.get(art.artifact_id), index, source=wrds.source(),
                             clock=Clock(T0))


def test_cli_refuses_another_frequency():
    from types import SimpleNamespace

    args = SimpleNamespace(partition_freq="Y", start=START, end=END, start_year=None,
                           end_year=None, temp=True)
    with pytest.raises(JobError, match="monthly"):
        KeyDevJob.from_args(args, None)
