"""wrds_client.capitaliq.keydev: monthly parquets of event versions (fake WRDS, no network)."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pandas as pd
import polars as pl
import pyarrow.parquet as pq
import pytest

from datalake import DatalakeIndex
from datalake.jobs import JobError, JobRunner
from datalake.layout import Layout
from tests.wrds_client.capitaliq.fakes import HISTORY_START, FakeWRDS, make_events
from wrds_client.capitaliq.keydev import (
    KEYDEV_SCHEMA,
    KeyDevJob,
    build_versions,
    read_keydev,
    verify_artifact,
)

T0 = datetime(2026, 1, 2, tzinfo=timezone.utc)
START, END = date(2008, 1, 1), date(2008, 4, 30)
REDATED_AT = datetime(2024, 3, 15, 2, 15, 13)


class Clock:
    def __init__(self, t: datetime) -> None:
        self.t = t

    def __call__(self) -> datetime:
        return self.t


def runner(index: DatalakeIndex) -> JobRunner:
    return JobRunner(index, allow_dirty=True, handle_signals=False, heartbeat_s=0.05)


def first_id_in(wrds: FakeWRDS, month: int) -> int:
    ev = wrds.events
    return int(ev.loc[ev["announceddate"].dt.month == month, "keydevid"].iloc[0])


@pytest.fixture
def wrds():
    """Jan-Apr 2008, plus one February event that CIQ re-dated to March (the Audacy case)."""
    fake = FakeWRDS(*make_events(START, END))
    fake.redated = first_id_in(fake, 2)
    fake.revise(fake.redated, REDATED_AT, announceddate=datetime(2008, 3, 14, 22, 14))
    return fake


@pytest.fixture
def lake(tmp_path, wrds):
    index = DatalakeIndex(tmp_path / "lake")
    art = runner(index).start(KeyDevJob.new(Layout("M", START, END), temp=True,
                                            source=wrds.source(), clock=Clock(T0)))
    return wrds, index, art


def test_one_file_per_month_in_the_raw_layer(lake):
    wrds, index, art = lake
    assert not art.partial and art.layer == "raw"
    assert art.path.parent.parent.name == "CapIQ"
    months = sorted(p.stem for p in art.path.glob("2008-*.parquet"))
    assert months == ["2008-01", "2008-02", "2008-03", "2008-04"]
    assert wrds.calls == {"links": 4, "events": 4, "dim": 3}
    for path in art.path.glob("2008-*.parquet"):
        assert pq.read_schema(path).remove_metadata().equals(KEYDEV_SCHEMA)
    assert verify_artifact(index.get(art.artifact_id)) == []


def test_current_view_has_one_row_per_event_versions_view_all(lake):
    wrds, _, art = lake
    current = read_keydev(art)
    assert current["keydevid"].is_unique().all()
    assert current.height == wrds.events["keydevid"].nunique()
    every = read_keydev(art, versions=True)
    assert every.height == len(wrds.events)               # one row per event version


def test_redated_event_keeps_its_history(lake):
    wrds, _, art = lake
    kid = wrds.redated
    rows = read_keydev(art, versions=True).filter(pl.col("keydevid") == kid).sort(
        "speffectivedate")
    assert rows["announcedate"].to_list() == [rows["announcedate"][0], date(2008, 3, 14)]
    assert rows["announcedate"][0].month == 2
    assert rows["first_version"].to_list() == [True, False]
    assert rows["sptodate"][0] == REDATED_AT - timedelta(seconds=1)
    assert rows["sptodate"][1] is None
    feb = pl.read_parquet(art.path / "2008-02.parquet")
    mar = pl.read_parquet(art.path / "2008-03.parquet")
    assert kid in feb["keydevid"].to_list() and kid in mar["keydevid"].to_list()
    now = read_keydev(art).filter(pl.col("keydevid") == kid)
    assert now["announcedate"].item() == date(2008, 3, 14)
    then = read_keydev(art, as_of="2020-01-01").filter(pl.col("keydevid") == kid)
    assert then["announcedate"].item().month == 2          # what CIQ showed before 2024


def test_as_of_before_the_history_uses_the_oldest_version_and_entry_date(lake):
    _, _, art = lake
    cut = datetime(2008, 3, 1)
    assert cut < HISTORY_START
    pit = read_keydev(art, as_of=cut)
    assert pit.height and (pit["entereddateutc"] <= cut).all()
    assert pit["first_version"].all()
    assert pit.height < read_keydev(art).height


def test_companies_keep_role_event_type_and_validity_together(lake):
    wrds, _, art = lake
    df = read_keydev(art)
    multi = df.filter(pl.col("companies").list.len() == 2).row(0, named=True)
    assert [c["objectroletype"] for c in multi["companies"]] == ["Target", "Buyer"]
    assert all(c["sptodate"] is None for c in multi["companies"])
    exploded = read_keydev(art, explode=True)
    assert {"company_companyid", "company_gvkey", "company_objectroletype"} <= set(
        exploded.columns)
    cur = wrds.links[wrds.links["sptodate"].isna()]
    assert exploded.height == len(cur)


def test_a_link_only_attaches_to_the_versions_it_overlaps():
    links, events = make_events(date(2008, 1, 1), date(2008, 1, 5), per_day=1)
    kid = events["keydevid"].iloc[0]
    later = datetime(2023, 4, 26, 13, 19, 29)
    extra = links[links["keydevid"] == kid].assign(companyid=999.0, gvkey="000999",
                                                   speffectivedate=later)
    fake = FakeWRDS(pd.concat([links, extra], ignore_index=True), events)
    fake.revise(int(kid), datetime(2024, 1, 1))
    lk, ev = fake.source().by_ids([int(kid)])
    table, counts = build_versions(lk, ev, T0)
    rows = pl.from_arrow(table).sort("speffectivedate")
    ids = [sorted(c["companyid"] for c in comps) for comps in rows["companies"].to_list()]
    assert ids == [[int(links.loc[links["keydevid"] == kid, "companyid"].iloc[0]), 999]] * 2
    old = [c for c in rows["companies"][0] if c["companyid"] == 999][0]
    assert old["speffectivedate"] == later and not old["first_version"]
    assert counts["link_versions_unmatched"] == 0


def test_update_writes_new_and_changed_versions_only(lake):
    wrds, index, art = lake
    before = {p.name: p.read_bytes() for p in art.path.glob("*.parquet")}
    t1 = datetime(2026, 1, 3, 8, 0)
    edited = first_id_in(wrds, 4)
    wrds.revise(edited, t1, situation="Revised paragraph.")
    wrds.add(*make_events(date(2008, 5, 1), date(2008, 5, 10), first_id=5000), at=t1)
    job = KeyDevJob.for_update(index.get(art.artifact_id), index, source=wrds.source(),
                               clock=Clock(T0 + timedelta(days=1)))
    assert job.units()[0].key == "changes"
    done = runner(index).update_job(art.artifact_id, job)
    assert {p.name: p.read_bytes() for p in done.path.glob("*.parquet")
            if p.name in before} == before                  # base never rewritten
    delta = pl.read_parquet(next(done.path.glob("update-*.parquet")))
    new_ids = set(wrds.events.loc[wrds.events["keydevid"] >= 5000, "keydevid"].astype(int))
    assert set(delta["keydevid"].to_list()) == new_ids | {edited}
    assert delta.filter(pl.col("keydevid") == edited).height == 2   # closed + new version
    now = read_keydev(done).filter(pl.col("keydevid") == edited)
    assert now["situation"].item() == "Revised paragraph."
    assert read_keydev(done).height == wrds.events["keydevid"].nunique()
    assert verify_artifact(index.get(art.artifact_id)) == []


def test_update_without_changes_writes_an_empty_delta(lake):
    wrds, index, art = lake
    done = runner(index).update_job(art.artifact_id, KeyDevJob.for_update(
        index.get(art.artifact_id), index, source=wrds.source(),
        clock=Clock(T0 + timedelta(days=1))))
    assert pq.read_metadata(next(done.path.glob("update-*.parquet"))).num_rows == 0


def test_update_vintage_must_move_forward(lake):
    wrds, index, art = lake
    with pytest.raises(JobError, match="later"):
        KeyDevJob.for_update(index.get(art.artifact_id), index, source=wrds.source(),
                             clock=Clock(T0))


def test_empty_month_is_an_empty_file(tmp_path):
    wrds = FakeWRDS(*make_events(date(2008, 1, 1), date(2008, 1, 31)))
    art = runner(DatalakeIndex(tmp_path / "lake")).start(
        KeyDevJob.new(Layout("M", date(2008, 1, 1), date(2008, 2, 29)), temp=True,
                      source=wrds.source(), clock=Clock(T0)))
    assert pq.read_metadata(art.path / "2008-02.parquet").num_rows == 0


def test_cli_refuses_another_frequency():
    from types import SimpleNamespace

    args = SimpleNamespace(partition_freq="Y", start=START, end=END, start_year=None,
                           end_year=None, temp=True)
    with pytest.raises(JobError, match="monthly"):
        KeyDevJob.from_args(args, None)
