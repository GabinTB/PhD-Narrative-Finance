"""ravenpack.annotations: every raw field into stories + story x entity tables."""
from __future__ import annotations

import io
import json
import random
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from datalake import DatalakeIndex
from datalake.jobs import JobRunner
from datalake.layout import Layout
from ravenpack.annotations import ingest as ing
from ravenpack.annotations.feeds import RPA1_ALL
from ravenpack.annotations.fields import ENTITY_KEY, EVENT_INDEX, RPA1, STORY_KEY, TIMESTAMP
from ravenpack.annotations.ingest import (
    IngestError,
    LevelError,
    RPA1IngestJob,
    ingest_month,
    typed,
    verify_entities,
    verify_headlines,
)

FIELDS = RPA1
DATE_FMT = "%Y-%m-%d %H:%M:%S"


# ---------------------------------------------------------------------------
# Synthetic raw: every field, realistic levels
# ---------------------------------------------------------------------------

def raw_month(year: int, month: int, n_stories: int = 40, seed: int = 0) -> pl.DataFrame:
    """Raw rows (all strings, the vendor's header order) of one month: stories with
    1-3 entities, some entities in several events, event indexes shuffled within a
    pair, nulls in the event fields, awkward headlines (commas, quotes, newlines)."""
    rng = random.Random(seed * 1000 + year * 12 + month)
    rows: list[dict[str, str | None]] = []
    t = datetime(year, month, 1)
    for s in range(n_stories):
        t += timedelta(seconds=rng.randint(0, 3600), microseconds=rng.choice([0, 412000]))
        sid = f"S{year}{month:02d}{s:04d}" + "A" * 20
        story = {
            "TIMESTAMP_UTC": t.strftime(DATE_FMT) + (f".{t.microsecond // 1000:03d}"),
            "NEWS_TYPE": rng.choice(["FULL-ARTICLE", "NEWS-FLASH"]),
            "RP_SOURCE_ID": f"SRC{s % 3}", "SOURCE_NAME": f"Source {s % 3}",
            "CSS": f"{rng.uniform(-1, 1):.2f}", "NIP": f"{rng.uniform(-1, 1):.2f}",
            **{c: str(rng.choice([-1, 0, 1])) for c in ("PEQ", "BEE", "BMQ", "BAM", "BCA", "BER")},
            "PRODUCT_KEY": "RPA", "PROVIDER_ID": rng.choice(["MRVR", "DJ"]),
            "PROVIDER_STORY_ID": f"P{s}",
            "HEADLINE": rng.choice([f"Plain headline {s}", f'Quoted "news", item {s}',
                                    f"Two\nlines {s}"]),
        }
        n_ent = rng.randint(1, 3)
        events = []
        for e in range(n_ent):
            entity = {"RP_ENTITY_ID": f"E{rng.randint(0, 999):06d}{e}",
                      "ENTITY_TYPE": rng.choice(["COMP", "PLCE", "ORGA"]),
                      "ENTITY_NAME": f"Entity {e}", "COUNTRY_CODE": rng.choice(["US", "GB", None]),
                      "RELEVANCE": str(rng.randint(0, 100)), "ANL_CHG": str(rng.choice([0, 1])),
                      "MCQ": str(rng.choice([-1, 0, 1]))}
            for _ in range(rng.choice([1, 1, 2, 3])):
                events.append(entity)
        rng.shuffle(events)                       # event order != entity order
        story["RP_STORY_EVENT_COUNT"] = str(len(events))
        for idx, entity in enumerate(events, 1):
            has_event = rng.random() < 0.4
            det = {c: None for c in FIELDS.detection_fields}
            det["RP_STORY_EVENT_INDEX"] = str(idx)
            if has_event:
                det.update({
                    "EVENT_SENTIMENT_SCORE": f"{rng.uniform(-1, 1):.2f}",
                    "EVENT_RELEVANCE": str(rng.randint(0, 100)),
                    "EVENT_SIMILARITY_KEY": f"K{rng.randint(0, 99)}",
                    "EVENT_SIMILARITY_DAYS": f"{rng.uniform(0, 365):.5f}",
                    "TOPIC": "business", "GROUP": "earnings", "TYPE": "revenue",
                    "CATEGORY": "revenue-up", "EVENT_TEXT": f"text, with comma {idx}",
                    "EVENT_START_DATE_UTC": (t - timedelta(days=30)).strftime(DATE_FMT),
                    "REPORTING_PERIOD": "FY-2007",
                })
            rows.append({STORY_KEY: sid, **story, **entity, **det})
    return pl.DataFrame(rows, schema={c: pl.String for c in FIELDS.raw_columns}, orient="row"
                        ).select(FIELDS.raw_columns)


def write_zip(raw_dir: Path, year: int, month: int, frame: pl.DataFrame) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    zp = RPA1_ALL.zip_path(raw_dir, year)
    member = RPA1_ALL.member(zp, year, month)
    buf = io.StringIO()
    frame.write_csv(buf, null_value="")
    with zipfile.ZipFile(zp, "a" if zp.exists() else "w") as zf:
        zf.writestr(member, buf.getvalue())


def _ingest(tmp_path: Path, frame: pl.DataFrame, block_bytes: int = ing.BLOCK_BYTES,
            tag: str = "o") -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    raw = tmp_path / f"raw_{tag}"
    write_zip(raw, 2008, 1, frame)
    hd, ed = tmp_path / f"h_{tag}", tmp_path / f"e_{tag}"
    hd.mkdir(), ed.mkdir()
    zp = RPA1_ALL.zip_path(raw, 2008)
    with zipfile.ZipFile(zp) as zf:
        ingest_month(RPA1_ALL, zf, RPA1_ALL.member(zp, 2008, 1), "2008-01", hd, ed, block_bytes)
    return (pl.read_parquet(hd / "2008-01.parquet"), pl.read_parquet(ed / "2008-01.parquet"),
            json.loads((hd / "2008-01.report.json").read_text()))


def rebuild_raw(stories: pl.DataFrame, entities: pl.DataFrame) -> pl.DataFrame:
    """Raw rows back from the two tables (explode the lists, join the stories)."""
    det = FIELDS.detection_fields
    exploded = entities.drop(TIMESTAMP).explode(det, empty_as_null=True)
    return exploded.join(stories.drop("N_ENTITIES", "N_DETECTIONS"), on=STORY_KEY,
                         how="left").select(FIELDS.raw_columns)


# ---------------------------------------------------------------------------
# Content
# ---------------------------------------------------------------------------

class TestTables:
    def test_every_raw_row_and_field_round_trips(self, tmp_path):
        raw = raw_month(2008, 1)
        stories, entities, _ = _ingest(tmp_path, raw)
        assert stories.schema == FIELDS.headlines_schema
        assert entities.schema == FIELDS.entities_schema
        want = typed(raw, FIELDS, "t").sort(STORY_KEY, ENTITY_KEY, EVENT_INDEX)
        got = rebuild_raw(stories, entities).sort(STORY_KEY, ENTITY_KEY, EVENT_INDEX)
        assert got.equals(want)

    def test_stories_keep_raw_order_and_counts(self, tmp_path):
        raw = raw_month(2008, 1)
        stories, entities, report = _ingest(tmp_path, raw)
        assert stories[STORY_KEY].to_list() == raw[STORY_KEY].unique(maintain_order=True).to_list()
        per_story = raw.group_by(STORY_KEY).agg(pl.len().alias("n"),
                                                pl.col(ENTITY_KEY).n_unique().alias("e"))
        chk = stories.join(per_story, on=STORY_KEY)
        assert (chk["N_DETECTIONS"] == chk["n"]).all() and (chk["N_ENTITIES"] == chk["e"]).all()
        assert (report["raw_rows"], report["stories"], report["pairs"]) == (
            raw.height, stories.height, entities.height)
        assert report["rows_by_provider"] == dict(raw["PROVIDER_ID"].value_counts().iter_rows())
        assert report["stories_outside_month"] == 0
        assert "Two\nlines" in "".join(stories["HEADLINE"].to_list())

    def test_detection_lists_are_aligned_and_ordered_by_event_index(self, tmp_path):
        stories, entities, _ = _ingest(tmp_path, raw_month(2008, 1))
        lens = entities.select([pl.col(c).list.len() for c in FIELDS.detection_fields])
        assert all((lens[c] == lens[EVENT_INDEX]).all() for c in lens.columns)
        assert entities[EVENT_INDEX].list.eval(pl.element().diff().drop_nulls() > 0
                                               ).list.all().all()
        assert (entities[EVENT_INDEX].list.len() > 1).any()        # multi-event pairs present
        assert entities["CATEGORY"].list.eval(pl.element().is_null()).list.any().any()

    def test_entities_keep_first_appearance_order(self, tmp_path):
        raw = raw_month(2008, 1)
        _, entities, _ = _ingest(tmp_path, raw)
        want = raw.select(STORY_KEY, ENTITY_KEY).unique(maintain_order=True)
        assert entities.select(STORY_KEY, ENTITY_KEY).equals(want)

    def test_block_boundaries_do_not_change_the_output(self, tmp_path):
        raw = raw_month(2008, 1, n_stories=80)
        big = _ingest(tmp_path, raw, tag="big")
        small = _ingest(tmp_path, raw, block_bytes=4096, tag="small")
        assert big[0].equals(small[0]) and big[1].equals(small[1]) and big[2] == small[2]


    def test_pandas_na_strings_are_null_like_the_old_ingest(self, tmp_path):
        raw = raw_month(2008, 1)
        sid = raw[STORY_KEY][0]
        raw = raw.with_columns(
            pl.when(pl.col(STORY_KEY) == sid).then(pl.lit("None")).otherwise(pl.col("HEADLINE"))
            .alias("HEADLINE"),
            pl.when(pl.col(STORY_KEY) == sid).then(pl.lit("NA")).otherwise(pl.col("COUNTRY_CODE"))
            .alias("COUNTRY_CODE"))
        stories, entities, _ = _ingest(tmp_path, raw)
        assert stories.filter(pl.col(STORY_KEY) == sid)["HEADLINE"].to_list() == [None]
        assert entities.filter(pl.col(STORY_KEY) == sid)["COUNTRY_CODE"].null_count() > 0
        assert stories["HEADLINE"].null_count() == 1


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

class TestGuards:
    def _expect(self, tmp_path, frame, err, match):
        tmp_path.mkdir(exist_ok=True)
        with pytest.raises(err, match=match):
            _ingest(tmp_path, frame)
        assert not list((tmp_path / "h_o").glob("*")) and not list((tmp_path / "e_o").glob("*"))

    def test_non_contiguous_story(self, tmp_path):
        raw = raw_month(2008, 1)
        first = raw.filter(pl.col(STORY_KEY) == raw[STORY_KEY][0])
        if first.height < 2:
            raw = pl.concat([first, raw])
        moved = pl.concat([raw.slice(1), raw.slice(0, 1)])          # first row at the end
        self._expect(tmp_path, moved, IngestError, "contiguous")

    def test_story_field_varying_inside_a_story(self, tmp_path):
        raw = raw_month(2008, 1)
        sid = raw.group_by(STORY_KEY).len().filter(pl.col("len") > 1)[STORY_KEY][0]
        idx = raw.with_row_index().filter(pl.col(STORY_KEY) == sid)["index"][-1]
        raw = raw.with_row_index().with_columns(
            pl.when(pl.col("index") == idx).then(pl.lit("other")).otherwise(pl.col("HEADLINE"))
            .alias("HEADLINE")).drop("index")
        self._expect(tmp_path, raw, LevelError, "HEADLINE")

    def test_entity_field_varying_inside_a_pair(self, tmp_path):
        raw = raw_month(2008, 1)
        pair = raw.group_by(STORY_KEY, ENTITY_KEY).len().filter(pl.col("len") > 1).row(0)
        raw = raw.with_row_index()
        in_pair = (pl.col(STORY_KEY) == pair[0]) & (pl.col(ENTITY_KEY) == pair[1])
        idx = raw.filter(in_pair)["index"][0]
        raw = raw.with_columns(pl.when(pl.col("index") == idx).then(pl.lit("1"))
                               .otherwise(pl.col("RELEVANCE")).alias("RELEVANCE"),
                               pl.when(pl.col("index") == idx).then(pl.lit("ZZ"))
                               .otherwise(pl.col("COUNTRY_CODE")).alias("COUNTRY_CODE")
                               ).drop("index")
        self._expect(tmp_path, raw, LevelError, "story x entity")

    @pytest.mark.parametrize("column,value", [("RELEVANCE", "abc"), ("RELEVANCE", "1.5"),
                                              ("RELEVANCE", "300"), ("PEQ", "2.5"),
                                              ("CSS", "x"), ("TIMESTAMP_UTC", "2008/01/01")])
    def test_unparseable_values(self, tmp_path, column, value):
        raw = raw_month(2008, 1)
        sid = raw[STORY_KEY][0]
        raw = raw.with_columns(pl.when(pl.col(STORY_KEY) == sid).then(pl.lit(value))
                               .otherwise(pl.col(column)).alias(column))
        self._expect(tmp_path, raw, IngestError, f"do not parse.*{column}")

    def test_unknown_and_missing_columns(self, tmp_path):
        raw = raw_month(2008, 1)
        self._expect(tmp_path / "a", raw.with_columns(pl.lit("x").alias("NEW_VENDOR_FIELD")),
                     IngestError, "unknown \\['NEW_VENDOR_FIELD'\\]")
        self._expect(tmp_path / "b", raw.drop("MCQ"), IngestError, "missing \\['MCQ'\\]")

    def test_duplicate_detection(self, tmp_path):
        raw = raw_month(2008, 1)
        sid = raw[STORY_KEY][0]
        first = raw.filter(pl.col(STORY_KEY) == sid)
        raw = pl.concat([first.head(1), raw])
        self._expect(tmp_path, raw, IngestError, "duplicated")


# ---------------------------------------------------------------------------
# The job: siblings, groups, crash between renames, update, verifiers
# ---------------------------------------------------------------------------

MONTHS = (1, 2, 3)
LAYOUT = Layout("M", date(2008, 1, 1), date(2008, 3, 31))


@pytest.fixture(autouse=True)
def _registered(monkeypatch):
    # resume / update look the job up by kind; independent of the installed entry points
    import datalake.jobs as jobs_mod

    monkeypatch.setitem(jobs_mod._REGISTRY, RPA1IngestJob.kind, RPA1IngestJob)


@pytest.fixture
def env(tmp_path):
    raw = tmp_path / "raw"
    for m in MONTHS:
        write_zip(raw, 2008, m, raw_month(2008, m, n_stories=15))
    index = DatalakeIndex(tmp_path / "lake")
    runner = JobRunner(index, allow_dirty=True, handle_signals=False, log_file=False)
    yield raw, index, runner
    index.close()


def _sibling(index, art):
    (eid,) = [c for c in index.children(art.artifact_id) if c.startswith("rp_headline_entities")]
    return index.get(eid)


class TestJob:
    def test_siblings_under_the_feed_group(self, env):
        raw, index, runner = env
        art = runner.start(RPA1IngestJob(raw, LAYOUT, temp=True))
        ent = _sibling(index, art)
        assert not art.partial and not ent.partial
        assert art.path.parent.parent.name == "RavenPack" == ent.path.parent.parent.name
        assert art.meta.hyperparams["feed"] == "rpa1_all_entities"
        assert ent.meta.hyperparams["headlines_id"] == art.artifact_id
        assert ent.meta.sources == [art.artifact_id]
        assert sorted(art.file_hashes) == sorted(
            [f"2008-0{m}.parquet" for m in MONTHS] + [f"2008-0{m}.report.json" for m in MONTHS])
        assert sorted(ent.file_hashes) == [f"2008-0{m}.parquet" for m in MONTHS]
        assert verify_headlines(art) == [] and verify_entities(ent) == []

    def test_crash_between_renames_reruns_the_month(self, env, monkeypatch, tmp_path):
        raw, index, runner = env
        ref = runner.start(RPA1IngestJob(raw, LAYOUT, temp=True))
        ref_files = {p.name: p.read_bytes() for p in ref.path.glob("2008-*")}

        lake2 = DatalakeIndex(tmp_path / "lake2")
        runner2 = JobRunner(lake2, allow_dirty=True, handle_signals=False, log_file=False)
        real = ing.MonthWriter.commit
        calls = [0]

        def crashing(self, *a):
            calls[0] += 1
            if calls[0] == 2:                    # second month: entities renamed, then dies
                for w in self.writers:
                    w.close()
                self.tmp[0].replace(self.final[0])
                raise RuntimeError("killed after the entities rename")
            return real(self, *a)

        monkeypatch.setattr(ing.MonthWriter, "commit", crashing)
        with pytest.raises(RuntimeError):
            runner2.start(RPA1IngestJob(raw, LAYOUT, temp=True))
        monkeypatch.setattr(ing.MonthWriter, "commit", real)
        part = lake2.list("rp_headlines", include_partial=True)[0]
        assert not (part.path / "2008-02.parquet").exists()
        assert (_sibling(lake2, part).path / "2008-02.parquet").exists()
        done = runner2.resume(part.artifact_id, raw_dir=raw)
        assert {p.name: p.read_bytes() for p in done.path.glob("2008-*")} == ref_files
        assert verify_entities(_sibling(lake2, done)) == []
        lake2.close()

    def test_update_adds_new_months_only(self, env):
        raw, index, runner = env
        art = runner.start(RPA1IngestJob(raw, LAYOUT, temp=True))
        before = {p.name: p.read_bytes() for p in art.path.glob("2008-*")}
        write_zip(raw, 2008, 4, raw_month(2008, 4, n_stories=10))
        done = runner.update(art.artifact_id, raw_dir=raw)
        after = {p.name: p.read_bytes() for p in done.path.glob("2008-*")}
        assert {k: after[k] for k in before} == before
        assert set(after) - set(before) == {"2008-04.parquet", "2008-04.report.json"}
        ent = _sibling(index, done)
        assert len(done.meta.runs) == 2 and len(ent.meta.runs) == 2 and not ent.partial
        assert (ent.path / "2008-04.parquet").exists()
        assert verify_headlines(done) == [] and verify_entities(ent) == []

    def test_missing_raw_month_is_skipped_and_reported(self, env, tmp_path):
        raw, index, runner = env
        art = runner.start(RPA1IngestJob(raw, Layout("M", date(2008, 1, 1), date(2008, 5, 31)),
                                         temp=True))
        assert sorted(p.stem for p in art.path.glob("*.parquet")) == ["2008-01", "2008-02",
                                                                       "2008-03"]

    def test_only_monthly_layouts(self, env):
        raw, _, _ = env
        with pytest.raises(ValueError, match="must be M"):
            RPA1IngestJob(raw, Layout("D", date(2008, 1, 1), date(2008, 1, 31)))

    def test_verifier_flags_misaligned_lists(self, env):
        raw, index, runner = env
        art = runner.start(RPA1IngestJob(raw, LAYOUT, temp=True))
        ent = _sibling(index, art)
        path = ent.path / "2008-02.parquet"
        frame = pl.read_parquet(path)
        frame.with_columns(pl.col("TYPE").list.head(0)).write_parquet(path)
        assert any("misaligned" in f.message for f in verify_entities(ent))
