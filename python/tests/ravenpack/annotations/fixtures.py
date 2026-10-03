"""Test helper: register an ``rp_headlines`` artifact and its ``rp_headline_entities``
sibling from small frames, schema-exact (``ravenpack.annotations.fields.RPA1``) and
laid out as the ingestion lays them out (feed hyperparams, entities sourced on the
headlines, group RavenPack).

    hl, ent = register_annotations(index, {"2008-01": stories}, {"2008-01": entities})

``stories``: RP_STORY_ID, TIMESTAMP_UTC (datetime or ISO text, read as UTC), optional
HEADLINE / PROVIDER_ID; every other story field is null. ``entities``: RP_STORY_ID,
RP_ENTITY_ID, optional RELEVANCE (default 100); one row per story x entity. Without
entities, each story gets one entity outside any universe (every real story has one).
"""
from __future__ import annotations

import json
from datetime import datetime

import polars as pl

from datalake import Artifact, DatalakeIndex
from datalake.layout import Layout
from datalake.periods import parse_key
from ravenpack.annotations.feeds import RPA1_ALL
from ravenpack.annotations.fields import ENTITY_KEY, RPA1, STORY_KEY, TIMESTAMP

UTC = pl.Datetime("us", "UTC")


def _utc(col: pl.Series) -> pl.Series:
    if col.dtype == pl.String:
        return col.str.to_datetime("%Y-%m-%d %H:%M:%S%.f", time_unit="us", time_zone="UTC",
                                   strict=False).fill_null(
            col.str.to_datetime("%Y-%m-%d", time_unit="us", time_zone="UTC", strict=False))
    if isinstance(col.dtype, pl.Datetime) and col.dtype.time_zone is None:
        return col.dt.replace_time_zone("UTC").cast(UTC)
    return col.cast(UTC)


def headlines_frame(stories: pl.DataFrame, entities: pl.DataFrame) -> pl.DataFrame:
    schema = RPA1.headlines_schema
    s = stories.with_columns(_utc(stories[TIMESTAMP]).alias(TIMESTAMP))
    counts = entities.group_by(STORY_KEY).agg(pl.len().cast(pl.Int32).alias("N_ENTITIES"))
    s = s.join(counts, on=STORY_KEY, how="left").with_columns(
        pl.col("N_ENTITIES").fill_null(0), pl.col("N_ENTITIES").fill_null(0).alias("N_DETECTIONS"))
    return s.select([pl.col(c).cast(t) if c in s.columns else pl.lit(None, dtype=t).alias(c)
                     for c, t in schema.items()])


def entities_frame(stories: pl.DataFrame, entities: pl.DataFrame) -> pl.DataFrame:
    schema = RPA1.entities_schema
    ts = stories.select(STORY_KEY, _utc(stories[TIMESTAMP]).alias(TIMESTAMP))
    e = entities.join(ts, on=STORY_KEY, how="left")
    if "RELEVANCE" not in e.columns:
        e = e.with_columns(pl.lit(100).alias("RELEVANCE"))
    cols = []
    for c, t in schema.items():
        if c in e.columns:
            cols.append(pl.col(c).cast(t))
        elif c == "RP_STORY_EVENT_INDEX":
            cols.append(pl.lit([1], dtype=t).alias(c))          # one detection, no event
        elif isinstance(t, pl.List):
            cols.append(pl.lit([None], dtype=t).alias(c))
        else:
            cols.append(pl.lit(None, dtype=t).alias(c))
    return e.select(cols)


def register_annotations(index: DatalakeIndex, stories: dict[str, pl.DataFrame],
                         entities: dict[str, pl.DataFrame] | None = None, *,
                         layout: Layout | None = None,
                         temp: bool = True) -> tuple[Artifact, Artifact]:
    keys = sorted(stories)
    if layout is None:
        first, last = parse_key(keys[0]), parse_key(keys[-1])
        layout = Layout(first.freq, first.first, last.last)
    if entities is None:
        entities = {k: pl.DataFrame({STORY_KEY: f[STORY_KEY],
                                     ENTITY_KEY: [f"X{i}" for i in range(f.height)]})
                    for k, f in stories.items()}
    version = "v1.0.0" + ("__TEMP" if temp else "")
    hp = {"feed": RPA1_ALL.name, **layout.hyperparams()}
    with index.run(kind=RPA1_ALL.headlines_kind, pipeline="t", pipeline_version=version,
                   hyperparams=hp, group=RPA1_ALL.group) as run:
        for k in keys:
            frame = headlines_frame(stories[k], entities[k])
            frame.write_parquet(run.out_dir / f"{k}.parquet")
            report = {"month": k, "feed": RPA1_ALL.name, "stories": frame.height,
                      "raw_rows": int(frame["N_DETECTIONS"].sum()),
                      "pairs": int(frame["N_ENTITIES"].sum())}
            (run.out_dir / f"{k}.report.json").write_text(json.dumps(report))
    hl = index.get(run.artifact_id)
    with index.run(kind=RPA1_ALL.entities_kind, pipeline="t", pipeline_version=version,
                   hyperparams={**hp, "headlines_id": hl.artifact_id}, sources=[hl],
                   group=RPA1_ALL.group) as run:
        for k in keys:
            entities_frame(stories[k], entities[k]).write_parquet(run.out_dir / f"{k}.parquet")
    return hl, index.get(run.artifact_id)


def stories_frame(ids: list[str], times: list[datetime], headlines: list[str] | None = None,
                  ) -> pl.DataFrame:
    data = {STORY_KEY: ids, TIMESTAMP: times}
    if headlines is not None:
        data["HEADLINE"] = headlines
    return pl.DataFrame(data)
