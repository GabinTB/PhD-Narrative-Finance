"""Feeds: where a RavenPack Annotations product lies on disk and what it becomes.

A feed is data, not code: another Annotations 1.0 edition (or another vendor
delivery in the same CSV layout) is one more ``FeedSpec``; a product with another
schema brings its own ``FieldSet``. Providers inside a product (web, Dow Jones,
...) are not feeds: they stay a column (PROVIDER_ID), filtered downstream.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from ravenpack.annotations.fields import RPA1, FieldSet


@dataclass(frozen=True)
class FeedSpec:
    name: str                     # recorded in the artifact's hyperparams (identity)
    raw_subdir: tuple[str, ...]   # under $RAW_DATA_PATH
    zip_name: str                 # .format(year=)
    member_name: str              # .format(stem=, year=, month=)
    fields: FieldSet
    headlines_kind: str
    entities_kind: str
    group: str

    def raw_dir(self) -> Path:
        root = os.environ.get("RAW_DATA_PATH")
        if not root:
            raise ValueError("RAW_DATA_PATH is not set; pass the raw directory explicitly")
        return Path(root).joinpath(*self.raw_subdir)

    def zip_path(self, raw_dir: Path, year: int) -> Path:
        return Path(raw_dir) / self.zip_name.format(year=year)

    def member(self, zip_path: Path, year: int, month: int) -> str:
        return self.member_name.format(stem=zip_path.stem, year=year, month=month)


RPA1_ALL = FeedSpec(
    name="rpa1_all_entities",
    raw_subdir=("RavenPack", "headlines_edge_v1.0"),
    zip_name="RavenPackAnalytics_AllEntities_1.0_{year}.zip",
    member_name="{stem}/{year}-{month:02d}.csv",
    fields=RPA1,
    headlines_kind="rp_headlines",
    entities_kind="rp_headline_entities",
    group="RavenPack",
)

FEEDS: dict[str, FeedSpec] = {f.name: f for f in (RPA1_ALL,)}


def feed(name: str) -> FeedSpec:
    if name not in FEEDS:
        raise ValueError(f"unknown feed {name!r} (known: {sorted(FEEDS)})")
    return FEEDS[name]


__all__ = ["FeedSpec", "RPA1_ALL", "FEEDS", "feed"]
