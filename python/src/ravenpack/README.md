## Data access

This module processes RavenPack Annotations 1.0 data. The data itself is 
proprietary and requires a RavenPack license. No data is included in this 
repository. The ingestion pipeline expects raw zip files obtained directly 
from RavenPack.
## Annotations ingestion (`ravenpack.annotations`)

`jobs start rp_headlines --start 2000-01-01 --end 2025-12-31` reads the raw zips
(`$RAW_DATA_PATH/RavenPack/headlines_edge_v1.0`, one CSV per month) once and writes two
sibling artifacts under `derived/RavenPack/`, one parquet per month each:

| artifact | one row per | columns |
|---|---|---|
| `rp_headlines` | story | `RP_STORY_ID`, the story-level fields, `N_ENTITIES`, `N_DETECTIONS`; plus `{YYYY-MM}.report.json` (counts, providers) |
| `rp_headline_entities` | story x entity | `RP_STORY_ID`, `RP_ENTITY_ID`, `TIMESTAMP_UTC`, the entity-level fields, every detection-level field as a list aligned across fields and ordered by `RP_STORY_EVENT_INDEX` |

Every raw field is kept, typed, at its level (`annotations/fields.py`):

- **story**: TIMESTAMP_UTC (UTC, microseconds), HEADLINE, NEWS_TYPE, RP_SOURCE_ID,
  SOURCE_NAME, PRODUCT_KEY, PROVIDER_ID, PROVIDER_STORY_ID, RP_STORY_EVENT_COUNT, CSS, NIP,
  PEQ, BEE, BMQ, BAM, BCA, BER;
- **story x entity**: ENTITY_TYPE, ENTITY_NAME, COUNTRY_CODE, RELEVANCE, ANL_CHG, MCQ;
- **detection** (one per raw row, mostly null when the row carries no event): the event
  index, sentiment, relevance, similarity, taxonomy (TOPIC / GROUP / TYPE / SUB_TYPE /
  PROPERTY / CATEGORY), dates, position, reporting period, related entity, EVENT_TEXT.

`COUNTRY_CODE` is per story x entity, not per story (the older `ravenpack_headlines`
kept the first value of each story). Empty strings and pandas' default NA strings
(`NA`, `None`, `nan`, ...) read as null, as in `ravenpack_headlines`: Namibia's
country code `NA` is therefore null.

Exploding the lists and joining the stories gives back every raw row exactly. A month
is refused, nothing written, when:

- a raw column is unknown or missing;
- a value does not parse to its type;
- a field breaks its level;
- a story's rows are not contiguous;
- a (story, entity, event index) repeats.

The product mixes providers (`PROVIDER_ID`: MRVR = web, DJ = Dow Jones, ...). They stay
a column; select a feed downstream. Another Annotations product in the same CSV layout
is a new `FeedSpec` (`annotations/feeds.py`) plus a three-line `AnnotationsIngestJob`
subclass and an entry point. `jobs update <id>` adds the raw months after the last one.
