# wrds_client.capitaliq

- `mapping.py`: CapitalIQ identifier and GICS backfill for universe rows (see its docstring).
- `keydev.py`: the `ciq_keydev` job, Capital IQ Key Developments as monthly Parquet files with one row per event **version**.

## ciq_keydev

CIQ keeps the history of each event as validity periods: a version is valid from `speffectivedate` until `sptodate`, and `sptodate` is empty while it's current. When CIQ edits an event, re-dates it, or reuses its `keydevid` for another event, it closes the current version and opens a new one. Two examples:

- On 2024-03-15, Audacy shelf registrations from 2017–2021 were re-dated to 2024-03-14.
- `keydevid` 1 was an Advent rumour (2003), then an International Star deal, and is now a Turpaz acquisition.

This history only starts in 2018-04, the first `speffectivedate` in WRDS. So the oldest version of each event (`first_version`) counts as valid from the beginning.

Each month is pulled through indexes only (plans checked with `EXPLAIN`):

1. The month's event ids come from `ciq_keydev.wrds_keydev`, through its `announcedate` index.
2. Every version of those events and of their company links is fetched by `keydevid`: links from `wrds_keydev`, text from `ciqkeydev`. A window function over all versions finds each event's oldest one.
3. The versions announced in the month are written. Each event version carries the company link versions that overlap its validity.

```bash
.venv/bin/jobs start ciq_keydev --start 2000-01-01 --end 2026-09-30 [--temp]
.venv/bin/jobs update <ciq_keydev id>     # events changed since the newest version stored
.venv/bin/jobs resume <id>
```

The job writes to `Datalake/raw/CapIQ/ciq_keydev/<id>/`:

| file | content |
|---|---|
| `YYYY-MM.parquet` | one row per event version announced in the month (`KEYDEV_SCHEMA`), keyed by (`keydevid`, `speffectivedate`). Each row has the validity (`speffectivedate`, `sptodate`, `first_version`), text, every CIQ date (local and UTC), the source type, and `companies`. `companies` is a list of `{companyid, gvkey, companyname, keydeveventtypeid, eventtype, keydevtoobjectroletypeid, objectroletype, speffectivedate, sptodate, first_version}`, sorted by role then company. |
| `update-<vintage>.parquet` | the versions an update found new or changed (`row_sha256`; a version that was closed counts as changed). The file is empty when nothing changed. |
| `dim-<table>.parquet` | lookups: event type → category, role, time zone |
| `plan-<vintage>.json` | what an execution pulls: its mode; for an update, the `since` timestamp |

How an update works: it finds the events with a link version newer than the latest `speffectivedate` already stored (indexed), then re-fetches all their versions by `keydevid`. Base files are never rewritten.

```python
from wrds_client.capitaliq import read_keydev

now = read_keydev(art, start="2024-01-01", end="2024-12-31")   # current versions, one row per event
pit = read_keydev(art, as_of="2015-06-30")                      # what CIQ showed then
every = read_keydev(art, versions=True)                         # full history
links = read_keydev(art, explode=True)                          # one row per event x company
```

What `as_of=t` keeps:

- the event versions valid at `t` (the oldest counts from the beginning) and entered by `t` (`entereddateutc`);
- within those, the company links valid at `t`. A link CIQ recorded after the event's oldest version (a company added later) only counts from its own `speffectivedate`.

Validity timestamps are naive, in CIQ's database time.
