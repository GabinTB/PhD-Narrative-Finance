# wrds_client.capitaliq

- `mapping.py`: CapitalIQ identifier and GICS backfill for universe rows (see its docstring).
- `keydev.py`: the `ciq_keydev` job, Capital IQ Key Developments as one Parquet file per month.

## ciq_keydev

Each month is pulled in two steps, because only one WRDS table is indexed by date (checked with `EXPLAIN`):

1. Event ids and company links come from `ciq_keydev.wrds_keydev` through its `announcedate` index. This table has one row per event × company × role.
2. Headline, situation and dates come from `ciq_keydev.ciqkeydev`, one row per event, matched by `keydevid` through its index. Each event's text is read once. That table has no date index.

```bash
.venv/bin/jobs start ciq_keydev --start 2000-01-01 --end 2026-09-30 [--temp]
.venv/bin/jobs update <ciq_keydev id>     # re-pulls the last 3 months, adds new months
.venv/bin/jobs resume <id>
```

The job writes to `Datalake/raw/CapIQ/ciq_keydev/<id>/`:

| file | content |
|---|---|
| `YYYY-MM.parquet` | one row per event announced in the month (`KEYDEV_SCHEMA`): text, every CIQ date (local and UTC), source type, and `companies`. `companies` is a list of `{companyid, gvkey, companyname, keydeveventtypeid, eventtype, keydevtoobjectroletypeid, objectroletype}`, sorted by role then company. |
| `update-<vintage>-YYYY-MM.parquet` | events of an update that are new or changed (`row_sha256`). The file is empty when nothing changed. |
| `dim-<table>.parquet` | lookups: event type → category, role, time zone |
| `plan-<vintage>.json` | what an execution pulls: mode and month range |

The event type sits in the company struct because CIQ stores it per event–company link. CIQ gives no relevance score per company.

```python
from wrds_client.capitaliq import read_keydev

events = read_keydev(art, start="2024-01-01", end="2024-12-31")       # one row per event
links = read_keydev(art, start="2024-01-01", explode=True)             # one row per event x company
pit = read_keydev(art, as_of="2015-06-30")                             # entered by CIQ by then
```

Point-in-time: files are partitioned by the announcement date, but CIQ back-filled its history: `entereddateutc` can be years later than the announcement. Filter on it (`as_of`) for availability.

An update does not pick up events that CIQ deletes or back-fills into older months. A rebuild does.
