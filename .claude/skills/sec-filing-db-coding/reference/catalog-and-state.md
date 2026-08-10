# Catalog, EDGAR state, and point-in-time queries

## Why SQLite

The catalog is small, written concurrently during download, must be crash-safe, and must travel
as a single file for the state export. That is SQLite's exact profile. A server database adds
operational burden and, fatally for the export requirement, does not move as a file.

Failure then reduces to "a file is missing", which content addressing detects and a re-fetch
heals.

## Schema

```sql
CREATE TABLE schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);  -- includes ('schema_version', '1')

-- Mirror of EDGAR's index. All filings EDGAR knows about, metadata only.
-- Append-only: one row per (accession, seen_at) observation, never upsert in place.
CREATE TABLE edgar_state (
    accession         TEXT NOT NULL,
    cik               INTEGER NOT NULL,
    form              TEXT NOT NULL,
    filing_date       DATE NOT NULL,
    period_of_report  DATE,
    primary_doc_url   TEXT,
    is_amendment      BOOLEAN NOT NULL,
    seen_at           TIMESTAMP NOT NULL,
    PRIMARY KEY (accession, seen_at)
);
CREATE INDEX idx_state_cik_form ON edgar_state (cik, form, filing_date);

-- What we actually hold.
CREATE TABLE filings (
    accession         TEXT PRIMARY KEY,
    cik               INTEGER NOT NULL,
    form              TEXT NOT NULL,
    filing_date       DATE NOT NULL,
    period_of_report  DATE,
    filer_category    TEXT,          -- dei:EntityFilerCategory, drives absent_by_rule
    is_amendment      BOOLEAN NOT NULL,
    raw_sha           TEXT,          -- NULL until fetched
    status            TEXT NOT NULL, -- see status values below
    received_at       TIMESTAMP,     -- when WE first stored it
    last_attempt_at   TIMESTAMP,
    attempt_count     INTEGER NOT NULL DEFAULT 0,
    error             TEXT
);
CREATE INDEX idx_filings_cik_form_date ON filings (cik, form, filing_date);
CREATE INDEX idx_filings_status ON filings (status);

-- Derived cache warmth. Not a first-class object.
CREATE TABLE derived (
    raw_sha     TEXT NOT NULL,
    preproc_id  TEXT NOT NULL,
    derived_sha TEXT NOT NULL,
    created_at  TIMESTAMP NOT NULL,
    PRIMARY KEY (raw_sha, preproc_id)
);

-- Company name resolution. CIK is canonical; names change on M&A.
CREATE TABLE companies (
    cik        INTEGER NOT NULL,
    name       TEXT NOT NULL,
    valid_from DATE,
    valid_to   DATE,
    PRIMARY KEY (cik, name, valid_from)
);
CREATE INDEX idx_companies_name ON companies (name);
```

### Status values

Three states, not two. Collapsing them means permanently-absent filings are re-attempted on
every run, forever.

| Status | Meaning |
|---|---|
| `pending` | known from `edgar_state`, never fetched |
| `stored` | fetched, blob present, `raw_sha` set |
| `failed` | fetch attempted and failed; `error` and `attempt_count` populated |
| `not_available` | EDGAR does not serve it (withdrawn, redacted); do not retry |

The missing-data diff reads `status`, not row presence.

## The `received_at` decision

One timestamp only: when your system first stored the filing. It answers "what did I hold as of
date X", which is what the extract manifest's source state reference needs.

The SEC-side dissemination axis was considered and dropped. There is no published feed of SEC
staff corrections or deletions (they are handled case by case over email), so the axis cannot be
backfilled and cannot be sourced externally. Do not add it.

## EDGAR state table

### Why it exists

With a mirror of EDGAR's index, "what am I missing for this CIK" is a join:

```sql
SELECT s.accession
FROM edgar_state s
LEFT JOIN filings f USING (accession)
WHERE s.cik = ?
  AND s.form IN (...)
  AND (f.accession IS NULL OR f.status IN ('pending', 'failed'));
```

Without it, the same question requires either a live EDGAR query per CIK or a filesystem walk.

### Daily scheduler

Pull EDGAR's **index feeds**, never the filings:

- `https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{q}/master.idx` for the quarterly
  index (backfill and reconciliation)
- `https://data.sec.gov/submissions/CIK##########.json` for per-firm truth. Note this caps at
  roughly 1,000 recent filings; the additional shards listed under `filings.files` must also be
  fetched or high-volume filers silently lose early years.

Insert one row per observation with `seen_at = now`. Never upsert.

Storage cost is negligible: index metadata is bytes per row, so the full EDGAR index is
megabytes to low GB even though the document corpus is over 1,000 TB.

### EDGAR's mutation semantics

Verified behaviour, and the reason append-only is safe:

- Amendments never touch the original. A correction is a **new filing** with a `/A` suffix and
  its own fresh accession number. "The amendment becomes the authoritative record" is a
  disclosure statement, not a data operation; the original accession remains publicly
  retrievable and unmodified.
- Accession numbers are assigned automatically, are unique, and are never reused or corrected.
- The only genuine mutation is SEC staff deletion or redaction of an existing filing, which the
  SEC does only in rare and unusual circumstances.

So: new accessions only, originals immutable, daily index diff catches everything.

**Disappearance handling.** An accession present in yesterday's observation and absent today is
the redaction case. Log it as an anomaly, set the filing's status to `not_available`, and do not
delete the historical `edgar_state` rows. This is rare; the point is having a record that
EDGAR's own view changed, not building machinery around it.

## Company name resolution

CIK is the canonical, permanent key. Store it on every filing.

Names are a convenience lookup and are time-varying (M&A, rebranding), so they carry validity
intervals. Resolve a name **as of the filing date**, not as of today, or a firm's pre-merger
filings get attributed to the wrong entity.

Tickers are deliberately out of scope. If they are ever added, the same bitemporal treatment
applies and is more urgent, since tickers are reassigned between unrelated companies.

## Point-in-time queries

Three lookups. The naming is load-bearing.

```python
get_specific(accession) -> Filing
```
Primary-key lookup. Returns the filing plus metadata (CIK, company name as of filing date, form,
filing_date, period_of_report, filer_category, is_amendment).

```python
get_latest_as_of(cik, date, form="10-K") -> Filing | None
```
The latest filing with `filing_date <= date`. **Point-in-time.** This is what a reader could
actually have seen at that date, and the only safe basis for signal construction.

```python
get_covering_period(cik, date, form="10-K") -> Filing | None
```
The filing whose `period_of_report` interval contains `date`. **Look-ahead**: at
`date = 2015-06-01` this returns the FY2015 10-K, a document that did not exist until 2016.
Legitimate for descriptive work, never for backtests.

### Why they must stay separate

The two answer different questions and return different documents. At `t = 2016-01-15`,
point-in-time returns the FY2014 10-K (FY2015 not yet filed); period-covering returns FY2016.
A user who assumes the wrong one gets look-ahead bias with no error raised. Do not provide a
single `get_doc_at` that picks a default.

### Amendments compose for free

If a FY2015 10-K is amended by a `10-K/A` filed 2016-08-01, then `get_latest_as_of(cik,
2016-09-01)` returns the amendment and `get_latest_as_of(cik, 2016-06-01)` returns the original,
automatically, because the `/A` is just another filing with its own `filing_date`. No
special-casing. This is a further reason point-in-time is the correct default: under
period-covering semantics the version choice has to be resolved by hand.

### Validity intervals are derived, not stored

Do not persist a `valid_from`/`valid_to` per filing. The interval is a view computed from
`filing_date` ordering within `(cik, form)` at query time. Storing it goes stale the moment a new
filing or amendment arrives.

## State export and rebuild

```python
state = store.export_state()          # catalog only, no blobs, no parquet
Store.rebuild_db_from_state(state, path)
```

The exported state is: the `filings` catalog, the `edgar_state` snapshot, the `companies`
resolver, and `schema_version`. No documents, no section text.

`rebuild_db_from_state` rehydrates the catalog. A subsequent `get(config)` re-downloads and
re-derives from EDGAR to reconstruct the data. **Reproducibility by re-execution, not by data
transfer.** This is the right default for open-sourcing: publish code, config, and state, and
anyone reconstructs the dataset themselves.

The assumption is that EDGAR still serves those accessions identically, which holds given
append-only plus immutable accessions, except for the rare redaction case. Ship the actual data
(HuggingFace, Kaggle) as a belt-and-braces fallback, not as the primary path.
