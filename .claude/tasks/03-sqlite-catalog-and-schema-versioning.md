# Task 03 — SQLite catalog and schema versioning

**Goal**: Stand up `catalog.db` with the full schema from `reference/catalog-and-state.md`,
including `schema_meta.schema_version`.

**Depends on**: 01

**Files**:
- `src/edgar_tools/catalog.py` — connection management, schema creation/migration, typed row
  accessors
- `tests/edgar_tools/test_catalog.py`

**Detail**:
- Tables exactly as specified: `schema_meta`, `edgar_state` (append-only, `PRIMARY KEY (accession,
  seen_at)`, never upsert), `filings` (`PRIMARY KEY (accession)`, `status` in `{pending, stored,
  failed, not_available}` — three states, not two, per the doc's reasoning), `derived`
  (`PRIMARY KEY (raw_sha, preproc_id)`), `companies` (`PRIMARY KEY (cik, name, valid_from)`).
- Indices: `idx_state_cik_form`, `idx_filings_cik_form_date`, `idx_filings_status`,
  `idx_companies_name`.
- `schema_meta` seeded with `('schema_version', '1')` on creation.
- One connection factory (`root: str | Path -> catalog.db` under it), WAL mode for concurrent
  read during download, foreign-key-off (SQLite defaults) is fine since this schema has no FKs by
  design.
- `received_at` is the only timestamp on `filings` beyond `last_attempt_at` — do not add an SEC
  dissemination-side timestamp; it can't be backfilled or sourced (see the doc's rationale).
- Company resolution: `resolve_name(cik, as_of) -> str | None`, resolving **as of the filing
  date**, not today.

**Tests**:
- Fresh catalog creation produces all 5 tables + all 4 indices; `schema_meta['schema_version'] ==
  '1'`.
- Inserting two `edgar_state` rows for the same accession at different `seen_at` keeps both (no
  upsert).
- `filings.status` transitions `pending -> stored` and `pending -> failed` are representable and
  distinguishable in a query (a `failed` row must not look like it was never attempted).
- `companies` name resolution picks the name valid at a given historical date, not the latest one.

**Done when**: four gates green; a catalog built from scratch matches the schema in
`reference/catalog-and-state.md` byte-for-byte in `CREATE TABLE` semantics (column names, types,
keys).

**Commit**: `edgar_tools: SQLite catalog schema and schema versioning`
