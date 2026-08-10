# Task 04 — EDGAR state table and the daily index sync

**Goal**: Populate and incrementally refresh `edgar_state` from EDGAR's own index feeds, per
`reference/catalog-and-state.md`.

**Depends on**: 03

**Files**:
- `src/edgar_tools/edgar_state.py` — index-feed clients, diff/sync logic
- `tests/edgar_tools/test_edgar_state.py`
- `tests/edgar_tools/fixtures/edgar_state/` — small recorded index-feed responses (JSON/idx
  snippets), not live fetches

**Detail**:
- Two sources, both required:
  - `https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{q}/master.idx` — quarterly index,
    for backfill/reconciliation.
  - `https://data.sec.gov/submissions/CIK##########.json` — per-firm truth. **Caps at ~1,000
    recent filings**; fetch the additional shards under `filings.files` too, or high-volume
    filers silently lose early years. Do not skip this.
- Insert one `edgar_state` row per observation with `seen_at = now()`; never upsert.
- **Disappearance handling**: an accession present yesterday and absent today is a redaction. Log
  it as an anomaly, set the corresponding `filings.status = 'not_available'` if held, and do
  **not** delete historical `edgar_state` rows.
- This task only produces `edgar_state` rows (metadata). It must not fetch or store any filing
  document — that's task 05.
- One process-wide rate limiter is introduced here (task 05 shares it) — see rule 10 in
  `.claude/tasks/README.md`. A real contact email in `User-Agent` on every request, sourced from
  `config.runtime.user_agent`.

**Tests**:
- Parsing a recorded `master.idx` snippet yields correctly typed rows (accession, cik, form,
  filing_date).
- Parsing a recorded per-firm submissions JSON (including a synthetic "shard" reference) picks up
  filings beyond the ~1,000-row cap by following the shard.
- A second sync run against a fixture where one accession has disappeared marks it as an anomaly
  and does not remove the earlier `edgar_state` row.
- No test hits the network — everything is fixture-driven (rule from `.claude/tasks/README.md`:
  never fetch from EDGAR inside a test).

**Done when**: four gates green; `edgar_state` sync is idempotent (running it twice with no new
EDGAR activity produces no new rows beyond the expected new `seen_at` observation policy you
choose and document).

**Commit**: `edgar_tools: EDGAR state table and daily index sync`
