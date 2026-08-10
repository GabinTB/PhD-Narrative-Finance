# Task conventions — `edgar_tools`

## Structure

Tasks are numbered and ordered. Each is one commit, or a small series of commits under one
theme. Do not merge tasks; do not skip ahead.

Each task file states:

- **Goal** — one sentence
- **Depends on** — prior tasks
- **Files** — what is created or modified
- **Detail** — the actual directives
- **Tests** — what must be tested, specifically
- **Done when** — the acceptance gate
- **Commit** — the message

## Non-negotiable gates

Every task ends green on all four, **scoped to this module** (the repo root also holds unrelated
thesis notebooks with their own pre-existing lint debt — do not let this module's CI depend on
fixing that, and do not fix it incidentally as part of an edgar_tools task):

```bash
uv run ruff check src/edgar_tools tests/edgar_tools
uv run ruff format --check src/edgar_tools tests/edgar_tools
uv run pytest tests/edgar_tools
uv run mypy src/edgar_tools          # from task 01 onward
```

A task with new code and no new tests is not complete. A task that leaves any of the four red is
not complete. Do not commit through a failure with a note to fix it later.

## Workflow

1. Work happens on `dev/edgar-tools` (or a task-scoped branch off it) until the module reaches a
   state Gabin is happy making visible on `main`.
2. Open a GitHub issue before modifying source files, once the repo's issue tracker is in use for
   this module; reference the issue in the commit message. Not yet mandatory while the branch is
   private, but adopt it before merging to `main`.
3. Task files for the coding agent live under `.claude/tasks/` in this repo (this directory).

## Rules the agent must not break

These come from `../skills/sec-filing-db-coding/SKILL.md` and its `reference/` documents. They
are settled design decisions. Deviating from any of them requires flagging it explicitly, not
implementing it silently.

1. All extractors run on every filing. No early exit, no filing-level cascade.
2. Extractor selection is at the firm level, never the filing level.
3. Each extractor consumes its own text representation. Offsets are scoped to `repr_id`.
4. `download` is config-free and never produces sections.
5. Blob writes complete and verify before the catalog row is inserted.
6. Priority order is read from config, never hardcoded.
7. The section store is long format. Wide is an analysis-time view.
8. `end_date: null` is resolved to a concrete date before hashing.
9. `get_latest_as_of` and `get_covering_period` stay separate functions.
10. One process-wide rate limiter. A real contact email in `User-Agent` on every request.

## Placeholders

Where a dependency is not ready (notably the `edgar-crawler` fork), build against the `Extractor`
protocol with a stub that raises `NotImplementedError`, and mark its tests `pytest.mark.skipif`.
Never stub by returning fake data; a silent fake will propagate into the corpus.

## Test data

Commit a small fixture set of real filings under `tests/edgar_tools/fixtures/filings/`, chosen to
span:

- pre-2001 plain text
- 2001-2010 HTML
- 2011-2019 HTML
- post-2019 inline XBRL
- one smaller reporting company missing Item 1A and 7A (`absent_by_rule`)
- one filing with a combined "Items 1 and 2" header
- one 10-K/A partial amendment
- one filing with incorporation by reference for Item 7

Keep them small. Truncate where truncation does not affect the property under test. Never fetch
from EDGAR inside a test.

## Task list

| # | Task | File |
|---|---|---|
| 01 | Scaffold, uv, tooling, CI | `01-scaffold-uv-tooling-ci.md` |
| 02 | Config schema, resolution, canonicalisation, hashing | `02-config-schema-and-hashing.md` |
| 03 | SQLite catalog and schema versioning | `03-sqlite-catalog-and-schema-versioning.md` |
| 04 | EDGAR state table and the daily index sync | `04-edgar-state-table-and-daily-index-sync.md` |
| 05 | Raw store, rate limiter, `download` | `05-raw-store-rate-limiter-download.md` |
| 06 | Derived cache and preprocessing registry | `06-derived-cache-and-preprocessing-registry.md` |
| 07 | `Extractor` protocol, registry, validation layer | `07-extractor-protocol-registry-validation-layer.md` |
| 08 | `edgartools` adapter | `08-edgartools-adapter.md` |
| 09 | `edgar-crawler` adapter (stub, then fork) | `09-edgar-crawler-adapter-stub.md` |
| 10 | `itemseg` adapters (BERT and CRF) | `10-itemseg-adapters.md` |
| 11 | `own-regex` adapter with ablation flags | `11-own-regex-adapter.md` |
| 12 | Firm-level selection and agreement diagnostic | `12-firm-level-selection-and-agreement-diagnostic.md` |
| 13 | Section store, `get`, `query` | `13-section-store-get-query.md` |
| 14 | Point-in-time lookups | `14-point-in-time-lookups.md` |
| 15 | Healing layer and two-phase build | `15-healing-layer-and-two-phase-build.md` |
| 16 | Export, manifest, extracts index | `16-export-manifest-and-extracts-index.md` |
| 17 | State export and `rebuild_db_from_state` | `17-state-export-and-rebuild.md` |
| 18 | Benchmark harness | `18-benchmark-harness.md` |
| 19 | Notebooks, docs, convergence pass | `19-notebooks-docs-convergence-pass.md` |
