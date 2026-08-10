# Task 05 — Raw store, rate limiter, `download`

**Goal**: Implement the content-addressed raw blob tier and the public, config-free
`Store.download()`, per `reference/architecture.md`.

**Depends on**: 03, 04

**Files**:
- `src/edgar_tools/raw_store.py` — content-addressed blob writer/reader (`raw/{sha[:2]}/{sha}.txt.zst`)
- `src/edgar_tools/rate_limiter.py` — single process-wide, file-lock-backed limiter
- `src/edgar_tools/store.py` — introduces `Store` with `download()` only at this stage
- `tests/edgar_tools/test_raw_store.py`
- `tests/edgar_tools/test_rate_limiter.py`

**Detail**:
- Store the **full SGML submission** (`{accession}.txt` as EDGAR serves it), not the stripped
  primary document — it's the strict superset every extractor derives from. zstd-compressed,
  named by SHA-256 of the uncompressed bytes.
- **Write ordering is load-bearing** (rule 5): write blob to a temp file on the same filesystem,
  fsync, verify SHA-256 matches, atomic rename into final path, *then* insert the catalog
  `filings` row in one transaction. If the process crashes between these, nothing in the catalog
  claims a filing that isn't on disk — this is what makes crash-resume free. Write a test that
  simulates a crash between blob-write and catalog-insert and asserts recovery is a plain re-run.
- Two force flags, not one: `force_refetch` (invalidates raw tier, re-downloads) and
  `force_reextract` (later tasks; invalidates derived+sections only). Only wire `force_refetch`
  here.
- Rate limiter: one process-wide, persisted (file lock or equivalent) so two processes cannot
  both believe they own the 10 req/s budget — SEC's limit is per-IP, not per call site. Every
  request carries `User-Agent: <name> (<email>)` from `config.runtime.user_agent`; reject startup
  if that's missing or looks like a placeholder.
- `Store.download(ciks, forms, ...)` reads the `edgar_state ⋈ filings` gap (the LEFT JOIN from
  `reference/catalog-and-state.md`), fetches whatever's `pending`/`failed`, and is **config-free**
  — it must not accept `items_to_extract`, `tools`, or anything extraction-related. This is rule
  4 in `.claude/tasks/README.md`.
- fsspec paths from this commit on (rule: portability). Local disk and a future remote object
  store differ only by the `root` config string — do not hardcode `pathlib.Path` filesystem calls
  where an fsspec call would do.

**Tests**:
- Round-trip: write raw bytes → read back → identical bytes, correct SHA in filename.
- Re-downloading identical content is a no-op (dedup via content address) — assert no duplicate
  blob and no duplicate catalog row.
- Corrupted blob (SHA mismatch on write) is rejected before the catalog row is inserted.
- `download()` rejects an `items_to_extract`/`tools` kwarg — config-free is enforced, not just
  documented.
- Concurrent rate-limiter acquisition from two simulated processes never exceeds the configured
  rate (use a fake clock, not real sleeps).
- No test hits the real network — mock the HTTP layer.

**Done when**: four gates green; `download()` is demonstrably resumable after a simulated
crash with no manual cleanup.

**Commit**: `edgar_tools: raw store, rate limiter, download()`
