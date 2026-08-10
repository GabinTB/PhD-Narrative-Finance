# Task 06 — Derived cache and preprocessing registry

**Goal**: Implement the evictable, regenerable derived-text tier and the `preproc_id` registry,
per `reference/architecture.md`.

**Depends on**: 05

**Files**:
- `src/edgar_tools/derived_store.py` — `(raw_sha, preproc_id) -> derived_sha` cache, backed by
  `derived/{preproc_id}/{sha[:2]}/{sha}.txt.zst` and the `derived` catalog table
- `src/edgar_tools/preproc.py` — registry of preprocessing frameworks (`inscriptis-2.5.0`,
  `edgarcrawler-strip-v1.2-notables`, DOM-for-`edgartools`, etc.) with stable, version-bearing IDs
- `tests/edgar_tools/test_derived_store.py`

**Detail**:
- Derived text is a **pure function** of `(raw_sha, preproc_id)` — treat it strictly as a cache:
  never exported (task 16 must skip it), evictable under disk pressure with no data loss, and
  represented by exactly one `derived(raw_sha, preproc_id, derived_sha)` row (not a first-class
  catalog object with its own lifecycle).
- `preproc_id` must be stable and must change the moment output changes — bump it on any library
  version change (e.g. `inscriptis` 2.5.0 → 2.5.1) or option change.
- `force_reextract` (introduced properly here): invalidates derived + sections, touches no
  network. Confirm it is meaningfully cheaper than `force_refetch` in a test (no HTTP calls made).
- Parallelise across filings, not within — CPU-bound Python, use a process pool. Do not add
  threading for the parse step (GIL-bound).
- `get_or_build(raw_sha, preproc_id, builder: Callable[[bytes], str]) -> str` is the core entry
  point later extractors call through; it must be safe to call concurrently for the same key
  without double-computing (lock or check-then-insert under a transaction).

**Tests**:
- First call computes and persists; second call for the same `(raw_sha, preproc_id)` does not
  re-invoke the builder (cache hit).
- Different `preproc_id` for the same `raw_sha` produces a distinct derived blob.
- Simulated eviction (delete the blob, keep the catalog row) is detected and triggers a rebuild
  rather than serving a stale/missing read.
- `force_reextract=True` rebuilds derived text without any network call (assert via a mock that
  raises if the HTTP layer is touched).

**Done when**: four gates green; re-running extraction under `force_reextract` on a warm cache
completes without touching the network, verified by test.

**Commit**: `edgar_tools: derived cache and preprocessing registry`
