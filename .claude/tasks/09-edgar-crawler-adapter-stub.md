# Task 09 — `edgar-crawler` adapter (stub, then fork)

**Goal**: Build the `edgar-crawler` adapter against the `Extractor` protocol as a stub, ready to
swap in the fork the moment it lands, per `reference/extractors.md`.

**Depends on**: 07

**Files**:
- `src/edgar_tools/extractors/edgar_crawler_adapter.py`
- `tests/edgar_tools/test_edgar_crawler_adapter.py`

**Detail**:
- **This is an open decision (SKILL.md #2), flagged not guessed.** `edgar-crawler` is not on
  PyPI, has no `pyproject.toml`; the fork is handled in a separate repo/pipeline. Build the
  adapter class fully against the `Extractor` protocol now; its `extract()` raises
  `NotImplementedError` until the fork lands. Never stub by returning fake data — a silent fake
  propagates into the corpus (rule from `.claude/tasks/README.md`).
- Mark the stub's would-be-real tests `pytest.mark.skipif(condition=True, reason="edgar-crawler
  fork not yet available")` so the suite stays green without pretending coverage exists.
- Document, in the module docstring, the three properties the fork must preserve (from
  `reference/extractors.md`): line anchoring (`\n[^\S\r\n]*`, prevents mid-sentence
  cross-reference matches), monotonic ordering (a `positions` list rejecting any candidate
  starting before the previous item's end), and terminator generality (fall through successive
  candidate terminators).
- What the fork must expose, so the adapter can wrap it cleanly once it lands: a per-document
  callable that raises on failure (not a batch script that loops/prints), ablation flags for the
  own-regex-style additions so arms stay switchable at runtime, and a stable `version` string
  that changes on any extraction-affecting change.
- **Verify on fork**: whether it downloads the complete submission `.txt` or only the primary
  document — that determines whether exhibit stripping becomes this module's problem. Leave a
  `# TODO(fork):` marker for this check; do not guess an answer into the code.
- Its `preproc_id` is its own strip + table removal — distinct from inscriptis (`itemseg`,
  `own-regex`) and from edgartools's DOM.

**Tests**:
- Registry resolves `"edgar-crawler"` to the stub adapter; calling `.extract()` raises
  `NotImplementedError` with a message pointing at this task file.
- The would-be functional tests exist, are `skipif`-marked, and are discoverable (not simply
  absent) so CI visibly shows what's pending rather than silently omitting coverage.
- Registry-level `run_all` (task 07) tolerates one extractor raising `NotImplementedError` without
  aborting the other extractors' runs on the same filing (rule 1: all extractors run on every
  filing — a stub failing must not cascade).

**Done when**: four gates green; nothing downstream (validation, selection) can mistake the stub
for a working extractor — a stub result must never reach `VALID`.

**Commit**: `edgar_tools: edgar-crawler adapter stub (fork pending)`
