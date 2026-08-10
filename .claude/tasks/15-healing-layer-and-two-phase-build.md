# Task 15 — Healing layer and two-phase build

**Goal**: Implement the LLM boundary-detection healing layer and the mandatory two-phase build,
per `reference/healing.md`.

**Depends on**: 07 (validation trigger), 13 (section store to rebuild against)

**Files**:
- `src/edgar_tools/healing.py` — `HealingCache`, resolver, two-phase orchestration
- `tests/edgar_tools/test_healing.py`
- `tests/edgar_tools/fixtures/healing/` — recorded LLM responses (no live API calls in tests)

**Detail**:
- **Not an extractor, not in the priority chain.** Fires only where **no** source passes
  validation. Returns **character offsets**, never text — the module slices the text itself.
  Paraphrase from a text-returning model would be indistinguishable from real drift; offsets make
  that structurally impossible. Enforce this at the type level (the healing resolver's return type
  has no text field).
- **No generated regexes, ever.** Rejected for three reasons documented in the reference doc:
  unvalidatable (fixes one filing, silently mis-fires on hundreds already processed), ReDoS risk
  (LLM-authored regex catastrophic backtracking), unreproducible. Do not add a "propose a pattern"
  code path even as an experiment.
- **Mandatory two-phase build** — not optional, and this is the part most likely to get skipped
  under time pressure:
  - **Phase 1, discovery**: run extraction, collect every validation failure, resolve via LLM,
    persist resolutions to a versioned artifact (`HealingCache`).
  - **Phase 2, freeze and rebuild**: version the cache, fold its hash into the extraction config
    hash (task 02's `row_config_hash` already has a `healing.cache_version` slot — wire it here),
    re-extract everything from scratch against the frozen state. This must reuse `force_reextract`
    (task 06) — no network touched, cache-only.
  - If resolutions accumulate mid-run instead, a filing processed in hour one sees a different
    (unhealed) extractor outcome than one processed in hour nine, and provenance can't tell —
    this breaks firm-level consistency (task 12) invisibly. A test must catch this class of bug,
    not just document it.
- `HealingCache` fields exactly as specified: `version, model_snapshot (pinned, not a moving
  alias), prompt_hash, temperature (0.0, asserted at call time), offsets: dict[(accession, item,
  repr_id), (start, end)]`. Keyed on the tuple, not on a pattern — a lookup table, not a rule
  engine, deterministic on replay.
- Trigger hooks to the **validation layer** (task 07), not to a null return — a wrong-content
  "success" (e.g. `SPAN_OVERSIZED`) must trigger healing too, since that's the failure class that
  actually damages drift. Do **not** trigger on `ABSENT_BY_RULE` — the model will invent
  boundaries for sections that legitimately don't exist.
- Prompt: line-numbered inscriptis text; ask for first/last line ID of the item, convert to
  character offsets afterward. Never prompt with raw character positions.
- What healing cannot fix (self-contradicting documents — wrong item numbering, item content
  placed under the wrong header) gets a `manual_audit` status, not a forced resolution.
- Item 1C: route through healing by default unless explicitly excluded in config (task 10 already
  flags this on the itemseg side — this is where it gets a real resolution path).

**Tests**:
- A validation-failure fixture (e.g. `SPAN_OVERSIZED`) triggers healing; an `ABSENT_BY_RULE`
  fixture does not.
- Healing resolver returns offsets only — assert no code path can extract `.text` directly from
  the resolver's return value without an explicit slice-and-inspect step.
- `HealingCache` round-trips through save/load with all fields intact; `temperature != 0.0` at
  call time raises.
- Two-phase build: a discovery-phase run against a fixture, followed by a freeze+rebuild, produces
  identical output on repeated freeze+rebuild calls (deterministic replay) and makes zero mocked
  network calls during phase 2.
- A firm-series fixture where healing resolves filing A but not filing B (still failing) — the
  consistency check (task 12 tie-in) must flag this as a broken run for that firm/item, not select
  silently across the healed/unhealed boundary.

**Done when**: four gates green; the discovery→freeze→rebuild sequence is exercised end-to-end in
a test with no live LLM calls (mocked/recorded responses only).

**Commit**: `edgar_tools: LLM healing layer and two-phase build`
