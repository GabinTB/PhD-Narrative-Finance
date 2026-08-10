# Task 19 — Notebooks, docs, convergence pass

**Goal**: Close out the module: worked-example notebooks, documentation reconciliation, and a
full re-read of every reference doc against the shipped code to catch drift accumulated over
tasks 01–18.

**Depends on**: all prior tasks

**Files**:
- `content/` (or a module-local `dev/` notebook, matching wherever this repo's other chapters keep
  worked examples — follow existing convention, e.g. `content/chapter-2/central-banker-speeches/`)
  — one or two notebooks demonstrating `download → get → query → get_latest_as_of → export`
  end-to-end against a small real (or fixture) universe
- `.claude/skills/sec-filing-db-coding/SKILL.md` — update any section that drifted from the
  as-built code (open decisions that got resolved, e.g. the edgar-crawler fork landing, the
  benchmark's priority order)
- `.claude/tasks/README.md` — mark closed/superseded items
- `README.md` (repo root) or a short module `README.md` under `src/edgar_tools/` — practical
  quickstart, matching the public API in `SKILL.md`

**Detail**:
- Re-read every `reference/*.md` doc against the actual shipped code, file by file, and fix any
  place where an "open decision" flagged in `SKILL.md` was resolved during implementation but the
  doc still says "unresolved" — or conversely, where code silently resolved something the doc
  still flags as open (that's a rule violation from `.claude/tasks/README.md`'s "flagged, not
  guessed" principle and must be fixed, not just documented).
- Confirm the five SKILL.md open decisions' actual status at this point:
  1. Extractor priority order — should now be resolved by task 18's benchmark; update config
     defaults and SKILL.md accordingly, or explicitly note it's still pending and why.
  2. `edgar-crawler` fork — update the adapter (task 09) if it landed; otherwise confirm the stub
     is still correctly non-silent.
  3. Amendment merge policy — confirm it was correctly left **out** of this module (a research
     decision downstream), not accidentally implemented.
  4. Per-config vs. global section store — confirm still per-config, or document the migration if
     it happened.
  5. LLM healing provider — document whichever provider ended up wired in task 15.
- Notebook(s) must exercise the point-in-time functions (task 14) explicitly enough to demonstrate
  the look-ahead-safe vs. look-ahead pattern side by side — this is the module's central
  correctness property and deserves to be visible, not just tested.
- Verify the `edgar` dependency group in root `pyproject.toml` is complete and `uv sync --group
  dev --group edgar` from a clean checkout is sufficient to run everything in this module,
  including the notebooks.
- Decide, and document in `SKILL.md`, whether `dev/edgar-tools` is ready to open a PR into `main`
  at this point, or what's still blocking (e.g. fork still pending, benchmark not yet run against
  real gold data).

**Tests**: no new unit tests required specifically for this task beyond what prior tasks already
cover; the deliverable is documentation/notebook convergence. If the doc reconciliation above
surfaces an actual code/doc mismatch that changes behaviour, fix the code and add the regression
test under the task where it logically belongs (retroactively, noted in this task's commit).

**Done when**: four gates green; every reference doc and `SKILL.md` accurately describes the
code as shipped, with no remaining "flagged not guessed" item silently resolved in code without a
doc update, and a working end-to-end notebook exists.

**Commit**: `edgar_tools: notebooks, docs reconciliation, convergence pass`
