# Task 11 — `own-regex` adapter with ablation flags

**Goal**: Implement the title-bearing/comma-lookbehind regex extractor, fixing the prototype bugs
documented in `reference/extractors.md`.

**Depends on**: 07

**Files**:
- `src/edgar_tools/extractors/own_regex_adapter.py`
- `tests/edgar_tools/test_own_regex_adapter.py`

**Detail**: two independent ablation flags — **must stay separate**, not one bundled arm, or a
regression can't be attributed:
- `title_keyword` — start patterns require the item title immediately after the number (e.g. Item
  7 must be followed by `management's discussion|md&a`). Prevents matching bare cross-references.
- `comma_lookbehind` — `(?<!,\s)` on start patterns, suppressing matches inside comma-separated
  cross-reference lists ("as set forth in Items 1, 1A, and 7").

Fixes required (carried over from the prototype — these are bugs, not preferences, per the
reference doc):
1. **`ITEM1A_END`'s second alternative terminates on Item 1** (the preceding item) — remove it;
   let the Item 2 branch carry the terminator, or extend to Item 3.
2. **Add line anchoring** (`\n[^\S\r\n]*`) to start patterns — only correct once the HTML-to-text
   step preserves block boundaries (inscriptis does; naive tag stripping doesn't).
3. **Add monotonic ordering** — thread a `positions` list as edgar-crawler does; items are
   extracted independently in the prototype, discarding the strict-ordering constraint.
4. **Widen `ITEM7A_END` terminators** — currently accepts only `item 8 ... Finan`; fall through
   to 9, 9A, 9B so a deviant Item 8 heading doesn't return `None` for a cleanly-matched 7A start.
5. **Table removal on the DOM, before stripping** — do this via inscriptis (already the chosen
   representation), not via post-hoc delimiter-heavy-line detection on stripped tags, which finds
   nothing on post-2001 HTML.
- Keep the **longest-span-wins pairing** from the prototype — it's a genuine TOC defence (a TOC
  start pairs with an adjacent TOC end and produces a short span that loses).
- Consumes inscriptis text — same `preproc_id` as `itemseg`, deliberately, for comparability.

**Tests**:
- One test per fix above, each reproducing the original bug on a minimal fixture and asserting
  it's resolved (e.g. an Item 1A span that used to truncate at a stray "Item 1" cross-reference
  now runs to the correct terminator).
- `title_keyword` and `comma_lookbehind` are independently toggleable and independently testable
  — a test with only one flag on must show only that flag's effect.
- Monotonic ordering: a synthetic fixture with an out-of-order regex match (e.g. a stray "Item 7"
  string appearing before the real Item 3 section) does not produce a candidate that starts before
  the prior item's end.
- Longest-span-wins: a fixture with both a TOC-embedded pseudo-match and the real section confirms
  the real (longer) span wins.

**Done when**: four gates green; all five documented bugs have a named regression test.

**Commit**: `edgar_tools: own-regex extractor with ablation flags, prototype bugs fixed`
