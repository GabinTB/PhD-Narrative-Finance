# Validation and firm-level selection

## Why validation gates everything

Without accept rules, the fallback never fires: a 40-word truncated Item 1A counts as
"available", the healing layer never sees it, and the selection logic treats it as a valid year
in the firm's series.

The failure that damages drift most is not a null return. It is a regex that returns *something
wrong*:

- Item 1A truncated at 200 words because the body contains "see Item 2. Properties"
- Item 7 that swallowed Item 8's financial statements because the 7A terminator never matched
- A section returned at high confidence containing raw HTML because an anchor landed wrong

Those look like successes. Validation is what distinguishes them.

## ValidationResult

```python
from enum import Enum

class ValidationStatus(str, Enum):
    VALID           = "valid"
    TOO_SHORT       = "too_short"
    TOO_LONG        = "too_long"
    SPAN_OVERSIZED  = "span_oversized"
    TERMINATOR_IN_TEXT = "terminator_in_text"
    BOILERPLATE_XREF = "boilerplate_xref"
    NOT_FOUND       = "not_found"
    ABSENT_BY_RULE  = "absent_by_rule"
    ERROR           = "error"

@dataclass(frozen=True)
class ValidationResult:
    status: ValidationStatus
    detail: str | None = None
```

`passed_validation` on the output row is `status == VALID`.

## Rules

Applied in order; first failure wins.

1. **`ABSENT_BY_RULE`** — checked first, before anything else. See below.
2. **`NOT_FOUND`** — the extractor returned nothing for this item.
3. **`TOO_SHORT`** — word count below `validation.min_words` (default 50).
4. **`SPAN_OVERSIZED`** — extracted span exceeds `validation.max_doc_fraction` of the cleaned
   document (default 0.6). Catches the swallowed-following-items failure.
5. **`TERMINATOR_IN_TEXT`** — the terminating item header appears inside the extracted text.
   Means the span ran past its own boundary.
6. **`BOILERPLATE_XREF`** — the text is a pure incorporation-by-reference stub ("incorporated
   herein by reference to...") with no substantive content. Genuinely short and genuinely not
   the section.
7. **`TOO_LONG`** — absolute word ceiling, as a backstop for pathological output.

`edgartools` provides its own size-band flag and `confidence`. Store both, but apply this
module's rules independently so all extractors are gated identically. Do not delegate validation
to any one backend.

## `ABSENT_BY_RULE` is not a failure

This will bite as soon as the universe extends past large caps.

Smaller reporting companies are **not required** to provide Item 1A or Item 7A. Their absence is
correct. A healing loop pointed at them will happily invent boundaries for sections that do not
exist.

Check `filer_category` (from `dei:EntityFilerCategory` on the cover page, stored on the filing
row) before treating a missing item as a parse failure.

Two more legitimate cases that must not be treated as failures:

- **Combined headers.** "Items 1 and 2. Business and Properties" is common and legal. Detect and
  attribute the combined span to both items, flagged as such.
- **Incorporation by reference.** Item 7 reduced to two sentences pointing at an annual report
  exhibit is genuinely short, not broken. This is `BOILERPLATE_XREF`, distinct from `TOO_SHORT`.

## Firm-level selection

The rule that protects the drift measure. Do not implement this at the filing level.

```
For each (cik, item, form):
    candidates = tools where validation == VALID for EVERY filing in that firm's series
    if candidates non-empty:
        chosen = highest priority in candidates      # priority from config
    else:
        chosen = tool with the longest unbroken valid run
        usable_range = that run only
    emit selection row: (cik, form, item, chosen, coverage, run_start, run_end)
```

Then, downstream: **never build a drift pair across an extractor switch.** Drop the pair, flag
the firm-year. This is a hard rule; the pair would contain the difference between two parsers,
and that difference is correlated with filing format changes, which is when real drift also
happens.

### Priority comes from config

Never hardcode an ordering. The benchmark has not run; the ordering is an open question. The
selection function reads `config.tools` order or an explicit `priority` list.

### Format-regime split

Priority may legitimately differ by regime, since the mechanisms differ:

- **HTML era (roughly 2001 onward)**: anchors and typography are available.
- **Plain-text era (pre-2001)**: no anchors, no styles, so DOM-based detection degrades to its
  own regex fallback.

The regime switch happens at the same calendar point for every firm, so it is a common shock
absorbable with a dummy, not firm-specific extractor noise. The firm-level consistency rule
still applies **within** regime.

Also note extraction quality is not stationary across the sample for a third reason: pre-2001
plain text, 2001-2019 HTML of varying quality, 2019+ inline XBRL. Check that mean drift does not
jump at format transitions.

## Agreement diagnostic

Free, and it preempts the obvious referee question about extraction robustness.

Wherever two or more tools pass validation on the same `(accession, item)`, compute token-level
agreement between their outputs. Low agreement means "both succeeded" is an illusion for that
filing type.

**Compare after normalisation of the extracted text, not by offsets.** Offsets are scoped to
`repr_id` and are not comparable across representations. Normalisation: casefold, strip
non-alphanumeric, collapse whitespace, then token-set IoU.

This is why `edgartools` cannot be forced onto a shared input representation. It is inherently
DOM-based and would lose its entire advantage. The agreement metric works across representations;
a shared input does not.

Report the agreement distribution in any paper using this data.

## Sample-selection consequence

Extraction failures are almost certainly correlated **within** firm: a firm using an unusual
filing agent uses it every year.

Two implications:

- Pair-correctness is higher than p², which helps panel size.
- The firms lost are a systematic group, not a random draw.

Measure the within-firm correlation of failure, and check whether dropped firms differ on size,
industry, or filer status. If they do, that belongs in the limitations section.
