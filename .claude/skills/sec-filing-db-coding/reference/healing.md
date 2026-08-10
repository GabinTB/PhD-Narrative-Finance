# LLM healing layer

## What it is and is not

A resolver for filings where **no** extractor passes validation. It is not an extractor, is not
in the priority chain, and does not compete with the four sources.

It returns **character offsets** into a named representation. It never returns text.

## Why offsets, not text

If a model returns text, it will paraphrase, normalise, and drop clauses. Paraphrase is
indistinguishable from drift in the downstream measure. Offsets are verifiable (slice and
inspect), cheap to store, and make paraphrase structurally impossible. The module slices the
text itself.

## Why not generated regexes

The original instinct was to have an LLM propose a new regex when an item fails to be detected,
then save it. Rejected for three reasons:

1. **Unvalidatable.** A pattern that fixes the triggering filing can silently mis-fire on
   hundreds already processed, with no way to detect it short of re-checking everything.
2. **ReDoS.** LLM-written regexes catastrophically backtrack. One nested quantifier and a 3 MB
   filing hangs the pipeline. Guarding this requires a timeout wrapper and a backtracking check
   on every generated pattern, which is real work for no benefit.
3. **Unreproducible.** "We used an LLM to author regexes during the run" is not a method anyone
   can replicate.

## Why not fine-tuning

Measure the residual first, then decide:

| Residual | Action |
|---|---|
| under ~2% | LLM the failures. A few thousand calls at negligible cost. |
| 2-10% | LLM the failures; if one structural pattern dominates, hand-write one or two patterns. Cheaper and more auditable than a model. |
| over 10% | Something is broken upstream, almost certainly the HTML-to-text step. No model fixes a bad text representation, it learns to work around it. |

Fine-tuning would be distillation of the LLM labels into a smaller model. That pays off only at
volumes where API cost dominates, which a one-time corpus build of tens of thousands of filings
does not reach. The labels are needed first regardless, so it is strictly a later step and never
a replacement.

## The stationarity problem

If resolutions accumulate as a run proceeds, a filing processed in hour one sees a different
extractor than one processed in hour nine. That breaks the firm-level consistency rule
**invisibly**, because provenance would say the same tool name for both.

### Two-phase build

Mandatory. Not optional.

**Phase 1, discovery.** Run extraction, collect every validation failure, resolve them with the
LLM, persist resolutions to a versioned artifact.

**Phase 2, freeze and rebuild.** Version the healing cache, fold its hash into the extraction
config hash, and re-extract everything from scratch against the frozen state.

Phase 2 is cheap, which is exactly why `force_refetch` and `force_reextract` are separate flags.
Re-extraction reads cached derived text and touches neither the network nor the parser.

This also makes the result publishable: ship the frozen cache as an artifact and anyone
reproduces the corpus exactly.

## The cache

```python
@dataclass(frozen=True)
class HealingCache:
    version: str
    model_snapshot: str            # exact model identifier, not a moving alias
    prompt_hash: str
    temperature: float             # pinned to 0.0
    offsets: dict[tuple[str, str, str], tuple[int, int]]
    #        (accession, item, repr_id) -> (start, end)
```

Keyed on `(accession, item, repr_id)`, not on pattern. A lookup table of resolved boundaries, not
a rule engine. Deterministic on replay, trivially auditable, and structurally incapable of
mis-firing on filings it was not built for.

`repr_id` in the key is required: offset 41,203 means a different place in inscriptis text than
in edgar-crawler text.

## Provenance requirements

Without these, `extractor = "llm"` says nothing and the firm-level consistency rule is fake. Two
runs six months apart against a moving Vertex endpoint are two different extractors.

- `model_snapshot`: the exact pinned model identifier. Reject moving aliases.
- `prompt_hash`: SHA-256 of the prompt template.
- `temperature`: 0.0, asserted at call time.

All three are folded into the cache version, which is folded into the config hash.

## Trigger

Hook to the **validation layer**, not to a null return. Healing on "returned None" catches only
the easy failures and misses the wrong-content ones that actually damage drift.

Do not trigger on `ABSENT_BY_RULE`. Those sections legitimately do not exist and the model will
invent boundaries for them.

## Prompt design

Line-numbered inscriptis text. Ask for the line ID of the first and last line of the item, then
convert to character offsets. Line IDs are more stable for a model to reason about than raw
character positions, and they match the format used in the GPT4ItemSeg approach.

## Consistency rule

Never mix LLM-resolved and rule-based extraction across years within a firm, for the same reason
extractors cannot be mixed. Healing outputs are a distinct source in the selection logic.

## What healing cannot fix

The itemseg paper's error analysis groups its residual into random errors, very short reports,
incorrect headings, and unusual item placement. Two examples:

- A 2002 Allstates WorldCargo filing labels "Controls and Procedures" as Item 14 when it should
  be Item 9A. Every approach followed the wrong heading.
- A 2006 Hub International filing places Item 7A content inside Part II of Item 7. Every
  approach missed it.

These are semantic judgments about self-contradicting documents. Not learnable from more data,
not fixable by a generated regex. The healing layer resolves what it can and everything else
gets a `manual_audit` status.

## Item 1C

The SEC added Item 1C (cybersecurity) for fiscal years ending on or after 15 December 2023,
inside the research window. Trained models cannot recognise items they were never trained on and
will fail on it silently. Only prompt-based approaches adapt without retraining.

Either route Item 1C through healing by default, or exclude it explicitly in config. Do not let
it fail silently.

## Reporting

The defensible framing for a paper:

> Sections extracted with [tool]; the N% that failed validation were resolved by LLM boundary
> detection, with a manually audited random sample of 200 showing X% agreement.

That takes an afternoon. A fine-tuned boundary model is a second paper.
