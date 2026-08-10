---
name: sec-filing-db-coding
description: Build and maintain `edgar_tools`, a sub-module of phd-narrative-finance that acquires SEC EDGAR filings, stores them in a content-addressed raw layer, extracts item-level sections with four independent extractors, and exports versioned, labelled research datasets. Use this skill for any work on that module: scaffolding, storage layers, the EDGAR state table, extractor adapters, validation, the extraction benchmark, export archives, or the manifest format. Trigger on mentions of edgar_tools, EDGAR item extraction, edgar-crawler, edgartools, itemseg/BERT4ItemSeg, filing section extraction, disclosure drift data, or the research extract manifest.
---

# edgar_tools

## What this is

A standalone sub-module at `src/edgar_tools/` that turns SEC EDGAR into a reproducible,
versioned research dataset of item-level filing sections.

It exists to support disclosure drift research: measuring year-over-year semantic change in
10-K and 10-Q sections as a leading indicator of quality deterioration. That downstream use
dictates almost every design decision here, so read the constraint before reading the design.

## The constraint that drives everything

Drift is measured as a change between the same firm's section text in year *t* and year *t-1*.
If those two years' text came from different extractors, or from different HTML-to-text
representations, the measured "drift" contains the difference between two parsers. That
difference is correlated with filing format changes, which is exactly when real disclosure
change also happens. The confound is not separable after the fact.

Three rules follow, and nothing in this module may violate them:

1. **All extractors run on every filing. No early exit, no filing-level cascade.**
   Extraction is cheap next to embedding. A cascade that stops at the first success destroys
   the agreement signal precisely where it would be informative.
2. **Extractor selection happens at the firm level, not the filing level.** For each firm and
   item, pick the source with complete valid coverage across that firm's whole series. Never
   build a drift pair across an extractor switch.
3. **Each extractor gets the text representation it expects.** BERT4ItemSeg was trained on
   inscriptis output; feeding it edgar-crawler's stripped text puts it off-distribution and it
   degrades silently rather than failing. Representations are not interchangeable.

## Storage: three tiers, two permanent

| Tier | Content | Persistence | Format |
|---|---|---|---|
| Raw | full SGML submission as EDGAR served it | permanent, append-only | `raw/{sha[:2]}/{sha}.txt.zst` |
| Derived | per-framework text (inscriptis, strip, DOM) | evictable cache | `derived/{preproc_id}/{sha[:2]}/{sha}.txt.zst` |
| Sections | extracted item text plus provenance | permanent | partitioned parquet under `extracted/{confighash}/` |

The middle tier is a cache, not a dataset. It is a pure function of `(raw_sha, preproc_id)`, so
it is regenerable, evictable under disk pressure, and never exported. It exists only because the
DOM build and inscriptis rendering are expensive (seconds per large filing) and must not sit
inside an iteration loop.

Blobs never go in a database. Metadata never goes on a bare filesystem. See
`reference/architecture.md`.

## Catalog: SQLite, one file

The catalog answers "do I already hold this?" and "what does EDGAR have that I don't?". It is
small, transactional, and must travel as a single file for the state export. SQLite, not a
server. See `reference/catalog-and-state.md`.

A separate `edgar_state` table mirrors EDGAR's *index* completely (all filings, metadata only,
no documents) while the blob store holds documents *selectively* (your universe, your forms).
That asymmetry is deliberate: the full EDGAR corpus is over 1,000 TB, but its index is
megabytes to low GB. Missing data is then a `LEFT JOIN`, not a guess.

EDGAR is append-only for these purposes. Amendments are new filings with new accession numbers;
originals are never modified. The only mutation is rare staff deletion or redaction, which the
daily diff surfaces as a disappearance.

## Extractors: four sources, one interface

| Name | Method | Representation | Status |
|---|---|---|---|
| `edgartools` | rule-based over the DOM: TOC anchors (0.95), typography (0.7-0.9), regex (0.6) | raw HTML, parsed in-process | unbenchmarked |
| `edgar-crawler` | line-anchored regex with monotonic ordering | its own strip + table removal | fork pending |
| `itemseg` | BERT4ItemSeg (SBERT + BiLSTM over lines) and CRF variant | inscriptis | published: 0.9826 macro-F1 core items |
| `own-regex` | title-bearing patterns, comma-list lookbehind, as ablation flags | inscriptis | unbenchmarked |

**Do not assert a priority order before the benchmark runs.** The only published number covers
`itemseg`. The claim that `edgartools` is strongest on HTML is an inference from reading its
source, not a measurement. `reference/benchmark.md` specifies the experiment that settles it.

A fifth path, the LLM healing layer, is not an extractor and is not in the priority chain. It
fires only where no source passes validation, returns character offsets rather than text, and is
frozen into a versioned cache before the final build. See `reference/healing.md`.

## Public API

```python
from edgar_tools import Store

store = Store("path/or/abfss://uri")

# Acquisition only. Config-free. Never produces sections.
store.download(ciks=[...], forms=["10-K", "10-Q"])

# Ensure raw -> ensure derived -> extract -> persist -> return.
df = store.get(config="config.json")          # or a dict
df = store.get(config, allow_download=False)  # read-only over what exists

# Read-only SQL over the section store.
df = store.query("SELECT ... FROM sections WHERE item = '1A'")

# Point-in-time lookups.
store.get_specific(accession)
store.get_latest_as_of(cik, date, form="10-K")     # filing_date <= date, latest
store.get_covering_period(cik, date, form="10-K")  # period contains date. LOOK-AHEAD.

# Export.
store.export(confighash, label="...", title="...")       # ZIP archive
store.flat_export(confighash, label="...", title="...")  # single parquet

# Reproducibility.
state = store.export_state()
Store.rebuild_db_from_state(state, path)
```

`download` is config-free acquisition. `get` is the config-dependent step, because sectioning
depends on which items, which tools, and which preprocessing. Making `download` produce sections
would bake a default config into the raw layer.

`get_latest_as_of` and `get_covering_period` must never be collapsed into one function. The
first is point-in-time and safe for signal construction; the second looks ahead and is safe only
for descriptive work. Naming the semantics is the whole point: a user who assumes the wrong one
gets look-ahead bias in a backtest with no error raised. Amendments compose correctly under
point-in-time semantics without special-casing, since a `10-K/A` is just another filing with its
own `filing_date`.

## Versioning rule of thumb

This module implements a general rule that applies beyond EDGAR, and will be applied to
RavenPack computation and other sources:

> Raw and lightly-derived data live on the server, append-only and content-addressed. Every
> research project is based on a labelled, immutable extract of that store. No research project
> reads the raw store directly.

Every extract carries a `MANIFEST.json` plus rendered `README.md` in the archive root. Manifest
fields are specified in `reference/output-and-export.md`. All manifest headers are additionally
written to a central `extracts_index.parquet` so extracts are searchable without opening each
archive.

Reproducibility is by re-execution, not by data transfer: publish code, config, and catalog
state, and anyone reconstructs the dataset from EDGAR. Data export exists as a fallback for
convenience and for the rare-redaction case.

## Portability

Home datalake and any future managed lakehouse (e.g. MS Fabric OneLake) are the same substrate:
parquet in a lakehouse. Use fsspec paths everywhere from the first commit so local disk and a
remote object store differ by a config string. No Postgres fork, no storage-engine abstraction
beyond fsspec.

## Reference documents

- `reference/architecture.md` — storage tiers, content addressing, write ordering, portability
- `reference/catalog-and-state.md` — SQLite schema, EDGAR state table, resolver, point-in-time queries
- `reference/config-and-hashing.md` — config schema, canonicalisation, what is and is not hashed
- `reference/extractors.md` — the `Extractor` protocol, per-framework preprocessing, each adapter
- `reference/validation-and-selection.md` — validation rules, firm-level selection, agreement diagnostic
- `reference/healing.md` — LLM boundary detection, the frozen cache, two-phase build
- `reference/output-and-export.md` — section schema, partitioning, archive format, manifest
- `reference/benchmark.md` — the extraction benchmark protocol

## Task plan

`.claude/tasks/` holds a commit-oriented build plan. Read `.claude/tasks/README.md` first for
conventions. Every task ends with a green `pytest`, `ruff check`, `ruff format --check`, and
(from task 01 onward) `mypy src/`. No task is complete without tests.

## Open decisions, flagged not guessed

These are unresolved. Do not resolve them silently in code.

1. **Extractor priority order.** Unknown until the benchmark runs. Code must read the order from
   config, never hardcode it.
2. **edgar-crawler fork.** Not on PyPI, no `pyproject.toml`. The fork is being handled in a
   separate repo and pipeline. Build the adapter against the `Extractor` protocol with a stub
   until it lands.
3. **Amendment merge policy.** Whether a `10-K/A` is merged onto its parent or dropped is a
   research decision, not an extraction one. Tag `is_amendment` from day one; leave the policy
   out of this module.
4. **Per-config vs global section store.** Currently per-config (`extracted/{confighash}/`).
   Migration to a global store keyed on `(accession, item, tool, preproc_id)` is possible later
   without a schema change if configs turn out to overlap heavily.
5. **LLM provider for healing.** Self-hosted models or a hosted API, via the OpenAI SDK or
   equivalent. Interface should not assume a provider.
