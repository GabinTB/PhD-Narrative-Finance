# Config, canonicalisation and hashing

## Role of the config

The config is the sole specification of what `get` produces. It is the selection, and its hash
is the dataset identity. There is no second source of truth: do not accept loose `start`/`end`
kwargs alongside a config that already carries a date range, or they will diverge.

Accepted as a path or a dict. A file is a dict with a location; canonicalise to the dict
internally and hash that.

## Schema

```json
{
  "schema_version": "1",

  "universe": {
    "ciks": ["0000320193", "0000789019"],
    "company_names": [],
    "resolve_as_of": "2026-07-29"
  },

  "filings": {
    "forms": ["10-K", "10-Q"],
    "start_date": "2004-01-01",
    "end_date": "2025-12-31"
  },

  "items_to_extract": {
    "10-K": ["1", "1A", "3", "7", "7A"],
    "10-Q": ["part_1__2", "part_2__1A"]
  },

  "tools": ["edgartools", "edgar-crawler", "itemseg", "own-regex"],

  "preproc": {
    "edgartools":    { "engine": "dom",        "lib_version": "5.43.0" },
    "edgar-crawler": { "engine": "ec-strip",   "lib_version": "1.2.0", "remove_tables": true },
    "itemseg":       { "engine": "inscriptis", "lib_version": "2.5.0" },
    "own-regex":     { "engine": "inscriptis", "lib_version": "2.5.0",
                       "title_keyword": true, "comma_lookbehind": true }
  },

  "extract_options": {
    "include_signature": false
  },

  "healing": {
    "enabled": false,
    "cache_version": null
  },

  "validation": {
    "min_words": 50,
    "max_doc_fraction": 0.6,
    "reject_if_terminator_present": true,
    "reject_boilerplate_xref": true
  },

  "runtime": {
    "user_agent": "Name (email)",
    "workers": 8,
    "root": "/data/edgar"
  }
}
```

## `items_to_extract` must be form-keyed

A flat list is ambiguous the moment more than one form type is requested. Item 7 is MD&A in a
10-K; in a 10-Q, MD&A is Part I Item 2. `["1A", "7"]` cannot express both.

edgar-crawler's own config has this flaw. Override it with the mapping above. `null` or `{}`
means all items for all requested forms.

10-Q item keys follow edgar-crawler's convention: `part_{n}` for a whole part,
`part_{n}__{item}` for an item within a part.

## Amendments

Not in defaults. `"forms": ["10-K", "10-Q"]` is the default; a user who wants amendments writes
`["10-K", "10-K/A", "10-Q", "10-Q/A"]`.

An amendment is a form type, not an item. There is nothing inside a 10-K to extract under the
key `"amendments"`.

Tag `is_amendment` on every row regardless, so the deferred merge policy is available later
without a re-scan.

## Resolution before hashing

The config that gets hashed and stored is the **resolved** config, not the user's input.

- `end_date: null` (meaning "present") must be resolved to a concrete date at call time and
  written into the stored config before hashing. Otherwise the same config run a month apart
  produces different corpora under the same hash.
- `company_names` must be resolved to CIKs as of `resolve_as_of` and the expanded CIK list
  written in.
- CIKs normalised to zero-padded 10-character strings and sorted.

## Canonicalisation

Before hashing:

1. Drop non-semantic keys (below)
2. Sort all object keys recursively
3. Sort all arrays whose order is not semantic (CIK lists, form lists, item lists, tool lists)
4. Serialise with `json.dumps(..., sort_keys=True, separators=(",", ":"), ensure_ascii=False)`
5. SHA-256 the UTF-8 bytes

### Excluded from the hash

Everything under `runtime`:

- `user_agent`
- `workers`
- `root` and any other path
- network settings, retry counts, timeouts

Changing a contact email or a worker count must not produce a new dataset identity.

### Included in the hash, and easy to forget

- Every `preproc` entry including `lib_version`. Bumping inscriptis or edgartools changes the
  output text, therefore changes the dataset, therefore must change the hash. This is correct
  behaviour, not an annoyance: it is what makes resume safe.
- `healing.cache_version` when healing is enabled.
- The extractor code version. Fold the module's own git commit SHA, or a version string bumped
  on any extraction-affecting change, into the hashed payload.
- `validation` thresholds, since they determine which rows are marked valid and therefore which
  extractor is selected per firm.

## Archive naming

The hash is a lookup key, not a decoder. A hash is one-way, and a reversible encoding of a
1,000-CIK universe runs past filename limits.

Human-readable prefix plus 8 hex characters of the hash:

```
10K10Q_2004-2025_msciworld_i1-1a-3-7-7a_a3f9c1e2.zip
```

The full resolved config lives inside the archive at `config.json`, readable without extracting
the whole archive. The name never needs decoding.

## Resume compatibility

A previously-extracted row is reusable only if it was produced under a compatible config.
Reusing rows extracted with `remove_tables: false` in a run configured with `remove_tables: true`
mixes two text representations inside one dataset, silently.

Store, per row, the hash of the **extraction-relevant subset** of the config: the row's `tool`,
its `preproc` entry, `extract_options`, `validation`, `healing.cache_version`, and the extractor
code version. Reuse only on exact match; otherwise re-extract.

This is a narrower hash than the full config hash (it excludes universe and date range, which do
not affect a given row's text) and it is what makes incremental builds across overlapping
configs correct.
