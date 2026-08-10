# Storage architecture

## Principle

Two kinds of data with opposite requirements. Do not force them into one store.

- **Blobs**: raw filings, derived text. Large, immutable, write-once, read-by-key.
  Filesystem, content-addressed, compressed.
- **Metadata**: which CIK, which accession, do I hold it. Small, mutable during download,
  queried by point lookup. SQLite.
- **Analytical output**: section text plus provenance. Read-mostly, scanned columnar in bulk.
  Partitioned parquet.

Putting blobs in a relational database bloats it, slows it, and kills portability. Putting
metadata on a bare filesystem means "do I have this?" degrades to a directory walk.

## Layout

```
{root}/
  catalog.db                                   # SQLite: filings, edgar_state, resolver, derived
  raw/{sha[:2]}/{sha}.txt.zst                  # full SGML submission, permanent
  derived/{preproc_id}/{sha[:2]}/{sha}.txt.zst # per-framework text, evictable cache
  extracted/{confighash}/
    config.json                                # resolved config, self-describing
    MANIFEST.json
    form=10-K/year=2020/part-*.parquet
    form=10-Q/year=2020/part-*.parquet
  exports/
    {title}_{confighash8}.zip
    extracts_index.parquet                     # searchable index of all manifests
```

## Raw tier

**Store the full SGML submission** (`{accession}.txt` as EDGAR serves it), zstd-compressed,
named by the SHA-256 of the uncompressed bytes.

Not the stripped primary document. The SGML submission is the strict superset every framework
derives from:

- `edgartools` parses it natively (it is what the `.nc` bulk bundle contains)
- `edgar-crawler`'s primary-document extraction is a projection of it
- `itemseg`'s inscriptis runs on the primary HTML pulled from it
- own-regex runs on any of the above

Store `edgartools` cannot use edgar-crawler's output: its advantage is anchors, typography and
filer-agent detection, all of which need markup that edgar-crawler has already discarded.

Not parquet. Raw filings are opaque byte streams fetched by key and never queried by column.

### Why content addressing

Naming by SHA-256 of content buys three things with no extra machinery:

- **Deduplication.** A re-requested filing, or an amendment reusing content, costs nothing.
- **Integrity.** The name is the checksum, so corruption is detectable without a side table.
- **Incremental export.** Same sha means same bytes, so shipping to another environment is a
  diff, not a full copy.

### Size scoping

The full EDGAR corpus is over 1,000 TB (roughly 20 million filings, 100 million exhibits, 400+
form types). Do not mirror it.

- 10-K alone: ~30 GB compressed, >300 GB uncompressed, all filers.
- Restricted to a research universe (order 1,500 firms, not 800,000 filers) and to 10-K, 10-Q,
  8-K, with exhibits and binaries dropped: tens of GB compressed.

Mirror EDGAR's *index* completely (see `catalog-and-state.md`); mirror its *documents*
selectively.

## Derived tier

A pure function of `(raw_sha, preproc_id)`. Therefore a cache.

`preproc_id` identifies the library, its version, and its options, e.g.
`inscriptis-2.5.0`, `edgarcrawler-strip-v1.2-notables`. It must be stable and must change when
output changes.

Properties that follow from being a cache:

- **Never exported.** Ship raw and sections; the receiving environment regenerates or trusts.
- **Evictable** under disk pressure with no data loss.
- **Not a first-class catalog object.** One `derived(raw_sha, preproc_id, derived_sha)` row so
  cache warmth is queryable, nothing more.

### Why it must exist

The expensive step is `raw SGML -> (DOM | inscriptis text | strip text)`. A large modern 10-K is
5-20 MB of inline-XBRL-bloated HTML; building a rich DOM over that is seconds. `edgartools` then
runs style parsing, table processing, agent detection, and a three-stage detection cascade with
cross-validation on top. inscriptis is layout-aware pure Python and comparable in order.

Local storage does not fix this. The cost is parsing, not I/O.

If that cost sits inside the iteration loop, every re-run of the analysis re-parses everything.
Persist derived text once and `force_reextract` becomes a coffee break instead of an afternoon.

### Parallelism

Parallelise across filings, not within. Each filing is independent and the work is CPU-bound
Python. Use a process pool; do not thread, the parse does not release the GIL where it matters.

## Sections tier

Permanent. Partitioned parquet. Schema and partitioning in `output-and-export.md`.

## Write ordering

Blob first, catalog second, atomically:

1. Write blob to a temp file in the same filesystem
2. fsync
3. Verify SHA-256 matches
4. Atomic rename into final path
5. Insert the catalog row in one transaction

If the catalog row goes first and the process crashes, the catalog claims a filing that is not
on disk, and every subsequent run trusts it. With this order, crash-resume falls out for free:
anything in the catalog is guaranteed present.

## Force flags

Two, because the costs differ by orders of magnitude:

- `force_refetch` — invalidates the raw tier. Re-downloads. Network-bound, hours.
- `force_reextract` — invalidates derived and sections. No network. Minutes.

The common case is re-extracting under new settings without touching EDGAR. A single
`force_redownload` flag cannot express it.

## Rate limiting

One process-wide limiter, persisted, shared across `download`, `get`'s implicit fetches, and the
daily scheduler. SEC's 10 requests/second is per IP, not per call site. If the scheduler and an
interactive `get` run concurrently with separate limiters, they collectively exceed it and the IP
is blocked.

Use a file lock or equivalent so two processes cannot both believe they own the budget.
A `User-Agent` header with a real contact email is mandatory on every request.

If an existing downloader library is used for acquisition, it must either accept an injected
limiter or be wrapped so all traffic passes one gate.

## Portability

Use fsspec paths from the first commit. Local disk (`/data/edgar`) and MS Fabric OneLake
(`abfss://...`) then differ by a config string. Both are parquet-in-a-lakehouse; there is no
storage-engine fork to plan for.

Keep the EDGAR store physically separate from anything vendor-licensed (RavenPack, MSCI
constituents). EDGAR filings are public disclosure and freely redistributable; co-locating
licensed data in the same tree makes the whole subtree non-exportable.

## Schema versioning

`catalog.db`, every archive, and the state export carry a `schema_version`. Without it, the
first schema change makes `rebuild_db_from_state` from an older export guesswork.
