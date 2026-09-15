"""Narrative scoring: taxonomy-based headline scoring pipeline.

Scores RavenPack headline embeddings against a named taxonomy of
"primitive" narrative descriptions, with:

  - taxonomy.py      Named taxonomies from the Narrative_Taxonomy monorepo,
                      mapped onto a canonical 4-level hierarchy.
  - descriptions.py   Embeds taxonomy descriptions (centroid/max/median pooling).
  - corrections.py    RAW / R1 (mean-centering) / R2 (mean-direction removal).
  - null_model.py      The F0 rejection floor (per-headline trim, pooled null,
                      effective-N, empirical quantile) -- the sole rejection
                      layer; there is no garbage-catcher layer.
  - scoring.py          Per-day/per-month scoring driver + datalake entry point.
  - aggregate.py        Roll primitive-level daily scores up the hierarchy.
  - schema.py            Polars schemas for every artifact this package writes.

See each module's docstring for the exact formulas -- they are specified
precisely and must not be approximated.
"""
