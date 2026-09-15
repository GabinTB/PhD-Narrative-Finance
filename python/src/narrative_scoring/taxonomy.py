"""Taxonomy abstraction over the Narrative_Taxonomy monorepo.

Taxonomies live flat, by name, directly under one root directory
(``RAW_DATA_PATH/Narrative_Taxonomy``), each as three files:

    {name}_taxonomy.authored.csv                        (required -- the ONLY csv ever read)
    {name}-primitive_semantic_paraphrases.jsonl          (required)
    {name}-primitive_headline_paraphrases.jsonl          (required)

A sibling ``{name}_taxonomy.csv`` (no ``.authored``) may exist alongside the
authored one -- it is a generated skeleton with ``DISPLAY_NAME``/
``DESCRIPTION`` blank, produced before authoring, and this module never
reads it. A taxonomy is *complete* exactly when the three files above exist
and cross-validate (see ``validate()``); a ``Legacy/`` subdirectory of the
root holds deprecated material this module never looks at.

Every taxonomy is treated as a RavenPack-schema child: ``primitives()``
always returns the full ``RAVENPACK_BASE_SCHEMA`` column set (missing ones
null-filled, never an error) plus whatever extra columns the source CSV
carries (e.g. ``OBSERVABILITY_CHANNEL``, ``POLARITY``), verbatim-named.
Pass a custom, extended ``base_schema`` (e.g. ``RAVENPACK_BASE_SCHEMA +
["MY_COL"]``) to widen what's guaranteed present.

Column mapping onto the canonical 4-level hierarchy:

    reservoir   <- TOPIC
    dimension   <- GROUP
    narrative   <- CATEGORY  (TYPE + "-" + SUB_TYPE if CATEGORY is absent)
    primitive   <- DISPLAY_NAME
    description <- DESCRIPTION

JSONL records are keyed by their ``display_name`` field for the CSV join --
NOT by ``id`` (an opaque per-record hash, not the primitive name).

``load_taxonomy`` validates all three files before handing back a
``TaxonomyVersion``; ``TaxonomyError`` lists every problem found, not just
the first.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl

# ---------------------------------------------------------------------------
# Canonical column mapping
# ---------------------------------------------------------------------------

CANONICAL_COLUMNS: dict[str, str] = {
    "reservoir":   "TOPIC",
    "dimension":   "GROUP",
    "narrative":   "CATEGORY",
    "primitive":   "DISPLAY_NAME",
    "description": "DESCRIPTION",
}

_CORE_REQUIRED = {"TOPIC", "GROUP", "DISPLAY_NAME", "DESCRIPTION"}

# The reference RavenPack taxonomy schema (Legacy/Vendor/RavenPack/vendor_taxonomy.csv).
# Every taxonomy is normalized to carry at least these columns -- missing ones
# are added as null, never an error. Import and extend this to require more:
#   base_schema = RAVENPACK_BASE_SCHEMA + ["MY_EXTRA_COL"]
RAVENPACK_BASE_SCHEMA: list[str] = [
    "TOPIC", "GROUP", "TYPE", "SUB_TYPE", "ROLE", "CATEGORY",
    "DISPLAY_NAME", "DESCRIPTION", "SCHEDULED", "VALID_ENTITY_TYPES", "TAGS",
]

PARAPHRASE_STYLES = ("semantic", "headline")


class TaxonomyError(ValueError):
    """Raised when a taxonomy fails structural validation."""


# ---------------------------------------------------------------------------
# File readers
# ---------------------------------------------------------------------------

def _read_taxonomy_csv(path: Path) -> pl.DataFrame:
    """Read a taxonomy CSV with every column as String, no dtype inference.

    Vendor taxonomy CSVs (e.g. RavenPack's own) carry columns we never use
    (like a SCHEDULED flag) that mix True/False with sentinel values such as
    "UNDEFINED" -- polars' schema inference samples the first N rows, guesses
    bool, then raises a parse error deep into the file. Since only a handful
    of known columns are ever selected (see _canonicalize), inference buys
    nothing and turning it off makes every taxonomy load regardless of what
    its other columns contain.
    """
    return pl.read_csv(path, infer_schema_length=0)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Parse a paraphrases JSONL into its raw list of records.

    Each record carries at least {"id", "display_name", "master",
    "paraphrases"} -- "id" is an opaque per-record hash, NOT usable as a
    join key; use _index_by_display_name for that.
    """
    records: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def _index_by_display_name(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {rec["display_name"]: rec for rec in records if "display_name" in rec}


# ---------------------------------------------------------------------------
# Canonicalization
# ---------------------------------------------------------------------------

def _canonicalize(df: pl.DataFrame, base_schema: list[str]) -> pl.DataFrame:
    """Map a physical taxonomy CSV frame onto the canonical hierarchy.

    Constructs CATEGORY from TYPE + "-" + SUB_TYPE when CATEGORY itself is
    not already a column. Every column in base_schema not already present
    is added as an all-null String column ("mandatory, even if empty");
    columns beyond base_schema present in the source are carried through
    verbatim as extra metadata rather than dropped.
    """
    if "CATEGORY" not in df.columns:
        if not {"TYPE", "SUB_TYPE"}.issubset(df.columns):
            raise TaxonomyError(
                "taxonomy frame has neither CATEGORY nor TYPE+SUB_TYPE columns"
            )
        df = df.with_columns(
            (pl.col("TYPE") + "-" + pl.col("SUB_TYPE")).alias("CATEGORY")
        )

    missing_core = (_CORE_REQUIRED | {"CATEGORY"}) - set(df.columns)
    if missing_core:
        raise TaxonomyError(f"taxonomy frame missing required column(s): {sorted(missing_core)}")

    for col in base_schema:
        if col not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=pl.String).alias(col))

    rename_map = {
        "TOPIC": "reservoir",
        "GROUP": "dimension",
        "CATEGORY": "narrative",
        "DISPLAY_NAME": "primitive",
        "DESCRIPTION": "description",
    }
    ordered_baseline = [c for c in base_schema if c not in rename_map]
    extra_cols = [c for c in df.columns if c not in base_schema and c not in rename_map]
    keep = list(rename_map.keys()) + ordered_baseline + extra_cols

    return df.select(keep).rename(rename_map)


# ---------------------------------------------------------------------------
# TaxonomyVersion
# ---------------------------------------------------------------------------

@dataclass
class TaxonomyVersion:
    """One named taxonomy in the Narrative_Taxonomy monorepo."""

    name: str                     # e.g. "Evergreen_v5"
    path: Path                    # the Narrative_Taxonomy root (flat, no per-taxonomy subdir)
    paraphrase_style: str = "headline"    # "semantic" | "headline"
    base_schema: list[str] = field(default_factory=lambda: list(RAVENPACK_BASE_SCHEMA))

    # -- file layout ---------------------------------------------------

    @property
    def csv_path(self) -> Path:
        return self.path / f"{self.name}_taxonomy.authored.csv"

    def jsonl_path(self, style: str | None = None) -> Path:
        style = style or self.paraphrase_style
        return self.path / f"{self.name}-primitive_{style}_paraphrases.jsonl"

    # -- canonical frame ------------------------------------------------

    def primitives(self) -> pl.DataFrame:
        """Canonical frame: reservoir, dimension, narrative, primitive, description,
        plus every base_schema column (null-filled if absent) and any extras."""
        return _canonicalize(_read_taxonomy_csv(self.csv_path), self.base_schema)

    # -- paraphrases / masters -------------------------------------------

    def paraphrases(self) -> dict[str, list[str]]:
        """primitive -> [paraphrase, ...]. Reads the style selected at load time."""
        return {
            name: rec["paraphrases"]
            for name, rec in _index_by_display_name(_read_jsonl(self.jsonl_path())).items()
        }

    def masters(self) -> dict[str, str]:
        """primitive -> master description from the JSONL (not the CSV DESCRIPTION)."""
        return {
            name: rec["master"]
            for name, rec in _index_by_display_name(_read_jsonl(self.jsonl_path())).items()
        }

    # -- K ----------------------------------------------------------------

    @property
    def k(self) -> int:
        """Paraphrases per primitive (validated to be constant within the file)."""
        counts = {len(v) for v in self.paraphrases().values()}
        if len(counts) > 1:
            raise TaxonomyError(f"inconsistent paraphrase counts (K): {sorted(counts)}")
        return next(iter(counts), 0)

    # -- validation ---------------------------------------------------------

    def validate(self) -> list[str]:
        """Return a list of error strings; empty means valid.

        Checks: the authored CSV exists and is readable; DISPLAY_NAME and
        DESCRIPTION are both required non-empty (that's what "authored"
        means); unique primitives; both semantic and headline paraphrase
        JSONLs exist, CSV<->JSONL bijection (by display_name), constant K
        per file, every JSONL entry has a non-empty 'master'.
        """
        errors: list[str] = []

        if not self.csv_path.exists():
            return [f"authored taxonomy CSV missing: {self.csv_path.name}"]
        try:
            tax_csv = _read_taxonomy_csv(self.csv_path)
        except (FileNotFoundError, pl.exceptions.PolarsError) as exc:
            return [f"cannot read taxonomy CSV ({self.csv_path.name}): {exc}"]

        for col in ("DISPLAY_NAME", "DESCRIPTION"):
            if col not in tax_csv.columns:
                errors.append(f"taxonomy CSV missing column: {col}")
        if "DISPLAY_NAME" not in tax_csv.columns:
            return errors  # nothing further can be checked without primitive names

        tax_names = tax_csv["DISPLAY_NAME"].to_list()
        if len(tax_names) != len(set(tax_names)):
            errors.append("taxonomy CSV has duplicate DISPLAY_NAME values")

        empty_names = tax_csv.filter(
            pl.col("DISPLAY_NAME").is_null() | (pl.col("DISPLAY_NAME") == "")
        )
        if len(empty_names) > 0:
            errors.append(f"taxonomy CSV has {len(empty_names)} empty DISPLAY_NAME value(s)")

        if "DESCRIPTION" in tax_csv.columns:
            empty_desc = tax_csv.filter(
                pl.col("DESCRIPTION").is_null() | (pl.col("DESCRIPTION") == "")
            )
            if len(empty_desc) > 0:
                errors.append(
                    f"taxonomy CSV has {len(empty_desc)} empty DESCRIPTION value(s): "
                    f"{empty_desc['DISPLAY_NAME'].to_list()[:5]}"
                )

        tax_set = set(tax_names)

        for style in PARAPHRASE_STYLES:
            path = self.jsonl_path(style)
            if not path.exists():
                errors.append(f"{style} paraphrases JSONL file missing: {path.name}")
                continue
            try:
                records = _read_jsonl(path)
            except (json.JSONDecodeError, KeyError) as exc:
                errors.append(f"{style} paraphrases JSONL unreadable: {exc}")
                continue

            display_names = [r.get("display_name") for r in records]
            if len(display_names) != len(set(display_names)):
                errors.append(f"{style} paraphrases JSONL has duplicate display_name entries")

            keys = set(display_names)
            if keys != tax_set:
                missing = tax_set - keys
                extra = keys - tax_set
                errors.append(
                    f"{style} paraphrases JSONL mismatch with CSV: "
                    f"missing={len(missing)}, extra={len(extra)}"
                )

            ks = {len(r.get("paraphrases", [])) for r in records}
            if len(ks) > 1:
                errors.append(
                    f"{style} paraphrases: inconsistent paraphrase counts (K): {sorted(ks)}"
                )

            missing_master = [r.get("display_name") for r in records if not r.get("master")]
            if missing_master:
                errors.append(
                    f"{style} paraphrases: {len(missing_master)} entries missing 'master'"
                )

        return errors


def load_taxonomy(
    root: Path,
    name: str,
    *,
    paraphrase_style: str = "headline",
    base_schema: list[str] | None = None,
) -> TaxonomyVersion:
    """Load and validate. Raises TaxonomyError listing all problems if invalid.

    ``root`` is the Narrative_Taxonomy directory itself (taxonomies live
    flat, by name, directly under it -- no per-taxonomy subdirectory).
    """
    tv = TaxonomyVersion(
        name=name,
        path=Path(root),
        paraphrase_style=paraphrase_style,
        base_schema=list(base_schema) if base_schema is not None else list(RAVENPACK_BASE_SCHEMA),
    )
    errors = tv.validate()
    if errors:
        detail = "\n".join(f"  - {e}" for e in errors)
        raise TaxonomyError(f"taxonomy {name!r} at {tv.path} failed validation:\n{detail}")
    return tv


def list_taxonomies(root: Path) -> list[str]:
    """Names of every complete-or-not taxonomy directly under root (never Legacy/).

    A name is discovered from `{name}_taxonomy.authored.csv`; call
    load_taxonomy(root, name) to validate a given one before using it.
    """
    root = Path(root)
    suffix = "_taxonomy.authored.csv"
    return sorted(
        p.name[: -len(suffix)]
        for p in root.glob(f"*{suffix}")
        if p.is_file()
    )
