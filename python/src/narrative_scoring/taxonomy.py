"""Taxonomy abstraction over evergreen and RavenPack primitive taxonomies.

Both taxonomy families share a physical column schema.  This module maps
those physical columns onto a canonical 4-level hierarchy so the rest of the
pipeline never has to know which family it is working with:

    reservoir   <- TOPIC
    dimension   <- GROUP
    narrative   <- CATEGORY  (TYPE + "-" + SUB_TYPE if CATEGORY is absent)
    primitive   <- DISPLAY_NAME
    description <- DESCRIPTION

RavenPack CSVs additionally carry a ROLE column (entity role); it plays no
part in the hierarchy but is carried through as metadata rather than dropped.

A taxonomy VERSION directory holds three required files and three optional
ones (the garbage catcher -- some taxonomies, e.g. vendor-supplied ones,
ship without one):

    {family}_taxonomy.csv
    {family}_taxonomy_primitive_paraphrases.jsonl               (pure)
    {family}_taxonomy_primitive_paraphrases-headlined.jsonl     (headlined)
    garbage-catching_taxonomy.csv                                  [optional]
    garbage-catching_taxonomy_primitive_paraphrases.jsonl          [optional]
    garbage-catching_taxonomy_primitive_paraphrases-headlined.jsonl [optional]

The garbage catcher is optional as a *set*: if ``garbage-catching_taxonomy.csv``
is absent, the whole garbage catcher is treated as absent (``has_garbage``
is False) and none of its checks run; if it is present, all three garbage
files are required and validated exactly as the primary taxonomy's are.

``load_taxonomy`` validates every file present before handing back a
``TaxonomyVersion``; ``TaxonomyError`` lists every problem found, not just
the first.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
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

_REQUIRED_PHYSICAL = {"TOPIC", "GROUP", "DISPLAY_NAME", "DESCRIPTION"}


class TaxonomyError(ValueError):
    """Raised when a taxonomy version fails structural validation."""


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------

def _read_taxonomy_csv(path: Path) -> pl.DataFrame:
    """Read a taxonomy CSV with every column as String, no dtype inference.

    Vendor taxonomy CSVs (e.g. RavenPack's own) carry columns we never use
    (like a SCHEDULED flag) that mix True/False with sentinel values such as
    "UNDEFINED" -- polars' schema inference samples the first N rows, guesses
    bool, then raises a parse error deep into the file. Since only a handful
    of known string columns are ever selected (see _canonicalize), inference
    buys nothing and turning it off makes every vendor taxonomy load
    regardless of what its other columns contain.
    """
    return pl.read_csv(path, infer_schema_length=0)


def _read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    """primitive id -> its JSONL record ({"id", "master", "paraphrases"})."""
    out: dict[str, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        out[rec["id"]] = rec
    return out


# ---------------------------------------------------------------------------
# Canonicalization
# ---------------------------------------------------------------------------

def _canonicalize(df: pl.DataFrame) -> pl.DataFrame:
    """Map a physical taxonomy CSV frame onto the canonical hierarchy.

    Constructs CATEGORY from TYPE + "-" + SUB_TYPE when CATEGORY itself is
    not already a column.  ROLE, when present, is carried through unchanged
    (RavenPack garbage catcher only) rather than dropped or folded into the
    hierarchy.
    """
    if "CATEGORY" not in df.columns:
        if not {"TYPE", "SUB_TYPE"}.issubset(df.columns):
            raise TaxonomyError(
                "taxonomy frame has neither CATEGORY nor TYPE+SUB_TYPE columns"
            )
        df = df.with_columns(
            (pl.col("TYPE") + "-" + pl.col("SUB_TYPE")).alias("CATEGORY")
        )

    missing = (_REQUIRED_PHYSICAL | {"CATEGORY"}) - set(df.columns)
    if missing:
        raise TaxonomyError(f"taxonomy frame missing required column(s): {sorted(missing)}")

    keep = ["TOPIC", "GROUP", "CATEGORY", "DISPLAY_NAME", "DESCRIPTION"]
    if "ROLE" in df.columns:
        keep.append("ROLE")

    return df.select(keep).rename(
        {
            "TOPIC": "reservoir",
            "GROUP": "dimension",
            "CATEGORY": "narrative",
            "DISPLAY_NAME": "primitive",
            "DESCRIPTION": "description",
        }
    )


# ---------------------------------------------------------------------------
# TaxonomyVersion
# ---------------------------------------------------------------------------

@dataclass
class TaxonomyVersion:
    """One versioned taxonomy directory: primitives, paraphrases, garbage catcher."""

    version: str
    path: Path
    family: str = "evergreen"
    paraphrase_style: str = "headlined"

    # -- file layout ---------------------------------------------------

    def _csv_path(self, prefix: str) -> Path:
        return self.path / f"{prefix}_taxonomy.csv"

    def _jsonl_path(self, prefix: str, style: str) -> Path:
        suffix = "" if style == "pure" else "-headlined"
        return self.path / f"{prefix}_taxonomy_primitive_paraphrases{suffix}.jsonl"

    @property
    def csv_path(self) -> Path:
        return self._csv_path(self.family)

    @property
    def garbage_csv_path(self) -> Path:
        return self._csv_path("garbage-catching")

    def jsonl_path(self, style: str | None = None) -> Path:
        return self._jsonl_path(self.family, style or self.paraphrase_style)

    def garbage_jsonl_path(self, style: str | None = None) -> Path:
        return self._jsonl_path("garbage-catching", style or self.paraphrase_style)

    @property
    def has_garbage(self) -> bool:
        """Whether this taxonomy version ships a garbage catcher at all.

        Detected from the CSV's presence alone (the JSONLs are required to
        follow if the CSV is there -- see validate()); some taxonomies, e.g.
        vendor-supplied ones, are shipped without any garbage catcher.
        """
        return self.garbage_csv_path.exists()

    def _require_garbage(self) -> None:
        if not self.has_garbage:
            raise TaxonomyError(
                f"{self.family!r} {self.version!r} at {self.path} has no garbage-catching "
                "taxonomy (garbage-catching_taxonomy.csv not found); check has_garbage first"
            )

    # -- canonical frames ------------------------------------------------

    def primitives(self) -> pl.DataFrame:
        """Canonical frame: reservoir, dimension, narrative, primitive, description."""
        return _canonicalize(_read_taxonomy_csv(self.csv_path))

    def garbage_primitives(self) -> pl.DataFrame:
        """Same canonical frame for the garbage catcher. Raises if has_garbage is False."""
        self._require_garbage()
        return _canonicalize(_read_taxonomy_csv(self.garbage_csv_path))

    # -- paraphrases / masters -------------------------------------------

    def paraphrases(self) -> dict[str, list[str]]:
        """primitive -> [paraphrase, ...]. Reads the style selected at load time."""
        return {
            pid: rec["paraphrases"]
            for pid, rec in _read_jsonl(self.jsonl_path()).items()
        }

    def masters(self) -> dict[str, str]:
        """primitive -> master description from the JSONL (not the CSV DESCRIPTION)."""
        return {
            pid: rec["master"]
            for pid, rec in _read_jsonl(self.jsonl_path()).items()
        }

    def garbage_paraphrases(self) -> dict[str, list[str]]:
        """Raises if has_garbage is False."""
        self._require_garbage()
        return {
            pid: rec["paraphrases"]
            for pid, rec in _read_jsonl(self.garbage_jsonl_path()).items()
        }

    def garbage_masters(self) -> dict[str, str]:
        """Raises if has_garbage is False."""
        self._require_garbage()
        return {
            pid: rec["master"]
            for pid, rec in _read_jsonl(self.garbage_jsonl_path()).items()
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

        Checks: unique primitives; non-empty descriptions; CSV<->JSONL
        bijection for the taxonomy (pure and headlined styles), and for the
        garbage catcher too IF one is present (has_garbage -- entirely
        skipped otherwise, e.g. vendor taxonomies that ship without one);
        constant K within each file (K may differ between taxonomy and
        garbage -- that is legal); no primitive-name overlap between taxonomy
        and garbage; every JSONL entry has a non-empty 'master'.
        """
        errors: list[str] = []

        try:
            tax_csv = _read_taxonomy_csv(self.csv_path)
        except (FileNotFoundError, pl.exceptions.PolarsError) as exc:
            return [f"cannot read taxonomy CSV ({self.csv_path.name}): {exc}"]

        has_garbage = self.has_garbage
        gc_csv: pl.DataFrame | None = None
        if has_garbage:
            try:
                gc_csv = _read_taxonomy_csv(self.garbage_csv_path)
            except (FileNotFoundError, pl.exceptions.PolarsError) as exc:
                return [f"cannot read garbage CSV ({self.garbage_csv_path.name}): {exc}"]

        for col in ("DISPLAY_NAME", "DESCRIPTION"):
            if col not in tax_csv.columns:
                errors.append(f"taxonomy CSV missing column: {col}")
        if has_garbage and "DISPLAY_NAME" not in gc_csv.columns:
            errors.append("garbage CSV missing DISPLAY_NAME column")

        if "DISPLAY_NAME" not in tax_csv.columns:
            return errors  # nothing further can be checked without primitive IDs
        if has_garbage and "DISPLAY_NAME" not in gc_csv.columns:
            return errors

        tax_ids = tax_csv["DISPLAY_NAME"].to_list()
        if len(tax_ids) != len(set(tax_ids)):
            errors.append("taxonomy CSV has duplicate DISPLAY_NAME values")

        gc_ids: list[str] = []
        if has_garbage:
            gc_ids = gc_csv["DISPLAY_NAME"].to_list()
            if len(gc_ids) != len(set(gc_ids)):
                errors.append("garbage CSV has duplicate DISPLAY_NAME values")

        if "DESCRIPTION" in tax_csv.columns:
            empty = tax_csv.filter(
                pl.col("DESCRIPTION").is_null() | (pl.col("DESCRIPTION") == "")
            )
            if len(empty) > 0:
                errors.append(
                    f"taxonomy CSV has {len(empty)} empty DESCRIPTION value(s): "
                    f"{empty['DISPLAY_NAME'].to_list()[:5]}"
                )

        tax_set = set(tax_ids)
        gc_set = set(gc_ids)

        jsonl_specs = [
            ("taxonomy pure", self.jsonl_path("pure"), tax_set),
            ("taxonomy headlined", self.jsonl_path("headlined"), tax_set),
        ]
        if has_garbage:
            jsonl_specs += [
                ("garbage pure", self.garbage_jsonl_path("pure"), gc_set),
                ("garbage headlined", self.garbage_jsonl_path("headlined"), gc_set),
            ]
        parsed: dict[str, dict[str, dict[str, Any]]] = {}
        for name, path, csv_set in jsonl_specs:
            if not path.exists():
                errors.append(f"{name} JSONL file missing: {path.name}")
                continue
            try:
                data = _read_jsonl(path)
            except (json.JSONDecodeError, KeyError) as exc:
                errors.append(f"{name} JSONL unreadable: {exc}")
                continue
            parsed[name] = data
            keys = set(data.keys())
            if keys != csv_set:
                missing = csv_set - keys
                extra = keys - csv_set
                errors.append(
                    f"{name} JSONL mismatch with CSV: missing={len(missing)}, extra={len(extra)}"
                )

        for name, data in parsed.items():
            ks = {len(v.get("paraphrases", [])) for v in data.values()}
            if len(ks) > 1:
                errors.append(f"{name}: inconsistent paraphrase counts (K): {sorted(ks)}")

            missing_master = [pid for pid, v in data.items() if not v.get("master")]
            if missing_master:
                errors.append(f"{name}: {len(missing_master)} entries missing 'master'")

        overlap = tax_set & gc_set
        if overlap:
            errors.append(
                f"{len(overlap)} DISPLAY_NAME value(s) appear in both taxonomy and garbage: "
                f"{sorted(overlap)[:5]}"
            )

        return errors


def load_taxonomy(
    root: Path,
    version: str,
    *,
    family: str = "evergreen",
    paraphrase_style: str = "headlined",
) -> TaxonomyVersion:
    """Load and validate. Raises TaxonomyError listing all problems if invalid."""
    path = Path(root) / version
    tv = TaxonomyVersion(
        version=version, path=path, family=family, paraphrase_style=paraphrase_style
    )
    errors = tv.validate()
    if errors:
        detail = "\n".join(f"  - {e}" for e in errors)
        raise TaxonomyError(
            f"taxonomy {family!r} {version!r} at {path} failed validation:\n{detail}"
        )
    return tv
