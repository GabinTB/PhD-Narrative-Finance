"""Sidecar writers: meta.json, README.md, and file hashing.

Every artifact directory carries two sidecars:

  meta.json   machine-readable RunMeta plus file hashes.  The index is a cache
              over these; `datalake reindex` rebuilds index.db from them.

  README.md   the same content rendered for a human browsing the tree in a
              file manager or on Google Drive, where no tooling is available.

Both are written atomically (tmp + replace) so a crash never leaves a sidecar
that parses but lies.
"""
from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from pathlib import Path
from typing import Any

from datalake.artifact import RunMeta

log = logging.getLogger(__name__)

META_FILENAME = "meta.json"
README_FILENAME = "README.md"

# blake2b over xxhash: stdlib, no new dependency, ~1GB/s which is fast enough
# for a once-per-artifact operation.  Swap here if hashing ever dominates.
HASH_ALGORITHM = "blake2b"
_HASH_DIGEST_SIZE = 32          # 256-bit digest, hex-encodes to 64 chars
_HASH_CHUNK_BYTES = 8 * 1024 * 1024


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def hash_file(path: Path) -> tuple[str, int]:
    """Return (hex digest, size in bytes) for one file.

    Streams in chunks so a multi-GB parquet never lands in memory.
    """
    digest = hashlib.blake2b(digest_size=_HASH_DIGEST_SIZE)
    size = 0
    with path.open("rb") as fh:
        while chunk := fh.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def hash_directory(
    directory: Path,
    pattern: str = "*",
    exclude: frozenset[str] = frozenset({META_FILENAME, README_FILENAME}),
) -> dict[str, dict[str, Any]]:
    """Hash every matching file directly inside `directory` (non-recursive).

    Sidecars are excluded by default: meta.json cannot contain its own hash,
    and README.md is derived from it.

    Returns {filename: {"digest": ..., "size_bytes": ..., "algorithm": ...}},
    sorted by filename so the mapping is stable across runs.
    """
    results: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.glob(pattern)):
        if not path.is_file() or path.name in exclude:
            continue
        if path.name.endswith(".tmp"):
            # A .tmp file inside a completed artifact means a crashed write.
            # Skip it here; `verify` reports it.
            log.warning("skipping stray tmp file while hashing: %s", path)
            continue
        digest, size = hash_file(path)
        results[path.name] = {
            "algorithm": HASH_ALGORITHM,
            "digest": digest,
            "size_bytes": size,
        }
    return results


# ---------------------------------------------------------------------------
# Git provenance
# ---------------------------------------------------------------------------

def git_commit(repo_dir: Path | None = None) -> str | None:
    """Current HEAD SHA, with '-dirty' appended if the tree has changes.

    Returns None when git is unavailable or the directory is not a repo: a
    missing commit is recorded honestly rather than faked, and `verify`
    reports artifacts that lack one.
    """
    cwd = str(repo_dir) if repo_dir else None
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd, stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None

    try:
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=cwd, stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        dirty = ""

    return f"{sha}-dirty" if dirty else sha


# ---------------------------------------------------------------------------
# meta.json
# ---------------------------------------------------------------------------

def write_meta(
    directory: Path,
    meta: RunMeta,
    file_hashes: dict[str, dict[str, Any]] | None = None,
) -> Path:
    """Write meta.json atomically.  Returns the written path."""
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "artifact_id": meta.artifact_id,
        **meta.to_dict(),
        "files": file_hashes or {},
    }
    path = directory / META_FILENAME
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    tmp.replace(path)
    return path


def read_meta(directory: Path) -> tuple[RunMeta, dict[str, dict[str, Any]]]:
    """Read meta.json back into (RunMeta, file_hashes).

    Raises FileNotFoundError when the sidecar is absent, which for a directory
    that looks like an artifact means a run that never completed its first
    write.
    """
    path = directory / META_FILENAME
    payload = json.loads(path.read_text())
    files = payload.pop("files", {})
    payload.pop("schema_version", None)
    payload.pop("artifact_id", None)
    return RunMeta.from_dict(payload), files


# ---------------------------------------------------------------------------
# README.md
# ---------------------------------------------------------------------------

def _format_bytes(n: int) -> str:
    step = 1024.0
    value = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < step:
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= step
    return f"{value:.1f} PB"


def render_readme(
    meta: RunMeta,
    file_hashes: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Render RunMeta as human-readable Markdown.

    Written for someone browsing the datalake with no tooling: a file manager,
    a Google Drive web view, a GitHub blob page.
    """
    files = file_hashes or {}
    lines: list[str] = [f"# {meta.artifact_id}", ""]

    if meta.partial:
        lines += [
            "> **INCOMPLETE RUN.** This artifact was never finalised; the "
            "producing run crashed or was interrupted. Do not consume it "
            "downstream.",
            "",
        ]
    if meta.deprecated:
        reason = meta.deprecation_reason or "no reason recorded"
        lines += [f"> **DEPRECATED.** {reason}", ""]

    lines += [
        f"- **Kind**: `{meta.kind}`",
        f"- **Pipeline**: `{meta.pipeline}` version `{meta.pipeline_version}`",
    ]
    if meta.pipeline_commit:
        lines.append(f"- **Commit**: `{meta.pipeline_commit}`")
    if meta.pipeline_repo:
        lines.append(f"- **Repo**: {meta.pipeline_repo}")
    lines.append(f"- **Started**: {meta.run_start}")
    lines.append(f"- **Finished**: {meta.run_end or 'never (incomplete)'}")
    lines.append("")

    if meta.hyperparams:
        lines += ["## Hyperparameters", ""]
        for key in sorted(meta.hyperparams):
            lines.append(f"- `{key}` = `{meta.hyperparams[key]!r}`")
        lines.append("")

    if meta.model_card is not None:
        card = meta.model_card
        lines += ["## Model", ""]
        lines.append(f"- **Model**: `{card.model_id}` version `{card.version}`")
        if card.architecture:
            lines.append(f"- **Architecture**: {card.architecture}")
        if card.dim is not None:
            lines.append(f"- **Dimension**: {card.dim}")
        if card.pooling:
            lines.append(f"- **Pooling**: {card.pooling}")
        if card.trained_on:
            lines.append(f"- **Trained on**: {card.trained_on}")
        lines.append(f"- **Repo**: {card.repo}")
        if card.commit:
            lines.append(f"- **Model commit**: `{card.commit}`")
        if card.weights_public:
            lines.append("- **Weights**: public, fetchable from the repo above")
        else:
            digest = card.weights_sha256 or "not recorded"
            lines.append(
                f"- **Weights**: private, not distributable "
                f"(sha256 `{digest}`)"
            )
        if card.notes:
            lines.append(f"- **Notes**: {card.notes}")
        lines.append("")

    if meta.sources:
        lines += ["## Inputs", ""]
        for source in meta.sources:
            lines.append(f"- `{source}`")
        lines.append("")

    if files:
        total = sum(entry["size_bytes"] for entry in files.values())
        algorithm = next(iter(files.values()))["algorithm"]
        lines += [
            "## Files",
            "",
            f"{len(files)} files, {_format_bytes(total)} total. "
            f"Digests are `{algorithm}`.",
            "",
            "| File | Size | Digest |",
            "| --- | ---: | --- |",
        ]
        for name in sorted(files):
            entry = files[name]
            short = entry["digest"][:16]
            lines.append(
                f"| `{name}` | {_format_bytes(entry['size_bytes'])} | `{short}...` |"
            )
        lines.append("")

    if meta.notes:
        lines += ["## Notes", "", meta.notes, ""]

    lines += [
        "---",
        "",
        "Generated by the datalake layer. Do not edit by hand: this file is "
        "regenerated from `meta.json` and any manual change will be lost.",
        "",
    ]
    return "\n".join(lines)


def write_readme(
    directory: Path,
    meta: RunMeta,
    file_hashes: dict[str, dict[str, Any]] | None = None,
) -> Path:
    """Write README.md atomically.  Returns the written path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / README_FILENAME
    tmp = path.with_suffix(".md.tmp")
    tmp.write_text(render_readme(meta, file_hashes))
    tmp.replace(path)
    return path


def write_sidecars(
    directory: Path,
    meta: RunMeta,
    file_hashes: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Write meta.json and README.md together.

    meta.json is written first: it is the source of truth, and if the process
    dies between the two writes a stale README is a cosmetic problem while a
    missing meta.json is a correctness one.
    """
    write_meta(directory, meta, file_hashes)
    write_readme(directory, meta, file_hashes)
