"""Historical flat files (``/history``)."""

from __future__ import annotations

import builtins
from pathlib import Path

from pydantic import TypeAdapter

from ._base import RavenPackModel, Resource, validated

__all__ = ["FlatFile", "HistoryResource"]


class FlatFile(RavenPackModel):
    """A yearly zip of monthly CSVs from the Edge historical archive."""

    id: str
    """File name, e.g. ``RavenPackEdge_FileName_2021.zip``."""
    url: str
    """Download path, relative to the data API host."""
    size: int | None = None
    """Size in bytes."""
    md5: str | None = None


_FLAT_FILES = TypeAdapter(list[FlatFile])


class HistoryResource(Resource):
    """Bulk historical archive of RavenPack Edge (yearly zips of monthly CSVs)."""

    @validated
    def list(self, flatfile_package: str) -> list[FlatFile]:
        """List the files of a flat-file package (``GET /history/{flatfile_package}``).

        Files cover up to the end of the prior month.

        Raises:
            BadRequestError: Invalid flat-file package.
            PermissionDeniedError: Not entitled to the package.
        """
        data = self._json("GET", self._http.url("data", f"history/{flatfile_package}"))
        return _FLAT_FILES.validate_python(data)

    @validated
    def download(
        self, flat_file: FlatFile, destination: str | Path, *, verify_checksum: bool = True
    ) -> Path:
        """Download one flat file (several GB) to ``destination`` (file or existing directory).

        Raises:
            ChecksumMismatchError: MD5 mismatch.
        """
        target = Path(destination)
        if target.is_dir():
            target = target / flat_file.id
        return self._http.download(
            self._http.absolute_url("data", flat_file.url),
            target,
            expected_md5=flat_file.md5 if verify_checksum else None,
        )

    @validated
    def download_package(
        self,
        flatfile_package: str,
        directory: str | Path,
        *,
        verify_checksum: bool = True,
        overwrite: bool = False,
    ) -> builtins.list[Path]:
        """Download every file of a package into ``directory``.

        Existing files are skipped unless ``overwrite`` is set.
        """
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        paths: builtins.list[Path] = []
        for flat_file in self.list(flatfile_package):
            target = root / flat_file.id
            if target.exists() and not overwrite:
                paths.append(target)
                continue
            paths.append(self.download(flat_file, target, verify_checksum=verify_checksum))
        return paths
