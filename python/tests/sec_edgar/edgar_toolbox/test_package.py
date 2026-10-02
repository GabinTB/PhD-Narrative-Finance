"""Smoke test for the edgar_tools package skeleton (task 01).

Real behaviour arrives with later tasks; this only pins that the package is
importable, typed (py.typed present), and carries a version string.
"""

from pathlib import Path

import sec_edgar.edgar_toolbox as edgar_toolbox


def test_imports() -> None:
    assert edgar_toolbox.__version__ == "0.0.0"


def test_py_typed_marker_present() -> None:
    package_dir = Path(edgar_toolbox.__file__).parent
    assert (package_dir / "py.typed").exists()
