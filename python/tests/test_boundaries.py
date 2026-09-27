"""Architecture boundary: narrative_scoring does scoring only; all NLP is in ``nlp``.

narrative_scoring may use nlp objects (corrections, reference vectors,
embedders through ``nlp``'s own API) but never an inference stack or the
RavenPack NLP jobs directly. Checked on the AST, so lazy imports inside
functions count too.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
FORBIDDEN = ("ravenbert", "torch", "transformers", "sentence_transformers", "httpx",
             "openai", "ray", "ravenpack.headlines.embed", "ravenpack.headlines.sentiment_model")


def forbidden_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        else:
            continue
        for name in names:
            if any(name == f or name.startswith(f + ".") for f in FORBIDDEN):
                found.append(f"{path.name}:{node.lineno} imports {name}")
    return found


def test_narrative_scoring_has_no_nlp_stack_imports():
    files = list((SRC / "narrative_scoring").rglob("*.py"))
    assert files
    offenders = [hit for p in files for hit in forbidden_imports(p)]
    assert offenders == []


@pytest.mark.parametrize("code", ["import torch\n", "from ravenbert.embedding.model import X\n",
                                  "def f():\n    from ravenpack.headlines.embed import g\n"])
def test_the_check_catches_a_forbidden_import(tmp_path, code):
    bad = tmp_path / "bad.py"
    bad.write_text(code)
    assert forbidden_imports(bad)
