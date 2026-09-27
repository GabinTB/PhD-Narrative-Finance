"""datalake.lineage: recorded sources vs the inputs a run combines."""
from __future__ import annotations

import logging

import pytest

from datalake import DatalakeIndex
from datalake.lineage import LineageError, lineage_problems, require_lineage


@pytest.fixture
def lake(tmp_path):
    index = DatalakeIndex(tmp_path / "lake")
    ids = {}
    for name in ("a", "b"):
        with index.run(kind="parent", pipeline="t", pipeline_version="v0",
                       hyperparams={"name": name}) as r:
            pass
        ids[name] = r.artifact_id
    with index.run(kind="child", pipeline="t", pipeline_version="v0",
                   hyperparams={"n": 1}, sources=[ids["a"]]) as r:
        pass
    child = index.get(r.artifact_id)
    with index.run(kind="orphan", pipeline="t", pipeline_version="v0") as r:
        pass
    yield index, ids, child, index.get(r.artifact_id)
    index.close()


def test_matching_mismatching_and_unrecorded(lake):
    _, ids, child, orphan = lake
    assert lineage_problems(child, {"parent": ids["a"]}) == ([], [])
    mismatches, _ = lineage_problems(child, {"parent": ids["b"]})
    assert len(mismatches) == 1 and ids["a"] in mismatches[0] and ids["b"] in mismatches[0]
    assert lineage_problems(orphan, {"parent": ids["a"]}) == ([], ["parent"])
    assert lineage_problems(child, {"parent": None}) == ([], [])       # not checked


def test_require_lineage_raises_on_mismatch_and_warns_when_unrecorded(lake, caplog):
    _, ids, child, orphan = lake
    with pytest.raises(LineageError, match="refusing to mix"):
        require_lineage([(child, {"parent": ids["b"]}), (None, {"parent": "x"})])
    with caplog.at_level(logging.WARNING, logger="datalake.lineage"):
        require_lineage([(orphan, {"parent": ids["a"]}), (child, {"parent": ids["a"]})])
    assert "cannot be verified" in caplog.text and orphan.artifact_id in caplog.text
