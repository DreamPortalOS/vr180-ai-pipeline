"""Export cache receipts must still describe materialized files (issue #448)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from studio.graph import run_graph
from studio.models import EdgeSpec, NodeSpec, Project
from studio.nodes.base import StudioNode
from studio.nodes.registry import NODE_REGISTRY


class LocalSource(StudioNode):
    def run(self, *, params, inputs, work_dir, node_id):
        return {"video": params["path"]}


@pytest.fixture
def setup_export(tmp_path, monkeypatch):
    monkeypatch.setitem(NODE_REGISTRY, "test.cache_source", LocalSource)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"original local bytes")
    project = Project(
        nodes=[
            NodeSpec(id="source", type="test.cache_source", params={"path": str(source)}),
            NodeSpec(id="export", type="export.bundle", params={"filename": "original.mp4"}),
        ],
        edges=[EdgeSpec(id="edge", from_node="source", from_port="video", to_node="export", to_port="video")],
    )
    cache: dict[str, dict[str, Any]] = {}

    def run(work_root=None):
        return run_graph(project, work_dir=str(work_root or tmp_path), cache=cache).results[project.nodes[1].id]

    return source, project, cache, run


def assert_receipt(result, source):
    manifest = result.outputs["manifest"]
    disk = json.loads(Path(manifest["manifest_path"]).read_text(encoding="utf-8"))
    assert disk == {key: value for key, value in manifest.items() if key != "manifest_path"}
    dest = Path(result.outputs["path"])
    assert dest.read_bytes() == source.read_bytes()
    assert disk["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()


def test_unchanged_export_keeps_cache_hit(setup_export):
    source, _, _, run = setup_export
    assert not run().cache_hit
    second = run()
    assert second.cache_hit
    assert_receipt(second, source)


def test_filename_a_b_a_restores_manifest(setup_export):
    source, project, _, run = setup_export
    run()
    project.nodes[1].params["filename"] = "renamed.mp4"
    run()
    project.nodes[1].params["filename"] = "original.mp4"
    restored = run()
    assert not restored.cache_hit
    assert Path(restored.outputs["path"]).name == "original.mp4"
    assert_receipt(restored, source)


def test_source_a_b_a_restores_overwritten_bytes(setup_export, tmp_path):
    source, project, _, run = setup_export
    run()
    other = tmp_path / "other.mp4"
    other.write_bytes(b"different source bytes")
    project.nodes[0].params["path"] = str(other)
    run()
    project.nodes[0].params["path"] = str(source)
    restored = run()
    assert not restored.cache_hit
    assert_receipt(restored, source)


def test_changed_source_at_same_path_invalidates(setup_export):
    source, _, _, run = setup_export
    run()
    source.write_bytes(b"modified local bytes")  # same length as the old source
    modified = run()
    assert not modified.cache_hit
    assert_receipt(modified, source)


@pytest.mark.parametrize("artifact", ["video", "manifest"])
@pytest.mark.parametrize("action", ["delete", "tamper"])
def test_missing_or_tampered_artifact_recomputes(setup_export, artifact, action):
    source, _, _, run = setup_export
    first = run()
    path = Path(first.outputs["path"] if artifact == "video" else first.outputs["manifest"]["manifest_path"])
    if action == "delete":
        path.unlink()
    else:
        path.write_bytes(b"corrupted artifact")
    repaired = run()
    assert not repaired.cache_hit
    assert_receipt(repaired, source)


@pytest.mark.parametrize("scope", ["work_root", "node_id"])
def test_shared_cache_must_not_return_another_export_location(setup_export, tmp_path, scope):
    source, project, _, run = setup_export
    old = run()
    if scope == "node_id":
        project.nodes[1].id = "other_export"
        project.edges[0].to_node = "other_export"
        result = run()
    else:
        result = run(tmp_path / "other_root")
    assert not result.cache_hit
    assert result.outputs["path"] != old.outputs["path"]
    assert_receipt(result, source)


def test_legacy_cache_without_hash_recomputes(setup_export):
    source, _, _, run = setup_export
    first = run()
    first.outputs["manifest"].pop("sha256")
    repaired = run()
    assert not repaired.cache_hit
    assert_receipt(repaired, source)
