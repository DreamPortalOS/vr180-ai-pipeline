"""Tests for studio project model."""

from __future__ import annotations

import pytest

from studio.models import Project, StudioModelError, empty_demo_project


def test_demo_project_roundtrip() -> None:
    demo = empty_demo_project()
    text = demo.to_json()
    loaded = Project.from_json(text)
    assert loaded.name == demo.name
    assert len(loaded.nodes) == 3
    assert len(loaded.edges) == 3
    assert {n.id for n in loaded.nodes} == {"n_script", "n_mock", "n_export"}


def test_validate_rejects_unknown_edge_endpoint() -> None:
    demo = empty_demo_project()
    demo.edges[0].to_node = "missing"
    with pytest.raises(StudioModelError, match="unknown to node"):
        demo.validate()


def test_validate_rejects_duplicate_node_ids() -> None:
    demo = empty_demo_project()
    demo.nodes[1].id = demo.nodes[0].id
    with pytest.raises(StudioModelError, match="duplicate node ids"):
        demo.validate()


def test_from_dict_rejects_bad_version() -> None:
    with pytest.raises(StudioModelError, match="unsupported project version"):
        Project.from_dict({"version": 99, "nodes": [], "edges": []})


def test_save_load_path(tmp_path) -> None:
    demo = empty_demo_project()
    path = demo.save(tmp_path / "p.studio.json")
    assert path.is_file()
    loaded = Project.load(path)
    assert loaded.to_dict() == demo.to_dict()
