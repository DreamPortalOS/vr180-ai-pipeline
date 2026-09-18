"""Tests for studio graph engine and M0 nodes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from studio.graph import GraphError, list_node_types, run_graph, topological_order
from studio.models import EdgeSpec, NodeSpec, Project, empty_demo_project


def test_topological_order_demo() -> None:
    order = topological_order(empty_demo_project())
    assert order.index("n_script") < order.index("n_mock") < order.index("n_export")


def test_cycle_rejected() -> None:
    project = Project(
        nodes=[
            NodeSpec(id="a", type="script.storyboard", params={"prompt": "x"}),
            NodeSpec(id="b", type="tool.preview"),
        ],
        edges=[
            EdgeSpec(id="e1", from_node="a", from_port="prompt", to_node="b", to_port="value"),
            EdgeSpec(id="e2", from_node="b", from_port="summary", to_node="a", to_port="prompt"),
        ],
    )
    with pytest.raises(GraphError, match="cycle"):
        topological_project = project  # noqa: F841
        topological_order(project)


def test_run_demo_mock_export(tmp_path) -> None:
    demo = empty_demo_project()
    # Keep CI fast: tiny mock geometry via node params.
    for node in demo.nodes:
        if node.type == "video.mock":
            node.params["size"] = 64
            node.params["fps"] = 5
            node.params["duration"] = 1

    report = run_graph(demo, work_dir=str(tmp_path))
    assert report.results["n_script"].status == "ok"
    assert report.results["n_mock"].status == "ok"
    assert report.results["n_export"].status == "ok"

    video = report.results["n_mock"].outputs["video"]
    assert Path(video).is_file()
    export_path = report.results["n_export"].outputs["path"]
    assert Path(export_path).is_file()
    manifest = report.results["n_export"].outputs["manifest"]
    assert Path(manifest["manifest_path"]).is_file()
    assert json.loads(Path(manifest["manifest_path"]).read_text(encoding="utf-8"))["export_path"] == export_path


def test_cache_hit_on_second_run(tmp_path) -> None:
    demo = empty_demo_project()
    for node in demo.nodes:
        if node.type == "video.mock":
            node.params["size"] = 64
            node.params["fps"] = 5
            node.params["duration"] = 1
    shared_cache: dict = {}
    run_graph(demo, work_dir=str(tmp_path), cache=shared_cache)
    report2 = run_graph(demo, work_dir=str(tmp_path), cache=shared_cache)
    assert report2.results["n_mock"].cache_hit is True
    assert report2.results["n_export"].cache_hit is True


def test_empty_prompt_fails() -> None:
    project = Project(
        nodes=[NodeSpec(id="s", type="script.storyboard", params={"prompt": "  "})],
        edges=[],
    )
    with pytest.raises(GraphError, match="prompt must not be empty"):
        run_graph(project, work_dir=".")


def test_list_node_types_includes_m0() -> None:
    types = {t["type"] for t in list_node_types()}
    assert {"script.storyboard", "video.mock", "tool.preview", "export.bundle"} <= types
