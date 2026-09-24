"""Tests for studio graph engine and M0 nodes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from studio.graph import (
    HISTORY_DEPTH,
    GraphError,
    extract_gallery,
    list_node_types,
    reorder_shots,
    run_graph,
    topological_order,
)
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


# ---------------------------------------------------------------------------
# issue #418: per-node run history + drawer reorder helper
# ---------------------------------------------------------------------------


def test_run_history_records_each_run(tmp_path) -> None:
    """Every run appends to the node's history; the canvas ◀▶ reads from it."""
    demo = empty_demo_project()
    for node in demo.nodes:
        if node.type == "video.mock":
            node.params["size"] = 64
            node.params["fps"] = 5
            node.params["duration"] = 1
    shared_cache: dict = {}
    history: dict = {}
    run_graph(demo, work_dir=str(tmp_path), cache=shared_cache, history=history)
    run_graph(demo, work_dir=str(tmp_path), cache=shared_cache, history=history)
    # n_mock ran twice (once cold, once cache-hit) → 2 entries.
    entries = history["n_mock"].recent()
    assert len(entries) == 2
    assert entries[0]["status"] == "ok"
    assert entries[1]["cache_hit"] is True
    # Report also carries a history snapshot.
    report = run_graph(demo, work_dir=str(tmp_path), cache=shared_cache, history=history)
    assert "history" in report.to_dict()
    assert len(report.to_dict()["history"]["n_mock"]) >= 1


def test_run_history_caps_at_depth(tmp_path) -> None:
    demo = empty_demo_project()
    for node in demo.nodes:
        if node.type == "video.mock":
            node.params["size"] = 64
            node.params["fps"] = 5
            node.params["duration"] = 1
    shared_cache: dict = {}
    history: dict = {}
    for _ in range(HISTORY_DEPTH + 3):
        run_graph(demo, work_dir=str(tmp_path), cache=shared_cache, history=history)
    assert len(history["n_mock"].entries) == HISTORY_DEPTH


def test_run_history_records_errors(tmp_path) -> None:
    """A failing node also lands in history so the ◀▶ switcher surfaces it."""
    project = Project(
        nodes=[NodeSpec(id="s", type="script.storyboard", params={"prompt": "  "})],
        edges=[],
    )
    history: dict = {}
    with pytest.raises(GraphError):
        run_graph(project, work_dir=str(tmp_path), history=history)
    assert history["s"].entries[-1]["status"] == "error"


def test_reorder_shots_puts_listed_ids_first() -> None:
    shots = [{"id": "shot_01"}, {"id": "shot_02"}, {"id": "shot_03"}]
    out = reorder_shots(shots, ["shot_03", "shot_01"])
    assert [s["id"] for s in out] == ["shot_03", "shot_01", "shot_02"]


def test_reorder_shots_ignores_unknown_ids() -> None:
    """A stale shot_order stored in an older project must not drop shots."""
    shots = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    out = reorder_shots(shots, ["zzz", "a"])
    assert [s["id"] for s in out] == ["a", "b", "c"]


def test_reorder_shots_empty_order_is_identity() -> None:
    shots = [{"id": "a"}, {"id": "b"}]
    assert reorder_shots(shots, []) == shots


def test_extract_gallery_carries_motion_and_source_node() -> None:
    """The drawer card needs motion + source_node; guard extract_gallery keeps them."""
    report = run_graph(empty_demo_project(), work_dir=".")
    # Inject a stills result as image.batch_stills would produce.
    from studio.graph import NodeRunResult

    report.results["n_stills"] = NodeRunResult(
        node_id="n_stills",
        status="ok",
        outputs={
            "stills": {
                "shots": [
                    {
                        "id": "shot_01",
                        "description": "wide establishing",
                        "image": None,
                        "duration": 4,
                        "motion": "dolly_in",
                        "index": 0,
                    }
                ]
            }
        },
    )
    gal = extract_gallery(report)
    assert gal is not None
    assert gal["shots"][0]["motion"] == "dolly_in"
    assert gal["shots"][0]["source_node"] == "n_stills"
