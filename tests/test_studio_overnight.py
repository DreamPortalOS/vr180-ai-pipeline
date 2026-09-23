"""Overnight: dirty subgraph rerun, gallery extraction, paid-provider guard."""

from __future__ import annotations

from pathlib import Path

import pytest

from studio.graph import (
    ancestors_and_self,
    descendants_and_self,
    extract_gallery,
    run_graph,
)
from studio.models import EdgeSpec, NodeSpec, Project
from studio.templates import production_pipeline_project


def test_ancestors_and_descendants() -> None:
    project = production_pipeline_project()
    anc = ancestors_and_self(project, "n_concat")
    assert {"n_concat", "n_clips", "n_review", "n_stills", "n_polish", "n_shots", "n_brief"} <= anc
    assert "n_vr" not in anc
    desc = descendants_and_self(project, "n_shots")
    assert "n_export" in desc
    assert "n_brief" not in desc or "n_shots" == "n_brief"


def test_only_downstream_of_skips_siblings(tmp_path) -> None:
    project = Project(
        nodes=[
            NodeSpec(id="a", type="script.storyboard", params={"prompt": "hello", "duration": 1}),
            NodeSpec(id="b", type="video.mock", params={"size": 64, "fps": 5, "duration": 1}),
            NodeSpec(id="c", type="export.bundle", params={"filename": "x.mp4"}),
        ],
        edges=[
            EdgeSpec(id="e1", from_node="a", from_port="prompt", to_node="b", to_port="prompt"),
            EdgeSpec(id="e2", from_node="b", from_port="video", to_node="c", to_port="video"),
            EdgeSpec(id="e3", from_node="a", from_port="prompt", to_node="c", to_port="prompt"),
        ],
    )
    report = run_graph(project, work_dir=str(tmp_path), only_downstream_of="b")
    assert report.results["a"].status == "ok"
    assert report.results["b"].status == "ok"
    assert report.results["c"].status == "skipped"


def test_dirty_from_forces_downstream_recompute(tmp_path) -> None:
    project = Project(
        nodes=[
            NodeSpec(id="a", type="script.storyboard", params={"prompt": "hi", "duration": 1}),
            NodeSpec(id="b", type="video.mock", params={"size": 64, "fps": 5, "duration": 1}),
        ],
        edges=[
            EdgeSpec(id="e1", from_node="a", from_port="prompt", to_node="b", to_port="prompt"),
            EdgeSpec(id="e2", from_node="a", from_port="duration", to_node="b", to_port="duration"),
        ],
    )
    cache: dict = {}
    r1 = run_graph(project, work_dir=str(tmp_path), cache=cache)
    assert r1.results["b"].status == "ok"
    r2 = run_graph(project, work_dir=str(tmp_path), cache=cache)
    assert r2.results["b"].cache_hit is True
    # change params on b and mark dirty
    project.nodes[1].params["size"] = 80
    r3 = run_graph(project, work_dir=str(tmp_path), cache=cache, dirty_from="b")
    assert r3.results["b"].status == "ok"
    assert r3.results["b"].cache_hit is False
    # upstream still cache hit
    assert r3.results["a"].cache_hit is True


def test_extract_gallery_from_production(tmp_path) -> None:
    project = production_pipeline_project()
    for node in project.nodes:
        if node.type == "image.batch_stills":
            node.params["width"] = 96
            node.params["height"] = 96
        if node.type == "video.from_stills":
            node.params["size"] = 96
            node.params["fps"] = 10
        if node.type == "script.shot_list":
            node.params["shot_texts"] = "open\npush"
            node.params["shot_durations"] = "1,1"
        if node.type == "convert.dome":
            node.params["size"] = 96
        if node.type == "audio.bgm_tone":
            node.params["duration"] = 2
    report = run_graph(project, work_dir=str(tmp_path))
    gallery = extract_gallery(report)
    assert gallery is not None
    assert gallery["sheet"]
    assert Path(gallery["sheet"]).is_file()
    assert gallery["count"] >= 2
    assert gallery["shots"][0]["image"]


def test_minimax_not_registered() -> None:
    """Owner decision: MiniMax tops out at 2K, so the whole line was dropped (#374)."""
    from integrations.factory import list_providers

    assert "minimax" not in list_providers()


def test_studio_video_node_rejects_minimax() -> None:
    from studio.nodes.seedance_video import SeedanceVideoNode

    with pytest.raises(ValueError, match="unknown video provider"):
        SeedanceVideoNode().run(
            params={"provider": "minimax"},
            inputs={"prompt": "x"},
            work_dir=".",
            node_id="n",
        )
