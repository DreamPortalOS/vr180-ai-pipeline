"""Tests for Studio M1 nodes: LLM polish, Seedance (mock), quality check."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from studio.graph import list_node_types, run_graph
from studio.models import EdgeSpec, NodeSpec, Project
from studio.nodes.llm import LlmPolishNode
from studio.nodes.quality import QualityCheckNode
from studio.nodes.seedance_video import SeedanceVideoNode, estimate_cost_yuan


def test_m1_types_registered() -> None:
    types = {t["type"] for t in list_node_types()}
    assert {"text.llm_polish", "video.seedance", "qa.source_quality"} <= types


def test_llm_polish_mock_rewrites_prompt() -> None:
    node = LlmPolishNode()
    out = node.run(
        params={"provider": "mock"},
        inputs={"prompt": "FPV drone flying over a river valley"},
        work_dir=".",
        node_id="n",
    )
    assert out["meta"]["provider"] == "mock"
    assert "river" in out["prompt"].lower() or "FPV" in out["prompt"]
    assert len(out["prompt"]) >= 20


def test_llm_polish_requires_base_url_for_litellm() -> None:
    node = LlmPolishNode()
    with pytest.raises(ValueError, match="base_url"):
        node.run(
            params={"provider": "litellm"},
            inputs={"prompt": "a quiet harbour"},
            work_dir=".",
            node_id="n",
        )


def test_estimate_cost() -> None:
    assert estimate_cost_yuan("4k", 10) == 50.0
    assert estimate_cost_yuan("480p", 5) == 1.0
    assert estimate_cost_yuan("999p", 5) is None


def test_seedance_paid_requires_confirm() -> None:
    node = SeedanceVideoNode()
    with pytest.raises(ValueError, match="confirm_paid"):
        node.run(
            params={"provider": "seedance", "resolution": "480p"},
            inputs={"prompt": "x"},
            work_dir=".",
            node_id="n",
        )


def test_seedance_mock_renders(tmp_path) -> None:
    node = SeedanceVideoNode()
    out = node.run(
        params={"provider": "mock", "size": 64, "ratio": "1:1"},
        inputs={"prompt": "harbour dawn"},
        work_dir=str(tmp_path),
        node_id="n_seed",
    )
    assert Path(out["video"]).is_file()
    assert out["meta"]["provider"] == "mock"
    assert out["meta"]["cost_estimate_yuan"] == 0


def test_quality_mock_mode(tmp_path) -> None:
    fake = tmp_path / "clip.mp4"
    fake.write_bytes(b"not-a-real-video")
    node = QualityCheckNode()
    out = node.run(
        params={"mode": "mock"},
        inputs={"video": str(fake)},
        work_dir=str(tmp_path),
        node_id="q",
    )
    assert out["passed"] == 1
    assert out["report"]["mode"] == "mock"


def test_quality_missing_file_fails(tmp_path) -> None:
    node = QualityCheckNode()
    out = node.run(
        params={"mode": "auto"},
        inputs={"video": str(tmp_path / "missing.mp4")},
        work_dir=str(tmp_path),
        node_id="q",
    )
    assert out["passed"] == 0
    assert out["report"]["summary"]["overall"] == "fail"


def test_m1_graph_script_polish_mock_video_export(tmp_path) -> None:
    project = Project(
        name="m1-chain",
        nodes=[
            NodeSpec(
                id="s",
                type="script.storyboard",
                params={"prompt": "walk through a neon alley at night", "duration": 1, "aspect_ratio": "1:1"},
            ),
            NodeSpec(id="polish", type="text.llm_polish", params={"provider": "mock", "target": "vr180"}),
            NodeSpec(id="vid", type="video.seedance", params={"provider": "mock", "size": 64, "ratio": "1:1"}),
            NodeSpec(id="qa", type="qa.source_quality", params={"mode": "mock"}),
            NodeSpec(id="exp", type="export.bundle", params={"filename": "m1.mp4"}),
        ],
        edges=[
            EdgeSpec(id="e1", from_node="s", from_port="prompt", to_node="polish", to_port="prompt"),
            EdgeSpec(id="e2", from_node="polish", from_port="prompt", to_node="vid", to_port="prompt"),
            EdgeSpec(id="e3", from_node="s", from_port="duration", to_node="vid", to_port="duration"),
            EdgeSpec(id="e4", from_node="vid", from_port="video", to_node="qa", to_port="video"),
            EdgeSpec(id="e5", from_node="qa", from_port="video", to_node="exp", to_port="video"),
            EdgeSpec(id="e6", from_node="polish", from_port="prompt", to_node="exp", to_port="prompt"),
        ],
    )
    report = run_graph(project, work_dir=str(tmp_path))
    assert report.results["polish"].status == "ok"
    assert report.results["vid"].status == "ok"
    assert report.results["qa"].outputs["passed"] == 1
    export = report.results["exp"].outputs["path"]
    assert Path(export).is_file()
    manifest = json.loads(Path(report.results["exp"].outputs["manifest"]["manifest_path"]).read_text(encoding="utf-8"))
    assert "prompt" in manifest
