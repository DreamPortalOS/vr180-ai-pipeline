"""Tests for Studio M2: dome convert, coverage, VR180 mock."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from studio.coverage import analyze_frame
from studio.graph import list_node_types, run_graph
from studio.models import EdgeSpec, NodeSpec, Project
from studio.nodes.convert import DomeConvertNode, DomeCoverageNode, Vr180ConvertNode


def test_m2_types_registered() -> None:
    types = {t["type"] for t in list_node_types()}
    assert {"convert.dome", "qa.dome_coverage", "convert.vr180"} <= types


def test_dual_export_template_runs(tmp_path) -> None:
    from studio.templates import dual_export_demo_project

    project = dual_export_demo_project()
    for node in project.nodes:
        if node.type == "video.seedance":
            node.params["size"] = 64
    report = run_graph(project, work_dir=str(tmp_path))
    assert report.results["n_dome"].status == "ok"
    assert report.results["n_cov"].status == "ok"
    assert report.results["n_vr"].status == "ok"
    assert Path(report.results["n_exp"].outputs["path"]).is_file()


def _make_dome_like_frame(size: int = 256, content_radius: float = 0.6) -> np.ndarray:
    """Synthetic domemaster: lit disc of given radius on black."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    yy, xx = np.mgrid[0:size, 0:size]
    dx = (xx + 0.5) / size * 2 - 1
    dy = (yy + 0.5) / size * 2 - 1
    rr = np.sqrt(dx * dx + dy * dy)
    img[rr <= content_radius] = (200, 200, 200)
    return img


def test_coverage_detects_short_content() -> None:
    stats = analyze_frame(_make_dome_like_frame(content_radius=0.6))
    assert stats.level == "bad"
    assert stats.coverage_deg < 70
    assert stats.outer_fill < 0.1


def test_coverage_full_disc_ok() -> None:
    stats = analyze_frame(_make_dome_like_frame(content_radius=0.99))
    assert stats.level == "ok"
    assert stats.coverage_deg >= 85


def test_dome_convert_and_coverage(tmp_path) -> None:
    # Tiny synthetic source video
    src = tmp_path / "src.mp4"
    writer = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"mp4v"), 5, (64, 64))
    assert writer.isOpened()
    for _ in range(5):
        writer.write(np.full((64, 64, 3), 180, dtype=np.uint8))
    writer.release()

    project = Project(
        name="m2",
        nodes=[
            NodeSpec(id="dome", type="convert.dome", params={"size": 128, "coverage_h": 150}),
            NodeSpec(id="cov", type="qa.dome_coverage", params={"min_deg": 70}),
            NodeSpec(id="vr", type="convert.vr180", params={"mode": "mock"}),
        ],
        edges=[
            EdgeSpec(id="e1", from_node="dome", from_port="video", to_node="cov", to_port="video"),
        ],
    )
    # Wire source into dome via manual input injection: run dome node directly
    dome = DomeConvertNode()
    out = dome.run(
        params={"size": 128, "coverage_h": 150},
        inputs={"video": str(src)},
        work_dir=str(tmp_path),
        node_id="dome",
    )
    assert Path(out["video"]).is_file()

    cov = DomeCoverageNode()
    report = cov.run(
        params={"min_deg": 70},
        inputs={"video": out["video"]},
        work_dir=str(tmp_path),
        node_id="cov",
    )
    assert "coverage_deg" in report["report"]
    assert report["report"]["level"] in {"ok", "warn", "bad"}

    vr = Vr180ConvertNode()
    vr_out = vr.run(
        params={"mode": "mock"},
        inputs={"video": str(src)},
        work_dir=str(tmp_path),
        node_id="vr",
    )
    assert Path(vr_out["video"]).is_file()
    assert vr_out["meta"]["mode"] == "mock"

    # graph wiring sanity (types registered)
    assert project.nodes
