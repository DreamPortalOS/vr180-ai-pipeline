"""Tests for project store API and SeedVR2 mock upscale node."""

from __future__ import annotations

from pathlib import Path

import pytest

from studio.graph import list_node_types, run_graph
from studio.models import EdgeSpec, NodeSpec, Project, empty_demo_project
from studio.nodes.upscale import Seedvr2UpscaleNode
from studio.projects import ProjectStore, ProjectStoreError


def test_project_store_roundtrip(tmp_path) -> None:
    store = ProjectStore(tmp_path / "projects")
    demo = empty_demo_project()
    saved = store.save(demo.to_dict(), project_id="demo")
    assert saved["id"] == "demo"
    assert Path(saved["path"]).is_file()
    loaded = store.load("demo")
    assert loaded["name"] == demo.name
    items = store.list_projects()
    assert any(i["id"] == "demo" for i in items)
    assert store.delete("demo") is True
    with pytest.raises(ProjectStoreError):
        store.load("demo")


def test_project_store_rejects_bad_json(tmp_path) -> None:
    store = ProjectStore(tmp_path / "p")
    with pytest.raises(ProjectStoreError):
        store.save({"nodes": [{"id": "x"}], "edges": []})  # missing type


def test_seedvr2_type_registered() -> None:
    assert "video.seedvr2" in {t["type"] for t in list_node_types()}


def test_seedvr2_mock_upscale(tmp_path) -> None:
    import cv2
    import numpy as np

    src = tmp_path / "in.mp4"
    w = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"mp4v"), 5, (64, 64))
    assert w.isOpened()
    for _ in range(4):
        w.write(np.full((64, 64, 3), 120, dtype=np.uint8))
    w.release()

    out = Seedvr2UpscaleNode().run(
        params={"mode": "mock", "scale": 2, "filename": "up.mp4"},
        inputs={"video": str(src)},
        work_dir=str(tmp_path),
        node_id="up",
    )
    assert Path(out["video"]).is_file()
    assert out["meta"]["mode"] == "mock"
    assert out["meta"]["scale"] == 2

    cap = cv2.VideoCapture(out["video"])
    ok, frame = cap.read()
    cap.release()
    assert ok and frame is not None
    assert frame.shape[0] >= 128  # ~2x of 64


def test_seedvr2_real_mode_errors_without_cuda(tmp_path, monkeypatch) -> None:
    import cv2
    import numpy as np

    import studio.nodes.upscale as up_mod

    src = tmp_path / "in.mp4"
    w = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"mp4v"), 5, (32, 32))
    for _ in range(2):
        w.write(np.zeros((32, 32, 3), dtype=np.uint8))
    w.release()

    def _boom(*_a, **_k):
        raise RuntimeError("CUDA requested but not available")

    monkeypatch.setattr("pipeline.video_upscaler.SeedVR2Upscaler.__init__", _boom)
    with pytest.raises(RuntimeError, match="mock"):
        up_mod.Seedvr2UpscaleNode().run(
            params={"mode": "seedvr2"},
            inputs={"video": str(src)},
            work_dir=str(tmp_path),
            node_id="up",
        )


def test_api_projects(tmp_path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from studio.server import create_app

    client = TestClient(create_app(default_work_dir=str(tmp_path / "w")))
    demo = empty_demo_project()
    res = client.post("/api/projects", json={"project": demo.to_dict(), "project_id": "demo1"})
    assert res.status_code == 200, res.text
    assert res.json()["id"] == "demo1"

    res = client.get("/api/projects")
    assert res.status_code == 200
    assert any(p["id"] == "demo1" for p in res.json())

    res = client.get("/api/projects/demo1")
    assert res.status_code == 200
    assert res.json()["name"] == demo.name

    # run with seedvr2 mock in graph
    project = Project(
        nodes=[
            NodeSpec(id="mock", type="video.mock", params={"size": 64, "fps": 5, "duration": 1}),
            NodeSpec(id="up", type="video.seedvr2", params={"mode": "mock", "scale": 2}),
            NodeSpec(id="exp", type="export.bundle", params={"filename": "e.mp4"}),
        ],
        edges=[
            EdgeSpec(id="e1", from_node="mock", from_port="video", to_node="up", to_port="video"),
            EdgeSpec(id="e2", from_node="up", from_port="video", to_node="exp", to_port="video"),
        ],
    )
    report = run_graph(project, work_dir=str(tmp_path / "run"))
    assert report.results["up"].status == "ok"
    assert Path(report.results["exp"].outputs["path"]).is_file()

    res = client.delete("/api/projects/demo1")
    assert res.status_code == 200
