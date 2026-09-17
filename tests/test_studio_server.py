"""Tests for studio FastAPI app (M0). Skips if FastAPI is unavailable."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from studio.models import empty_demo_project
from studio.server import create_app


@pytest.fixture()
def client(tmp_path):
    app = create_app(default_work_dir=str(tmp_path / "studio_work"))
    return TestClient(app)


def test_health(client) -> None:
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_node_types(client) -> None:
    res = client.get("/api/node-types")
    assert res.status_code == 200
    types = {t["type"] for t in res.json()}
    assert "video.mock" in types


def test_validate_and_run_demo(client, tmp_path) -> None:
    demo = empty_demo_project()
    for node in demo.nodes:
        if node.type == "video.mock":
            node.params["size"] = 64
            node.params["fps"] = 5
            node.params["duration"] = 1
    payload = {"project": demo.to_dict(), "work_dir": str(tmp_path / "out")}

    res = client.post("/api/validate", json=payload)
    assert res.status_code == 200
    assert res.json()["ok"] is True

    res = client.post("/api/run", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert body["results"]["n_export"]["status"] == "ok"


def test_run_rejects_unknown_node_type(client, tmp_path) -> None:
    payload = {
        "project": {
            "version": 1,
            "name": "bad",
            "nodes": [{"id": "x", "type": "does.not.exist", "pos": [0, 0], "params": {}}],
            "edges": [],
        },
        "work_dir": str(tmp_path / "out2"),
    }
    res = client.post("/api/run", json=payload)
    assert res.status_code == 400
    assert "unknown node type" in res.json()["detail"]


def test_index_served(client) -> None:
    res = client.get("/")
    assert res.status_code == 200
    assert "Immersive Node Studio" in res.text


def test_dual_export_template_endpoint(client) -> None:
    res = client.get("/api/templates/dual-export")
    assert res.status_code == 200
    body = res.json()
    assert body["name"] == "demo-dual-export"
    types = {n["type"] for n in body["nodes"]}
    assert "convert.dome" in types
    assert "qa.dome_coverage" in types
