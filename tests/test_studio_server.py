"""Tests for studio FastAPI app (M0). Skips if FastAPI is unavailable."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")

import cv2
from fastapi.testclient import TestClient

from studio.coverage import analyze_frame
from studio.models import empty_demo_project
from studio.server import create_app

STATIC_DIR = Path(__file__).resolve().parent.parent / "studio" / "static"


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
    res = client.get("/api/templates/production")
    assert res.status_code == 200
    body = res.json()
    assert body["name"].startswith("production")
    types = {n["type"] for n in body["nodes"]}
    assert "script.shot_list" in types
    assert "video.concat" in types
    assert "audio.mux" in types


def test_preview3d_and_dome_assets_served(client) -> None:
    res = client.get("/preview3d.js")
    assert res.status_code == 200
    assert "DomePreview" in res.text
    res = client.get("/")
    assert "domeCanvas" in res.text
    assert "preview3d.js" in res.text
    assert "画幅" in res.text


# ── POST /api/coverage (issue #405) ────────────────────────────────────────
# The 3D preview asks the server for a coverage reading so it uses the same
# scan as scripts/dome_qa.py and the qa.dome_coverage node, not a third
# client-side algorithm. These frame helpers mirror the synthetic masters in
# tests/test_dome_coverage_unified.py (no ffmpeg, no models).

SIZE = 256


def _noise_dome(fill_r: float, size: int = SIZE, seed: int = 0) -> np.ndarray:
    """Noise-filled disc out to r=fill_r on a black square (BGR uint8)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(float)
    rr = np.sqrt((xx - (size - 1) / 2.0) ** 2 + (yy - (size - 1) / 2.0) ** 2) / (size / 2.0)
    noise = rng.integers(0, 256, size=(size, size)).astype(np.uint8)
    frame = np.zeros((size, size, 3), dtype=np.uint8)
    mask = rr <= fill_r
    frame[mask] = np.stack([noise[mask]] * 3, axis=1)
    return frame


def _png_bytes(frame: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", frame)
    assert ok, "png encode failed"
    return buf.tobytes()


def test_coverage_endpoint_matches_shared_scan(client) -> None:
    """The endpoint reads the same coverage_r as studio.coverage.analyze_frame
    on the same frame — one algorithm, not a third one."""
    frame = _noise_dome(0.95)
    res = client.post("/api/coverage", content=_png_bytes(frame), headers={"Content-Type": "image/png"})
    assert res.status_code == 200
    body = res.json()
    # Same keys as CoverageStats.to_dict().
    for key in ("coverage_radius", "coverage_deg", "level", "ring_profile"):
        assert key in body
    expected = analyze_frame(frame)
    assert body["coverage_radius"] == pytest.approx(expected.coverage_radius, abs=1e-9)
    assert body["coverage_deg"] == pytest.approx(expected.coverage_deg, abs=1e-9)
    assert body["level"] == expected.level


def test_coverage_endpoint_compliant_preset_passes(client) -> None:
    """A fully-filled disc (the 合规 preset's synthetic image) reads >= 85 deg."""
    res = client.post("/api/coverage", content=_png_bytes(_noise_dome(0.98)), headers={"Content-Type": "image/png"})
    assert res.status_code == 200
    body = res.json()
    assert body["coverage_deg"] >= 85.0
    assert body["level"] == "ok"


def test_coverage_endpoint_bad_preset_is_red(client) -> None:
    """A disc filled only to r=0.6 (the 坏母版 preset) reads level='bad'."""
    res = client.post("/api/coverage", content=_png_bytes(_noise_dome(0.6)), headers={"Content-Type": "image/png"})
    assert res.status_code == 200
    body = res.json()
    assert body["level"] == "bad"
    assert body["coverage_deg"] < 75.0


def test_coverage_endpoint_rejects_empty_body(client) -> None:
    res = client.post("/api/coverage", content=b"", headers={"Content-Type": "image/png"})
    assert res.status_code == 400
    assert "empty" in res.json()["detail"].lower()


def test_coverage_endpoint_rejects_garbage(client) -> None:
    res = client.post("/api/coverage", content=b"not an image", headers={"Content-Type": "image/png"})
    assert res.status_code == 400
    assert "image" in res.json()["detail"].lower()


def test_coverage_endpoint_accepts_jpeg(client) -> None:
    """The endpoint decodes any cv2-supported format, not only PNG."""
    ok, buf = cv2.imencode(".jpg", _noise_dome(0.95), [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    assert ok
    res = client.post("/api/coverage", content=buf.tobytes(), headers={"Content-Type": "image/jpeg"})
    assert res.status_code == 200
    assert res.json()["level"] == "ok"


# ── preview3d.js no longer ships a second coverage algorithm (issue #405) ──


def test_preview3d_uses_server_coverage_not_a_client_scan() -> None:
    """Guard: the browser must fetch /api/coverage and must NOT keep a
    client-side coverage scan (analyzeImage / edgeAt) that could disagree with
    the server on the same master."""
    js = (STATIC_DIR / "preview3d.js").read_text(encoding="utf-8")
    # The new contract: POST the image, draw the circle from the reply.
    assert "/api/coverage" in js
    assert "fetch(" in js
    # The old third algorithm is gone as CODE — its function definitions and
    # call site are removed (comments may still name it for context).
    assert "function analyzeImage" not in js
    assert "function edgeAt" not in js
    assert "analyzeImage(img)" not in js
    # Presets still generate their image on the frontend, but readings go via
    # the server (makeDomeCanvas feeds analyzeOnServer, not a hardcoded number).
    assert "makeDomeCanvas" in js
    assert "analyzeOnServer" in js


# ── issue #418: per-node history endpoint for the ◀▶ inline switcher ──


def test_node_history_endpoint_returns_empty_before_run(client) -> None:
    """An unknown / never-run node returns 200 with an empty list (not 404)
    so the frontend can treat "no history" as a clean first-run state."""
    res = client.get("/api/node/never_seen/history")
    assert res.status_code == 200
    body = res.json()
    assert body["node_id"] == "never_seen"
    assert body["history"] == []
    assert body["count"] == 0


def test_node_history_endpoint_records_runs(client, tmp_path) -> None:
    """After two runs the endpoint returns ≥2 entries for a node that ran each
    time (cold + cache-hit), proving the ◀▶ switcher has data to cycle."""
    demo = empty_demo_project()
    for node in demo.nodes:
        if node.type == "video.mock":
            node.params["size"] = 64
            node.params["fps"] = 5
            node.params["duration"] = 1
    payload = {"project": demo.to_dict(), "work_dir": str(tmp_path / "out")}
    client.post("/api/run", json=payload)
    client.post("/api/run", json=payload)
    res = client.get("/api/node/n_mock/history")
    assert res.status_code == 200
    body = res.json()
    assert body["count"] >= 1
    assert all(e["status"] == "ok" for e in body["history"])


def test_run_payload_carry_history_snapshot(client, tmp_path) -> None:
    """The /api/run reply embeds a history map so the canvas can paint the
    ◀▶ arrows from a single response."""
    demo = empty_demo_project()
    for node in demo.nodes:
        if node.type == "video.mock":
            node.params["size"] = 64
            node.params["fps"] = 5
            node.params["duration"] = 1
    payload = {"project": demo.to_dict(), "work_dir": str(tmp_path / "out")}
    res = client.post("/api/run", json=payload)
    body = res.json()
    assert "history" in body
    assert "n_mock" in body["history"]
