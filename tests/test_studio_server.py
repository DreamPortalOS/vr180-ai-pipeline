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


def test_dome_proj_math_module_served(client) -> None:
    """issue #416: the pure-JS projection module is served before preview3d.js."""
    res = client.get("/dome_proj.js")
    assert res.status_code == 200
    assert "dirToMasterUV" in res.text
    assert "exportCliParams" in res.text


def test_dome3d_standalone_page_served(client) -> None:
    """issue #416: the standalone dome 3D / specs page is served (no CDN)."""
    res = client.get("/dome3d.html")
    assert res.status_code == 200
    assert "domeCanvas" in res.text
    assert "dome_proj.js" in res.text


def test_camera_viewport_owns_its_texture_per_context(client) -> None:
    """issue #416: the camera 2D viewport must NOT sample the preview's texture.

    ``CameraView`` renders in its own WebGL context, and a texture object
    belongs to the context that created it — binding a foreign one raises
    INVALID_OPERATION and the sampler reads black (verified in headless Chrome:
    bindError 1282, black sample).  So the viewport has to own a texture in its
    own context, fed from the shared source image, and the preview has to expose
    the upload for it.  A regression here silently blanks the 2D frame again,
    which is exactly the bug this pins."""
    js = client.get("/preview3d.js").text
    assert "uploadTexture" in js, "DomePreview must expose uploadTexture(gl, tex)"
    assert "this.tex = this.gl.createTexture()" in js, "CameraView needs its own texture"
    assert "p.uploadTexture(gl, this.tex)" in js, "CameraView must upload into its own context"
    # the old cross-context bind is what must be gone
    assert "gl.bindTexture(gl.TEXTURE_2D, p.tex)" not in js, (
        "CameraView must not bind the preview's texture (cross-context -> black)"
    )


def test_camera_drag_tool_locks_the_orbit(client) -> None:
    """issue #416: arming 机位拖拽 must not also orbit the 3D view.

    The orbit handler and the camera-drag handler are both bound to the dome
    canvas, so without a lock a single drag moved the camera *and* spun the view
    the operator was aiming against.  Found in headless Chrome (a 60x30px drag
    with the tool armed moved the camera +30° yaw / -15° pitch and the orbit
    +0.60 az / +0.30 el); after the fix the orbit delta is exactly 0 while the
    camera still moves.  There is no browser in CI, so the flag is pinned at the
    source level: it must exist, gate the orbit's mousedown, and be driven by the
    toggle — not merely declared."""
    js = client.get("/preview3d.js").text
    assert "this.orbitLocked = false" in js, "DomePreview needs the orbit lock"
    assert "if (this.orbitLocked) return;" in js, "the orbit mousedown must honour the lock"
    assert "preview.orbitLocked = dragCam.active;" in js, (
        "the 机位拖拽 toggle must arm/disarm the lock together with cameraShow"
    )


def test_index_ships_orientation_tools_and_camera_viewport(client) -> None:
    """issue #416 acceptance: the orientation sliders, draggable panel, and
    camera viewport are all present in the served index.html."""
    html = client.get("/").text
    # draggable right panel (left-edge resizer)
    assert "rightRail" in html and "railResizer" in html
    # yaw/pitch/roll sliders + quick buttons + front-convention toggle
    for ident in ("orientYawRow", "orientPitchRow", "orientRollRow", "frontIsBottom"):
        assert ident in html, f"index.html missing #{ident}"
    for btn in ("btnOrientReset", "btnRot90cw", "btnRot90ccw", "btnFlipH", "btnFlipV", "btnOrientExport"):
        assert btn in html, f"index.html missing #{btn}"
    # CLI export readout uses the --dome-* flag names
    assert "--dome-pitch" in html and "--dome-yaw" in html and "--dome-roll" in html
    # camera viewport + its sliders
    assert "camCanvas" in html
    for ident in ("camYawRow", "camPitchRow", "camHalfFovRow", "btnCamReset", "btnCamDrag"):
        assert ident in html, f"index.html missing #{ident}"


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
