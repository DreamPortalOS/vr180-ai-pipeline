"""Tests for input nodes + POST /api/upload (issue #417).

Acceptance items covered here:

* TestClient: uploading png/mp4 returns metadata and a de-duplicated path;
  unsupported extensions are rejected; oversize returns 413.
* The three input nodes pass their values to downstream nodes (mock downstream).

Everything runs on the CPU-only CI image: images are encoded with cv2/Pillow,
videos with the ffmpeg already required on PATH, and no models are touched.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

import cv2
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from studio.graph import run_graph
from studio.models import EdgeSpec, NodeSpec, Project
from studio.nodes import get_node_class
from studio.nodes.preview import PreviewNode
from studio.server import create_app
from studio.uploads import DEFAULT_MAX_BYTES

PNG = (64, 32)


@pytest.fixture()
def client(tmp_path):
    app = create_app(default_work_dir=str(tmp_path / "studio_work"))
    return TestClient(app)


def _png_bytes(shape: tuple[int, int] = PNG) -> bytes:
    """A small but genuinely decodable PNG."""
    frame = np.full((shape[1], shape[0], 3), 200, dtype=np.uint8)
    ok, buf = cv2.imencode(".png", frame)
    assert ok
    return buf.tobytes()


@pytest.fixture(scope="module")
def tiny_mp4_bytes(tmp_path_factory) -> bytes:
    """A 1s 128x96 clip, rendered once for the whole module."""
    name = tmp_path_factory.mktemp("fixture") / "tiny.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=128x96:rate=10:duration=1",
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        "libx264",
        str(name),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=120)
    if proc.returncode != 0:
        pytest.skip(f"ffmpeg could not render a test clip: {proc.stderr[-200:]}")
    return name.read_bytes()


# ── POST /api/upload ───────────────────────────────────────────────────────


def test_upload_png_returns_metadata(client, tmp_path) -> None:
    res = client.post("/api/upload", files={"file": ("shot.png", _png_bytes(), "image/png")})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["kind"] == "image"
    assert body["width"] == PNG[0]
    assert body["height"] == PNG[1]
    assert body["duration"] is None
    # Poster frame is only meaningful for videos.
    assert body["poster"] is None
    path = body["path"]
    assert path.endswith(".png")
    assert Path(path).is_file()
    assert Path(path).stat().st_size == len(_png_bytes())
    # Stored under work_root/uploads/, content-addressed by sha256[:12].
    assert "/uploads/" in path or "\\uploads\\" in path
    # Content-addressed: sha256[:12], i.e. exactly 12 lowercase hex chars.
    assert re.fullmatch(r"[0-9a-f]{12}", Path(path).stem)


def test_upload_mp4_returns_duration_res_and_poster(client, tiny_mp4_bytes) -> None:
    res = client.post("/api/upload", files={"file": ("clip.mp4", tiny_mp4_bytes, "video/mp4")})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["kind"] == "video"
    assert body["width"] == 128
    assert body["height"] == 96
    assert body["duration"] is not None and 0.5 <= body["duration"] <= 1.5
    assert body["poster"], "a poster frame must be extracted for the canvas"
    assert Path(body["poster"]).suffix == ".png"
    assert Path(body["poster"]).is_file()


def test_upload_dedupes_identical_bytes(client) -> None:
    data = _png_bytes()
    first = client.post("/api/upload", files={"file": ("a.png", data, "image/png")})
    second = client.post("/api/upload", files={"file": ("b.png", data, "image/png")})
    assert first.status_code == second.status_code == 200
    # Same bytes ⇒ same path, regardless of the original filename.
    assert first.json()["path"] == second.json()["path"]
    store = Path(first.json()["path"]).parent
    assert len(list(store.glob("*.png"))) == 1


def test_upload_rejects_unsupported_extension(client) -> None:
    res = client.post("/api/upload", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert res.status_code == 400
    assert "unsupported" in res.json()["detail"].lower()
    for name in ("app.exe", "data.bin", "script.py"):
        res = client.post("/api/upload", files={"file": (name, b"MZ", "application/octet-stream")})
        assert res.status_code == 400, name


def test_upload_rejects_no_extension(client) -> None:
    res = client.post("/api/upload", files={"file": ("README", b"data", "application/octet-stream")})
    assert res.status_code == 400


def test_upload_rejects_garbage_with_allowed_extension(client, tmp_path) -> None:
    """A .png named over garbage bytes must 400, not 500.

    The extension is only a first filter; the bytes have to decode too.
    (Regression: ``Image.open`` raises ``UnidentifiedImageError`` and the
    endpoint used to crash with a 500.)
    """
    for name, payload in (
        ("evil.png", b"MZ\x90\x00not really an image"),
        ("evil.webp", b"\x00\x00\x00\x00"),
    ):
        res = client.post("/api/upload", files={"file": (name, payload, "image/png")})
        assert res.status_code == 400, (name, res.status_code, res.text)
        assert "decode" in res.json()["detail"].lower()
    # The bogus file must not linger in the store (otherwise a re-upload with
    # real bytes would collide on the sha and be served back malformed).
    store = Path((tmp_path / "studio_work") / "uploads")
    assert not list(store.glob("*.png"))


def test_upload_rejects_garbage_video(client) -> None:
    res = client.post("/api/upload", files={"file": ("evil.mp4", b"not a container", "video/mp4")})
    assert res.status_code == 400, res.text
    assert "decode" in res.json()["detail"].lower()


def test_upload_oversize_returns_413(client, tmp_path, monkeypatch) -> None:
    import studio.server as server_mod

    monkeypatch.setattr(server_mod, "DEFAULT_MAX_BYTES", 32)
    res = client.post("/api/upload", files={"file": ("big.png", b"\x00" * 64, "image/png")})
    assert res.status_code == 413
    assert "32" in res.json()["detail"]


def test_upload_empty_body(client) -> None:
    res = client.post(
        "/api/upload",
        data="",
        headers={"Content-Type": "multipart/form-data; boundary=----x"},
    )
    assert res.status_code == 400


def test_upload_stored_files_are_served_by_media_route(client) -> None:
    res = client.post("/api/upload", files={"file": ("served.png", _png_bytes(), "image/png")})
    assert res.status_code == 200
    served = client.get("/api/media", params={"path": res.json()["path"]})
    assert served.status_code == 200
    assert served.headers["content-type"].startswith("image/")
    assert served.content == _png_bytes()


def test_upload_only_persists_paths_not_contents(client) -> None:
    """A project that references an upload stores the path, never the bytes."""
    res = client.post("/api/upload", files={"file": ("json.png", _png_bytes(), "image/png")})
    assert res.status_code == 200
    stored = res.json()["path"]
    project = Project(
        nodes=[NodeSpec(id="n_img", type="input.image", params={"path": stored})],
        edges=[],
    )
    text = project.to_json()
    # JSON escapes Windows separators, so compare against an escaped form.
    assert stored.replace("\\", "\\\\") in text
    # The PNG magic bytes must never appear in the project document.
    assert "\x89PNG" not in text
    assert "iVBORw0KGgo" not in text


# ── input nodes ────────────────────────────────────────────────────────────


def test_input_text_node_outputs_text(tmp_path) -> None:
    out = get_node_class("input.text")().run(
        params={"text": "  a slow dolly through a harbour  "},
        inputs={},
        work_dir=str(tmp_path),
        node_id="t",
    )
    assert out == {"text": "a slow dolly through a harbour"}
    with pytest.raises(ValueError, match="non-empty"):
        get_node_class("input.text")().run(params={}, inputs={}, work_dir=str(tmp_path), node_id="t")


def test_input_image_node_outputs_image(tmp_path) -> None:
    src = tmp_path / "src.png"
    Image.new("RGB", (40, 20), (9, 21, 33)).save(src)
    out = get_node_class("input.image")().run(
        params={"path": str(src)}, inputs={}, work_dir=str(tmp_path), node_id="im"
    )
    assert out["image"] == str(src)
    assert out["meta"]["kind"] == "image"
    assert (out["meta"]["width"], out["meta"]["height"]) == (40, 20)
    assert out["meta"]["size_bytes"] > 0


def test_input_video_node_outputs_video(tmp_path) -> None:
    src = tmp_path / "src.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=64x48:rate=8:duration=1",
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        "libx264",
        str(src),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=120)
    if proc.returncode != 0:
        pytest.skip(f"ffmpeg could not render a test clip: {proc.stderr[-200:]}")
    out = get_node_class("input.video")().run(
        params={"path": str(src)}, inputs={}, work_dir=str(tmp_path), node_id="vid"
    )
    assert out["video"] == str(src)
    assert out["meta"]["kind"] == "video"
    assert (out["meta"]["width"], out["meta"]["height"]) == (64, 48)
    assert out["meta"]["duration"] is not None and 0.5 <= out["meta"]["duration"] <= 1.5
    assert out["meta"]["poster"] and Path(out["meta"]["poster"]).is_file()
    # The poster frame must land under this node's own studio_out dir.
    assert "studio_out" in out["meta"]["poster"]
    assert "vid" in out["meta"]["poster"]


def test_input_nodes_reject_wrong_extension(tmp_path) -> None:
    bad = tmp_path / "shot.txt"
    bad.write_text("not a picture", encoding="utf-8")
    with pytest.raises(ValueError, match="not an accepted input type"):
        get_node_class("input.image")().run(params={"path": str(bad)}, inputs={}, work_dir=str(tmp_path), node_id="im")
    with pytest.raises(ValueError, match="not an accepted input type"):
        get_node_class("input.video")().run(params={"path": str(bad)}, inputs={}, work_dir=str(tmp_path), node_id="vid")


def test_input_nodes_fail_on_missing_file(tmp_path) -> None:
    missing = tmp_path / "nope.png"
    with pytest.raises(FileNotFoundError):
        get_node_class("input.image")().run(
            params={"path": str(missing)}, inputs={}, work_dir=str(tmp_path), node_id="im"
        )


def test_input_image_resolves_relative_path_against_work_dir(tmp_path) -> None:
    """A project JSON saved on one machine stays portable."""
    src = tmp_path / "sub" / "src.png"
    src.parent.mkdir(parents=True)
    Image.new("RGB", (12, 12)).save(src)
    out = get_node_class("input.image")().run(
        params={"path": "sub/src.png"},
        inputs={},
        work_dir=str(tmp_path),
        node_id="im",
    )
    assert out["image"] == str(src)


# ── input nodes → downstream (mock downstream) ─────────────────────────────
# The acceptance criterion: all three input nodes' output ports can feed an
# existing text/image/video consumer. ``tool.preview`` is the mock downstream
# here — it requires a value on its ``value`` port and echoes it back, so a
# successful run proves the value actually travelled the wire.


def _project(src_node: NodeSpec, from_port: str, out_dir: Path) -> Project:
    sink = NodeSpec(id="n_sink", type="tool.preview", params={"label": "mock downstream"})
    project = Project(
        name="input-wire",
        nodes=[src_node, sink],
        edges=[EdgeSpec(id="e1", from_node=src_node.id, from_port=from_port, to_node="n_sink", to_port="value")],
    )
    project.validate()
    return project


def test_input_text_feeds_downstream(tmp_path) -> None:
    report = run_graph(
        _project(NodeSpec(id="n_text", type="input.text", params={"text": "hello downstream"}), "text", tmp_path),
        work_dir=str(tmp_path),
    )
    assert report.results["n_text"].status == "ok"
    assert report.results["n_sink"].status == "ok"
    assert report.results["n_sink"].outputs["value"] == "hello downstream"


def test_input_image_feeds_downstream(tmp_path) -> None:
    src = tmp_path / "wire.png"
    Image.new("RGB", (16, 16)).save(src)
    report = run_graph(
        _project(NodeSpec(id="n_img", type="input.image", params={"path": str(src)}), "image", tmp_path),
        work_dir=str(tmp_path),
    )
    assert report.results["n_img"].status == "ok"
    assert report.results["n_sink"].status == "ok"
    assert report.results["n_sink"].outputs["value"] == str(src)
    # The mock downstream saw a real path that exists.
    assert Path(report.results["n_sink"].outputs["value"]).is_file()


def test_input_video_feeds_downstream(tmp_path) -> None:
    src = tmp_path / "wire.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=48x32:rate=8:duration=1",
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        "libx264",
        str(src),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=120)
    if proc.returncode != 0:
        pytest.skip(f"ffmpeg could not render a test clip: {proc.stderr[-200:]}")
    report = run_graph(
        _project(NodeSpec(id="n_vid", type="input.video", params={"path": str(src)}), "video", tmp_path),
        work_dir=str(tmp_path),
    )
    assert report.results["n_vid"].status == "ok"
    assert report.results["n_sink"].status == "ok"
    assert report.results["n_sink"].outputs["value"] == str(src)


def test_input_nodes_can_be_upstream_of_real_consumer(tmp_path) -> None:
    """input.video → convert.vr180 (mode=mock) → preview: end-to-end."""
    src = tmp_path / "e2e.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=48x32:rate=8:duration=1",
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        "libx264",
        str(src),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=120)
    if proc.returncode != 0:
        pytest.skip(f"ffmpeg could not render a test clip: {proc.stderr[-200:]}")
    project = Project(
        name="input-to-convert",
        nodes=[
            NodeSpec(id="n_vid", type="input.video", params={"path": str(src)}),
            NodeSpec(id="n_conv", type="convert.vr180", params={"mode": "mock"}),
            NodeSpec(id="n_prev", type="tool.preview", params={}),
        ],
        edges=[
            EdgeSpec(id="e1", from_node="n_vid", from_port="video", to_node="n_conv", to_port="video"),
            EdgeSpec(id="e2", from_node="n_conv", from_port="video", to_node="n_prev", to_port="value"),
        ],
    )
    report = run_graph(project, work_dir=str(tmp_path))
    assert report.results["n_vid"].status == "ok"
    assert report.results["n_conv"].status == "ok"
    assert report.results["n_prev"].status == "ok"
    # The converted file is a new artefact, distinct from the input path.
    assert report.results["n_prev"].outputs["value"] != str(src)
    assert Path(report.results["n_prev"].outputs["value"]).is_file()


def test_input_nodes_in_catalogue(tmp_path) -> None:
    """The palette must expose the three input nodes."""
    from studio.graph import list_node_types

    cat = {t["type"]: t for t in list_node_types()}
    assert "input.text" in cat and "input.image" in cat and "input.video" in cat
    assert all(cat[t]["category"] == "input" for t in ("input.text", "input.image", "input.video"))
    # Output port types must match the wire types downstream nodes expect.
    out = {p["name"]: p["type"] for p in cat["input.text"]["outputs"]}
    assert out == {"text": "text"}
    out = {p["name"]: p["type"] for p in cat["input.image"]["outputs"]}
    assert out["image"] == "image"
    out = {p["name"]: p["type"] for p in cat["input.video"]["outputs"]}
    assert out["video"] == "video"


def test_preview_requires_value() -> None:
    """The mock downstream must actually require an input (not silently pass)."""
    with pytest.raises(ValueError, match="requires an input"):
        PreviewNode().run(params={}, inputs={}, work_dir=".", node_id="x")


def test_default_max_bytes_is_2gb() -> None:
    assert DEFAULT_MAX_BYTES == 2 * 1024 * 1024 * 1024
