"""Regression guards for the owner's T1/T2 retest feedback of 2026-09-29.

- T1-1: no gateway configured => the drawer must say so instead of silently
  showing gradient placeholders.
- T1-4: the queue dropdown must be closable, never forced open by polling, and
  finished jobs must be clearable.
- T2: widening the right rail must resize the grid track, not overflow it.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import time
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parent.parent / "studio" / "static"
APP_JS = STATIC / "app.js"
PREVIEW_JS = STATIC / "preview3d.js"
INDEX_HTML = STATIC / "index.html"
STYLE_CSS = STATIC / "style.css"


def _function_body(src: str, name: str) -> str:
    start = src.index(f"function {name}(")
    brace = src.index("{", start)
    depth = 0
    for i in range(brace, len(src)):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return src[brace : i + 1]
    raise AssertionError(f"unbalanced body for {name}")


def test_queue_visibility_is_operator_controlled() -> None:
    body = _function_body(APP_JS.read_text(encoding="utf-8"), "renderQueue")
    # The old bug: toggle("hidden", state.jobs.length === 0) re-opened the panel
    # on every poll tick, so it could never be closed.
    assert 'toggle("hidden", state.jobs.length === 0)' not in body
    assert 'toggle("hidden", !state.queueOpen)' in body


def test_finished_jobs_have_no_cancel_button() -> None:
    body = _function_body(APP_JS.read_text(encoding="utf-8"), "renderQueue")
    assert '"已取消"' not in body, "ok jobs must not be labelled 已取消"
    assert "queue-cancel" in body and "active" in body


def test_queue_close_clear_and_escape_wired() -> None:
    src = APP_JS.read_text(encoding="utf-8")
    html = INDEX_HTML.read_text(encoding="utf-8")
    for el in ("btnQueueClose", "btnQueueClear"):
        assert f'id="{el}"' in html
        assert el in src
    assert "/api/jobs/clear" in src
    assert re.search(r'evt\.key === "Escape" && state\.queueOpen', src)


def test_rail_resize_drives_grid_track() -> None:
    js = PREVIEW_JS.read_text(encoding="utf-8")
    css = STYLE_CSS.read_text(encoding="utf-8")
    body = _function_body(js, "initRailResizer")
    assert "--rail-w" in body
    assert "rail.style.width" not in body, "sizing the aside overflows its grid column"
    assert "var(--rail-w" in css
    assert "minmax(0, 1fr)" in css


def test_gateway_warning_banner_wired() -> None:
    src = APP_JS.read_text(encoding="utf-8")
    html = INDEX_HTML.read_text(encoding="utf-8")
    assert 'id="gwWarn"' in html and 'id="gwWarnLink"' in html
    assert "showGatewayWarning(" in _function_body(src, "loadSettings")
    boot = _function_body(src, "boot")
    assert "loadSettings()" in boot


pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from studio.server import JobManager, create_app  # noqa: E402


def test_job_manager_clear_finished_keeps_running() -> None:
    mgr = JobManager()
    done = mgr.create("a", 1)
    done.finished_at = time.time()
    live = mgr.create("b", 1)
    assert mgr.clear_finished() == 1
    ids = [j["id"] for j in mgr.list_jobs()]
    assert ids == [live.id]


def test_clear_endpoint(tmp_path) -> None:
    client = TestClient(create_app(default_work_dir=str(tmp_path / "w")))
    res = client.post("/api/jobs/clear")
    assert res.status_code == 200
    assert res.json() == {"cleared": 0}


def test_motion_filter_known_and_static() -> None:
    from studio.nodes.production import motion_filter

    assert motion_filter("static", frames=48, fps=24, size=512) is None
    assert motion_filter("weird", frames=48, fps=24, size=512) is None
    zin = motion_filter("dolly_in", frames=48, fps=24, size=512)
    assert zin is not None and "zoompan" in zin and "d=48" in zin and "s=512x512" in zin
    assert "1+0.18*on/47" in zin
    assert "1.18-0.18*on/47" in motion_filter("pull_out", frames=48, fps=24, size=512)
    assert "(1-on/47)" in motion_filter("pan_left", frames=48, fps=24, size=512)


def test_clip_command_is_list_argv() -> None:
    from studio.nodes.production import clip_command

    cmd = clip_command("a.png", "b.mp4", dur=2, fps=24, size=512, motion="dolly_in")
    assert isinstance(cmd, list) and cmd[-1] == "b.mp4"
    assert "-frames:v" in cmd and cmd[cmd.index("-frames:v") + 1] == "48"
    static = clip_command("a.png", "b.mp4", dur=2, fps=24, size=512, motion="static")
    assert "-loop" in static


def test_batch_shot_with_still_renders_real_clip(tmp_path) -> None:
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not on PATH")
    from PIL import Image

    still = tmp_path / "s1.png"
    Image.new("RGB", (320, 180), (40, 90, 160)).save(still)
    client = TestClient(create_app(default_work_dir=str(tmp_path / "w")))
    shot = {"id": "s1", "description": "d", "duration": 1, "image": str(still), "motion": "dolly_in"}
    res = client.post("/api/batch-shots", json={"shots": [shot, {"id": "s2", "duration": 1}]})
    assert res.status_code == 200, res.text
    ids = [j["job_id"] for j in res.json()["jobs"]]
    deadline = time.time() + 60
    while time.time() < deadline:
        listed = {j["id"]: j for j in client.get("/api/jobs").json()}
        if all(listed[i]["status"] in {"ok", "error", "cancelled"} for i in ids):
            break
        time.sleep(0.1)
    real, mock = listed[ids[0]], listed[ids[1]]
    assert real["status"] == "ok", real
    clip = next(iter(real["report"]["results"].values()))["outputs"]["videos"]["clips"][0]["video"]
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "stream=width", "-of", "csv=p=0", clip],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode == 0:
        assert probe.stdout.strip() == "512"
    # a shot with no still still falls back to the mock clip
    assert mock["status"] == "ok"
    assert "video" in next(iter(mock["report"]["results"].values()))["outputs"]


def test_drawer_card_shows_batch_video_result() -> None:
    src = APP_JS.read_text(encoding="utf-8")
    assert "function harvestShotVideos(" in src
    body = _function_body(src, "batchGenerateShots")
    assert "image:" in body and "motion:" in body
    assert "shot-video" in _function_body(src, "renderDrawerCards")


def test_settings_file_with_bom_is_read(tmp_path) -> None:
    """PowerShell 5 ``Set-Content -Encoding utf8`` writes a BOM; do not drop the file."""
    from studio.settings import StudioSettings

    p = tmp_path / "studio_settings.json"
    p.write_bytes(b'\xef\xbb\xbf{"litellm_base_url": "http://gw", "litellm_api_key": "k"}')
    s = StudioSettings.load(p)
    assert s.litellm_base_url == "http://gw"
    assert s.litellm_api_key == "k"
