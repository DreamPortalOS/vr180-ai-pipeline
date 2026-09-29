"""Regression guards for the owner's T1/T2 retest feedback of 2026-09-29.

- T1-1: no gateway configured => the drawer must say so instead of silently
  showing gradient placeholders.
- T1-4: the queue dropdown must be closable, never forced open by polling, and
  finished jobs must be clearable.
- T2: widening the right rail must resize the grid track, not overflow it.
"""

from __future__ import annotations

import re
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
