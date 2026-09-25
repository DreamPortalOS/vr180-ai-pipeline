"""Source-level guards for studio/static/app.js regressions found in browser QA."""

from __future__ import annotations

import re
from pathlib import Path

APP_JS = Path(__file__).resolve().parent.parent / "studio" / "static" / "app.js"
INDEX_HTML = APP_JS.parent / "index.html"
STYLE_CSS = APP_JS.parent / "style.css"


def _function_body(src: str, name: str) -> str:
    start = src.index(f"async function {name}(")
    brace = src.index("{", start)
    depth = 0
    for i in range(brace, len(src)):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return src[brace : i + 1]
    raise AssertionError(f"unterminated function {name}")


def _function_body_sync(src: str, name: str) -> str:
    """Find a non-async ``function name(`` body by brace matching."""
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
    raise AssertionError(f"unterminated function {name}")


def test_upload_file_reads_the_response_body_once() -> None:
    """#417: a second ``res.json()`` threw "body stream already read" on every
    successful upload, so dropped files never became input nodes."""
    body = _function_body(APP_JS.read_text(encoding="utf-8"), "uploadFile")
    reads = re.findall(r"\bres\.(json|text|blob|arrayBuffer)\(\)", body)
    assert len(reads) == 1, f"uploadFile must read the response once, found {reads}"


def test_draw_thumb_box_geometry_declared_before_branches() -> None:
    """#417: declaring ``bx/by/bw/bh`` inside one branch made the other branch
    ReferenceError and abort the whole canvas redraw. The box geometry must be
    declared once before the loaded/placeholder branches use it."""
    src = APP_JS.read_text(encoding="utf-8")
    body = _function_body_sync(src, "draw")
    # ``bx`` is assigned exactly once in draw() now (shared by all branches).
    assigns = re.findall(r"(?<![a-zA-Z])bx\s*=", body)
    assert len(assigns) == 1, f"bx should be assigned once in draw(), found {len(assigns)}"


def test_draw_history_arrows_guarded_by_history_length() -> None:
    """The ◀▶ arrows must only render when a node has ≥2 history entries, so a
    never-run node card doesn't draw dead controls."""
    src = APP_JS.read_text(encoding="utf-8")
    body = _function_body_sync(src, "drawHistoryArrows")
    assert "entries.length < 2" in body, "drawHistoryArrows must early-return on <2 entries"
    assert "return" in body


def test_node_run_thumb_resolves_history_index() -> None:
    """The ◀▶ switcher reads from ``rep.history[node.id]`` entries, not just the
    latest ``results`` — guard the history-index branch stays present."""
    src = APP_JS.read_text(encoding="utf-8")
    body = _function_body_sync(src, "nodeRunThumb")
    assert "rep.history" in body and "historyIdx" in body


def test_mousedown_history_arrow_click_does_not_start_drag() -> None:
    """Clicking a ◀▶ arrow must step history and return without entering node
    drag mode (otherwise every arrow click also moves the node)."""
    src = APP_JS.read_text(encoding="utf-8")
    # The mousedown listener short-circuits on history clicks.
    assert 'click.kind === "history"' in src
    assert "stepNodeHistory" in src


def test_drawer_state_persisted_to_local_storage() -> None:
    """Open/closed, height, shot order and checks must round-trip through
    localStorage so a refresh keeps the lead's layout (acceptance criterion)."""
    src = APP_JS.read_text(encoding="utf-8")
    assert "LS_DRAWER" in src and "localStorage.setItem" in src
    assert "localStorage.getItem" in src


def test_b_key_toggles_drawer_outside_text_fields() -> None:
    """``B`` toggles the drawer but must not fire while typing in an input or
    textarea (would eat every "b" keystroke in the text node editor)."""
    src = APP_JS.read_text(encoding="utf-8")
    assert 'evt.key.toLowerCase() === "b"' in src
    assert "isContentEditable" in src or 'tagName === "INPUT"' in src


def test_index_html_has_drawer_and_no_right_gallery() -> None:
    """The right-rail gallery box is gone; the bottom drawer markup exists."""
    html = INDEX_HTML.read_text(encoding="utf-8")
    assert "shotDrawer" in html
    assert "galleryBox" not in html, "right-rail gallery must be removed"
    # Both card and table views are present.
    assert "drawerCards" in html and "drawerTableWrap" in html


def test_style_css_has_drawer_and_collapsed_rules() -> None:
    css = STYLE_CSS.read_text(encoding="utf-8")
    assert ".shot-drawer" in css
    assert ".shot-drawer.collapsed" in css
    assert ".shot-card" in css and ".shot-card .thumb.placeholder" in css


PREVIEW_JS = APP_JS.parent / "preview3d.js"


def test_camera_slider_rows_resolve_to_real_ids() -> None:
    """#416: camSlider built ``cam${key}Row`` (camyawRow) while index.html has
    camYawRow, so the camera sliders never rendered in the Studio panel."""
    src = PREVIEW_JS.read_text(encoding="utf-8")
    html = INDEX_HTML.read_text(encoding="utf-8")
    keys = re.findall(r'camSlider\("(\w+)"', src)
    assert keys, "camSlider calls not found"
    assert "charAt(0).toUpperCase()" in src, "camSlider must capitalise the key when building the row id"
    for key in keys:
        row_id = f"cam{key[0].upper()}{key[1:]}Row"
        assert f'id="{row_id}"' in html, f"index.html has no #{row_id} for camSlider('{key}')"
