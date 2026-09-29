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


def _read_app() -> str:
    return APP_JS.read_text(encoding="utf-8")


def test_issue_419_run_modes_queued_not_synchronous() -> None:
    """Runs go through POST /api/jobs (the queue), not a blocking /api/run, so
    the UI can show progress and offer cancel (队列与停止)."""
    src = _read_app()
    assert "/api/jobs" in src, "runProject must enqueue via /api/jobs"
    assert "run_mode" in src, "the run payload must carry run_mode"
    # All three scopes the card asks for.
    for mode in ('"all"', '"node"', '"downstream"'):
        assert mode in src, f"run mode {mode} must be selectable"


def test_issue_419_locked_nodes_skip_execution() -> None:
    """The locked flag must round-trip the project JSON so the executor can skip
    locked nodes (锁定节点). It appears in both serialisation directions."""
    src = _read_app()
    assert "locked" in src, "node lock flag must be tracked"
    # normaliseProject and toServerProject both carry it.
    assert "!!n.locked" in src, "normalizeProject must read n.locked"
    # graph.py is the executor side.
    graph = (APP_JS.parent.parent / "graph.py").read_text(encoding="utf-8")
    assert "locked" in graph and '"locked"' in graph, "run_graph must handle locked nodes"


def test_issue_419_queue_ui_and_cancel_wired() -> None:
    """The queue panel lists running jobs and offers cancel (顶栏任务数 + 取消)."""
    src = _read_app()
    for sym in ("/api/jobs", "stop-all", "cancel", "queueCount", "renderQueue"):
        assert sym in src, f"app.js missing queue symbol {sym}"
    html = INDEX_HTML.read_text(encoding="utf-8")
    for el in ("queuePanel", "queueList", "queueCount", "btnQueueStopAll", "btnStop"):
        assert f'id="{el}"' in html, f"index.html missing queue element #{el}"


def test_issue_419_port_drag_out_filters_by_type() -> None:
    """Dragging from an output port onto empty canvas opens a spotlight filtered
    to nodes whose inputs accept that port's type (端口拖出按类型过滤)."""
    src = _read_app()
    assert "typesAccepting" in src, "port drag-out must compute compatible node types"
    assert "filterType" in src, "the spotlight must accept a type filter"
    assert "autoConnect" in src, "picking a node after port drag-out must auto-connect"


def test_issue_419_spotlight_shortcuts_present() -> None:
    """空格/斜杠/双击/右键 open the add-node spotlight (Spotlight 加节点)."""
    src = _read_app()
    assert "openSpotlight" in src, "openSpotlight must exist"
    assert 'evt.key === "/"' in src or 'code === "/"' in src, '"/" must open the spotlight'
    assert "Space" in src, "Space must open the spotlight"
    assert "contextmenu" in src, "right-click must open the spotlight"
    assert "dblclick" in src, "double-click must open the spotlight"


def test_issue_419_batch_shots_submit_only_checked() -> None:
    """只跑勾选镜头: the drawer submits exactly the checked shot ids.

    #418 owns the bottom drawer and the ``shotChecked`` state ({nodeId: [ids]});
    #419 adds the 批量生成视频 button and a collector that walks that state.
    """
    src = _read_app()
    assert "shotChecked" in src, "checked-shot state must exist"
    assert "collectCheckedShots" in src, "a collector must gather only checked shots"
    assert "/api/batch-shots" in src, "batch generation must POST to /api/batch-shots"
    html = INDEX_HTML.read_text(encoding="utf-8")
    for el in ("shotDrawer", "btnBatchGen"):
        assert f'id="{el}"' in html, f"index.html missing drawer element #{el}"


def test_issue_419_new_status_colors_have_legend() -> None:
    """locked/cancelled statuses get distinct colors AND legend entries, so the
    canvas readout matches the legend (取消/锁定)."""
    src = _read_app()
    assert '"locked"' in src or "locked" in src, "locked color branch must exist"
    assert '"cancelled"' in src or "cancelled" in src, "cancelled color branch must exist"
    html = INDEX_HTML.read_text(encoding="utf-8")
    assert "锁定" in html and "取消" in html, "status legend must include 锁定/取消"


def test_all_changed_js_parse_with_node_check() -> None:
    """Every JS asset must be syntactically valid; CI has no browser, so a
    ReferenceError/SyntaxError would be invisible until manual QA (the class of
    bug this guard closes for #417/#416/#418)."""
    import shutil
    import subprocess

    node = shutil.which("node")
    if node is None:
        return  # node unavailable in CI; the static asserts above still guard
    for js in sorted(APP_JS.parent.glob("*.js")):
        proc = subprocess.run([node, "--check", str(js)], capture_output=True, text=True, timeout=30)
        assert proc.returncode == 0, f"node --check failed for {js.name}:\n{proc.stderr}"


def test_batch_generate_uses_the_drawer_scope_and_default_selection() -> None:
    """#419: collectCheckedShots read only explicit shotChecked entries on project
    nodes, so the default all-checked drawer (and the __run_gallery__ scope)
    collected nothing and 批量生成 never fired."""
    src = APP_JS.read_text(encoding="utf-8")
    start = src.index("function collectCheckedShots()")
    body = src[start : src.index("function updateBatchButton()", start)]
    assert "state.drawerScope" in body
    assert "checkedShotIds(" in body, "must reuse the drawer's default-all-checked rule"
    assert "state.drawerScope = scope" in src


# ----------------------------------------------------------------- #428
# Dome panel loads MP4/MOV as a <video> texture (texImage2D(video) per
# tick), with player controls (play/pause, scrub, loop) and coverage POSTed
# only when paused — never continuously during playback. The image load
# path must not regress.


def test_issue_428_video_texture_branch_exists() -> None:
    """uploadTexture must dispatch on kind=="video" and feed the
    HTMLVideoElement to texImage2D; setVideo must bind it as the source."""
    src = PREVIEW_JS.read_text(encoding="utf-8")
    assert "setVideo(v)" in src, "DomePreview must expose setVideo(video)"
    assert 'this.kind = "video"' in src, "a video source kind must be tracked"
    # texImage2D's source is the video when kind is video, else the image.
    assert 'this.kind === "video" ? this.video : this.image' in src, (
        "uploadTexture must pass the HTMLVideoElement to texImage2D for video"
    )
    assert "gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, src)" in src, (
        "texImage2D must upload the chosen source"
    )


def test_issue_428_per_frame_upload_only_on_play_or_seek() -> None:
    """The per-tick video upload must be gated so it fires only while the video
    is playing OR right after a seek — not every redraw while paused (would
    burn GPU on an unchanged frame). This is the acceptance criterion:
    '每帧纹理更新只在播放或 seek 时发生'."""
    src = PREVIEW_JS.read_text(encoding="utf-8")
    assert "_videoFrameIsFresh" in src, "a freshness gate must exist"
    # The gate early-returns (no upload) when paused AND not seeking.
    assert "this.video.paused" in src and "this.seeking" in src, (
        "_videoFrameIsFresh must early-return when paused and not seeking"
    )
    # _tickVideoTexture (called from _draw) gates the upload on freshness.
    assert "if (!this._videoFrameIsFresh()) return;" in src, (
        "_tickVideoTexture must skip the upload when no fresh frame is available"
    )
    assert "this._tickVideoTexture();" in src, "_draw must drive the per-tick upkeep"
    # The camera viewport must NOT reset texVersion for video (it gates on the
    # version bump instead of the static hasTexture flag, so a video that
    # wasn't ready at load still recovers once frames arrive).
    assert 'p.kind !== "video" && !p.hasTexture' in src


def test_issue_428_coverage_measured_only_when_paused() -> None:
    """Coverage POSTs happen on pause/seek/load — never continuously during
    playback. measureCurrentFrame must skip while playing, and timeupdate
    must only refresh the UI (not POST)."""
    src = PREVIEW_JS.read_text(encoding="utf-8")
    assert "measureCurrentFrame" in src
    # Skips while playing (or while a seek is mid-flight).
    assert "!videoEl.paused" in src and "videoEl.seeking" in src
    # pause / seeked both trigger one measurement.
    assert '"pause"' in src and '"seeked"' in src
    # timeupdate only refreshes the UI — it must NOT call measureCurrentFrame.
    assert 'addEventListener("timeupdate", () => refreshPlayerUi())' in src, (
        "timeupdate must not POST coverage on every frame during playback"
    )


def test_issue_428_player_controls_resolve_to_real_ids() -> None:
    """The player markup must exist in index.html with the ids preview3d.js
    looks up, so a wrong-case id doesn't silently disable a control (the class
    of bug this guard closes for #416)."""
    html = INDEX_HTML.read_text(encoding="utf-8")
    for ident in (
        "domePlayer",
        "btnPlayPause",
        "domeProgress",
        "domeLoop",
        "domeCurTime",
        "domeDur",
    ):
        assert f'id="{ident}"' in html, f"index.html missing player element #{ident}"


def test_issue_428_player_styles_present() -> None:
    css = STYLE_CSS.read_text(encoding="utf-8")
    for cls in (".dome-player", ".player-progress", ".player-time", ".dome-video"):
        assert cls in css, f"style.css missing player rule {cls}"
    # The hidden state for the player must be defined (toggled until a video loads).
    assert ".dome-player.hidden" in css
    # A disabled control must visually read as disabled (the play button starts
    # disabled until a video is bound).
    assert "button:disabled" in css


def test_issue_428_image_load_path_not_regressed() -> None:
    """Loading an image still measures it on the server and applies the stats;
    the new video dispatch did not replace the image branch."""
    src = PREVIEW_JS.read_text(encoding="utf-8")
    assert "new Image()" in src, "image load branch must remain"
    assert "analyzeOnServer(imageToCanvas(img))" in src
    assert "applyStats(stats, img)" in src
    # The file input still accepts both images and video.
    html = INDEX_HTML.read_text(encoding="utf-8")
    assert 'accept="image/*,video/*"' in html
