"""Tests for scripts/make_handoff.py — the WORKLOG handoff summary generator.

No real ``gh`` process is spawned: :func:`scripts.make_handoff._gh` takes a
``runner`` kwarg so we inject a fake that returns canned JSON payloads. Local
artefact scanning runs against ``tmp_path`` mp4 files (no real video data) plus
in-line sidecar JSON. Worklog idempotence is verified end-to-end on a temp file.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from scripts.make_handoff import (
    HandoffData,
    MergedPR,
    OpenIssue,
    VideoArtefact,
    _anchor,
    _worklog_heading,
    append_to_worklog,
    build_handoff,
    fetch_merged_prs,
    fetch_open_issues,
    render_json,
    render_markdown,
    scan_artefacts,
)

# ---------------------------------------------------------------------------
# Fake gh runner
# ---------------------------------------------------------------------------


def _fake_runner(stdout_by_argv: dict[tuple[str, ...], list[dict]]) -> SimpleNamespace:
    """Return a fake subprocess runner keyed by gh argv.

    ``stdout_by_argv`` maps a tuple argv-prefix to a JSON-serializable list.
    The runner matches on argv starting with ``("pr", "list")`` or
    ``("issue", "list")`` — exactly the two query shapes :func:`_gh` issues.
    """

    def runner(cmd: list[str], **_: object) -> SimpleNamespace:
        argv = tuple(cmd)
        # Strip gh binary name from the front (the real _gh prepends gh_bin).
        if argv and argv[0] in {"gh", "fake-gh"}:
            argv = argv[1:]
        data: list[dict] | None = None
        for key, val in stdout_by_argv.items():
            if argv[: len(key)] == key:
                data = val
                break
        if data is None:
            return SimpleNamespace(returncode=1, stdout="[]", stderr="unknown query")
        return SimpleNamespace(returncode=0, stdout=json.dumps(data), stderr="")

    return SimpleNamespace(runner=runner)


def _runner_for(
    prs: list[dict] | None = None,
    issues: list[dict] | None = None,
) -> SimpleNamespace:
    if prs is None:
        prs = []
    if issues is None:
        issues = []
    return _fake_runner(
        {
            ("pr", "list"): prs,
            ("issue", "list"): issues,
        }
    )


# ---------------------------------------------------------------------------
# Merged PR fetch + time-window
# ---------------------------------------------------------------------------


def _now() -> datetime:
    return datetime.now(timezone.utc)


def test_fetch_merged_prs_empty_by_default():
    runner = _runner_for(prs=[], issues=[])
    result = fetch_merged_prs(_now(), runner=runner.runner)
    assert result == []


def test_fetch_merged_prs_sorted_by_merged_at():
    """PRs are sorted by mergedAt ascending (the digest is time-ordered)."""
    prs = [
        {"number": 2, "title": "B", "mergedAt": "2026-09-01T10:00:00+00:00"},
        {"number": 1, "title": "A", "mergedAt": "2026-09-01T08:00:00+00:00"},
        {"number": 3, "title": "C", "mergedAt": "2026-09-01T09:00:00+00:00"},
    ]
    runner = _runner_for(prs=prs)
    # Window starts before all of them so none are filtered out.
    since = datetime(2026, 9, 1, 0, 0, 0, tzinfo=timezone.utc)
    result = fetch_merged_prs(since, runner=runner.runner)
    assert [p.number for p in result] == [1, 3, 2]
    assert result[0].title == "A"


def test_fetch_merged_prs_filters_to_time_window():
    """Only PRs with mergedAt >= since are kept, sorted ascending.

    gh pr list returns a page of recent merges (no --since); the Python side
    must slice the window. PRs straddling the boundary on both sides must be
    partitioned correctly.
    """
    since = datetime(2026, 9, 1, 9, 0, 0, tzinfo=timezone.utc)
    prs = [
        # Before the window — must be dropped.
        {"number": 170, "title": "old-pre-window", "mergedAt": "2026-09-01T06:00:00+00:00"},
        {"number": 172, "title": "just-before", "mergedAt": "2026-09-01T08:59:59+00:00"},
        # At/after the window — must be kept (boundary is inclusive).
        {"number": 175, "title": "boundary-exact", "mergedAt": "2026-09-01T09:00:00+00:00"},
        {"number": 178, "title": "inside-1", "mergedAt": "2026-09-01T11:00:00+00:00"},
        {"number": 181, "title": "inside-2", "mergedAt": "2026-09-01T13:00:00+00:00"},
        # Z-suffix timestamps (gh's real shape) must parse too.
        {"number": 185, "title": "z-suffix", "mergedAt": "2026-09-01T15:00:00Z"},
    ]
    runner = _runner_for(prs=prs)
    result = fetch_merged_prs(since, runner=runner.runner)

    assert [p.number for p in result] == [175, 178, 181, 185]
    # And the kept list is still sorted by mergedAt ascending.
    assert [p.merged_at for p in result] == sorted(p.merged_at for p in result)


def test_fetch_merged_prs_argv_has_no_since_and_uses_limit():
    """Regression for issue #188: gh pr list has NO --since flag.

    The fabricated --since was rejected by gh, _gh silently returned [], and
    the digest showed "无合并 PR" while real merges piled up. The argv must
    use the real --limit flag instead and must NOT contain --since at all.
    """
    captured: list[list[str]] = []

    def runner(cmd: list[str], **_: object) -> SimpleNamespace:
        captured.append(list(cmd))
        return SimpleNamespace(returncode=0, stdout="[]", stderr="")

    since_dt = datetime(2026, 8, 30, 0, 0, 0, tzinfo=timezone.utc)
    fetch_merged_prs(since_dt, runner=runner)

    assert len(captured) == 1
    cmd = captured[0]
    assert cmd[0] == "gh"
    assert "--since" not in cmd  # the fabricated flag must never come back
    assert "--limit" in cmd
    limit_arg = cmd[cmd.index("--limit") + 1]
    assert int(limit_arg) == 100  # default; large enough for a busy window
    assert "--state" in cmd and "merged" in cmd
    # --json takes a single comma-joined fields argument containing mergedAt.
    assert "--json" in cmd
    json_arg = cmd[cmd.index("--json") + 1]
    assert "mergedAt" in json_arg and "number" in json_arg and "title" in json_arg


def test_fetch_merged_prs_gh_failure_surfaces_error(capsys):
    """gh returning non-zero must NOT be silently swallowed (issue #188 core).

    The fabricated --since was rejected by gh, _gh returned [] with no output,
    and the digest showed "无合并 PR" — the bug was silent. Now the stderr
    must be surfaced, labeled with which query failed, so a future bad flag
    can never hide the same way.
    """

    def runner(cmd: list[str], **_: object) -> SimpleNamespace:
        return SimpleNamespace(
            returncode=1,
            stdout="fatal: not a git repo",
            stderr="unknown flag: --since",
        )

    result = fetch_merged_prs(_now(), runner=runner)
    assert result == []
    err = capsys.readouterr().err
    # The failure is surfaced, labeled, and carries the gh stderr verbatim.
    assert "[make_handoff]" in err
    assert "gh pr list" in err
    assert "failed" in err
    assert "unknown flag: --since" in err


def test_fetch_merged_prs_gh_garbage_stdout_surfaces_error(capsys):
    """Non-JSON stdout is also surfaced rather than silently dropped."""

    def runner(cmd: list[str], **_: object) -> SimpleNamespace:
        return SimpleNamespace(returncode=0, stdout="not json at all", stderr="")

    result = fetch_merged_prs(_now(), runner=runner)
    assert result == []
    err = capsys.readouterr().err
    assert "[make_handoff]" in err
    assert "non-JSON" in err


def test_build_handoff_since_window():
    """build_handoff with an explicit --since slices the window in Python.

    The window is no longer a gh flag (no --since on gh pr list); it is
    applied client-side. The digest's ``since`` label still records it, and
    the gh argv uses --limit not --since.
    """
    fixed = "2026-08-29T00:00:00Z"
    captured: list[list[str]] = []

    def runner(cmd: list[str], **_: object) -> SimpleNamespace:
        captured.append(list(cmd))
        return SimpleNamespace(returncode=0, stdout="[]", stderr="")

    data = build_handoff(since=fixed, runner=runner, video_dir=Path("/nonexistent"))
    assert "2026-08-29" in data.since
    # Both queries should have been issued (PR list then issue list).
    assert len(captured) == 2
    pr_cmd = captured[0]
    assert "--since" not in pr_cmd  # no fabricated flag
    assert "--limit" in pr_cmd


def test_build_handoff_since_window_filters_in_python():
    """The --since window is applied on the Python side against mergedAt."""
    fixed = "2026-08-29T00:00:00Z"
    prs = [
        {"number": 10, "title": "before window", "mergedAt": "2026-08-28T12:00:00Z"},
        {"number": 11, "title": "in window", "mergedAt": "2026-08-29T06:00:00Z"},
        {"number": 12, "title": "later", "mergedAt": "2026-08-30T06:00:00Z"},
    ]
    runner = _runner_for(prs=prs, issues=[])
    data = build_handoff(since=fixed, runner=runner.runner, video_dir=Path("/nonexistent"))
    assert [p.number for p in data.merged_prs] == [11, 12]


def test_build_handoff_since_hours_default():
    """Without --since, the default 12-hour window is used to filter in Python."""
    captured: list[list[str]] = []

    def runner(cmd: list[str], **_: object) -> SimpleNamespace:
        captured.append(list(cmd))
        return SimpleNamespace(returncode=0, stdout="[]", stderr="")

    _ = build_handoff(since_hours=12, runner=runner, video_dir=Path("/nonexistent"))
    pr_cmd = captured[0]
    assert "--since" not in pr_cmd  # no fabricated flag
    assert "--limit" in pr_cmd


# ---------------------------------------------------------------------------
# Open issues: grouping + owner-action flags
# ---------------------------------------------------------------------------


def _labels(*names: str) -> list[dict]:
    return [{"name": n} for n in names]


def test_fetch_open_issues_classifies_stage_and_priority():
    issues = [
        {
            "number": 173,
            "title": "segment_concat",
            "labels": _labels("stage:done", "prio:P2"),
        },
        {
            "number": 174,
            "title": "prompt_library",
            "labels": _labels("stage:ready", "prio:P1"),
        },
        {
            "number": 180,
            "title": "blocked card",
            "labels": _labels("stage:blocked", "prio:P0", "needs:muso-decision"),
        },
    ]
    runner = _runner_for(issues=issues)
    result = fetch_open_issues(runner=runner.runner)

    by_num = {i.number: i for i in result}
    assert by_num[173].stage == "stage:done"
    assert by_num[173].priority == "P2"
    assert by_num[174].stage == "stage:ready"
    assert by_num[180].needs_owner is True
    assert "needs:muso-decision" in by_num[180].labels


def test_fetch_open_issues_sorts_ready_before_done():
    """stage:ready cards sort before stage:done; within a stage, P0 first."""
    issues = [
        {"number": 1, "title": "t1", "labels": _labels("stage:done", "prio:P0")},
        {"number": 2, "title": "t2", "labels": _labels("stage:ready", "prio:P1")},
        {"number": 3, "title": "t3", "labels": _labels("stage:ready", "prio:P0")},
    ]
    runner = _runner_for(issues=issues)
    result = fetch_open_issues(runner=runner.runner)
    nums = [i.number for i in result]
    # ready cards (2, 3) before done (1); within ready, P0 (3) before P1 (2).
    assert nums[0] == 3
    assert nums[1] == 2
    assert nums[2] == 1


def test_fetch_open_issues_handles_string_labels():
    """gh sometimes returns labels as a list of strings; parsing should still work."""
    issues = [
        {"number": 1, "title": "t", "labels": ["stage:ready", "prio:P0", "needs:hw-verify"]},
    ]
    runner = _runner_for(issues=issues)
    result = fetch_open_issues(runner=runner.runner)
    assert len(result) == 1
    assert result[0].stage == "stage:ready"
    assert result[0].priority == "P0"
    assert result[0].needs_owner is True


# ---------------------------------------------------------------------------
# Markdown rendering sections
# ---------------------------------------------------------------------------


def _owner_issue(n: int, title: str, flag: str = "needs:muso-decision") -> OpenIssue:
    return OpenIssue(number=n, title=title, labels=[flag, "stage:ready", "prio:P0"])


def _active_issue(n: int, title: str) -> OpenIssue:
    return OpenIssue(number=n, title=title, labels=["stage:in-progress", "prio:P1"])


def test_render_markdown_sections():
    """All five sections render with the expected rows."""
    data = HandoffData(
        since="2026-09-01T00:00:00+00:00 .. now",
        merged_prs=[MergedPR(number=179, title="pip timeout 推广", merged_at="2026-09-01T09:00:00+00:00")],
        open_issues=[
            _active_issue(175, "segment_concat"),
            _owner_issue(180, "owner decision needed"),
        ],
        artefacts=[
            VideoArtefact(
                filename="scene_v1.mp4",
                mtime=1.0,
                resolution="5760×2880",
                has_audio="yes",
                qa_verdict="VR180 (180° 3D SBS)",
            ),
        ],
    )
    md = render_markdown(data)

    assert "① 本轮合并" in md
    assert "#179" in md and "pip timeout 推广" in md
    assert "2026-09-01 09:00" in md

    assert "② 当前在跑 / 排队" in md
    assert "#175" in md and "segment_concat" in md

    assert "③ 待 owner 决策/操作" in md
    assert "⚠️" in md
    assert "#180" in md and "needs:muso-decision" in md

    assert "④ 可验收产物" in md
    assert "scene_v1.mp4" in md and "5760×2880" in md and "VR180 (180° 3D SBS)" in md
    assert "有音轨" in md

    assert "⑤ 已知风险/未完成项" in md
    assert "#180" in md  # P0 risk, so also appears in ⑤


def test_render_markdown_owner_item_absent_from_section_2():
    """Owner-action cards must NOT appear in ②; they live only in ③."""
    data = HandoffData(
        open_issues=[
            _active_issue(175, "active"),
            _owner_issue(180, "owner needed", flag="needs:hw-verify"),
        ],
    )
    md = render_markdown(data)
    section2_start = md.index("## ② 当前在跑 / 排队")
    section3_start = md.index("## ③ 待 owner 决策/操作")
    section2 = md[section2_start:section3_start]
    assert "#175" in section2
    assert "#180" not in section2
    assert "#180" in md[section3_start:]


def test_render_markdown_empty_sections():
    """Every section renders a fallback line when its data is empty."""
    data = HandoffData(since="..")
    md = render_markdown(data)
    assert "无合并 PR" in md
    assert "无在跑/排队卡" in md
    assert "无待 owner 决策项" in md
    assert "无成片" in md
    assert "无 P0 风险项" in md


def test_render_json_parses_and_roundtrips():
    data = HandoffData(
        since="2026-09-01T00:00:00+00:00 .. now",
        merged_prs=[MergedPR(1, "t", "2026-09-01T09:00:00+00:00")],
        open_issues=[OpenIssue(2, "t2", ["stage:ready", "prio:P0"])],
        artefacts=[VideoArtefact("v.mp4", 1.0, "5760×2880", "yes", "VR180 (180° 3D SBS)")],
    )
    obj = json.loads(render_json(data))
    assert obj["merged_prs"][0]["number"] == 1
    assert obj["artefacts"][0]["resolution"] == "5760×2880"


# ---------------------------------------------------------------------------
# Artefact scanner
# ---------------------------------------------------------------------------


def test_scan_artefacts_ordering_and_sidecar(tmp_path):
    """Newest mp4 first; sidecar QA verdict / resolution / audio are attached."""
    (tmp_path / "older.mp4").write_bytes(b"fake")
    (tmp_path / "newer.mp4").write_bytes(b"fake")
    sidecar_data = {
        "immersive": {"eye_resolution": [5760, 2880]},
        "qa": {"verdict": "VR180 (180° 3D SBS)", "checks": {"audio stream": {"status": "pass"}}},
    }
    (tmp_path / "newer.json").write_text(json.dumps(sidecar_data), encoding="utf-8")

    result = scan_artefacts(tmp_path)
    assert [a.filename for a in result] == ["newer.mp4", "older.mp4"]
    newer = result[0]
    assert newer.resolution == "5760×2880"
    assert newer.has_audio == "yes"
    assert newer.qa_verdict == "VR180 (180° 3D SBS)"
    assert result[1].resolution == ""  # no sidecar
    assert result[1].qa_verdict == "unknown"


def test_scan_artefacts_no_audio_flag(tmp_path):
    (tmp_path / "silent.mp4").write_bytes(b"fake")
    sidecar_data = {
        "qa": {
            "verdict": "plain 2D",
            "checks": {
                "audio stream": {
                    "status": "warn",
                    "detail": "no audio stream — video will be silent",
                }
            },
        }
    }
    (tmp_path / "silent.json").write_text(json.dumps(sidecar_data), encoding="utf-8")
    result = scan_artefacts(tmp_path)
    assert result[0].has_audio == "no"


def test_scan_artefacts_empty_qa_degrades(tmp_path):
    """An empty qa dict (no checks) must degrade to unknown, not raise."""
    (tmp_path / "bare.mp4").write_bytes(b"fake")
    (tmp_path / "bare.json").write_text(json.dumps({"qa": {}}), encoding="utf-8")
    result = scan_artefacts(tmp_path)
    assert result[0].has_audio == "unknown"
    assert result[0].qa_verdict == "unknown"


def test_scan_artefacts_missing_video_dir():
    result = scan_artefacts(Path("/proc/nonexistent-video-dir"))
    assert result == []


# ---------------------------------------------------------------------------
# Worklog append idempotence
# ---------------------------------------------------------------------------


def test_append_worklog_creates_file(tmp_path):
    path = tmp_path / "WORKLOG.md"
    since = "2026-09-01T00:00:00Z"
    appended, msg = append_to_worklog("# 交付摘要 (最近 12 小时)", since, worklog_path=path)
    assert appended is True
    assert "交付摘要 (2026-09-01)" in path.read_text(encoding="utf-8")
    assert "2026-09-01" in msg


def test_append_worklog_idempotent(tmp_path):
    """Calling append twice for the same day does not duplicate the entry."""
    path = tmp_path / "WORKLOG.md"
    since = "2026-09-01T00:00:00Z"
    digest = "# 交付摘要 (最近 12 小时)"

    appended_a, _ = append_to_worklog(digest, since, worklog_path=path)
    appended_b, msg = append_to_worklog(digest, since, worklog_path=path)

    assert appended_a is True
    assert appended_b is False
    assert "skipped" in msg
    content = path.read_text(encoding="utf-8")
    assert content.count("## 交付摘要 (2026-09-01)") == 1
    assert content.count("handoff: begin") == 1


def test_worklog_heading_extract_day():
    assert _worklog_heading("2026-09-01T00:00:00Z") == "## 交付摘要 (2026-09-01)"
    assert _worklog_heading("2026-09-01T09:00:00+00:00") == "## 交付摘要 (2026-09-01)"


def test_anchor_finds_matching_heading():
    body = "# WORKLOG\n\n## 交付摘要 (2026-09-01)\n\nhello\n"
    assert _anchor(body, "## 交付摘要 (2026-09-01)") == body.index("## 交付摘要 (2026-09-01)")
    assert _anchor(body, "## 交付摘要 (2026-09-02)") is None
