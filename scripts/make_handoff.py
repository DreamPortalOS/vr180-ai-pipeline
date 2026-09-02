#!/usr/bin/env python3
"""K-6 (#157): WORKLOG handoff summary generator.

Turns the lead's nightly "what did we ship / what's blocked / what needs the
owner" into a one-page Markdown (or JSON) digest by reading:

1. ``gh pr list --state merged --limit N --json number,title,mergedAt`` --
   recently merged PRs; the time window (``--since-hours`` / ``--since``,
   default last 12h) is applied on the Python side by filtering on
   ``mergedAt``. ``gh pr list`` has no ``--since`` flag, so we fetch a page of
   recent merges and slice it ourselves.
2. ``gh issue list --state open ...`` -- currently open cards, grouped by
   stage / priority. Cards carrying ``needs:muso-decision`` or
   ``needs:hw-verify`` are flagged as **owner action required**.
3. ``video/*.mp4`` -- the newest finished takes plus their same-stem sidecar
   JSON (QA verdict + resolution, when present).

The script is **read-only**: it never pushes, comments, or edits GitHub. The
only write it performs is the optional ``--append-worklog`` step, which
appends the digest under a date-anchored heading inside ``WORKLOG.md`` and
skips the append when that heading already exists (idempotent).

Tests inject fake ``gh`` output and fixture sidecar JSONs so CI (CPU-only,
no network, no GitHub auth) never touches the real CLI.

Usage:
    python scripts/make_handoff.py
    python scripts/make_handoff.py --since "2026-08-30T00:00:00Z"
    python scripts/make_handoff.py --json
    python scripts/make_handoff.py --append-worklog
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = REPO_ROOT / "video"
WORKLOG_PATH = REPO_ROOT / "WORKLOG.md"

DEFAULT_SINCE_HOURS = 12

# Labels that move a card into the "owner action required" section.
OWNER_ACTION_LABELS = {"needs:muso-decision", "needs:hw-verify"}

# Stage ordering for the "in-flight / queued" section.
STAGE_ORDER = {
    "stage:ready": 0,
    "stage:in-progress": 1,
    "stage:blocked": 2,
    "stage:review": 3,
    "stage:done": 4,
}
# Priority ordering (P0 first).
PRIO_ORDER = {"P0": 0, "P1": 1, "P2": 2, "P3": 3, "P4": 4}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class MergedPR:
    number: int
    title: str
    merged_at: str


@dataclass
class OpenIssue:
    number: int
    title: str
    labels: list[str] = field(default_factory=list)
    stage: str = "unclassified"
    priority: str = "P2"

    @property
    def needs_owner(self) -> bool:
        return bool(set(self.labels) & OWNER_ACTION_LABELS)


@dataclass
class VideoArtefact:
    filename: str
    mtime: float
    resolution: str = ""
    has_audio: str = "unknown"
    qa_verdict: str = "unknown"


@dataclass
class HandoffData:
    since: str = ""
    merged_prs: list[MergedPR] = field(default_factory=list)
    open_issues: list[OpenIssue] = field(default_factory=list)
    artefacts: list[VideoArtefact] = field(default_factory=list)


# ---------------------------------------------------------------------------
# gh CLI data source (injectable for tests)
# ---------------------------------------------------------------------------


def _gh(
    argv: list[str],
    gh_bin: str = "gh",
    runner: Any = subprocess.run,
) -> list[dict[str, Any]]:
    """Run a ``gh`` JSON query and return the parsed list.

    Tests inject a fake ``runner`` (or return JSON) so no real gh process is
    spawned. The real path uses list-form ``subprocess.run`` (no shell=True).

    On gh failure (returncode != 0 or unparseable stdout) the error is surfaced
    to stderr with a label naming the failed query, then an empty list is
    returned. It is **never** silently swallowed -- a silently-empty result is
    exactly the class of bug that let a fabricated ``--since`` flag hide for a
    full release (issue #188): gh rejected the flag, _gh returned ``[]``, and
    the digest printed "无合并 PR" with no hint anything had gone wrong.
    """
    cmd = [gh_bin, *argv]
    result = runner(cmd, capture_output=True, text=True, timeout=30, check=False)
    if result.returncode != 0:
        stderr = (getattr(result, "stderr", "") or "").strip()
        print(
            f"[make_handoff] gh {' '.join(argv[:2])} failed (rc={result.returncode}): {stderr}",
            file=sys.stderr,
        )
        return []
    try:
        parsed = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        print(
            f"[make_handoff] gh {' '.join(argv[:2])} returned non-JSON: {exc}",
            file=sys.stderr,
        )
        return []
    if isinstance(parsed, list):
        return parsed
    print(
        f"[make_handoff] gh {' '.join(argv[:2])} returned non-list JSON",
        file=sys.stderr,
    )
    return []


DEFAULT_MERGED_LIMIT = 100


def _parse_merged_at(merged_at: str) -> datetime | None:
    """Parse gh's ``mergedAt`` ISO-8601 into an aware UTC datetime, or None."""
    try:
        dt = datetime.fromisoformat(merged_at.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def fetch_merged_prs(
    since: datetime,
    gh_bin: str = "gh",
    runner: Any = subprocess.run,
    limit: int = DEFAULT_MERGED_LIMIT,
) -> list[MergedPR]:
    """Fetch PRs merged at or after *since* (UTC), sorted by merged_at.

    ``gh pr list`` has **no ``--since`` flag** (issue #188): it was fabricated
    and gh rejected it, silently producing an empty list. The real query is
    ``gh pr list --state merged --limit N --json number,title,mergedAt``; the
    time window is then applied on the Python side by filtering on
    ``mergedAt``. ``--limit`` defaults to 100 because gh's own default (30) is
    too small for a busy window and would drop real merges.
    """
    since_utc = since.astimezone(timezone.utc)
    rows = _gh(
        [
            "pr",
            "list",
            "--state",
            "merged",
            "--limit",
            str(limit),
            "--json",
            "number,title,mergedAt",
        ],
        gh_bin=gh_bin,
        runner=runner,
    )
    out: list[MergedPR] = []
    for row in rows:
        merged_at = row.get("mergedAt")
        if not merged_at:
            continue
        merged_dt = _parse_merged_at(merged_at)
        if merged_dt is None:
            # Keep unparseable timestamps out of a time-windowed list.
            continue
        if merged_dt < since_utc:
            continue
        out.append(MergedPR(number=row.get("number", 0), title=row.get("title", ""), merged_at=merged_at))
    out.sort(key=lambda p: p.merged_at)
    return out


def fetch_open_issues(
    gh_bin: str = "gh",
    runner: Any = subprocess.run,
) -> list[OpenIssue]:
    """Fetch open issues (cards) and classify by stage / priority labels."""
    rows = _gh(
        ["issue", "list", "--state", "open", "--json", "number,title,labels"],
        gh_bin=gh_bin,
        runner=runner,
    )
    out: list[OpenIssue] = []
    for row in rows:
        label_names = _label_names(row.get("labels", []))
        out.append(
            OpenIssue(
                number=row.get("number", 0),
                title=row.get("title", ""),
                labels=label_names,
                stage=_pick_stage(label_names),
                priority=_pick_priority(label_names),
            )
        )
    out.sort(key=_issue_sort_key)
    return out


# ---------------------------------------------------------------------------
# Local artefact scanner
# ---------------------------------------------------------------------------


FFPROBE_BIN = "ffprobe"
FFPROBE_TIMEOUT = 15  # seconds — never let one mp4 stall the whole digest


def _probe_with_ffprobe(
    mp4_path: Path,
    runner: Any = subprocess.run,
    ffprobe_bin: str = FFPROBE_BIN,
    timeout: float = FFPROBE_TIMEOUT,
    probe_resolution: bool = True,
    probe_audio: bool = True,
) -> tuple[str, str]:
    """Probe an mp4's resolution + audio presence via ``ffprobe``.

    Returns ``(resolution, has_audio)`` where ``has_audio`` is one of
    ``"yes"`` / ``"no"`` / ``"unknown"`` (matching the sidecar-derived
    vocabulary). Resolution is ``"WxH"`` (plain ASCII ``x`` — this is the
    machine-readable fallback, not the display form) or ``""`` on failure.

    ``probe_resolution`` / ``probe_audio`` gate which probe runs so a
    partial sidecar (one field known, one missing) only pays for the
    missing probe.

    Used only as a **fallback** when a sidecar is missing or a field is
    empty: ffprobe needs no metadata file to read container-level facts.
    ``QA 判定`` is intentionally NOT probed here — it requires the sidecar's
    QA conclusion and we do not run ``vr180_qa.py`` (that would slow the
    digest down).

    subprocess is **always list-form, never ``shell=True``** (CLAUDE.md red
    line), and the runner is injectable so tests assert without a real
    ffprobe binary. timeout / non-zero return / missing binary / any
    exception → ``("", "unknown")``; the summary must never crash or hang
    on one bad file.
    """
    resolution = ""
    audio = "unknown"
    if probe_resolution:
        cmd = [
            ffprobe_bin,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0",
            str(mp4_path),
        ]
        try:
            res = runner(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        except (FileNotFoundError, subprocess.SubprocessError, OSError):
            return "", "unknown"
        if res.returncode == 0:
            resolution = _parse_ffprobe_resolution(getattr(res, "stdout", "") or "")
    if probe_audio:
        audio_cmd = [
            ffprobe_bin,
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "csv=p=0",
            str(mp4_path),
        ]
        try:
            ares = runner(audio_cmd, capture_output=True, text=True, timeout=timeout, check=False)
        except (FileNotFoundError, subprocess.SubprocessError, OSError):
            return resolution, "unknown"
        if ares.returncode == 0:
            audio = _parse_ffprobe_audio(getattr(ares, "stdout", "") or "")
    return resolution, audio


def _parse_ffprobe_resolution(stdout: str) -> str:
    """``ffprobe -show_entries stream=width,height`` → ``"WxH"`` or ``""``.

    Output is ``csv=p=0`` so a video stream line is ``W,H``. Take the first
    parseable numeric pair; anything else (empty, a single token, text) → ``""``.
    """
    for line in stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            w, h = int(parts[0]), int(parts[1])
            if w and h:
                return f"{w}x{h}"
    return ""


def _parse_ffprobe_audio(stdout: str) -> str:
    """``ffprobe -select_streams a`` stdout → ``"yes"`` if any audio stream, else ``"no"``.

    With ``-select_streams a`` ffprobe prints nothing when there is no audio
    stream (and a codec name / ``audio`` when there is). Any non-empty line
    ⇒ audio present; empty ⇒ silent.
    """
    return "yes" if stdout.strip() else "no"


def scan_artefacts(
    video_dir: Path | None = None,
    runner: Any = subprocess.run,
    ffprobe_bin: str = FFPROBE_BIN,
    ffprobe_timeout: float = FFPROBE_TIMEOUT,
) -> list[VideoArtefact]:
    """Scan ``video/*.mp4`` newest-first, attaching sidecar QA verdict/resolution.

    The sidecar JSON lives beside the mp4 with the same stem
    (e.g. ``scene_v1.mp4`` ↔ ``scene_v1.json``). Fields consumed are the
    same shape the pipeline sidecar writer emits
    (``qa.verdict``, ``immersive.eye_resolution`` or ``spatial_metadata``).

    When a sidecar is missing — or a specific field is empty — **resolution
    and audio presence fall back to a direct ``ffprobe`` probe** of the mp4
    (issue #211). Those two facts are container-level and need no sidecar.
    ``QA 判定`` stays ``unknown`` without a sidecar: probing it would mean
    running ``vr180_qa.py``, which is too slow for a nightly digest.

    The ``runner`` is injectable (defaults to ``subprocess.run``) so tests
    never spawn a real ffprobe binary; it mirrors the ``_gh`` runner pattern.
    """
    root = video_dir or VIDEO_DIR
    if not root.is_dir():
        return []

    videos = sorted(root.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    out: list[VideoArtefact] = []
    for mp4 in videos:
        sidecar = root / f"{mp4.stem}.json"
        verdict = "unknown"
        resolution = ""
        audio = "unknown"
        has_sidecar = sidecar.is_file()
        if has_sidecar:
            try:
                data = json.loads(sidecar.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                data = {}
            qa = data.get("qa") or {}
            verdict = qa.get("verdict") or "unknown"
            audio = _audio_status(data, qa)
            resolution = _resolution_from_sidecar(data)
        # Fallback: probe the mp4 itself for whatever the sidecar couldn't
        # answer (no sidecar, or a field came back empty/unknown). Only the
        # missing field is probed so a partial sidecar pays for one probe.
        need_res = not resolution
        need_audio = audio == "unknown"
        if need_res or need_audio:
            probe_res, probe_audio = _probe_with_ffprobe(
                mp4,
                runner=runner,
                ffprobe_bin=ffprobe_bin,
                timeout=ffprobe_timeout,
                probe_resolution=need_res,
                probe_audio=need_audio,
            )
            if need_res and probe_res:
                resolution = probe_res
            if need_audio and probe_audio != "unknown":
                audio = probe_audio
        out.append(
            VideoArtefact(
                filename=mp4.name,
                mtime=mp4.stat().st_mtime,
                resolution=resolution,
                has_audio=audio,
                qa_verdict=verdict,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Label parsing helpers
# ---------------------------------------------------------------------------


def _label_names(labels: Any) -> list[str]:
    """Normalize gh's label payload (list of objects or list of strings)."""
    if not labels:
        return []
    if isinstance(labels, str):
        return labels.split(",")
    if not isinstance(labels, list):
        return []
    names: list[str] = []
    for lbl in labels:
        if isinstance(lbl, str):
            names.append(lbl)
        elif isinstance(lbl, dict):
            name = lbl.get("name")
            if isinstance(name, str):
                names.append(name)
    return names


def _pick_stage(labels: list[str]) -> str:
    matches = [lbl for lbl in labels if lbl.startswith("stage:")]
    return matches[-1] if matches else "unclassified"


def _pick_priority(labels: list[str]) -> str:
    matches = [lbl for lbl in labels if lbl.startswith("prio:")]
    return matches[-1].removeprefix("prio:") if matches else "P2"


def _issue_sort_key(issue: OpenIssue) -> tuple[int, int, int]:
    prio_key = issue.priority.removeprefix("prio:")
    return (STAGE_ORDER.get(issue.stage, 9), PRIO_ORDER.get(prio_key, 9), -issue.number)


def _audio_status(data: dict, qa: dict) -> str:
    """Derive audio presence from the sidecar qa block.

    Real sidecars (pipeline.sidecar / scripts.vr180_qa) carry qa.checks as a
    dict keyed by check name, e.g.

        {"qa": {"checks": {"audio stream": {"status": "pass", "detail": "..."}}}}

    ``status == "pass"`` => audio present. Any non-pass status (most commonly
    ``"warn"`` with a "no audio stream" detail) means silent. Missing /
    non-dict fields degrade gracefully to "unknown" rather than raising.
    """
    checks = qa.get("checks") if isinstance(qa, dict) else None
    if not isinstance(checks, dict):
        return "unknown"
    audio = checks.get("audio stream")
    if not isinstance(audio, dict):
        return "unknown"
    status = audio.get("status")
    if status == "pass":
        return "yes"
    if isinstance(status, str):
        return "no"
    return "unknown"


def _resolution_from_sidecar(data: dict) -> str:
    """Best-effort resolution string from immersive/spatial_metadata blocks."""
    eye = _get_nested(data, ["immersive", "eye_resolution"])
    if isinstance(eye, (list, tuple)) and len(eye) >= 2:
        try:
            w, h = int(eye[0]), int(eye[1])
            if w and h:
                return f"{w}×{h}"
        except (TypeError, ValueError):
            pass
    spatial = data.get("spatial_metadata") or {}
    if isinstance(spatial, dict):
        w = spatial.get("width")
        h = spatial.get("height")
        try:
            if w and h:
                return f"{int(w)}×{int(h)}"
        except (TypeError, ValueError):
            pass
    return ""


def _get_nested(data: dict, path: list[str]) -> Any:
    cur: Any = data
    for key in path:
        if isinstance(cur, dict):
            cur = cur.get(key)
        else:
            return None
    return cur


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def render_markdown(data: HandoffData) -> str:
    """Render the five-section handoff digest.

    Sections:
        1. 本轮合并
        2. 当前在跑 / 排队
        3. 待 owner 决策/操作
        4. 可验收产物
        5. 已知风险/未完成项
    """
    lines: list[str] = []
    lines.append(f"# 交付摘要 ({data.since or '最近 12 小时'})")
    lines.append("")

    lines.append("## ① 本轮合并")
    if data.merged_prs:
        for pr in data.merged_prs:
            when = _format_time(pr.merged_at)
            lines.append(f"- **#{pr.number}** {pr.title} — *{when}*")
    else:
        lines.append("- _无合并 PR_")
    lines.append("")

    lines.append("## ② 当前在跑 / 排队")
    active = [i for i in data.open_issues if not i.needs_owner]
    if active:
        for issue in active:
            lines.append(
                f"- **#{issue.number}** {issue.title} — stage:{issue.stage.split(':', 1)[-1]} / {issue.priority}"
            )
    else:
        lines.append("- _无在跑/排队卡_")
    lines.append("")

    lines.append("## ③ 待 owner 决策/操作 ⚠️")
    owner_items = [i for i in data.open_issues if i.needs_owner]
    if owner_items:
        for issue in owner_items:
            flags = sorted(set(issue.labels) & OWNER_ACTION_LABELS)
            lines.append(f"- **#{issue.number}** {issue.title} — 需要: {', '.join(flags)}")
    else:
        lines.append("- _无待 owner 决策项_")
    lines.append("")

    lines.append("## ④ 可验收产物")
    lines.append("_分辨率/音轨在缺 sidecar 时由 ffprobe 直接探测；QA 判定仍需 sidecar_")
    if data.artefacts:
        for art in data.artefacts:
            res = art.resolution or "—"
            audio = {"yes": "有音轨", "no": "无音轨", "unknown": "音轨未知"}.get(art.has_audio, art.has_audio)
            lines.append(f"- `{art.filename}` — {res} / {audio} / QA: {art.qa_verdict}")
    else:
        lines.append("- _无成片_")
    lines.append("")

    lines.append("## ⑤ 已知风险/未完成项")
    risks = [i for i in data.open_issues if i.priority == "P0"]
    if risks:
        for issue in risks:
            lines.append(f"- **#{issue.number}** {issue.title} — {issue.stage}")
    else:
        lines.append("- _无 P0 风险项_")
    lines.append("")

    return "\n".join(lines)


def render_json(data: HandoffData) -> str:
    return json.dumps(asdict(data), ensure_ascii=False, indent=2, default=str)


# ---------------------------------------------------------------------------
# Worklog append (idempotent)
# ---------------------------------------------------------------------------


_WORKLOG_HEADER_RE = re.compile(r"^##\s+交付摘要\s*\(\s*(\d{4}-\d{2}-\d{2})\s*\)\s*$", re.MULTILINE)


def _worklog_heading(since: str) -> str:
    day = _format_date(since) or datetime.now(timezone.utc).date().isoformat()
    return f"## 交付摘要 ({day})"


def _anchor(worklog: str, heading: str) -> int | None:
    for m in _WORKLOG_HEADER_RE.finditer(worklog):
        if f"## 交付摘要 ({m.group(1)})" == heading:
            return m.start()
    return None


def append_to_worklog(markdown: str, since: str, worklog_path: Path | None = None) -> tuple[bool, str]:
    """Append *markdown* under a date-anchored heading in ``WORKLOG.md``.

    Idempotent: if the heading for the given day already exists and its block
    already contains the digest, this is a no-op. Returns
    ``(appended: bool, message: str)``.
    """
    path = worklog_path or WORKLOG_PATH
    heading = _worklog_heading(since)
    fence = "\n\n<!-- handoff: begin -->\n" + markdown.strip() + "\n<!-- handoff: end -->\n"

    if path.is_file():
        existing = path.read_text(encoding="utf-8")
        pos = _anchor(existing, heading)
        if pos is not None:
            # Heading exists -- leave it alone (idempotent).
            return False, f"skipped: {heading} already present in {path.name}"
        with path.open("a", encoding="utf-8") as fh:
            fh.write("\n" + heading + fence)
        return True, f"appended {heading} to {path.name}"

    # No existing worklog -- create it.
    with path.open("w", encoding="utf-8") as fh:
        fh.write("# WORKLOG\n\n" + heading + fence)
    return True, f"created {path.name} with {heading}"


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------


def _format_time(iso: str) -> str:
    """Render an ISO timestamp as a short local-ish string."""
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError):
        return iso or "—"


def _format_date(since: str) -> str:
    if not since:
        return ""
    try:
        return datetime.fromisoformat(since.replace("Z", "+00:00")).date().isoformat()
    except (TypeError, ValueError):
        return since[:10] if len(since) >= 10 else ""


# ---------------------------------------------------------------------------
# Top-level build
# ---------------------------------------------------------------------------


def build_handoff(
    since_hours: int | None = None,
    since: str = "",
    gh_bin: str = "gh",
    runner: Any = subprocess.run,
    video_dir: Path | None = None,
) -> HandoffData:
    """Assemble all data sources into a :class:`HandoffData` payload."""
    if since:
        try:
            since_dt = datetime.fromisoformat(since.replace("Z", "+00:00")).astimezone(timezone.utc)
        except (TypeError, ValueError):
            since_dt = datetime.now(timezone.utc) - timedelta(hours=DEFAULT_SINCE_HOURS)
    else:
        hours = since_hours if since_hours is not None else DEFAULT_SINCE_HOURS
        since_dt = datetime.now(timezone.utc) - timedelta(hours=hours)

    return HandoffData(
        since=f"{since_dt.isoformat()} .. now",
        merged_prs=fetch_merged_prs(since_dt, gh_bin=gh_bin, runner=runner),
        open_issues=fetch_open_issues(gh_bin=gh_bin, runner=runner),
        artefacts=scan_artefacts(video_dir, runner=runner),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="K-6: WORKLOG handoff summary generator.")
    parser.add_argument(
        "--since",
        default="",
        help="ISO-8601 timestamp; only PRs merged after this are listed (default: last 12h).",
    )
    parser.add_argument(
        "--since-hours",
        type=int,
        default=None,
        help="Time window in hours (default: 12). Ignored when --since is set.",
    )
    parser.add_argument("--json", action="store_true", help="Emit structured JSON instead of Markdown.")
    parser.add_argument(
        "--append-worklog",
        action="store_true",
        help="Append the Markdown digest under a date-anchored heading in WORKLOG.md.",
    )
    parser.add_argument("--gh", default="gh", help="Path to gh binary (default: gh).")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data = build_handoff(since=args.since, since_hours=args.since_hours, gh_bin=args.gh)

    if args.json:
        print(render_json(data))
        return 0

    markdown = render_markdown(data)
    print(markdown)

    if args.append_worklog:
        _, msg = append_to_worklog(markdown, args.since or data.since)
        print(f"# worklog: {msg}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
