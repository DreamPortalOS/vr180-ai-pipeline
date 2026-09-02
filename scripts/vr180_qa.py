#!/usr/bin/env python3
"""VR180 output QA validator — machine acceptance check before delivery.

Catches the failures Quest field-testing surfaced *before* a file goes out:
recognized as 360 2D instead of 180 3D, wrong stereo layout, sub-standard
per-eye resolution.

Checks:
  1. sv3d / st3d ISOBMFF boxes present (reuses pipeline.spherical_injector
     box-scanning primitives — no mp4 box parsing is re-implemented here)
  2. Stereo mode is left-right SBS (st3d mode byte == 2)
  3. Frame layout is SBS with square eyes (width == 2 * height)
  4. Resolution / fps / bitrate report; per-eye width < 2880 px → WARN
     (V-1 standard tier is 2880×2880 per eye)

Verdicts:
  VR180 (180° 3D SBS)  — sv3d+st3d present, SBS stereo, square-eye layout
  fulldome domemaster  — square fisheye frame, no stereo metadata (not HMD)
  plain 2D             — everything else

Usage:
    python scripts/vr180_qa.py video.mp4
    python scripts/vr180_qa.py video.mp4 --json

Exit code is non-zero if any check fails (❌). Read-only: the input file is
never modified.
"""

from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path

# K-15 (#205): let this script run directly (``python scripts/vr180_qa.py``)
# without the caller having to set PYTHONPATH — put the repo root on sys.path
# before importing the ``pipeline`` package.  Idempotent (no duplicate entries)
# and a no-op when PYTHONPATH already points here.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipeline.spherical_injector import _STEREO_LEFT_RIGHT, _find_box_recursive  # noqa: E402

# V-1 standard tier: 2880×2880 per eye (5760×2880 SBS frame).
MIN_PER_EYE_WIDTH = 2880

VERDICT_VR180 = "VR180 (180° 3D SBS)"
VERDICT_FULLDOME = "fulldome domemaster（方形鱼眼、无立体元数据——非头显用）"
VERDICT_PLAIN_2D = "plain 2D"

# mp4 box types we look for, scanned recursively through moov/trak/mdia/minf/stbl
_SCAN_BOXES = (b"sv3d", b"st3d")


@dataclass
class Check:
    name: str
    status: str  # "pass" | "warn" | "fail"
    detail: str


@dataclass
class QAReport:
    path: str
    verdict: str = ""
    checks: list[Check] = field(default_factory=list)
    width: int = 0
    height: int = 0
    fps: float = 0.0
    bitrate_kbps: float = 0.0
    duration_s: float = 0.0
    codec: str = ""
    # H-1: audio passthrough (issue #73). Empty string means "no audio stream".
    audio_codec: str = ""
    audio_bitrate_kbps: float = 0.0

    @property
    def failed(self) -> bool:
        return any(c.status == "fail" for c in self.checks)


def _probe(path: str, ffprobe: str = "ffprobe") -> dict:
    """Read container/stream metadata via ffprobe (JSON)."""
    cmd = [
        ffprobe,
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed (exit {result.returncode}): {result.stderr.strip()}")
    return json.loads(result.stdout)


def _video_stream(probe: dict) -> dict:
    for stream in probe.get("streams", []):
        if stream.get("codec_type") == "video":
            return stream
    raise ValueError("no video stream found")


def _parse_fps(stream: dict) -> float:
    rate = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/1"
    try:
        frac = Fraction(rate)
    except (ValueError, ZeroDivisionError):
        return 0.0
    return float(frac) if frac.denominator else 0.0


def _extract_stream_info(probe: dict, report: QAReport) -> None:
    stream = _video_stream(probe)
    fmt = probe.get("format", {})
    report.width = int(stream.get("width", 0))
    report.height = int(stream.get("height", 0))
    report.fps = _parse_fps(stream)
    report.codec = stream.get("codec_name", "unknown")
    bitrate = stream.get("bit_rate") or fmt.get("bit_rate") or 0
    report.bitrate_kbps = int(bitrate) / 1000.0
    report.duration_s = float(fmt.get("duration", 0.0) or 0.0)
    _extract_audio_info(probe, report)


def _extract_audio_info(probe: dict, report: QAReport) -> None:
    """Populate audio_codec / audio_bitrate_kbps from the first audio stream.

    H-1: the QA report surfaces audio presence so operators can tell a silent
    VR180 from one with a track at a glance. Absence is a warning, not a fail.
    """
    for stream in probe.get("streams", []):
        if stream.get("codec_type") == "audio":
            report.audio_codec = stream.get("codec_name", "unknown")
            abr = stream.get("bit_rate") or 0
            report.audio_bitrate_kbps = int(abr) / 1000.0
            return


def _audio_check(report: QAReport) -> Check:
    """Build the audio QA check. Present → pass with codec/bitrate; absent → warn."""
    if report.audio_codec:
        return Check(
            "audio stream",
            "pass",
            f"{report.audio_codec} @ {report.audio_bitrate_kbps:.0f} kbps",
        )
    return Check(
        "audio stream",
        "warn",
        "no audio stream — video will be silent (consider --copy-audio-from or a provider with AAC)",
    )


def _scan_boxes(path: str) -> dict[str, dict]:
    """Scan for sv3d/st3d boxes using spherical_injector's recursive finder.

    Returns {box_type: {"offset": int, "stereo_mode": int|None}}.
    """
    found: dict[str, dict] = {}
    data = bytearray(Path(path).read_bytes())
    for box_type in _SCAN_BOXES:
        offset = _find_box_recursive(data, box_type, 0, len(data))
        if offset == -1:
            continue
        entry: dict = {"offset": offset, "stereo_mode": None}
        if box_type == b"st3d":
            # st3d = size(4) + type(4) + version_flags(4) + stereo_mode(1)
            size = struct.unpack(">I", data[offset : offset + 4])[0]
            if size >= 13 and offset + 13 <= len(data):
                entry["stereo_mode"] = data[offset + 12]
        found[box_type.decode("ascii")] = entry
    return found


def run_qa(path: str, ffprobe: str = "ffprobe") -> QAReport:
    """Run all QA checks against *path* and return a report. Read-only."""
    report = QAReport(path=path)

    if not Path(path).is_file():
        report.checks.append(Check("input file", "fail", f"file not found: {path}"))
        report.verdict = VERDICT_PLAIN_2D
        return report

    # ── stream metadata (ffprobe) ─────────────────────────────────────────
    try:
        probe = _probe(path, ffprobe=ffprobe)
        _extract_stream_info(probe, report)
    except (RuntimeError, ValueError, json.JSONDecodeError) as exc:
        report.checks.append(Check("ffprobe metadata", "fail", str(exc)))
        report.verdict = VERDICT_PLAIN_2D
        return report

    report.checks.append(
        Check(
            "stream info",
            "pass",
            f"{report.width}×{report.height} {report.codec} "
            f"{report.fps:.3g} fps, {report.bitrate_kbps:.0f} kbps, {report.duration_s:.1f}s",
        )
    )

    # H-1: audio presence (warn, never fail).
    report.checks.append(_audio_check(report))

    # ── ISOBMFF boxes ─────────────────────────────────────────────────────
    boxes = _scan_boxes(path)
    has_sv3d = "sv3d" in boxes
    has_st3d = "st3d" in boxes

    if has_sv3d and has_st3d:
        report.checks.append(Check("sv3d/st3d boxes", "pass", "sv3d + st3d present"))
    elif has_sv3d or has_st3d:
        missing = "st3d" if has_sv3d else "sv3d"
        report.checks.append(Check("sv3d/st3d boxes", "fail", f"incomplete: missing {missing} box"))
    else:
        report.checks.append(Check("sv3d/st3d boxes", "fail", "no spherical metadata boxes found"))

    st3d_mode = boxes.get("st3d", {}).get("stereo_mode")
    if has_st3d:
        if st3d_mode == _STEREO_LEFT_RIGHT:
            report.checks.append(Check("stereo mode", "pass", "left-right SBS"))
        else:
            report.checks.append(Check("stereo mode", "fail", f"st3d mode={st3d_mode}, expected left-right (2)"))
    else:
        report.checks.append(Check("stereo mode", "fail", "no st3d box — cannot confirm SBS"))

    # ── frame layout ──────────────────────────────────────────────────────
    w, h = report.width, report.height
    if w and h:
        if w == 2 * h:
            report.checks.append(Check("SBS square-eye layout", "pass", f"{w}×{h} = 2×({h}×{h})"))
        elif w == h:
            report.checks.append(
                Check("SBS square-eye layout", "warn", f"square frame {w}×{h} — domemaster-like, not SBS")
            )
        else:
            report.checks.append(
                Check("SBS square-eye layout", "fail", f"{w}×{h}: width ≠ 2×height, not per-eye-square SBS")
            )

        per_eye = w // 2 if w == 2 * h else w
        if per_eye < MIN_PER_EYE_WIDTH:
            report.checks.append(
                Check(
                    "per-eye resolution",
                    "warn",
                    f"per-eye width {per_eye}px < {MIN_PER_EYE_WIDTH}px (V-1 standard tier)",
                )
            )
        else:
            report.checks.append(Check("per-eye resolution", "pass", f"per-eye width {per_eye}px"))

    # ── verdict ───────────────────────────────────────────────────────────
    sbs_layout = bool(w and h and w == 2 * h)
    if has_sv3d and has_st3d and st3d_mode == _STEREO_LEFT_RIGHT and sbs_layout:
        report.verdict = VERDICT_VR180
    elif w and h and w == h and not (has_sv3d and has_st3d):
        report.verdict = VERDICT_FULLDOME
    else:
        report.verdict = VERDICT_PLAIN_2D

    return report


_ICON = {"pass": "✅", "warn": "⚠️", "fail": "❌"}


def format_human(report: QAReport) -> str:
    """Render the human-readable per-check report."""
    lines = [f"VR180 QA — {report.path}", "=" * 60]
    for check in report.checks:
        lines.append(f"{_ICON[check.status]} {check.name}: {check.detail}")
    lines.append("=" * 60)
    lines.append(f"Verdict: {report.verdict}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="VR180 output QA validator — machine acceptance check before delivery."
    )
    parser.add_argument("video", help="Path to the video file to validate (read-only)")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON report")
    parser.add_argument("--ffprobe", default="ffprobe", help="Path to ffprobe binary")
    args = parser.parse_args(argv)

    report = run_qa(args.video, ffprobe=args.ffprobe)

    if args.json:
        print(json.dumps(asdict(report), ensure_ascii=False, indent=2))
    else:
        print(format_human(report))

    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
