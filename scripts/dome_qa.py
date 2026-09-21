#!/usr/bin/env python3
"""Fulldome (domemaster) output QA validator — machine acceptance before delivery.

VR180 has ``scripts/vr180_qa.py``; dome output had no automatic gate, so a
release whose content only filled r≈0.6 slipped through on eyeballing alone.
This script closes that hole for the Route-1 fisheye domemaster produced by
``pipeline/fulldome_mapper.py``: a square canvas holding an inscribed image
circle on pure black, mono (no stereo metadata).

Checks (each PASS/FAIL + measured value, worst frame wins over the sample):
  1. resolution — frame is square and exactly ``--size``² (default 4096²)
  2. circular mask — pixels outside the inscribed circle average < ``--mask-thresh``
     (black), pixels inside average above it (non-black)
  3. coverage radius — annulus-by-annulus "has content" scan (ring std / mean
     local-std above ``--content-thresh``); the outermost content ring is
     reported as r/R and must be >= ``--min-coverage`` (default 0.9)
  4. zenith orientation — centre-disc vs bottom (forward-horizon) brightness /
     content is REPORTED ONLY, never FAILs
  5. mono metadata — the file must NOT contain st3d/sv3d boxes (dome is mono);
     stream info comes from ``ffprobe`` (subprocess list form, JSON parsed)
  6. frame sampling — ``--frames`` frames (default 5) spread across the file;
     every per-frame metric aggregates the worst value

Usage:
    python scripts/dome_qa.py dome.mp4
    python scripts/dome_qa.py dome.mp4 --json
    python scripts/dome_qa.py dome.mp4 --size 2048 --frames 3

Exit codes:
    0 — all checks pass
    1 — at least one check FAILED

The --json output carries a top-level ``summary`` object with
{pass, warn, fail, overall} so callers need not parse the human text.
Read-only: the input file is never modified.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path

# Let this script run directly (``python scripts/dome_qa.py``) without the
# caller having to set PYTHONPATH — same bootstrap as scripts/vr180_qa.py.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from pipeline.spherical_injector import _find_box_recursive  # noqa: E402

DEFAULT_SIZE = 4096
DEFAULT_FRAMES = 5
DEFAULT_MIN_COVERAGE = 0.9
DEFAULT_MASK_THRESH = 8.0
DEFAULT_CONTENT_THRESH = 5.0

# Annulus resolution of the coverage scan (1/25 R = 0.04 R per ring).
NBINS = 25
# Frames are downscaled to this max dimension before analysis — coverage r/R
# and mask means are scale-invariant, full-4K analysis only costs RAM.
ANALYZE_MAX_DIM = 1024


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
    coverage_r: float = 0.0
    outside_mean: float = 0.0
    inside_mean: float = 0.0
    zenith_center: float = 0.0
    zenith_bottom: float = 0.0
    frames_sampled: int = 0

    @property
    def failed(self) -> bool:
        return any(c.status == "fail" for c in self.checks)

    @property
    def warned(self) -> bool:
        return any(c.status == "warn" for c in self.checks)

    @property
    def summary(self) -> dict:
        counts = {"pass": 0, "warn": 0, "fail": 0}
        for check in self.checks:
            counts[check.status] = counts.get(check.status, 0) + 1
        overall = "fail" if self.failed else ("warn" if self.warned else "pass")
        return {"pass": counts["pass"], "warn": counts["warn"], "fail": counts["fail"], "overall": overall}


def _probe(path: str, ffprobe: str = "ffprobe") -> dict:
    """Read container/stream metadata via ffprobe (JSON, list-form argv)."""
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


def _scan_stereo_boxes(path: str) -> list[str]:
    """Return which of sv3d/st3d ISOBMFF boxes are present in *path*.

    Uses pipeline.spherical_injector's recursive box finder (the same
    primitive scripts/vr180_qa.py uses); unstructured bytes fall back to a
    raw substring search so a FAIL is never silently missed.
    """
    data = bytes(Path(path).read_bytes())
    found: list[str] = []
    for name in ("sv3d", "st3d"):
        box = name.encode("ascii")
        try:
            hit = _find_box_recursive(bytearray(data), box, 0, len(data)) != -1
        except Exception:
            hit = box in data
        if hit:
            found.append(name)
    return found


def _gray(frame: np.ndarray) -> np.ndarray:
    """BGR/RGB/gray frame -> float32 gray image."""
    arr = np.asarray(frame, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    return arr


def _radial_map(h: int, w: int) -> np.ndarray:
    """Per-pixel r/R from the frame centre (R = half the side)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return np.sqrt((xx - (w - 1) / 2.0) ** 2 + (yy - (h - 1) / 2.0) ** 2) / (min(h, w) / 2.0)


def analyze_frame(
    frame: np.ndarray,
    mask_thresh: float = DEFAULT_MASK_THRESH,
    content_thresh: float = DEFAULT_CONTENT_THRESH,
) -> dict:
    """Pure-numpy per-frame analysis (no I/O — unit-testable without ffmpeg).

    Returns dict with outside_mean / inside_mean (mask means), coverage_r
    (outermost content ring as r/R), zenith_center / zenith_bottom (mean
    texture energy of centre disc vs bottom patch), plus the ring table.
    """
    gray = _gray(frame)
    h, w = gray.shape
    rr = _radial_map(h, w)
    inside = rr <= 1.0

    outside_mean = float(gray[~inside].mean()) if (~inside).any() else 0.0
    inside_mean = float(gray[inside].mean()) if inside.any() else 0.0

    # Local texture energy: |gray - blurred| isolates edges/texture while a
    # flat pedestal (e.g. uniform gray fill) scores ~0.
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=3.0)
    energy = np.abs(gray - blur)

    ring_std = np.zeros(NBINS, dtype=np.float64)
    ring_energy = np.zeros(NBINS, dtype=np.float64)
    for i in range(NBINS):
        lo, hi = i / NBINS, (i + 1) / NBINS
        m = (rr > lo) & (rr <= hi) & inside
        if m.any():
            ring_std[i] = float(gray[m].std())
            ring_energy[i] = float(energy[m].mean())

    content = (ring_std > content_thresh) | (ring_energy > content_thresh)
    outer = int(np.nonzero(content)[0].max()) if content.any() else -1
    coverage_r = (outer + 1) / NBINS if outer >= 0 else 0.0

    center = rr <= 0.25
    zenith_center = float(energy[center].mean()) if center.any() else 0.0
    fwd = (rr <= 1.0) & (np.mgrid[0:h, 0:w][0].astype(np.float32) > h * 0.72)
    zenith_bottom = float(energy[fwd].mean()) if fwd.any() else 0.0

    return {
        "outside_mean": outside_mean,
        "inside_mean": inside_mean,
        "coverage_r": coverage_r,
        "zenith_center": zenith_center,
        "zenith_bottom": zenith_bottom,
        "ring_std": ring_std.tolist(),
        "ring_energy": ring_energy.tolist(),
    }


def _downscale(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    m = max(h, w)
    if m <= ANALYZE_MAX_DIM:
        return frame
    scale = ANALYZE_MAX_DIM / float(m)
    return cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)


def sample_frames(
    path: str,
    n_frames: int = DEFAULT_FRAMES,
    size: int = DEFAULT_SIZE,
    ffmpeg: str = "ffmpeg",
) -> list[np.ndarray]:
    """Extract *n_frames* evenly spread frames as BGR numpy arrays.

    Frame indices come from the ffprobe duration/fps when available, else
    fall back to wall-spread timestamps; every subprocess call is list-form.
    """
    try:
        probe = _probe(path, ffprobe="ffprobe")
        stream = _video_stream(probe)
        fps = _parse_fps(stream) or 30.0
        duration = float(probe.get("format", {}).get("duration", 0.0) or 0.0)
    except Exception:
        fps, duration = 30.0, 0.0
    total = max(1, round(fps * duration)) if duration > 0 else n_frames * 10
    idxs = sorted({min(total - 1, round((i + 0.5) * total / n_frames)) for i in range(max(1, n_frames))})
    frames: list[np.ndarray] = []
    for idx in idxs:
        t = idx / fps
        cmd = [
            ffmpeg,
            "-v",
            "error",
            "-ss",
            f"{t:.3f}",
            "-i",
            path,
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{size}x{size}",
            "pipe:1",
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        if result.returncode != 0 or not result.stdout:
            continue
        raw = np.frombuffer(result.stdout, dtype=np.uint8)
        expect = size * size * 3
        if raw.size < expect:
            continue
        frames.append(raw[:expect].reshape(size, size, 3))
    return frames


def run_qa(
    path: str,
    size: int = DEFAULT_SIZE,
    n_frames: int = DEFAULT_FRAMES,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    mask_thresh: float = DEFAULT_MASK_THRESH,
    content_thresh: float = DEFAULT_CONTENT_THRESH,
    ffprobe: str = "ffprobe",
    ffmpeg: str = "ffmpeg",
    frames: list[np.ndarray] | None = None,
    probe_data: dict | None = None,
    stereo_boxes: list[str] | None = None,
) -> QAReport:
    """Run all dome QA checks against *path*. Read-only.

    ``frames`` / ``probe_data`` / ``stereo_boxes`` are injectable seams so
    CI without ffmpeg can test the full gate on synthetic numpy frames;
    when omitted the file is probed/sampled for real.
    """
    report = QAReport(path=path)

    if not Path(path).is_file() and frames is None:
        report.checks.append(Check("input file", "fail", f"file not found: {path}"))
        report.verdict = "invalid input"
        return report

    try:
        probe = probe_data if probe_data is not None else _probe(path, ffprobe=ffprobe)
        stream = _video_stream(probe)
        fmt = probe.get("format", {})
        report.width = int(stream.get("width", 0))
        report.height = int(stream.get("height", 0))
        report.fps = _parse_fps(stream)
        report.codec = stream.get("codec_name", "unknown")
        bitrate = stream.get("bit_rate") or fmt.get("bit_rate") or 0
        report.bitrate_kbps = int(bitrate) / 1000.0
        report.duration_s = float(fmt.get("duration", 0.0) or 0.0)
    except (RuntimeError, ValueError, json.JSONDecodeError) as exc:
        report.checks.append(Check("ffprobe metadata", "fail", str(exc)))
        report.verdict = "invalid input"
        return report

    report.checks.append(
        Check(
            "stream info",
            "pass",
            f"{report.width}x{report.height} {report.codec} "
            f"{report.fps:.3g} fps, {report.bitrate_kbps:.0f} kbps, {report.duration_s:.1f}s",
        )
    )

    # ── 1. resolution ─────────────────────────────────────────────────────
    if report.width == size and report.height == size:
        report.checks.append(Check("resolution", "pass", f"{report.width}x{report.height} == {size}x{size}"))
    elif report.width == report.height and report.width:
        report.checks.append(
            Check("resolution", "fail", f"{report.width}x{report.height}: square but not {size}x{size}")
        )
    else:
        report.checks.append(
            Check("resolution", "fail", f"{report.width}x{report.height}: not square, expected {size}x{size}")
        )

    # ── frames (worst-of-sample aggregation) ──────────────────────────────
    sampled = frames if frames is not None else sample_frames(path, n_frames=n_frames, size=size, ffmpeg=ffmpeg)
    if not sampled:
        report.checks.append(Check("frame sampling", "fail", "no frames could be extracted"))
        report.verdict = "invalid input"
        return report
    report.frames_sampled = len(sampled)
    report.checks.append(Check("frame sampling", "pass", f"{len(sampled)} frame(s) sampled"))

    per = [analyze_frame(_downscale(f), mask_thresh=mask_thresh, content_thresh=content_thresh) for f in sampled]
    report.coverage_r = min(p["coverage_r"] for p in per)
    report.outside_mean = max(p["outside_mean"] for p in per)
    report.inside_mean = min(p["inside_mean"] for p in per)
    report.zenith_center = sum(p["zenith_center"] for p in per) / len(per)
    report.zenith_bottom = sum(p["zenith_bottom"] for p in per) / len(per)

    # ── 2. circular mask ──────────────────────────────────────────────────
    if report.outside_mean < mask_thresh:
        report.checks.append(
            Check("circular mask (outside black)", "pass", f"outside mean {report.outside_mean:.2f} < {mask_thresh:g}")
        )
    else:
        report.checks.append(
            Check(
                "circular mask (outside black)",
                "fail",
                f"outside mean {report.outside_mean:.2f} >= {mask_thresh:g} (corners not black)",
            )
        )
    if report.inside_mean > mask_thresh:
        report.checks.append(Check("circular mask (inside non-black)", "pass", f"inside mean {report.inside_mean:.2f}"))
    else:
        report.checks.append(
            Check("circular mask (inside non-black)", "fail", f"inside mean {report.inside_mean:.2f} (circle black)")
        )

    # ── 3. coverage radius ────────────────────────────────────────────────
    if report.coverage_r >= min_coverage:
        report.checks.append(Check("coverage radius", "pass", f"r/R={report.coverage_r:.2f} >= {min_coverage:g}"))
    else:
        report.checks.append(
            Check(
                "coverage radius",
                "fail",
                f"r/R={report.coverage_r:.2f} < {min_coverage:g} (content missing near rim)",
            )
        )

    # ── 4. zenith (report only, never FAIL) ─────────────────────────────────
    report.checks.append(
        Check(
            "zenith orientation",
            "pass",
            f"centre energy {report.zenith_center:.2f} vs bottom {report.zenith_bottom:.2f} (info only)",
        )
    )

    # ── 5. mono metadata (no stereo boxes) ────────────────────────────────
    try:
        boxes = stereo_boxes if stereo_boxes is not None else _scan_stereo_boxes(path)
    except OSError as exc:
        report.checks.append(Check("mono metadata", "fail", f"box scan failed: {exc}"))
        report.verdict = "invalid input"
        return report
    if boxes:
        report.checks.append(
            Check("mono metadata", "fail", f"stereo boxes present ({', '.join(boxes)}): dome must be mono")
        )
    else:
        report.checks.append(Check("mono metadata", "pass", "no st3d/sv3d — mono domemaster"))

    report.verdict = "domemaster" if not report.failed else "not domemaster"
    return report


_ICON = {"pass": "✅", "warn": "⚠️", "fail": "❌"}


def format_human(report: QAReport) -> str:
    """Render the human-readable per-check report."""
    lines = [f"Dome QA — {report.path}", "=" * 60]
    for check in report.checks:
        lines.append(f"{_ICON[check.status]} {check.name}: {check.detail}")
    lines.append("=" * 60)
    lines.append(f"Verdict: {report.verdict}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fulldome domemaster QA validator.")
    parser.add_argument("video", help="Path to the video file to validate (read-only)")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON report")
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE, help="Expected square size (default 4096)")
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES, help="Frames to sample (default 5)")
    parser.add_argument("--min-coverage", type=float, default=DEFAULT_MIN_COVERAGE, help="Min coverage r/R (0.9)")
    parser.add_argument("--mask-thresh", type=float, default=DEFAULT_MASK_THRESH, help="Outside black threshold")
    parser.add_argument("--content-thresh", type=float, default=DEFAULT_CONTENT_THRESH, help="Ring content threshold")
    parser.add_argument("--ffprobe", default="ffprobe", help="Path to ffprobe binary")
    parser.add_argument("--ffmpeg", default="ffmpeg", help="Path to ffmpeg binary")
    args = parser.parse_args(argv)

    report = run_qa(
        args.video,
        size=args.size,
        n_frames=args.frames,
        min_coverage=args.min_coverage,
        mask_thresh=args.mask_thresh,
        content_thresh=args.content_thresh,
        ffprobe=args.ffprobe,
        ffmpeg=args.ffmpeg,
    )

    payload = asdict(report)
    payload["summary"] = report.summary

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(format_human(report))

    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
