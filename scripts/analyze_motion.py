#!/usr/bin/env python3
"""Camera-motion diagnostic report for one clip — "how does this thing move?"

This is the operator-facing half of #331.  ``pipeline/motion.py`` measures;
this script decodes a video, runs the measurement, and turns it into a report
that answers three questions without the reader having to interpret a number:

1. **How much does the camera roll, really?**  Reported twice: the integrated
   per-frame curve *and* the long-baseline direct registration that does not
   integrate anything.  When the two disagree the report says so loudly — that
   disagreement is the whole reason this card exists.  #327 read only the
   first number, got 21.16°, and built a plan on it; the second number was
   1.37°.
2. **Is it drift or is it shake?**  The roll trajectory is split into
   frequency bands.  All four clips measured so far put >95% of their roll
   energy below 0.3 Hz, i.e. a slowly leaning horizon rather than jitter —
   which is why a 1-second smoothing window could only ever have removed ~6%
   of it.  Whatever gets built next needs to know which of the two it faces.
3. **Can any of this be trusted?**  Per-frame confidence, the share of frames
   that failed, and *why* they failed.  Frames the estimator could not measure
   are reported as unmeasured, never as zero motion.

Deliberately **not** here: any correction.  Nothing is stabilised, rotated or
re-encoded; the input video is only ever read.

Usage:
    python scripts/analyze_motion.py --video video/gen_1x1_4k_drone.mp4
    python scripts/analyze_motion.py --video clip.mp4 --json out/clip_motion.json
    python scripts/analyze_motion.py --video clip.mp4 --json          # to stdout
    python scripts/analyze_motion.py --video clip.mp4 --strict        # exit 2 on warning
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipeline.motion import (  # noqa: E402
    DEFAULT_BAND_EDGES,
    DEFAULT_BASELINE_STRIDE,
    DEFAULT_DRIFT_WARNING_DEG,
    ConfidenceGates,
    MotionTrack,
    TrackingConfig,
    band_label,
    band_rms_deg,
    dominant_band,
    estimate_motion,
)

log = logging.getLogger("analyze-motion")

# ---------------------------------------------------------------------------
# Verdict thresholds (module constants, all overridable from the CLI)
# ---------------------------------------------------------------------------

#: Working resolution, long edge, in pixels.  The measurement is a global 2-D
#: similarity fit, so 4K buys nothing but decode time; 960 is what the
#: synthetic calibration was run at.
DEFAULT_WORK_SIZE: int = 960

#: Cumulative roll peak-to-peak (degrees) below which a clip reads as steady.
#: The two direct-registration readings from real clips were 1.37° and 1.86°,
#: and the owner's dizziness complaint attaches to the latter, so the line
#: sits between them rather than at a round number pulled from nowhere.
STEADY_P2P_DEG: float = 1.5

#: Above this share of frames failing the confidence gates, no verdict is
#: issued at all — the clip is reported as unmeasured rather than steady.
MIN_RELIABLE_RATIO: float = 0.6

#: How many failed frames to list individually in the JSON before truncating.
MAX_LISTED_FAILURES: int = 40

VERDICT_STEADY = "steady"
VERDICT_LOW_FREQ_DRIFT = "low_freq_drift"
VERDICT_HIGH_FREQ_SHAKE = "high_freq_shake"
VERDICT_UNMEASURED = "unmeasured"

#: One plain-language sentence per verdict.  Written in Chinese because the
#: person who reads this report writes the cards in Chinese.
_SUMMARY_TEMPLATES = {
    VERDICT_STEADY: "该片基本平稳：累积横滚峰峰 {p2p:.2f}°，高频 RMS {high:.3f}°。",
    VERDICT_LOW_FREQ_DRIFT: (
        "该片主要是低频缓慢漂移：累积横滚峰峰 {p2p:.2f}°，"
        "低频（<{edge:g}Hz）RMS {low:.3f}° 是高频（>{high_edge:g}Hz）RMS {high:.3f}° 的 {ratio:.1f} 倍。"
    ),
    VERDICT_HIGH_FREQ_SHAKE: (
        "该片主要是高频抖动：累积横滚峰峰 {p2p:.2f}°，"
        "高频（>{high_edge:g}Hz）RMS {high:.3f}° 已达低频（<{edge:g}Hz）RMS {low:.3f}° 的 {shake_ratio:.1f} 倍。"
    ),
    VERDICT_UNMEASURED: (
        "该片的相机运动无法可信测量：仅 {reliable:.0%} 的帧通过可信度门限，"
        "低于 {min_reliable:.0%}，报告里的横滚数字不可采信。"
    ),
}


# ---------------------------------------------------------------------------
# Report container
# ---------------------------------------------------------------------------


@dataclass
class MotionReport:
    """Everything the diagnostic prints or serialises, in one place."""

    video: str
    n_frames: int
    fps: float
    frame_shape: tuple[int, int]

    net_roll_deg: float
    cum_p2p_deg: float
    per_frame_std_deg: float
    per_frame_mean_deg: float

    bands: dict[str, float]
    dominant_band: str
    low_freq_rms_deg: float
    high_freq_rms_deg: float
    band_edges: tuple[float, ...]

    reliable_ratio: float
    n_untrusted: int
    untrusted: list[dict]
    tracking: dict

    long_baseline: dict
    verdict: str
    summary: str

    warnings: list[str] = field(default_factory=list)

    @property
    def drift_warning(self) -> bool:
        """True when the integrated and long-baseline readings disagree."""
        return bool(self.long_baseline.get("drift_warning", False))


# ---------------------------------------------------------------------------
# Decoding (the only I/O in the measurement path)
# ---------------------------------------------------------------------------


def load_gray_frames(
    video: str | Path,
    *,
    work_size: int = DEFAULT_WORK_SIZE,
    max_frames: int | None = None,
) -> tuple[list[np.ndarray], float]:
    """Decode ``video`` to a list of grayscale frames, downscaled to ``work_size``.

    Only ever *reads* the file.  Frames are downscaled (never upscaled) so the
    long edge is at most ``work_size``: the estimator fits a global similarity,
    which gains nothing from 4K and costs several minutes of decode.
    """
    path = Path(video)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise OSError(f"cannot open video: {path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or float("nan")
        src_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        scale = min(1.0, work_size / max(src_w, src_h, 1))
        size = (max(1, round(src_w * scale)), max(1, round(src_h * scale)))

        frames: list[np.ndarray] = []
        while max_frames is None or len(frames) < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            if scale < 1.0:
                frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    finally:
        capture.release()

    if len(frames) < 2:
        raise OSError(f"decoded only {len(frames)} frame(s) from {path}; need at least 2")
    return frames, fps


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def _median(values: Sequence[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else float("nan")


def _long_baseline_dict(track: MotionTrack) -> dict:
    check = track.long_baseline
    return {
        "stride": check.stride,
        "threshold_deg": check.threshold_deg,
        "verdict": check.verdict,
        "drift_warning": check.drift_warning,
        "available": check.available,
        "max_abs_diff_deg": check.max_abs_diff_deg,
        "message": check.message,
        "endpoint": None if check.endpoint is None else _segment_dict(check.endpoint),
        "segments": [_segment_dict(seg) for seg in check.segments],
    }


def _segment_dict(segment) -> dict:
    return {
        "start": segment.start,
        "end": segment.end,
        "direct_roll_deg": segment.direct_roll_deg,
        "integrated_roll_deg": segment.integrated_roll_deg,
        "diff_deg": segment.diff_deg,
        "direct_trusted": segment.direct_trusted,
        "direct_n_tracked": segment.direct_n_tracked,
        "direct_inlier_ratio": segment.direct_inlier_ratio,
        "direct_reason": segment.direct_reason,
    }


def classify(
    *,
    cum_p2p_deg: float,
    low_freq_rms_deg: float,
    high_freq_rms_deg: float,
    reliable_ratio: float,
    steady_p2p_deg: float = STEADY_P2P_DEG,
    min_reliable_ratio: float = MIN_RELIABLE_RATIO,
) -> str:
    """Pick one of the four verdicts.

    Order matters: an unmeasurable clip is decided first, because "we could
    not see any roll" and "there was no roll" look identical in the numbers
    and only one of them is a finding.
    """
    if not np.isfinite(cum_p2p_deg) or reliable_ratio < min_reliable_ratio:
        return VERDICT_UNMEASURED
    if cum_p2p_deg <= steady_p2p_deg:
        return VERDICT_STEADY
    low = 0.0 if not np.isfinite(low_freq_rms_deg) else low_freq_rms_deg
    high = 0.0 if not np.isfinite(high_freq_rms_deg) else high_freq_rms_deg
    return VERDICT_LOW_FREQ_DRIFT if low >= high else VERDICT_HIGH_FREQ_SHAKE


def analyse_track(
    track: MotionTrack,
    *,
    video: str = "",
    band_edges: Sequence[float] = DEFAULT_BAND_EDGES,
    steady_p2p_deg: float = STEADY_P2P_DEG,
    min_reliable_ratio: float = MIN_RELIABLE_RATIO,
) -> MotionReport:
    """Turn a measured :class:`MotionTrack` into a report."""
    edges = tuple(float(e) for e in band_edges)
    bands = band_rms_deg(track.cumulative_roll_deg, track.fps, edges)
    low_key = band_label(None, edges[0])
    mid_high = [band_label(edges[i], edges[i + 1]) for i in range(len(edges) - 1)][1:]
    high_keys = [*mid_high, band_label(edges[-1], None)]
    low_rms = bands.get(low_key, float("nan"))
    high_vals = [bands[k] for k in high_keys if np.isfinite(bands.get(k, float("nan")))]
    high_rms = float(math.sqrt(sum(v * v for v in high_vals))) if high_vals else float("nan")

    increments = track.increments
    verdict = classify(
        cum_p2p_deg=track.cumulative_p2p_deg,
        low_freq_rms_deg=low_rms,
        high_freq_rms_deg=high_rms,
        reliable_ratio=track.reliable_ratio,
        steady_p2p_deg=steady_p2p_deg,
        min_reliable_ratio=min_reliable_ratio,
    )
    ratio_low_high = low_rms / high_rms if high_rms and np.isfinite(high_rms) and high_rms > 0 else float("inf")
    summary = _SUMMARY_TEMPLATES[verdict].format(
        p2p=track.cumulative_p2p_deg,
        low=low_rms,
        high=high_rms,
        edge=edges[0],
        high_edge=edges[1] if len(edges) > 1 else edges[0],
        ratio=ratio_low_high,
        shake_ratio=(high_rms / low_rms) if low_rms and np.isfinite(low_rms) and low_rms > 0 else float("inf"),
        reliable=track.reliable_ratio,
        min_reliable=min_reliable_ratio,
    )

    warnings: list[str] = []
    if track.long_baseline.drift_warning or not track.long_baseline.available:
        warnings.append(track.long_baseline.message)
    if track.n_untrusted:
        warnings.append(
            f"{track.n_untrusted}/{len(increments)} frame steps failed the confidence gates and "
            "are reported as unmeasured (bridged by interpolation in the cumulative curve, "
            "flagged in cumulative_filled) — they are NOT zero motion"
        )

    trusted = [m for m in increments if m.trusted]
    return MotionReport(
        video=str(video),
        n_frames=track.n_frames,
        fps=track.fps,
        frame_shape=track.frame_shape,
        net_roll_deg=track.net_roll_deg,
        cum_p2p_deg=track.cumulative_p2p_deg,
        per_frame_std_deg=track.per_frame_std_deg,
        per_frame_mean_deg=float(np.mean(track.trusted_roll_increments))
        if track.trusted_roll_increments.size
        else float("nan"),
        bands=bands,
        dominant_band=dominant_band(bands),
        low_freq_rms_deg=low_rms,
        high_freq_rms_deg=high_rms,
        band_edges=edges,
        reliable_ratio=track.reliable_ratio,
        n_untrusted=track.n_untrusted,
        untrusted=[
            {"index": m.index, "reason": m.reason, "raw_roll_deg": m.raw_roll_deg} for m in increments if not m.trusted
        ][:MAX_LISTED_FAILURES],
        tracking={
            "n_tracked_median": _median([m.n_tracked for m in trusted]),
            "fb_err_px_median": _median([m.fb_err_px for m in trusted]),
            "inlier_ratio_median": _median([m.inlier_ratio for m in trusted]),
            "resid_px_median": _median([m.resid_px for m in trusted]),
            "spread_median": _median([m.spread for m in trusted]),
        },
        long_baseline=_long_baseline_dict(track),
        verdict=verdict,
        summary=summary,
        warnings=warnings,
    )


def analyse_frames(
    frames: Sequence[np.ndarray],
    fps: float,
    *,
    video: str = "",
    stride: int = DEFAULT_BASELINE_STRIDE,
    drift_threshold_deg: float = DEFAULT_DRIFT_WARNING_DEG,
    band_edges: Sequence[float] = DEFAULT_BAND_EDGES,
    cfg: TrackingConfig | None = None,
    gates: ConfidenceGates | None = None,
    steady_p2p_deg: float = STEADY_P2P_DEG,
    min_reliable_ratio: float = MIN_RELIABLE_RATIO,
    **estimator_overrides,
) -> MotionReport:
    """Measure ``frames`` and report on them.  No I/O, so unit-testable directly."""
    track = estimate_motion(
        frames,
        fps=fps,
        cfg=cfg,
        gates=gates,
        stride=stride,
        drift_threshold_deg=drift_threshold_deg,
        **estimator_overrides,
    )
    return analyse_track(
        track,
        video=video,
        band_edges=band_edges,
        steady_p2p_deg=steady_p2p_deg,
        min_reliable_ratio=min_reliable_ratio,
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _fmt(value: float, digits: int = 3) -> str:
    return "n/a" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def report_to_dict(report: MotionReport) -> dict:
    """JSON-ready dict.  Non-finite floats become ``None`` so the JSON is valid."""

    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [clean(v) for v in obj]
        if isinstance(obj, (np.floating, float)):
            return None if not np.isfinite(obj) else float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        return obj

    return clean(
        {
            "video": report.video,
            "n_frames": report.n_frames,
            "fps": report.fps,
            "frame_shape": list(report.frame_shape),
            "roll": {
                "net_deg": report.net_roll_deg,
                "cum_p2p_deg": report.cum_p2p_deg,
                "per_frame_mean_deg": report.per_frame_mean_deg,
                "per_frame_std_deg": report.per_frame_std_deg,
            },
            "bands": {
                "edges_hz": list(report.band_edges),
                "rms_deg": report.bands,
                "dominant": report.dominant_band,
                "low_freq_rms_deg": report.low_freq_rms_deg,
                "high_freq_rms_deg": report.high_freq_rms_deg,
            },
            "confidence": {
                "reliable_ratio": report.reliable_ratio,
                "n_untrusted": report.n_untrusted,
                "untrusted": report.untrusted,
                "tracking": report.tracking,
            },
            "long_baseline": report.long_baseline,
            "verdict": report.verdict,
            "summary": report.summary,
            "warnings": report.warnings,
        }
    )


def render_report(report: MotionReport) -> str:
    """Human-readable report body."""
    lines: list[str] = ["Camera Motion Report", "=" * 62]
    lines.append(f"source          : {report.video or '<in-memory frames>'}")
    lines.append(f"frames / fps    : {report.n_frames} @ {_fmt(report.fps, 2)}")
    lines.append(f"work resolution : {report.frame_shape[1]}x{report.frame_shape[0]}")
    lines.append("")
    lines.append("Roll (deg, + = camera rolled clockwise / picture turns counter-clockwise)")
    lines.append(f"  integrated net       : {_fmt(report.net_roll_deg)}")
    lines.append(f"  integrated peak-peak : {_fmt(report.cum_p2p_deg)}")
    lines.append(f"  per-frame mean / std : {_fmt(report.per_frame_mean_deg, 4)} / {_fmt(report.per_frame_std_deg, 4)}")
    lines.append("")

    check = report.long_baseline
    endpoint = check.get("endpoint")
    lines.append(f"Long-baseline cross-check (stride {check['stride']}, threshold {check['threshold_deg']}deg)")

    def _segment_line(seg: dict, label: str) -> str:
        tail = "" if seg["direct_trusted"] else f"  ({seg['direct_reason'] or 'rejected'})"
        return (
            f"  {label:<14} direct {_fmt(seg['direct_roll_deg'])}  "
            f"integrated {_fmt(seg['integrated_roll_deg'])}  "
            f"diff {_fmt(seg['diff_deg'])}  trusted={seg['direct_trusted']}{tail}"
        )

    if endpoint:
        lines.append(_segment_line(endpoint, f"endpoint 0->{endpoint['end']}"))
    for seg in check["segments"]:
        lines.append(_segment_line(seg, f"{seg['start']}->{seg['end']}"))
    icon = {"consistent": "OK", "drift_suspected": "WARN", "unavailable": "UNKNOWN"}[check["verdict"]]
    lines.append(f"  verdict: [{icon}] {check['message']}")
    lines.append("")

    lines.append("Roll energy by frequency band (RMS deg of the cumulative curve)")
    for key, value in report.bands.items():
        marker = "  <-- dominant" if key == report.dominant_band else ""
        lines.append(f"  {key:<12} {_fmt(value, 4):>10}{marker}")
    lines.append("")

    lines.append("Per-frame confidence")
    lines.append(f"  reliable frames : {report.reliable_ratio:.1%} ({report.n_untrusted} rejected)")
    track = report.tracking
    lines.append(
        f"  medians         : n_tracked {_fmt(track['n_tracked_median'], 0)}  "
        f"fb_err {_fmt(track['fb_err_px_median'], 4)}px  "
        f"inliers {_fmt(track['inlier_ratio_median'])}  "
        f"resid {_fmt(track['resid_px_median'])}px  "
        f"spread {_fmt(track['spread_median'])}"
    )
    for failure in report.untrusted[:10]:
        lines.append(f"    step {failure['index']:>5}: {failure['reason']}")
    if report.n_untrusted > 10:
        lines.append(f"    ... {report.n_untrusted - 10} more (full list in --json)")
    lines.append("")

    for warning in report.warnings:
        lines.append(f"[WARN] {warning}")
    lines.append(f"VERDICT: {report.verdict}")
    lines.append(report.summary)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="analyze_motion.py",
        description="Measure and diagnose a clip's camera motion (read-only; corrects nothing).",
    )
    parser.add_argument("--video", required=True, help="Input video (read-only).")
    parser.add_argument(
        "--json",
        nargs="?",
        const="-",
        default=None,
        metavar="PATH",
        help="Write the report as JSON to PATH, or to stdout when given with no value.",
    )
    parser.add_argument(
        "--work-size", type=int, default=DEFAULT_WORK_SIZE, help="Long-edge working resolution in pixels."
    )
    parser.add_argument("--max-frames", type=int, default=None, help="Analyse only the first N frames.")
    parser.add_argument(
        "--stride",
        type=int,
        default=DEFAULT_BASELINE_STRIDE,
        help="Long-baseline stride in frames for the integration-free cross-check.",
    )
    parser.add_argument(
        "--drift-threshold",
        type=float,
        default=DEFAULT_DRIFT_WARNING_DEG,
        help="Degrees of integrated-vs-direct disagreement that raise the drift warning.",
    )
    parser.add_argument(
        "--steady-p2p",
        type=float,
        default=STEADY_P2P_DEG,
        help="Cumulative roll peak-to-peak (deg) below which the clip is called steady.",
    )
    parser.add_argument(
        "--min-reliable",
        type=float,
        default=MIN_RELIABLE_RATIO,
        help="Minimum share of frames passing the confidence gates before a verdict is issued.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress the human-readable report.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 2 when the cross-check warns or the clip cannot be measured.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        frames, fps = load_gray_frames(args.video, work_size=args.work_size, max_frames=args.max_frames)
    except OSError as exc:
        log.error("analyze_motion: %s", exc)
        return 1

    report = analyse_frames(
        frames,
        fps,
        video=str(args.video),
        stride=args.stride,
        drift_threshold_deg=args.drift_threshold,
        steady_p2p_deg=args.steady_p2p,
        min_reliable_ratio=args.min_reliable,
    )

    if not args.quiet:
        print(render_report(report))

    if args.json is not None:
        payload = json.dumps(report_to_dict(report), indent=2, ensure_ascii=False)
        if args.json == "-":
            print(payload)
        else:
            out_path = Path(args.json)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(payload, encoding="utf-8")
            log.info("-> %s", out_path)

    if args.strict and (report.drift_warning or report.verdict == VERDICT_UNMEASURED):
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess in tests
    raise SystemExit(main())
