#!/usr/bin/env python3
"""Depth temporal-stability metric — turn "is it dizzying?" into a number.

Subjective headset checks for depth jitter/flicker take hours of owner
feedback. This script measures the *frame-to-frame stability of a depth map
sequence*, which is the direct prerequisite for clean binocular fusion:
depth jitter -> each eye warps independently -> the two eyes cannot fuse.
With this, DepthCrafter (#77) vs. single-frame depth, or parameter tweaks,
can be compared quantitatively on the desktop instead of in the headset.

Three metrics (pure functions, unit-testable):
    temporal_jitter  - mean absolute difference between adjacent, normalised
                       depth frames.  Lower is more stable.
    flicker_ratio    - share of pixels whose inter-frame difference changes
                       sign (direction flip), capturing high-frequency flicker.
                       Lower is more stable.
    edge_consistency - IoU of depth edges (Sobel) between adjacent frames.
                       Higher is more stable.

Each metric is graded against an empirical threshold into a three-tier
verdict (OK / warn / fail).

All heavy work is injectable: ``depths`` can be supplied directly (or via a
fake loader) so CI runs without GPU, models, or ffmpeg.

Usage:
    python scripts/depth_stability.py --depth-npy-dir out/depths
    python scripts/depth_stability.py --depth-npy-dir out/depths --json out/report.json
    python scripts/depth_stability.py --depth-npy-dir out/depths --json out/report.json --print
    python scripts/depth_stability.py --compare A.json B.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger("depth-stability")


# ---------------------------------------------------------------------------
# Thresholds (module constants)
# ---------------------------------------------------------------------------

# Initial values — to be re-calibrated by lead against real headset data.
# temporal_jitter: normalised mean-absolute depth diff between adjacent frames.
# A fully-static sequence is 0; values above ~0.05 start to read as jittery.
JITTER_OK: float = 0.03
JITTER_WARN: float = 0.08

# flicker_ratio: fraction of pixels whose depth-diff sign flips across frames.
# 0 = no flicker; >0.20 means a large fraction of pixels are chattering.
FLICKER_OK: float = 0.05
FLICKER_WARN: float = 0.20

# edge_consistency: IoU of Sobel depth edges between adjacent frames.
# Higher is better; below ~0.6 the edge layout is drifting frame to frame.
EDGE_OK: float = 0.70
EDGE_WARN: float = 0.50


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class MetricResult:
    """One metric's computed value, three-tier verdict, and thresholds."""

    name: str
    value: float
    ok: str  # "OK" | "WARN" | "FAIL"
    thresholds: dict[str, float]
    higher_is_better: bool


@dataclass
class StabilityReport:
    """Aggregate report for one depth sequence."""

    temporal_jitter: MetricResult
    flicker_ratio: MetricResult
    edge_consistency: MetricResult
    n_frames: int

    # Composite verdict: worst tier across the three metrics.
    overall: str  # "OK" | "WARN" | "FAIL"

    def __str__(self) -> str:
        """Human-readable report (also written when --print is used)."""
        lines: list[str] = ["Depth Temporal Stability Report", "=" * 38]
        lines.append(f"Frames analysed: {self.n_frames}")
        lines.append("")
        lines.append(f"{'metric':<20} {'value':>8} {'verdict':>8}")
        lines.append("-" * 38)
        for m in (self.temporal_jitter, self.flicker_ratio, self.edge_consistency):
            icon = {"OK": "✅", "WARN": "⚠️", "FAIL": "❌"}[m.ok]
            lines.append(f"{m.name:<20} {m.value:>8.4f} {icon} {m.ok:>4}")
        lines.append("-" * 38)
        icon = {"OK": "✅", "WARN": "⚠️", "FAIL": "❌"}[self.overall]
        lines.append(f"{'OVERALL':<20} {'':>8} {icon} {self.overall:>4}")
        lines.append("")
        lines.append("Thresholds (initial values, pending lead calibration on real data):")
        lines.append(
            f"  temporal_jitter  lower is better — OK < {JITTER_OK:.2f} ; WARN < {JITTER_WARN:.2f} ; FAIL otherwise"
        )
        lines.append(
            f"  flicker_ratio    lower is better — OK < {FLICKER_OK:.2f} ; WARN < {FLICKER_WARN:.2f} ; FAIL otherwise"
        )
        lines.append(
            f"  edge_consistency higher is better — OK >= {EDGE_OK:.2f} ; WARN >= {EDGE_WARN:.2f} ; FAIL otherwise"
        )
        return "\n".join(lines)


def _to_json(obj) -> object:
    """Recursively convert dataclasses/numpy scalars to JSON-serialisable types."""
    import dataclasses

    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _to_json(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


# ---------------------------------------------------------------------------
# Pure metric functions
# ---------------------------------------------------------------------------


def _normalise(depth: np.ndarray) -> np.ndarray:
    """Normalise a single depth map to [0, 1] (or zeros if flat)."""
    depth = np.asarray(depth, dtype=np.float32)
    d_min = depth.min()
    d_max = depth.max()
    if d_max > d_min:
        return (depth - d_min) / (d_max - d_min)
    return np.zeros_like(depth)


def temporal_jitter(depths: Sequence[np.ndarray]) -> float:
    """Mean absolute difference between adjacent, normalised depth frames.

    Computed as the average (over each adjacent pair) of the per-pixel mean
    absolute difference.  Range [0, 1]; lower = more stable.  A fully-static
    sequence yields 0.
    """
    if len(depths) < 2:
        raise ValueError(f"temporal_jitter needs at least 2 frames, got {len(depths)}")
    norms = [_normalise(d) for d in depths]
    diffs = [np.abs(norms[i + 1] - norms[i]).mean() for i in range(len(norms) - 1)]
    return float(np.mean(diffs))


def flicker_ratio(depths: Sequence[np.ndarray]) -> float:
    """Fraction of pixels whose inter-frame depth-diff sign flips.

    For each adjacent triple (t-1, t, t+1) and each pixel, a flicker is a
    sign change in the first difference: sign(d_t - d_{t-1}) !=
    sign(d_{t+1} - d_t).  Returns the mean share of flickering pixels across
    all triples.  Range [0, 1]; lower = less flicker.
    """
    if len(depths) < 3:
        return 0.0
    norms = [_normalise(d) for d in depths]
    ratios: list[float] = []
    for i in range(1, len(norms) - 1):
        diff_prev = norms[i] - norms[i - 1]
        diff_next = norms[i + 1] - norms[i]
        flip = (diff_prev > 0) != (diff_next > 0)
        ratios.append(float(flip.mean()))
    return float(np.mean(ratios))


def _sobel_edges(depth: np.ndarray) -> np.ndarray:
    """Binary edge map via combined Sobel gradients.

    Returns a float32 boolean mask of the same shape as ``depth``.  No cv2
    dependency — the 3x3 Sobel kernels are applied with numpy so the module
    stays importable on headless CI without opencv.
    """
    d = np.asarray(_normalise(depth), dtype=np.float32)
    if d.shape[0] < 3 or d.shape[1] < 3:
        return np.zeros_like(d)
    gx_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    gy_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

    def _correlate(kernel: np.ndarray) -> np.ndarray:
        kh, kw = kernel.shape
        h, w = d.shape
        out = np.zeros((h - kh + 1, w - kw + 1), dtype=np.float32)
        strips = np.lib.stride_tricks.sliding_window_view(d, kernel.shape)
        out += (strips * kernel).sum(axis=(-2, -1))
        return out

    gx = np.abs(_correlate(gx_kernel))
    gy = np.abs(_correlate(gy_kernel))
    mag = np.sqrt(gx**2 + gy**2)
    threshold = mag.max() * 0.25 if mag.max() > 0 else 0.0
    return (mag > threshold).astype(np.float32)


def edge_consistency(depths: Sequence[np.ndarray]) -> float:
    """Mean IoU of depth edges between adjacent frames.

    Edges are detected with combined Sobel gradients.  Returns the average IoU
    across adjacent pairs.  Range [0, 1]; higher = more stable edge layout.
    """
    if len(depths) < 2:
        raise ValueError(f"edge_consistency needs at least 2 frames, got {len(depths)}")
    edges = [_sobel_edges(d) for d in depths]
    ious: list[float] = []
    for i in range(len(edges) - 1):
        a = edges[i]
        b = edges[i + 1]
        intersection = float(np.minimum(a, b).sum())
        union = float(np.maximum(a, b).sum())
        ious.append(intersection / union if union > 0 else 1.0)
    return float(np.mean(ious))


# ---------------------------------------------------------------------------
# Grading / report assembly
# ---------------------------------------------------------------------------


def _grade(value: float, thresholds: dict[str, float], higher_is_better: bool) -> str:
    """Three-tier verdict from a value and its {OK, WARN} thresholds."""
    if higher_is_better:
        if value >= thresholds["OK"]:
            return "OK"
        if value >= thresholds["WARN"]:
            return "WARN"
        return "FAIL"
    # lower_is_better
    if value < thresholds["OK"]:
        return "OK"
    if value < thresholds["WARN"]:
        return "WARN"
    return "FAIL"


def _tier_rank(verdict: str) -> int:
    return {"OK": 0, "WARN": 1, "FAIL": 2}[verdict]


def compute_report(
    depths: Sequence[np.ndarray],
    jitter_ok: float = JITTER_OK,
    jitter_warn: float = JITTER_WARN,
    flicker_ok: float = FLICKER_OK,
    flicker_warn: float = FLICKER_WARN,
    edge_ok: float = EDGE_OK,
    edge_warn: float = EDGE_WARN,
) -> StabilityReport:
    """Compute all three metrics and their verdicts for a depth sequence."""
    if len(depths) < 2:
        raise ValueError(f"compute_report needs at least 2 frames, got {len(depths)}")

    tj = temporal_jitter(depths)
    fr = flicker_ratio(depths)
    ec = edge_consistency(depths)

    tj_result = MetricResult(
        name="temporal_jitter",
        value=tj,
        ok=_grade(tj, {"OK": jitter_ok, "WARN": jitter_warn}, higher_is_better=False),
        thresholds={"OK": jitter_ok, "WARN": jitter_warn},
        higher_is_better=False,
    )
    fr_result = MetricResult(
        name="flicker_ratio",
        value=fr,
        ok=_grade(fr, {"OK": flicker_ok, "WARN": flicker_warn}, higher_is_better=False),
        thresholds={"OK": flicker_ok, "WARN": flicker_warn},
        higher_is_better=False,
    )
    ec_result = MetricResult(
        name="edge_consistency",
        value=ec,
        ok=_grade(ec, {"OK": edge_ok, "WARN": edge_warn}, higher_is_better=True),
        thresholds={"OK": edge_ok, "WARN": edge_warn},
        higher_is_better=True,
    )

    overall_rank = max(_tier_rank(m.ok) for m in (tj_result, fr_result, ec_result))
    overall = {0: "OK", 1: "WARN", 2: "FAIL"}[overall_rank]

    return StabilityReport(
        temporal_jitter=tj_result,
        flicker_ratio=fr_result,
        edge_consistency=ec_result,
        n_frames=len(depths),
        overall=overall,
    )


# ---------------------------------------------------------------------------
# Depth loading — injectable, so tests supply fake sequences
# ---------------------------------------------------------------------------


def load_depth_npy_dir(depth_dir: str | Path) -> list[np.ndarray]:
    """Load a sorted list of ``depth_*.npy`` frames from a directory.

    Each .npy is expected to hold a single 2-D depth map (H, W) or
    per-frame (H, W, C) that gets averaged over the last axis.
    """
    ddir = Path(depth_dir)
    files = sorted(ddir.glob("depth_*.npy"))
    if not files:
        raise FileNotFoundError(f"No depth_*.npy files in {ddir}")
    frames: list[np.ndarray] = []
    for f in files:
        arr = np.load(f)
        arr = np.atleast_2d(arr)
        if arr.ndim == 3:
            arr = arr.mean(axis=-1)
        frames.append(arr)
    return frames


def load_depth_meta(depth_dir: str | Path) -> dict | None:
    """Read the ``meta.json`` written by the pipeline's depth stage (I-6, #121).

    Returns ``None`` when no meta file exists (e.g. a pre-#121 depth cache, or
    a hand-built npy dir) — the caller then prints a 'source unknown' notice so
    an A/B comparison can never be silently mis-attributed.
    """
    meta_path = Path(depth_dir) / "meta.json"
    if not meta_path.exists():
        return None
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)


def _format_source(meta: dict | None) -> list[str]:
    """Human-readable provenance lines for the report header (I-6, #121).

    When the depth dir carries ``meta.json`` (written by run_pipeline's depth
    stage), the report leads with the model + key params so an A/B run is
    self-attributing.  Without it, an explicit '来源未知' reminder is shown so
    the operator cannot mistake one model's maps for another's.
    """
    if meta is None:
        return [
            "Source: 来源未知 (no meta.json found — depth dir may pre-date #121 "
            "or be hand-built; A/B attribution is not guaranteed)",
        ]
    model = meta.get("depth_model", "?")
    lines = [f"Source: depth_model={model}"]
    params = {k: v for k, v in meta.items() if k not in ("depth_model", "timestamp")}
    if params:
        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
        lines.append(f"Params: {param_str}")
    if meta.get("timestamp"):
        lines.append(f"Generated: {meta['timestamp']}")
    return lines


def run_depth_stage(video: str, max_frames: int | None = None) -> list[np.ndarray]:
    """Run the existing pipeline depth stage on a video and return frames.

    Heavy imports are deferred to keep this module importable without the
    full pipeline stack in pure unit tests.
    """
    from scripts import run_pipeline as rp

    args = rp.parse_args(["--input", video, "--output", "/tmp/_ds_out.mp4"])
    rp.apply_quality_preset(args)
    if max_frames is not None:
        args.max_frames = max_frames
    frames = list(rp.read_frames(video, max_frames=max_frames))
    if not frames:
        raise RuntimeError(f"No frames read from {video}")
    return rp.run_depth_stage(args, frames)


# ---------------------------------------------------------------------------
# Compare / report rendering
# ---------------------------------------------------------------------------

METRIC_ORDER: tuple[str, ...] = ("temporal_jitter", "flicker_ratio", "edge_consistency")


def _metric_val(report: dict, key: str) -> float:
    return float(report[key]["value"])


def render_compare(a_path: str | Path, b_path: str | Path) -> str:
    """Render a Markdown comparison table between two JSON reports."""
    with open(a_path, encoding="utf-8") as fa, open(b_path, encoding="utf-8") as fb:
        a = json.load(fa)
        b = json.load(fb)

    lines: list[str] = [
        f"Depth Stability Compare: `{Path(a_path).name}` vs `{Path(b_path).name}`",
        "",
        f"| metric | {Path(a_path).name} | {Path(b_path).name} | delta | winner |",
        "| --- | ---: | ---: | ---: | --- |",
    ]

    def _better(metric: str, va: float, vb: float) -> str:
        # lower_is_better for jitter/flicker, higher_is_better for edge
        lower = metric in ("temporal_jitter", "flicker_ratio")
        if lower:
            return Path(a_path).name if va < vb else (Path(b_path).name if vb < va else "tie")
        return Path(a_path).name if va > vb else (Path(b_path).name if vb > va else "tie")

    for key in METRIC_ORDER:
        va = _metric_val(a, key)
        vb = _metric_val(b, key)
        delta = vb - va
        lines.append(f"| {key} | {va:.4f} | {vb:.4f} | {delta:+.4f} | {_better(key, va, vb)} |")

    def _overall(report: dict) -> str:
        return report.get("overall", "?")

    a_ov = _overall(a)
    b_ov = _overall(b)
    winner = _better("temporal_jitter", _tier_rank(a_ov), _tier_rank(b_ov))
    lines.append(f"| OVERALL | {a_ov} | {b_ov} |  | {winner} |")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.  Accept optional *argv* for testing."""
    parser = argparse.ArgumentParser(
        description="Measure temporal stability of a depth map sequence "
        "(jitter / flicker / edge consistency) — turn 'is it dizzying?' into a number.",
    )
    parser.add_argument("--input", "-i", help="Source video; requires --depth-stage to run the depth model")
    parser.add_argument("--depth-npy-dir", help="Read depth_*.npy frames directly (preferred, no model needed)")
    parser.add_argument(
        "--depth-stage",
        action="store_true",
        help="Estimate depth from --input via the existing depth stage (loads the model)",
    )
    parser.add_argument("--max-frames", type=int, default=None, help="Analyse only the first N frames")
    parser.add_argument("--json", "-o", help="Write report as JSON to this path")
    parser.add_argument("--print", action="store_true", help="Print the human-readable report to stdout")
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("A.JSON", "B.JSON"),
        help="Print a comparison table between two JSON reports (no depth loading)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.  Returns exit code (0 = success)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args(argv)

    # --compare is a self-contained mode: reads two JSON files, prints table.
    if args.compare:
        try:
            print(render_compare(args.compare[0], args.compare[1]))
            return 0
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

    # Normal mode: load a depth sequence then compute metrics.
    if not args.depth_npy_dir and not (args.input and args.depth_stage):
        print(
            "Error: provide --depth-npy-dir <dir> or --input <video> --depth-stage",
            file=sys.stderr,
        )
        return 1

    try:
        meta: dict | None = None
        if args.depth_npy_dir:
            depths = load_depth_npy_dir(args.depth_npy_dir)
            meta = load_depth_meta(args.depth_npy_dir)
        else:
            depths = run_depth_stage(args.input, max_frames=args.max_frames)

        if args.max_frames is not None and len(depths) > args.max_frames:
            depths = depths[: args.max_frames]

        report = compute_report(depths)

        # I-6 (#121): print provenance (model + params from meta.json) at the
        # top of the report so an A/B run is self-attributing; '来源未知' when
        # there is no meta so the operator cannot silently mis-attribute.
        source_lines = _format_source(meta)
        for line in source_lines:
            print(line)
        print("")

        if args.json:
            out = Path(args.json)
            out.parent.mkdir(parents=True, exist_ok=True)
            payload = _to_json(report)
            payload = {"source": meta if meta is not None else "来源未知", **payload}
            out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            log.info("Report written to %s", out)

        if args.print:
            print(report)

        if not args.json and not args.print:
            print(report)

        return 0
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
