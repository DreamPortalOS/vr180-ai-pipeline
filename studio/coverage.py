"""Domemaster coverage measurement — the single shared algorithm.

This module owns the **one** coverage-radius algorithm used by both the Studio
``qa.dome_coverage`` node (``studio/nodes/convert.py``) and the release
acceptance gate ``scripts/dome_qa.py``. Issue #401 unified the two: previously
the Studio node judged pass/fail with a permissive luma threshold while
``dome_qa.py`` used a stricter full-ring-fill scan, so the two could disagree
on the same master (the Studio node reported ``level="bad"`` yet
``passed=True``).

Geometry (same as web/dome-preview and ``scripts/dome_qa.py``):

  inscribed circle of the square frame = 180° hemisphere,
  centre = zenith, rim = horizon,
  normalised radius r maps linearly to zenith angle: theta = r * 90°.

Coverage is a *full-circumference* measure: a ring (annulus) passes only when
at least ``RING_FILL`` of its pixels are content (bright OR textured), so a disc
that merely fills one side of every ring — the VR180→dome half-dome defect —
cannot push the reported radius out to the rim. This is the algorithm tuned on
the real 4K master ``video/dome_v10_4096.mp4`` (content ends ~r/R 0.66); both
consumers analyse at the same ``ANALYZE_MAX_DIM`` downscale so they read the
same number on the same frame.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

# ── shared algorithm constants ───────────────────────────────────────────────
# Annulus resolution of the coverage scan (1/50 R = 0.02 R per ring).
NBINS = 50
# A ring passes the coverage scan only when this fraction of its pixels are
# content (full-circumference fill), not merely "any azimuth has content".
RING_FILL = 0.9
# Per-pixel content floor: a pixel is "content" when its gray value OR its
# local texture energy exceeds this. 16 matches the brightness floor used to
# derive the gate on a real 4K domemaster (content ends ~r/R 0.65).
CONTENT_THRESH = 16.0
# Frames are downscaled to this max dimension before analysis — coverage r/R
# and mask means are scale-invariant, full-4K analysis only costs RAM.
ANALYZE_MAX_DIM = 1024

# Soft-edge threshold (a ring "softly covered" at half-fill) for the secondary
# soft_radius reading the UI exposes alongside the solid coverage radius.
SOFT_FILL = 0.5
# Outer annulus over which the "outer fill" fraction is reported (r in [LO, HI]).
OUTER_LO = 0.80
OUTER_HI = 0.98
# Level classification by solid coverage zenith angle (degrees).
OK_DEG = 85.0
WARN_DEG = 75.0

# Kept for backward compatibility with callers that imported the old name.
DEFAULT_LUMA_THRESHOLD = CONTENT_THRESH


@dataclass(frozen=True)
class CoverageStats:
    coverage_radius: float
    coverage_deg: float
    soft_radius: float
    soft_deg: float
    solid_angle_frac: float
    outer_fill: float
    content_frac: float
    level: str
    text: str
    # Mask means + per-ring profile — shared with scripts/dome_qa.py so its
    # circular-mask / ring-profile checks use the same scan, not a second one.
    outside_mean: float
    inside_mean: float
    ring_profile: list[dict]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _gray(frame: np.ndarray) -> np.ndarray:
    """BGR/RGB/gray frame -> float32 gray image (channel mean)."""
    arr = np.asarray(frame, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    return arr


def _radial_map(h: int, w: int) -> np.ndarray:
    """Per-pixel r/R from the frame centre (R = half the short side)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return np.sqrt((xx - (w - 1) / 2.0) ** 2 + (yy - (h - 1) / 2.0) ** 2) / (min(h, w) / 2.0)


def _downscale(frame: np.ndarray) -> np.ndarray:
    """Downscale so the longest side is at most ``ANALYZE_MAX_DIM`` (area interp)."""
    h, w = frame.shape[:2]
    m = max(h, w)
    if m <= ANALYZE_MAX_DIM:
        return frame
    scale = ANALYZE_MAX_DIM / float(m)
    import cv2

    return cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)


def analyze_frame(
    frame_bgr: np.ndarray,
    *,
    content_thresh: float = CONTENT_THRESH,
    ring_fill: float = RING_FILL,
) -> CoverageStats:
    """Measure how far content reaches on a domemaster frame (BGR/RGB/gray ndarray).

    This is the single shared scan. A ring passes only when at least
    ``ring_fill`` of its pixels are content (bright OR textured) — a
    full-circumference measure, so a half-dome disc collapses to its short
    side rather than reading as fully covered.
    """
    import cv2

    gray = _gray(frame_bgr)
    h, w = gray.shape
    rr = _radial_map(h, w)
    inside = rr <= 1.0

    outside_mean = float(gray[~inside].mean()) if (~inside).any() else 0.0
    inside_mean = float(gray[inside].mean()) if inside.any() else 0.0

    # Local texture energy: |gray - blurred| isolates edges/texture while a
    # flat pedestal (e.g. uniform gray fill) scores ~0.
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=3.0)
    energy = np.abs(gray - blur)

    # Per-pixel "has content": bright OR textured. A ring is judged by how
    # much of its circumference is content (fill fraction), not by whether any
    # single azimuth happens to be bright — the half-dome regression this gate
    # exists to catch.
    has_content = (gray > content_thresh) | (energy > content_thresh)

    ring_fill_frac = np.zeros(NBINS, dtype=np.float64)
    ring_mean = np.zeros(NBINS, dtype=np.float64)
    for i in range(NBINS):
        lo, hi = i / NBINS, (i + 1) / NBINS
        m = (rr > lo) & (rr <= hi) & inside
        if m.any():
            ring_fill_frac[i] = float(has_content[m].mean())
            ring_mean[i] = float(gray[m].mean())

    passing = ring_fill_frac >= ring_fill
    outer = int(np.nonzero(passing)[0].max()) if passing.any() else -1
    coverage_radius = (outer + 1) / NBINS if outer >= 0 else 0.0

    # Soft radius: outermost ring at least half-covered (the "almost reached"
    # edge the UI draws alongside the solid coverage edge).
    soft_passing = ring_fill_frac >= SOFT_FILL
    soft_outer = int(np.nonzero(soft_passing)[0].max()) if soft_passing.any() else -1
    soft_radius = max(coverage_radius, (soft_outer + 1) / NBINS if soft_outer >= 0 else 0.0)

    coverage_deg = coverage_radius * 90.0
    theta = math.radians(coverage_deg)

    # Outer annulus fill fraction (r in [OUTER_LO, OUTER_HI]) and whole-disc
    # content fraction — the human-readable "外圈 …% 有内容" figure.
    outer_annulus = inside & (rr >= OUTER_LO) & (rr <= OUTER_HI)
    outer_total = float(outer_annulus.sum())
    outer_fill = float(has_content[outer_annulus].sum()) / outer_total if outer_total > 0 else 0.0
    total_in = float(inside.sum())
    content_frac = float(has_content[inside].sum()) / total_in if total_in > 0 else 0.0

    if coverage_deg >= OK_DEG:
        level, text = "ok", "合规：内容铺满整圆，覆盖到地平线附近，可直接出片。"
    elif coverage_deg >= WARN_DEG:
        level, text = "warn", "轻微留白：外圈最后几度没有内容，观众抬头看边缘会见到黑边。"
    else:
        level = "bad"
        text = (
            f"不合规：实心内容只铺到天顶角 {coverage_deg:.1f}°"
            f"（r={coverage_radius:.2f}），外圈 {OUTER_LO:.2f}–{OUTER_HI:.2f} 只有 {outer_fill * 100:.1f}% 有内容。"
            "出片前必须重做投影映射或扩幅，否则球幕四周是黑的。"
        )

    ring_profile = [
        {"r": (i + 1) / NBINS, "fill": float(ring_fill_frac[i]), "mean": float(ring_mean[i])} for i in range(NBINS)
    ]

    return CoverageStats(
        coverage_radius=coverage_radius,
        coverage_deg=coverage_deg,
        soft_radius=soft_radius,
        soft_deg=soft_radius * 90.0,
        solid_angle_frac=1.0 - math.cos(theta),
        outer_fill=outer_fill,
        content_frac=content_frac,
        level=level,
        text=text,
        outside_mean=outside_mean,
        inside_mean=inside_mean,
        ring_profile=ring_profile,
    )


def analyze_media(
    path: str | Path,
    *,
    frame_index: int = 0,
    content_thresh: float = CONTENT_THRESH,
    ring_fill: float = RING_FILL,
) -> CoverageStats:
    """Analyze a still image or a sampled video frame using the shared scan."""
    import cv2

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"media not found: {path}")
    suffix = path.suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"failed to decode image: {path}")
        return analyze_frame(_downscale(frame), content_thresh=content_thresh, ring_fill=ring_fill)

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_index))
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"failed to read frame {frame_index} from {path}")
    return analyze_frame(_downscale(frame), content_thresh=content_thresh, ring_fill=ring_fill)
