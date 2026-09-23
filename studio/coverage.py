"""Domemaster coverage measurement — same geometry as web/dome-preview.

Assumptions (verified on the bad master that only reached r≈0.6):
  inscribed circle of the square frame = 180° hemisphere,
  centre = zenith, rim = horizon,
  normalised radius r maps linearly to zenith angle: theta = r * 90°.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

SIZE = 256
BINS = 128
SOLID_FILL = 0.98
SOFT_FILL = 0.5
OUTER_LO = 0.80
OUTER_HI = 0.98
DEFAULT_LUMA_THRESHOLD = 4.0


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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def analyze_frame(frame_bgr: np.ndarray, luma_threshold: float = DEFAULT_LUMA_THRESHOLD) -> CoverageStats:
    """Measure how far content reaches on a domemaster frame (BGR ndarray)."""
    import cv2

    bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2BGR) if frame_bgr.ndim == 2 else frame_bgr
    small = cv2.resize(bgr, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float64)
    lum = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    has = (lum >= luma_threshold).astype(np.float64)

    ys = ((np.arange(SIZE) + 0.5) / SIZE) * 2 - 1
    xs = ((np.arange(SIZE) + 0.5) / SIZE) * 2 - 1
    dx, dy = np.meshgrid(xs, ys)
    rr = np.sqrt(dx * dx + dy * dy)
    inside = rr <= 1.0

    total = np.zeros(BINS, dtype=np.float64)
    content = np.zeros(BINS, dtype=np.float64)
    bins = np.minimum((rr * BINS).astype(int), BINS - 1)
    for b in range(BINS):
        mask = inside & (bins == b)
        total[b] = float(mask.sum())
        content[b] = float(has[mask].sum())

    fill = np.where(total > 0, content / np.maximum(total, 1), 0.0)
    total_in = float(inside.sum())
    content_in = float(has[inside].sum())
    outer = inside & (rr >= OUTER_LO) & (rr <= OUTER_HI)
    outer_total = float(outer.sum())
    outer_content = float(has[outer].sum())

    def edge_at(threshold: float) -> float:
        for k in range(BINS - 1, 0, -1):
            if fill[k] >= threshold and fill[k - 1] >= threshold:
                return (k + 1) / BINS
        return 0.0

    coverage_radius = edge_at(SOLID_FILL)
    soft_radius = max(coverage_radius, edge_at(SOFT_FILL))
    coverage_deg = coverage_radius * 90.0
    theta = math.radians(coverage_deg)
    outer_fill = (outer_content / outer_total) if outer_total > 0 else 0.0
    content_frac = (content_in / total_in) if total_in > 0 else 0.0

    if coverage_deg >= 85:
        level, text = "ok", "合规：内容铺满整圆，覆盖到地平线附近，可直接出片。"
    elif coverage_deg >= 75:
        level, text = "warn", "轻微留白：外圈最后几度没有内容，观众抬头看边缘会见到黑边。"
    else:
        level = "bad"
        text = (
            f"不合规：实心内容只铺到天顶角 {coverage_deg:.1f}°"
            f"（r={coverage_radius:.2f}），外圈 0.80–0.98 只有 {outer_fill * 100:.1f}% 有内容。"
            "出片前必须重做投影映射或扩幅，否则球幕四周是黑的。"
        )

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
    )


def analyze_media(
    path: str | Path, *, frame_index: int = 0, luma_threshold: float = DEFAULT_LUMA_THRESHOLD
) -> CoverageStats:
    """Analyze a still image or a sampled video frame."""
    import cv2

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"media not found: {path}")
    suffix = path.suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"failed to decode image: {path}")
        return analyze_frame(frame, luma_threshold=luma_threshold)

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
    return analyze_frame(frame, luma_threshold=luma_threshold)
