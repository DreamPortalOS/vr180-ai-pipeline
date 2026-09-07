#!/usr/bin/env python3
"""Estimate a source clip's horizontal FOV from how straight its lines unwarp.

Why
---
``run_pipeline.py --src-hfov`` decides how wide the source content is spread
over the sphere.  Nobody has ever *measured* it: a generation prompt that says
"ultra wide / 16mm / fisheye" is a style hint, not a lens spec, so today the
owner eyeballs a number per clip.  Ten degrees of error is not a cosmetic white
edge — it is angular misregistration (turn your head 50° and the object is no
longer at 50°).  This script turns that guess into a measurement.

Principle (zero new dependencies — ffmpeg ``v360`` + OpenCV only)
----------------------------------------------------------------
Straight lines in the world (horizons, tree trunks, building edges) stay
straight under a *rectilinear* rendering and bow under a wide, non-rectilinear
one.  So: push each frame through a sphere round trip at every candidate hfov —

    1. ``v360=input=fisheye:ih_fov=C:iv_fov=<equidistant vfov(C)>:output=hequirect``
    2. ``v360=input=hequirect:output=flat:h_fov=<window>``

— and score how straight the result is.  Only the *correct* C unbends the
picture; too small pincushions it, too large barrels it.  Lowest score wins.

A note on step 1's ``input=fisheye`` (this deliberately departs from the
issue-card text, which wrote ``input=flat``)
............................................
``flat`` is ffmpeg's *pinhole* model, and pinhole → sphere → pinhole composes
to a plain homography: straight lines stay straight for **every** candidate, so
the score curve is exactly flat and the method cannot work.  Verified on the
synthetic grid before this file was written.  The recoverable signal lives in
the source's *departure* from rectilinearity, so the source is modelled as an
equidistant (fisheye-like) wide rendering, which is what wide AI footage and
real wide lenses actually look like.  Everything else — the two-stage round
trip through ``hequirect``, the candidate grid, the straightness score — is as
the card specifies.

The honest corollary: a source that really is a clean pinhole render carries no
fov evidence at all.  Then the score curve *is* flat, and this script says
"置信度低" instead of inventing a number.  See ``--min-margin``.

Score
-----
Per rectified frame: Canny edges (restricted to the pixels the round trip
actually covered), then ``HoughLinesP`` for the long straight runs.  The score
is ``edge_pixels * short_side / Σ segment_length²`` — "how much edge do you
have to spend to buy a unit of squared straight run".  Lower is straighter.
Squared length is load-bearing: a bowed line still yields short chordal
segments, so *plain* segment count barely separates the candidates, while the
length² weighting rewards the one candidate that produces genuinely long runs.

Usage
-----
    python scripts/calibrate_hfov.py video/src_720p_v2.mp4
    python scripts/calibrate_hfov.py clip.mp4 --frames 8 --grid 70:140:2
    python scripts/calibrate_hfov.py clip.mp4 --json                 # stdout
    python scripts/calibrate_hfov.py clip.mp4 --json out/hfov.json --plot out/hfov.png

The input may be a video or a still image; a still is scored as a single frame.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Defaults (module constants so tests can reference them instead of literals)
# ---------------------------------------------------------------------------

#: Candidate grid, ``start:stop:step`` in degrees. 60–150 brackets everything
#: from a normal lens to an action-cam ultra-wide.
DEFAULT_GRID: str = "60:150:5"

#: Frames sampled evenly across the clip. More frames = steadier score, linear cost.
DEFAULT_FRAMES: int = 5

#: Rectilinear window rendered out of the sphere, in degrees.  Fixed across
#: candidates on purpose: the output is always ``out_fov`` degrees wide over
#: ``width`` pixels, so a residual in output pixels means the same angle for
#: every candidate and the scores stay comparable.
DEFAULT_OUT_FOV: float = 90.0

#: Frames are downscaled to this width before analysis (speed; edge geometry
#: survives fine). Sources narrower than this are left alone.
DEFAULT_WIDTH: int = 640

#: Minimum Hough segment length, as a fraction of the frame's short side.
DEFAULT_MIN_LINE_FRAC: float = 0.15

#: ``HoughLinesP``'s ``maxLineGap``, as a fraction of the minimum segment
#: length (never below :data:`HOUGH_MIN_GAP_PX`).  It has to scale with the
#: frame.  A fixed few-pixel gap makes a *straight* line fail to register
#: whenever anything interrupts its edge — a grid crossing, an occluder, a
#: texture break — and because the correct candidate is precisely the one whose
#: lines are long and therefore most interrupted, a tight gap penalises the
#: right answer hardest.  Measured on the synthetic grid: at ``maxLineGap=4``
#: the true 100° candidate scored *worst* (Hough found no segments at all at
#: 480x480 and 640x480, so the recommendation ran away to the 150° grid edge).
#: At these values 100° is recovered with a 0.44–0.68 relative margin across
#: 320x180 → 640x360 and both line densities tried.
HOUGH_GAP_FRAC: float = 0.35
HOUGH_MIN_GAP_PX: float = 8.0

#: Relative depth the winning score must have below the rest of the curve
#: before the result is called trustworthy.  See :func:`_curve_margin`.
DEFAULT_MIN_MARGIN: float = 0.15

#: Intermediate hequirect is this multiple of the analysis width, so the sphere
#: never becomes the resolution bottleneck of the round trip.
EQUIRECT_SCALE: int = 2

#: Seconds allowed for one ffmpeg call.
FFMPEG_TIMEOUT: float = 300.0

_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"})

#: Printed verbatim when the curve is too flat to trust — the card requires the
#: literal string, and the operator needs the fallback instruction with it.
LOW_CONFIDENCE_TEXT = (
    "置信度低：评分曲线过平，画面里没有足够的直线证据。"
    "不要直接采用推荐值，请人工用 v360 output=flat 目视比对几档 hfov 后再定。"
)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def pinhole_vfov(hfov_deg: float, width: int, height: int) -> float:
    """Vertical fov of a *rectilinear* frame with ``hfov_deg`` and this aspect.

    ``vfov = 2 * atan(tan(hfov / 2) * height / width)`` — the same pinhole
    relation :meth:`pipeline.equirectangular_mapper.EquirectangularMapper._calc_vertical_fov`
    uses, kept here so this script stays standalone.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"width/height must be positive, got {width}x{height}")
    hfov_rad = math.radians(hfov_deg)
    return math.degrees(2.0 * math.atan(math.tan(hfov_rad / 2.0) * height / width))


def equidistant_vfov(hfov_deg: float, width: int, height: int) -> float:
    """Vertical fov of an *equidistant* (fisheye) frame with ``hfov_deg``.

    Equidistant means radius is proportional to angle, so the angular scale is
    isotropic and the vertical fov is just the aspect-scaled horizontal one —
    no ``tan`` anywhere, unlike :func:`pinhole_vfov`.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"width/height must be positive, got {width}x{height}")
    return hfov_deg * height / width


def parse_grid(spec: str) -> list[float]:
    """Parse ``"start:stop:step"`` into an inclusive list of candidate hfovs."""
    parts = spec.split(":")
    if len(parts) != 3:
        raise ValueError(f"--grid must be start:stop:step, got {spec!r}")
    try:
        start, stop, step = (float(p) for p in parts)
    except ValueError as exc:
        raise ValueError(f"--grid values must be numbers, got {spec!r}") from exc
    if step <= 0:
        raise ValueError(f"--grid step must be > 0, got {step}")
    if stop < start:
        raise ValueError(f"--grid stop must be >= start, got {spec!r}")
    if start <= 0.0 or stop >= 180.0:
        raise ValueError(f"--grid must stay inside (0, 180) degrees, got {spec!r}")
    n = math.floor((stop - start) / step + 1e-9)
    return [round(start + i * step, 6) for i in range(n + 1)]


# ---------------------------------------------------------------------------
# ffmpeg plumbing — always list form, never shell=True
# ---------------------------------------------------------------------------


def _even(value: float) -> int:
    """Round to a positive even int (ffmpeg's rawvideo/scale are happier)."""
    return max(2, round(value / 2.0) * 2)


def run_v360(
    frames: Sequence[np.ndarray],
    vfilter: str,
    out_width: int,
    out_height: int,
    ffmpeg: str = "ffmpeg",
    timeout: float = FFMPEG_TIMEOUT,
) -> list[np.ndarray]:
    """Push grayscale ``frames`` through ``vfilter`` in one ffmpeg call.

    Raw ``gray`` in, raw ``gray`` out over pipes — a third of the bytes of
    rgb24, and edge geometry is all this script looks at.  One subprocess for
    the whole batch, not one per frame.
    """
    if not frames:
        return []
    in_h, in_w = frames[0].shape[:2]
    for f in frames:
        if f.shape[:2] != (in_h, in_w) or f.ndim != 2 or f.dtype != np.uint8:
            raise ValueError("run_v360 expects a batch of same-sized uint8 grayscale frames")
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostats",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-s",
        f"{in_w}x{in_h}",
        "-i",
        "pipe:0",
        "-vf",
        vfilter,
        "-threads",
        "1",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "pipe:1",
    ]
    payload = b"".join(np.ascontiguousarray(f).tobytes() for f in frames)
    proc = subprocess.run(cmd, input=payload, capture_output=True, timeout=timeout, check=False)
    if proc.returncode != 0:
        tail = proc.stderr.decode("utf-8", "replace").strip().splitlines()[-5:]
        raise RuntimeError("ffmpeg v360 failed: " + " | ".join(tail))
    stride = out_width * out_height
    expected = stride * len(frames)
    if len(proc.stdout) < expected:
        raise RuntimeError(f"ffmpeg v360 returned {len(proc.stdout)} bytes, expected {expected}")
    return [
        np.frombuffer(proc.stdout[i * stride : (i + 1) * stride], dtype=np.uint8).reshape(out_height, out_width)
        for i in range(len(frames))
    ]


def build_roundtrip_filter(
    hfov_deg: float,
    width: int,
    height: int,
    out_fov: float = DEFAULT_OUT_FOV,
    equirect_scale: int = EQUIRECT_SCALE,
) -> str:
    """The two-stage v360 chain for one candidate hfov.

    Stage 1 lifts the frame onto the sphere *believing* it is an equidistant
    rendering ``hfov_deg`` wide; stage 2 renders a rectilinear window back out.
    Composition is a genuine warp (not a homography), so straightness of the
    result is evidence about ``hfov_deg``.
    """
    equi = _even(width * equirect_scale)
    return (
        f"v360=input=fisheye:ih_fov={hfov_deg:.4f}:iv_fov={equidistant_vfov(hfov_deg, width, height):.4f}:"
        f"output=hequirect:h_fov=180:v_fov=180:w={equi}:h={equi},"
        f"v360=input=hequirect:ih_fov=180:iv_fov=180:output=flat:"
        f"h_fov={out_fov:.4f}:v_fov={pinhole_vfov(out_fov, width, height):.4f}:w={width}:h={height}"
    )


class V360Rectifier:
    """Runs the round trip for one candidate and reports which pixels are real.

    The rectified frame has regions the source never covered (v360 fills them
    with a constant).  Those borders are strong, *curved* edges that would
    corrupt the score, so they are excluded via a coverage mask measured — not
    guessed — by sending an all-black and an all-white probe frame through the
    very same filter chain in the very same ffmpeg call: a pixel the source
    reaches tracks its probe, a pixel it does not is identical in both.
    """

    def __init__(
        self,
        out_fov: float = DEFAULT_OUT_FOV,
        equirect_scale: int = EQUIRECT_SCALE,
        ffmpeg: str = "ffmpeg",
        erode_px: int = 11,
    ) -> None:
        self.out_fov = out_fov
        self.equirect_scale = equirect_scale
        self.ffmpeg = ffmpeg
        self.erode_px = erode_px

    def __call__(self, frames: Sequence[np.ndarray], hfov_deg: float) -> tuple[np.ndarray, list[np.ndarray]]:
        h, w = frames[0].shape[:2]
        vfilter = build_roundtrip_filter(hfov_deg, w, h, self.out_fov, self.equirect_scale)
        probes = [np.zeros((h, w), np.uint8), np.full((h, w), 255, np.uint8)]
        out = run_v360([*probes, *frames], vfilter, w, h, ffmpeg=self.ffmpeg)
        spread = out[1].astype(np.int16) - out[0].astype(np.int16)
        mask = (spread > 200).astype(np.uint8) * 255
        if self.erode_px > 1:
            mask = cv2.erode(mask, np.ones((self.erode_px, self.erode_px), np.uint8))
        return mask, list(out[2:])


#: A rectifier is anything that maps (frames, hfov) -> (coverage mask, frames).
Rectifier = Callable[[Sequence[np.ndarray], float], tuple[np.ndarray, list[np.ndarray]]]


# ---------------------------------------------------------------------------
# Straightness score
# ---------------------------------------------------------------------------


def edge_map(gray: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Canny edges, blurred first, clipped to the covered region."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150, L2gradient=True)
    if mask is not None:
        edges = cv2.bitwise_and(edges, mask)
    return edges


def hough_gap(min_length: float) -> float:
    """``maxLineGap`` for a run of ``min_length`` px. See :data:`HOUGH_GAP_FRAC`."""
    return max(HOUGH_MIN_GAP_PX, min_length * HOUGH_GAP_FRAC)


def straight_segments(edges: np.ndarray, min_length: float) -> np.ndarray:
    """Lengths of the Hough segments at least ``min_length`` px long."""
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=max(10, int(min_length)),
        minLineLength=float(min_length),
        maxLineGap=hough_gap(min_length),
    )
    if lines is None:
        return np.zeros(0, dtype=np.float64)
    seg = lines[:, 0, :].astype(np.float64)
    return np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1])


def straightness_score(
    gray: np.ndarray,
    mask: np.ndarray | None = None,
    min_line_frac: float = DEFAULT_MIN_LINE_FRAC,
) -> tuple[float | None, int, int]:
    """Score one rectified frame. Returns ``(score, segment_count, edge_pixels)``.

    ``score = edge_pixels * short_side / Σ length²``; lower means straighter.
    ``None`` when the frame carries no usable line evidence at all — a blank or
    featureless frame — which the caller reports as low confidence rather than
    as a flattering zero.

    Noise is a *different* failure and is deliberately not caught here: Hough
    will happily chain random edges into long "segments", so a noise frame does
    get a score, and often a flattering one.  What it cannot do is get a
    *different* score at different candidates — the round trip rearranges noise
    into more noise.  So noise is rejected one level up, by the flat-curve
    margin rule in :func:`_curve_margin`, not by this function.
    """
    h, w = gray.shape[:2]
    edges = edge_map(gray, mask)
    edge_pixels = int(np.count_nonzero(edges))
    if edge_pixels < 200:
        return None, 0, edge_pixels
    short_side = min(w, h)
    lengths = straight_segments(edges, min_line_frac * short_side)
    energy = float((lengths**2).sum())
    if energy <= 0.0:
        return None, 0, edge_pixels
    return edge_pixels * short_side / energy, int(lengths.size), edge_pixels


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateScore:
    """One row of the score table."""

    hfov: float
    score: float | None
    segments: int
    edge_pixels: int
    frames_scored: int

    def to_dict(self) -> dict:
        return {
            "hfov": self.hfov,
            "score": None if self.score is None else round(self.score, 6),
            "segments": self.segments,
            "edge_pixels": self.edge_pixels,
            "frames_scored": self.frames_scored,
        }


@dataclass
class CalibrationResult:
    """Everything the CLI prints, serialises or plots."""

    recommended_hfov: float | None
    confidence: str
    margin: float
    scores: list[CandidateScore]
    runners_up: list[float] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    source: str = ""
    frames: int = 0
    analysis_size: tuple[int, int] = (0, 0)
    out_fov: float = DEFAULT_OUT_FOV
    min_margin: float = DEFAULT_MIN_MARGIN

    @property
    def is_confident(self) -> bool:
        return self.confidence == "high"

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "frames": self.frames,
            "analysis_size": list(self.analysis_size),
            "out_fov": self.out_fov,
            "recommended_hfov": self.recommended_hfov,
            "confidence": self.confidence,
            "margin": round(self.margin, 6),
            "min_margin": self.min_margin,
            "runners_up": self.runners_up,
            "scores": [s.to_dict() for s in self.scores],
            "notes": self.notes,
        }


def _curve_margin(scores: Sequence[CandidateScore], best: CandidateScore) -> float:
    """How convincingly ``best`` beats the rest of the curve, in [0, 1].

    The minimum of ``two`` readings, so a curve has to be convincing both ways:

    * **runner-up gap** — the card's "最低与次低差", but skipping the winner's
      immediate grid neighbours: they sample the *same* basin, so on a sharp
      minimum with a fine grid they are close by construction and would fake
      flatness.
    * **depth** — how far the winner sits below the curve's median.  A curve
      with one lucky dip and no overall shape scores badly here.

    Both are relative, so the units of the score never matter.
    """
    valid = [s for s in scores if s.score is not None]
    if len(valid) < 3 or best.score is None or best.score <= 0.0:
        return 0.0
    step = _grid_step(scores)
    outside = [s.score for s in valid if abs(s.hfov - best.hfov) > step * 1.5]
    runner = min(outside) if outside else min(s.score for s in valid if s is not best)
    gap = 0.0 if runner <= 0.0 else (runner - best.score) / runner
    median = float(np.median([s.score for s in valid]))
    depth = 0.0 if median <= 0.0 else (median - best.score) / median
    return max(0.0, min(gap, depth))


def _grid_step(scores: Sequence[CandidateScore]) -> float:
    if len(scores) < 2:
        return 0.0
    return min(abs(b.hfov - a.hfov) for a, b in pairwise(scores)) or 0.0


def calibrate(
    frames: Sequence[np.ndarray],
    candidates: Sequence[float],
    rectifier: Rectifier | None = None,
    min_line_frac: float = DEFAULT_MIN_LINE_FRAC,
    min_margin: float = DEFAULT_MIN_MARGIN,
    progress: Callable[[float, float | None], None] | None = None,
) -> CalibrationResult:
    """Score every candidate hfov and pick the straightest.

    ``rectifier`` is injectable so the scoring/summary logic is testable
    without ffmpeg; it defaults to the real :class:`V360Rectifier`.
    """
    if not frames:
        raise ValueError("no frames to calibrate on")
    if not candidates:
        raise ValueError("empty candidate grid")
    rectify = rectifier if rectifier is not None else V360Rectifier()

    rows: list[CandidateScore] = []
    for hfov in candidates:
        mask, rectified = rectify(frames, hfov)
        per_frame: list[float] = []
        segments = 0
        edge_pixels = 0
        for frame in rectified:
            score, n_seg, n_edge = straightness_score(frame, mask, min_line_frac)
            segments += n_seg
            edge_pixels += n_edge
            if score is not None:
                per_frame.append(score)
        # Median over frames: one busy or blank frame must not swing a candidate.
        agg = float(np.median(per_frame)) if per_frame else None
        rows.append(CandidateScore(hfov, agg, segments, edge_pixels, len(per_frame)))
        if progress is not None:
            progress(hfov, agg)

    valid = [r for r in rows if r.score is not None]
    notes: list[str] = []
    if not valid:
        notes.append("没有任何候选档位找到可用的直线证据（画面可能无直线、过暗或过糊）。")
        return CalibrationResult(
            recommended_hfov=None,
            confidence="low",
            margin=0.0,
            scores=rows,
            notes=notes,
            min_margin=min_margin,
        )

    ordered = sorted(valid, key=lambda r: r.score)
    best = ordered[0]
    margin = _curve_margin(rows, best)
    confidence = "high" if margin >= min_margin else "low"
    if len(valid) < len(rows):
        notes.append(f"{len(rows) - len(valid)} 个候选档位无直线证据，已跳过。")
    if best.hfov in (rows[0].hfov, rows[-1].hfov):
        notes.append(f"推荐值落在候选网格边界 {best.hfov:g}°，真实值可能在网格之外，建议用 --grid 扩大范围。")
        confidence = "low"
    return CalibrationResult(
        recommended_hfov=best.hfov,
        confidence=confidence,
        margin=margin,
        scores=rows,
        runners_up=[r.hfov for r in ordered[1:3]],
        notes=notes,
        min_margin=min_margin,
    )


# ---------------------------------------------------------------------------
# Frame loading
# ---------------------------------------------------------------------------


def _probe_video(path: Path, ffprobe: str = "ffprobe") -> tuple[int, int, float]:
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height:format=duration",
        "-of",
        "json",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=FFMPEG_TIMEOUT, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path}: {proc.stderr.decode('utf-8', 'replace').strip()[:300]}")
    info = json.loads(proc.stdout.decode("utf-8", "replace") or "{}")
    streams = info.get("streams") or []
    if not streams:
        raise RuntimeError(f"no video stream in {path}")
    width = int(streams[0]["width"])
    height = int(streams[0]["height"])
    duration = float((info.get("format") or {}).get("duration") or 0.0)
    return width, height, duration


def _grab_frame(path: Path, timestamp: float, width: int, height: int, ffmpeg: str = "ffmpeg") -> np.ndarray | None:
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostats",
        "-ss",
        f"{timestamp:.3f}",
        "-i",
        str(path),
        "-frames:v",
        "1",
        "-vf",
        f"scale={width}:{height}:flags=bicubic,format=gray",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=FFMPEG_TIMEOUT, check=False)
    if proc.returncode != 0 or len(proc.stdout) < width * height:
        return None
    return np.frombuffer(proc.stdout[: width * height], dtype=np.uint8).reshape(height, width)


def _fit_size(src_w: int, src_h: int, target_width: int) -> tuple[int, int]:
    width = _even(min(target_width, src_w))
    return width, _even(width * src_h / src_w)


def load_frames(
    path: str | Path,
    count: int = DEFAULT_FRAMES,
    target_width: int = DEFAULT_WIDTH,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
) -> list[np.ndarray]:
    """Sample ``count`` grayscale frames spread evenly over a clip (or read a still)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"source not found: {path}")
    if count < 1:
        raise ValueError(f"--frames must be >= 1, got {count}")

    if path.suffix.lower() in _IMAGE_SUFFIXES:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"could not read image {path}")
        h, w = img.shape[:2]
        width, height = _fit_size(w, h, target_width)
        if (width, height) != (w, h):
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        return [img]

    src_w, src_h, duration = _probe_video(path, ffprobe=ffprobe)
    width, height = _fit_size(src_w, src_h, target_width)
    stamps = [0.0] if duration <= 0.0 else [duration * (i + 0.5) / count for i in range(count)]
    frames = [f for f in (_grab_frame(path, t, width, height, ffmpeg=ffmpeg) for t in stamps) if f is not None]
    if not frames:
        raise RuntimeError(f"could not decode any frame from {path}")
    return frames


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def format_report(result: CalibrationResult) -> str:
    """The human-readable score table + verdict."""
    w, h = result.analysis_size
    lines = [
        f"source          : {result.source}",
        f"frames analysed : {result.frames} @ {w}x{h}",
        f"round-trip window: {result.out_fov:g}°  (v360 fisheye→hequirect→flat)",
        "",
        "  hfov      score   segments   frames",
        "  " + "-" * 36,
    ]
    best = result.recommended_hfov
    for row in result.scores:
        marker = "*" if row.hfov == best else " "
        score = "     --" if row.score is None else f"{row.score:7.3f}"
        lines.append(f" {marker}{row.hfov:6.1f}  {score}   {row.segments:8d}  {row.frames_scored:7d}")
    lines.append("")
    if best is None:
        lines.append("recommended --src-hfov : (none)")
    else:
        lines.append(f"recommended --src-hfov : {best:g}")
        if result.runners_up:
            lines.append("runners-up             : " + ", ".join(f"{v:g}°" for v in result.runners_up))
    lines.append(f"margin                 : {result.margin * 100:.1f}% (need >= {result.min_margin * 100:.1f}%)")
    lines.append(f"confidence             : {result.confidence}")
    if not result.is_confident:
        lines.append("")
        lines.append(LOW_CONFIDENCE_TEXT)
    for note in result.notes:
        lines.append(f"note: {note}")
    return "\n".join(lines)


def render_plot(result: CalibrationResult, path: str | Path, size: tuple[int, int] = (900, 480)) -> Path:
    """Draw the score curve to a PNG with OpenCV (no new plotting dependency)."""
    valid = [r for r in result.scores if r.score is not None]
    if not valid:
        raise ValueError("nothing to plot: no candidate produced a score")
    width, height = size
    pad_l, pad_r, pad_t, pad_b = 70, 20, 40, 50
    canvas = np.full((height, width, 3), 255, np.uint8)
    x0, x1 = pad_l, width - pad_r
    y0, y1 = pad_t, height - pad_b
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (200, 200, 200), 1)

    hfovs = [r.hfov for r in valid]
    vals = [r.score for r in valid]
    lo_x, hi_x = min(hfovs), max(hfovs)
    lo_y, hi_y = min(vals), max(vals)
    span_x = (hi_x - lo_x) or 1.0
    span_y = (hi_y - lo_y) or 1.0

    def to_px(hfov: float, score: float) -> tuple[int, int]:
        px = x0 + int((hfov - lo_x) / span_x * (x1 - x0))
        py = y1 - int((score - lo_y) / span_y * (y1 - y0))
        return px, py

    pts = np.array([to_px(r.hfov, r.score) for r in valid], np.int32)
    cv2.polylines(canvas, [pts], False, (180, 90, 40), 2, cv2.LINE_AA)
    for px, py in pts:
        cv2.circle(canvas, (int(px), int(py)), 3, (180, 90, 40), -1, cv2.LINE_AA)
    if result.recommended_hfov is not None:
        best = min(valid, key=lambda r: r.score)
        bx, by = to_px(best.hfov, best.score)
        cv2.circle(canvas, (bx, by), 7, (40, 40, 200), 2, cv2.LINE_AA)
        cv2.line(canvas, (bx, y0), (bx, y1), (40, 40, 200), 1, cv2.LINE_AA)

    font, fs = cv2.FONT_HERSHEY_SIMPLEX, 0.45
    cv2.putText(canvas, f"{lo_x:g}", (x0 - 10, y1 + 20), font, fs, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{hi_x:g}", (x1 - 20, y1 + 20), font, fs, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.putText(canvas, "src hfov (deg)", ((x0 + x1) // 2 - 50, y1 + 38), font, fs, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{hi_y:.2f}", (8, y0 + 6), font, fs, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{lo_y:.2f}", (8, y1), font, fs, (60, 60, 60), 1, cv2.LINE_AA)
    title = f"straightness vs hfov  ->  {result.recommended_hfov:g} deg ({result.confidence})"
    cv2.putText(canvas, title, (x0, y0 - 14), font, 0.55, (30, 30, 30), 1, cv2.LINE_AA)

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out), canvas):
        raise RuntimeError(f"could not write plot to {out}")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="calibrate_hfov",
        description="Estimate a source clip's horizontal FOV via a v360 round trip + line straightness.",
    )
    parser.add_argument("source", help="video or still image to measure")
    parser.add_argument(
        "--frames", type=int, default=DEFAULT_FRAMES, help=f"frames to sample (default {DEFAULT_FRAMES})"
    )
    parser.add_argument(
        "--grid", default=DEFAULT_GRID, help=f"candidate hfovs start:stop:step (default {DEFAULT_GRID})"
    )
    parser.add_argument(
        "--out-fov",
        type=float,
        default=DEFAULT_OUT_FOV,
        help=f"rectified window in degrees (default {DEFAULT_OUT_FOV:g})",
    )
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help=f"analysis width (default {DEFAULT_WIDTH})")
    parser.add_argument(
        "--min-line-frac",
        type=float,
        default=DEFAULT_MIN_LINE_FRAC,
        help=f"min Hough segment length as a fraction of the short side (default {DEFAULT_MIN_LINE_FRAC})",
    )
    parser.add_argument(
        "--min-margin",
        type=float,
        default=DEFAULT_MIN_MARGIN,
        help=f"relative curve depth required for high confidence (default {DEFAULT_MIN_MARGIN})",
    )
    parser.add_argument(
        "--json",
        nargs="?",
        const="-",
        metavar="PATH",
        help="emit JSON; bare --json prints to stdout, --json PATH writes a file",
    )
    parser.add_argument("--plot", metavar="PATH", help="write the score curve to a PNG")
    parser.add_argument("--quiet", action="store_true", help="suppress the per-candidate progress lines")
    parser.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg binary (default: ffmpeg)")
    parser.add_argument("--ffprobe", default="ffprobe", help="ffprobe binary (default: ffprobe)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    to_stdout = args.json == "-"
    try:
        candidates = parse_grid(args.grid)
        frames = load_frames(args.source, args.frames, args.width, ffmpeg=args.ffmpeg, ffprobe=args.ffprobe)
        h, w = frames[0].shape[:2]

        def progress(hfov: float, score: float | None) -> None:
            shown = "--" if score is None else f"{score:.3f}"
            print(f"  scanning {hfov:6.1f}° -> {shown}", file=sys.stderr)

        result = calibrate(
            frames,
            candidates,
            rectifier=V360Rectifier(out_fov=args.out_fov, ffmpeg=args.ffmpeg),
            min_line_frac=args.min_line_frac,
            min_margin=args.min_margin,
            progress=None if args.quiet else progress,
        )
        result.source = str(args.source)
        result.frames = len(frames)
        result.analysis_size = (w, h)
        result.out_fov = args.out_fov

        if args.plot:
            render_plot(result, args.plot)

        payload = json.dumps(result.to_dict(), indent=2, ensure_ascii=False)
        if to_stdout:
            print(payload)
        else:
            if args.json:
                out = Path(args.json)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(payload, encoding="utf-8")
            print(format_report(result))
        return 0
    except (ValueError, FileNotFoundError, RuntimeError, subprocess.TimeoutExpired, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
