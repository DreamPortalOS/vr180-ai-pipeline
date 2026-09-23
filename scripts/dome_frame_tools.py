#!/usr/bin/env python3
"""Dome frame geometry tools (D-3, issue #408): 16:9 ⇄ 1:1 + rim brightening.

Owner D-3 generates dome material with Gemini first (free tier) and keeps
Seedance for the locked final shots.  Gemini only emits **16:9 / 9:16**, while a
fulldome domemaster is a **1:1** canvas whose content lives inside the inscribed
circle (IMERSA), so a pair of conversions is needed in both directions.  A third
tool exists because Gemini's dome frames are consistently dark towards the rim —
on the S3/S4 material the outer annulus carried only 10–13 % of the centre
luminance — and a radial lift costs far less than regenerating the shot.

Subcommands
-----------
``pad169 <in.jpg|png> <out>``
    1:1 → 16:9: place the square image centred on a **black** canvas, height
    unchanged, ``width = round(height * 16 / 9)`` rounded up to even (ffmpeg
    needs even dimensions).  Written by OpenCV — PNG stays lossless, JPEG is
    lossy (q=95).

``crop11 <in.mp4|jpg|png> <out>``
    16:9 → 1:1: centred square crop (side = input height) and erase everything
    outside the inscribed circle to pure black.  Images go through OpenCV;
    videos go through **ffmpeg** (subprocess *list* form, never ``shell=True``)
    with ``crop=ih:ih`` plus a ``geq`` circle mask, and the audio track is
    stream-copied (``-c:a copy``) so the clip keeps its sound untouched.  The
    mask is exact in the frames ffmpeg builds; the H.264 4:2:0 encode
    afterwards puts a 1–3 px chroma/ringing halo on the hard circle edge (mean
    over the whole surround is still < 1.5 / 255).

``rimlift <in> <out> [--target 0.55] [--start 0.5] [--max-gain 4]``
    Radial brightness lift by radius profile: for every ring ``r >= start`` the
    mean luminance is pushed towards ``target × centre luminance`` (centre is
    measured over ``r < 0.3``), the per-ring gain is capped at ``max_gain`` and
    the gain map is Gaussian-smoothed (σ ≈ 15 px at 1024²) so ring boundaries
    cannot show up as steps.  ``r < start`` is left bit-exact untouched and
    everything outside the inscribed circle stays pure black.  Images only —
    the card does not require the video path; run it on the still frames.

Usage:
    python scripts/dome_frame_tools.py pad169 gemini_1x1.png frame_169.png
    python scripts/dome_frame_tools.py crop11 frame_169.png frame_1x1.png
    python scripts/dome_frame_tools.py crop11 gemini_169.mp4 dome_1x1.mp4
    python scripts/dome_frame_tools.py rimlift dome_1x1.png dome_lifted.png
    python scripts/dome_frame_tools.py rimlift dome.png out.png --target 0.6 --max-gain 6

Only the named output file is written; the input is never modified.  Geometry
uses the same radius convention as the fulldome acceptance gate
``scripts/dome_qa.py`` (``studio.coverage`` — centre ``(size-1)/2``, ``R =
size/2``), so ``rimlift`` brightens exactly the annuli that gate measures.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

# One radius-map definition for the whole repo (issue #401 unified the QA gate
# on this one); rimlift's r must be the same r dome_qa.py measures.
from studio.coverage import _radial_map  # noqa: E402

log = logging.getLogger("dome_frame_tools")

#: Target canvas aspect for ``pad169`` (16:9).
ASPECT_W = 16
ASPECT_H = 9

#: Container suffixes routed through ffmpeg by ``crop11``; everything else is
#: treated as a still image and handled by OpenCV.
VIDEO_EXTS = frozenset({".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"})

#: ``crop11`` video encode settings — the dome-orientation mapper uses the same
#: crf, and ``yuv420p`` keeps the result playable in the browser preview.
CROP11_CRF = 18
CROP11_PIX_FMT = "yuv420p"
#: Filtergraph label for the masked 1:1 stream (``_``-prefixed, like the labels
#: in ``pipeline/fulldome_mapper.py``).
_CROP11_LABEL = "_dome_1x1"
#: Wall-clock limit for one ffmpeg pass (matches the mapper's 3600 s).
FFMPEG_TIMEOUT_SEC = 3600

#: ``rimlift`` defaults (issue #408 spec).
DEFAULT_TARGET = 0.55
DEFAULT_START = 0.5
DEFAULT_MAX_GAIN = 4.0
#: "Centre" luminance for the target computation: mean over ``r < CENTER_RADIUS``.
CENTER_RADIUS = 0.3
#: Gain-map Gaussian σ, expressed at the 1024² reference and scaled with size.
RIM_SIGMA_PX = 15.0
RIM_SIGMA_REF = 1024.0
#: Annuli used for the radius-profile statistics.
N_RINGS = 32


# --------------------------------------------------------------------------- #
# Still-image I/O
# --------------------------------------------------------------------------- #


def is_video(path: str | Path) -> bool:
    """True when ``path``'s suffix names a video container (see ``VIDEO_EXTS``)."""
    return Path(path).suffix.lower() in VIDEO_EXTS


def _read_image(path: str | Path) -> np.ndarray:
    """Read ``path`` as a 3-channel BGR uint8 array (OpenCV, always BGR).

    ``cv2.imread`` returns ``None`` for a missing file, an unsupported format
    or a *video* container — all three used to surface much later as an
    ``AttributeError`` on ``None.shape``.  ``IMREAD_COLOR`` normalises
    grayscale / RGBA inputs to 3-channel BGR so the geometry code below only
    ever deals with one layout.
    """
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(
            f"cannot read image {path!s} — missing file, unsupported format, or a video "
            "(pad169/rimlift take stills; crop11 handles videos)"
        )
    return img


def _write_image(path: str | Path, img: np.ndarray) -> str:
    """Write ``img`` to ``path`` (PNG stays lossless, JPEG uses OpenCV's q=95)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out), img):
        raise OSError(f"could not write image {out!s} (unknown extension or unwritable path)")
    return str(out)


# --------------------------------------------------------------------------- #
# pad169 — 1:1 → 16:9
# --------------------------------------------------------------------------- #


def pad169_width(height: int) -> int:
    """Even canvas width for a 16:9 canvas of the given ``height``.

    ``round(h * 16 / 9)`` as the card specifies, then rounded up to even:
    ffmpeg cannot encode an odd-width ``yuv420p`` frame, so the padded result
    is expected to survive a video pass.  1024 → 1820 (already even).
    """
    if height <= 0:
        raise ValueError(f"height must be positive, got {height}")
    width = round(height * ASPECT_W / ASPECT_H)
    return width + (width % 2)


def pad169_image(img: np.ndarray) -> np.ndarray:
    """Centre a square image on a black 16:9 canvas (height unchanged).

    The image is pasted **verbatim** — a view of the same rows and columns, no
    resampling — so the padded centre is bit-identical to the input (the
    round-trip guarantee ``crop11`` relies on).  The bars are exactly 0.
    """
    if img.ndim != 3 or img.shape[0] != img.shape[1]:
        raise ValueError(f"pad169 expects a 1:1 image, got {img.shape[1]}x{img.shape[0]}")
    h = img.shape[0]
    w = pad169_width(h)
    canvas = np.zeros((h, w, img.shape[2]), dtype=img.dtype)
    x0 = (w - h) // 2
    canvas[:, x0 : x0 + h] = img
    return canvas


def pad169_file(input_path: str | Path, output_path: str | Path) -> str:
    """``pad169`` on files: read → pad → write.  Returns the output path."""
    padded = pad169_image(_read_image(input_path))
    written = _write_image(output_path, padded)
    log.info(f"pad169: {input_path} → {written} ({padded.shape[1]}x{padded.shape[0]}, black side bars)")
    return written


# --------------------------------------------------------------------------- #
# crop11 — 16:9 → 1:1 (centre square + inscribed-circle mask)
# --------------------------------------------------------------------------- #


def center_square_bounds(width: int, height: int) -> tuple[int, int, int]:
    """``(x0, y0, side)`` of the centred square crop that ffmpeg's ``crop=ih:ih`` takes.

    ``crop``'s default position is the centre, computed with integer division
    (``x = (in_w - out_w) / 2``), and the side is the input *height* — the
    largest square a landscape frame offers.  Keeping that arithmetic in one
    place is what makes the OpenCV (still) and ffmpeg (video) paths agree
    pixel for pixel.
    """
    if width < height:
        raise ValueError(f"crop11 needs a landscape or square frame, got {width}x{height} (side = height)")
    side = height
    return (width - side) // 2, (height - side) // 2, side


def circle_mask(height: int, width: int) -> np.ndarray:
    """Boolean ``r <= 1`` mask: the inscribed circle, in ``dome_qa.py``'s r.

    ``r`` comes from ``studio.coverage._radial_map`` (centre ``(size-1)/2``,
    ``R = min(h, w)/2``) so "inside the circle" means precisely what the dome
    acceptance gate measures as ``r <= 1``.
    """
    return _radial_map(height, width) <= 1.0


def crop11_image(img: np.ndarray) -> np.ndarray:
    """Centre square crop of ``img``, everything outside the circle set to 0."""
    h, w = img.shape[:2]
    x0, y0, side = center_square_bounds(w, h)
    square = np.ascontiguousarray(img[y0 : y0 + side, x0 : x0 + side])
    out = np.zeros_like(square)
    inside = circle_mask(side, side)
    out[inside] = square[inside]
    return out


def geq_circle_expression() -> str:
    """The ``geq`` predicate marking the inscribed circle of the current frame.

    ``X``/``Y`` are pixel centre coordinates (hence ``(W-1)/2`` for the centre)
    and ``W``/``H`` are the *cropped* frame's dimensions, so the mask follows
    the source resolution without the tool having to probe the file first —
    no second ffmpeg input, no temp PNG to write and clean up.
    """
    return "lte(hypot(X-(W-1)/2,Y-(H-1)/2),min(W,H)/2)"


def crop11_filter() -> str:
    """The filtergraph: ``crop=ih:ih`` → ``geq`` circle mask → ``yuv420p``.

    ``format=gbrp`` is what makes the mask exact: ``geq`` then addresses
    full-range RGB planes with no chroma subsampling to smear the circle edge,
    and ``r(X,Y)``/``g(X,Y)``/``b(X,Y)`` pass the inside pixels through
    unchanged while the outside gets a literal 0 in every plane.

    The frames handed to the encoder are therefore exactly black outside the
    circle; what the *decoder* gives back is not, quite: 4:2:0 shares one
    chroma sample per 2×2 luma block, so the blocks straddling the circle edge
    carry content chroma onto their black luma, and the DCT rings a little
    further out.  Measured on a 180² testsrc2 clip at crf 18: mean 1.3/255 over
    the whole surround (the dome gate's own outside threshold is 8), halo
    amplitude < 16 beyond 3 px.  Nothing in a 4:2:0 bitstream can avoid that;
    ``yuv444p`` could, at the cost of player compatibility.
    """
    circle = geq_circle_expression()
    return (
        "[0:v]crop=ih:ih,format=gbrp,"
        f"geq=r='if({circle},r(X,Y),0)':g='if({circle},g(X,Y),0)':b='if({circle},b(X,Y),0)',"
        f"format={CROP11_PIX_FMT}[{_CROP11_LABEL}]"
    )


def build_crop11_command(
    input_path: str | Path,
    output_path: str | Path,
    ffmpeg: str = "ffmpeg",
) -> list[str]:
    """The exact ffmpeg argv for the video path — **list form**, never ``shell=True``.

    Audio is ``-map 0:a:0?`` + ``-c:a copy``: the optional map keeps a silent
    source working and ``copy`` keeps the track byte-identical, the same
    guarantee ``pipeline/fulldome_mapper.py`` gives a dome master.
    """
    return [
        ffmpeg,
        "-y",
        "-v",
        "error",
        "-i",
        str(input_path),
        "-filter_complex",
        crop11_filter(),
        "-map",
        f"[{_CROP11_LABEL}]",
        "-map",
        "0:a:0?",
        "-c:a",
        "copy",
        "-c:v",
        "libx264",
        "-crf",
        str(CROP11_CRF),
        "-pix_fmt",
        CROP11_PIX_FMT,
        str(output_path),
    ]


def crop11_video(input_path: str | Path, output_path: str | Path, ffmpeg: str = "ffmpeg") -> str:
    """Run :func:`build_crop11_command` and return the output path (raises on failure)."""
    cmd = build_crop11_command(input_path, output_path, ffmpeg)
    log.debug("ffmpeg command: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=FFMPEG_TIMEOUT_SEC)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg crop11 failed (exit {result.returncode}):\nstderr:\n{result.stderr[:2000]}")
    log.info(f"crop11: {input_path} → {output_path} (centre square, circle mask, audio copy)")
    return str(output_path)


def crop11_file(input_path: str | Path, output_path: str | Path, ffmpeg: str = "ffmpeg") -> str:
    """``crop11`` on files: video containers via ffmpeg, stills via OpenCV."""
    if is_video(input_path):
        return crop11_video(input_path, output_path, ffmpeg)
    written = _write_image(output_path, crop11_image(_read_image(input_path)))
    log.info(f"crop11: {input_path} → {written} (centre square, circle mask)")
    return written


# --------------------------------------------------------------------------- #
# rimlift — radial brightening towards a fraction of the centre luminance
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RimProfile:
    """Radius-profile statistics of one ``rimlift`` pass (logged and asserted on).

    ``ring_gains`` is the *raw* per-annulus gain before smoothing — the number
    a reviewer wants when a lift looks wrong ("was the cap binding?").
    """

    center_level: float
    target_level: float
    ring_centers: tuple[float, ...]
    ring_gains: tuple[float, ...]


def radius_profile_gains(
    gray: np.ndarray,
    radius: np.ndarray,
    *,
    start: float = DEFAULT_START,
    target: float = DEFAULT_TARGET,
    max_gain: float = DEFAULT_MAX_GAIN,
    n_rings: int = N_RINGS,
) -> RimProfile:
    """Measure the brightness profile and derive one gain per annulus.

    Rings run from ``start`` to ``r = 1`` (the inscribed-circle rim) in
    ``n_rings`` steps.  Each ring's gain is ``target_level / ring_mean`` with
    ``target_level = target × centre`` (centre = mean over ``r < 0.3``), clamped
    into ``[1, max_gain]``: a gain below 1 would *darken* a ring that is already
    at or above the target, and the cap stops a nearly-black rim from being
    multiplied into noise.

    A ring whose mean is 0 (pure black) keeps gain 1: multiplying black by any
    factor is still black, so the ring is left bit-exact rather than pretended
    to be fixed.
    """
    if not 0.0 <= start < 1.0:
        raise ValueError(f"--start must be in [0, 1), got {start}")
    if target < 0.0:
        raise ValueError(f"--target must be >= 0, got {target}")
    if max_gain < 1.0:
        raise ValueError(f"--max-gain must be >= 1, got {max_gain}")
    if n_rings < 1:
        raise ValueError(f"n_rings must be >= 1, got {n_rings}")

    inside = radius <= 1.0
    centre_sel = inside & (radius < CENTER_RADIUS)
    center_level = float(gray[centre_sel].mean()) if centre_sel.any() else 0.0
    target_level = float(target * center_level)

    edges = np.linspace(start, 1.0, n_rings + 1)
    ring_centers = tuple(float(c) for c in (edges[:-1] + edges[1:]) / 2.0)
    ring_gains = np.ones(n_rings, dtype=np.float64)
    for i, (lo, hi) in enumerate(pairwise(edges)):
        ring_sel = inside & (radius >= lo) & (radius <= hi)
        if not ring_sel.any():
            continue
        ring_mean = float(gray[ring_sel].mean())
        if ring_mean <= 0.0:
            continue
        ring_gains[i] = min(max(target_level / ring_mean, 1.0), max_gain)

    return RimProfile(center_level, target_level, ring_centers, tuple(float(g) for g in ring_gains))


def gain_map_from_profile(
    radius: np.ndarray,
    profile: RimProfile,
    *,
    start: float = DEFAULT_START,
    sigma_px: float = RIM_SIGMA_PX,
) -> np.ndarray:
    """The 2D gain field: per-ring gains interpolated in r, then Gaussian-smoothed.

    Smoothing is what keeps the 32 annuli from showing up as rings of their own;
    σ is specified at the 1024² reference and scaled with the frame so a 512²
    preview and a 4096² master get the same *visual* feathering.  The blurred
    map is then pinned back to exactly 1 for ``r < start``: without that, a
    bright rim's gain bleeds inward and the "inner region is bit-exact"
    guarantee silently stops holding.
    """
    h, w = radius.shape
    nodes_x = np.concatenate([[0.0], np.asarray(profile.ring_centers)])
    nodes_y = np.concatenate([[1.0], np.asarray(profile.ring_gains)])
    gain = np.interp(radius, nodes_x, nodes_y).astype(np.float32)
    sigma = max(1e-3, sigma_px * min(h, w) / RIM_SIGMA_REF)
    gain = cv2.GaussianBlur(gain, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)
    gain[radius < start] = 1.0
    return gain


def rimlift_image(
    img: np.ndarray,
    *,
    target: float = DEFAULT_TARGET,
    start: float = DEFAULT_START,
    max_gain: float = DEFAULT_MAX_GAIN,
) -> tuple[np.ndarray, RimProfile]:
    """Lift the outer rings of a square domemaster image.  Returns ``(out, profile)``.

    ``r >= start`` is scaled by the smoothed gain map, ``r < start`` comes back
    bit-exact (gain pinned to 1) and everything outside the inscribed circle is
    forced to pure black, so the result stays a valid domemaster whatever the
    source's corners held.
    """
    if img.ndim != 3 or img.shape[0] != img.shape[1]:
        raise ValueError(
            f"rimlift expects a 1:1 domemaster image, got {img.shape[1]}x{img.shape[0]} — run crop11 first"
        )
    gray = img.astype(np.float32).mean(axis=2)
    radius = _radial_map(img.shape[0], img.shape[1])
    profile = radius_profile_gains(gray, radius, start=start, target=target, max_gain=max_gain)
    if profile.center_level <= 0.0 or profile.target_level <= 0.0:
        # A black frame (or --target 0) gives the lift nothing to aim at.
        gain = np.ones_like(gray)
    else:
        gain = gain_map_from_profile(radius, profile, start=start)
    lifted = np.clip(img.astype(np.float32) * gain[..., None], 0.0, 255.0)
    lifted[radius > 1.0] = 0.0
    return np.rint(lifted).astype(np.uint8), profile


def rimlift_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    target: float = DEFAULT_TARGET,
    start: float = DEFAULT_START,
    max_gain: float = DEFAULT_MAX_GAIN,
) -> str:
    """``rimlift`` on files (stills only — the card does not require video)."""
    if is_video(input_path):
        raise ValueError("rimlift handles still images only — extract the frames first (crop11 handles video)")
    lifted, profile = rimlift_image(
        _read_image(input_path),
        target=target,
        start=start,
        max_gain=max_gain,
    )
    written = _write_image(output_path, lifted)
    log.info(
        f"rimlift: {input_path} → {written} "
        f"(centre {profile.center_level:.1f}, target {profile.target_level:.1f}, "
        f"ring gain {min(profile.ring_gains, default=1.0):.2f}–{max(profile.ring_gains, default=1.0):.2f})"
    )
    return written


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    """The ``pad169`` / ``crop11`` / ``rimlift`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="dome_frame_tools.py",
        description="Dome frame geometry tools: 16:9 ⇄ 1:1 conversion + radial rim lift (issue #408).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    pad = sub.add_parser("pad169", help="centre a 1:1 image on a black 16:9 canvas")
    pad.add_argument("input", help="1:1 source image (jpg/png)")
    pad.add_argument("output", help="16:9 output image")

    crop = sub.add_parser("crop11", help="centre-square crop + inscribed-circle black mask")
    crop.add_argument("input", help="16:9 (or square) source: mp4/jpg/png")
    crop.add_argument("output", help="1:1 output (mp4 for a video input)")
    crop.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg binary used for the video path")

    rim = sub.add_parser("rimlift", help="brighten the outer rings towards --target × centre luminance")
    rim.add_argument("input", help="1:1 domemaster image (run crop11 first)")
    rim.add_argument("output", help="output image")
    rim.add_argument(
        "--target",
        type=float,
        default=DEFAULT_TARGET,
        help="rim target as a fraction of the centre luminance (default 0.55)",
    )
    rim.add_argument(
        "--start",
        type=float,
        default=DEFAULT_START,
        help="normalised radius where the lift starts; r < start is untouched (default 0.5)",
    )
    rim.add_argument(
        "--max-gain",
        type=float,
        default=DEFAULT_MAX_GAIN,
        help="per-ring gain cap (default 4)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one subcommand.  Returns the process exit code (0 ok, 1 failed)."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        if args.command == "pad169":
            pad169_file(args.input, args.output)
        elif args.command == "crop11":
            crop11_file(args.input, args.output, ffmpeg=args.ffmpeg)
        else:
            rimlift_file(args.input, args.output, target=args.target, start=args.start, max_gain=args.max_gain)
    except (OSError, RuntimeError, ValueError) as exc:
        log.error(f"{args.command} failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
