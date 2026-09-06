"""
F-2 — Temporal spherical accumulation fill (issue #262)
=======================================================

Pure-function layer.  **Not wired** into ``streaming_pipeline`` /
``run_pipeline`` — that is the next card.

Why
---
A planar source projected onto a 180° hemisphere never fills it: at 126°
horizontal FOV the outer ~27° of the per-eye equirectangular canvas is an
``alpha == 0`` hole (see ``SOURCE_SPEC.md``).  Forward flight, however, has a
property no other camera motion has: content flows **radially outward** from
the canvas centre and eventually leaves the frame — so the patch of sphere
that is *outside* the current FOV was really seen a few frames ago.  The
outer ring therefore does not need to be generated, only remembered and
moved.

What
----
:class:`SphereAccumulator` keeps a persistent premultiplied-RGBA buffer ``B``
of one hemisphere (square half-equirect canvas, forward axis at the centre —
the ``with_alpha=True`` output of
:class:`pipeline.equirectangular_mapper.EquirectangularMapper` for one eye).
Per frame ``V_t``:

1. ``B ← warp_radial(B, radial_scale)`` — a **global scalar** radial
   expansion about the canvas centre (``cv2.remap``).  Skipped when
   ``radial_scale <= 1`` (static / backward).  Per-pixel depth-driven flow is
   left to the next card.
2. ``B ← V_t over B`` — the current frame's covered area (``alpha > 0``)
   overwrites the buffer.  A short feather (default 4°, see
   :data:`DEFAULT_SEAM_FEATHER_DEG`) *inside* the current frame's edge blends
   it into the remembered content so no seam shows.  Where the buffer is still
   empty the current frame keeps full weight, so the cold start (frame 0) is
   byte-exact ``frame + fade``.
3. Output = ``B`` as RGB-on-black with the outermost
   ``fade_start → fade_end`` angle-weighted fade to black, reusing
   :func:`pipeline.outpainter.compute_edge_feather_weights` (issue #244).  The
   fade is applied to the *output only*: the buffer keeps unfaded content,
   otherwise the ring would darken cumulatively frame after frame.

Usage::

    acc = SphereAccumulator(size=1920)               # per-eye canvas size
    for rgba, scale in zip(eye_frames, radial_scales):
        rgb = acc.update(rgba, scale)                  # (1920, 1920, 3) uint8
"""

import math

import cv2
import numpy as np

from pipeline.outpainter import (
    DEFAULT_EDGE_FEATHER_END,
    DEFAULT_EDGE_FEATHER_START,
    compute_edge_feather_weights,
    resolve_edge_feather,
)

#: Width (degrees, 0–180 FOV scale) of the feather inside the current frame's
#: content edge that blends it into the accumulated buffer.  The card asks for
#: 3–5°; 4° is the middle of that range.
DEFAULT_SEAM_FEATHER_DEG = 4.0


# ---------------------------------------------------------------------------
#  Pure functions
# ---------------------------------------------------------------------------


def radial_warp_maps(shape: tuple[int, int], radial_scale: float) -> tuple[np.ndarray, np.ndarray]:
    """``cv2.remap`` maps that expand an ``(H, W)`` canvas radially about its centre.

    A pixel at offset ``p`` from the centre of the *output* samples the input
    at ``p / radial_scale`` — content moves outward for ``radial_scale > 1``.
    The centre is the pixel-centre convention of
    ``pipeline.outpainter._hemisphere_pixel_angles`` (``(W - 1) / 2``).
    """
    h, w = shape
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    xs = (np.arange(w, dtype=np.float32) - cx) / radial_scale + cx
    ys = (np.arange(h, dtype=np.float32) - cy) / radial_scale + cy
    map_x = np.ascontiguousarray(np.broadcast_to(xs[None, :], (h, w)))
    map_y = np.ascontiguousarray(np.broadcast_to(ys[:, None], (h, w)))
    return map_x, map_y


def warp_radial(buffer: np.ndarray, radial_scale: float) -> np.ndarray:
    """Radially expand *buffer* ``(H, W[, C≤4])`` about its centre by *radial_scale*.

    Bilinear resampling; anything pulled in from outside the canvas is 0
    (transparent).  A premultiplied-alpha buffer stays fringe-free under this
    interpolation, which is why :class:`SphereAccumulator` stores one.
    """
    map_x, map_y = radial_warp_maps(buffer.shape[:2], radial_scale)
    return cv2.remap(buffer, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def seam_weights(covered: np.ndarray, feather_px: float) -> np.ndarray:
    """Per-pixel weight of the current frame *inside* its covered area.

    1 deep inside the content, ramping linearly down to 0 at the last covered
    pixel before the hole over *feather_px* pixels; exactly 0 in the hole.
    Only hole pixels count as an edge — the canvas border is the hemisphere
    rim, not a seam.
    """
    cov = (np.asarray(covered) > 0).astype(np.uint8)
    if not cov.any():
        return np.zeros(cov.shape, dtype=np.float32)
    if feather_px <= 0.0 or cov.all():
        return cov.astype(np.float32)
    dist = cv2.distanceTransform(cov, cv2.DIST_L2, 5)  # px to the nearest hole pixel (1 at the edge)
    return np.clip((dist - 1.0) / feather_px, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
#  Accumulator
# ---------------------------------------------------------------------------


class SphereAccumulator:
    """Temporal accumulation buffer for one hemisphere (see module docstring).

    Args:
        size: Per-eye canvas size in pixels (square, ``size × size``).
        fade_start_deg: Angle (0–180 FOV scale) where the output fade to black
            starts.  Default 165.
        fade_end_deg: Angle where the output is fully black.  Default 180.
            ``None`` for **both** fade args switches the fade off (raw
            accumulation); validation is :func:`pipeline.outpainter.resolve_edge_feather`.
        seam_feather_deg: Width of the blend inside the current frame's content
            edge (keyword-only, default :data:`DEFAULT_SEAM_FEATHER_DEG`).
    """

    def __init__(
        self,
        size: int,
        fade_start_deg: float | None = DEFAULT_EDGE_FEATHER_START,
        fade_end_deg: float | None = DEFAULT_EDGE_FEATHER_END,
        *,
        seam_feather_deg: float = DEFAULT_SEAM_FEATHER_DEG,
    ):
        if int(size) != size or size < 2:
            raise ValueError(f"size must be an integer >= 2, got {size!r}")
        if not (math.isfinite(seam_feather_deg) and seam_feather_deg >= 0.0):
            raise ValueError(f"seam_feather_deg must be a finite non-negative angle, got {seam_feather_deg!r}")
        self._size = int(size)
        self._fade = resolve_edge_feather(fade_start_deg, fade_end_deg)
        self._seam_feather_deg = float(seam_feather_deg)
        # 1 px = 180 / size degrees at the canvas centre (same rule as the outpainter feather).
        self._seam_px = self._seam_feather_deg * self._size / 180.0
        self._buf: np.ndarray | None = None  # (size, size, 4) float32: premultiplied RGB 0..255, alpha 0..1
        self._frames = 0
        self._fade_cache: tuple[np.ndarray, np.ndarray] | None = None

    # ------------------------------------------------------------------ #
    #  introspection
    # ------------------------------------------------------------------ #

    @property
    def size(self) -> int:
        return self._size

    @property
    def fade(self) -> tuple[float, float] | None:
        """``(start, end)`` in degrees, or ``None`` when the fade is off."""
        return self._fade

    @property
    def seam_feather_deg(self) -> float:
        return self._seam_feather_deg

    @property
    def frames_seen(self) -> int:
        """Number of :meth:`update` calls since construction / :meth:`reset`."""
        return self._frames

    @property
    def alpha(self) -> np.ndarray:
        """Coverage of the accumulated buffer ``(size, size)`` float32 in ``[0, 1]`` (copy).

        1 where the buffer holds real content, 0 where nothing has been seen yet.
        Zeros before the first :meth:`update`.
        """
        if self._buf is None:
            return np.zeros((self._size, self._size), dtype=np.float32)
        return self._buf[:, :, 3].copy()

    def reset(self) -> None:
        """Forget everything (scene cut / new clip)."""
        self._buf = None
        self._frames = 0
        self._fade_cache = None

    # ------------------------------------------------------------------ #
    #  main entry
    # ------------------------------------------------------------------ #

    def update(self, frame_rgba: np.ndarray, radial_scale: float) -> np.ndarray:
        """Accumulate one frame and return the composited hemisphere.

        Args:
            frame_rgba: Current frame on the equirect canvas, ``(size, size, 4)``
                uint8.  ``alpha == 0`` marks the hole outside the source FOV; RGB
                there is ignored.
            radial_scale: Radial expansion of this frame relative to the previous
                one (``> 1`` while flying forward; estimated by the caller from
                depth / displacement).  ``<= 1`` means no expansion — the buffer is
                only composited, never shrunk.

        Returns:
            RGB ``(size, size, 3)`` uint8 — current frame over the remembered
            content, faded to black over ``fade_start → fade_end``.
        """
        frame = self._check_frame(frame_rgba)
        if not math.isfinite(radial_scale) or radial_scale <= 0.0:
            raise ValueError(f"radial_scale must be a finite positive number, got {radial_scale!r}")

        if self._buf is None:
            buf = np.zeros((self._size, self._size, 4), dtype=np.float32)
        elif radial_scale > 1.0:
            buf = warp_radial(self._buf, float(radial_scale))
        else:
            buf = self._buf

        # --- V_t over B_{t-1} (premultiplied) ---
        v_alpha = frame[:, :, 3].astype(np.float32) / 255.0
        v_rgb = frame[:, :, :3].astype(np.float32)
        b_alpha = buf[:, :, 3]
        # Current-frame weight: full where the buffer is empty (cold start), the
        # seam feather where remembered content exists to blend into.
        feather = seam_weights(frame[:, :, 3], self._seam_px)
        a_eff = v_alpha * (1.0 - b_alpha * (1.0 - feather))
        out_rgb = a_eff[:, :, None] * v_rgb + (1.0 - a_eff)[:, :, None] * buf[:, :, :3]
        out_alpha = a_eff + (1.0 - a_eff) * b_alpha

        new_buf = np.empty_like(buf)
        new_buf[:, :, :3] = out_rgb
        new_buf[:, :, 3] = out_alpha
        self._buf = new_buf
        self._frames += 1

        # --- output-only angle-weighted fade at the accumulated content edge ---
        weights = self._fade_weights(out_alpha > 0.0)
        if weights is not None:
            out_rgb = out_rgb * weights[:, :, None]
        return np.clip(np.rint(out_rgb), 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------ #
    #  helpers
    # ------------------------------------------------------------------ #

    def _check_frame(self, frame_rgba: np.ndarray) -> np.ndarray:
        frame = np.asarray(frame_rgba)
        expected = (self._size, self._size, 4)
        if frame.shape != expected:
            raise ValueError(f"frame_rgba must have shape {expected} (square RGBA canvas), got {frame.shape}")
        if frame.dtype != np.uint8:
            raise TypeError(f"frame_rgba must be uint8, got {frame.dtype}")
        return frame

    def _fade_weights(self, covered: np.ndarray) -> np.ndarray | None:
        """Fade weights for the composited coverage; cached while the coverage mask is unchanged."""
        if self._fade is None:
            return None
        if self._fade_cache is not None and np.array_equal(self._fade_cache[0], covered):
            return self._fade_cache[1]
        start, end = self._fade
        weights = compute_edge_feather_weights(covered.astype(np.uint8) * 255, start, end)
        self._fade_cache = (covered.copy(), weights)
        return weights
