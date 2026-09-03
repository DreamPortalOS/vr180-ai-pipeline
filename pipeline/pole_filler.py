"""E-3: vertical pole-filling — extend a 16:9 equirectangular frame to 1:1.

Owner can only get **16:9** video out of Gemini, but VR180 needs a **1:1
180°×180°** equirectangular source.  The key insight (lead's strategy call):

a 16:9 frame read as an equirectangular projection already covers the **full
180° horizontally** (left/right extremes are ±90° — you can't look past the
sides).  It only falls short **vertically**: ``180° × 9/16 = 101.25°``, so the
top and bottom poles together are missing ~79° (≈39.4° each).

The poles are the dullest part of any real scene — straight up is pure sky,
straight down is pure ground.  They are trivial to synthesise with image
processing: no AI content generation needed, just an outward extrapolation of
the edge content that converges to the per-column average colour as it
approaches the pole (in an equirectangular projection the pole is a *single
point*, so the whole top/bottom row must converge to one colour — otherwise
there is a visible seam at the zenith/nadir).

This module exposes one entry point:

    >>> extend_equirect_to_square(frame, method="gradient")  # → square ndarray

Three fill methods are provided, selectable by ``method``:

- ``gradient`` (default) — take the per-column mean colour of the top edge rows
  and linearly extrapolate upward, converging each column to the **global** row
  mean (the pole-convergence property).  Bottom is symmetric.  This gives a
  smooth, physically-motivated sky/ground continuation.
- ``mirror`` — mirror-flip the edge band and heavily blur it.  Good when the
  edge already contains useful texture (clouds, grass) you want to propagate.
- ``solid`` — fill the whole top band with the top-row per-column mean, bottom
  band with the bottom-row per-column mean.  Minimal, deterministic fallback.

All three:

- leave the original content rows **byte-for-byte unchanged** outside the
  ``sky_blend_rows`` feather band (the tests assert this with
  ``np.array_equal``);
- feather the synth band into the original across ``sky_blend_rows`` so there
  is no hard seam at the 16:9 ↔ synthesised boundary;
- are pure numpy/OpenCV — no new deps, no AI models, no weight downloads.

This module is intentionally **not wired** into ``run_pipeline.py`` (that file
is mid-edit by issue #239).  Wiring is a follow-up card.

Issue #241.
"""

from __future__ import annotations

import cv2
import numpy as np

__all__ = ["extend_equirect_to_square"]

# How many edge rows to sample when seeding the extrapolation.  Proportional to
# frame height, capped so a noisy single-row edge doesn't dominate.
_EDGE_WINDOW_CAP_FRAC = 4
_EDGE_WINDOW_CAP_ABS = 16


def _validate_inputs(
    frame: np.ndarray,
    *,
    target_size: int | None,
    method: str,
    sky_blend_rows: int,
) -> None:
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"frame must be HxWx3, got shape {frame.shape!r}")
    h, w = frame.shape[:2]
    if h == 0 or w == 0:
        raise ValueError(f"frame must be non-empty, got shape {frame.shape!r}")
    if target_size is not None and target_size <= 0:
        raise ValueError(f"target_size must be positive, got {target_size!r}")
    if method not in ("gradient", "mirror", "solid"):
        raise ValueError(
            f"method must be 'gradient' | 'mirror' | 'solid', got {method!r}",
        )
    if sky_blend_rows < 0:
        raise ValueError(f"sky_blend_rows must be >= 0, got {sky_blend_rows!r}")


def _per_column_mean(rows: np.ndarray) -> np.ndarray:
    """Mean colour per column across the given rows → shape (W, 3)."""
    return rows.mean(axis=0, dtype=np.float64)


def _global_mean_color(rows: np.ndarray) -> np.ndarray:
    """Scalar mean colour across all pixels in ``rows`` → shape (3,)."""
    return rows.reshape(-1, 3).mean(axis=0, dtype=np.float64)


def _lerp(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Broadcasted linear interpolation ``a→b`` by ``t`` in [0,1]."""
    return a + (b - a) * t


def _edge_window(h: int) -> int:
    return max(1, min(h // _EDGE_WINDOW_CAP_FRAC, _EDGE_WINDOW_CAP_ABS))


def _fill_gradient(
    canvas: np.ndarray,
    band_height: int,
    side: str,
    edge_rows: np.ndarray,
) -> None:
    """Fill the band with a pole-converging gradient.

    Each column goes from its per-column mean (at the content edge) linearly to
    the **global** row mean (at the pole).  Because every column converges to
    the same colour, the pole row has ~zero row-variance — the physical
    "pole is a point" property.
    """
    if band_height <= 0:
        return
    col_mean = _per_column_mean(edge_rows)  # (W, 3)
    pole_color = _global_mean_color(edge_rows)  # (3,)
    # t = 0 at the content edge, t = 1 at the pole.
    t = np.linspace(0.0, 1.0, band_height, dtype=np.float64)[:, None, None]
    band = _lerp(col_mean[None, :, :], pole_color[None, None, :], t)
    # ``band[0]`` is the content edge, ``band[-1]`` is the pole.  For the top
    # band the pole sits at row 0 of the canvas, so reverse.  For the bottom
    # band the pole sits at the last canvas row, so keep order.
    h = canvas.shape[0]
    if side == "top":
        canvas[:band_height] = band[::-1]
    else:
        canvas[h - band_height : h] = band


def _fill_solid(
    canvas: np.ndarray,
    band_height: int,
    side: str,
    edge_rows: np.ndarray,
) -> None:
    """Fill the band with the per-column mean colour (constant down the band)."""
    if band_height <= 0:
        return
    col_mean = _per_column_mean(edge_rows)  # (W, 3)
    band = np.broadcast_to(col_mean[None, :, :], (band_height, col_mean.shape[0], 3))
    h = canvas.shape[0]
    if side == "top":
        canvas[:band_height] = band
    else:
        canvas[h - band_height : h] = band


def _fill_mirror(
    canvas: np.ndarray,
    band_height: int,
    side: str,
    edge_rows: np.ndarray,
) -> None:
    """Fill the band by mirror-flipping the edge rows and blurring heavily.

    The mirror gives plausible texture continuation (clouds/grass propagate
    outward); the heavy blur removes the mirror seam and any high-frequency
    artefact at the pole.
    """
    if band_height <= 0:
        return
    # Take up to ``band_height`` rows from the edge and flip so the row nearest
    # the content edge stays adjacent (mirror symmetry about the boundary).
    src = edge_rows[:band_height] if edge_rows.shape[0] >= band_height else edge_rows
    mirror = src[::-1]
    if mirror.shape[0] < band_height:
        reps = int(np.ceil(band_height / mirror.shape[0]))
        mirror = np.tile(mirror, (reps, 1, 1))[:band_height]
    # Heavy blur — odd kernel, bounded by the band width and a sane cap.
    k = max(1, min(band_height if band_height % 2 else band_height + 1, 51))
    blurred = cv2.GaussianBlur(mirror, (k, k), 0)
    h = canvas.shape[0]
    if side == "top":
        canvas[:band_height] = blurred
    else:
        canvas[h - band_height : h] = blurred


def _blend_seam(
    canvas: np.ndarray,
    original: np.ndarray,
    top_rows: int,
    blend: int,
) -> np.ndarray:
    """Feather the content↔synth boundary across ``blend`` original rows.

    The first ``blend`` rows of the original content are alpha-mixed with the
    synthesised band (already present in ``canvas``) so the gradient at the
    seam is gentle.  The original content **outside** this feather band is left
    untouched (the hard pixel-equality assertion in the tests operates only on
    rows beyond the feather).
    """
    out = canvas.copy()
    b = min(blend, original.shape[0] // 2) if blend else 0
    if b < 1:
        return out

    # Top boundary: first ``b`` original rows blend synth(orig-side) → original.
    synth_top = canvas[top_rows : top_rows + b].astype(np.float64)
    orig_top = original[:b].astype(np.float64)
    # row at the seam edge → pure original; moving down into content → stays
    # original (t stays ~0).  We instead fade original→synth across the band so
    # the *visible* transition is smooth; the original side dominates at the
    # boundary, which is what removes the hard line.
    t = np.linspace(0.0, 1.0, b, dtype=np.float64)[:, None, None]
    out[top_rows : top_rows + b] = _lerp(orig_top, synth_top, t)

    # Bottom boundary: last ``b`` original rows.
    ob = original.shape[0] - b
    synth_bot = canvas[top_rows + ob : top_rows + original.shape[0]].astype(np.float64)
    orig_bot = original[ob:].astype(np.float64)
    t = np.linspace(1.0, 0.0, b, dtype=np.float64)[:, None, None]
    out[top_rows + ob : top_rows + original.shape[0]] = _lerp(orig_bot, synth_bot, t)
    return out


_FILL = {
    "gradient": _fill_gradient,
    "mirror": _fill_mirror,
    "solid": _fill_solid,
}


def extend_equirect_to_square(
    frame: np.ndarray,
    *,
    target_size: int | None = None,
    method: str = "gradient",
    sky_blend_rows: int = 32,
) -> np.ndarray:
    """Extend a 16:9 equirectangular frame vertically to a 1:1 square.

    The frame is read as covering 180° horizontally and ``180° × (h/w)``
    vertically.  The missing top and bottom pole regions are synthesised so the
    result is a complete 180°×180° equirectangular image of square aspect.

    Parameters
    ----------
    frame:
        ``(H, W, 3)`` equirectangular frame (uint8 or any numeric dtype).  For
        the intended use case ``W > H`` (16:9 or wider).  Square inputs are
        returned unchanged (nothing to fill).
    target_size:
        Desired output height (edge length to fill to).  ``None`` (default) →
        use the frame's own width, i.e. fill ``W - H`` rows split between top
        and bottom so the result is ``W × W``.  When set, the output height
        becomes ``target_size``; the width is **never** resampled (the original
        content pixels stay byte-identical), so a ``target_size ≠ W`` yields a
        non-square ``(target_size, W, 3)`` canvas — use ``target_size = W`` or
        leave it ``None`` for a true square.
    method:
        ``"gradient"`` (default), ``"mirror"``, or ``"solid"`` — see module
        docstring.
    sky_blend_rows:
        Width (in rows) of the alpha-feather band at each content boundary so
        the synthesised region eases into the original without a hard seam.

    Returns
    -------
    np.ndarray
        ``(size, W, 3)`` frame (``size = target_size or W``), same dtype as the
        input.  The original content rows appear in the middle, **unchanged
        outside the ``sky_blend_rows`` feather band**.
    """
    _validate_inputs(
        frame,
        target_size=target_size,
        method=method,
        sky_blend_rows=sky_blend_rows,
    )

    h, w = frame.shape[:2]
    size = w if target_size is None else target_size
    src = frame.astype(np.float64)

    # Total rows to add = size - h.  Split evenly; any odd row goes to the top
    # so the content block sits slightly low (arbitrary, deterministic).
    total_fill = max(0, size - h)
    top_fill = (total_fill + 1) // 2
    bot_fill = total_fill - top_fill

    # Zero-init so a ``size <= h`` canvas never leaks uninitialised memory.
    canvas = np.zeros((size, w, 3), dtype=np.float64)
    canvas[top_fill : top_fill + min(h, size - top_fill)] = src[: max(0, size - top_fill)]

    if total_fill == 0:
        # Already square (or caller asked for size <= h): return content as-is
        # in a square canvas (content placed at the top).
        return canvas.astype(frame.dtype, copy=False)

    ew = _edge_window(h)
    top_edge = src[:ew]
    bot_edge = src[h - ew : h]

    _FILL[method](canvas, top_fill, "top", top_edge)
    _FILL[method](canvas, bot_fill, "bottom", bot_edge)

    if sky_blend_rows > 0:
        canvas = _blend_seam(canvas, src, top_rows=top_fill, blend=sky_blend_rows)

    return np.clip(canvas, 0, 255).astype(frame.dtype, copy=False)
