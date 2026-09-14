"""G-8 (#355) — the two eyes must agree around a depth discontinuity.

The owner saw two things on the headset: an opaque drone that looked locally
*see-through*, and rock faces that *flowed* while the camera pushed forward.
Both eased when the player UI pushed the image back — i.e. both scale with
parallax, so neither is a content bug: the eyes simply do not match.

Two mechanisms produce that mismatch, and this module pins both:

1. A depth cliff of Δ px tears a Δ-wide band that exists in neither eye.  The
   renderer forward-warps with a z-buffer so the band becomes an explicit
   hole, and fills every hole from the **background side** of the contour.
2. A depth *slope* ``d'`` renders the same surface ``(1+d')/(1-d')`` times
   wider in one eye than the other — the canyon wall.  The gradient limiter
   caps that ratio.

Both depend on one sign: in this pipeline the depth backends emit inverse
depth (larger = nearer), so ``_compute_disparity`` returns *smaller = nearer*.
``test_near_object_renders_with_crossed_disparity`` pins it, because getting it
backwards silently turns the fix into the artefact.
"""

import numpy as np
import pytest

from pipeline.stereo_renderer import StereoRenderer

BG = (0, 0, 255)  # far background — pure blue
FG = (255, 0, 0)  # near foreground — pure red
RECT = slice(80, 120)
H, W = 40, 200


def _scene() -> tuple[np.ndarray, np.ndarray]:
    """A near red rectangle on a far blue field (the card's synthetic case).

    Depth follows the pipeline's convention — the backends emit inverse depth,
    so the *near* rectangle carries the *larger* value.
    """
    frame = np.zeros((H, W, 3), np.uint8)
    frame[:, :] = BG
    frame[:, RECT] = FG
    depth = np.full((H, W), 0.05, np.float32)  # far field
    depth[:, RECT] = 0.95  # near rectangle
    return frame, depth


def _renderer(**kw) -> StereoRenderer:
    opts = {
        "temporal_smooth": False,
        "edge_align": False,
        "gradient_limit": None,
        "max_disparity": 0.05,
    }
    opts.update(kw)
    return StereoRenderer(**opts)


def test_near_object_renders_with_crossed_disparity():
    """The near rectangle must sit *in front of* the screen plane.

    Crossed disparity means the object lands further right in the left eye than
    in the right eye.  This is the sign every other test here depends on: flip
    it and the z-buffer lets the background paint over the object while the
    fill smears foreground into the hole — precisely the "opaque body looks
    see-through" report.
    """
    frame, depth = _scene()
    left, right = _renderer().render(frame, depth)

    row = H // 2
    left_red = np.flatnonzero(left[row, :, 0] > 128)
    right_red = np.flatnonzero(right[row, :, 0] > 128)
    assert left_red.size and right_red.size

    offset = right_red.mean() - left_red.mean()
    assert offset < 0, f"near object rendered uncrossed (behind the screen): {offset=:.1f}"


def test_disocclusion_is_filled_from_the_background_side():
    """The revealed band takes background colour, never a foreground smear.

    With ``max_disparity=0.05`` on a 200px-wide frame the near rectangle clips
    to -10px and the far field sits at +6px.  In the left eye (``x - d``) the
    rectangle lands 10px right and the background 6px left, so a 16px band just
    left of the rectangle is uncovered.  Its two neighbours are foreground
    (-10) and background (+6); the fill must take the *farther* one.
    """
    frame, depth = _scene()
    renderer = _renderer()

    # The band is derived from the renderer's own disparity, not guessed.
    disparity = renderer._compute_disparity(depth, 1.0)
    fg_disp, bg_disp = disparity[0, 100], disparity[0, 0]
    assert fg_disp < bg_disp, "foreground must be nearer (smaller disparity) than background"

    left, _ = renderer.render(frame, depth)
    hole = left[:, int(RECT.start - bg_disp) : int(RECT.start - fg_disp)]
    assert hole.size > 0

    # Closer to background blue than to foreground red, by a wide margin.
    to_bg = np.abs(hole.astype(int) - np.array(BG)).mean()
    to_fg = np.abs(hole.astype(int) - np.array(FG)).mean()
    assert to_bg < to_fg, f"disocclusion took the foreground side ({to_bg=:.1f} {to_fg=:.1f})"
    assert to_bg < 8.0, f"fill is not clean background ({to_bg=:.1f})"


def test_foreground_silhouette_does_not_grow():
    """A foreground-side fill would inflate the object; the width must hold.

    This is the guard against the "just blur it" pseudo-fix: smearing the
    foreground into the hole widens the red region beyond its source width.
    """
    frame, depth = _scene()
    left, right = _renderer().render(frame, depth)
    source_width = RECT.stop - RECT.start

    for name, view in (("left", left), ("right", right)):
        red_cols = np.flatnonzero(view[H // 2, :, 0] > 128)
        width = red_cols.max() - red_cols.min() + 1
        assert width <= source_width, f"{name} eye foreground grew {width} > {source_width}"


def test_nearer_surface_wins_the_z_fight():
    """Where both surfaces land on one pixel, the object survives — not the wall.

    An inverted z-buffer loses this: the far field paints over the rectangle
    and punches holes *inside* an opaque body.
    """
    frame, depth = _scene()
    left, right = _renderer().render(frame, depth)
    source_width = RECT.stop - RECT.start

    for name, view in (("left", left), ("right", right)):
        red = int((view[H // 2, :, 0] > 128).sum())
        assert red == source_width, f"{name} eye lost {source_width - red} px of the near object"


def test_both_eyes_agree_outside_the_contour():
    """The two eyes must not disagree about what the far field contains.

    Matching the left eye's band against the right eye's at the correct offset
    is the synthetic analogue of the lead's block-matching measurement.
    """
    frame, depth = _scene()
    left, right = _renderer().render(frame, depth)
    # The far field shifts -6px in the left eye and +6px in the right, so the
    # same content is 12px apart.
    band_l = left[:, 150:180].astype(int)
    band_r = right[:, 162:192].astype(int)
    assert np.abs(band_l - band_r).mean() < 1.0


def test_fill_ratio_is_reported():
    """The diagnostic field records what fraction of pixels were holes."""
    frame, depth = _scene()
    renderer = _renderer()
    renderer.render(frame, depth)

    ratio = renderer.last_fill_ratio
    assert set(ratio) == {"left", "right"}
    assert all(0.0 < v < 0.5 for v in ratio.values()), ratio


def test_uniform_depth_translates_exactly_with_only_border_holes():
    """A flat depth map shifts everything equally — no *interior* hole at all.

    The whole frame moves by one constant disparity (+6px here), so the only
    uncovered pixels are the 6-column strip walking off one edge.  The interior
    must be a bit-exact translate: no z-buffer dropouts, no resampling blur.
    """
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)
    depth = np.full((H, W), 0.5, np.float32)

    renderer = _renderer()
    shift = int(renderer._compute_disparity(depth, 1.0)[0, 0])
    left, right = renderer.render(frame, depth)

    assert np.array_equal(left[:, : W - shift], frame[:, shift:])
    assert np.array_equal(right[:, shift:], frame[:, : W - shift])
    assert renderer.last_fill_ratio == {"left": shift / W, "right": shift / W}


def test_edge_align_snaps_disparity_onto_the_image_edge():
    """A depth silhouette fatter than the object is pulled back to the edge.

    The image edge sits at column 120; the depth edge is deliberately 2px "fat"
    at column 122.  Guided filtering with the image as guide snaps the
    disparity transition back onto the image edge.  The correction is bounded
    by the filter radius (2% of the short side), which is the point: it fixes
    the depth-map misalignment that fringes a contour, and leaves genuinely
    distant structure alone.
    """
    size, img_edge, depth_edge = 500, 120, 122
    frame = np.zeros((size, size, 3), np.uint8)
    frame[:, :] = BG
    frame[:, img_edge:] = FG
    disparity = np.zeros((size, size), np.float32)
    disparity[:, depth_edge:] = 10.0

    aligned = StereoRenderer(temporal_smooth=False)._align_disparity_edges(disparity, frame)

    crossing = int(np.argmax(aligned[size // 2] > 5.0))
    assert crossing == img_edge, f"edge not snapped onto the image edge (at {crossing})"


# ---------------------------------------------------------------------------
# Gradient limiting (G-8, #355) — the canyon-wall half of the report
# ---------------------------------------------------------------------------


def test_gradient_limit_bounds_the_horizontal_slope():
    """A cliff becomes a ramp of exactly the requested slope."""
    disparity = np.zeros((4, 120), np.float32)
    disparity[:, 60:] = -24.0  # a 24px cliff into the foreground

    for limit in (2.0, 1.0, 0.5):
        out = StereoRenderer._limit_disparity_gradient(disparity, limit)
        slope = np.abs(np.diff(out, axis=1)).max()
        assert slope <= limit + 1e-4, f"{limit=} left a slope of {slope:.3f}"


def test_gradient_limit_preserves_the_foreground_and_only_touches_its_neighbourhood():
    """It pulls background forward; it never pushes the object back.

    The object keeps its full pop-out (that is the 3D the viewer came for), and
    everything more than ``Δ/limit`` px from the contour is bit-exact — so this
    is a local correction, not the "flatten everything until the ratio looks
    good" move the card rules out.
    """
    disparity = np.zeros((4, 120), np.float32)
    disparity[:, 60:] = -24.0

    out = StereoRenderer._limit_disparity_gradient(disparity, 1.0)

    assert out.min() == pytest.approx(disparity.min()), "foreground pop-out was eaten"
    assert (out <= disparity + 1e-6).all(), "limiter pushed a surface away from the viewer"
    # Δ=24 at limit 1.0 ⇒ a 24px ramp; column 35 is 25px clear of the contour.
    assert np.array_equal(out[:, :35], disparity[:, :35])


def test_gradient_limit_is_a_no_op_on_an_already_smooth_field():
    """Not a blur: a field that already satisfies the bound comes back bit-exact."""
    x = np.arange(200, dtype=np.float32)
    disparity = np.tile(0.1 * x, (4, 1))  # slope 0.1 everywhere

    out = StereoRenderer._limit_disparity_gradient(disparity, 1.0)

    assert np.array_equal(out, disparity)


def test_gradient_limit_shrinks_the_inter_eye_width_ratio_on_a_slanted_surface():
    """The canyon wall: a steep depth ramp renders two different widths.

    A wall whose disparity slides by ``d'`` per pixel comes out ``1 - d'`` wide
    in one eye and ``1 + d'`` in the other.  Tightening the limit must pull
    that ratio towards 1, monotonically — this is the knob the operator has if
    a slanted surface still refuses to fuse.
    """
    depth = np.tile(np.linspace(1.0, 0.0, 400, dtype=np.float32), (40, 1))
    renderer = _renderer(max_disparity=0.1)
    raw = renderer._compute_disparity(depth, 1.0)

    def width_ratio(limit):
        field = raw if limit is None else renderer._limit_disparity_gradient(raw, limit)
        slope = float(np.abs(np.diff(field, axis=1)).max())
        assert limit is None or slope <= limit + 1e-4, f"{limit=} left a slope of {slope:.3f}"
        return (1 + slope) / (1 - slope)

    assert width_ratio(0.01) < width_ratio(0.05) < width_ratio(None)


def test_gradient_limit_is_on_by_default_and_disablable():
    """Default behaviour is limited; ``None`` restores the unbounded field."""
    assert StereoRenderer().gradient_limit == 0.9

    frame, depth = _scene()
    off = StereoRenderer(temporal_smooth=False, edge_align=False, gradient_limit=None)
    on = StereoRenderer(temporal_smooth=False, edge_align=False, gradient_limit=0.9)

    assert not np.array_equal(on.render(frame, depth)[0], off.render(frame, depth)[0])


def test_the_default_gradient_limit_leaves_nothing_to_invent():
    """The point of ``limit < 1``: every output pixel is real, stretched content.

    Below 1 the per-eye mapping stays monotone *and* each destination column
    stays within a pixel of some source pixel, so the sub-pixel splat covers
    the interior outright — the invented band the owner reads as "the opaque
    drone is see-through" cannot form.  Unlimited, the same scene tears.
    """
    frame, depth = _scene()
    border = 12.0 / W  # the strip that legitimately walks off the frame edge

    limited = _renderer(gradient_limit=0.9)
    limited.render(frame, depth)
    assert all(v <= border for v in limited.last_fill_ratio.values()), limited.last_fill_ratio

    unlimited = _renderer(gradient_limit=None)
    unlimited.render(frame, depth)
    assert all(v > border for v in unlimited.last_fill_ratio.values()), unlimited.last_fill_ratio


def test_legacy_backward_remap_path_still_available():
    """``occlusion_aware=False`` keeps the pre-#355 behaviour for A/B work."""
    frame, depth = _scene()
    left, right = _renderer(occlusion_aware=False).render(frame, depth)

    assert left.shape == frame.shape
    assert right.shape == frame.shape
    # The legacy path is exactly what this card replaces: it smears the
    # foreground outward, so the red region is wider than its source.
    red_cols = np.flatnonzero(left[H // 2, :, 0] > 128)
    assert red_cols.size > 0


@pytest.mark.parametrize("occlusion_aware", [True, False])
@pytest.mark.parametrize("gradient_limit", [None, 1.0])
def test_output_shape_and_dtype_are_preserved(occlusion_aware, gradient_limit):
    frame, depth = _scene()
    left, right = _renderer(occlusion_aware=occlusion_aware, gradient_limit=gradient_limit).render(frame, depth)

    for view in (left, right):
        assert view.shape == frame.shape
        assert view.dtype == np.uint8
