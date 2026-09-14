"""G-8 (#355) — disocclusions must be filled from the background side.

The owner saw "an extra edge" around the drone: a band roughly as wide as the
local disparity where the two eyes disagree.  Its cause is that a *backward*
remap moves a silhouette's content but not the silhouette itself, so the band
carries foreground colour in one eye and background colour in the other.

These tests pin the fix: a forward z-buffered splat makes the band an explicit
hole, and every hole is filled from the **background side** of the contour.
"""

import numpy as np
import pytest

from pipeline.stereo_renderer import StereoRenderer

BG = (0, 0, 255)  # far background — pure blue
FG = (255, 0, 0)  # near foreground — pure red
RECT = slice(80, 120)
H, W = 40, 200


def _scene() -> tuple[np.ndarray, np.ndarray]:
    """A near red rectangle on a far blue field (the card's synthetic case)."""
    frame = np.zeros((H, W, 3), np.uint8)
    frame[:, :] = BG
    frame[:, RECT] = FG
    depth = np.full((H, W), 0.9, np.float32)
    depth[:, RECT] = 0.02
    return frame, depth


def _renderer(**kw) -> StereoRenderer:
    opts = {"temporal_smooth": False, "edge_align": False, "max_disparity": 0.05}
    opts.update(kw)
    return StereoRenderer(**opts)


def test_disocclusion_is_filled_from_the_background_side():
    """The revealed band takes background colour, never a foreground smear.

    With ``max_disparity=0.05`` on a 200px-wide frame the foreground disparity
    is +6px and the background clips to -10px, so in the left eye the rectangle
    lands at columns 74..113 and nothing reaches columns 114..129.  That band is
    the disocclusion: its left neighbour is foreground (disparity +6), its right
    neighbour background (disparity -10).  The fill must choose the *smaller*
    disparity — the background.
    """
    frame, depth = _scene()
    renderer = _renderer()

    # The band is derived from the renderer's own disparity, not guessed.
    disparity = renderer._compute_disparity(depth, 1.0)
    fg_disp, bg_disp = disparity[0, 100], disparity[0, 0]
    assert fg_disp > bg_disp, "foreground must be nearer than background"

    left, _ = renderer.render(frame, depth)
    hole = left[:, int(RECT.stop - fg_disp) : int(RECT.stop - bg_disp)]
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


def test_both_eyes_agree_outside_the_contour():
    """The two eyes must not disagree about what the band contains.

    Matching the left eye's band against the right eye's at the correct offset
    is the synthetic analogue of the lead's block-matching measurement.
    """
    frame, depth = _scene()
    left, right = _renderer().render(frame, depth)
    # Background shifts by -10px in the left eye and +10px in the right, so the
    # far field is the same content 20px apart.
    band_l = left[:, 140:180].astype(int)
    band_r = right[:, 160:200].astype(int)
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
    by the filter radius (1% of the short side), which is the point: it fixes
    the few-pixel depth-map misalignment that fringes a contour, and leaves
    genuinely distant structure alone.
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
def test_output_shape_and_dtype_are_preserved(occlusion_aware):
    frame, depth = _scene()
    left, right = _renderer(occlusion_aware=occlusion_aware).render(frame, depth)

    for view in (left, right):
        assert view.shape == frame.shape
        assert view.dtype == np.uint8
