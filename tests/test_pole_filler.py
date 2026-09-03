"""Tests for E-3 vertical pole-filler (issue #241).

Acceptance criteria from the card, exercised on small synthetic frames (no
models, no ffmpeg, no weights — pure numpy/OpenCV):

  1. output is square, edge = input width (or ``target_size``);
  2. the original content rows are byte-for-byte unchanged outside the
     feather band;
  3. ``gradient`` mode is monotonic down each column toward the pole and the
     pole row has ~zero row-variance (pole-convergence);
  4. no hard seam: across a vertical line, adjacent-row differences have no
     spike at the content↔synth boundary (threshold assertion);
  5. all three methods run and produce identical shapes;
  6. non-16:9 inputs don't crash — behaviour is fixed by the proportionality
     of the fill split.
"""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.pole_filler import extend_equirect_to_square

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _gradient_frame(h: int, w: int) -> np.ndarray:
    """A frame whose top rows are blue-ish (sky) and bottom green-ish (ground),
    with a horizontal gradient between — so pole extrapolation is meaningful.
    """
    top = np.array([180, 210, 240], dtype=np.float64)  # light blue
    bot = np.array([40, 120, 40], dtype=np.float64)  # green
    row_colors = np.linspace(0, 1, h)[:, None, None] * (bot - top)[None, None, :] + top[None, None, :]
    frame = np.broadcast_to(row_colors, (h, w, 3)).copy()
    # add a faint column-wise pattern so per-column means differ
    col_pattern = (np.sin(np.linspace(0, np.pi, w)) * 10)[None, :, None]
    frame = np.clip(frame + col_pattern, 0, 255)
    return frame.astype(np.uint8)


@pytest.fixture()
def frame_169() -> np.ndarray:
    # 160×90 → 16:9.  Fills 70 rows (35 top, 35 bottom) to reach 160×160.
    return _gradient_frame(90, 160)


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------


def test_output_is_square_default(frame_169: np.ndarray) -> None:
    out = extend_equirect_to_square(frame_169)
    assert out.shape == (160, 160, 3)


def test_output_is_square_target_size(frame_169: np.ndarray) -> None:
    # target_size overrides the edge length (height); width stays = input width
    # so original content pixels are never resampled (the "don't touch content"
    # invariant).  200 > 160 → adds 40 rows of synth above/below.
    out = extend_equirect_to_square(frame_169, target_size=200)
    assert out.shape == (200, 160, 3)


@pytest.mark.parametrize("method", ["gradient", "mirror", "solid"])
def test_all_methods_same_shape(frame_169: np.ndarray, method: str) -> None:
    out = extend_equirect_to_square(frame_169, method=method)
    assert out.shape == (160, 160, 3)


# ---------------------------------------------------------------------------
# Original content preserved (outside the feather band)
# ---------------------------------------------------------------------------


def test_original_content_unchanged_outside_feather(frame_169: np.ndarray) -> None:
    # sky_blend_rows default 32; original block = rows [35, 125).  The feather
    # touches the first/last 32 rows → the untouched core is rows [67, 93).
    out = extend_equirect_to_square(frame_169, method="gradient")
    top_fill = (160 - 90 + 1) // 2  # 35
    blend = 32
    core_start = top_fill + blend
    core_end = top_fill + 90 - blend
    orig_core = frame_169[core_start - top_fill : core_end - top_fill]
    out_core = out[core_start:core_end]
    assert np.array_equal(orig_core, out_core), "original content rows must be byte-identical outside feather"


def test_original_content_unchanged_zero_blend(frame_169: np.ndarray) -> None:
    # With no feather, the ENTIRE original block must be pixel-identical.
    out = extend_equirect_to_square(frame_169, method="solid", sky_blend_rows=0)
    top_fill = (160 - 90 + 1) // 2
    out_block = out[top_fill : top_fill + 90]
    assert np.array_equal(frame_169, out_block)


# ---------------------------------------------------------------------------
# Gradient mode: monotonic + pole convergence
# ---------------------------------------------------------------------------


def test_gradient_top_band_monotonic_per_column(frame_169: np.ndarray) -> None:
    """Each column's top-band luminance is monotonic from edge → pole."""
    out = extend_equirect_to_square(frame_169, method="gradient", sky_blend_rows=0)
    top_fill = (160 - 90 + 1) // 2  # 35
    band = out[:top_fill].astype(np.float64)
    lum = band.mean(axis=2)  # (35, W) per-column luminance
    # The band runs pole(row0) → edge(row top_fill-1).  Monotonic = no sign
    # flips in the first difference (allow equality for flat regions).
    diff = np.diff(lum, axis=0)
    # A column is monotonic if all non-zero diffs have the same sign.
    nonneg = np.all(diff >= 0, axis=0) | np.all(diff <= 0, axis=0)
    assert nonneg.all(), "every column must be monotonic edge→pole"


def test_gradient_pole_row_low_variance(frame_169: np.ndarray) -> None:
    """The pole (top) row converges to the global mean → ~zero row variance."""
    out = extend_equirect_to_square(frame_169, method="gradient", sky_blend_rows=0)
    pole_row = out[0].astype(np.float64)
    # row variance across columns for each channel
    var = pole_row.var(axis=0)
    # Original top edge rows had a sin pattern → real variance.  The pole must
    # be near-zero by construction (converges to global scalar mean).
    assert var.max() < 5.0, f"pole row must be ~flat (low variance), got {var}"


def test_gradient_bottom_pole_low_variance(frame_169: np.ndarray) -> None:
    out = extend_equirect_to_square(frame_169, method="gradient", sky_blend_rows=0)
    pole_row = out[-1].astype(np.float64)
    assert pole_row.var(axis=0).max() < 5.0


# ---------------------------------------------------------------------------
# No hard seam at the boundary
# ---------------------------------------------------------------------------


def test_no_hard_seam_top_boundary(frame_169: np.ndarray) -> None:
    """Across a vertical line, adjacent-row difference has no spike at the
    content↔synth boundary (within the feathered region)."""
    out = extend_equirect_to_square(frame_169, method="gradient").astype(np.float64)
    top_fill = (160 - 90 + 1) // 2  # 35
    # Look at column w/2, a few rows on each side of the boundary.
    col = out[:, 80, 0]
    diffs = np.abs(np.diff(col))
    # The boundary is at row top_fill.  The feather spans [top_fill, top_fill+32).
    # A "hard seam" would be a diff at the boundary much larger than the local
    # baseline.  Baseline = median diff in a calm interior region.
    interior_diffs = np.abs(np.diff(col[top_fill + 40 : top_fill + 80]))
    baseline = np.median(interior_diffs) + 1.0  # +1 to avoid zero baseline
    boundary_region_diffs = diffs[top_fill - 2 : top_fill + 34]
    # No single transition in the feathered boundary should exceed ~6× the
    # interior baseline (generous — a hard seam would be 10–50×).
    assert (boundary_region_diffs < 6 * baseline).all(), (
        f"hard seam detected: boundary diffs {boundary_region_diffs.max()} > {6 * baseline} (baseline {baseline})"
    )


def test_no_hard_seam_bottom_boundary(frame_169: np.ndarray) -> None:
    out = extend_equirect_to_square(frame_169, method="gradient").astype(np.float64)
    bot_start = (160 - 90 + 1) // 2 + 90  # = 125
    col = out[:, 80, 0]
    diffs = np.abs(np.diff(col))
    interior_diffs = np.abs(np.diff(col[bot_start - 80 : bot_start - 40]))
    baseline = np.median(interior_diffs) + 1.0
    boundary_region_diffs = diffs[bot_start - 34 : bot_start + 2]
    assert (boundary_region_diffs < 6 * baseline).all()


# ---------------------------------------------------------------------------
# Non-16:9 inputs
# ---------------------------------------------------------------------------


def test_square_input_returned_unchanged() -> None:
    frame = _gradient_frame(64, 64)
    out = extend_equirect_to_square(frame)
    assert out.shape == (64, 64, 3)
    assert np.array_equal(out, frame)


def test_non_169_ratio_does_not_crash() -> None:
    # 4:3 input (120×90) — fills 30 rows (15/15) to reach 120×120.
    frame = _gradient_frame(90, 120)
    out = extend_equirect_to_square(frame)
    assert out.shape == (120, 120, 3)


def test_tall_input_target_larger_than_height() -> None:
    # Caller can force a square smaller than the width but larger than the
    # height: only a small band is added.  100 > h=90 → 10 rows to fill.
    frame = _gradient_frame(90, 160)
    out = extend_equirect_to_square(frame, target_size=100)
    assert out.shape == (100, 160, 3)


def test_size_smaller_than_height_returns_canvas_with_content() -> None:
    # size < h: nothing to fill (total_fill clamps to 0).  The canvas is
    # size×w with content placed at the top; rows beyond h are zero-filled by
    # the module (no uninitialised memory leaks out).  Shape contract holds.
    frame = np.full((90, 64, 3), 100, dtype=np.uint8)
    out = extend_equirect_to_square(frame, target_size=40)
    assert out.shape == (40, 64, 3)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_invalid_method_raises() -> None:
    frame = _gradient_frame(16, 32)
    with pytest.raises(ValueError, match="method"):
        extend_equirect_to_square(frame, method="nope")


def test_invalid_frame_shape_raises() -> None:
    with pytest.raises(ValueError, match="HxWx3"):
        extend_equirect_to_square(np.zeros((10, 10), dtype=np.uint8))


def test_negative_sky_blend_raises() -> None:
    frame = _gradient_frame(16, 32)
    with pytest.raises(ValueError, match="sky_blend_rows"):
        extend_equirect_to_square(frame, sky_blend_rows=-1)
