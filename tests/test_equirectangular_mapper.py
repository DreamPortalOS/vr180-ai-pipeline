"""Tests for :mod:`pipeline.equirectangular_mapper`.

Prior to issue #240 this module had **zero** tests, which is how a docstring
claiming "filled with black (not stretched)" survived while the default ffmpeg
path actually did neither — ffmpeg's v360 clamps edge pixels into the
periphery. These tests run the *real* ffmpeg v360 mapping on a tiny synthetic
frame and assert issue #255's contract: out-of-FOV pixels are **pure black RGB
on the default path**, because alpha does not survive the ``yuv420p`` encode
downstream.

**The fixture frame matters.** Issue #240's tests used a frame with a *black*
border, so v360's edge-clamped smear was itself black and every "the hole is
black" assertion passed vacuously — which is exactly how PR #254 shipped green
CI while the operator still saw a cream band on the real render. The frame
below therefore has a **bright cream border** (the very RGB the operator
reported), so a smeared hole is loudly distinguishable from a black one.

The ffmpeg path is exercised only when ffmpeg + the v360 filter are present
(skipped otherwise, so CI stays green on a minimal runner). The OpenCV fallback
is always exercised.
"""

from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pytest

from pipeline.equirectangular_mapper import EquirectangularMapper

# Small, deterministic synthetic frame: a solid red patch dead-centre on the
# cream background the operator reported smearing into the periphery
# (RGB 237,218,193). Both colours are far from black, so *any* black pixel in
# the output can only have come from the fix.
_H, _W = 36, 64
_CREAM = (237, 218, 193)
_FRAME = np.full((_H, _W, 3), _CREAM, dtype=np.uint8)
_FRAME[12:24, 22:42] = (255, 0, 0)  # red patch


def _black_fraction(rgb: np.ndarray) -> float:
    """Fraction of pixels that are exactly RGB (0,0,0)."""
    return float((rgb[:, :, :3].sum(axis=2) == 0).mean())


def test_fixture_frame_has_no_black_pixels():
    """Guard the guard: if the fixture ever regains a black border, every
    "hole is black" assertion below silently becomes vacuous (the #254 trap).
    """
    assert _black_fraction(_FRAME) == 0.0


def _ffmpeg_v360_available() -> bool:
    if not shutil.which("ffmpeg"):
        return False
    try:
        out = subprocess.run(["ffmpeg", "-filters"], capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return False
    return "v360" in out


_HAS_V360 = _ffmpeg_v360_available()
_FFMPEG = pytest.mark.skipif(not _HAS_V360, reason="ffmpeg v360 unavailable")


# --------------------------------------------------------------------------- #
# ffmpeg v360 path (the default) — issue #240 regression
# --------------------------------------------------------------------------- #


@_FFMPEG
class TestFfmpegAlphaMask:
    """v360 must request the alpha mask *and* composite it onto black."""

    def _mapper(self) -> EquirectangularMapper:
        return EquirectangularMapper(output_width=_W, output_height=_H, src_hfov=90.0)

    def test_filter_string_contains_alpha_mask(self):
        # The filter we hand to ffmpeg must request the alpha plane. If a
        # future edit drops ``alpha_mask=1`` this assertion turns red — that
        # is the regression this card guards.
        m = self._mapper()
        flt = m._v360_filter(_W, _H)
        assert "alpha_mask=1" in flt

    def test_filter_string_composites_onto_black(self):
        # issue #255: the alpha mask alone is inert once the frame is encoded
        # to yuv420p. The chain must also flatten the hole onto black *inside*
        # ffmpeg, and must terminate in an opaque pixel format by default.
        flt = self._mapper()._v360_filter(_W, _H)
        assert "overlay" in flt
        assert flt.endswith("format=rgb24")

    def test_filter_string_keeps_alpha_when_requested(self):
        flt = self._mapper()._v360_filter(_W, _H, with_alpha=True)
        assert flt.endswith("format=rgba")

    def test_output_is_rgba_when_alpha_requested(self):
        rgba = self._mapper().map_single(_FRAME, with_alpha=True)
        assert rgba.ndim == 3 and rgba.shape[2] == 4

    def test_default_returns_three_channels(self):
        # Downstream consumers expect the historical 3-channel RGB contract.
        rgb = self._mapper().map_single(_FRAME)
        assert rgb.ndim == 3 and rgb.shape[2] == 3

    def test_corners_are_alpha_zero(self):
        rgba = self._mapper().map_single(_FRAME, with_alpha=True)
        a = rgba[:, :, 3]
        for corner in (a[0, 0], a[0, -1], a[-1, 0], a[-1, -1]):
            assert corner == 0

    def test_there_is_an_uncovered_region(self):
        # The bug: before alpha_mask, *every* pixel was "covered" by a smeared
        # edge. After the fix a real alpha==0 hole must exist (ratio > 0).
        # If this assertion is ever green-without-the-fix, the fix regressed.
        rgba = self._mapper().map_single(_FRAME, with_alpha=True)
        hole_ratio = float((rgba[:, :, 3] == 0).mean())
        assert hole_ratio > 0.0, f"expected alpha==0 region, got ratio {hole_ratio}"

    def test_content_pixels_unchanged_with_or_without_alpha(self):
        m = self._mapper()
        rgb = m.map_single(_FRAME)
        rgba = m.map_single(_FRAME, with_alpha=True)
        # The RGB planes must be bit-identical regardless of the alpha flag —
        # alpha_mask only adds a plane, it does not resample content.
        assert np.array_equal(rgb, rgba[:, :, :3])

    def test_content_pixel_value_preserved(self):
        # The bright red centre must survive the mapping untouched (the patch
        # is well inside the 90° FOV, so v360 samples it directly).
        rgba = self._mapper().map_single(_FRAME, with_alpha=True)
        cy, cx = _H // 2, _W // 2
        assert tuple(rgba[cy, cx, :3]) == (255, 0, 0)
        assert rgba[cy, cx, 3] == 255

    def test_hole_rgb_is_black(self):
        # Where the alpha mask says "no coverage", the RGB must be zero so a
        # compositor that ignores alpha sees pure black (no cream smear).
        rgba = self._mapper().map_single(_FRAME, with_alpha=True)
        hole = rgba[rgba[:, :, 3] == 0][:, :3]
        assert hole.shape[0] > 0
        assert int(hole.max()) == 0

    # -- issue #255: the DEFAULT (with_alpha=False) path is what ships ------ #

    def test_default_corners_are_black(self):
        # The headline acceptance criterion. Before #255 these corners were
        # RGB(237,218,193) — v360's clamped copy of the source's edge pixel.
        rgb = self._mapper().map_single(_FRAME)
        for r, c in [(0, 0), (0, -1), (-1, 0), (-1, -1)]:
            assert tuple(int(x) for x in rgb[r, c]) == (0, 0, 0), f"corner ({r},{c}) = {rgb[r, c]}, expected black"

    def test_default_has_substantial_black_region(self):
        # A 90° source in a 180° dome leaves a large uncovered periphery. If
        # this drops to ~0 the smear is back.
        rgb = self._mapper().map_single(_FRAME)
        assert _black_fraction(rgb) > 0.2

    def test_default_does_not_blacken_inside_the_fov(self):
        # The fix must only touch the hole. Assert the centre region — well
        # inside the 90° FOV — survives, so an over-broad mask can't pass by
        # blackening the whole frame.
        rgb = self._mapper().map_single(_FRAME)
        centre = rgb[_H // 2 - 4 : _H // 2 + 4, _W // 2 - 6 : _W // 2 + 6]
        assert _black_fraction(centre) == 0.0
        assert int(centre.max()) > 0

    def test_default_black_region_matches_the_alpha_hole(self):
        # The composited black must be exactly the alpha==0 set: no more
        # (content eaten), no less (smear left behind).
        m = self._mapper()
        rgb = m.map_single(_FRAME)
        alpha = m.map_single(_FRAME, with_alpha=True)[:, :, 3]
        assert np.array_equal(rgb.sum(axis=2) == 0, alpha == 0)

    def test_stereo_pair_corners_are_black(self):
        # map_stereo_pair shares map_single's path, so both eyes must agree.
        sbs = self._mapper().map_stereo_pair(_FRAME, _FRAME)
        assert sbs.shape == (_H, _W * 2, 3)
        for r, c in [(0, 0), (0, _W - 1), (0, _W), (0, -1), (-1, 0), (-1, -1)]:
            assert tuple(int(x) for x in sbs[r, c]) == (0, 0, 0)

    def test_sequence_batch_path_is_black(self, tmp_path):
        # The batched map_sequence path builds its own ffmpeg call from the
        # same _v360_filter; verify the composite survives the image-sequence
        # invocation (different rate/frame-count handling than a single PNG).
        m = self._mapper()
        frames = [_FRAME, _FRAME, _FRAME]
        out = m.map_sequence(frames, frames, str(tmp_path))
        assert len(out) == 3
        for sbs in out:
            assert tuple(int(x) for x in sbs[0, 0]) == (0, 0, 0)
            assert _black_fraction(sbs) > 0.2


# --------------------------------------------------------------------------- #
# OpenCV fallback path
# --------------------------------------------------------------------------- #


class TestOpenCvFallback:
    """Out-of-FOV pixels are pure black; alpha hole available on request."""

    def _mapper(self) -> EquirectangularMapper:
        return EquirectangularMapper(output_width=_W, output_height=_H, src_hfov=90.0, use_ffmpeg=False)

    def test_default_three_channels(self):
        rgb = self._mapper().map_single(_FRAME)
        assert rgb.ndim == 3 and rgb.shape[2] == 3

    def test_with_alpha_returns_rgba(self):
        rgba = self._mapper().map_single(_FRAME, with_alpha=True)
        assert rgba.ndim == 3 and rgba.shape[2] == 4

    def test_corners_alpha_zero_and_rgb_black(self):
        rgba = self._mapper().map_single(_FRAME, with_alpha=True)
        for r, c in [(0, 0), (0, -1), (-1, 0), (-1, -1)]:
            assert rgba[r, c, 3] == 0
            assert tuple(rgba[r, c, :3]) == (0, 0, 0)

    def test_there_is_an_uncovered_region(self):
        rgba = self._mapper().map_single(_FRAME, with_alpha=True)
        hole_ratio = float((rgba[:, :, 3] == 0).mean())
        assert hole_ratio > 0.0

    def test_content_pixel_alpha_full(self):
        rgba = self._mapper().map_single(_FRAME, with_alpha=True)
        cy, cx = _H // 2, _W // 2
        assert rgba[cy, cx, 3] == 255

    def test_rgb_identical_with_or_without_alpha(self):
        m = self._mapper()
        rgb = m.map_single(_FRAME)
        rgba = m.map_single(_FRAME, with_alpha=True)
        assert np.array_equal(rgb, rgba[:, :, :3])

    def test_default_corners_are_black(self):
        # Same #255 contract as the ffmpeg path — the two must not diverge.
        rgb = self._mapper().map_single(_FRAME)
        for r, c in [(0, 0), (0, -1), (-1, 0), (-1, -1)]:
            assert tuple(int(x) for x in rgb[r, c]) == (0, 0, 0)

    def test_default_does_not_blacken_inside_the_fov(self):
        rgb = self._mapper().map_single(_FRAME)
        centre = rgb[_H // 2 - 4 : _H // 2 + 4, _W // 2 - 6 : _W // 2 + 6]
        assert _black_fraction(centre) == 0.0


# --------------------------------------------------------------------------- #
# SBS pairing still honours the 3-channel contract
# --------------------------------------------------------------------------- #


class TestSbsContract:
    def test_sbs_is_three_channels_and_double_width(self):
        m = EquirectangularMapper(output_width=_W, output_height=_H, src_hfov=90.0, use_ffmpeg=False)
        sbs = m.map_stereo_pair(_FRAME, _FRAME)
        assert sbs.shape == (_H, _W * 2, 3)
