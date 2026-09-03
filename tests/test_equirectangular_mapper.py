"""Tests for :mod:`pipeline.equirectangular_mapper`.

Prior to issue #240 this module had **zero** tests, which is how a docstring
claiming "filled with black (not stretched)" survived while the default ffmpeg
path actually did neither — ffmpeg's v360 clamps edge pixels into the
periphery. These tests run the *real* ffmpeg v360 mapping on a tiny synthetic
frame and assert the behaviour that issue #240 introduces: out-of-FOV pixels
are a genuine ``alpha=0`` hole, not a smear.

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

# Small, deterministic synthetic frame.  A solid red patch dead-centre on a
# black background: content that is bright and clearly *not* the fill colour,
# so "untouched content pixel" assertions are meaningful.
_H, _W = 36, 64
_FRAME = np.zeros((_H, _W, 3), dtype=np.uint8)
_FRAME[12:24, 22:42] = (255, 0, 0)  # red patch


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
    """The v360 filter must be built with ``alpha_mask=1`` (issue #240)."""

    def _mapper(self) -> EquirectangularMapper:
        return EquirectangularMapper(output_width=_W, output_height=_H, src_hfov=90.0)

    def test_filter_string_contains_alpha_mask(self):
        # The filter we hand to ffmpeg must request the alpha plane. If a
        # future edit drops ``alpha_mask=1`` this assertion turns red — that
        # is the regression this card guards.
        m = self._mapper()
        flt = m._v360_filter(_W, _H)
        assert "alpha_mask=1" in flt

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
        # compositor that ignores alpha sees pure black (no white smear).
        rgba = self._mapper().map_single(_FRAME, with_alpha=True)
        hole = rgba[rgba[:, :, 3] == 0][:, :3]
        assert hole.shape[0] > 0
        assert int(hole.max()) == 0


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


# --------------------------------------------------------------------------- #
# SBS pairing still honours the 3-channel contract
# --------------------------------------------------------------------------- #


class TestSbsContract:
    def test_sbs_is_three_channels_and_double_width(self):
        m = EquirectangularMapper(output_width=_W, output_height=_H, src_hfov=90.0, use_ffmpeg=False)
        sbs = m.map_stereo_pair(_FRAME, _FRAME)
        assert sbs.shape == (_H, _W * 2, 3)
