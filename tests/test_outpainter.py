"""Tests for pipeline/outpainter.py — 180° Outpaint Fill + Edge Feather (#244)."""

import contextlib
import logging
import math
import os
import shutil
import subprocess
import sys
import threading
import time

import cv2
import numpy as np
import pytest

from pipeline.equirectangular_mapper import EquirectangularMapper
from pipeline.outpainter import (
    _FEATHER_CACHE_MAXSIZE,
    _GEOMETRY_CACHE,
    _WEIGHTS_CACHE,
    DEFAULT_EDGE_FEATHER_END,
    DEFAULT_EDGE_FEATHER_START,
    AIOutpaintBackend,
    MockAIOutpaintBackend,
    Outpainter,
    _clear_feather_caches,
    _geometry_tables,
    _gradient_outpaint_single,
    _weights_cache_key,
    alpha_to_fill_mask,
    compute_edge_feather_weights,
    content_edge_angles,
    detect_black_boundary_mask,
    resolve_edge_feather,
    sbs_edge_feather_weights,
)

# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


def _make_frame(h: int = 192, w: int = 768) -> np.ndarray:
    """Create a synthetic RGB frame with content in the middle and black at top/bottom."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    # Fill middle third with content
    mid_start = h // 4
    mid_end = 3 * h // 4
    frame[mid_start:mid_end, :, :] = 128
    # Top and bottom stay black (0, 0, 0)
    return frame


def _all_black_frame(h: int = 192, w: int = 768) -> np.ndarray:
    """All-black frame — degenerate case."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def _no_black_frame(h: int = 192, w: int = 768) -> np.ndarray:
    """Frame with no black boundaries — all content."""
    return np.full((h, w, 3), 128, dtype=np.uint8)


# ---------------------------------------------------------------------------
#  Tests: detect_black_boundary_mask
# ---------------------------------------------------------------------------


class TestDetectBlackBoundaryMask:
    def test_detects_top_and_bottom_black(self):
        frame = _make_frame()
        mask = detect_black_boundary_mask(frame, threshold=10, top_ratio=0.3, bottom_ratio=0.3)
        h, _w = frame.shape[:2]
        # Top region should be masked
        assert np.all(mask[: h // 4, :] == 255), "Top black rows should be masked"
        # Middle should NOT be masked
        assert np.all(mask[h // 4 : 3 * h // 4, :] == 0), "Middle content rows should not be masked"
        # Bottom region should be masked
        assert np.all(mask[3 * h // 4 :, :] == 255), "Bottom black rows should be masked"

    def test_no_black_detected(self):
        frame = _no_black_frame()
        mask = detect_black_boundary_mask(frame, threshold=10)
        assert np.all(mask == 0), "No black boundaries should produce empty mask"

    def test_all_black_frame(self):
        frame = _all_black_frame()
        mask = detect_black_boundary_mask(frame, threshold=10)
        # #244: the scan follows the black band past the ratio window, so a frame
        # that is black all the way through is masked in full (degenerate case) …
        assert np.all(mask == 255), "An all-black frame has no content edge — everything is band"
        # … and the gradient filler then leaves it untouched (nothing to source from).
        result = Outpainter(mode="gradient").process([frame])
        assert np.array_equal(result[0], frame)

    def test_ratio_is_a_window_not_a_cap(self):
        """#244 root cause (OpenCV path): with the default 0.25 the scan ended
        *inside* a band taller than 25%, so the fill was sourced from a black row
        and ``--outpaint gradient`` changed nothing.  The scan now follows the band
        to the content edge whatever the ratio."""
        h, w = 192, 768
        frame = _make_frame(h, w)  # bands are 25% tall
        for ratio in (0.05, 0.25, 0.5):
            mask = detect_black_boundary_mask(frame, threshold=10, top_ratio=ratio, bottom_ratio=ratio)
            assert np.all(mask[: h // 4, :] == 255), f"ratio={ratio}: top band must be fully masked"
            assert np.all(mask[3 * h // 4 :, :] == 255), f"ratio={ratio}: bottom band must be fully masked"
            assert np.all(mask[h // 4 : 3 * h // 4, :] == 0), f"ratio={ratio}: content must stay unmasked"

    def test_dark_content_beyond_window_is_not_eaten(self):
        """Past the window only rows that are black in *every* pixel extend the band."""
        h, w = 192, 768
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[40:, :, :] = 3  # dim content: row mean far below the threshold …
        frame[40:, ::50, :] = 60  # … but not black in every pixel
        mask = detect_black_boundary_mask(frame, threshold=10, top_ratio=0.1, bottom_ratio=0.0)
        assert np.all(mask[:40] == 255), "true black band (rows 0-39) is masked in full"
        assert np.all(mask[40:] == 0), "dim-but-textured content beyond the window is kept"


# ---------------------------------------------------------------------------
#  Tests: _gradient_outpaint_single
# ---------------------------------------------------------------------------


class TestGradientOutpaintSingle:
    def test_basic_outpaint(self):
        h, w = 192, 768
        frame = _make_frame(h, w)
        mask = detect_black_boundary_mask(frame, threshold=10, top_ratio=0.3, bottom_ratio=0.3)

        result = _gradient_outpaint_single(frame, mask)

        # Output should be same shape and type
        assert result.shape == (h, w, 3)
        assert result.dtype == np.uint8

        # Masked regions should have non-zero content (no longer black)
        assert np.any(result[: h // 4, :, :] > 0), "Top masked region should be filled"
        assert np.any(result[3 * h // 4 :, :, :] > 0), "Bottom masked region should be filled"

        # Middle content should be preserved
        assert np.allclose(result[h // 4 : 3 * h // 4, :, :], frame[h // 4 : 3 * h // 4, :, :]), (
            "Middle content should be unchanged"
        )

    def test_no_mask(self):
        frame = _make_frame()
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        result = _gradient_outpaint_single(frame, mask)
        assert np.array_equal(result, frame), "No-op mask should return frame unchanged"

    def test_full_mask(self):
        frame = _all_black_frame()
        mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255
        result = _gradient_outpaint_single(frame, mask)
        assert result.shape == frame.shape, "Should not crash on degenerate case"


# ---------------------------------------------------------------------------
#  Tests: Outpainter class
# ---------------------------------------------------------------------------


class TestOutpainter:
    def test_mode_none_passthrough(self):
        frames = [_make_frame() for _ in range(3)]
        op = Outpainter(mode="none")
        result = op.process(frames)
        assert len(result) == len(frames)
        assert all(np.array_equal(r, f) for r, f in zip(result, frames, strict=True)), "None mode should passthrough"

    def test_mode_gradient(self):
        frames = [_make_frame() for _ in range(3)]
        op = Outpainter(mode="gradient")
        result = op.process(frames)

        assert len(result) == len(frames)
        for r, f in zip(result, frames, strict=True):
            assert r.shape == f.shape
            # Black boundaries should be filled (some pixel values > 0 in top rows)
            assert np.any(r[: f.shape[0] // 4, :, :] > 0), "Top boundary should be filled"

    def test_gradient_no_black(self):
        frames = [_no_black_frame() for _ in range(3)]
        op = Outpainter(mode="gradient")
        result = op.process(frames)
        assert len(result) == len(frames)
        assert all(np.array_equal(r, f) for r, f in zip(result, frames, strict=True)), (
            "No black boundaries -> frame unchanged"
        )

    def test_mode_ai_requires_backend(self):
        with pytest.raises(ValueError, match="requires an 'ai_backend' argument"):
            Outpainter(mode="ai")

    def test_mode_ai_with_mock_backend(self):
        backend = MockAIOutpaintBackend()
        frames = [_make_frame() for _ in range(3)]
        op = Outpainter(mode="ai", ai_backend=backend)
        result = op.process(frames)
        assert len(result) == len(frames)
        # Mock fills with green — verify at least some pixels changed
        assert np.any(result[0] != frames[0])

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="Unknown outpaint mode"):
            Outpainter(mode="invalid")

    def test_empty_frames(self):
        op = Outpainter(mode="gradient")
        result = op.process([])
        assert result == []

    def test_custom_threshold(self):
        """Very high threshold should treat all near-black as masked."""
        frame = _make_frame()
        # Frame content is 128, threshold > 128 means everything appears "black"
        mask = detect_black_boundary_mask(frame, threshold=200, top_ratio=0.5, bottom_ratio=0.5)
        # Top half should be masked
        assert np.all(mask[: frame.shape[0] // 2, :] == 255), "High threshold should mask more"


# ---------------------------------------------------------------------------
#  Tests: MockAIOutpaintBackend
# ---------------------------------------------------------------------------


class TestMockAIOutpaintBackend:
    def test_fills_with_green(self):
        backend = MockAIOutpaintBackend()
        h, w = 64, 256
        frame = _make_frame(h, w)
        mask = detect_black_boundary_mask(frame, threshold=10)

        result = backend.outpaint([frame], mask)
        assert len(result) == 1

        # Masked regions should be green (2D boolean mask broadcasts)
        mask_bool_2d = mask > 0
        assert np.all(result[0][mask_bool_2d] == [0, 255, 0]), "Mock should fill with green"

        # Non-masked regions should be unchanged
        assert np.all(result[0][~mask_bool_2d] == frame[~mask_bool_2d]), "Non-masked pixels should be unchanged"


# ---------------------------------------------------------------------------
#  Tests: AIOutpaintBackend ABC
# ---------------------------------------------------------------------------


class TestAIOutpaintBackendABC:
    def test_abc_cannot_instantiate(self):
        with pytest.raises(TypeError):
            AIOutpaintBackend()  # type: ignore


# ---------------------------------------------------------------------------
#  Tests: Outpainter property
# ---------------------------------------------------------------------------


class TestOutpainterProperty:
    def test_mode_property(self):
        op = Outpainter(mode="gradient")
        assert op.mode == "gradient"

    def test_mode_property_none(self):
        op = Outpainter(mode="none")
        assert op.mode == "none"


# ===========================================================================
#  Issue #244 — alpha-driven fill mask, angle-weighted edge feather,
#  ``--outpaint gradient`` effective on both mapper paths
# ===========================================================================

_EYE = 192  # per-eye size for the #244 fixtures (square hemisphere, tiny but real geometry)
_SRC_H, _SRC_W = 36, 64  # 16:9 synthetic source
_SRC_RGB = (200, 150, 100)


def _ffmpeg_v360_available() -> bool:
    if not shutil.which("ffmpeg"):
        return False
    try:
        out = subprocess.run(["ffmpeg", "-filters"], capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return False
    return "v360" in out


_FFMPEG = pytest.mark.skipif(not _ffmpeg_v360_available(), reason="ffmpeg v360 unavailable")


def _mapped_sbs(use_ffmpeg: bool, src_hfov: float, eye: int = _EYE) -> tuple[np.ndarray, np.ndarray]:
    """Real (tiny) equirect mapping of a solid 16:9 source → ``(sbs_rgb, sbs_alpha)``."""
    src = np.full((_SRC_H, _SRC_W, 3), _SRC_RGB, dtype=np.uint8)
    mapper = EquirectangularMapper(output_width=eye, output_height=eye, src_hfov=src_hfov, use_ffmpeg=use_ffmpeg)
    rgba = mapper.map_single(src, with_alpha=True)
    rgb, alpha = rgba[:, :, :3], rgba[:, :, 3]
    return np.concatenate([rgb, rgb], axis=1), np.concatenate([alpha, alpha], axis=1)


def _full_sbs(eye: int = _EYE, value: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """A source that fills the whole hemisphere: uniform content, alpha 255 everywhere."""
    sbs = np.full((eye, 2 * eye, 3), value, dtype=np.uint8)
    return sbs, np.full((eye, 2 * eye), 255, dtype=np.uint8)


def _changed_pixels(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(np.any(a != b, axis=2)))


def _equator_ray(frame_sbs: np.ndarray, eye: int) -> np.ndarray:
    """Brightness (channel 0) along the left eye's equator, centre → right edge."""
    return frame_sbs[eye // 2, eye // 2 : eye, 0].astype(int)


def _ray_index_at_fov(eye: int, fov_deg: float) -> int:
    """Index into :func:`_equator_ray` of the pixel whose centre sits at *fov_deg* (0–180 scale)."""
    col = round((fov_deg / 360.0 + 0.5) * eye - 0.5)
    return col - eye // 2


def _capture_warnings(logger_name: str, fn) -> list[str]:
    logger = logging.getLogger(logger_name)
    records: list[logging.LogRecord] = []

    class _Handler(logging.Handler):
        def emit(self, record):
            records.append(record)

    h = _Handler(level=logging.WARNING)
    logger.addHandler(h)
    try:
        fn()
    finally:
        logger.removeHandler(h)
    return [r.getMessage() for r in records]


# ---------------------------------------------------------------------------
#  resolve_edge_feather — defaults, enabling, validation
# ---------------------------------------------------------------------------


class TestResolveEdgeFeather:
    def test_off_by_default(self):
        assert resolve_edge_feather(None, None) is None
        assert Outpainter().edge_feather is None
        assert Outpainter(mode="gradient").edge_feather is None

    def test_single_bound_enables_with_documented_default(self):
        assert (DEFAULT_EDGE_FEATHER_START, DEFAULT_EDGE_FEATHER_END) == (165.0, 180.0)
        assert resolve_edge_feather(110, None) == (110.0, 180.0)
        assert resolve_edge_feather(None, 170) == (165.0, 170.0)
        assert Outpainter(edge_feather_start=165).edge_feather == (165.0, 180.0)

    @pytest.mark.parametrize(
        ("start", "end"),
        [(170, 165), (181, 180), (-1, 180), (0, 200), (float("nan"), 180), (165, float("inf"))],
    )
    def test_invalid_angles_raise(self, start, end):
        with pytest.raises(ValueError):
            resolve_edge_feather(start, end)
        with pytest.raises(ValueError):
            Outpainter(edge_feather_start=start, edge_feather_end=end)

    def test_start_equals_end_is_accepted_as_a_hard_cut(self):
        assert resolve_edge_feather(180, 180) == (180.0, 180.0)
        assert Outpainter(edge_feather_start=150, edge_feather_end=150).edge_feather == (150.0, 150.0)


# ---------------------------------------------------------------------------
#  Regression: not passing the feather args leaves the output byte-identical
# ---------------------------------------------------------------------------


class TestEdgeFeatherOffIsByteIdentical:
    def test_mode_none_is_passthrough_with_and_without_alpha(self):
        sbs, alpha = _mapped_sbs(use_ffmpeg=False, src_hfov=126.0)
        frames = [sbs, sbs.copy()]
        op = Outpainter(mode="none")
        assert op.process(frames) is frames
        assert op.process(frames, alpha=alpha) is frames

    def test_full_hemisphere_frame_untouched_without_feather(self):
        sbs, alpha = _full_sbs()
        out = Outpainter(mode="none").process([sbs], alpha=alpha)
        assert out[0] is sbs

    def test_gradient_output_identical_with_explicit_none_feather(self):
        sbs, alpha = _mapped_sbs(use_ffmpeg=False, src_hfov=126.0)
        plain = Outpainter(mode="gradient").process([sbs], alpha=alpha)[0]
        explicit = Outpainter(mode="gradient", edge_feather_start=None, edge_feather_end=None)
        assert np.array_equal(plain, explicit.process([sbs], alpha=alpha)[0])


# ---------------------------------------------------------------------------
#  Feather on a source that fills the hemisphere → fade at the 165°→180° rim
# ---------------------------------------------------------------------------


class TestEdgeFeatherFullHemisphere:
    def _fade(self, start, end, eye=_EYE):
        sbs, alpha = _full_sbs(eye)
        out = Outpainter(mode="none", edge_feather_start=start, edge_feather_end=end).process([sbs], alpha=alpha)[0]
        return sbs, out

    def test_default_165_180_ray_is_monotone_to_zero_without_steps(self):
        _, out = self._fade(165, 180)
        ray = _equator_ray(out, _EYE)
        assert ray[0] == 200, "centre must be untouched"
        assert ray[-1] == 0, "the rim must be fully black"
        assert np.all(np.diff(ray) <= 0), "brightness must be non-increasing from centre to rim"
        # No step: the 15° (FOV) ramp spans ~8 px at 192²/eye → ~25/px; a hard cut would be 200.
        assert (-np.diff(ray)).max() <= 40, f"largest step {(-np.diff(ray)).max()} looks like a hard edge"
        # The fade starts at 165°: 160° is still full brightness, 172° is already darker.
        assert ray[_ray_index_at_fov(_EYE, 160)] == 200
        assert 0 < ray[_ray_index_at_fov(_EYE, 172)] < 200

    def test_ramp_is_angle_weighted_not_pixel_linear(self):
        """Along the vertical centre line the very same 165°→180° ramp must appear
        (θ reaches 90° at the poles too) — the fade is a function of angle, and by
        symmetry of the hemisphere both axes give the same profile."""
        _, out = self._fade(165, 180)
        horiz = _equator_ray(out, _EYE)
        vert = out[_EYE // 2 :, _EYE // 2, 0].astype(int)  # centre column, centre → bottom
        assert np.abs(horiz - vert).max() <= 1

    def test_wide_feather_110_180(self):
        _, out = self._fade(110, 180)
        ray = _equator_ray(out, _EYE)
        assert ray[0] == 200 and ray[-1] == 0
        assert np.all(np.diff(ray) <= 0)
        assert ray[_ray_index_at_fov(_EYE, 100)] == 200, "inside 110° → untouched"
        assert 0 < ray[_ray_index_at_fov(_EYE, 150)] < 200, "150° sits in the wide ramp"
        assert (-np.diff(ray)).max() < 15, "a 70° ramp is far gentler than the default 15° one"

    def test_hard_cut_180_only_touches_the_border_ring(self):
        sbs, out = self._fade(180, 180)
        eye = _EYE
        m = eye // 8
        for x0 in (0, eye):  # both eyes
            assert np.array_equal(out[m:-m, x0 + m : x0 + eye - m], sbs[m:-m, x0 + m : x0 + eye - m])
            assert np.all(out[:, x0] == 0) and np.all(out[:, x0 + eye - 1] == 0)
        assert np.all(out[0] == 0) and np.all(out[-1] == 0)

    def test_end_below_180_is_black_beyond_end(self):
        _, out = self._fade(150, 160)
        ray = _equator_ray(out, _EYE)
        assert ray[_ray_index_at_fov(_EYE, 170)] == 0
        assert ray[_ray_index_at_fov(_EYE, 130)] == 200

    def test_no_alpha_means_fully_covered(self):
        sbs, alpha = _full_sbs()
        with_alpha = Outpainter(edge_feather_start=165).process([sbs], alpha=alpha)[0]
        without = Outpainter(edge_feather_start=165).process([sbs])[0]
        assert np.array_equal(with_alpha, without)
        assert _changed_pixels(without, sbs) > 0


# ---------------------------------------------------------------------------
#  Feather on the production geometry (--src-hfov 126, 16:9): anchored at the
#  alpha edge (~126° horizontally, ~96° vertically), not at the 180° rim
# ---------------------------------------------------------------------------


class TestEdgeFeatherPartialContent:
    @staticmethod
    def _expected_vfov(hfov: float) -> float:
        return math.degrees(2 * math.atan(math.tan(math.radians(hfov / 2)) * _SRC_H / _SRC_W))

    def _ray_inside_content(self, out: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        content = np.flatnonzero(alpha[_EYE // 2, _EYE // 2 : _EYE] > 0)
        return _equator_ray(out, _EYE)[: content[-1] + 1]

    def test_content_edge_angles_match_the_pinhole_geometry(self):
        _, alpha = _mapped_sbs(use_ffmpeg=False, src_hfov=126.0)
        edge = content_edge_angles(alpha[:, :_EYE])
        n = edge.shape[0]
        assert abs(edge[0] - 63.0) < 1.5, "ψ=0 (right): half of 126°"
        assert abs(edge[n // 4] - self._expected_vfov(126.0) / 2) < 1.5, "ψ=90° (up): half the pinhole vfov"
        assert edge.max() <= 90.0

    def test_content_edge_angles_full_coverage_is_the_rim(self):
        assert np.all(content_edge_angles(np.full((64, 64), 255, np.uint8)) == 90.0)

    def test_fade_meets_the_alpha_edge(self):
        sbs, alpha = _mapped_sbs(use_ffmpeg=False, src_hfov=126.0)
        out = Outpainter(edge_feather_start=165, edge_feather_end=180).process([sbs], alpha=alpha)[0]
        orig = int(sbs[_EYE // 2, _EYE // 2, 0])
        ray = self._ray_inside_content(out, alpha)
        assert ray[0] == orig, "centre untouched"
        assert ray[-1] == 0, "last content pixel before the hole must be black"
        assert np.all(np.diff(ray) <= 0), "monotone from centre to the content edge"
        assert (-np.diff(ray)).max() <= 0.2 * orig, "no step"
        assert 0 < ray[-4] < orig, "the ramp lives inside the content (anchored at ~126°, not 180°)"
        assert np.all(out[alpha == 0] == 0), "hole stays black"
        assert np.array_equal(out[:, :_EYE], out[:, _EYE:]), "both eyes treated alike"

    def test_feather_wider_than_content_half_angle_is_clamped(self):
        sbs, alpha = _mapped_sbs(use_ffmpeg=False, src_hfov=126.0)
        out = Outpainter(edge_feather_start=0, edge_feather_end=180).process([sbs], alpha=alpha)[0]
        orig = int(sbs[_EYE // 2, _EYE // 2, 0])
        ray = self._ray_inside_content(out, alpha)
        assert ray[0] >= 0.97 * orig, "ramp starts at the centre — never overshoots it"
        assert ray[-1] == 0
        assert np.all(np.diff(ray) <= 0)

    def test_wide_feather_on_partial_content(self):
        sbs, alpha = _mapped_sbs(use_ffmpeg=False, src_hfov=126.0)
        wide = Outpainter(edge_feather_start=110).process([sbs], alpha=alpha)[0]
        orig = int(sbs[_EYE // 2, _EYE // 2, 0])
        ray = self._ray_inside_content(wide, alpha)
        assert ray[0] == orig and ray[-1] == 0
        assert np.all(np.diff(ray) <= 0)
        default = Outpainter(edge_feather_start=165).process([sbs], alpha=alpha)[0]
        assert _changed_pixels(wide, sbs) > _changed_pixels(default, sbs)

    def test_alpha_shape_mismatch_raises(self):
        sbs, alpha = _mapped_sbs(use_ffmpeg=False, src_hfov=126.0)
        with pytest.raises(ValueError, match="alpha shape"):
            Outpainter(edge_feather_start=165).process([sbs], alpha=alpha[:, :_EYE])


# ---------------------------------------------------------------------------
#  ``--outpaint gradient`` must change pixels on BOTH mapper paths
# ---------------------------------------------------------------------------


class TestGradientChangesPixelsOnBothMapperPaths:
    """#244 root causes: gradient changed 0 pixels on both mapper paths.

    A 16:9 source at 90° hfov leaves a black band ~34% tall — taller than the
    default 25% scan window — so the pre-#244 black-row mask stopped *inside*
    the band and the filler smeared black over black.  Both ``_check`` cases
    (alpha-driven mask and black-row fallback) are RED on the pre-#244 code:
    the former because ``process()`` had no ``alpha`` argument, the latter
    because ``changed_pixels == 0``.
    """

    def _check(self, use_ffmpeg: bool) -> None:
        sbs, alpha = _mapped_sbs(use_ffmpeg=use_ffmpeg, src_hfov=90.0)
        op = Outpainter(mode="gradient")  # CLI defaults: threshold 10, ratios 0.25
        via_alpha = op.process([sbs], alpha=alpha)[0]
        via_black = op.process([sbs])[0]

        for out in (via_alpha, via_black):
            assert out.shape == sbs.shape and out.dtype == np.uint8
            assert _changed_pixels(out, sbs) > 0, "gradient must actually paint something"

        # Content is untouched: exactly (alpha path) / wherever the row scan saw content.
        assert np.array_equal(via_alpha[alpha > 0], sbs[alpha > 0])
        row_mask = detect_black_boundary_mask(sbs, threshold=10, top_ratio=0.25, bottom_ratio=0.25)
        assert np.array_equal(via_black[row_mask == 0], sbs[row_mask == 0])

        # The hole right above the content (centre column) is now painted on both.
        top = np.flatnonzero(alpha[:, _EYE // 2] > 0)[0]
        assert via_alpha[top - 1, _EYE // 2].max() > 0
        assert via_black[top - 1, _EYE // 2].max() > 0
        # Alpha also knows about the side holes → at least as much coverage as the row scan.
        assert _changed_pixels(via_alpha, sbs) >= _changed_pixels(via_black, sbs)

    def test_opencv_path(self):
        self._check(use_ffmpeg=False)

    @_FFMPEG
    def test_ffmpeg_path(self):
        self._check(use_ffmpeg=True)

    def test_alpha_to_fill_mask(self):
        alpha = np.array([[0, 255], [1, 0]], dtype=np.uint8)
        assert alpha_to_fill_mask(alpha).tolist() == [[255, 0], [0, 255]]
        rgba = np.zeros((2, 2, 4), dtype=np.uint8)
        rgba[0, 0, 3] = 255
        assert alpha_to_fill_mask(rgba).tolist() == [[0, 255], [255, 255]]

    def test_ai_backend_receives_the_alpha_hole(self):
        sbs, alpha = _mapped_sbs(use_ffmpeg=False, src_hfov=126.0)
        out = Outpainter(mode="ai", ai_backend=MockAIOutpaintBackend()).process([sbs], alpha=alpha)[0]
        assert np.all(out[alpha == 0] == [0, 255, 0])
        assert np.array_equal(out[alpha > 0], sbs[alpha > 0])

    def test_gradient_then_feather_fades_at_the_rim(self):
        """With a fill the hemisphere is fully covered, so the feather anchors at the
        physical rim and must not undo what the filler just painted."""
        sbs, alpha = _mapped_sbs(use_ffmpeg=False, src_hfov=126.0)
        out = Outpainter(mode="gradient", edge_feather_start=165).process([sbs], alpha=alpha)[0]
        assert np.all(out[:, 0] == 0) and np.all(out[:, _EYE - 1] == 0)
        edge_col = np.flatnonzero(alpha[_EYE // 2, _EYE // 2 : _EYE] > 0)[-1] + _EYE // 2
        assert out[_EYE // 2, edge_col + 2].max() > 0, "the smear just outside the content survives"


# ---------------------------------------------------------------------------
#  scripts/run_pipeline.py wiring: flags default off, alpha side-car, gating
# ---------------------------------------------------------------------------


class TestRunPipelineWiring:
    @pytest.fixture
    def rp(self):
        scripts = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
        sys.path.insert(0, scripts)
        try:
            import run_pipeline

            yield run_pipeline
        finally:
            sys.modules.pop("run_pipeline", None)
            with contextlib.suppress(ValueError):
                sys.path.remove(scripts)

    @staticmethod
    def _args(rp, tmp_path, *extra):
        base = ["--temp-dir", str(tmp_path), "--src-hfov", "126", "--output-width", "64", "--output-height", "64"]
        return rp.parse_args([*base, "--no-ffmpeg-v360", *extra])

    @staticmethod
    def _mapper():
        return EquirectangularMapper(output_width=64, output_height=64, src_hfov=126.0, use_ffmpeg=False)

    def test_flags_default_to_none(self, rp):
        args = rp.parse_args([])
        assert args.edge_feather_start is None and args.edge_feather_end is None
        args = rp.parse_args(["--edge-feather-start", "110"])
        assert args.edge_feather_start == 110.0 and args.edge_feather_end is None

    def test_invalid_feather_fails_fast_at_parse_time(self, rp, capsys):
        with pytest.raises(SystemExit):
            rp.parse_args(["--edge-feather-start", "170", "--edge-feather-end", "165"])
        assert "edge feather" in capsys.readouterr().err

    def test_outpaint_none_without_feather_is_a_no_op(self, rp, tmp_path):
        args = rp.parse_args(["--temp-dir", str(tmp_path)])
        frames = [_full_sbs(64)[0]]
        assert rp.run_outpaint_stage(args, frames) is frames
        assert not list(tmp_path.rglob("*.png"))

    def test_feather_runs_even_with_outpaint_none(self, rp, tmp_path):
        args = rp.parse_args(["--temp-dir", str(tmp_path), "--edge-feather-start", "165"])
        sbs = _full_sbs(64)[0]
        out = rp.run_outpaint_stage(args, [sbs])
        assert _changed_pixels(out[0], sbs) > 0 and np.all(out[0][:, -1] == 0)
        assert (tmp_path / "equirect" / "equirect_000000.png").exists()

    def test_alpha_sidecar_roundtrip(self, rp, tmp_path):
        args = self._args(rp, tmp_path)
        src = np.full((_SRC_H, _SRC_W, 3), _SRC_RGB, dtype=np.uint8)
        alpha = rp._save_equirect_alpha(args, self._mapper(), src)
        assert alpha.shape == (64, 128) and (alpha == 0).any() and (alpha > 0).any()
        path = rp._equirect_alpha_path(args)
        assert os.path.exists(path) and not path.endswith(".png"), "must not be picked up by the *.png frame globs"
        assert np.array_equal(rp._load_equirect_alpha(args, (64, 128, 3)), alpha)
        assert rp._load_equirect_alpha(args, (32, 64, 3)) is None, "a stale side-car of another size is ignored"

    def test_missing_sidecar_is_none(self, rp, tmp_path):
        assert rp._load_equirect_alpha(self._args(rp, tmp_path), (64, 128, 3)) is None

    def test_equirect_stage_writes_the_sidecar(self, rp, tmp_path):
        args = self._args(rp, tmp_path)
        src = np.full((_SRC_H, _SRC_W, 3), _SRC_RGB, dtype=np.uint8)
        sbs = rp.run_equirect_stage(args, [src], [src])
        assert len(sbs) == 1 and sbs[0].shape == (64, 128, 3)
        alpha = rp._load_equirect_alpha(args, sbs[0].shape)
        assert alpha is not None and np.all(sbs[0][alpha == 0] == 0)

    def test_outpaint_stage_uses_the_sidecar(self, rp, tmp_path):
        args = self._args(rp, tmp_path, "--edge-feather-start", "165")
        src = np.full((_SRC_H, _SRC_W, 3), _SRC_RGB, dtype=np.uint8)
        rgba = self._mapper().map_single(src, with_alpha=True)
        sbs = np.concatenate([rgba[:, :, :3]] * 2, axis=1)
        rp._save_equirect_alpha(args, self._mapper(), src)
        out = rp.run_outpaint_stage(args, [sbs])[0]
        alpha = rgba[:, :, 3]
        last = np.flatnonzero(alpha[32, 32:64] > 0)[-1] + 32
        assert out[32, last].max() == 0, "anchored at the content edge: last content pixel is black"
        assert out[32, 32].max() > 0, "centre survives"

    def test_streaming_no_longer_warns_that_feather_is_ignored(self, rp):
        """F-1 (#261): the streaming path now applies the feather per frame, so the
        swallowed-arg detector must not name the flags any more (was the inverse
        assertion when #244 shipped batch-only).  Full streaming coverage lives in
        ``tests/test_streaming_feather.py``."""
        args = rp.parse_args([])
        args.streaming, args.stage, args.edge_feather_start = True, "all", 165.0
        warned = _capture_warnings("vr180-pipeline", lambda: rp._warn_streaming_unsupported_args(args))
        assert not any("--edge-feather" in m for m in warned), warned


# ===========================================================================
#  Issue #271 — the vectorised gradient filler is byte-exact vs the pre-#271
#  row/column loops (shared by batch ``Outpainter`` and ``StreamingPipeline``)
# ===========================================================================


def _ref_smear_weights(distance, band):
    return np.clip(1.5 * (1.0 - distance / np.maximum(band, 1.0)), 0.0, 1.0)


def _ref_smear_columns(out, src_col, cols):
    band = float(max(abs(c - src_col) for c in cols))
    for c in cols:
        out[:, c] = out[:, src_col] * _ref_smear_weights(float(abs(c - src_col)), band)


def _reference_gradient_outpaint_single(frame, mask):
    """Pre-#271 ``pipeline.outpainter._gradient_outpaint_single`` (git e5fa036), verbatim.

    Kept here so the tests pin the *bytes* of the vectorised version against
    the original arithmetic (float32 frame, float64 vertical weights, float64
    scalar column weights, full-frame Gaussian, boolean restore).
    """
    mask_bool = mask > 0
    if not np.any(mask_bool):
        return frame.copy()
    content = ~mask_bool
    if not np.any(content):
        return frame.copy()

    h, w = mask_bool.shape
    cols = np.arange(w)
    out = frame.astype(np.float32)

    col_has = content.any(axis=0)
    first = np.where(col_has, content.argmax(axis=0), 0)
    last = np.where(col_has, h - 1 - content[::-1, :].argmax(axis=0), h - 1)
    src_first = out[first, cols]
    src_last = out[last, cols]
    band_top = first.astype(np.float32)
    band_bot = (h - 1 - last).astype(np.float32)

    for r in range(int(first[col_has].max())):
        d = first - r
        sel = col_has & (d > 0)
        if sel.any():
            out[r, sel] = src_first[sel] * _ref_smear_weights(d[sel], band_top[sel])[:, None]
    for r in range(int(last[col_has].min()) + 1, h):
        d = r - last
        sel = col_has & (d > 0)
        if sel.any():
            out[r, sel] = src_last[sel] * _ref_smear_weights(d[sel], band_bot[sel])[:, None]

    if not col_has.all():
        padded = np.concatenate([[True], col_has, [True]])
        run_starts = np.flatnonzero(padded[:-1] & ~padded[1:])
        run_ends = np.flatnonzero(~padded[:-1] & padded[1:]) - 1
        for a, b in zip(run_starts, run_ends, strict=True):
            if a == 0 and b == w - 1:
                continue
            if a == 0:
                _ref_smear_columns(out, src_col=b + 1, cols=range(a, b + 1))
            elif b == w - 1:
                _ref_smear_columns(out, src_col=a - 1, cols=range(a, b + 1))
            else:
                mid = (a + b) // 2
                _ref_smear_columns(out, src_col=a - 1, cols=range(a, mid + 1))
                _ref_smear_columns(out, src_col=b + 1, cols=range(mid + 1, b + 1))

    out_u8 = np.clip(np.rint(out), 0, 255).astype(np.uint8)
    blur_ksize = (1, max(3, h // 32 * 2 + 1))
    out_u8 = cv2.GaussianBlur(out_u8, blur_ksize, sigmaX=0, sigmaY=h / 16.0)
    out_u8[content] = frame[content]
    return out_u8


def _noise_frame(h, w, seed, channels=3):
    return np.random.default_rng(seed).integers(0, 256, size=(h, w, channels), dtype=np.uint8)


def _pinhole_hole_mask_sbs(eye, hfov_deg, aspect=16 / 9):
    """Analytic VR180 SBS fill mask of a pinhole source: a rounded-rectangle hole per eye.

    Same spherical convention as ``EquirectangularMapper`` (longitude across the
    eye width, colatitude down the height, evaluated at pixel centres); a
    direction is covered when it meets the source plane inside ±hfov/2 × ±vfov/2.
    """
    lon = ((np.arange(eye) + 0.5) / eye - 0.5) * np.pi
    colat = (np.arange(eye) + 0.5) / eye * np.pi
    sin_colat = np.sin(colat)[:, None]
    x = np.sin(lon)[None, :] * sin_colat
    y = np.broadcast_to(np.cos(colat)[:, None], (eye, eye))
    z = np.cos(lon)[None, :] * sin_colat
    tan_h = math.tan(math.radians(hfov_deg) / 2)
    tan_v = tan_h / aspect
    with np.errstate(divide="ignore", invalid="ignore"):
        covered = (z > 0) & (np.abs(x / z) <= tan_h) & (np.abs(y / z) <= tan_v)
    eye_mask = np.where(covered, 0, 255).astype(np.uint8)
    return np.concatenate([eye_mask, eye_mask], axis=1)


def _holed(frame, mask):
    """Black out the hole, as the mapper renders ``alpha == 0``."""
    out = frame.copy()
    out[mask > 0] = 0
    return out


def _rect_hole_mask(h, w, r0, r1, c0, c1):
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[r0:r1, c0:c1] = 255
    return mask


def _equivalence_cases():
    """``(name, frame, mask)`` — ≥3 mask shapes × several sizes, odd sizes, both mask dtypes, 3/4 channels."""
    cases = []
    # rectangular interior hole: content columns whose masked pixels sit *between* first/last
    m = _rect_hole_mask(256, 640, 80, 176, 200, 440)
    cases.append(("rect-hole", _holed(_noise_frame(256, 640, 1), m), m))
    # rectangular hole + top/bottom bands, odd size (interior masked pixels AND the vertical smear)
    m = _rect_hole_mask(193, 777, 60, 120, 300, 500)
    m[:23] = 255
    m[-31:] = 255
    cases.append(("rect-hole+bands-odd", _holed(_noise_frame(193, 777, 2), m), m))
    # black-row fallback mask (full-width bands)
    f = _make_frame(192, 768)
    cases.append(("black-row-bands", f, detect_black_boundary_mask(f, threshold=10, top_ratio=0.3, bottom_ratio=0.3)))
    # real mapper alpha hole (126° and 90° pinhole), noisy content
    for hfov in (126.0, 90.0):
        sbs, alpha = _mapped_sbs(use_ffmpeg=False, src_hfov=hfov)
        m = alpha_to_fill_mask(alpha)
        cases.append((f"mapper-{int(hfov)}", _holed(_noise_frame(*sbs.shape[:2], 3), m), m))
    # analytic 126° hole at an odd per-eye size spanning several 512-column strips
    m = _pinhole_hole_mask_sbs(901, 126.0)
    cases.append(("analytic-126-odd-multistrip", _holed(_noise_frame(901, 1802, 4), m), m))
    # bool mask dtype, hole touching the top border only
    m = _rect_hole_mask(128, 256, 0, 40, 0, 256) > 0
    cases.append(("bool-mask-top-band", _holed(_noise_frame(128, 256, 5), m), m))
    # full hemisphere — no hole at all
    sbs, alpha = _full_sbs()
    cases.append(("no-hole", sbs, alpha_to_fill_mask(alpha)))
    # hole everywhere — degenerate, nothing to source from
    cases.append(("all-hole", _noise_frame(64, 128, 6), np.full((64, 128), 255, np.uint8)))
    # four channels (the reference is channel-count agnostic)
    m = _pinhole_hole_mask_sbs(96, 126.0)
    cases.append(("rgba-126", _holed(_noise_frame(96, 192, 7, channels=4), m), m))
    # side holes only, content touching the top and bottom borders (no vertical smear at all)
    m = np.zeros((160, 900), np.uint8)
    m[:, :140] = 255
    m[:, 700:] = 255
    cases.append(("side-runs-only", _holed(_noise_frame(160, 900, 8), m), m))
    return cases


class TestGradientOutpaintVectorisedIsByteExact:
    @pytest.mark.parametrize("case", _equivalence_cases(), ids=lambda c: c[0])
    def test_matches_pre_271_reference(self, case):
        _name, frame, mask = case
        expected = _reference_gradient_outpaint_single(frame, mask)
        got = _gradient_outpaint_single(frame, mask)
        assert got.dtype == expected.dtype and got.shape == expected.shape
        assert np.array_equal(got, expected)
        assert np.array_equal(got[mask == 0], frame[mask == 0]), "content pixels are never touched"

    def test_input_frame_and_mask_are_not_modified(self):
        m = _pinhole_hole_mask_sbs(128, 126.0)
        frame = _holed(_noise_frame(128, 256, 9), m)
        frame_copy, mask_copy = frame.copy(), m.copy()
        out = _gradient_outpaint_single(frame, m)
        assert out is not frame and not np.shares_memory(out, frame)
        assert np.array_equal(frame, frame_copy) and np.array_equal(m, mask_copy)

    def test_batch_path_sees_the_same_bytes(self):
        """``Outpainter`` (and ``StreamingPipeline``) call ``_gradient_outpaint_single`` per frame."""
        sbs, alpha = _mapped_sbs(use_ffmpeg=False, src_hfov=126.0)
        mask = alpha_to_fill_mask(alpha)
        frame = _holed(_noise_frame(*sbs.shape[:2], 10), mask)
        via_class = Outpainter(mode="gradient").process([frame], alpha=alpha)[0]
        assert np.array_equal(via_class, _reference_gradient_outpaint_single(frame, mask))


# --- 5760×2880 (2880²/eye, ``--quality standard``): byte-exact and fast ------

_PROD_EYE = 2880
_IS_CI = bool(os.environ.get("CI"))
# Card: ≤ 0.25 s measured, asserted with head-room at 0.4 s.  GitHub-hosted
# runners have 2 vCPUs and OpenCV's Gaussian is multi-threaded, so the absolute
# budgets are relaxed there; the relative bound against the reference on the
# same machine holds everywhere.
_FULL_FRAME_BUDGET_S = 2.0 if _IS_CI else 0.4
_NO_HOLE_BUDGET_S = 0.2 if _IS_CI else 0.02


def _min_time(fn, repeats):
    best = float("inf")
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - t0)
    return best, result


@pytest.fixture(scope="module")
def production_frame_and_mask():
    mask = _pinhole_hole_mask_sbs(_PROD_EYE, 126.0)
    return _holed(_noise_frame(_PROD_EYE, 2 * _PROD_EYE, 271), mask), mask


class TestGradientOutpaintProductionSize:
    def test_5760x2880_is_byte_exact_and_at_least_3x_faster(self, production_frame_and_mask):
        frame, mask = production_frame_and_mask
        assert frame.shape == (2880, 5760, 3) and 0.6 < (mask > 0).mean() < 0.75, "the 126° hole covers ~68%"
        t_ref, expected = _min_time(lambda: _reference_gradient_outpaint_single(frame, mask), repeats=1)
        t_new, got = _min_time(lambda: _gradient_outpaint_single(frame, mask), repeats=3)
        assert np.array_equal(got, expected)
        assert t_new <= _FULL_FRAME_BUDGET_S, f"{t_new:.3f} s > {_FULL_FRAME_BUDGET_S} s budget"
        assert t_new * 3 <= t_ref, f"only {t_ref / t_new:.1f}× faster than the reference ({t_ref:.2f} s)"

    def test_no_hole_returns_fast(self, production_frame_and_mask):
        frame, _ = production_frame_and_mask
        no_hole = np.zeros(frame.shape[:2], np.uint8)
        t, out = _min_time(lambda: _gradient_outpaint_single(frame, no_hole), repeats=5)
        assert out is not frame and np.array_equal(out, frame)
        assert t <= _NO_HOLE_BUDGET_S, f"{t * 1000:.1f} ms > {_NO_HOLE_BUDGET_S * 1000:.0f} ms budget"


# ===========================================================================
#  Issue #264 — feather caches: byte-exact against the pre-#264 code, fast on
#  a hit, bounded, LRU, thread-safe.  The reference below is the previous
#  implementation copied verbatim (git 997bc51) so every assertion is against
#  the bytes the pipeline produced before this change.
# ===========================================================================


def _ref_hemisphere_pixel_angles(h, w):
    """Pre-#264 ``pipeline.outpainter._hemisphere_pixel_angles`` (git 997bc51), verbatim."""
    lon = ((np.arange(w, dtype=np.float64) + 0.5) / w - 0.5) * np.pi
    colat = (np.arange(h, dtype=np.float64) + 0.5) / h * np.pi
    sin_colat = np.sin(colat)[:, None]
    x = np.sin(lon)[None, :] * sin_colat
    y = np.broadcast_to(np.cos(colat)[:, None], (h, w))
    z = np.cos(lon)[None, :] * sin_colat
    theta = np.degrees(np.arccos(np.clip(z, -1.0, 1.0)))
    psi = np.mod(np.arctan2(y, x), 2.0 * np.pi)
    return theta.astype(np.float32), psi.astype(np.float32)


def _ref_content_edge_angles(alpha_eye, n_psi=1440):
    """Pre-#264 ``pipeline.outpainter.content_edge_angles`` (git 997bc51), verbatim."""
    h, w = alpha_eye.shape
    covered = alpha_eye > 0
    n_theta = max(h, w) + 1
    theta = np.linspace(0.0, np.pi / 2.0, n_theta)[:, None]
    psi = np.linspace(0.0, 2.0 * np.pi, n_psi, endpoint=False)[None, :]
    sin_t = np.sin(theta)
    x = sin_t * np.cos(psi)
    y = sin_t * np.sin(psi)
    z = np.broadcast_to(np.cos(theta), x.shape)
    lon = np.arctan2(x, z)
    colat = np.arccos(np.clip(y, -1.0, 1.0))
    u = np.clip(np.floor((lon / np.pi + 0.5) * w).astype(np.int64), 0, w - 1)
    v = np.clip(np.floor(colat / np.pi * h).astype(np.int64), 0, h - 1)
    hole = ~covered[v, u]
    hole[-1, :] = True
    first = hole.argmax(axis=0)
    return np.degrees(theta[first, 0]).astype(np.float32)


def _ref_compute_edge_feather_weights(alpha_eye, start_deg, end_deg):
    """Pre-#264 ``pipeline.outpainter.compute_edge_feather_weights`` (git 997bc51), verbatim."""
    if alpha_eye.ndim == 3:
        alpha_eye = alpha_eye[:, :, -1]
    h, w = alpha_eye.shape
    covered = alpha_eye > 0
    if not covered.any():
        return np.zeros((h, w), dtype=np.float32)

    theta_p, psi_p = _ref_hemisphere_pixel_angles(h, w)
    edge = _ref_content_edge_angles(alpha_eye)
    n = edge.shape[0]
    pos = psi_p.astype(np.float64) / (2.0 * np.pi) * n
    i0 = np.floor(pos).astype(np.int64) % n
    frac = (pos - np.floor(pos)).astype(np.float32)
    theta_edge = edge[i0] * (1.0 - frac) + edge[(i0 + 1) % n] * frac

    px_deg = 180.0 / h
    s, e = start_deg / 2.0, end_deg / 2.0
    e_eff = np.minimum(e, theta_edge) - px_deg
    width = np.maximum(np.minimum(e - s, e_eff), 1e-6)
    weights = np.clip((e_eff - theta_p) / width, 0.0, 1.0).astype(np.float32)

    inner = cv2.erode(
        covered.astype(np.uint8), np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0
    )
    weights[inner == 0] = 0.0
    return weights


def _fov_mask(h, w, half_angle_deg):
    """Coverage of a pinhole source: pixels within *half_angle_deg* of the forward axis."""
    theta, _ = _ref_hemisphere_pixel_angles(h, w)
    return (theta < half_angle_deg).astype(np.uint8) * 255


def _timed(fn):
    t0 = time.perf_counter()
    result = fn()
    return time.perf_counter() - t0, result


def _feather_cases():
    """(name, alpha, start, end) — sizes × masks × angles, incl. RGBA and a real mapper alpha."""
    rng = np.random.default_rng(264)
    rgba = np.zeros((64, 64, 4), np.uint8)
    rgba[..., 3] = _fov_mask(64, 64, 60.0)
    _, mapper_alpha = _mapped_sbs(use_ffmpeg=False, src_hfov=126.0)
    return [
        ("64x64_full_165_180", np.full((64, 64), 255, np.uint8), 165.0, 180.0),
        ("96x128_fov126_165_180", _fov_mask(96, 128, 63.0), 165.0, 180.0),
        ("128x96_fov90_110_180", _fov_mask(128, 96, 45.0), 110.0, 180.0),
        ("80x80_fov126_hard_cut_150", _fov_mask(80, 80, 63.0), 150.0, 150.0),
        ("72x72_ragged_noise", (rng.random((72, 72)) > 0.3).astype(np.uint8) * 255, 165.0, 180.0),
        ("64x64_rgba_165_180", rgba, 165.0, 180.0),
        ("mapper_eye_fov126_noncontiguous", mapper_alpha[:, :_EYE], 165.0, 180.0),
    ]


@pytest.fixture
def fresh_feather_caches():
    _clear_feather_caches()
    yield
    _clear_feather_caches()


class TestFeatherCacheIsByteExact:
    @pytest.mark.parametrize("case", _feather_cases(), ids=lambda c: c[0])
    def test_cold_and_warm_calls_match_pre_264_reference(self, case, fresh_feather_caches):
        _name, alpha, start, end = case
        expected = _ref_compute_edge_feather_weights(alpha, start, end)
        cold = compute_edge_feather_weights(alpha, start, end)  # geometry + weights both miss
        warm = compute_edge_feather_weights(alpha, start, end)  # weights hit
        for got in (cold, warm):
            assert got.dtype == expected.dtype and got.shape == expected.shape
            assert np.array_equal(got, expected)
        assert _WEIGHTS_CACHE.hits == 1 and len(_WEIGHTS_CACHE) == 1

    @pytest.mark.parametrize("case", _feather_cases(), ids=lambda c: c[0])
    def test_same_size_new_mask_reuses_geometry_and_matches_reference(self, case, fresh_feather_caches):
        """The SphereAccumulator case: the mask advances every frame, so only the
        size-keyed geometry tables can hit — the result must still be byte-exact."""
        _name, alpha, start, end = case
        compute_edge_feather_weights(alpha, start, end)  # warm the geometry tables
        plane = alpha[..., -1] if alpha.ndim == 3 else alpha
        other = np.ascontiguousarray(plane).copy()
        h, w = other.shape
        other[h // 3 : h // 2, w // 4 : w // 2] = 0  # punch a hole: new edge, new guard ring
        geo_hits, weights_hits = _GEOMETRY_CACHE.hits, _WEIGHTS_CACHE.hits
        got = compute_edge_feather_weights(other, start, end)
        assert _GEOMETRY_CACHE.hits == geo_hits + 1 and len(_GEOMETRY_CACHE) == 1
        assert _WEIGHTS_CACHE.hits == weights_hits, "a changed mask must never be served from the weights cache"
        assert np.array_equal(got, _ref_compute_edge_feather_weights(other, start, end))

    @pytest.mark.parametrize("n_psi", [1440, 360, 7])
    def test_content_edge_angles_matches_reference_for_any_ray_count(self, n_psi, fresh_feather_caches):
        alpha = _fov_mask(90, 120, 55.0)
        expected = _ref_content_edge_angles(alpha, n_psi)
        assert np.array_equal(content_edge_angles(alpha, n_psi), expected)
        assert np.array_equal(content_edge_angles(alpha, n_psi), expected), "from cached tables"
        assert [k[2] for k in _GEOMETRY_CACHE.ordered_keys()] == [n_psi]

    def test_sbs_and_outpainter_paths_see_the_same_bytes(self, fresh_feather_caches):
        sbs, alpha = _mapped_sbs(use_ffmpeg=False, src_hfov=126.0)
        w = alpha.shape[1] // 2
        expected = np.concatenate(
            [
                _ref_compute_edge_feather_weights(alpha[:, :w], 165.0, 180.0),
                _ref_compute_edge_feather_weights(alpha[:, w:], 165.0, 180.0),
            ],
            axis=1,
        )
        assert np.array_equal(sbs_edge_feather_weights(alpha, 165.0, 180.0), expected)
        assert np.array_equal(sbs_edge_feather_weights(alpha, 165.0, 180.0), expected)
        via_class = Outpainter(edge_feather_start=165, edge_feather_end=180).process([sbs], alpha=alpha)[0]
        assert np.array_equal(
            via_class, Outpainter(edge_feather_start=165, edge_feather_end=180).process([sbs], alpha=alpha)[0]
        )

    def test_empty_mask_returns_zeros_and_is_not_cached(self, fresh_feather_caches):
        out = compute_edge_feather_weights(np.zeros((40, 50), np.uint8), 165.0, 180.0)
        assert out.shape == (40, 50) and out.dtype == np.float32 and not out.any()
        assert len(_WEIGHTS_CACHE) == 0 and len(_GEOMETRY_CACHE) == 0


class TestFeatherCacheIsolation:
    def test_returned_array_is_a_private_writable_copy(self, fresh_feather_caches):
        alpha = _fov_mask(64, 64, 60.0)
        expected = _ref_compute_edge_feather_weights(alpha, 165.0, 180.0)
        first = compute_edge_feather_weights(alpha, 165.0, 180.0)
        assert first.flags.writeable
        first[:] = -1.0  # a caller scribbling on its result...
        second = compute_edge_feather_weights(alpha, 165.0, 180.0)
        assert second is not first and not np.shares_memory(second, first)
        assert np.array_equal(second, expected), "...cannot poison the cache"
        second[:] = -1.0
        assert np.array_equal(compute_edge_feather_weights(alpha, 165.0, 180.0), expected)

    def test_input_alpha_is_not_modified(self, fresh_feather_caches):
        alpha = _fov_mask(64, 64, 60.0)
        keep = alpha.copy()
        compute_edge_feather_weights(alpha, 165.0, 180.0)
        compute_edge_feather_weights(alpha, 165.0, 180.0)
        assert np.array_equal(alpha, keep)

    def test_mutating_the_callers_alpha_after_a_call_does_not_alias_the_cache(self, fresh_feather_caches):
        alpha = _fov_mask(64, 64, 60.0)
        compute_edge_feather_weights(alpha, 165.0, 180.0)
        alpha[20:30, 20:30] = 0
        got = compute_edge_feather_weights(alpha, 165.0, 180.0)
        assert np.array_equal(got, _ref_compute_edge_feather_weights(alpha, 165.0, 180.0))

    def test_digest_collision_is_caught_by_the_exact_check(self, fresh_feather_caches):
        """The dict key is a ::4 sub-sample at 256²; a change off that grid must
        still be detected (``np.array_equal`` confirm) and recomputed."""
        alpha = _fov_mask(256, 256, 60.0)
        base = compute_edge_feather_weights(alpha, 165.0, 180.0)
        tweaked = alpha.copy()
        tweaked[101, 101] = 0  # 101 % 4 != 0 → invisible to the sub-sample digest
        assert _weights_cache_key(tweaked, 165.0, 180.0) == _weights_cache_key(alpha, 165.0, 180.0)
        misses = _WEIGHTS_CACHE.misses
        got = compute_edge_feather_weights(tweaked, 165.0, 180.0)
        assert _WEIGHTS_CACHE.misses == misses + 1
        assert np.array_equal(got, _ref_compute_edge_feather_weights(tweaked, 165.0, 180.0))
        assert not np.array_equal(got, base)

    def test_different_angles_are_different_entries(self, fresh_feather_caches):
        alpha = _fov_mask(64, 64, 60.0)
        a = compute_edge_feather_weights(alpha, 165.0, 180.0)
        b = compute_edge_feather_weights(alpha, 110.0, 180.0)
        assert len(_WEIGHTS_CACHE) == 2 and not np.array_equal(a, b)
        assert np.array_equal(b, _ref_compute_edge_feather_weights(alpha, 110.0, 180.0))

    def test_geometry_tables_are_read_only_and_shared(self, fresh_feather_caches):
        t1 = _geometry_tables(60, 70, 1440)
        assert _geometry_tables(60, 70, 1440) is t1
        assert all(not a.flags.writeable for a in t1)
        with pytest.raises(ValueError):
            t1.theta_p[0, 0] = 0.0


# ---------------------------------------------------------------------------
#  Issue #276 — the integer index tables are stored as int32.  The values are
#  the pre-#276 int64 ones (git 1ed2a53 arithmetic, reproduced verbatim below);
#  the weights stay byte-exact against the pre-#264 reference through
#  ``TestFeatherCacheIsByteExact`` above, which runs unchanged.
# ---------------------------------------------------------------------------


def _ref_int64_index_tables(h, w, n_psi):
    """Pre-#276 ``_geometry_tables`` index arithmetic (git 1ed2a53), verbatim: ``(i0, i1, ray_flat)`` int64."""
    _theta_p, psi_p = _ref_hemisphere_pixel_angles(h, w)
    pos = psi_p.astype(np.float64) / (2.0 * np.pi) * n_psi
    i0 = np.floor(pos).astype(np.int64) % n_psi
    i1 = (i0 + 1) % n_psi
    n_theta = max(h, w) + 1
    theta = np.linspace(0.0, np.pi / 2.0, n_theta)[:, None]
    psi = np.linspace(0.0, 2.0 * np.pi, n_psi, endpoint=False)[None, :]
    sin_t = np.sin(theta)
    x = sin_t * np.cos(psi)
    y = sin_t * np.sin(psi)
    z = np.broadcast_to(np.cos(theta), x.shape)
    lon = np.arctan2(x, z)
    colat = np.arccos(np.clip(y, -1.0, 1.0))
    u = np.clip(np.floor((lon / np.pi + 0.5) * w).astype(np.int64), 0, w - 1)
    v = np.clip(np.floor(colat / np.pi * h).astype(np.int64), 0, h - 1)
    return i0, i1, v * w + u


def _index_table_bytes(h, w, n_psi, itemsize):
    """Bytes of ``i0 + i1 + ray_flat`` for one size at the given integer width."""
    return (2 * h * w + (max(h, w) + 1) * n_psi) * itemsize


class TestFeatherGeometryTablesAreInt32:
    @pytest.mark.parametrize(("h", "w", "n_psi"), [(64, 64, 1440), (96, 128, 1440), (128, 96, 360), (90, 120, 7)])
    def test_index_tables_are_int32_with_the_pre_276_int64_values(self, h, w, n_psi, fresh_feather_caches):
        t = _geometry_tables(h, w, n_psi)
        assert t.i0.dtype == t.i1.dtype == t.ray_flat.dtype == np.int32
        assert t.theta_p.dtype == t.frac.dtype == t.omf.dtype == t.ray_theta_deg.dtype == np.float32
        for got, ref in zip((t.i0, t.i1, t.ray_flat), _ref_int64_index_tables(h, w, n_psi), strict=True):
            assert ref.dtype == np.int64 and np.array_equal(got, ref)
        assert t.ray_flat.min() >= 0 and t.ray_flat.max() < h * w
        assert t.i0.min() >= 0 and t.i0.max() < n_psi and t.i1.min() >= 0 and t.i1.max() < n_psi

    def test_index_tables_take_half_the_bytes_of_int64(self, fresh_feather_caches):
        h, w, n_psi = 96, 128, 1440
        t = _geometry_tables(h, w, n_psi)
        index_bytes = t.i0.nbytes + t.i1.nbytes + t.ray_flat.nbytes
        assert index_bytes == _index_table_bytes(h, w, n_psi, 4) == _index_table_bytes(h, w, n_psi, 8) // 2
        float_bytes = t.theta_p.nbytes + t.frac.nbytes + t.omf.nbytes + t.ray_theta_deg.nbytes
        assert sum(a.nbytes for a in t) == index_bytes + float_bytes

    def test_oversized_canvas_is_refused_before_anything_is_allocated(self, fresh_feather_caches):
        """H*W (or n_psi) beyond int32 would wrap the indices: refuse instead of returning garbage."""
        with pytest.raises(ValueError, match="int32"):
            _geometry_tables(2**16, 2**15 + 1, 1440)  # H*W = 2**31 + 2**16
        with pytest.raises(ValueError, match="int32"):
            _geometry_tables(64, 64, 2**31 + 1)
        assert len(_GEOMETRY_CACHE) == 0

    def test_production_2880_index_tables_are_83_mb_not_166(self, fresh_feather_caches):
        """The card's number: one 2880² size holds 82.9 MB of int32 indices (165.9 MB as int64)."""
        t = _geometry_tables(_PROD_EYE, _PROD_EYE, 1440)
        index_bytes = t.i0.nbytes + t.i1.nbytes + t.ray_flat.nbytes
        assert index_bytes == _index_table_bytes(_PROD_EYE, _PROD_EYE, 1440, 4) == 82_949_760
        assert _index_table_bytes(_PROD_EYE, _PROD_EYE, 1440, 8) == 165_899_520 == 2 * index_bytes
        assert t.ray_flat.min() >= 0 and t.ray_flat.max() < _PROD_EYE * _PROD_EYE <= np.iinfo(np.int32).max
        assert t.i0.max() < 1440 and t.i1.max() < 1440


# Timing: ~0.25 s cold on a desktop CPU; a weights hit is a digest + one
# ``np.array_equal`` + one copy (a few ms).  Card: hits < 5 % of the cold call.
# Percentiles rather than every single sample so a GC pause / scheduler tick
# cannot flake the assertion; the worst case is still pinned far below "recomputed".
_FEATHER_TIMING_EYE = 1536


class TestFeatherCacheTiming:
    def test_100_calls_same_size_same_mask_hit_under_5_percent_of_cold(self, fresh_feather_caches):
        alpha = _fov_mask(_FEATHER_TIMING_EYE, _FEATHER_TIMING_EYE, 63.0)
        t_cold, cold = _timed(lambda: compute_edge_feather_weights(alpha, 165.0, 180.0))
        hits = []
        for _ in range(99):
            t, got = _timed(lambda: compute_edge_feather_weights(alpha, 165.0, 180.0))
            hits.append(t)
            assert np.array_equal(got, cold)
        assert _WEIGHTS_CACHE.hits == 99
        hits.sort()
        median, p95, worst = hits[len(hits) // 2], hits[int(len(hits) * 0.95)], hits[-1]
        budget = 0.05 * t_cold
        msg = f"cold {t_cold * 1e3:.0f} ms; hits median {median * 1e3:.1f} / p95 {p95 * 1e3:.1f} / max {worst * 1e3:.1f} ms"
        assert median < budget, msg
        assert p95 < (budget * 3 if _IS_CI else budget), msg
        assert worst < 0.25 * t_cold, msg

    def test_same_size_new_mask_reuses_geometry_and_is_at_least_2x_faster(self, fresh_feather_caches):
        n = _FEATHER_TIMING_EYE
        a1, a2 = _fov_mask(n, n, 63.0), _fov_mask(n, n, 62.0)
        t_cold, _ = _timed(lambda: compute_edge_feather_weights(a1, 165.0, 180.0))
        geo_hits = _GEOMETRY_CACHE.hits
        t_new, got = _timed(lambda: compute_edge_feather_weights(a2, 165.0, 180.0))
        assert _GEOMETRY_CACHE.hits == geo_hits + 1 and len(_GEOMETRY_CACHE) == 1
        assert t_new * 2 <= t_cold, f"new-mask call {t_new:.3f} s vs cold {t_cold:.3f} s"
        assert np.array_equal(got, _ref_compute_edge_feather_weights(a2, 165.0, 180.0))


class TestFeatherCacheIsBounded:
    def test_ten_sizes_keep_at_most_four_entries_and_evict_lru(self, fresh_feather_caches):
        sizes = [(32 + 4 * i, 40 + 4 * i) for i in range(10)]
        for h, w in sizes:
            compute_edge_feather_weights(_fov_mask(h, w, 60.0), 165.0, 180.0)
            assert len(_GEOMETRY_CACHE) <= _FEATHER_CACHE_MAXSIZE == 4
            assert len(_WEIGHTS_CACHE) <= _FEATHER_CACHE_MAXSIZE
        assert [k[:2] for k in _GEOMETRY_CACHE.ordered_keys()] == sizes[-4:], "four most recent survive, oldest first"
        assert [k[:2] for k in _WEIGHTS_CACHE.ordered_keys()] == sizes[-4:]
        # An evicted size is rebuilt (miss) and is still byte-exact.
        misses = _GEOMETRY_CACHE.misses
        old = _fov_mask(*sizes[0], 60.0)
        got = compute_edge_feather_weights(old, 165.0, 180.0)
        assert np.array_equal(got, _ref_compute_edge_feather_weights(old, 165.0, 180.0))
        assert _GEOMETRY_CACHE.misses == misses + 1 and len(_GEOMETRY_CACHE) == 4

    def test_a_weights_hit_keeps_that_sizes_geometry_most_recently_used(self, fresh_feather_caches):
        a = _fov_mask(48, 48, 60.0)
        compute_edge_feather_weights(a, 165.0, 180.0)  # size A
        for h in (52, 56, 60):
            compute_edge_feather_weights(_fov_mask(h, h, 60.0), 165.0, 180.0)  # B, C, D → cache full
        compute_edge_feather_weights(a, 165.0, 180.0)  # weights hit on A → A becomes most recent
        compute_edge_feather_weights(_fov_mask(64, 64, 60.0), 165.0, 180.0)  # E evicts B, not A
        kept = {k[:2] for k in _GEOMETRY_CACHE.ordered_keys()}
        assert (48, 48) in kept and (52, 52) not in kept and len(kept) == 4


class TestFeatherCacheThreadSafety:
    def test_concurrent_callers_get_exact_results_and_the_caches_stay_bounded(self, fresh_feather_caches):
        masks = [
            _fov_mask(96, 96, 60.0),
            _fov_mask(96, 96, 50.0),
            _fov_mask(80, 112, 63.0),
            _fov_mask(112, 80, 63.0),
            _fov_mask(64, 64, 45.0),
            _fov_mask(72, 72, 55.0),
        ]
        expected = [_ref_compute_edge_feather_weights(m, 165.0, 180.0) for m in masks]
        errors = []

        def worker(seed):
            rng = np.random.default_rng(seed)
            try:
                for _ in range(40):
                    i = int(rng.integers(len(masks)))
                    if not np.array_equal(compute_edge_feather_weights(masks[i], 165.0, 180.0), expected[i]):
                        errors.append(f"thread {seed}: mask {i} mismatch")
            except Exception as exc:
                errors.append(f"thread {seed}: {exc!r}")

        threads = [threading.Thread(target=worker, args=(s,)) for s in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, errors
        assert len(_GEOMETRY_CACHE) <= _FEATHER_CACHE_MAXSIZE and len(_WEIGHTS_CACHE) <= _FEATHER_CACHE_MAXSIZE
