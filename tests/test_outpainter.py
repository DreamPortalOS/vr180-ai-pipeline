"""Tests for pipeline/outpainter.py — 180° Outpaint Fill + Edge Feather (#244)."""

import contextlib
import logging
import math
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

from pipeline.equirectangular_mapper import EquirectangularMapper
from pipeline.outpainter import (
    DEFAULT_EDGE_FEATHER_END,
    DEFAULT_EDGE_FEATHER_START,
    AIOutpaintBackend,
    MockAIOutpaintBackend,
    Outpainter,
    _gradient_outpaint_single,
    alpha_to_fill_mask,
    content_edge_angles,
    detect_black_boundary_mask,
    resolve_edge_feather,
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

    def test_streaming_warns_that_feather_is_ignored(self, rp):
        args = rp.parse_args([])
        args.streaming, args.stage, args.edge_feather_start = True, "all", 165.0
        warned = _capture_warnings("vr180-pipeline", lambda: rp._warn_streaming_unsupported_args(args))
        assert any("--edge-feather-start" in m for m in warned), warned
