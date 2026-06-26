"""Tests for pipeline/outpainter.py — 180° Outpaint Fill."""

import numpy as np
import pytest

from pipeline.outpainter import (
    AIOutpaintBackend,
    MockAIOutpaintBackend,
    Outpainter,
    _gradient_outpaint_single,
    detect_black_boundary_mask,
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
        # All black: top_ratio and bottom_ratio regions get masked
        h = frame.shape[0]
        top_end = int(h * 0.2)
        bottom_start = int(h * (1.0 - 0.2))
        assert np.all(mask[:top_end, :] == 255), "Top boundary masked"
        assert np.all(mask[bottom_start:, :] == 255), "Bottom boundary masked"
        # Middle (unscanned area) stays unmasked — boundary-only detection
        assert np.all(mask[top_end:bottom_start, :] == 0), "Middle stays unmasked"


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

    def test_mask_params_propagated(self):
        """Verify that mask parameters are passed through correctly."""
        h, w = 192, 768
        frame = _make_frame(h, w)
        # Use very small top/bottom ratios — only scan 5% from each end
        mask = detect_black_boundary_mask(frame, threshold=10, top_ratio=0.05, bottom_ratio=0.05)
        top_masked = int(h * 0.05)
        bottom_start = int(h * (1 - 0.05))
        # Only the very top and very bottom rows should be masked
        assert np.any(mask[:top_masked, :] == 255), "Top 5% should be masked"
        assert np.all(mask[top_masked:bottom_start, :] == 0), "Middle should not be masked"

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
