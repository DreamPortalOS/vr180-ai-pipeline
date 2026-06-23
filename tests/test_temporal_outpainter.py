"""Tests for Temporal Outpainter (Phase 3 R&D)."""

import numpy as np

from pipeline.research.temporal_outpainter import (
    OutpaintQualityMetrics,
    TemporalOutpainter,
)


class TestBoundaryMask:
    """Test VR180 boundary mask detection."""

    def test_mask_shape(self):
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        outpainter = TemporalOutpainter()
        mask = outpainter.detect_boundary_mask(frame)
        assert mask.shape == (100, 200)
        assert mask.dtype == np.uint8

    def test_poles_masked(self):
        frame = np.zeros((180, 360, 3), dtype=np.uint8)
        outpainter = TemporalOutpainter(pole_angle_deg=30.0)
        mask = outpainter.detect_boundary_mask(frame)
        # pole_frac = 30/180 = 0.1667, pole_h = int(180 * 0.1667 / 2) = 15
        # Top pole region should be masked
        assert np.all(mask[:15, :] == 255)
        # Bottom pole region should be masked
        assert np.all(mask[165:, :] == 255)

    def test_edges_masked(self):
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        outpainter = TemporalOutpainter(edge_margin_pct=0.05)
        mask = outpainter.detect_boundary_mask(frame)
        # Left edge
        assert np.all(mask[:, :10] == 255)
        # Right edge
        assert np.all(mask[:, 190:] == 255)

    def test_center_not_masked(self):
        frame = np.zeros((180, 360, 3), dtype=np.uint8)
        outpainter = TemporalOutpainter(pole_angle_deg=30.0, edge_margin_pct=0.05)
        mask = outpainter.detect_boundary_mask(frame)
        # Center should be unmasked (original content)
        center_region = mask[50:130, 50:310]
        assert np.all(center_region == 0)


class TestOpticalFlow:
    """Test optical flow computation."""

    def test_flow_shape(self):
        outpainter = TemporalOutpainter(flow_pyramid_scale=1.0)
        f1 = np.random.randint(0, 255, (60, 80, 3), dtype=np.uint8)
        f2 = np.random.randint(0, 255, (60, 80, 3), dtype=np.uint8)
        flow = outpainter.compute_optical_flow(f1, f2)
        assert flow.shape == (60, 80, 2)

    def test_identical_frames_zero_flow(self):
        outpainter = TemporalOutpainter(flow_pyramid_scale=1.0)
        frame = np.random.randint(0, 255, (60, 80, 3), dtype=np.uint8)
        flow = outpainter.compute_optical_flow(frame, frame)
        # Flow should be near zero for identical frames
        assert np.abs(flow).mean() < 0.5

    def test_grayscale_input(self):
        outpainter = TemporalOutpainter(flow_pyramid_scale=1.0)
        f1 = np.random.randint(0, 255, (60, 80), dtype=np.uint8)
        f2 = np.random.randint(0, 255, (60, 80), dtype=np.uint8)
        flow = outpainter.compute_optical_flow(f1, f2)
        assert flow.shape == (60, 80, 2)


class TestWarpFrameByFlow:
    """Test flow-based frame warping."""

    def test_zero_flow_identity(self):
        outpainter = TemporalOutpainter()
        frame = np.random.randint(0, 255, (60, 80, 3), dtype=np.uint8)
        flow = np.zeros((60, 80, 2), dtype=np.float32)
        warped = outpainter.warp_frame_by_flow(frame, flow)
        np.testing.assert_array_equal(warped, frame)

    def test_output_shape_preserved(self):
        outpainter = TemporalOutpainter()
        frame = np.random.randint(0, 255, (60, 80, 3), dtype=np.uint8)
        flow = np.random.randn(60, 80, 2).astype(np.float32) * 0.5
        warped = outpainter.warp_frame_by_flow(frame, flow)
        assert warped.shape == frame.shape


class TestOutpaintFrame:
    """Test single-frame outpainting."""

    def test_outpaint_preserves_original_content(self):
        """Non-masked regions should be unchanged."""
        outpainter = TemporalOutpainter(max_iterations=1)
        h, w = 60, 80
        frame = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        ctx = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[:5, :] = 255  # Only mask top 5 rows

        result, _metrics = outpainter.outpaint_frame(frame, [ctx], mask)

        # Non-masked region should be identical
        np.testing.assert_array_equal(result[10:, :], frame[10:, :])

    def test_masked_region_filled(self):
        """Masked regions should have content (not all zeros)."""
        outpainter = TemporalOutpainter(max_iterations=2)
        h, w = 60, 80
        frame = np.random.randint(50, 200, (h, w, 3), dtype=np.uint8)
        ctx = np.random.randint(50, 200, (h, w, 3), dtype=np.uint8)
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[:10, :] = 255

        result, _metrics = outpainter.outpaint_frame(frame, [ctx], mask)

        # Masked region should not be all zeros
        assert result[:10, :].mean() > 0

    def test_no_context_returns_original(self):
        outpainter = TemporalOutpainter()
        frame = np.random.randint(0, 255, (60, 80, 3), dtype=np.uint8)
        mask = np.zeros((60, 80), dtype=np.uint8)
        mask[:5, :] = 255

        result, metrics = outpainter.outpaint_frame(frame, [], mask)
        np.testing.assert_array_equal(result, frame)
        assert metrics.iterations_used == 0
        assert metrics.converged is True

    def test_empty_mask_returns_original(self):
        outpainter = TemporalOutpainter()
        frame = np.random.randint(0, 255, (60, 80, 3), dtype=np.uint8)
        ctx = np.random.randint(0, 255, (60, 80, 3), dtype=np.uint8)
        mask = np.zeros((60, 80), dtype=np.uint8)

        result, _metrics = outpainter.outpaint_frame(frame, [ctx], mask)
        np.testing.assert_array_equal(result, frame)

    def test_metrics_have_required_fields(self):
        outpainter = TemporalOutpainter(max_iterations=1)
        frame = np.random.randint(0, 255, (60, 80, 3), dtype=np.uint8)
        ctx = np.random.randint(0, 255, (60, 80, 3), dtype=np.uint8)
        mask = np.zeros((60, 80), dtype=np.uint8)
        mask[:5, :] = 255

        _, metrics = outpainter.outpaint_frame(frame, [ctx], mask)
        assert isinstance(metrics, OutpaintQualityMetrics)
        assert hasattr(metrics, "ssim")
        assert hasattr(metrics, "psnr")
        assert hasattr(metrics, "coverage_pct")
        assert hasattr(metrics, "iterations_used")
        assert hasattr(metrics, "converged")


class TestOutpaintSequence:
    """Test full-sequence outpainting."""

    def test_outpaint_returns_correct_count(self):
        outpainter = TemporalOutpainter(max_iterations=1)
        frames = [
            np.random.randint(0, 255, (40, 60, 3), dtype=np.uint8)
            for _ in range(5)
        ]
        results, metrics_list = outpainter.outpaint(frames)
        assert len(results) == 5
        assert len(metrics_list) == 5

    def test_auto_mask_detection(self):
        outpainter = TemporalOutpainter(max_iterations=1)
        frames = [
            np.random.randint(0, 255, (60, 80, 3), dtype=np.uint8)
            for _ in range(3)
        ]
        results, _metrics_list = outpainter.outpaint(frames, mask=None)
        assert len(results) == 3

    def test_empty_frames_returns_empty(self):
        outpainter = TemporalOutpainter()
        results, metrics_list = outpainter.outpaint([])
        assert results == []
        assert metrics_list == []


class TestQualityMetrics:
    """Test PSNR and SSIM computation."""

    def test_psnr_identical_frames(self):
        outpainter = TemporalOutpainter()
        frame = np.random.randint(0, 255, (60, 80, 3), dtype=np.uint8)
        mask = np.zeros((60, 80), dtype=bool)
        psnr = outpainter._compute_psnr(frame, frame.copy(), mask)
        assert psnr == float('inf')

    def test_ssim_identical_frames(self):
        outpainter = TemporalOutpainter()
        frame = np.random.randint(0, 255, (60, 80, 3), dtype=np.uint8)
        mask = np.zeros((60, 80), dtype=bool)
        ssim = outpainter._compute_ssim(frame, frame.copy(), mask)
        assert ssim > 0.99
