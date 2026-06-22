"""Tests for VR180 Orientation Matrix (Task 1.2)."""

import numpy as np
import pytest

from pipeline.research.orientation_matrix import (
    apply_flip,
    apply_transpose,
    generate_orientation_matrix,
    generate_ffmpeg_filter_map,
)


class TestApplyFlip:
    """Test cv2.flip wrapper."""

    def test_none_returns_copy(self):
        frame = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        result = apply_flip(frame, "none")
        np.testing.assert_array_equal(result, frame)
        # Should be a copy, not same object
        assert result is not frame

    def test_vflip_flips_vertically(self):
        frame = np.zeros((4, 6, 3), dtype=np.uint8)
        frame[0, :, :] = 255  # top row white
        result = apply_flip(frame, "vflip")
        # Bottom row should now be white
        assert result[3, 0, 0] == 255
        assert result[0, 0, 0] == 0

    def test_hflip_flips_horizontally(self):
        frame = np.zeros((4, 6, 3), dtype=np.uint8)
        frame[:, 0, :] = 255  # left column white
        result = apply_flip(frame, "hflip")
        # Right column should now be white
        assert result[0, 5, 0] == 255
        assert result[0, 0, 0] == 0

    def test_both_is_180_rotation(self):
        frame = np.zeros((4, 6, 3), dtype=np.uint8)
        frame[0, 0, :] = 255  # top-left white
        result = apply_flip(frame, "both")
        # Bottom-right should be white
        assert result[3, 5, 0] == 255

    def test_invalid_type_raises(self):
        frame = np.zeros((4, 6, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="Unknown flip type"):
            apply_flip(frame, "invalid")


class TestApplyTranspose:
    """Test ffmpeg-style transpose wrapper."""

    def test_none_returns_copy(self):
        frame = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        result = apply_transpose(frame, "none")
        np.testing.assert_array_equal(result, frame)

    def test_t1_rotates_90cw(self):
        frame = np.zeros((4, 8, 3), dtype=np.uint8)
        frame[0, :, :] = 255  # top row white
        result = apply_transpose(frame, "t1")
        # After 90° CW, dimensions should swap
        assert result.shape == (8, 4, 3)

    def test_t2_rotates_90ccw(self):
        frame = np.zeros((4, 8, 3), dtype=np.uint8)
        result = apply_transpose(frame, "t2")
        assert result.shape == (8, 4, 3)

    def test_invalid_type_raises(self):
        frame = np.zeros((4, 6, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="Unknown transpose type"):
            apply_transpose(frame, "bad")


class TestGenerateOrientationMatrix:
    """Test the full orientation grid generation."""

    def test_grid_dimensions(self):
        frame = np.random.randint(0, 255, (60, 80, 3), dtype=np.uint8)
        grid, combo_map = generate_orientation_matrix(frame)
        # 4 flip types × 5 transpose types = 20 combos
        assert len(combo_map) == 20
        # Grid should have correct dimensions
        label_h = 40
        expected_h = 5 * (60 + label_h)  # 5 transpose rows
        expected_w = 4 * 80              # 4 flip cols
        assert grid.shape == (expected_h, expected_w, 3)

    def test_combo_map_keys(self):
        frame = np.random.randint(0, 255, (60, 80, 3), dtype=np.uint8)
        _, combo_map = generate_orientation_matrix(frame)
        # Should have (row, col) keys for all combinations
        for row in range(5):
            for col in range(4):
                assert (row, col) in combo_map

    def test_combo_map_has_required_fields(self):
        frame = np.random.randint(0, 255, (60, 80, 3), dtype=np.uint8)
        _, combo_map = generate_orientation_matrix(frame)
        for key, info in combo_map.items():
            assert "label" in info
            assert "short" in info
            assert "flip" in info
            assert "transpose" in info
            assert "frame" in info

    def test_custom_flip_types(self):
        frame = np.random.randint(0, 255, (60, 80, 3), dtype=np.uint8)
        _, combo_map = generate_orientation_matrix(
            frame, flip_types=["none", "vflip"], transpose_types=["none"]
        )
        assert len(combo_map) == 2

    def test_transposed_frames_have_correct_size(self):
        """All frames in combo_map should be resized to original dimensions."""
        frame = np.random.randint(0, 255, (60, 80, 3), dtype=np.uint8)
        _, combo_map = generate_orientation_matrix(frame)
        for key, info in combo_map.items():
            assert info["frame"].shape[:2] == (60, 80)


class TestFFmpegFilterMap:
    """Test ffmpeg filter string generation."""

    def test_report_contains_all_combos(self):
        frame = np.random.randint(0, 255, (60, 80, 3), dtype=np.uint8)
        _, combo_map = generate_orientation_matrix(frame)
        report = generate_ffmpeg_filter_map(combo_map)
        # Should have 20 lines with -vf
        assert report.count("-vf") == 20

    def test_report_has_header(self):
        frame = np.random.randint(0, 255, (60, 80, 3), dtype=np.uint8)
        _, combo_map = generate_orientation_matrix(frame)
        report = generate_ffmpeg_filter_map(combo_map)
        assert "# FFmpeg Filter Equivalents" in report

    def test_identity_has_no_filter(self):
        frame = np.random.randint(0, 255, (60, 80, 3), dtype=np.uint8)
        _, combo_map = generate_orientation_matrix(frame)
        report = generate_ffmpeg_filter_map(combo_map)
        assert "(no filter)" in report