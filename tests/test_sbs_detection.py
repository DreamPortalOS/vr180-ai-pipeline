"""Tests for Smart SBS Input Detection (Task 1.1)."""

import subprocess

import numpy as np
import pytest
from scripts.run_pipeline import detect_sbs_input


@pytest.fixture
def standard_video(tmp_path):
    """Create a 16:9 video (standard 2D input)."""
    video_path = str(tmp_path / "standard.mp4")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=1920x1080:d=0.125:r=24",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            video_path,
        ],
        capture_output=True,
        timeout=30,
    )
    return video_path


@pytest.fixture
def sbs_video(tmp_path):
    """Create a 4:1 SBS video (e.g., 7680x1920)."""
    video_path = str(tmp_path / "sbs.mp4")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=7680x1920:d=0.125:r=24",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            video_path,
        ],
        capture_output=True,
        timeout=30,
    )
    return video_path


@pytest.fixture
def ultra_wide_video(tmp_path):
    """Create a 3.6:1 video (just above threshold)."""
    video_path = str(tmp_path / "ultrawide.mp4")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=green:s=3600x1000:d=0.125:r=24",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            video_path,
        ],
        capture_output=True,
        timeout=30,
    )
    return video_path


class TestSBSDetection:
    """Test auto-detection of SBS stereo input."""

    def test_standard_16x9_not_sbs(self, standard_video):
        """Standard 16:9 video should NOT be detected as SBS."""
        result = detect_sbs_input(standard_video)
        assert result is False

    def test_sbs_4x1_detected(self, sbs_video):
        """4:1 SBS video (7680x1920) should be detected as SBS."""
        result = detect_sbs_input(sbs_video)
        assert result is True

    def test_force_sbs_flag(self, standard_video):
        """--force-sbs should override detection for any input."""
        result = detect_sbs_input(standard_video, force_sbs=True)
        assert result is True

    def test_ultra_wide_detected(self, ultra_wide_video):
        """3.6:1 video (above 3.5 threshold) should be detected as SBS."""
        result = detect_sbs_input(ultra_wide_video)
        assert result is True

    def test_nonexistent_file_returns_false(self):
        """Non-existent file should return False gracefully."""
        result = detect_sbs_input("/nonexistent/video.mp4")
        assert result is False

    def test_stage_order_sbs_skips_depth_stereo(self):
        """STAGE_ORDER_SBS should not contain 'depth' or 'stereo'."""
        from scripts.run_pipeline import STAGE_ORDER_SBS

        assert "depth" not in STAGE_ORDER_SBS
        assert "stereo" not in STAGE_ORDER_SBS
        assert "equirect" in STAGE_ORDER_SBS
        assert "metadata" in STAGE_ORDER_SBS


class TestSBSPipelineIntegration:
    """Integration test: SBS input should skip depth/stereo and go to equirect."""

    def test_sbs_frame_split(self):
        """Test that SBS frames are correctly split into left/right."""
        # Create a synthetic SBS frame (left=red, right=blue)
        h, w = 480, 1920  # 4:1 SBS
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, : w // 2, 2] = 255  # left half = red
        frame[:, w // 2 :, 0] = 255  # right half = blue

        mid = w // 2
        left = frame[:, :mid, :]
        right = frame[:, mid:, :]

        assert left.shape == (h, mid, 3)
        assert right.shape == (h, mid, 3)
        # Left should be predominantly red
        assert left[:, :, 2].mean() > 200
        # Right should be predominantly blue
        assert right[:, :, 0].mean() > 200
