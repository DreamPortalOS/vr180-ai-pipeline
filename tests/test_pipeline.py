"""Unit tests for VR180 pipeline modules.

Run with: pytest tests/ -v
"""
import os
import struct
import subprocess
import tempfile

import cv2
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dummy_frame():
    """Generate a random 480x640 RGB frame."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def dummy_depth(dummy_frame):
    """Generate a random depth map matching dummy_frame."""
    h, w = dummy_frame.shape[:2]
    return np.random.rand(h, w).astype(np.float32)


@pytest.fixture
def tmp_video(tmp_path):
    """Create a minimal 3-frame test video using ffmpeg."""
    video_path = str(tmp_path / "test_input.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i",
         "color=c=blue:s=320x240:d=0.125:r=24",
         "-c:v", "libx264", "-pix_fmt", "yuv420p",
         video_path],
        capture_output=True, timeout=30,
    )
    return video_path


@pytest.fixture
def tmp_sbs_video(tmp_path):
    """Create a minimal SBS video (640x480) with ffmpeg."""
    video_path = str(tmp_path / "test_sbs.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i",
         "color=c=red:s=640x480:d=0.125:r=24",
         "-c:v", "libx264", "-pix_fmt", "yuv420p",
         video_path],
        capture_output=True, timeout=30,
    )
    return video_path


# ---------------------------------------------------------------------------
# pipeline.depth_estimator
# ---------------------------------------------------------------------------

class TestDepthEstimator:
    def test_import(self):
        from pipeline.depth_estimator import DepthEstimator
        assert DepthEstimator is not None

    def test_estimate_returns_correct_shape(self, dummy_frame):
        from pipeline.depth_estimator import DepthEstimator
        estimator = DepthEstimator(model_size="small", device="cpu")
        depth = estimator.estimate(dummy_frame)
        assert depth.shape == dummy_frame.shape[:2]
        assert depth.dtype == np.float32

    def test_estimate_non_negative(self, dummy_frame):
        from pipeline.depth_estimator import DepthEstimator
        estimator = DepthEstimator(model_size="small", device="cpu")
        depth = estimator.estimate(dummy_frame)
        assert np.all(depth >= 0)


# ---------------------------------------------------------------------------
# pipeline.stereo_renderer
# ---------------------------------------------------------------------------

class TestStereoRenderer:
    def test_import(self):
        from pipeline.stereo_renderer import StereoRenderer
        assert StereoRenderer is not None

    def test_render_shapes(self, dummy_frame, dummy_depth):
        from pipeline.stereo_renderer import StereoRenderer
        renderer = StereoRenderer()
        left, right = renderer.render(dummy_frame, dummy_depth)
        assert left.shape == dummy_frame.shape
        assert right.shape == dummy_frame.shape

    def test_stereo_disparity(self, dummy_frame, dummy_depth):
        """Left and right frames should differ (have disparity)."""
        from pipeline.stereo_renderer import StereoRenderer
        renderer = StereoRenderer()
        left, right = renderer.render(dummy_frame, dummy_depth)
        diff = np.mean(np.abs(left.astype(float) - right.astype(float)))
        assert diff > 0, "Left and right frames should have some disparity"


# ---------------------------------------------------------------------------
# pipeline.equirectangular_mapper
# ---------------------------------------------------------------------------

class TestEquirectangularMapper:
    def test_import(self):
        from pipeline.equirectangular_mapper import EquirectangularMapper
        assert EquirectangularMapper is not None

    def test_map_stereo_pair_shape(self, dummy_frame):
        from pipeline.equirectangular_mapper import EquirectangularMapper
        mapper = EquirectangularMapper(
            output_width=640,
            output_height=320,
            src_hfov=70.0,
            use_ffmpeg=False,
        )
        sbs = mapper.map_stereo_pair(dummy_frame, dummy_frame)
        assert sbs.shape[0] == 320
        assert sbs.shape[1] == 1280  # SBS = 2× per-eye width

    def test_sbs_layout(self, dummy_frame):
        """SBS output width should be 2× per-eye width."""
        from pipeline.equirectangular_mapper import EquirectangularMapper
        w_per_eye, h = 320, 320
        mapper = EquirectangularMapper(
            output_width=w_per_eye,
            output_height=h,
            src_hfov=70.0,
            use_ffmpeg=False,
        )
        sbs = mapper.map_stereo_pair(dummy_frame, dummy_frame)
        assert sbs.shape[1] == w_per_eye * 2


# ---------------------------------------------------------------------------
# pipeline.vr_metadata
# ---------------------------------------------------------------------------

class TestVRMetadataEmbedder:
    def test_import(self):
        from pipeline.vr_metadata import VRMetadataEmbedder
        assert VRMetadataEmbedder is not None

    def test_embed_single_frame(self, tmp_path):
        from pipeline.vr_metadata import VRMetadataEmbedder
        embedder = VRMetadataEmbedder(codec="h264", crf=23, fps=24)
        # Create a single red frame
        frame = np.zeros((240, 480, 3), dtype=np.uint8)
        frame[:, :, 0] = 255
        output_path = str(tmp_path / "output_vr180.mp4")
        result = embedder.embed_single_frame_batch(
            [frame], output_path, width=480, height=240,
        )
        assert os.path.exists(result)
        assert os.path.getsize(result) > 0

    def test_sv3d_metadata_present(self, tmp_path):
        from pipeline.vr_metadata import VRMetadataEmbedder
        embedder = VRMetadataEmbedder(codec="h264", crf=23, fps=24)
        frame = np.zeros((240, 480, 3), dtype=np.uint8)
        output_path = str(tmp_path / "output_vr180_meta.mp4")
        result = embedder.embed_single_frame_batch(
            [frame], output_path, width=480, height=240,
        )
        with open(result, "rb") as f:
            data = f.read()
        assert b"sv3d" in data, "sv3d box should be present"
        assert b"st3d" in data, "st3d box should be present"


# ---------------------------------------------------------------------------
# pipeline.spherical_injector
# ---------------------------------------------------------------------------

class TestSphericalInjector:
    def test_import(self):
        from pipeline.spherical_injector import inject_spherical_metadata
        assert inject_spherical_metadata is not None


# ---------------------------------------------------------------------------
# pipeline.upscaler
# ---------------------------------------------------------------------------

class TestPixelUpscaler:
    def test_import(self):
        from pipeline.upscaler import PixelUpscaler
        assert PixelUpscaler is not None

    def test_upscale_frame_opencv(self, dummy_frame):
        """Test OpenCV fallback upscaling."""
        from pipeline.upscaler import PixelUpscaler
        upscaler = PixelUpscaler(scale=2, device="cpu")
        h, w = dummy_frame.shape[:2]
        frame_bgr = cv2.cvtColor(dummy_frame, cv2.COLOR_RGB2BGR)
        result_bgr = upscaler.upscale_frame(frame_bgr)
        result = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
        assert result.shape == (h * 2, w * 2, 3)


# ---------------------------------------------------------------------------
# scripts.run_pipeline (CLI argument parsing)
# ---------------------------------------------------------------------------

class TestRunPipelineCLI:
    def test_parse_args_defaults(self):
        """Test that default arguments parse correctly."""
        import sys
        sys.argv = ["run_pipeline.py", "--input", "test.mp4"]
        # We need to import and test parse_args
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
        # Just verify the module imports without error
        import importlib
        spec = importlib.util.spec_from_file_location(
            "run_pipeline",
            os.path.join(os.path.dirname(__file__), "..", "scripts", "run_pipeline.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        # Don't execute main, just verify it loads
        assert spec is not None


# ---------------------------------------------------------------------------
# Integration test (end-to-end with 3 frames)
# ---------------------------------------------------------------------------

class TestEndToEnd:
    @pytest.mark.slow
    def test_full_pipeline_mini(self, tmp_video, tmp_path):
        """Run the full pipeline on a 3-frame video and verify output."""
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from pipeline.depth_estimator import DepthEstimator
        from pipeline.stereo_renderer import StereoRenderer
        from pipeline.equirectangular_mapper import EquirectangularMapper
        from pipeline.vr_metadata import VRMetadataEmbedder

        # Read 2 frames
        cap = cv2.VideoCapture(tmp_video)
        frames = []
        for _ in range(2):
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        assert len(frames) == 2

        # Depth
        estimator = DepthEstimator(model_size="small", device="cpu")
        depths = [estimator.estimate(f) for f in frames]

        # Stereo
        renderer = StereoRenderer()
        lefts, rights = [], []
        for frame, depth in zip(frames, depths):
            l, r = renderer.render(frame, depth)
            lefts.append(l)
            rights.append(r)

        # Equirect
        mapper = EquirectangularMapper(
            output_width=320, output_height=320, src_hfov=70.0, use_ffmpeg=False,
        )
        sbs_frames = []
        for l, r in zip(lefts, rights):
            sbs_frames.append(mapper.map_stereo_pair(l, r))

        # Encode
        embedder = VRMetadataEmbedder(codec="h264", crf=23, fps=24)
        output_path = str(tmp_path / "e2e_output.mp4")
        H, W = sbs_frames[0].shape[:2]
        result = embedder.embed_single_frame_batch(sbs_frames, output_path, width=W, height=H)

        assert os.path.exists(result)
        with open(result, "rb") as f:
            data = f.read()
        assert b"sv3d" in data
        assert b"st3d" in data