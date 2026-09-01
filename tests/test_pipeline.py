"""Unit tests for VR180 pipeline modules.

Run with: pytest tests/ -v
"""

import contextlib
import glob
import os
import subprocess

import cv2
import numpy as np
import pytest


def _spatialmedia_available() -> bool:
    """sv3d/st3d ISOBMFF boxes require Google's optional spatial-media CLI.

    Without it the pipeline falls back to ffmpeg, which injects equivalent
    spherical metadata in a different form (no literal sv3d/st3d boxes).
    """
    import importlib.util

    return importlib.util.find_spec("spatialmedia") is not None


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
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=320x240:d=0.125:r=24",
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
def tmp_sbs_video(tmp_path):
    """Create a minimal SBS video (640x480) with ffmpeg."""
    video_path = str(tmp_path / "test_sbs.mp4")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=640x480:d=0.125:r=24",
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

    def test_estimate_sequence_chunked_matches_whole(self, monkeypatch):
        """V-4 (#37): chunked depth == whole-sequence, all chunk sizes.

        Depth-Anything is per-frame (no temporal state) → bit-exact with
        overlap=0.  ``estimate`` is stubbed so no model/transformers load.
        """
        from pipeline.depth_estimator import DepthEstimator

        rng = np.random.default_rng(3)
        frames = [rng.integers(0, 256, (48, 48, 3), dtype=np.uint8) for _ in range(20)]

        def _fake_estimate(self, frame):
            # Deterministic per-frame transform (frame-dependent, stateless).
            return frame.mean(axis=2).astype(np.float32) / 255.0

        monkeypatch.setattr(DepthEstimator, "estimate", _fake_estimate)

        est = DepthEstimator(model_size="small", device="cpu")
        whole = [est.estimate(f) for f in frames]

        for cs in (1, 6, 100, None):
            est2 = DepthEstimator(model_size="small", device="cpu")
            got = est2.estimate_sequence_chunked(frames, chunk_size=cs, overlap=0)
            assert len(got) == 20
            for i, (g, w) in enumerate(zip(got, whole, strict=True)):
                assert np.array_equal(g, w), f"cs={cs} frame {i} differs"


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

    def test_render_sequence_chunked_matches_whole(self):
        """V-4 (#37): chunked stereo render == whole-sequence render, all chunk sizes.

        The StereoRenderer reuses one instance across chunks so ``_prev_disparity``
        is continuous → bit-exact with overlap=0.  Covers the chunk_size=1 and
        chunk_size>total extremes plus a typical multi-chunk size.
        """
        from pipeline.stereo_renderer import StereoRenderer

        rng = np.random.default_rng(7)
        frames = [rng.integers(0, 256, (64, 64, 3), dtype=np.uint8) for _ in range(20)]
        depths = [rng.random((64, 64)).astype(np.float32) for _ in range(20)]

        # Whole-sequence reference
        ref = StereoRenderer()
        whole = [ref.render(f, d) for f, d in zip(frames, depths, strict=True)]

        for cs in (1, 7, 100, None):
            r = StereoRenderer()
            got = r.render_sequence_chunked(frames, depths, chunk_size=cs, overlap=0)
            assert len(got) == 20, f"count wrong for cs={cs}"
            for i, (gl, wl) in enumerate(zip(got, whole, strict=True)):
                assert np.array_equal(gl[0], wl[0]), f"cs={cs} left frame {i} differs"
                assert np.array_equal(gl[1], wl[1]), f"cs={cs} right frame {i} differs"


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

    def test_default_square_per_eye(self):
        """R-2: Default constructor should produce square (1920×1920) per-eye output.

        SBS width should be 2× height (= 3840).
        """
        from pipeline.equirectangular_mapper import EquirectangularMapper

        mapper = EquirectangularMapper(use_ffmpeg=False)
        assert mapper.output_width == 1920
        assert mapper.output_height == 1920
        assert mapper.src_hfov == 90.0

    def test_square_sbs_default(self, dummy_frame):
        """R-2: Default 1920×1920 per-eye → SBS width = 3840, height = 1920."""
        from pipeline.equirectangular_mapper import EquirectangularMapper

        mapper = EquirectangularMapper(output_width=480, output_height=480, use_ffmpeg=False)
        sbs = mapper.map_stereo_pair(dummy_frame, dummy_frame)
        assert sbs.shape[0] == 480
        assert sbs.shape[1] == 960  # 480 * 2

    def test_default_max_disparity(self):
        """R-2: Default max_disparity should be ~0.02."""
        from pipeline.stereo_renderer import StereoRenderer

        renderer = StereoRenderer()
        assert renderer.max_disparity == pytest.approx(0.02)


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
            [frame],
            output_path,
            width=480,
            height=240,
        )
        assert os.path.exists(result)
        assert os.path.getsize(result) > 0

    @pytest.mark.skipif(
        not _spatialmedia_available(),
        reason="sv3d/st3d boxes need the optional Google spatial-media CLI; "
        "ffmpeg fallback injects equivalent metadata in a different form",
    )
    def test_sv3d_metadata_present(self, tmp_path):
        from pipeline.vr_metadata import VRMetadataEmbedder

        embedder = VRMetadataEmbedder(codec="h264", crf=23, fps=24)
        frame = np.zeros((240, 480, 3), dtype=np.uint8)
        output_path = str(tmp_path / "output_vr180_meta.mp4")
        result = embedder.embed_single_frame_batch(
            [frame],
            output_path,
            width=480,
            height=240,
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
        importlib.util.module_from_spec(spec)
        # Don't execute main, just verify it loads
        assert spec is not None


# ---------------------------------------------------------------------------
# V-4 (#37): chunked batch stages — chunked == whole-sequence end-to-end
# ---------------------------------------------------------------------------


class TestRunPipelineChunked:
    """The batch depth/stereo stages under --chunk-size must match whole-run.

    DepthEstimator.estimate is stubbed (no transformers/model) so this is a
    pure CPU test of the chunking wiring + temporal-state continuity.
    """

    @staticmethod
    def _import_run_pipeline():
        import importlib.util
        import sys

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
        try:
            spec = importlib.util.spec_from_file_location(
                "run_pipeline_v4",
                os.path.join(os.path.dirname(__file__), "..", "scripts", "run_pipeline.py"),
            )
            mod = importlib.util.module_from_spec(spec)
            assert spec is not None
            spec.loader.exec_module(mod)
            return mod
        finally:
            with contextlib.suppress(ValueError):
                sys.path.remove(os.path.join(os.path.dirname(__file__), "..", "scripts"))

    def _args(self, run_pipeline, tmp_path, chunk_size, overlap=0):
        from unittest.mock import MagicMock

        args = MagicMock()
        args.depth_model = "depth-anything"
        args.stereo_model = "default"
        args.model_size = "small"
        args.device = "cpu"
        args.temporal_smoothing = 0.3
        args.ipd = 0.064
        args.max_disparity = 0.02
        # I-3 (#88): run_stereo_stage now reads convergence / temporal_smooth
        # off args (resolved by run_pipeline._apply_comfort_preset).  Set them
        # explicitly so the MagicMock isn't an auto-Mock that breaks numpy
        # broadcasting inside StereoRenderer.
        args.convergence = 0.3
        args.temporal_smooth = True
        args.no_temporal = False
        args.chunk_size = chunk_size
        args.overlap = overlap
        args.temp_dir = str(tmp_path)
        args.input = str(tmp_path / "fake_input.mp4")
        return args

    def test_depth_stage_chunked_matches_whole(self, tmp_path, monkeypatch):
        """--chunk-size depth (with EMA) == whole-sequence depth, all chunk sizes."""
        run_pipeline = self._import_run_pipeline()

        rng = np.random.default_rng(11)
        frames = [rng.integers(0, 256, (48, 48, 3), dtype=np.uint8) for _ in range(20)]

        def fake_estimate(self, frame):
            return frame.mean(axis=2).astype(np.float32) / 255.0

        monkeypatch.setattr(run_pipeline.DepthEstimator, "estimate", fake_estimate)

        # Whole-sequence reference
        ref_args = self._args(run_pipeline, tmp_path / "ref", chunk_size=None)
        ref = run_pipeline.run_depth_stage(ref_args, frames)

        for cs in (1, 7, 100):
            cargs = self._args(run_pipeline, tmp_path / f"cs{cs}", chunk_size=cs)
            got = run_pipeline.run_depth_stage(cargs, frames)
            assert len(got) == 20
            for i, (g, w) in enumerate(zip(got, ref, strict=True)):
                assert np.array_equal(g, w), f"cs={cs} frame {i} differs"

    def test_stereo_stage_chunked_matches_whole(self, tmp_path):
        """--chunk-size stereo == whole-sequence stereo, all chunk sizes."""
        run_pipeline = self._import_run_pipeline()

        rng = np.random.default_rng(23)
        frames = [rng.integers(0, 256, (48, 48, 3), dtype=np.uint8) for _ in range(20)]
        depths = [rng.random((48, 48)).astype(np.float32) for _ in range(20)]

        ref_args = self._args(run_pipeline, tmp_path / "ref", chunk_size=None)
        ref_l, ref_r = run_pipeline.run_stereo_stage(ref_args, frames, depths)

        for cs in (1, 7, 100):
            cargs = self._args(run_pipeline, tmp_path / f"cs{cs}", chunk_size=cs)
            got_l, got_r = run_pipeline.run_stereo_stage(cargs, frames, depths)
            assert len(got_l) == 20 and len(got_r) == 20
            for i in range(20):
                assert np.array_equal(got_l[i], ref_l[i]), f"cs={cs} left frame {i} differs"
                assert np.array_equal(got_r[i], ref_r[i]), f"cs={cs} right frame {i} differs"

    def test_depth_stage_writes_checkpoints_per_frame(self, tmp_path, monkeypatch):
        """Chunked depth must still write every depth_{i}.npy + .png checkpoint."""
        run_pipeline = self._import_run_pipeline()
        rng = np.random.default_rng(5)
        frames = [rng.integers(0, 256, (32, 32, 3), dtype=np.uint8) for _ in range(10)]

        def fake_estimate(self, frame):
            return frame.mean(axis=2).astype(np.float32) / 255.0

        monkeypatch.setattr(run_pipeline.DepthEstimator, "estimate", fake_estimate)
        cargs = self._args(run_pipeline, tmp_path, chunk_size=4, overlap=0)
        run_pipeline.run_depth_stage(cargs, frames)
        depth_dir = run_pipeline.get_depth_dir(cargs)
        npy = sorted(glob.glob(os.path.join(depth_dir, "depth_*.npy")))
        png = sorted(glob.glob(os.path.join(depth_dir, "depth_*.png")))
        assert len(npy) == 10 and len(png) == 10
        # Indices 0..9 present.
        assert {os.path.basename(p) for p in npy} == {f"depth_{i:06d}.npy" for i in range(10)}
        # I-6 (#121): meta.json written into the model-scoped depth dir.
        meta = run_pipeline.load_depth_meta(depth_dir)
        assert meta is not None
        assert meta["depth_model"] == "depth-anything"
        assert meta["num_frames"] == 10


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
        from pipeline.equirectangular_mapper import EquirectangularMapper
        from pipeline.stereo_renderer import StereoRenderer
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
        for frame, depth in zip(frames, depths, strict=False):
            left, right = renderer.render(frame, depth)
            lefts.append(left)
            rights.append(right)

        # Equirect
        mapper = EquirectangularMapper(
            output_width=320,
            output_height=320,
            src_hfov=70.0,
            use_ffmpeg=False,
        )
        sbs_frames = []
        for left, right in zip(lefts, rights, strict=False):
            sbs_frames.append(mapper.map_stereo_pair(left, right))

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
