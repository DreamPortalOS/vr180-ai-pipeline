"""Tests for issue #45 — streaming path triple fix (V-1 acceptance findings).

Covers:
  - Defect 1: SBS layout — ffmpeg ``-s`` must be ``{2*output_width}x{output_height}``
    (horizontal concat), not the old top-bottom declaration.
  - Defect 2: streaming CLI branch injects sv3d/st3d metadata after encoding.
  - Defect 3: ``select_encoder`` pure function (hw/no-hw × large/small frames),
    stderr no longer a deadlock-prone PIPE, BrokenPipeError handling, and
    non-zero ffmpeg returncode raising RuntimeError.

All tests are CPU-only; subprocess / cv2 / model stages are mocked.
"""

import contextlib
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pipeline.streaming_pipeline import select_encoder  # noqa: E402


def _make_pipeline(**kwargs):
    """Build a StreamingPipeline with all heavy stages mocked out."""
    with (
        patch("pipeline.streaming_pipeline.DepthEstimator"),
        patch("pipeline.streaming_pipeline.StereoRenderer"),
        patch("pipeline.streaming_pipeline.EquirectangularMapper"),
    ):
        from pipeline.streaming_pipeline import StreamingPipeline

        kwargs.setdefault("device", "cpu")
        return StreamingPipeline(**kwargs)


class TestSelectEncoder(unittest.TestCase):
    """Pure encoder-selection function: 4 combos + libx264 large-frame refusal."""

    def test_hw_small_frame_h264(self):
        self.assertEqual(select_encoder("h264", 3840, hw=True), ["-c:v", "h264_nvenc"])

    def test_hw_small_frame_h265(self):
        self.assertEqual(select_encoder("h265", 3840, hw=True), ["-c:v", "hevc_nvenc"])

    def test_hw_large_frame_forces_hevc(self):
        # H.264 NVENC caps at 4096 wide — 8K SBS must go HEVC even if h264 asked.
        self.assertEqual(select_encoder("h264", 7680, hw=True), ["-c:v", "hevc_nvenc"])
        self.assertEqual(select_encoder("h265", 7680, hw=True), ["-c:v", "hevc_nvenc"])

    def test_sw_small_frame_h264(self):
        self.assertEqual(select_encoder("h264", 3840, hw=False), ["-c:v", "libx264"])

    def test_sw_small_frame_h265(self):
        self.assertEqual(
            select_encoder("h265", 3840, hw=False),
            ["-c:v", "libx265", "-preset", "fast"],
        )

    def test_sw_large_frame_rejects_libx264(self):
        # libx264 on a >4096-wide SBS frame OOMs — must be forced to libx265.
        args = select_encoder("h264", 7680, hw=False)
        self.assertEqual(args, ["-c:v", "libx265", "-preset", "fast"])
        self.assertNotIn("libx264", args)

    def test_boundary_4096_is_allowed_for_h264(self):
        self.assertEqual(select_encoder("h264", 4096, hw=True), ["-c:v", "h264_nvenc"])
        self.assertEqual(select_encoder("h264", 4096, hw=False), ["-c:v", "libx264"])


class TestSbsLayout(unittest.TestCase):
    """Defect 1: ffmpeg -s must match the horizontal SBS concat (2W × H)."""

    def test_ffmpeg_cmd_size_is_horizontal_sbs(self):
        p = _make_pipeline(output_width=3840, output_height=3840, bitrate="80M")
        # process_stream computes out_w = 2*output_width, out_h = output_height
        cmd = p._build_ffmpeg_cmd("out.mp4", p.output_width * 2, p.output_height)
        s_idx = cmd.index("-s")
        self.assertEqual(cmd[s_idx + 1], "7680x3840")

    def test_open_ffmpeg_writer_receives_horizontal_sbs_size(self):
        """The writer opened by process_stream must declare 2W×H (mock ffmpeg)."""
        import numpy as np

        p = _make_pipeline(output_width=100, output_height=50)
        p.depth_estimator.estimate.return_value = np.zeros((10, 10), dtype=np.float32)
        p.stereo_renderer.render.return_value = (
            np.zeros((10, 10, 3), dtype=np.uint8),
            np.zeros((10, 10, 3), dtype=np.uint8),
        )
        p.eq_mapper.map_stereo_pair.return_value = np.zeros((50, 200, 3), dtype=np.uint8)

        fake_frame = np.zeros((8, 8, 3), dtype=np.uint8)
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.get.side_effect = lambda prop: {
            3: 8,  # CAP_PROP_FRAME_WIDTH
            4: 8,  # CAP_PROP_FRAME_HEIGHT
            7: 1.0,  # CAP_PROP_FRAME_COUNT
            5: 30.0,  # CAP_PROP_FPS
        }.get(prop, 0.0)
        cap.read.side_effect = [(True, fake_frame), (False, None)]

        proc = MagicMock()
        proc.returncode = 0

        with (
            patch("pipeline.streaming_pipeline.cv2.VideoCapture", return_value=cap),
            patch("pipeline.streaming_pipeline.subprocess.Popen", return_value=proc) as popen_mock,
        ):
            p.process_stream("in.mp4", "out.mp4")

        cmd = popen_mock.call_args[0][0]
        s_idx = cmd.index("-s")
        self.assertEqual(cmd[s_idx + 1], "200x50")  # 2*100 x 50 — horizontal SBS

    def test_stderr_is_devnull_not_pipe(self):
        """Defect 3 / issue #49: stderr must be drained — a file, not PIPE/DEVNULL.

        DEVNULL hid fatal encoder errors (NVENC driver mismatch was invisible);
        an undrained PIPE deadlocks. The writer now sends stderr to a temp file
        so failures can report the ffmpeg error tail.
        """
        import subprocess

        p = _make_pipeline(output_width=100, output_height=50)
        with patch("pipeline.streaming_pipeline.subprocess.Popen") as popen_mock:
            p._open_ffmpeg_writer("out.mp4", 200, 50)
        kwargs = popen_mock.call_args[1]
        self.assertNotEqual(kwargs["stderr"], subprocess.PIPE)
        self.assertNotEqual(kwargs["stderr"], subprocess.DEVNULL)
        self.assertTrue(hasattr(kwargs["stderr"], "write"))  # a real file object


class TestFfmpegFailureHandling(unittest.TestCase):
    """Defect 3: BrokenPipeError and non-zero returncode must surface as errors."""

    def _pipeline_with_mocks(self):
        import numpy as np

        p = _make_pipeline(output_width=100, output_height=50)
        p.depth_estimator.estimate.return_value = np.zeros((10, 10), dtype=np.float32)
        p.stereo_renderer.render.return_value = (
            np.zeros((10, 10, 3), dtype=np.uint8),
            np.zeros((10, 10, 3), dtype=np.uint8),
        )
        p.eq_mapper.map_stereo_pair.return_value = np.zeros((50, 200, 3), dtype=np.uint8)

        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.get.side_effect = lambda prop: {3: 8, 4: 8, 7: 1.0, 5: 30.0}.get(prop, 0.0)
        cap.read.side_effect = [(True, np.zeros((8, 8, 3), dtype=np.uint8)), (False, None)]
        return p, cap

    def test_nonzero_returncode_raises_runtime_error(self):
        p, cap = self._pipeline_with_mocks()
        proc = MagicMock()
        proc.returncode = 1

        with (
            patch("pipeline.streaming_pipeline.cv2.VideoCapture", return_value=cap),
            patch("pipeline.streaming_pipeline.subprocess.Popen", return_value=proc),
            self.assertRaises(RuntimeError) as ctx,
        ):
            p.process_stream("in.mp4", "out.mp4")
        self.assertIn("exit code 1", str(ctx.exception))

    def test_broken_pipe_raises_runtime_error(self):
        p, cap = self._pipeline_with_mocks()
        proc = MagicMock()
        proc.stdin.write.side_effect = BrokenPipeError("pipe closed")
        proc.returncode = -9

        with (
            patch("pipeline.streaming_pipeline.cv2.VideoCapture", return_value=cap),
            patch("pipeline.streaming_pipeline.subprocess.Popen", return_value=proc),
            self.assertRaises(RuntimeError) as ctx,
        ):
            p.process_stream("in.mp4", "out.mp4")
        self.assertIn("ffmpeg encoder died", str(ctx.exception))


class TestStreamingMetadataInjection(unittest.TestCase):
    """Defect 2: streaming CLI branch injects sv3d/st3d after encoding."""

    def _run_main(self, inject_mock):
        sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
        try:
            import run_pipeline

            args = MagicMock()
            args.video_upscale = "none"
            args.device = "cpu"
            args.validate_input = False
            args.fps = 30
            args.streaming = True
            args.stage = "all"
            args.projection = "vr180"
            args.model_size = "small"
            args.ipd = 0.064
            args.max_disparity = 0.05
            args.output_width = 2880
            args.output_height = 2880
            args.src_hfov = 70.0
            args.codec = "h264"
            args.crf = 23
            args.bitrate = "45M"
            args.input = "in.mp4"
            args.output = "out.mp4"
            args.max_frames = None
            # I-3 (#88): main() now resolves --comfort via _apply_comfort_preset;
            # set the fields it reads so the MagicMock does not auto-stub them.
            args.comfort = "balanced"
            args.convergence = None
            args.no_temporal = False
            # D-2 (#79): main() now resolves --preset via apply_playback_preset;
            # set source (passthrough) + gop=None so the resolution is a no-op
            # on the MagicMock.
            args.preset = "source"
            args.gop = None

            pipeline_inst = MagicMock()
            pipeline_inst.process_stream.return_value = "out.mp4"

            with (
                patch.object(run_pipeline, "parse_args", return_value=args),
                patch.object(run_pipeline, "apply_quality_preset"),
                patch.object(run_pipeline, "StreamingPipeline", return_value=pipeline_inst),
                patch("pipeline.spherical_injector.inject_spherical_metadata", inject_mock),
                patch("os.replace") as replace_mock,
            ):
                run_pipeline.main()
            return replace_mock
        finally:
            with contextlib.suppress(ValueError):
                sys.path.remove(os.path.join(PROJECT_ROOT, "scripts"))
            sys.modules.pop("run_pipeline", None)

    def test_streaming_branch_injects_metadata(self):
        inject_mock = MagicMock()
        replace_mock = self._run_main(inject_mock)
        inject_mock.assert_called_once_with(
            "out.mp4",
            "out.mp4.vr.mp4",
            width=5760,  # 2 × output_width (horizontal SBS)
            height=2880,
            stereo_mode="sbs",
        )
        replace_mock.assert_called_once_with("out.mp4.vr.mp4", "out.mp4")

    def test_injection_failure_raises(self):
        inject_mock = MagicMock(side_effect=RuntimeError("spatialmedia boom"))
        with self.assertRaises(RuntimeError):
            self._run_main(inject_mock)


if __name__ == "__main__":
    unittest.main()
