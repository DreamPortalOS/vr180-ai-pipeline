"""Tests for issue #49 — NVENC availability probing + software fallback.

Covers:
  - ``probe_nvenc``: success / failure / ffmpeg-missing, plus per-process caching
    (only one subprocess per encoder per process).
  - Auto fallback: CUDA device + failed probe → software encoder (libx265 for
    large frames), with a warning mentioning the driver upgrade.
  - ``--hw-encoder`` tri-state passthrough (auto/on/off) from the CLI to
    StreamingPipeline.
  - OSError(errno=22) on stdin.write (Windows broken pipe) → RuntimeError with
    the ffmpeg stderr summary.

All tests are CPU-only and machine-independent; subprocess is fully mocked and
the probe cache is cleared around each test.
"""

import contextlib
import os
import subprocess
import sys
import unittest
from unittest.mock import MagicMock, patch

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pipeline.streaming_pipeline as sp  # noqa: E402
from pipeline.streaming_pipeline import StreamingPipeline, probe_nvenc  # noqa: E402


def _clear_probe_cache():
    sp._NVENC_PROBE_CACHE.clear()


def _make_pipeline(**kwargs):
    """Build a StreamingPipeline with all heavy stages mocked out.

    ``resolve_device`` is patched to a pass-through so tests can request
    device="cuda" on machines without CUDA/torch (CI determinism).
    """
    with (
        patch("pipeline.streaming_pipeline.DepthEstimator"),
        patch("pipeline.streaming_pipeline.StereoRenderer"),
        patch("pipeline.streaming_pipeline.EquirectangularMapper"),
        patch("pipeline.streaming_pipeline.resolve_device", side_effect=lambda d: d or "cpu"),
    ):
        kwargs.setdefault("device", "cpu")
        return StreamingPipeline(**kwargs)


class TestProbeNvenc(unittest.TestCase):
    """probe_nvenc: mocked subprocess, three outcomes + caching."""

    def setUp(self):
        _clear_probe_cache()

    tearDown = setUp

    def _completed(self, returncode=0, stderr=""):
        r = MagicMock(spec=subprocess.CompletedProcess)
        r.returncode = returncode
        r.stderr = stderr
        r.stdout = ""
        return r

    def test_probe_success(self):
        with patch.object(sp.subprocess, "run", return_value=self._completed(0)) as run_mock:
            self.assertTrue(probe_nvenc("hevc_nvenc"))
        run_mock.assert_called_once()
        cmd = run_mock.call_args[0][0]
        # list-form argv, tiny synthetic encode with the target encoder.
        self.assertIsInstance(cmd, list)
        self.assertIn("hevc_nvenc", cmd)
        self.assertIn("lavfi", cmd)
        self.assertIn("null", cmd)

    def test_probe_failure(self):
        with patch.object(
            sp.subprocess,
            "run",
            return_value=self._completed(1, "Driver does not support the required nvenc API version"),
        ):
            self.assertFalse(probe_nvenc("hevc_nvenc"))

    def test_probe_ffmpeg_missing(self):
        with patch.object(sp.subprocess, "run", side_effect=FileNotFoundError("ffmpeg not found")):
            self.assertFalse(probe_nvenc("hevc_nvenc"))

    def test_probe_result_cached_per_encoder(self):
        with patch.object(sp.subprocess, "run", return_value=self._completed(0)) as run_mock:
            self.assertTrue(probe_nvenc("hevc_nvenc"))
            self.assertTrue(probe_nvenc("hevc_nvenc"))
            self.assertTrue(probe_nvenc("hevc_nvenc"))
        # Same encoder probed once per process.
        run_mock.assert_called_once()

    def test_probe_cache_keyed_by_encoder(self):
        with patch.object(sp.subprocess, "run", return_value=self._completed(0)) as run_mock:
            probe_nvenc("hevc_nvenc")
            probe_nvenc("h264_nvenc")
        self.assertEqual(run_mock.call_count, 2)


class TestAutoFallback(unittest.TestCase):
    """hw_encoder=None (auto): CUDA + probe result decides NVENC vs software."""

    def setUp(self):
        _clear_probe_cache()

    tearDown = setUp

    def test_auto_cuda_probe_ok_uses_nvenc(self):
        with patch("pipeline.streaming_pipeline.probe_nvenc", return_value=True):
            p = _make_pipeline(device="cuda", codec="h264", output_width=1920, output_height=1920)
        self.assertTrue(p.hw_encoder)
        cmd = p._build_ffmpeg_cmd("out.mp4", 3840, 1920)
        self.assertIn("h264_nvenc", cmd)

    def test_auto_cuda_probe_fail_falls_back_to_software(self):
        with (
            patch("pipeline.streaming_pipeline.probe_nvenc", return_value=False),
            self.assertLogs("vr180-streaming", level="WARNING") as logs,
        ):
            p = _make_pipeline(device="cuda", codec="h264", output_width=1920, output_height=1920)
        self.assertFalse(p.hw_encoder)
        cmd = p._build_ffmpeg_cmd("out.mp4", 3840, 1920)
        self.assertIn("libx264", cmd)
        self.assertNotIn("nvenc", " ".join(cmd))
        warning_text = "\n".join(logs.output)
        self.assertIn("NVENC", warning_text)
        self.assertIn("610", warning_text)  # 升级 NVIDIA 驱动 ≥610 可启用硬编

    def test_auto_cuda_probe_fail_large_frame_uses_libx265(self):
        with patch("pipeline.streaming_pipeline.probe_nvenc", return_value=False):
            p = _make_pipeline(device="cuda", codec="h264", output_width=3840, output_height=3840)
        self.assertFalse(p.hw_encoder)
        cmd = p._build_ffmpeg_cmd("out.mp4", 7680, 3840)
        self.assertIn("libx265", cmd)
        self.assertIn("fast", cmd)

    def test_auto_non_cuda_never_probes(self):
        with patch("pipeline.streaming_pipeline.probe_nvenc") as probe_mock:
            p = _make_pipeline(device="cpu", codec="h264", output_width=1920, output_height=1920)
        self.assertFalse(p.hw_encoder)
        probe_mock.assert_not_called()

    def test_on_forces_nvenc_without_probe(self):
        with patch("pipeline.streaming_pipeline.probe_nvenc") as probe_mock:
            p = _make_pipeline(device="cuda", codec="h264", output_width=1920, output_height=1920, hw_encoder="on")
        self.assertTrue(p.hw_encoder)
        probe_mock.assert_not_called()

    def test_off_forces_software_without_probe(self):
        with patch("pipeline.streaming_pipeline.probe_nvenc") as probe_mock:
            p = _make_pipeline(device="cuda", codec="h264", output_width=1920, output_height=1920, hw_encoder="off")
        self.assertFalse(p.hw_encoder)
        probe_mock.assert_not_called()

    def test_bool_true_backward_compatible(self):
        with patch("pipeline.streaming_pipeline.probe_nvenc") as probe_mock:
            p = _make_pipeline(device="cuda", codec="h264", output_width=1920, output_height=1920, hw_encoder=True)
        self.assertTrue(p.hw_encoder)
        probe_mock.assert_not_called()

    def test_bool_false_backward_compatible(self):
        p = _make_pipeline(device="cuda", codec="h264", output_width=1920, output_height=1920, hw_encoder=False)
        self.assertFalse(p.hw_encoder)


class TestHwEncoderCliPassthrough(unittest.TestCase):
    """--hw-encoder {auto,on,off} reaches StreamingPipeline from the CLI."""

    def _run_main(self, hw_encoder_value):
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
            args.hw_encoder = hw_encoder_value
            # I-3 (#88): main() now resolves --comfort via _apply_comfort_preset,
            # which reads comfort / convergence / no_temporal off args.  Set them
            # so the MagicMock does not auto-stub to a Mock (truthy / non-None).
            args.comfort = "balanced"
            args.convergence = None
            args.no_temporal = False
            # D-2 (#79): main() now resolves --preset via apply_playback_preset,
            # which reads preset / gop / codec / crf / bitrate off args.  Set
            # source (passthrough) + gop=None so the resolution is a no-op on
            # the MagicMock (does not auto-stub to a Mock).
            args.preset = "source"
            args.gop = None
            # H-1.2 (#132): main() now remuxes audio after sv3d/st3d injection;
            # pin the flag off and stub the passthrough helper so no real
            # ffmpeg/ffprobe runs here.
            args.copy_audio_from = None

            pipeline_inst = MagicMock()
            pipeline_inst.process_stream.return_value = "out.mp4"

            with (
                patch.object(run_pipeline, "parse_args", return_value=args),
                patch.object(run_pipeline, "apply_quality_preset"),
                patch.object(run_pipeline, "StreamingPipeline", return_value=pipeline_inst) as sp_mock,
                patch.object(run_pipeline, "_maybe_copy_audio_from_input"),
                patch("pipeline.spherical_injector.inject_spherical_metadata"),
                patch("os.replace"),
            ):
                run_pipeline.main()
            return sp_mock
        finally:
            with contextlib.suppress(ValueError):
                sys.path.remove(os.path.join(PROJECT_ROOT, "scripts"))
            sys.modules.pop("run_pipeline", None)

    def test_auto_passthrough(self):
        sp_mock = self._run_main("auto")
        self.assertEqual(sp_mock.call_args.kwargs["hw_encoder"], "auto")

    def test_on_passthrough(self):
        sp_mock = self._run_main("on")
        self.assertEqual(sp_mock.call_args.kwargs["hw_encoder"], "on")

    def test_off_passthrough(self):
        sp_mock = self._run_main("off")
        self.assertEqual(sp_mock.call_args.kwargs["hw_encoder"], "off")

    def test_cli_flag_default_is_auto(self):
        sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
        try:
            import run_pipeline

            args = run_pipeline.parse_args(["--input", "in.mp4"])
            self.assertEqual(args.hw_encoder, "auto")
            args = run_pipeline.parse_args(["--input", "in.mp4", "--hw-encoder", "off"])
            self.assertEqual(args.hw_encoder, "off")
        finally:
            with contextlib.suppress(ValueError):
                sys.path.remove(os.path.join(PROJECT_ROOT, "scripts"))
            sys.modules.pop("run_pipeline", None)


class TestOSErrorPipeHandling(unittest.TestCase):
    """Issue #49: OSError(errno=22) on stdin.write → RuntimeError with stderr tail."""

    def _pipeline_with_mocks(self):
        import numpy as np

        p = _make_pipeline(device="cpu", output_width=100, output_height=50)
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

    @contextlib.contextmanager
    def _run_with_proc(self, p, cap, proc, stderr_text: bytes | None = None):
        """Patch cv2/Popen for process_stream with a mocked ffmpeg proc.

        ``stderr_text`` is written into the temp stderr file that
        _open_ffmpeg_writer attaches, simulating ffmpeg's own error output.
        Yields a holder dict exposing the attached stderr file object.
        """
        holder = {}

        def popen_factory(*args, **kwargs):
            # _open_ffmpeg_writer passes the temp stderr file as the stderr=
            # kwarg and attaches it to the proc after Popen returns.
            stderr_file = kwargs["stderr"]
            if stderr_text is not None:
                stderr_file.write(stderr_text)
                stderr_file.flush()
            holder["stderr_file"] = stderr_file
            return proc

        with (
            patch("pipeline.streaming_pipeline.cv2.VideoCapture", return_value=cap),
            patch("pipeline.streaming_pipeline.subprocess.Popen", side_effect=popen_factory),
        ):
            yield holder

    def _make_proc(self, write_side_effect=None, returncode=1):
        proc = MagicMock()
        if write_side_effect is not None:
            proc.stdin.write.side_effect = write_side_effect
        proc.returncode = returncode
        proc.poll.return_value = returncode
        return proc

    def test_oserror_22_raises_runtime_error_with_stderr_summary(self):
        p, cap = self._pipeline_with_mocks()
        proc = self._make_proc(OSError(22, "Invalid argument"))

        with (
            self._run_with_proc(p, cap, proc, stderr_text=b"Driver does not support the required nvenc API version"),
            self.assertRaises(RuntimeError) as ctx,
        ):
            p.process_stream("in.mp4", "out.mp4")
        msg = str(ctx.exception)
        self.assertIn("ffmpeg encoder died", msg)
        self.assertIn("errno=22", msg)
        self.assertIn("nvenc API version", msg)  # stderr tail surfaced
        self.assertIn("exit code 1", msg)

    def test_oserror_32_raises_runtime_error(self):
        p, cap = self._pipeline_with_mocks()
        # errno=32 (EPIPE) subclasses BrokenPipeError — either wrapper is fine,
        # but the failure must surface as RuntimeError, not a raw OSError.
        proc = self._make_proc(OSError(32, "Broken pipe"))

        with self._run_with_proc(p, cap, proc), self.assertRaises(RuntimeError) as ctx:
            p.process_stream("in.mp4", "out.mp4")
        self.assertIn("ffmpeg encoder died", str(ctx.exception))

    def test_unrelated_oserror_reraises(self):
        p, cap = self._pipeline_with_mocks()
        proc = self._make_proc(OSError(28, "No space left on device"))

        with self._run_with_proc(p, cap, proc), self.assertRaises(OSError) as ctx:
            p.process_stream("in.mp4", "out.mp4")
        self.assertEqual(ctx.exception.errno, 28)

    def test_broken_pipe_error_includes_stderr_summary(self):
        p, cap = self._pipeline_with_mocks()
        proc = self._make_proc(BrokenPipeError("pipe closed"))

        with (
            self._run_with_proc(p, cap, proc, stderr_text=b"nvenc API version mismatch"),
            self.assertRaises(RuntimeError) as ctx,
        ):
            p.process_stream("in.mp4", "out.mp4")
        self.assertIn("nvenc API version", str(ctx.exception))

    def test_nonzero_returncode_includes_stderr_summary(self):
        p, cap = self._pipeline_with_mocks()
        proc = self._make_proc(returncode=1)

        with (
            self._run_with_proc(p, cap, proc, stderr_text=b"Error initializing output stream: nvenc boom"),
            self.assertRaises(RuntimeError) as ctx,
        ):
            p.process_stream("in.mp4", "out.mp4")
        msg = str(ctx.exception)
        self.assertIn("exit code 1", msg)
        self.assertIn("nvenc boom", msg)

    def test_success_path_closes_stderr_file(self):
        p, cap = self._pipeline_with_mocks()
        proc = self._make_proc(returncode=0)

        with self._run_with_proc(p, cap, proc) as holder:
            p.process_stream("in.mp4", "out.mp4")
        self.assertTrue(holder["stderr_file"].closed)


if __name__ == "__main__":
    unittest.main()
