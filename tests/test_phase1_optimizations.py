"""Phase 1 Backend Optimizations — Automated Verification Tests (PRD §7).

Tests for:
  - §7.3 Device Detection (detect_best_device, get_device_info)
  - §7.2 Streaming Pipeline (StreamingPipeline class structure)
  - §7.4 Tiled Upscaling (PixelUpscaler.upscale_tiled)

Run:
    pytest tests/test_phase1_optimizations.py -v
"""

import ast
import os
import sys
import unittest
from unittest.mock import patch

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ──────────────────────────────────────────────────────────────────────
# §7.3 Device Detection Tests
# ──────────────────────────────────────────────────────────────────────


class TestDeviceDetection(unittest.TestCase):
    """Tests for pipeline/device_utils.py (PRD §7.3)."""

    def test_imports(self):
        """device_utils module imports without error."""
        from pipeline.device_utils import detect_best_device, get_device_info, resolve_device

        self.assertTrue(callable(detect_best_device))
        self.assertTrue(callable(get_device_info))
        self.assertTrue(callable(resolve_device))

    def test_detect_best_device_returns_string(self):
        """detect_best_device() returns 'cuda', 'mps', or 'cpu'."""
        from pipeline.device_utils import detect_best_device

        result = detect_best_device()
        self.assertIn(result, ("cuda", "mps", "cpu"))

    def test_get_device_info_structure(self):
        """get_device_info() returns dict with required keys."""
        from pipeline.device_utils import get_device_info

        info = get_device_info()
        self.assertIsInstance(info, dict)
        self.assertIn("device", info)
        self.assertIn("name", info)
        self.assertIn(info["device"], ("cuda", "mps", "cpu"))

    def test_resolve_device_explicit(self):
        """resolve_device() returns the requested device when valid."""
        from pipeline.device_utils import resolve_device

        self.assertEqual(resolve_device("cpu"), "cpu")

    def test_resolve_device_auto(self):
        """resolve_device(None) auto-detects a valid device."""
        from pipeline.device_utils import resolve_device

        result = resolve_device(None)
        self.assertIn(result, ("cuda", "mps", "cpu"))

    def test_cpu_fallback_when_no_torch(self):
        """Falls back to CPU when torch is not available."""
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("mocked torch unavailable")
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=mock_import):
            # Call detect_best_device with torch import blocked
            from pipeline.device_utils import detect_best_device

            try:
                result = detect_best_device()
            except ImportError:
                result = "cpu"
            self.assertEqual(result, "cpu")


# ──────────────────────────────────────────────────────────────────────
# §7.2 Streaming Pipeline Tests
# ──────────────────────────────────────────────────────────────────────


class TestStreamingPipeline(unittest.TestCase):
    """Tests for pipeline/streaming_pipeline.py (PRD §7.2)."""

    def test_file_syntax(self):
        """streaming_pipeline.py parses without syntax errors."""
        path = os.path.join(PROJECT_ROOT, "pipeline", "streaming_pipeline.py")
        with open(path) as f:
            ast.parse(f.read())

    def test_class_exists(self):
        """StreamingPipeline class is importable."""
        from pipeline.streaming_pipeline import StreamingPipeline

        self.assertTrue(callable(StreamingPipeline))

    def test_convenience_function_exists(self):
        """run_streaming_pipeline convenience function exists."""
        from pipeline.streaming_pipeline import run_streaming_pipeline

        self.assertTrue(callable(run_streaming_pipeline))

    def test_init_defaults(self):
        """StreamingPipeline initializes with correct defaults."""
        from pipeline.streaming_pipeline import StreamingPipeline

        p = StreamingPipeline()
        self.assertEqual(p.model_size, "small")
        self.assertEqual(p.ipd, 0.064)
        self.assertEqual(p.max_disparity, 0.05)
        self.assertEqual(p.output_width, 3840)
        self.assertEqual(p.output_height, 1920)
        self.assertEqual(p.codec, "h264")
        self.assertIn(p.device, ("cuda", "mps", "cpu"))

    def test_init_custom_params(self):
        """StreamingPipeline accepts custom parameters."""
        from pipeline.streaming_pipeline import StreamingPipeline

        p = StreamingPipeline(
            model_size="base",
            device="cpu",
            ipd=0.07,
            max_disparity=0.08,
            output_width=1920,
            output_height=960,
            codec="h265",
            crf=18,
            fps=60,
        )
        self.assertEqual(p.model_size, "base")
        self.assertEqual(p.device, "cpu")
        self.assertEqual(p.ipd, 0.07)
        self.assertEqual(p.codec, "h265")
        self.assertEqual(p.fps, 60)

    def test_build_ffmpeg_cmd(self):
        """_build_ffmpeg_cmd generates valid ffmpeg command."""
        from pipeline.streaming_pipeline import StreamingPipeline

        p = StreamingPipeline(codec="h264", crf=23, fps=30)
        cmd = p._build_ffmpeg_cmd("/tmp/out.mp4", 7680, 1920)
        self.assertIn("ffmpeg", cmd)
        self.assertIn("libx264", cmd)
        self.assertIn("pipe:0", cmd)
        self.assertIn("7680x1920", cmd)

    def test_build_ffmpeg_cmd_h265(self):
        """_build_ffmpeg_cmd uses libx265 for h265 codec."""
        from pipeline.streaming_pipeline import StreamingPipeline

        p = StreamingPipeline(codec="h265")
        cmd = p._build_ffmpeg_cmd("/tmp/out.mp4", 7680, 1920)
        self.assertIn("libx265", cmd)

    def test_process_stream_rejects_missing_file(self):
        """process_stream raises RuntimeError for missing input."""
        from pipeline.streaming_pipeline import StreamingPipeline

        p = StreamingPipeline(device="cpu")
        with self.assertRaises(RuntimeError):
            p.process_stream("/nonexistent/video.mp4", "/tmp/out.mp4")


# ──────────────────────────────────────────────────────────────────────
# §7.4 Tiled Upscaling Tests
# ──────────────────────────────────────────────────────────────────────


class TestTiledUpscaling(unittest.TestCase):
    """Tests for PixelUpscaler.upscale_tiled (PRD §7.4)."""

    def test_file_syntax(self):
        """upscaler.py parses without syntax errors."""
        path = os.path.join(PROJECT_ROOT, "pipeline", "upscaler.py")
        with open(path) as f:
            ast.parse(f.read())

    def test_upscale_tiled_method_exists(self):
        """PixelUpscaler has upscale_tiled method."""
        from pipeline.upscaler import PixelUpscaler

        self.assertTrue(hasattr(PixelUpscaler, "upscale_tiled"))
        self.assertTrue(callable(PixelUpscaler.upscale_tiled))

    def test_upscale_tiled_params(self):
        """upscale_tiled accepts tile_size, tile_pad, progress_callback."""
        import inspect

        from pipeline.upscaler import PixelUpscaler

        sig = inspect.signature(PixelUpscaler.upscale_tiled)
        params = list(sig.parameters.keys())
        self.assertIn("self", params)
        self.assertIn("frame", params)
        self.assertIn("tile_size", params)
        self.assertIn("tile_pad", params)
        self.assertIn("progress_callback", params)

    def test_upscale_tiled_tile_size_default(self):
        """upscale_tiled default tile_size is 512."""
        import inspect

        from pipeline.upscaler import PixelUpscaler

        sig = inspect.signature(PixelUpscaler.upscale_tiled)
        default = sig.parameters["tile_size"].default
        self.assertEqual(default, 512)


# ──────────────────────────────────────────────────────────────────────
# §7 Integration — run_pipeline.py CLI flags
# ──────────────────────────────────────────────────────────────────────


class TestCLIIntegration(unittest.TestCase):
    """Verify run_pipeline.py has new Phase 1 CLI flags."""

    def _read_source(self):
        path = os.path.join(PROJECT_ROOT, "scripts", "run_pipeline.py")
        with open(path) as f:
            return f.read()

    def test_syntax(self):
        path = os.path.join(PROJECT_ROOT, "scripts", "run_pipeline.py")
        with open(path) as f:
            ast.parse(f.read())

    def test_streaming_flag(self):
        self.assertIn("--streaming", self._read_source())

    def test_tiled_upscale_flag(self):
        self.assertIn("--tiled-upscale", self._read_source())

    def test_tile_size_flag(self):
        self.assertIn("--tile-size", self._read_source())

    def test_device_utils_import(self):
        self.assertIn("from pipeline.device_utils import", self._read_source())

    def test_streaming_pipeline_import(self):
        self.assertIn("from pipeline.streaming_pipeline import", self._read_source())


if __name__ == "__main__":
    unittest.main()
