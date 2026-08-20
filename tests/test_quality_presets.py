"""Tests for --quality presets and adaptive bitrate scaling (issue #34).

Covers:
  - pipeline.streaming_pipeline.resolve_quality (preset → resolution/streaming)
  - pipeline.streaming_pipeline.scaled_bitrate_mbps (pure bitrate scaling fn)
  - scripts.run_pipeline CLI wiring (--quality in --help, defaults, overrides)

All tests are CPU-only and mock ffmpeg/model dependencies.
"""

import os
import sys
import unittest
from types import SimpleNamespace

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pipeline.streaming_pipeline import (  # noqa: E402
    BASELINE_BITRATE_MBPS,
    DEFAULT_QUALITY,
    QUALITY_PRESETS,
    resolve_quality,
    scaled_bitrate_mbps,
)


class TestQualityPresets(unittest.TestCase):
    """Preset table sanity."""

    def test_preset_values(self):
        self.assertEqual(QUALITY_PRESETS["preview"], 1920)
        self.assertEqual(QUALITY_PRESETS["standard"], 2880)
        self.assertEqual(QUALITY_PRESETS["high"], 3840)

    def test_default_quality_is_standard(self):
        self.assertEqual(DEFAULT_QUALITY, "standard")


class TestResolveQuality(unittest.TestCase):
    """Preset → (eye_size, streaming) mapping."""

    def test_preview(self):
        self.assertEqual(resolve_quality("preview"), (1920, False))

    def test_standard_streams(self):
        self.assertEqual(resolve_quality("standard"), (2880, True))

    def test_high_streams(self):
        self.assertEqual(resolve_quality("high"), (3840, True))

    def test_explicit_eye_size_overrides_preset(self):
        eye_size, streaming = resolve_quality("high", explicit_eye_size=2560)
        self.assertEqual(eye_size, 2560)
        self.assertTrue(streaming)  # streaming flag still comes from preset

    def test_unknown_preset_raises(self):
        with self.assertRaises(ValueError):
            resolve_quality("ultra")


class TestScaledBitrate(unittest.TestCase):
    """Pure bitrate scaling function."""

    def test_reference_resolution_is_baseline(self):
        self.assertAlmostEqual(scaled_bitrate_mbps(1920), BASELINE_BITRATE_MBPS)

    def test_3840_is_4x_baseline(self):
        self.assertAlmostEqual(scaled_bitrate_mbps(3840), 4 * BASELINE_BITRATE_MBPS)

    def test_2880_is_2_25x_baseline(self):
        self.assertAlmostEqual(scaled_bitrate_mbps(2880), 2.25 * BASELINE_BITRATE_MBPS)

    def test_max_cap_truncates(self):
        self.assertEqual(scaled_bitrate_mbps(3840, max_mbps=50.0), 50.0)

    def test_cap_not_applied_below_limit(self):
        self.assertAlmostEqual(scaled_bitrate_mbps(1920, max_mbps=100.0), BASELINE_BITRATE_MBPS)

    def test_custom_base(self):
        self.assertAlmostEqual(scaled_bitrate_mbps(1920, base_mbps=10.0), 10.0)
        self.assertAlmostEqual(scaled_bitrate_mbps(3840, base_mbps=10.0), 40.0)


class TestCliQualityFlag(unittest.TestCase):
    """CLI wiring in scripts/run_pipeline.py."""

    def _parse(self, argv):
        sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
        try:
            import run_pipeline

            return run_pipeline.parse_args(argv)
        finally:
            sys.path.remove(os.path.join(PROJECT_ROOT, "scripts"))
            sys.modules.pop("run_pipeline", None)

    def test_quality_default_is_standard(self):
        args = self._parse(["--input", "x.mp4"])
        self.assertEqual(args.quality, "standard")

    def test_quality_choices_accepted(self):
        for q in ("preview", "standard", "high"):
            args = self._parse(["--input", "x.mp4", "--quality", q])
            self.assertEqual(args.quality, q)

    def test_help_lists_quality(self):
        import subprocess

        env = dict(os.environ, PYTHONPATH=PROJECT_ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""))
        result = subprocess.run(
            [sys.executable, os.path.join(PROJECT_ROOT, "scripts", "run_pipeline.py"), "--help"],
            capture_output=True,
            text=True,
            env=env,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("--quality", result.stdout)
        self.assertIn("preview", result.stdout)
        self.assertIn("standard", result.stdout)
        self.assertIn("high", result.stdout)


class TestApplyQualityPreset(unittest.TestCase):
    """apply_quality_preset: preset fills defaults, explicit flags win."""

    def _apply(self, **overrides):
        sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
        try:
            import run_pipeline

            args = SimpleNamespace(
                quality="standard",
                output_width=None,
                output_height=None,
                streaming=False,
                projection="vr180",
                bitrate=None,
                max_bitrate=200.0,
            )
            for k, v in overrides.items():
                setattr(args, k, v)
            run_pipeline.apply_quality_preset(args)
            return args
        finally:
            sys.path.remove(os.path.join(PROJECT_ROOT, "scripts"))
            sys.modules.pop("run_pipeline", None)

    def test_standard_enables_streaming_2880(self):
        args = self._apply()
        self.assertEqual(args.output_width, 2880)
        self.assertEqual(args.output_height, 2880)
        self.assertTrue(args.streaming)
        self.assertEqual(args.bitrate, "45M")  # 20 × 2.25

    def test_high_enables_streaming_3840(self):
        args = self._apply(quality="high")
        self.assertEqual(args.output_width, 3840)
        self.assertEqual(args.output_height, 3840)
        self.assertTrue(args.streaming)
        self.assertEqual(args.bitrate, "80M")  # 20 × 4

    def test_preview_keeps_batch_path(self):
        args = self._apply(quality="preview")
        self.assertEqual(args.output_width, 1920)
        self.assertEqual(args.output_height, 1920)
        self.assertFalse(args.streaming)
        self.assertEqual(args.bitrate, "20M")

    def test_explicit_output_width_overrides_preset(self):
        args = self._apply(quality="high", output_width=2560)
        self.assertEqual(args.output_width, 2560)
        self.assertEqual(args.output_height, 2560)
        self.assertTrue(args.streaming)

    def test_explicit_output_height_preserved(self):
        args = self._apply(output_height=1440)
        self.assertEqual(args.output_width, 2880)
        self.assertEqual(args.output_height, 1440)

    def test_explicit_streaming_flag_preserved(self):
        args = self._apply(quality="preview", streaming=True)
        self.assertTrue(args.streaming)

    def test_explicit_bitrate_preserved(self):
        args = self._apply(bitrate="123M")
        self.assertEqual(args.bitrate, "123M")

    def test_bitrate_capped_by_max_bitrate(self):
        args = self._apply(quality="high", max_bitrate=50.0)
        self.assertEqual(args.bitrate, "50M")

    def test_fulldome_not_forced_to_streaming(self):
        args = self._apply(quality="high", projection="fulldome")
        self.assertFalse(args.streaming)


class TestStreamingPipelineBitrate(unittest.TestCase):
    """StreamingPipeline ffmpeg cmd honours bitrate override (no real ffmpeg)."""

    def test_bitrate_overrides_crf_in_cmd(self):
        from unittest.mock import patch

        with (
            patch("pipeline.streaming_pipeline.DepthEstimator"),
            patch("pipeline.streaming_pipeline.StereoRenderer"),
            patch("pipeline.streaming_pipeline.EquirectangularMapper"),
        ):
            from pipeline.streaming_pipeline import StreamingPipeline

            p = StreamingPipeline(codec="h264", crf=23, fps=30, bitrate="80M")
            cmd = p._build_ffmpeg_cmd("out.mp4", 7680, 3840)
        self.assertIn("-b:v", cmd)
        self.assertIn("80M", cmd)
        self.assertNotIn("-crf", cmd)

    def test_crf_used_when_no_bitrate(self):
        from unittest.mock import patch

        with (
            patch("pipeline.streaming_pipeline.DepthEstimator"),
            patch("pipeline.streaming_pipeline.StereoRenderer"),
            patch("pipeline.streaming_pipeline.EquirectangularMapper"),
        ):
            from pipeline.streaming_pipeline import StreamingPipeline

            p = StreamingPipeline(codec="h265", crf=18, fps=30)
            cmd = p._build_ffmpeg_cmd("out.mp4", 7680, 3840)
        self.assertIn("-crf", cmd)
        self.assertIn("18", cmd)
        self.assertNotIn("-b:v", cmd)


class TestVRMetadataBitrate(unittest.TestCase):
    """VRMetadataEmbedder accepts bitrate and prefers it over CRF."""

    def test_bitrate_stored(self):
        from pipeline.vr_metadata import VRMetadataEmbedder

        e = VRMetadataEmbedder(codec="h265", bitrate="45M")
        self.assertEqual(e.bitrate, "45M")

    def test_default_bitrate_is_none(self):
        from pipeline.vr_metadata import VRMetadataEmbedder

        e = VRMetadataEmbedder()
        self.assertIsNone(e.bitrate)


if __name__ == "__main__":
    unittest.main()
