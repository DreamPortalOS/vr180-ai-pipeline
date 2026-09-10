"""Tests for --quality presets and adaptive bitrate scaling (issue #34).

Covers:
  - pipeline.streaming_pipeline.resolve_quality (preset → resolution/streaming)
  - pipeline.streaming_pipeline.scaled_bitrate_mbps (pure bitrate scaling fn)
  - scripts.run_pipeline CLI wiring (--quality in --help, defaults, overrides)
  - D-3 (#346): the ``dome`` 4096²/eye tier and the NVENC_MAX_WIDTH boundary
    it lands on — including *real* ffmpeg encodes at 4096 / 8192 px wide.

Everything except the explicitly ffmpeg-backed tests at the bottom is CPU-only
and mocks ffmpeg/model dependencies.  The ffmpeg-backed tests write only into
``tmp_path`` and skip when ffmpeg/ffprobe are not on PATH.
"""

import json
import os
import shutil
import subprocess
import sys
import unittest
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pipeline.streaming_pipeline import (  # noqa: E402
    BASELINE_BITRATE_MBPS,
    DEFAULT_QUALITY,
    NVENC_MAX_WIDTH,
    QUALITY_PRESETS,
    RawFrameFFmpegWriter,
    resolve_quality,
    scaled_bitrate_mbps,
    select_encoder,
)


class TestQualityPresets(unittest.TestCase):
    """Preset table sanity."""

    def test_preset_values(self):
        self.assertEqual(QUALITY_PRESETS["preview"], 1920)
        self.assertEqual(QUALITY_PRESETS["standard"], 2880)
        self.assertEqual(QUALITY_PRESETS["high"], 3840)

    def test_dome_preset_is_4096(self):
        """D-3 (#346): dome-theatre master tier — 4096²/eye."""
        self.assertEqual(QUALITY_PRESETS["dome"], 4096)

    def test_preset_table_is_exactly_these_tiers(self):
        """Pin the whole table: adding a tier must not perturb the old ones."""
        self.assertEqual(
            QUALITY_PRESETS,
            {"preview": 1920, "standard": 2880, "high": 3840, "dome": 4096},
        )

    def test_preset_order_is_ascending(self):
        """Insertion order is the --help tier order; keep it ascending."""
        sizes = list(QUALITY_PRESETS.values())
        self.assertEqual(sizes, sorted(sizes))

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

    def test_dome_streams(self):
        self.assertEqual(resolve_quality("dome"), (4096, True))

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

    def test_4096_scales_from_the_same_curve(self):
        """D-3 (#346): dome gets no special-case bitrate, just the area curve."""
        self.assertAlmostEqual(scaled_bitrate_mbps(4096), (4096 / 1920) ** 2 * BASELINE_BITRATE_MBPS)
        # ...and stays under the 200 Mbps default cap, so the cap is a no-op here.
        self.assertAlmostEqual(scaled_bitrate_mbps(4096, max_mbps=200.0), scaled_bitrate_mbps(4096))

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
        # Driven off the table, not a literal list — a new tier that the CLI
        # forgot to allow (the D-3/#346 failure mode) fails right here.
        for q in QUALITY_PRESETS:
            args = self._parse(["--input", "x.mp4", "--quality", q])
            self.assertEqual(args.quality, q)

    def test_quality_dome_accepted(self):
        args = self._parse(["--input", "x.mp4", "--quality", "dome"])
        self.assertEqual(args.quality, "dome")

    def test_unknown_quality_rejected(self):
        with self.assertRaises(SystemExit):
            self._parse(["--input", "x.mp4", "--quality", "ultra"])

    def _help_text(self):
        env = dict(os.environ, PYTHONPATH=PROJECT_ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""))
        result = subprocess.run(
            [sys.executable, os.path.join(PROJECT_ROOT, "scripts", "run_pipeline.py"), "--help"],
            capture_output=True,
            text=True,
            env=env,
        )
        self.assertEqual(result.returncode, 0)
        return result.stdout

    def test_help_lists_quality(self):
        stdout = self._help_text()
        self.assertIn("--quality", stdout)
        self.assertIn("preview", stdout)
        self.assertIn("standard", stdout)
        self.assertIn("high", stdout)

    def test_help_lists_every_preset(self):
        """--help must name every tier in the table, dome included (#346)."""
        stdout = self._help_text()
        for q in QUALITY_PRESETS:
            self.assertIn(q, stdout, f"--help does not mention the {q!r} tier")


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

    def test_dome_enables_streaming_4096(self):
        """D-3 (#346): --quality dome → a 4096² master per eye."""
        args = self._apply(quality="dome")
        self.assertEqual(args.output_width, 4096)
        self.assertEqual(args.output_height, 4096)
        self.assertTrue(args.streaming)
        self.assertEqual(args.bitrate, f"{scaled_bitrate_mbps(4096, max_mbps=200.0):g}M")

    def test_dome_bitrate_above_high(self):
        """Sanity: more pixels than `high` must not mean fewer bits."""
        dome = self._apply(quality="dome")
        high = self._apply(quality="high")
        self.assertGreater(float(dome.bitrate.rstrip("M")), float(high.bitrate.rstrip("M")))

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


# ---------------------------------------------------------------------------
# D-3 (#346): the dome tier lands exactly on the H.264 NVENC width cap.
#
# NVENC_MAX_WIDTH is 4096 and the comparison in select_encoder is ``>``, so the
# cap is *inclusive*: a 4096-wide frame is still legal H.264.  That gives the
# dome tier two distinct encode paths and both need pinning:
#
#   * fulldome master  — 4096 wide (one domemaster)   → H.264 branch
#   * VR180 SBS frame  — 2×4096 = 8192 wide           → HEVC branch
#
# Getting this backwards is silent: you either lose the H.264 fast path for
# fulldome, or you hand NVENC an 8192-wide frame it will refuse to open.
# ---------------------------------------------------------------------------

DOME_EYE_SIZE = 4096
DOME_SBS_WIDTH = 2 * DOME_EYE_SIZE


class TestDomeEncoderSelection(unittest.TestCase):
    """select_encoder at and around the NVENC_MAX_WIDTH boundary."""

    def test_cap_is_4096(self):
        self.assertEqual(NVENC_MAX_WIDTH, 4096)
        self.assertEqual(QUALITY_PRESETS["dome"], NVENC_MAX_WIDTH)

    def test_fulldome_width_stays_on_h264(self):
        """4096 == the cap, and the cap is inclusive → H.264 is still valid."""
        self.assertEqual(select_encoder("h264", DOME_EYE_SIZE, hw=True), ["-c:v", "h264_nvenc"])
        self.assertEqual(select_encoder("h264", DOME_EYE_SIZE, hw=False), ["-c:v", "libx264"])

    def test_one_pixel_over_the_cap_switches_to_hevc(self):
        self.assertEqual(select_encoder("h264", NVENC_MAX_WIDTH + 1, hw=True), ["-c:v", "hevc_nvenc"])
        self.assertEqual(
            select_encoder("h264", NVENC_MAX_WIDTH + 1, hw=False),
            ["-c:v", "libx265", "-preset", "fast"],
        )

    def test_dome_sbs_width_goes_hevc(self):
        """The tier's actual VR180 frame is 8192 wide — over the cap."""
        self.assertGreater(DOME_SBS_WIDTH, NVENC_MAX_WIDTH)
        self.assertEqual(select_encoder("h264", DOME_SBS_WIDTH, hw=True), ["-c:v", "hevc_nvenc"])
        self.assertEqual(
            select_encoder("h264", DOME_SBS_WIDTH, hw=False),
            ["-c:v", "libx265", "-preset", "fast"],
        )

    def test_explicit_h265_unaffected_by_width(self):
        self.assertEqual(
            select_encoder("h265", DOME_EYE_SIZE, hw=False),
            ["-c:v", "libx265", "-preset", "fast"],
        )
        self.assertEqual(select_encoder("h265", DOME_EYE_SIZE, hw=True), ["-c:v", "hevc_nvenc"])


class TestExistingTiersUnchanged(unittest.TestCase):
    """The three pre-#346 tiers must produce byte-identical ffmpeg invocations.

    The ffmpeg argv *is* the encode contract — same encoder, same rate control,
    same flags, same frame geometry means the same bytes out.  These lists are
    frozen from ``origin/main`` before the dome tier existed.
    """

    FROZEN_TAIL: ClassVar[dict] = {
        "preview": (["-c:v", "libx264"], "3840x1920", 20.0),
        "standard": (["-c:v", "libx265", "-preset", "fast"], "5760x2880", 45.0),
        "high": (["-c:v", "libx265", "-preset", "fast"], "7680x3840", 80.0),
    }

    def _cmd_for(self, eye_size):
        from unittest.mock import patch

        with (
            patch("pipeline.streaming_pipeline.DepthEstimator"),
            patch("pipeline.streaming_pipeline.StereoRenderer"),
            patch("pipeline.streaming_pipeline.EquirectangularMapper"),
        ):
            from pipeline.streaming_pipeline import StreamingPipeline

            p = StreamingPipeline(codec="h264", crf=23, fps=30, hw_encoder=False)
            return p._build_ffmpeg_cmd("out.mp4", eye_size * 2, eye_size)

    def test_frozen_ffmpeg_argv_per_tier(self):
        for tier, (encoder, geometry, bitrate) in self.FROZEN_TAIL.items():
            with self.subTest(tier=tier):
                eye_size = QUALITY_PRESETS[tier]
                head = [
                    "ffmpeg",
                    "-y",
                    "-f",
                    "rawvideo",
                    "-vcodec",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-s",
                    geometry,
                    "-r",
                    "30",
                    "-i",
                    "pipe:0",
                ]
                tail = ["-crf", "23", "-movflags", "+faststart", "-pix_fmt", "yuv420p", "out.mp4"]
                expected = [*head, *encoder, *tail]
                self.assertEqual(self._cmd_for(eye_size), expected)
                self.assertAlmostEqual(scaled_bitrate_mbps(eye_size, max_mbps=200.0), bitrate)

    def test_default_tier_still_standard_2880(self):
        """Adding a bigger tier must not move the default off standard."""
        self.assertEqual(DEFAULT_QUALITY, "standard")
        self.assertEqual(resolve_quality(DEFAULT_QUALITY), (2880, True))


# ---------------------------------------------------------------------------
# Real-ffmpeg encode/decode proof for the dome widths.
#
# The pure-function tests above pin which encoder gets *chosen*; these pin that
# the chosen encoder actually accepts the frame and emits a decodable file.
# The default cases use a short frame height — the branch is width-driven, so a
# 4096×64 / 8192×64 frame exercises exactly the same decision at a fraction of
# the cost.  The `slow` variants repeat it at true dome geometry.
# ---------------------------------------------------------------------------

_HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
requires_ffmpeg = pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg/ffprobe not on PATH")


def _probe_stream(path):
    """Return the first video stream's ffprobe dict."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,pix_fmt",
        "-of",
        "json",
        str(path),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(out.stdout)["streams"][0]


def _assert_decodes(path):
    """Full decode pass — a file ffprobe can read but ffmpeg cannot decode is a bad master."""
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"decode failed: {result.stderr}"
    assert result.stderr.strip() == "", f"decoder reported errors: {result.stderr}"


def _encode_through_writer(path, width, height, *, frames=2, codec="h264"):
    """Push `frames` synthetic RGB frames through the production writer."""
    with RawFrameFFmpegWriter(str(path), width, height, codec=codec, fps=30, hw_encoder=False) as writer:
        for i in range(frames):
            frame = np.full((height, width, 3), (i + 1) * 40, dtype=np.uint8)
            # A non-flat frame so the encoder has something real to chew on.
            frame[:, : width // 2, 0] = 200
            writer.write(frame)


@requires_ffmpeg
@pytest.mark.parametrize(
    "width,expected_codec",
    [
        (DOME_EYE_SIZE, "h264"),  # fulldome master — exactly on the inclusive cap
        (DOME_SBS_WIDTH, "hevc"),  # VR180 SBS at the dome tier — over the cap
    ],
)
def test_dome_widths_encode_and_decode(tmp_path, width, expected_codec):
    """#346: the encoder select_encoder picks really does handle the width."""
    out = tmp_path / f"dome_{width}.mp4"
    _encode_through_writer(out, width, 64)

    assert out.exists() and out.stat().st_size > 0
    stream = _probe_stream(out)
    assert stream["codec_name"] == expected_codec
    assert stream["width"] == width
    assert stream["height"] == 64
    assert stream["pix_fmt"] == "yuv420p"
    _assert_decodes(out)


@requires_ffmpeg
@pytest.mark.slow
@pytest.mark.parametrize(
    "width,height,expected_codec",
    [
        (DOME_EYE_SIZE, DOME_EYE_SIZE, "h264"),  # 4096² fulldome domemaster
        (DOME_SBS_WIDTH, DOME_EYE_SIZE, "hevc"),  # 8192×4096 VR180 SBS
    ],
)
def test_dome_full_size_encode_and_decode(tmp_path, width, height, expected_codec):
    """Same proof at true dome geometry (slow: ~100 MB/frame through the pipe)."""
    out = tmp_path / f"dome_full_{width}x{height}.mp4"
    _encode_through_writer(out, width, height)

    stream = _probe_stream(out)
    assert stream["codec_name"] == expected_codec
    assert stream["width"] == width
    assert stream["height"] == height
    _assert_decodes(out)


if __name__ == "__main__":
    unittest.main()
