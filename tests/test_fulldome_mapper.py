"""Tests for pipeline.fulldome_mapper — FulldomeMapper class.

Unit tests mock ffmpeg/ffprobe. An integration test (marked slow) runs
with a real ffmpeg-generated test clip to verify the output domemaster.
"""

from __future__ import annotations

import json
import subprocess
import unittest.mock
from pathlib import Path

import pytest

from pipeline.fulldome_mapper import FulldomeMapper

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def dummy_input(tmp_path: Path) -> str:
    """Create a minimal valid-looking input path (file does NOT exist)."""
    return str(tmp_path / "nonexistent.mp4")


@pytest.fixture
def real_input(tmp_path: Path) -> str:
    """Create a real tiny test video with ffmpeg for integration tests."""
    mp4 = tmp_path / "test_src.mp4"
    # Generate a 1-second 320×240 test video (color bars, no audio)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=1:size=320x240:rate=30",
            "-frames:v",
            "5",
            "-pix_fmt",
            "yuv420p",
            str(mp4),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return str(mp4)


# ---------------------------------------------------------------------------
# Unit tests — construction & defaults
# ---------------------------------------------------------------------------


class TestFulldomeMapperConstruction:
    def test_default_params(self):
        mapper = FulldomeMapper()
        assert mapper.dome_fov == 180.0
        assert mapper.coverage_h_fov == 120.0
        assert mapper.coverage_v_fov is None
        assert mapper.output_size == 4096
        assert mapper.codec == "h264"
        assert mapper.crf == 18

    def test_custom_params(self):
        mapper = FulldomeMapper(
            dome_fov=200.0,
            coverage_h_fov=100.0,
            coverage_v_fov=60.0,
            output_size=2048,
            codec="h265",
            crf=22,
        )
        assert mapper.dome_fov == 200.0
        assert mapper.coverage_h_fov == 100.0
        assert mapper.coverage_v_fov == 60.0
        assert mapper.output_size == 2048
        assert mapper.codec == "h265"
        assert mapper.crf == 22

    def test_odd_output_size_is_rounded_up(self):
        """ffmpeg requires even dimensions; odd should be bumped by +1."""
        mapper = FulldomeMapper(output_size=2047)
        assert mapper.output_size == 2048

    def test_even_output_size_stays_unchanged(self):
        mapper = FulldomeMapper(output_size=4096)
        assert mapper.output_size == 4096


# ---------------------------------------------------------------------------
# Unit tests — input validation
# ---------------------------------------------------------------------------


class TestFulldomeMapperInputValidation:
    def test_nonexistent_input_raises(self):
        mapper = FulldomeMapper()
        with pytest.raises(FileNotFoundError, match="not found"):
            mapper.convert("C:\\nonexistent\\file.mp4", "out.mp4")


# ---------------------------------------------------------------------------
# Unit tests — _probe_coverage_v_fov (mocked ffprobe)
# ---------------------------------------------------------------------------


class TestProbeCoverageVFov:
    def test_16_9_source(self):
        mapper = FulldomeMapper(coverage_h_fov=120.0)
        with unittest.mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = unittest.mock.Mock(
                returncode=0,
                stdout=json.dumps({"streams": [{"width": 1920, "height": 1080}]}),
            )
            result = mapper._probe_coverage_v_fov("dummy.mp4")
            # aspect = 1080/1920 = 0.5625 → 120 * 0.5625 = 67.5
            assert abs(result - 67.5) < 0.01

    def test_4_3_source(self):
        mapper = FulldomeMapper(coverage_h_fov=120.0)
        with unittest.mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = unittest.mock.Mock(
                returncode=0,
                stdout=json.dumps({"streams": [{"width": 640, "height": 480}]}),
            )
            result = mapper._probe_coverage_v_fov("dummy.mp4")
            # aspect = 480/640 = 0.75 → 120 * 0.75 = 90.0
            assert abs(result - 90.0) < 0.01

    def test_ffprobe_failure_falls_back_to_90(self):
        mapper = FulldomeMapper(coverage_h_fov=120.0)
        with unittest.mock.patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("ffprobe not found")
            result = mapper._probe_coverage_v_fov("dummy.mp4")
            assert result == 90.0

    def test_malformed_ffprobe_output_falls_back(self):
        mapper = FulldomeMapper(coverage_h_fov=120.0)
        with unittest.mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = unittest.mock.Mock(
                returncode=0,
                stdout="not valid json",
            )
            result = mapper._probe_coverage_v_fov("dummy.mp4")
            assert result == 90.0

    def test_empty_streams_falls_back(self):
        mapper = FulldomeMapper(coverage_h_fov=120.0)
        with unittest.mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = unittest.mock.Mock(
                returncode=0,
                stdout=json.dumps({"streams": []}),
            )
            result = mapper._probe_coverage_v_fov("dummy.mp4")
            assert result == 90.0


# ---------------------------------------------------------------------------
# Unit tests — convert() with mocked subprocess
# ---------------------------------------------------------------------------


class TestConvertMocked:
    def test_convert_success(self, tmp_path: Path):
        """Verify the ffmpeg command is constructed correctly."""
        src = tmp_path / "input.mp4"
        src.write_text("fake video content")
        out = str(tmp_path / "output_dome.mp4")

        mapper = FulldomeMapper(
            coverage_h_fov=120.0,
            coverage_v_fov=75.0,  # explicit, no probing needed
            output_size=2048,
            codec="h264",
            crf=18,
        )

        with unittest.mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = unittest.mock.Mock(returncode=0, stdout="", stderr="")

            result = mapper.convert(str(src), out)

            assert result == out

            # Verify the subprocess command
            call_args = mock_run.call_args[0][0]
            assert "ffmpeg" in call_args[0]
            assert "-i" in call_args
            assert str(src) in call_args
            assert (
                "v360=input=flat:output=fisheye:ih_fov=120.0:iv_fov=75.0:h_fov=180.0:v_fov=180.0:w=2048:h=2048"
                in " ".join(call_args)
            )
            assert "-an" in call_args  # no audio
            assert out in call_args

    def test_convert_success_with_probe(self, tmp_path: Path):
        """When coverage_v_fov is None, the mapper should probe first."""
        src = tmp_path / "input.mp4"
        src.write_text("fake video content")
        out = str(tmp_path / "output_dome.mp4")

        mapper = FulldomeMapper(
            coverage_h_fov=120.0,
            coverage_v_fov=None,  # will probe
            output_size=2048,
        )

        with unittest.mock.patch("subprocess.run") as mock_run:
            # First call is ffprobe → return 16:9 source
            probe_result = unittest.mock.Mock(
                returncode=0,
                stdout=json.dumps({"streams": [{"width": 1920, "height": 1080}]}),
                stderr="",
            )
            # Second call is ffmpeg conversion
            ffmpeg_result = unittest.mock.Mock(returncode=0, stdout="", stderr="")
            mock_run.side_effect = [probe_result, ffmpeg_result]

            result = mapper.convert(str(src), out)
            assert result == out

            # Check that the ffmpeg command includes auto-computed iv_fov=67.5
            ffmpeg_call = mock_run.call_args_list[1].args[0]
            ffmpeg_str = " ".join(ffmpeg_call)
            assert "iv_fov=67.5" in ffmpeg_str

    def test_convert_ffmpeg_failure(self, tmp_path: Path):
        src = tmp_path / "input.mp4"
        src.write_text("fake video content")
        out = str(tmp_path / "output.mp4")

        mapper = FulldomeMapper(coverage_v_fov=75.0)

        with unittest.mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = unittest.mock.Mock(returncode=1, stdout="", stderr="ffmpeg error occurred")

            with pytest.raises(RuntimeError, match="ffmpeg v360 conversion failed"):
                mapper.convert(str(src), out)

    def test_convert_h265_codec(self, tmp_path: Path):
        """Verify libx265 is used when codec='h265'."""
        src = tmp_path / "input.mp4"
        src.write_text("fake")
        out = str(tmp_path / "out.mp4")

        mapper = FulldomeMapper(coverage_v_fov=75.0, codec="h265")

        with unittest.mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = unittest.mock.Mock(returncode=0, stdout="", stderr="")
            mapper.convert(str(src), out)

            call_args = mock_run.call_args[0][0]
            assert "libx265" in call_args

    def test_subprocess_uses_list_no_shell(self, tmp_path: Path):
        """Critical: subprocess.run must receive a list, never shell=True."""
        src = tmp_path / "input.mp4"
        src.write_text("fake")
        out = str(tmp_path / "out.mp4")

        mapper = FulldomeMapper(coverage_v_fov=75.0)

        with unittest.mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = unittest.mock.Mock(returncode=0, stdout="", stderr="")
            mapper.convert(str(src), out)

            # subprocess.run should have been called with a list as first positional arg
            call_kwargs = mock_run.call_args.kwargs
            # Ensure shell is NOT True
            assert call_kwargs.get("shell") is not True, "shell=True is forbidden for security"
            # First positional arg must be a list
            call_args = mock_run.call_args[0][0]
            assert isinstance(call_args, list)


# ---------------------------------------------------------------------------
# Integration test — actual ffmpeg (marked slow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestFulldomeMapperIntegration:
    """These tests use real ffmpeg to generate input and verify output.

    Run with:  pytest tests/test_fulldome_mapper.py -v -m slow
    """

    def test_output_is_square(self, real_input: str, tmp_path: Path):
        """The domemaster output must be square (width == height)."""
        out = str(tmp_path / "dome.mp4")
        mapper = FulldomeMapper(
            coverage_h_fov=120.0,
            coverage_v_fov=75.0,
            output_size=512,
            crf=28,
        )
        mapper.convert(real_input, out)
        assert Path(out).exists()

        # Probe output dimensions with ffprobe
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "json",
                out,
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
        info = json.loads(probe.stdout)
        w = info["streams"][0]["width"]
        h = info["streams"][0]["height"]
        assert w == h, f"Domemaster must be square, got {w}×{h}"

    def test_output_has_no_stereo_boxes(self, real_input: str, tmp_path: Path):
        """Fulldome output must NOT contain sv3d or st3d metadata boxes."""
        out = str(tmp_path / "dome.mp4")
        mapper = FulldomeMapper(
            coverage_h_fov=120.0,
            coverage_v_fov=75.0,
            output_size=512,
            crf=28,
        )
        mapper.convert(real_input, out)

        # Check for spherical video metadata boxes with ffprobe
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "stream=codec_name,width,height",
                "-of",
                "json",
                out,
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
        info = json.loads(probe.stdout)
        # Just having a valid video stream with no spherical metadata is sufficient
        assert len(info.get("streams", [])) >= 1

        # Also verify no sv3d/st3d in binary by grepping ffprobe side data output
        side_data = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream_side_data_list",
                "-of",
                "json",
                out,
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
        # If there are no side data entries, the output is clean
        side_json = json.loads(side_data.stdout) if side_data.stdout.strip() else {}
        streams = side_json.get("streams", [])
        if streams and "side_data_list" in streams[0]:
            for sd in streams[0]["side_data_list"]:
                assert "sv3d" not in json.dumps(sd).lower()
                assert "st3d" not in json.dumps(sd).lower()
