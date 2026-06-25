"""Unit tests for pipeline.spatial_converter.

Covers:
- Enum and dataclass imports
- Constructor edge cases (missing ffmpeg)
- Supported formats listing
- ISOBMFF metadata box injection (st3d, sv3d) for all 3 spatial modes
- ffmpeg command construction for SBS/MV-HEVC/SBS-mono paths (mocked)
- Error handling (unknown format)

Run with: pytest tests/test_spatial_converter.py -v
"""

import os
import shutil
import struct
import subprocess
import tempfile
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_converter():
    """Create a SpatialConverter with shutil.which patched to find ffmpeg."""
    from pipeline.spatial_converter import SpatialConverter

    with patch.object(shutil, "which", return_value="/usr/bin/ffmpeg"):
        return SpatialConverter()


def _has_ffmpeg() -> bool:
    """Check if ffmpeg is actually available on this system."""
    return shutil.which("ffmpeg") is not None


def _parse_isobmff_boxes(file_path: str) -> list[tuple[str, bytes]]:
    """Parse ISOBMFF top-level boxes from a binary file."""
    with open(file_path, "rb") as f:
        data = f.read()
    boxes = []
    pos = 0
    while pos + 8 <= len(data):
        size = struct.unpack(">I", data[pos : pos + 4])[0]
        if size < 8:
            break
        box_type = data[pos + 4 : pos + 8].decode("ascii", errors="replace")
        boxes.append((box_type, data[pos : pos + size]))
        pos += size
    return boxes


def _make_dummy_file(tmp_dir: str, suffix: str = ".mp4") -> str:
    """Create a dummy empty file and return its path."""
    fd, path = tempfile.mkstemp(suffix=suffix, dir=tmp_dir)
    os.close(fd)
    return path


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------


class TestImports:
    def test_spatial_converter_class(self):
        from pipeline.spatial_converter import SpatialConverter

        assert SpatialConverter is not None

    def test_spatial_format_enum(self):
        from pipeline.spatial_converter import SpatialFormat

        assert SpatialFormat.MV_HEVC.value == "mv-hevc"
        assert SpatialFormat.SBS_SPATIAL.value == "sbs-spatial"
        assert SpatialFormat.SBS_MONO.value == "sbs-mono"

    def test_spatial_projection_enum(self):
        from pipeline.spatial_converter import SpatialProjection

        assert SpatialProjection.EQUIRECTANGULAR.value == "equirectangular"
        assert SpatialProjection.RECTILINEAR.value == "rectilinear"

    def test_spatial_video_info_dataclass(self):
        from pipeline.spatial_converter import SpatialProjection, SpatialVideoInfo

        info = SpatialVideoInfo(
            width=3840,
            height=1920,
            fps=30.0,
            duration=10.0,
            codec="h264",
            format=SpatialProjection.EQUIRECTANGULAR,
            is_stereoscopic=True,
            stereo_mode="sbs",
            has_spatial_metadata=True,
            file_size=1024,
        )
        assert info.width == 3840
        assert info.height == 1920
        assert info.fps == 30.0
        assert info.stereo_mode == "sbs"


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_init_success(self):
        converter = _make_converter()
        assert converter.ffmpeg == "ffmpeg"
        assert converter.ffprobe == "ffprobe"

    def test_init_custom_paths(self):
        from pipeline.spatial_converter import SpatialConverter

        with patch.object(shutil, "which", return_value="/usr/bin/ffmpeg"):
            converter = SpatialConverter(ffmpeg_path="/my/ffmpeg", ffprobe_path="/my/ffprobe")
        assert converter.ffmpeg == "/my/ffmpeg"
        assert converter.ffprobe == "/my/ffprobe"

    def test_ffmpeg_not_found_raises(self):
        from pipeline.spatial_converter import SpatialConverter

        with patch.object(shutil, "which", return_value=None), pytest.raises(RuntimeError, match="ffmpeg not found"):
            SpatialConverter(ffmpeg_path="/nonexistent/ffmpeg")


# ---------------------------------------------------------------------------
# Supported Formats
# ---------------------------------------------------------------------------


class TestSupportedFormats:
    def test_returns_three_formats(self):
        converter = _make_converter()
        formats = converter.get_supported_formats()
        assert len(formats) == 3
        assert "mv-hevc" in formats
        assert "sbs-spatial" in formats
        assert "sbs-mono" in formats

    def test_descriptions(self):
        converter = _make_converter()
        formats = converter.get_supported_formats()
        assert "Apple Vision Pro" in formats["mv-hevc"]
        assert "Meta Quest" in formats["sbs-spatial"]
        assert "Legacy" in formats["sbs-mono"]


# ---------------------------------------------------------------------------
# Metadata Injection (ISOBMFF boxes)
# ---------------------------------------------------------------------------


class TestMetadataInjection:
    """Verify st3d and sv3d box bytes written by each _inject_*_metadata method."""

    @pytest.fixture
    def converter(self):
        return _make_converter()

    def test_mv_hevc_injection_contains_st3d_and_sv3d(self, converter, tmp_path):
        """MV-HEVC metadata should write st3d + sv3d (with svhd + proj) boxes."""
        file_path = os.path.join(tmp_path, "test.mp4")
        with open(file_path, "wb") as f:
            f.write(b"")

        converter._inject_mv_hevc_metadata(file_path, 1920, 1920)
        boxes = _parse_isobmff_boxes(file_path)

        box_types = [b[0] for b in boxes]
        assert "st3d" in box_types, "MV-HEVC should contain st3d box"
        assert "sv3d" in box_types, "MV-HEVC should contain sv3d box"

        # st3d should have stereo_mode = 0 (mono for MV-HEVC)
        st3d_data = next(b[1] for b in boxes if b[0] == "st3d")
        stereo_mode = st3d_data[12]
        assert stereo_mode == 0, "MV-HEVC st3d mode should be 0 (mono)"

        # sv3d should contain svhd + proj
        sv3d_data = next(b[1] for b in boxes if b[0] == "sv3d")
        assert b"svhd" in sv3d_data
        assert b"proj" in sv3d_data

    def test_sbs_spatial_injection_contains_st3d_and_sv3d(self, converter, tmp_path):
        """SBS spatial metadata should write st3d(mode=1) + sv3d boxes."""
        file_path = os.path.join(tmp_path, "test.mp4")
        with open(file_path, "wb") as f:
            f.write(b"")

        converter._inject_sbs_spatial_metadata(file_path, 3840, 1920)
        boxes = _parse_isobmff_boxes(file_path)

        box_types = [b[0] for b in boxes]
        assert "st3d" in box_types
        assert "sv3d" in box_types

        st3d_data = next(b[1] for b in boxes if b[0] == "st3d")
        stereo_mode = st3d_data[12]
        assert stereo_mode == 1, "SBS spatial st3d mode should be 1 (SBS)"

        sv3d_data = next(b[1] for b in boxes if b[0] == "sv3d")
        assert b"svhd" in sv3d_data
        assert b"proj" in sv3d_data

    def test_sbs_mono_injection_contains_st3d_only(self, converter, tmp_path):
        """SBS mono metadata should write only st3d(mode=0) box."""
        file_path = os.path.join(tmp_path, "test.mp4")
        with open(file_path, "wb") as f:
            f.write(b"")

        converter._inject_sbs_mono_metadata(file_path, 3840, 1920)
        boxes = _parse_isobmff_boxes(file_path)

        box_types = [b[0] for b in boxes]
        assert "st3d" in box_types
        assert "sv3d" not in box_types, "SBS mono should NOT contain sv3d"

        st3d_data = next(b[1] for b in boxes if b[0] == "st3d")
        stereo_mode = st3d_data[12]
        assert stereo_mode == 0, "SBS mono st3d mode should be 0"


# ---------------------------------------------------------------------------
# ffmpeg Command Construction (mocked subprocess)
# ---------------------------------------------------------------------------


class TestConvertCommandConstruction:
    """Verify correct ffmpeg cmd construction without running ffmpeg."""

    def test_convert_mv_hevc_command(self, tmp_path):
        """_convert_mv_hevc should construct split+crop+hstack filter for MV-HEVC."""
        converter = _make_converter()
        inp_path = _make_dummy_file(tmp_path)
        out_path = _make_dummy_file(tmp_path)
        tmp_output = _make_dummy_file(tmp_path)

        with (
            patch.object(converter, "_run_ffmpeg") as mock_run,
            patch.object(converter, "_inject_mv_hevc_metadata"),
            patch("tempfile.mktemp", return_value=tmp_output),
        ):
            converter._convert_mv_hevc(inp_path, out_path, 3840, 1920, 30.0, 18)

        assert mock_run.called
        cmd = mock_run.call_args[0][0]

        assert cmd[0] == "ffmpeg"
        assert "-i" in cmd
        input_idx = cmd.index("-i")
        assert cmd[input_idx + 1] == inp_path
        assert "-filter_complex" in cmd
        filter_idx = cmd.index("-filter_complex")
        filter_str = cmd[filter_idx + 1]
        assert "split=2" in filter_str
        assert "crop=1920:1920:0:0" in filter_str
        assert "crop=1920:1920:1920:0" in filter_str
        assert "hstack=inputs=2" in filter_str
        assert "libx265" in cmd
        assert "-tag:v" in cmd
        assert "hvc1" in cmd

    def test_convert_mv_hevc_odd_width(self, tmp_path):
        """MV-HEVC conversion should handle odd total width correctly."""
        converter = _make_converter()
        inp_path = _make_dummy_file(tmp_path)
        out_path = _make_dummy_file(tmp_path)
        tmp_output = _make_dummy_file(tmp_path)

        with (
            patch.object(converter, "_run_ffmpeg") as mock_run,
            patch.object(converter, "_inject_mv_hevc_metadata"),
            patch("tempfile.mktemp", return_value=tmp_output),
        ):
            converter._convert_mv_hevc(inp_path, out_path, 1921, 1920, 30.0, 18)

        cmd = mock_run.call_args[0][0]
        filter_str = cmd[cmd.index("-filter_complex") + 1]
        assert "crop=960:1920:0:0" in filter_str
        assert "crop=960:1920:960:0" in filter_str

    def test_convert_sbs_spatial_command(self, tmp_path):
        """_convert_sbs_spatial should pass-through video with metadata injection."""
        converter = _make_converter()
        inp_path = _make_dummy_file(tmp_path)
        out_path = _make_dummy_file(tmp_path)

        with (
            patch.object(converter, "_run_ffmpeg") as mock_run,
            patch.object(converter, "_inject_sbs_spatial_metadata") as mock_inject,
        ):
            result = converter._convert_sbs_spatial(inp_path, out_path, 3840, 1920, 30.0, 18)

        assert mock_run.called
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "ffmpeg"
        assert cmd[cmd.index("-i") + 1] == inp_path
        assert cmd[-1] == out_path
        assert "libx264" in cmd
        assert mock_inject.called
        assert result["spatial_mode"] == "sbs-spatial"

    def test_convert_sbs_mono_command(self, tmp_path):
        """_convert_sbs_mono should pass-through video with minimal metadata."""
        converter = _make_converter()
        inp_path = _make_dummy_file(tmp_path)
        out_path = _make_dummy_file(tmp_path)

        with (
            patch.object(converter, "_run_ffmpeg") as mock_run,
            patch.object(converter, "_inject_sbs_mono_metadata") as mock_inject,
        ):
            result = converter._convert_sbs_mono(inp_path, out_path, 3840, 1920, 30.0, 18)

        assert mock_run.called
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "ffmpeg"
        assert "libx264" in cmd
        assert mock_inject.called
        assert result["spatial_mode"] == "sbs-mono"

    def test_convert_unknown_format_raises(self, tmp_path):
        """convert() with unsupported format should raise ValueError."""
        from pipeline.spatial_converter import SpatialProjection

        converter = _make_converter()
        inp_path = _make_dummy_file(tmp_path)
        out_path = _make_dummy_file(tmp_path)

        with patch.object(converter, "get_video_info") as mock_info:
            mock_info.return_value = {"width": 3840, "height": 1920, "fps": 30.0, "duration": 5.0}
            with pytest.raises(ValueError, match="Unsupported format"):
                converter.convert(
                    inp_path,
                    out_path,
                    target_format="invalid-format",
                    projection=SpatialProjection.EQUIRECTANGULAR,
                )


# ---------------------------------------------------------------------------
# get_video_info (requires real ffmpeg)
# ---------------------------------------------------------------------------


class TestGetVideoInfo:
    @pytest.mark.skipif(not _has_ffmpeg(), reason="ffmpeg not available on this system")
    def test_get_video_info_real_file(self, tmp_path):
        """get_video_info should parse ffprobe output correctly."""
        from pipeline.spatial_converter import SpatialConverter

        video_path = str(tmp_path / "test_video.mp4")
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
            check=True,
        )

        converter = SpatialConverter()
        info = converter.get_video_info(video_path)

        assert info["width"] == 640
        assert info["height"] == 480
        assert info["fps"] == pytest.approx(24.0, rel=1e-1)
        assert info["duration"] > 0
        assert info["codec"] == "h264"
        assert info["file_size"] > 0


# ---------------------------------------------------------------------------
# Convert End-to-End (mocked ffmpeg)
# ---------------------------------------------------------------------------


class TestConvertPublicAPI:
    def test_convert_sbs_spatial_end_to_end(self, tmp_path):
        """convert() with SBS_SPATIAL should return correct result dict."""
        from pipeline.spatial_converter import SpatialFormat, SpatialProjection

        converter = _make_converter()
        inp_path = _make_dummy_file(tmp_path)
        out_path = _make_dummy_file(tmp_path)

        with (
            patch.object(converter, "get_video_info") as mock_info,
            patch.object(converter, "_run_ffmpeg") as mock_run,
            patch.object(converter, "_inject_sbs_spatial_metadata"),
        ):
            mock_info.return_value = {"width": 3840, "height": 1920, "fps": 30.0, "duration": 10.0}

            result = converter.convert(
                inp_path,
                out_path,
                target_format=SpatialFormat.SBS_SPATIAL,
                projection=SpatialProjection.EQUIRECTANGULAR,
                crf=20,
            )

        assert result["input_path"] == inp_path
        assert result["output_path"] == out_path
        assert result["target_format"] == "sbs-spatial"
        assert result["projection"] == "equirectangular"
        assert result["codec"] == "h264"
        assert result["spatial_mode"] == "sbs-spatial"
        assert mock_run.called

    def test_convert_mv_hevc_end_to_end(self, tmp_path):
        """convert() with MV_HEVC should return correct result dict."""
        from pipeline.spatial_converter import SpatialFormat, SpatialProjection

        converter = _make_converter()
        inp_path = _make_dummy_file(tmp_path)
        out_path = _make_dummy_file(tmp_path)
        tmp_output = _make_dummy_file(tmp_path)

        with (
            patch.object(converter, "get_video_info") as mock_info,
            patch.object(converter, "_run_ffmpeg") as mock_run,
            patch.object(converter, "_inject_mv_hevc_metadata"),
            patch("tempfile.mktemp", return_value=tmp_output),
        ):
            mock_info.return_value = {"width": 3840, "height": 1920, "fps": 30.0, "duration": 10.0}

            result = converter.convert(
                inp_path,
                out_path,
                target_format=SpatialFormat.MV_HEVC,
                projection=SpatialProjection.EQUIRECTANGULAR,
                crf=22,
            )

        assert result["target_format"] == "mv-hevc"
        assert result["projection"] == "equirectangular"
        assert result["codec"] == "hevc"
        assert result["spatial_mode"] == "mv-hevc"
        assert mock_run.called

    def test_convert_with_metadata(self, tmp_path):
        """convert() with optional metadata should embed it in result."""
        from pipeline.spatial_converter import SpatialFormat, SpatialProjection

        converter = _make_converter()
        inp_path = _make_dummy_file(tmp_path)
        out_path = _make_dummy_file(tmp_path)

        with (
            patch.object(converter, "get_video_info") as mock_info,
            patch.object(converter, "_run_ffmpeg"),
            patch.object(converter, "_inject_sbs_mono_metadata"),
        ):
            mock_info.return_value = {"width": 3840, "height": 1920, "fps": 30.0, "duration": 10.0}

            meta = {"title": "VR180 Test", "author": "CI"}
            result = converter.convert(
                inp_path,
                out_path,
                target_format=SpatialFormat.SBS_MONO,
                projection=SpatialProjection.EQUIRECTANGULAR,
                metadata=meta,
            )

        assert result["embedded_metadata"] == meta
