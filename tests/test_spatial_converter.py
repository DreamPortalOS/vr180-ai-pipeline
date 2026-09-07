"""Unit tests for pipeline.spatial_converter.

Covers:
- Enum and dataclass imports
- Constructor edge cases (missing ffmpeg)
- Supported formats listing
- st3d/sv3d injection for all 3 spatial modes goes through the shared
  pipeline.spherical_injector (#281): boxes inside moov, full proj tree with
  the RFC VR180 equi bounds, file still decodes
- ffmpeg command construction for SBS/MV-HEVC/SBS-mono paths (mocked)
- Error handling (unknown format)

Run with: pytest tests/test_spatial_converter.py -v
"""

import inspect
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from pipeline.spherical_injector import (
    _EQUI_BOUNDS_VR180,
    _STEREO_LEFT_RIGHT,
    _STEREO_MONO,
    _find_box_recursive,
    _walk_boxes,
    read_projection_bounds,
)
from tests.test_spherical_injector import _encode_testsrc, _ffmpeg_decode, _sv3d_tree

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


# (converter method, st3d stereo_mode byte it must map to)
_INJECT_CASES = [
    pytest.param("_inject_mv_hevc_metadata", _STEREO_MONO, id="mv-hevc->mono"),
    pytest.param("_inject_sbs_spatial_metadata", _STEREO_LEFT_RIGHT, id="sbs-spatial->sbs"),
    pytest.param("_inject_sbs_mono_metadata", _STEREO_MONO, id="sbs-mono->mono"),
]
# st3d byte -> the pipeline.spherical_injector mode name the converter must pass
_MODE_NAME = {_STEREO_MONO: "mono", _STEREO_LEFT_RIGHT: "sbs"}


class TestMetadataInjection:
    """#281: the converter no longer carries its own box writer.

    Every ``_inject_*_metadata`` goes through ``pipeline.spherical_injector``,
    so the boxes land *inside moov* (in the visual sample entry), carry the full
    ``svhd / proj { prhd, equi }`` tree with the RFC VR180 bounds, and the file
    still decodes.  The pre-#281 writer appended st3d/sv3d after the last
    top-level box (invisible to any parser), emitted a proj without prhd/equi,
    and tagged SBS as stereo_mode 1 — top-bottom per the RFC, not left-right.
    """

    @pytest.fixture
    def converter(self):
        return _make_converter()

    @pytest.fixture(scope="class")
    def clip(self, tmp_path_factory) -> Path:
        """One tiny ``-f lavfi testsrc`` H.264 clip per class; tests copy it."""
        path = tmp_path_factory.mktemp("i281_converter") / "src.mp4"
        _encode_testsrc(path, "h264", True)
        return path

    @pytest.mark.skipif(not _has_ffmpeg(), reason="ffmpeg not available on this system")
    @pytest.mark.parametrize(("method", "stereo_byte"), _INJECT_CASES)
    def test_boxes_live_inside_moov_with_full_proj_tree(self, converter, clip, tmp_path, method, stereo_byte):
        file_path = str(tmp_path / "out.mp4")
        shutil.copy2(clip, file_path)

        getattr(converter, method)(file_path, 256, 128)

        data = bytearray(Path(file_path).read_bytes())
        top_level = list(_walk_boxes(data, 0, len(data)))
        top_types = [t for _o, t, _s in top_level]
        assert b"st3d" not in top_types and b"sv3d" not in top_types, top_types  # nothing appended to the file
        moov_off, moov_sz = next((o, s) for o, t, s in top_level if t == b"moov")
        for box in (b"st3d", b"sv3d"):
            off = _find_box_recursive(data, box, 0, len(data))
            assert moov_off < off < moov_off + moov_sz, box
        assert _sv3d_tree(bytes(data)) == ([b"svhd", b"proj"], [b"prhd", b"equi"])
        st3d_off = _find_box_recursive(data, b"st3d", 0, len(data))
        assert data[st3d_off + 12] == stereo_byte
        assert read_projection_bounds(file_path) == _EQUI_BOUNDS_VR180 == (0, 0, 0x40000000, 0x40000000)
        rc, err = _ffmpeg_decode(Path(file_path))
        assert rc == 0, err
        assert not list(tmp_path.glob("*.vr.mp4"))  # temp file cleaned up

    @pytest.mark.parametrize(("method", "stereo_byte"), _INJECT_CASES)
    def test_mode_mapping_and_in_place_replace(self, converter, tmp_path, method, stereo_byte):
        """No ffmpeg needed: the shared injector receives the mapped mode and
        its output replaces the file in place, leaving no temp file behind."""
        file_path = tmp_path / "out.mp4"
        file_path.write_bytes(b"original")
        seen: list[dict] = []

        def fake_inject(input_path, output_path, **kwargs):
            seen.append({"input": input_path, "output": output_path, **kwargs})
            Path(output_path).write_bytes(b"injected")
            return output_path

        with patch("pipeline.spatial_converter.inject_spherical_metadata", fake_inject):
            getattr(converter, method)(str(file_path), 3840, 1920)

        assert len(seen) == 1
        assert seen[0]["input"] == str(file_path)
        assert seen[0]["output"] != str(file_path)
        assert seen[0]["stereo_mode"] == _MODE_NAME[stereo_byte]
        assert (seen[0]["width"], seen[0]["height"]) == (3840, 1920)
        assert file_path.read_bytes() == b"injected"
        assert not list(tmp_path.glob("*.vr.mp4"))

    def test_injector_failure_propagates_and_cleans_up(self, converter, tmp_path):
        file_path = tmp_path / "out.mp4"
        file_path.write_bytes(b"original")

        def failing_inject(input_path, output_path, **kwargs):
            Path(output_path).write_bytes(b"half-written")
            raise RuntimeError("VR metadata injection FAILED self-check")

        with (
            patch("pipeline.spatial_converter.inject_spherical_metadata", failing_inject),
            pytest.raises(RuntimeError, match="self-check"),
        ):
            converter._inject_sbs_spatial_metadata(str(file_path), 3840, 1920)

        assert file_path.read_bytes() == b"original"
        assert not list(tmp_path.glob("*.vr.mp4"))

    def test_no_private_box_writer_remains(self):
        """Acceptance grep for #281: spatial_converter must not build sv3d/st3d bytes itself."""
        import pipeline.spatial_converter as sc
        import pipeline.spherical_injector as si

        source = inspect.getsource(sc)
        for literal in ('b"sv3d"', "b'sv3d'", 'b"st3d"', "b'st3d'", "struct.pack"):
            assert literal not in source, literal
        assert sc.inject_spherical_metadata is si.inject_spherical_metadata


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
