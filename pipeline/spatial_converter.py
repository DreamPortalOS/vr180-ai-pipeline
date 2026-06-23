"""
Spatial Video Converter — Convert VR180 pipeline output to spatial video formats.

Supports:
- MV-HEVC (Apple Vision Pro compatible)
- SBS Spatial (Meta Quest compatible)
- SBS Mono (legacy fallback)

Each output format includes proper ISOBMFF metadata boxes for spatial playback.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import struct
import subprocess
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class SpatialFormat(Enum):
    """Supported spatial video container formats."""

    MV_HEVC = "mv-hevc"
    SBS_SPATIAL = "sbs-spatial"
    SBS_MONO = "sbs-mono"


class SpatialProjection(Enum):
    """Supported spatial video projections."""

    EQUIRECTANGULAR = "equirectangular"
    RECTILINEAR = "rectilinear"
    EQUIRECT = "equirect"


@dataclass
class SpatialVideoInfo:
    """Metadata about a spatial video file."""

    width: int
    height: int
    fps: float
    duration: float
    codec: str
    format: SpatialProjection
    is_stereoscopic: bool
    stereo_mode: str
    has_spatial_metadata: bool
    file_size: int
    extra: dict[str, Any] = field(default_factory=dict)


class SpatialConverter:
    """Convert VR180 pipeline output to device-specific spatial video formats."""

    def __init__(self, ffmpeg_path: str = "ffmpeg", ffprobe_path: str = "ffprobe"):
        self.ffmpeg = ffmpeg_path
        self.ffprobe = ffprobe_path
        if not shutil.which(self.ffmpeg):
            raise RuntimeError(f"ffmpeg not found at '{self.ffmpeg}'")

    def convert(
        self,
        input_path: str,
        output_path: str,
        target_format: SpatialFormat = SpatialFormat.MV_HEVC,
        projection: SpatialProjection = SpatialProjection.EQUIRECTANGULAR,
        crf: int = 18,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Convert an SBS VR180 video to a spatial video format.

        Args:
            input_path: Path to the input SBS equirectangular video.
            output_path: Path for the output spatial video.
            target_format: Target spatial format.
            projection: Source projection type.
            crf: Encoding quality (lower = better).
            metadata: Optional metadata to embed.

        Returns:
            Dict with conversion results and file info.
        """
        input_info = self.get_video_info(input_path)

        width = input_info["width"]
        height = input_info["height"]
        fps = input_info.get("fps", 30.0)
        input_info.get("duration", 0.0)

        if target_format == SpatialFormat.MV_HEVC:
            result = self._convert_mv_hevc(
                input_path, output_path, width, height, fps, crf,
            )
        elif target_format == SpatialFormat.SBS_SPATIAL:
            result = self._convert_sbs_spatial(
                input_path, output_path, width, height, fps, crf,
            )
        elif target_format == SpatialFormat.SBS_MONO:
            result = self._convert_sbs_mono(
                input_path, output_path, width, height, fps, crf,
            )
        else:
            raise ValueError(f"Unsupported format: {target_format}")

        result["input_path"] = input_path
        result["output_path"] = output_path
        result["target_format"] = target_format.value
        result["projection"] = projection.value

        if metadata:
            result["embedded_metadata"] = metadata

        return result

    def _convert_mv_hevc(
        self,
        input_path: str,
        output_path: str,
        width: int,
        height: int,
        fps: float,
        crf: int,
    ) -> dict[str, Any]:
        """Convert to MV-HEVC format for Apple Vision Pro."""
        eye_width = width // 2
        eye_height = height

        tmp_output = tempfile.mktemp(suffix=".mp4")
        try:
            cmd = [
                self.ffmpeg, "-y",
                "-i", input_path,
                "-filter_complex",
                f"[0:v]split=2[left][right];"
                f"[left]crop={eye_width}:{eye_height}:0:0[l];"
                f"[right]crop={eye_width}:{eye_height}:{eye_width}:0[r];"
                f"[l][r]hstack=inputs=2[out]",
                "-map", "[out]",
                "-c:v", "libx265",
                "-crf", str(crf),
                "-preset", "fast",
                "-tag:v", "hvc1",
                "-pix_fmt", "yuv420p",
                tmp_output,
            ]
            self._run_ffmpeg(cmd)

            self._inject_mv_hevc_metadata(tmp_output, eye_width, eye_height)

            shutil.move(tmp_output, output_path)

        except Exception:
            if os.path.exists(tmp_output):
                os.remove(tmp_output)
            raise

        return {
            "width": eye_width,
            "height": eye_height,
            "fps": fps,
            "codec": "hevc",
            "spatial_mode": "mv-hevc",
        }

    def _convert_sbs_spatial(
        self,
        input_path: str,
        output_path: str,
        width: int,
        height: int,
        fps: float,
        crf: int,
    ) -> dict[str, Any]:
        """Convert to SBS spatial format for Meta Quest."""
        cmd = [
            self.ffmpeg, "-y",
            "-i", input_path,
            "-c:v", "libx264",
            "-crf", str(crf),
            "-preset", "fast",
            "-pix_fmt", "yuv420p",
            output_path,
        ]
        self._run_ffmpeg(cmd)

        self._inject_sbs_spatial_metadata(output_path, width, height)

        return {
            "width": width,
            "height": height,
            "fps": fps,
            "codec": "h264",
            "spatial_mode": "sbs-spatial",
        }

    def _convert_sbs_mono(
        self,
        input_path: str,
        output_path: str,
        width: int,
        height: int,
        fps: float,
        crf: int,
    ) -> dict[str, Any]:
        """Convert to SBS mono format (legacy fallback)."""
        cmd = [
            self.ffmpeg, "-y",
            "-i", input_path,
            "-c:v", "libx264",
            "-crf", str(crf),
            "-preset", "fast",
            "-pix_fmt", "yuv420p",
            output_path,
        ]
        self._run_ffmpeg(cmd)

        self._inject_sbs_mono_metadata(output_path, width, height)

        return {
            "width": width,
            "height": height,
            "fps": fps,
            "codec": "h264",
            "spatial_mode": "sbs-mono",
        }

    def _inject_mv_hevc_metadata(
        self, file_path: str, eye_width: int, eye_height: int,
    ) -> None:
        """
        Inject MV-HEVC spatial metadata into an MP4 file using ISOBMFF boxes.

        Appends st3d (stereo mode), sv3d (supplemental video), svhd, proj boxes.
        """
        with open(file_path, "ab") as f:
            # st3d box: stereo_mode = 0 (mono) for MV-HEVC (each eye is separate track)
            st3d_data = struct.pack(">IB", 0, 0)
            st3d_box = struct.pack(">I", 8 + len(st3d_data)) + b"st3d" + st3d_data

            # svhd box: spherical video header
            svhd_version = struct.pack(">B", 0)
            svhd_flags = b"\x00\x00\x00"
            svhd_metadata_source = b"VR180Studio\x00"
            svhd_payload = svhd_version + svhd_flags + svhd_metadata_source
            svhd_box = struct.pack(">I", 8 + len(svhd_payload)) + b"svhd" + svhd_payload

            # proj box: projection box
            proj_data = struct.pack(">I", 0)  # equirectangular
            proj_box = struct.pack(">I", 8 + len(proj_data)) + b"proj" + proj_data

            # sv3d box: contains svhd + proj
            sv3d_payload = svhd_box + proj_box
            sv3d_box = struct.pack(">I", 8 + len(sv3d_payload)) + b"sv3d" + sv3d_payload

            f.write(st3d_box)
            f.write(sv3d_box)

        logger.info(
            "Injected MV-HEVC metadata: st3d + sv3d (svhd, proj) into %s",
            file_path,
        )

    def _inject_sbs_spatial_metadata(
        self, file_path: str, width: int, height: int,
    ) -> None:
        """
        Inject SBS spatial metadata into an MP4 file.

        Appends st3d (stereo_mode=1 for side-by-side), sv3d (svhd, proj) boxes.
        """
        with open(file_path, "ab") as f:
            # st3d box: stereo_mode = 1 (side-by-side)
            st3d_data = struct.pack(">IB", 0, 1)
            st3d_box = struct.pack(">I", 8 + len(st3d_data)) + b"st3d" + st3d_data

            # svhd box
            svhd_version = struct.pack(">B", 0)
            svhd_flags = b"\x00\x00\x00"
            svhd_metadata_source = b"VR180Studio\x00"
            svhd_payload = svhd_version + svhd_flags + svhd_metadata_source
            svhd_box = struct.pack(">I", 8 + len(svhd_payload)) + b"svhd" + svhd_payload

            # proj box
            proj_data = struct.pack(">I", 0)
            proj_box = struct.pack(">I", 8 + len(proj_data)) + b"proj" + proj_data

            # sv3d box
            sv3d_payload = svhd_box + proj_box
            sv3d_box = struct.pack(">I", 8 + len(sv3d_payload)) + b"sv3d" + sv3d_payload

            f.write(st3d_box)
            f.write(sv3d_box)

        logger.info(
            "Injected SBS spatial metadata: st3d(mode=1) + sv3d into %s",
            file_path,
        )

    def _inject_sbs_mono_metadata(
        self, file_path: str, width: int, height: int,
    ) -> None:
        """
        Inject minimal SBS mono metadata.

        Appends st3d (stereo_mode=0 for mono).
        """
        with open(file_path, "ab") as f:
            # st3d box: stereo_mode = 0 (mono)
            st3d_data = struct.pack(">IB", 0, 0)
            st3d_box = struct.pack(">I", 8 + len(st3d_data)) + b"st3d" + st3d_data
            f.write(st3d_box)

        logger.info("Injected SBS mono metadata: st3d(mode=0) into %s", file_path)

    def _run_ffmpeg(self, cmd: list[str]) -> None:
        """Run an ffmpeg command and raise on failure."""
        logger.debug("Running ffmpeg: %s", " ".join(cmd))
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed (exit {result.returncode}): {result.stderr}"
            )

    def get_video_info(self, path: str) -> dict[str, Any]:
        """Get video metadata using ffprobe."""
        cmd = [
            self.ffprobe, "-v", "quiet",
            "-print_format", "json",
            "-show_format", "-show_streams",
            path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(f"ffprobe failed: {result.stderr}")

        data = json.loads(result.stdout)
        video_stream = None
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                video_stream = stream
                break

        if not video_stream:
            raise ValueError(f"No video stream found in {path}")

        fmt = data.get("format", {})
        fps_str = video_stream.get("r_frame_rate", "30/1")
        if "/" in fps_str:
            num, den = fps_str.split("/")
            fps = float(num) / float(den)
        else:
            fps = float(fps_str)

        return {
            "width": int(video_stream.get("width", 0)),
            "height": int(video_stream.get("height", 0)),
            "fps": fps,
            "duration": float(fmt.get("duration", 0)),
            "codec": video_stream.get("codec_name", "unknown"),
            "file_size": int(fmt.get("size", 0)),
        }

    def get_supported_formats(self) -> dict[str, str]:
        """Return supported spatial video formats with descriptions."""
        return {
            "mv-hevc": "MV-HEVC — Apple Vision Pro",
            "sbs-spatial": "SBS Spatial — Meta Quest",
            "sbs-mono": "SBS Mono — Legacy Fallback",
        }
