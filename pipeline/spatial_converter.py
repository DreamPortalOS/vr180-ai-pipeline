"""
Spatial Video Converter — Convert VR180 pipeline output to spatial video formats.

Supports:
- MV-HEVC (Apple Vision Pro compatible)
- SBS Spatial (Meta Quest compatible)
- SBS Mono (legacy fallback)

Each output format includes proper ISOBMFF metadata boxes for spatial playback.
The boxes are written by the shared :mod:`pipeline.spherical_injector` (issue
#281) — this module only maps each format to a stereo mode.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pipeline.spherical_injector import inject_spherical_metadata

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
                input_path,
                output_path,
                width,
                height,
                fps,
                crf,
            )
        elif target_format == SpatialFormat.SBS_SPATIAL:
            result = self._convert_sbs_spatial(
                input_path,
                output_path,
                width,
                height,
                fps,
                crf,
            )
        elif target_format == SpatialFormat.SBS_MONO:
            result = self._convert_sbs_mono(
                input_path,
                output_path,
                width,
                height,
                fps,
                crf,
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
                self.ffmpeg,
                "-y",
                "-i",
                input_path,
                "-filter_complex",
                f"[0:v]split=2[left][right];"
                f"[left]crop={eye_width}:{eye_height}:0:0[l];"
                f"[right]crop={eye_width}:{eye_height}:{eye_width}:0[r];"
                f"[l][r]hstack=inputs=2[out]",
                "-map",
                "[out]",
                "-c:v",
                "libx265",
                "-crf",
                str(crf),
                "-preset",
                "fast",
                "-tag:v",
                "hvc1",
                "-pix_fmt",
                "yuv420p",
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
            self.ffmpeg,
            "-y",
            "-i",
            input_path,
            "-c:v",
            "libx264",
            "-crf",
            str(crf),
            "-preset",
            "fast",
            "-pix_fmt",
            "yuv420p",
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
            self.ffmpeg,
            "-y",
            "-i",
            input_path,
            "-c:v",
            "libx264",
            "-crf",
            str(crf),
            "-preset",
            "fast",
            "-pix_fmt",
            "yuv420p",
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

    # ------------------------------------------------------------------
    # Spherical metadata (st3d + sv3d) — shared injector, mode mapping only
    #
    # Issue #281: this module used to carry its own writer that *appended*
    # st3d/sv3d after the last top-level box (outside moov, invisible to any
    # parser), emitted a proj without prhd/equi, and tagged SBS as
    # stereo_mode 1 (top-bottom per the RFC; left-right is 2).  Each format
    # now maps to a mode that pipeline.spherical_injector understands and
    # the injector writes, fixes up and self-checks the boxes.
    # ------------------------------------------------------------------

    def _inject_mv_hevc_metadata(
        self,
        file_path: str,
        eye_width: int,
        eye_height: int,
    ) -> None:
        """Inject spherical metadata for the MV-HEVC output.

        Stereo mode ``mono`` (st3d 0), as before: each eye is meant to be its
        own view rather than a packed frame.
        """
        self._inject_spherical_metadata(file_path, "mono", eye_width, eye_height)

    def _inject_sbs_spatial_metadata(
        self,
        file_path: str,
        width: int,
        height: int,
    ) -> None:
        """Inject spherical metadata for the SBS spatial output: stereo mode ``sbs`` (st3d 2, left-right)."""
        self._inject_spherical_metadata(file_path, "sbs", width, height)

    def _inject_sbs_mono_metadata(
        self,
        file_path: str,
        width: int,
        height: int,
    ) -> None:
        """Inject spherical metadata for the SBS mono fallback: stereo mode ``mono`` (st3d 0)."""
        self._inject_spherical_metadata(file_path, "mono", width, height)

    def _inject_spherical_metadata(self, file_path: str, stereo_mode: str, width: int, height: int) -> None:
        """Run :func:`pipeline.spherical_injector.inject_spherical_metadata` on *file_path* in place.

        The injector writes input -> output, so it is pointed at a sibling temp
        file that then replaces *file_path*; on any failure the original is left
        untouched and the temp file is removed.
        """
        tmp_output = f"{file_path}.vr.mp4"
        try:
            inject_spherical_metadata(file_path, tmp_output, width=width, height=height, stereo_mode=stereo_mode)
            os.replace(tmp_output, file_path)
        finally:
            if os.path.exists(tmp_output):
                os.remove(tmp_output)

        logger.info("Injected st3d(%s) + sv3d into %s via pipeline.spherical_injector", stereo_mode, file_path)

    def _run_ffmpeg(self, cmd: list[str]) -> None:
        """Run an ffmpeg command and raise on failure."""
        logger.debug("Running ffmpeg: %s", " ".join(cmd))
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed (exit {result.returncode}): {result.stderr}")

    def get_video_info(self, path: str) -> dict[str, Any]:
        """Get video metadata using ffprobe."""
        cmd = [
            self.ffprobe,
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
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
