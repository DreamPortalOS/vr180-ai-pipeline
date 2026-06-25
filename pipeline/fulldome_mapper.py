"""Fulldome (球幕) mapper — single-pass ffmpeg v360 fisheye domemaster renderer.

No depth/stereo/spherical metadata — pure mono fisheye projection via ffmpeg.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

log = logging.getLogger("fulldome-mapper")


class FulldomeMapper:
    """Map a flat 2D video to a fisheye domemaster for fulldome projection.

    Uses a single ffmpeg v360 pass over the whole video (not per-frame).
    The resulting output is a square fisheye image circle on black background,
    suitable for dome projection systems.

    Parameters
    ----------
    dome_fov : float
        Fisheye field-of-view of the output domemaster in degrees (default 180,
        up to 220 for some dome systems).
    coverage_h_fov : float
        How many degrees of horizontal FOV the source flat video covers on the
        input sphere (default 120). Lower values = screen-like patch; higher
        values = fuller dome but more geometric stretch.
    coverage_v_fov : float | None
        Vertical coverage FOV. If None, auto-computed from source aspect ratio
        using the horizontal coverage (default None).
    output_size : int
        Width and height of the square output domemaster in pixels (default 4096).
        Must be even.
    codec : str
        Video codec for output (default "h264").
    crf : int
        Constant rate factor for encoding quality (default 18).
    """

    def __init__(
        self,
        dome_fov: float = 180.0,
        coverage_h_fov: float = 120.0,
        coverage_v_fov: float | None = None,
        output_size: int = 4096,
        codec: str = "h264",
        crf: int = 18,
    ) -> None:
        if output_size % 2 != 0:
            output_size += 1  # ffmpeg requires even dimensions
        self.dome_fov = dome_fov
        self.coverage_h_fov = coverage_h_fov
        self.coverage_v_fov = coverage_v_fov
        self.output_size = output_size
        self.codec = codec
        self.crf = crf

    def convert(self, input_path: str, output_path: str) -> str:
        """Run single ffmpeg v360 pass over the whole video.

        Parameters
        ----------
        input_path : str
            Path to the source flat video.
        output_path : str
            Path for the output fisheye domemaster video.

        Returns
        -------
        str
            The output path on success.
        """
        input_path_obj = Path(input_path)
        if not input_path_obj.exists():
            raise FileNotFoundError(f"Input video not found: {input_path}")

        # Auto-compute coverage_v_fov from source aspect ratio if not given
        coverage_v_fov = self.coverage_v_fov
        if coverage_v_fov is None:
            coverage_v_fov = self._probe_coverage_v_fov(input_path)

        v360_filter = (
            f"v360=input=flat:output=fisheye"
            f":ih_fov={self.coverage_h_fov}"
            f":iv_fov={coverage_v_fov}"
            f":h_fov={self.dome_fov}"
            f":v_fov={self.dome_fov}"
            f":w={self.output_size}"
            f":h={self.output_size}"
        )

        codec_map = {"h264": "libx264", "h265": "libx265"}
        encoder = codec_map.get(self.codec, f"libx{self.codec}")

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            input_path,
            "-vf",
            v360_filter,
            "-c:v",
            encoder,
            "-crf",
            str(self.crf),
            "-pix_fmt",
            "yuv420p",
            "-an",
            output_path,
        ]

        log.info(
            "Running fulldome conversion: "
            f"dome_fov={self.dome_fov}° "
            f"coverage=({self.coverage_h_fov}×{coverage_v_fov})° "
            f"output={self.output_size}×{self.output_size} "
            f"codec={self.codec} crf={self.crf}"
        )
        log.debug(f"ffmpeg command: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg v360 conversion failed (exit {result.returncode}):\nstderr:\n{result.stderr[:2000]}"
            )

        log.info(f"✅ Fulldome domemaster written to {output_path}")
        return output_path

    def _probe_coverage_v_fov(self, input_path: str) -> float:
        """Probe source video dimensions and derive vertical coverage FOV.

        The vertical coverage FOV is computed to preserve the source aspect
        ratio within the input sphere defined by coverage_h_fov.
        """
        import json

        probe_cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
            input_path,
        ]
        try:
            probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=15)
            if probe_result.returncode != 0:
                log.warning(f"ffprobe failed, falling back to default iv_fov=90: {probe_result.stderr}")
                return 90.0
            info = json.loads(probe_result.stdout)
            streams = info.get("streams", [])
            if not streams:
                return 90.0
            w = int(streams[0].get("width", 1920))
            h = int(streams[0].get("height", 1080))
        except Exception as exc:
            log.warning(f"Could not probe source dimensions: {exc}")
            return 90.0

        if h <= 0:
            return 90.0

        aspect = h / w
        computed = self.coverage_h_fov * aspect
        log.info(
            f"Source {w}×{h} → auto iv_fov = {computed:.1f}° (from ih_fov={self.coverage_h_fov}° × aspect={aspect:.4f})"
        )
        return computed
