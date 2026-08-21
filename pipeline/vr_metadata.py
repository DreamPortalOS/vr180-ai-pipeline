"""
Stage 4 — VR Metadata Embedding
================================

Embed Spherical Video V2 metadata into the output MP4 to signal VR180
format to YouTube VR and VR headset players.

This stage uses ffmpeg to:
1. Encode the SBS frame sequence to H.264/H.265
2. Inject the Google Spherical Video V2 RDF/XML metadata
3. Set correct stereo mode flags

Usage:
    from pipeline.vr_metadata import VRMetadataEmbedder
    embedder = VRMetadataEmbedder()
    embedder.embed_single_frame_batch(frames, "output.mp4")
    embedder.embed_frame_stream(iter(frames), "output.mp4")  # O(chunk) RAM
"""

import contextlib
import logging
import os
import subprocess
import tempfile
from collections.abc import Iterable

import numpy as np

from pipeline.spherical_injector import inject_spherical_metadata

log = logging.getLogger(__name__)

# Google Spherical Video V2 XML template (for VR180)
SPHERICAL_XML_TEMPLATE = """<?xml version="1.0"?>
<rdf:SphericalVideo
 xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
 xmlns:GSpherical="http://ns.google.com/videos/1.0/spherical/">
  <GSpherical:Spherical>true</GSpherical:Spherical>
  <GSpherical:Stitched>true</GSpherical:Stitched>
  <GSpherical:StitchingSoftware>vr180-ai-pipeline v0.2.0</GSpherical:StitchingSoftware>
  <GSpherical:ProjectionType>equirectangular</GSpherical:ProjectionType>
  <GSpherical:StereoMode>{stereo_mode}</GSpherical:StereoMode>
  <GSpherical:SourceCount>2</GSpherical:SourceCount>
  <GSpherical:InitialViewHeadingDegrees>0</GSpherical:InitialViewHeadingDegrees>
  <GSpherical:InitialViewPitchDegrees>0</GSpherical:InitialViewPitchDegrees>
  <GSpherical:InitialViewRollDegrees>0</GSpherical:InitialViewRollDegrees>
  <GSpherical:FullPanoWidthPixels>{pano_width}</GSpherical:FullPanoWidthPixels>
  <GSpherical:FullPanoHeightPixels>{pano_height}</GSpherical:FullPanoHeightPixels>
  <GSpherical:CroppedAreaImageWidthPixels>{pano_width}</GSpherical:CroppedAreaImageWidthPixels>
  <GSpherical:CroppedAreaImageHeightPixels>{pano_height}</GSpherical:CroppedAreaImageHeightPixels>
</rdf:SphericalVideo>"""

# How many bytes of ffmpeg stderr tail to keep for error messages.
STDERR_TAIL_BYTES = 4096


def _read_stderr_tail(stderr_file) -> str:
    """Read the tail of ffmpeg's captured stderr temp file for error messages."""
    try:
        stderr_file.flush()
        stderr_file.seek(0, 2)
        size = stderr_file.tell()
        stderr_file.seek(max(0, size - STDERR_TAIL_BYTES))
        tail = stderr_file.read()
        text = tail.decode("utf-8", errors="replace").strip()
        return text or "(ffmpeg produced no stderr output)"
    except (OSError, ValueError):
        return "(stderr unreadable)"


class VRMetadataEmbedder:
    """Embed VR metadata into video files for VR180 playback."""

    def __init__(
        self,
        codec: str = "h264",
        crf: int = 23,
        preset: str = "medium",
        fps: int = 30,
        stereo_mode: str = "sbs",
        bitrate: str | None = None,
    ):
        self.codec = codec
        self.crf = crf
        self.preset = preset
        self.fps = fps
        self.stereo_mode = stereo_mode
        self.bitrate = bitrate

    def _codec_name(self) -> str:
        return "libx265" if self.codec == "h265" else "libx264"

    def _spherical_xml(self, width: int, height: int) -> str:
        """Generate Spherical Video V2 XML string."""
        stereo_map = {"sbs": "side-by-side", "tb": "top-bottom"}
        mode = stereo_map.get(self.stereo_mode, "side-by-side")
        return SPHERICAL_XML_TEMPLATE.format(
            stereo_mode=mode,
            pano_width=width,
            pano_height=height,
        )

    def _build_encode_cmd(self, width: int, height: int, temp_path: str) -> list[str]:
        """Build the ffmpeg rawvideo-pipe encode command."""
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{width}x{height}",
            "-pix_fmt",
            "rgb24",
            "-r",
            str(self.fps),
            "-i",
            "-",
            "-c:v",
            self._codec_name(),
            "-preset",
            self.preset,
        ]
        if self.bitrate:
            # Explicit target bitrate (e.g. "80M") overrides CRF.
            cmd += ["-b:v", self.bitrate]
        else:
            cmd += ["-crf", str(self.crf)]
        cmd += [
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            temp_path,
        ]
        return cmd

    def _encode_and_inject(
        self,
        frames: Iterable[np.ndarray],
        output_path: str,
        width: int,
        height: int,
        xml_path: str,
        stream: bool,
    ) -> str:
        """Encode frames to temp MP4 via ffmpeg pipe, then inject sv3d/st3d."""
        temp_path = output_path + ".tmp.mp4"
        cmd = self._build_encode_cmd(width, height, temp_path)

        if stream:
            # Chunked path (V-4): write frames one at a time and drain stderr
            # from a temp file — peak RAM is one frame, never the joined bytes.
            stderr_file = tempfile.TemporaryFile(prefix="vr180-meta-stderr-")  # noqa: SIM115
            try:
                proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=stderr_file)
                frame_count = 0
                try:
                    for frame in frames:
                        proc.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())
                        frame_count += 1
                except (BrokenPipeError, OSError) as e:
                    raise RuntimeError(
                        f"ffmpeg encoder died after {frame_count} frames ({e}). "
                        f"stderr: {_read_stderr_tail(stderr_file)}"
                    ) from None
                finally:
                    with contextlib.suppress(BrokenPipeError, OSError):
                        proc.stdin.close()
                    proc.wait()
                if proc.returncode != 0:
                    raise RuntimeError(
                        f"FFmpeg encoding failed (exit code {proc.returncode}) after {frame_count} frames:\n"
                        f"{_read_stderr_tail(stderr_file)}"
                    )
                if frame_count == 0:
                    raise ValueError("No frames to encode")
            finally:
                with contextlib.suppress(OSError):
                    stderr_file.close()
        else:
            # Legacy whole-batch path: hand all frames to communicate(), which
            # writes stdin and drains stderr concurrently. Writing frames in a
            # loop while ffmpeg's stderr PIPE fills unread deadlocks once the
            # OS pipe buffer (~64 KB) is full.
            raw = b"".join(frame.astype(np.uint8).tobytes() for frame in frames)
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            _, stderr = proc.communicate(input=raw)

            if proc.returncode != 0:
                raise RuntimeError(f"FFmpeg encoding failed:\n{stderr.decode(errors='replace')}")

        # Post-process: inject spherical box via Python ISOBMFF writer
        inject_spherical_metadata(temp_path, output_path, width=width, height=height, stereo_mode=self.stereo_mode)
        with contextlib.suppress(OSError):
            os.unlink(temp_path)
        return output_path

    def embed_single_frame_batch(
        self,
        frames: list[np.ndarray],
        output_path: str,
        width: int | None = None,
        height: int | None = None,
    ) -> str:
        """Write frames as VR180 video with metadata via ffmpeg pipe.

        Uses ffmpeg's -spherical flag (ffmpeg 8.x+) if available, otherwise
        falls back to side data XML injection.

        Args:
            frames: List of numpy arrays (H, W, 3), uint8 (SBS equirect frames)
            output_path: Output MP4 path
            width, height: Override frame dimensions (auto-detected from frames)

        Returns:
            Path to output file
        """
        if not frames:
            raise ValueError("No frames to encode")

        H, W = frames[0].shape[:2]
        out_w = width or W
        out_h = height or H

        xml_content = self._spherical_xml(out_w, out_h)

        # Write XML to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
            f.write(xml_content)
            xml_path = f.name

        try:
            self._encode_and_inject(frames, output_path, out_w, out_h, xml_path, stream=False)
        except FileNotFoundError:
            print("[Metadata] ffmpeg not found!")
            raise
        finally:
            with contextlib.suppress(OSError):
                os.unlink(xml_path)

        print(f"[Metadata] ✅ VR180 video saved to {output_path}")
        return output_path

    def embed_frame_stream(
        self,
        frames: Iterable[np.ndarray],
        output_path: str,
        width: int,
        height: int,
    ) -> str:
        """Encode frames from an iterable with O(1)-per-frame RAM (V-4).

        Identical output to :meth:`embed_single_frame_batch`, but frames are
        consumed lazily and written to ffmpeg's stdin one at a time — the
        caller may pass a generator that reads checkpoint PNGs from disk, so
        peak memory is a single frame instead of the whole sequence plus its
        joined byte buffer.

        Args:
            frames: Iterable of (H, W, 3) uint8 SBS equirect frames.
            output_path: Output MP4 path.
            width, height: Frame dimensions (required — no auto-detect).

        Returns:
            Path to output file.
        """
        xml_content = self._spherical_xml(width, height)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
            f.write(xml_content)
            xml_path = f.name

        try:
            self._encode_and_inject(frames, output_path, width, height, xml_path, stream=True)
        except FileNotFoundError:
            print("[Metadata] ffmpeg not found!")
            raise
        finally:
            with contextlib.suppress(OSError):
                os.unlink(xml_path)

        print(f"[Metadata] ✅ VR180 video saved to {output_path}")
        return output_path

    def _embed_via_metadata_file(
        self,
        frames: list[np.ndarray],
        output_path: str,
        xml_path: str,
    ) -> str:
        """Fallback: embed VR metadata using ffmpeg -metadata flags.

        First encode to temp file, then remux with metadata.
        """
        H, W = frames[0].shape[:2]
        temp_path = output_path + ".tmp.mp4"

        # First pass: encode without metadata
        cmd1 = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{W}x{H}",
            "-pix_fmt",
            "rgb24",
            "-r",
            str(self.fps),
            "-i",
            "-",
            "-c:v",
            self._codec_name(),
            "-preset",
            self.preset,
            "-crf",
            str(self.crf),
            "-pix_fmt",
            "yuv420p",
            temp_path,
        ]

        # See embed_single_frame_batch: communicate() drains stderr concurrently
        # to avoid the pipe-buffer deadlock that a manual write loop hits.
        raw = b"".join(frame.astype(np.uint8).tobytes() for frame in frames)
        proc = subprocess.Popen(cmd1, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        _, stderr = proc.communicate(input=raw)

        if proc.returncode != 0:
            raise RuntimeError(f"FFmpeg first-pass failed:\n{stderr.decode(errors='replace')}")

        # Second pass: inject spherical metadata
        # Read the XML content
        with open(xml_path) as f:
            xml_content = f.read()

        cmd2 = [
            "ffmpeg",
            "-y",
            "-i",
            temp_path,
            "-c",
            "copy",
            "-metadata:s:v",
            f"spherical-v2={xml_content}",
            "-metadata:s:v",
            "stereo_mode=side-by-side",
            output_path,
        ]

        result = subprocess.run(cmd2, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[Metadata] Warning: metadata injection failed: {result.stderr[:200]}")
            # Fall back to temp file
            import shutil

            shutil.move(temp_path, output_path)
        else:
            with contextlib.suppress(OSError):
                os.unlink(temp_path)

        print(f"[Metadata] ✅ VR180 video saved to {output_path}")
        return output_path

    def embed(
        self,
        input_path: str,
        output_path: str,
        width: int = 7680,
        height: int = 1920,
    ) -> str:
        """Embed VR metadata into an already-encoded video file.

        Uses ISOBMFF binary injection (sv3d + st3d boxes) for proper
        Spherical Video V2 metadata that VR players can recognize.

        Args:
            input_path: Path to input MP4 video
            output_path: Path to output MP4 with VR metadata
            width: Full panorama width in pixels
            height: Full panorama height in pixels

        Returns:
            Path to output file
        """
        inject_spherical_metadata(input_path, output_path, width=width, height=height, stereo_mode=self.stereo_mode)
        return output_path
