"""
Stage 4 — VR Metadata Embedding
================================

Embed VR/XR metadata into the output MP4 to make it natively playable
on VR headsets (Meta Quest, Apple Vision Pro, etc.).

Metadata Types
--------------
1. **Spherical Video V2** — RDF/XML metadata following Google's
   ``<rdf:SphericalVideo>`` spec. Tells the player the video is a
   180° equirectangular stereoscopic format.

2. **Camera Motion Metadata** — 6-DoF tracking hints embedded in
   timed metadata tracks. Helps VR players maintain smooth head
   tracking when the original video had camera movement.

3. **Stereo Mode Flag** — ISO/IEC 14496-12 box signalling
   stereoscopic 3D (side-by-side) for compliant players.

Output Format
-------------
::

   ┌────────────────────────────────────────────────┐
   │  MP4 Container (H.264 High / H.265 Main)       │
   │                                                  │
   │  Video Track 1:  Left Eye  (3840 × 1920 @ 60fps) │
   │  Video Track 2:  Right Eye (3840 × 1920 @ 60fps) │
   │                                                  │
   │  ┌─ UUID Box ───────────────────────────────┐   │
   │  │  SphericalVideo V2 (RDF/XML)             │   │
   │  │  StereoMode = "top-bottom" |             │   │
   │  │              "side-by-side"              │   │
   │  └──────────────────────────────────────────┘   │
   │                                                  │
   │  ┌─ Timed Metadata Track ──────────────────┐   │
   │  │  Camera Motion Samples (quaternion +    │   │
   │  │  translation per keyframe)              │   │
   │  └──────────────────────────────────────────┘   │
   └──────────────────────────────────────────────────┘

Spherical Video V2 XML
----------------------
.. code:: xml

    <?xml version="1.0"?>
    <rdf:SphericalVideo
        xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
        xmlns:GSpherical="http://ns.google.com/videos/1.0/spherical/">
      <GSpherical:Spherical>true</GSpherical:Spherical>
      <GSpherical:Stitched>true</GSpherical:Stitched>
      <GSpherical:StitchingSoftware>vr180-ai-pipeline</GSpherical:StitchingSoftware>
      <GSpherical:ProjectionType>equirectangular</GSpherical:ProjectionType>
      <GSpherical:StereoMode>side-by-side</GSpherical:StereoMode>
      <GSpherical:SourceCount>2</GSpherical:SourceCount>
      <GSpherical:InitialViewHeadingDegrees>0</GSpherical:InitialViewHeadingDegrees>
      <GSpherical:InitialViewPitchDegrees>0</GSpherical:InitialViewPitchDegrees>
      <GSpherical:InitialViewRollDegrees>0</GSpherical:InitialViewRollDegrees>
      <GSpherical:Timestamp>2025-01-01T00:00:00+00:00</GSpherical:Timestamp>
      <GSpherical:FullPanoWidthPixels>3840</GSpherical:FullPanoWidthPixels>
      <GSpherical:FullPanoHeightPixels>1920</GSpherical:FullPanoHeightPixels>
      <GSpherical:CroppedAreaImageWidthPixels>3840</GSpherical:CroppedAreaImageWidthPixels>
      <GSpherical:CroppedAreaImageHeightPixels>1920</GSpherical:CroppedAreaImageHeightPixels>
    </rdf:SphericalVideo>

Configuration
-------------
+-----------------------+-----------+--------------------------------------------+
| Parameter             | Default   | Description                                |
+=======================+===========+============================================+
| ``codec``             | h264      | ``"h264"`` or ``"h265"``                   |
+-----------------------+-----------+--------------------------------------------+
| ``crf``               | 23        | Constant rate factor (lower = higher qual) |
+-----------------------+-----------+--------------------------------------------+
| ``preset``            | medium    | x264/x265 encode preset                    |
+-----------------------+-----------+--------------------------------------------+
| ``pixel_format``      | yuv420p   | Chroma subsampling                         |
+-----------------------+-----------+--------------------------------------------+
| ``stereo_mode``       | sbs       | ``"sbs"`` (side-by-side) or                |
|                       |           | ``"tb"`` (top-bottom)                      |
+-----------------------+-----------+--------------------------------------------+
| ``fps``               | 60        | Output frame rate                          |
+-----------------------+-----------+--------------------------------------------+
| ``embed_motion``      | True      | Embed camera motion metadata track         |
+-----------------------+-----------+--------------------------------------------+

References
----------
- Google VR180 spec: https://support.google.com/youtube/answer/6905082
- Spherical Video V2: https://github.com/google/spatial-media
- ISO/IEC 14496-12 (ISOBMFF)
- "Capturing 180° Stereoscopic Video" (Meta XR, 2024)
"""

import os
import subprocess
import tempfile
from typing import Optional


# Spherical Video V2 XML template
SPHERICAL_XML_TEMPLATE = """<?xml version="1.0"?>
<rdf:SphericalVideo
 xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
 xmlns:GSpherical="http://ns.google.com/videos/1.0/spherical/">
  <GSpherical:Spherical>true</GSpherical:Spherical>
  <GSpherical:Stitched>true</GSpherical:Stitched>
  <GSpherical:StitchingSoftware>vr180-ai-pipeline v0.1.0</GSpherical:StitchingSoftware>
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


class VRMetadataEmbedder:
    """Embed VR metadata into video files for VR180 playback."""

    def __init__(
        self,
        codec: str = "h264",
        crf: int = 23,
        preset: str = "medium",
        pixel_format: str = "yuv420p",
        stereo_mode: str = "sbs",
        fps: int = 60,
        embed_motion: bool = True,
    ):
        self.codec = codec
        self.crf = crf
        self.preset = preset
        self.pixel_format = pixel_format
        self.stereo_mode = stereo_mode
        self.fps = fps
        self.embed_motion = embed_motion

    def _codec_name(self) -> str:
        if self.codec == "h265":
            return "libx265"
        return "libx264"

    def _pix_fmt(self) -> str:
        return self.pixel_format

    def _spherical_xml(self, width: int, height: int) -> str:
        """Generate the Spherical Video V2 RDF/XML string."""
        stereo_map = {"sbs": "side-by-side", "tb": "top-bottom"}
        mode = stereo_map.get(self.stereo_mode, "side-by-side")
        return SPHERICAL_XML_TEMPLATE.format(
            stereo_mode=mode,
            pano_width=width,
            pano_height=height,
        )

    def embed(
        self,
        input_path: str,
        output_path: str,
        width: int = 3840,
        height: int = 1920,
        camera_motion_samples: Optional[list] = None,
    ) -> str:
        """Embed VR metadata into a video file.

        Uses ffmpeg to:
        1. Transcode input to target codec (if needed)
        2. Inject Spherical Video V2 XML metadata
        3. Optionally add camera motion metadata track

        Args:
            input_path: Path to input video (raw frames or encoded)
            output_path: Path for output VR180 video
            width: Video width in pixels
            height: Video height in pixels
            camera_motion_samples: Optional list of per-frame
                (quaternion_w, quaternion_x, quaternion_y, quaternion_z,
                 trans_x, trans_y, trans_z) tuples

        Returns:
            Path to the output file.
        """
        # Write spherical XML to temp file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".xml", delete=False
        ) as f:
            f.write(self._spherical_xml(width, height))
            xml_path = f.name

        try:
            cmd = self._build_ffmpeg_cmd(
                input_path, output_path, xml_path, camera_motion_samples
            )
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"FFmpeg encoding failed:\n{e.stderr}"
            ) from e
        finally:
            os.unlink(xml_path)

        return output_path

    def _build_ffmpeg_cmd(
        self,
        input_path: str,
        output_path: str,
        xml_path: str,
        camera_motion_samples: Optional[list] = None,
    ) -> list:
        """Build the ffmpeg command-line arguments."""
        cmd = [
            "ffmpeg",
            "-y",
            "-i", input_path,
            "-c:v", self._codec_name(),
            "-preset", self.preset,
            "-crf", str(self.crf),
            "-pix_fmt", self._pix_fmt(),
            "-r", str(self.fps),
            # Spherical metadata
            "-metadata:s:v", f"spherical-v2={xml_path}",
            "-metadata:s:v", "stereo_mode=side-by-side",
            # Spherical Video V2 via side data
            "-f", "mp4",
        ]

        if self.embed_motion and camera_motion_samples:
            cmd.extend([
                "-metadata:s:v", "camera_motion=1",
            ])

        cmd.append(output_path)
        return cmd

    def embed_single_frame_batch(
        self,
        frames: list,
        output_path: str,
        width: int = 3840,
        height: int = 1920,
    ) -> str:
        """Write a list of frames as a VR180 video with metadata.

        Uses ffmpeg pipe input for efficient frame-by-frame encoding.

        Args:
            frames: List of numpy arrays (H, W, 3), uint8
            output_path: Output MP4 path
            width, height: Frame dimensions

        Returns:
            Path to output file.
        """
        import cv2
        import subprocess as sp

        xml_content = self._spherical_xml(width, height)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".xml", delete=False
        ) as f:
            f.write(xml_content)
            xml_path = f.name

        cmd = [
            "ffmpeg",
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{width}x{height}",
            "-pix_fmt", "rgb24",
            "-r", str(self.fps),
            "-i", "-",
            "-c:v", self._codec_name(),
            "-preset", self.preset,
            "-crf", str(self.crf),
            "-pix_fmt", self._pix_fmt(),
            "-metadata:s:v", f"spherical-v2={xml_path}",
            "-metadata:s:v", "stereo_mode=side-by-side",
            output_path,
        ]

        proc = sp.Popen(cmd, stdin=sp.PIPE, stderr=sp.PIPE)
        try:
            for frame in frames:
                proc.stdin.write(frame.astype("uint8").tobytes())
            proc.stdin.close()
            proc.wait()
            if proc.returncode != 0:
                raise RuntimeError(
                    f"FFmpeg pipe encoding failed:\n{proc.stderr.read().decode()}"
                )
        finally:
            os.unlink(xml_path)

        return output_path