"""
Stage 3 — Equirectangular Projection
=====================================

Map planar stereo views onto a 180° hemisphere using equirectangular
projection. The output is a 7680×1920 SBS frame (3840×1920 per eye).

Two strategies:
1. **ffmpeg v360 filter** (recommended) — Uses ffmpeg's built-in
   v360 filter for fast, correct perspective→equirectangular mapping.
2. **OpenCV remap** (fallback) — Pure NumPy/OpenCV implementation.

Batched mode (default):
  Instead of calling ffmpeg 2×N times (once per eye per frame), the
  ``map_sequence()`` method writes all frames as a temporary image
  sequence and runs ffmpeg **once per eye** on the whole video.
  This yields ~10× speedup for long clips.

Usage:
    from pipeline.equirectangular_mapper import EquirectangularMapper
    mapper = EquirectangularMapper()
    sbs_frame = mapper.map_stereo_pair(left_frame, right_frame)
    sbs_frames = mapper.map_sequence(left_frames, right_frames, temp_dir)
"""

import os
import subprocess
from pathlib import Path

import cv2
import numpy as np


class EquirectangularMapper:
    """Map planar stereo views to VR180 equirectangular format.

    Output: 7680×1920 SBS frame (2 × 3840×1920 hemispheres).

    Key behavior for non-180° source footage:
    - Source content is placed centered in the equirectangular frame
    - Regions outside the source FOV are filled with black (not stretched)
    - Vertical flip is applied to match Quest/VR headset convention
    """

    def __init__(
        self,
        output_width: int = 1920,
        output_height: int = 1920,
        src_hfov: float = 90.0,
        use_ffmpeg: bool = True,
    ):
        """Configure the equirectangular mapper.

        Args:
            output_width: Per-eye equirectangular width (px).
                Default 1920 → square 1:1 per eye for comfortable VR180.
                3840 gives sharper full-resolution output at higher render cost.
            output_height: Per-eye equirectangular height (px).
                Default 1920 (square 1:1). Matches output_width for square per-eye.
            src_hfov: Source camera horizontal field of view (degrees).
                Default 90° — good tradeoff for most AI-generated and action-cam
                footage. Higher (e.g. 120°) fills more of the 180° dome but
                introduces more peripheral stretch. Lower (e.g. 70°) gives a
                "binoculars" feel with less stretch but worse immersion.
            use_ffmpeg: Prefer ffmpeg v360 filter when available.
        """
        self.output_width = output_width
        self.output_height = output_height
        self.src_hfov = src_hfov
        self.use_ffmpeg = use_ffmpeg
        self._mesh: tuple[np.ndarray, np.ndarray] | None = None

    def map_single(self, frame: np.ndarray) -> np.ndarray:
        """Map a single planar frame to equirectangular VR180.

        Args:
            frame: Input planar image (H, W, 3), uint8

        Returns:
            Equirectangular frame (output_height, output_width, 3), uint8
        """
        if self.use_ffmpeg and self._ffmpeg_available():
            return self._map_via_ffmpeg(frame)
        else:
            return self._map_via_opencv(frame)

    def _ffmpeg_available(self) -> bool:
        """Check if ffmpeg with v360 filter is available."""
        import shutil

        if not shutil.which("ffmpeg"):
            return False
        try:
            result = subprocess.run(["ffmpeg", "-filters"], capture_output=True, text=True, timeout=5)
            return "v360" in result.stdout
        except Exception:
            return False

    def _calc_vertical_fov(self, src_width: int, src_height: float) -> float:
        """Calculate vertical FOV from horizontal FOV and aspect ratio.

        For a pinhole camera: vfov = 2 * atan(tan(hfov/2) * height/width)
        """
        import math

        hfov_rad = math.radians(self.src_hfov)
        vfov_rad = 2.0 * math.atan(math.tan(hfov_rad / 2.0) * src_height / src_width)
        return math.degrees(vfov_rad)

    def _v360_filter(self, src_width: int, src_height: int) -> str:
        """Build v360 filter string for perspective → half-equirectangular."""
        src_vfov = self._calc_vertical_fov(src_width, src_height)
        return (
            f"v360=input=flat:output=hequirect:"
            f"ih_fov={self.src_hfov}:iv_fov={src_vfov:.2f}:"
            f"h_fov=180:v_fov=180:"
            f"w={self.output_width}:h={self.output_height}"
        )

    def _map_via_ffmpeg(self, frame: np.ndarray) -> np.ndarray:
        """Use ffmpeg v360 filter for equirectangular mapping.

        Maps a flat perspective image (with src_hfov FOV) onto a
        180° hemispherical equirectangular projection.
        """
        import tempfile

        H, W = frame.shape[:2]

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as in_f:
            in_path = in_f.name
            cv2.imwrite(in_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        out_path = in_path.replace(".png", "_eq.png")
        try:
            vfilter = self._v360_filter(W, H)
            cmd = ["ffmpeg", "-y", "-i", in_path, "-vf", vfilter, "-frames:v", "1", out_path]
            subprocess.run(cmd, check=True, capture_output=True, timeout=30)

            out_img = cv2.imread(out_path)
            if out_img is None:
                raise RuntimeError("ffmpeg v360 failed to produce output")
            return cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
        finally:
            try:
                os.unlink(in_path)
                if os.path.exists(out_path):
                    os.unlink(out_path)
            except OSError:
                pass

    def _map_via_opencv(self, frame: np.ndarray) -> np.ndarray:
        """Pure OpenCV equirectangular mapping (fallback).

        Pixels outside the source camera's FOV are filled with black.
        """
        import cv2

        H_src, W_src = frame.shape[:2]

        if self._mesh is None:
            self._build_mesh(W_src, H_src)

        sx, sy = self._mesh

        # Create mask for valid pixels (those within source bounds)
        valid_mask = sx >= 0

        # Replace invalid coords with 0 for remap (will be masked later)
        sx_safe = np.where(valid_mask, sx, 0.0).astype(np.float32)
        sy_safe = np.where(valid_mask, sy, 0.0).astype(np.float32)

        equirect = cv2.remap(
            frame, sx_safe, sy_safe, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
        )

        # Apply black fill for out-of-FOV regions
        if not np.all(valid_mask):
            equirect[~valid_mask] = [0, 0, 0]

        return equirect

    def _build_mesh(self, src_width: int, src_height: int):
        """Pre-compute equirectangular→planar mapping mesh.

        For each output pixel (u, v) in the equirect frame:
          1. Compute spherical direction (theta, phi)
          2. Project to the flat source camera's sensor plane
          3. Sample at (sx, sy)

        Pixels outside the source camera's FOV are marked as -1
        and filled with black instead of being stretched.
        """
        import math

        W_out, H_out = self.output_width, self.output_height

        # Output pixel grid
        u, v = np.meshgrid(np.arange(W_out), np.arange(H_out))
        u = u.astype(np.float32)
        v = v.astype(np.float32)

        # Spherical coordinates for a 180° hemisphere
        # theta: -90° to +90° (horizontal), phi: 0° (top) to 180° (bottom)
        theta = (u / W_out - 0.5) * np.pi  # [-π/2, π/2]
        phi = (v / H_out) * np.pi  # [0, π]

        # Ray direction in 3D
        ray_x = np.sin(theta) * np.sin(phi)
        ray_y = np.cos(phi)
        ray_z = np.cos(theta) * np.sin(phi)

        # Project onto source camera plane (pinhole model)
        hfov_rad = math.radians(self.src_hfov)
        fx = src_width / (2.0 * math.tan(hfov_rad / 2.0))
        fy = fx  # square pixels assumed

        cx, cy = src_width / 2.0, src_height / 2.0

        # Mask: only project rays that are in front of the camera (ray_z > 0)
        valid = ray_z > 0.01

        # ray_y is positive-up; image y is positive-down → negate for correct mapping
        sx = np.where(valid, fx * ray_x / np.maximum(ray_z, 1e-6) + cx, -1.0)
        sy = np.where(valid, -fy * ray_y / np.maximum(ray_z, 1e-6) + cy, -1.0)

        # Check if projected point is within source image bounds
        in_bounds = valid & (sx >= 0) & (sx < src_width) & (sy >= 0) & (sy < src_height)

        # Mark out-of-bounds pixels for black fill
        sx = np.where(in_bounds, sx, -1.0)
        sy = np.where(in_bounds, sy, -1.0)

        self._mesh = (sx.astype(np.float32), sy.astype(np.float32))

    def map_stereo_pair(self, left_frame: np.ndarray, right_frame: np.ndarray) -> np.ndarray:
        """Map left+right views into a SBS equirectangular frame.

        Each view is independently mapped to equirectangular,
        then concatenated side-by-side → 7680×1920.

        Args:
            left_frame: Left eye planar image (H, W, 3), uint8
            right_frame: Right eye planar image (H, W, 3), uint8

        Returns:
            SBS equirect frame (H_out, W_out*2, 3), uint8
        """
        left_eq = self.map_single(left_frame)
        right_eq = self.map_single(right_frame)
        return np.concatenate([left_eq, right_eq], axis=1)

    # ------------------------------------------------------------------
    # Batched processing — ~10× faster than per-frame ffmpeg calls
    # ------------------------------------------------------------------

    def _write_image_sequence(
        self,
        frames: list[np.ndarray],
        prefix: str,
        output_dir: str,
    ) -> tuple[int, int]:
        """Write frames as a PNG image sequence and return (height, width) of first frame."""
        for i, frame in enumerate(frames):
            path = os.path.join(output_dir, f"{prefix}_{i:06d}.png")
            cv2.imwrite(path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        H, W = frames[0].shape[:2]
        return H, W

    def _read_image_sequence(
        self,
        prefix: str,
        num_frames: int,
        input_dir: str,
    ) -> list[np.ndarray]:
        """Read back an image sequence as RGB ndarrays."""
        result = []
        for i in range(num_frames):
            path = os.path.join(input_dir, f"{prefix}_{i:06d}.png")
            img = cv2.imread(path)
            if img is None:
                raise RuntimeError(f"Missing frame: {path}")
            result.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return result

    def _run_ffmpeg_v360_on_dir(
        self,
        pattern: str,
        output_dir: str,
        out_prefix: str,
        w: int,
        h: int,
        num_frames: int,
    ):
        """Run ffmpeg v360 **once** on an image sequence directory.

        Writes output frames as ``{out_prefix}_000000.png`` etc.
        """
        vfilter = self._v360_filter(w, h)

        # Use %06d pattern for glob input
        in_pattern = os.path.join(output_dir, pattern).replace("\\", "/")
        out_pattern = os.path.join(output_dir, f"{out_prefix}_%06d.png").replace("\\", "/")

        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            "30",
            "-i",
            in_pattern,
            "-vf",
            vfilter,
            "-frames:v",
            str(num_frames),
            "-start_number",
            "0",
            out_pattern,
        ]
        subprocess.run(cmd, check=True, capture_output=True, timeout=600)

    def map_sequence(
        self,
        left_frames: list[np.ndarray],
        right_frames: list[np.ndarray],
        temp_dir: str,
    ) -> list[np.ndarray]:
        """Map a sequence of left/right frames to SBS equirectangular in batch.

        Instead of calling ffmpeg 2×N times, writes PNG sequences to
        ``temp_dir``, runs ffmpeg v360 **once per eye** on the full
        sequence, then reads back the results.  ~10× faster for long clips.

        Falls back to per-frame OpenCV mapping if ffmpeg v360 is unavailable.

        Args:
            left_frames: List of left eye frames (H, W, 3), uint8
            right_frames: List of right eye frames (H, W, 3), uint8
            temp_dir: Writable directory for intermediate PNG sequences.

        Returns:
            List of SBS equirect frames (H_out, W_out*2, 3), uint8
        """
        if not left_frames or not right_frames:
            raise ValueError("Empty frame lists")

        if self.use_ffmpeg and self._ffmpeg_available():
            return self._map_sequence_via_ffmpeg(left_frames, right_frames, temp_dir)
        else:
            # Fallback: call map_single per frame via existing per-frame path
            return [self.map_stereo_pair(left, right) for left, right in zip(left_frames, right_frames, strict=False)]

    def _map_sequence_via_ffmpeg(
        self,
        left_frames: list[np.ndarray],
        right_frames: list[np.ndarray],
        temp_dir: str,
    ) -> list[np.ndarray]:
        """Batch equirect via single ffmpeg call per eye."""
        tmp = Path(temp_dir) / "_equirect_batch"
        tmp.mkdir(parents=True, exist_ok=True)
        num = len(left_frames)

        # 1. Write input PNG sequences
        lw, lh = self._write_image_sequence(left_frames, "L_in", str(tmp))
        rw, rh = self._write_image_sequence(right_frames, "R_in", str(tmp))

        # 2. Run ffmpeg v360 once per eye on the whole sequence
        self._run_ffmpeg_v360_on_dir("L_in_%06d.png", str(tmp), "L_out", lw, lh, num)
        self._run_ffmpeg_v360_on_dir("R_in_%06d.png", str(tmp), "R_out", rw, rh, num)

        # 3. Read back equirect results
        left_eq = self._read_image_sequence("L_out", num, str(tmp))
        right_eq = self._read_image_sequence("R_out", num, str(tmp))

        # 4. Build SBS pairs
        sbs_frames = [np.concatenate([left, right], axis=1) for left, right in zip(left_eq, right_eq, strict=False)]

        # 5. Cleanup temp images (keep the dir itself for cache)
        import contextlib

        for fname in os.listdir(str(tmp)):
            fp = os.path.join(str(tmp), fname)
            with contextlib.suppress(OSError):
                os.unlink(fp)

        return sbs_frames

    def map_video(
        self,
        left_video: str,
        right_video: str,
        temp_dir: str,
        output_path: str,
        fps: int = 30,
    ) -> str:
        """Map an entire left/right eye video pair to an equirect SBS video.

        Uses a **single** ffmpeg v360 pass per eye on the whole video
        (if ffmpeg v360 is available), then concats left+right eq
        frames into SBS video.

        Falls back to per-frame OpenCV mapping if v360 filter
        is unavailable.

        Args:
            left_video: Path to left eye video file.
            right_video: Path to right eye video file.
            temp_dir: Temporary directory for frame extraction.
            output_path: Path for the output SBS equirectangular video.
            fps: Output framerate.

        Returns:
            ``output_path`` on success.
        """
        if self.use_ffmpeg and self._ffmpeg_available():
            return self._map_video_via_ffmpeg(left_video, right_video, temp_dir, output_path, fps)

        # Fallback: extract frames, map per-frame, re-encode
        extract_dir = Path(temp_dir) / "_equirect_vid_frames"
        extract_dir.mkdir(parents=True, exist_ok=True)

        # Extract left frames
        left_pat = str(extract_dir / "L_%06d.png").replace("\\", "/")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                left_video,
                "-vf",
                f"fps={fps}",
                "-start_number",
                "0",
                left_pat,
            ],
            check=True,
            capture_output=True,
            timeout=600,
        )

        # Extract right frames
        right_pat = str(extract_dir / "R_%06d.png").replace("\\", "/")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                right_video,
                "-vf",
                f"fps={fps}",
                "-start_number",
                "0",
                right_pat,
            ],
            check=True,
            capture_output=True,
            timeout=600,
        )

        # Count frames
        left_files = sorted(extract_dir.glob("L_*.png"))
        right_files = sorted(extract_dir.glob("R_*.png"))
        num = min(len(left_files), len(right_files))

        left_frames: list[np.ndarray] = []
        right_frames: list[np.ndarray] = []
        for i in range(num):
            lf = cv2.imread(str(left_files[i]))
            rf = cv2.imread(str(right_files[i]))
            if lf is None or rf is None:
                raise RuntimeError(f"Missing extracted frame at index {i}")
            left_frames.append(cv2.cvtColor(lf, cv2.COLOR_BGR2RGB))
            right_frames.append(cv2.cvtColor(rf, cv2.COLOR_BGR2RGB))

        sbs_frames = self.map_sequence(left_frames, right_frames, temp_dir)

        # Re-encode to output video
        out_pat = str(extract_dir / "SBS_%06d.png").replace("\\", "/")
        for i, sbs in enumerate(sbs_frames):
            cv2.imwrite(out_pat.replace("%06d", f"{i:06d}"), cv2.cvtColor(sbs, cv2.COLOR_RGB2BGR))

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-framerate",
                str(fps),
                "-i",
                out_pat,
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                output_path,
            ],
            check=True,
            capture_output=True,
            timeout=600,
        )

        # Cleanup frame dir
        import shutil

        shutil.rmtree(str(extract_dir), ignore_errors=True)

        return output_path

    def _map_video_via_ffmpeg(
        self,
        left_video: str,
        right_video: str,
        temp_dir: str,
        output_path: str,
        fps: int = 30,
    ) -> str:
        """Batch equirectangular mapping of a whole video pair.

        Uses a single ffmpeg v360 filter per eye, avoiding per-frame
        spawning overhead entirely.
        """
        # Determine source dimensions from first video
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
                "csv=p=0",
                left_video,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        parts = probe.stdout.strip().split(",")
        w, h = int(parts[0]), int(parts[1])

        vfilter = self._v360_filter(w, h)

        def _encode_eye(input_video: str, tag: str) -> str:
            outpath = os.path.join(temp_dir, f"_eq_eye_{tag}.mp4")
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                input_video,
                "-vf",
                vfilter,
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                outpath,
            ]
            subprocess.run(cmd, check=True, capture_output=True, timeout=600)
            return outpath

        left_eq = _encode_eye(left_video, "L")
        right_eq = _encode_eye(right_video, "R")

        # Concatenate side-by-side via hstack filter
        cmd_sbs = [
            "ffmpeg",
            "-y",
            "-i",
            left_eq,
            "-i",
            right_eq,
            "-filter_complex",
            "hstack=inputs=2",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            output_path,
        ]
        subprocess.run(cmd_sbs, check=True, capture_output=True, timeout=600)

        # Cleanup intermediate files
        import contextlib

        for f in [left_eq, right_eq]:
            with contextlib.suppress(OSError):
                os.unlink(f)

        return output_path
