"""
Stage 3 — Equirectangular Projection
=====================================

Map planar stereo views onto a 180° hemisphere using equirectangular
projection. The output is a 7680×1920 SBS frame (3840×1920 per eye).

Two strategies:
1. **ffmpeg v360 filter** (recommended) — Uses ffmpeg's built-in
   v360 filter for fast, correct perspective→equirectangular mapping.
2. **OpenCV remap** (fallback) — Pure NumPy/OpenCV implementation.

Usage:
    from pipeline.equirectangular_mapper import EquirectangularMapper
    mapper = EquirectangularMapper()
    sbs_frame = mapper.map_stereo_pair(left_frame, right_frame)
"""

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
        import subprocess

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

    def _map_via_ffmpeg(self, frame: np.ndarray) -> np.ndarray:
        """Use ffmpeg v360 filter for equirectangular mapping.

        Maps a flat perspective image (with src_hfov FOV) onto a
        180° hemispherical equirectangular projection.

        Key: iv_fov is calculated from the source aspect ratio, NOT set
        to src_hfov. This prevents stretching the content to fill the
        full 180° vertical range when the source only covers ~40-50°.
        """
        import os
        import subprocess
        import tempfile

        import cv2

        H, W = frame.shape[:2]
        src_vfov = self._calc_vertical_fov(W, H)

        # Write input frame to temp PNG
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as in_f:
            in_path = in_f.name
            cv2.imwrite(in_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        out_path = in_path.replace(".png", "_eq.png")
        try:
            # v360 filter: perspective → half-equirectangular (VR180)
            # ih_fov/iv_fov = source camera FOV; h_fov=180/v_fov=180 = full hemisphere output.
            # No vflip: ffmpeg v360 hequirect output is already Quest/YouTube-compatible.
            vfilter = (
                f"v360=input=flat:output=hequirect:"
                f"ih_fov={self.src_hfov}:iv_fov={src_vfov:.2f}:"
                f"h_fov=180:v_fov=180:"
                f"w={self.output_width}:h={self.output_height}"
            )

            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                in_path,
                "-vf",
                vfilter,
                "-frames:v",
                "1",
                out_path,
            ]
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
        Vertical flip is applied if flip_vertical is True.
        """
        import cv2

        H_src, W_src = frame.shape[:2]
        _W_out, _H_out = self.output_width, self.output_height

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
        # Use separate focal lengths for horizontal and vertical
        hfov_rad = math.radians(self.src_hfov)
        fx = src_width / (2.0 * math.tan(hfov_rad / 2.0))
        fy = fx  # square pixels assumed

        cx, cy = src_width / 2.0, src_height / 2.0

        # Mask: only project rays that are in front of the camera (ray_z > 0)
        # and within the source FOV
        valid = ray_z > 0.01  # slightly above zero for stability

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

    def reset_mesh(self):
        """Force mesh rebuild on next map call."""
        self._mesh = None
