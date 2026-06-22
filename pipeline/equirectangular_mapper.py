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
from typing import Optional, Tuple


class EquirectangularMapper:
    """Map planar stereo views to VR180 equirectangular format.

    Output: 7680×1920 SBS frame (2 × 3840×1920 hemispheres).
    """

    def __init__(
        self,
        output_width: int = 3840,
        output_height: int = 1920,
        src_hfov: float = 70.0,    # Source camera horizontal FOV (degrees)
        use_ffmpeg: bool = True,   # Prefer ffmpeg v360 when available
    ):
        self.output_width = output_width
        self.output_height = output_height
        self.src_hfov = src_hfov
        self.use_ffmpeg = use_ffmpeg
        self._mesh: Optional[Tuple[np.ndarray, np.ndarray]] = None

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
        import subprocess, shutil
        if not shutil.which("ffmpeg"):
            return False
        try:
            result = subprocess.run(
                ["ffmpeg", "-filters"],
                capture_output=True, text=True, timeout=5
            )
            return "v360" in result.stdout
        except Exception:
            return False

    def _map_via_ffmpeg(self, frame: np.ndarray) -> np.ndarray:
        """Use ffmpeg v360 filter for equirectangular mapping.

        Maps a flat perspective image (with src_hfov FOV) onto a
        180° hemispherical equirectangular projection.
        """
        import subprocess, tempfile, os
        import cv2

        H, W = frame.shape[:2]

        # Write input frame to temp PNG
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as in_f:
            in_path = in_f.name
            cv2.imwrite(in_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        out_path = in_path.replace(".png", "_eq.png")
        try:
            cmd = [
                "ffmpeg", "-y",
                "-i", in_path,
                "-vf",
                f"v360=input=flat:output=hequirect:"
                f"ih_fov={self.src_hfov}:iv_fov={self.src_hfov}:"
                f"h_fov=180:v_fov=180:"
                f"w={self.output_width}:h={self.output_height}",
                "-frames:v", "1",
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
        """Pure OpenCV equirectangular mapping (fallback)."""
        import cv2

        H_src, W_src = frame.shape[:2]
        W_out, H_out = self.output_width, self.output_height

        if self._mesh is None:
            self._build_mesh(W_src, H_src)

        sx, sy = self._mesh
        equirect = cv2.remap(frame, sx, sy, cv2.INTER_LANCZOS4,
                             borderMode=cv2.BORDER_REPLICATE)
        return equirect

    def _build_mesh(self, src_width: int, src_height: int):
        """Pre-compute equirectangular→planar mapping mesh.

        For each output pixel (u, v) in the 3840×1920 equirect frame:
          1. Compute spherical direction (theta, phi)
          2. Project to the flat source camera's sensor plane
          3. Sample at (sx, sy)
        """
        W_out, H_out = self.output_width, self.output_height

        # Output pixel grid
        u, v = np.meshgrid(np.arange(W_out), np.arange(H_out))
        u = u.astype(np.float32)
        v = v.astype(np.float32)

        # Spherical coordinates for a 180° hemisphere
        # theta: -90° to +90° (horizontal), phi: 0° (top) to 180° (bottom)
        theta = (u / W_out - 0.5) * np.pi      # [-π/2, π/2]
        phi = (v / H_out) * np.pi               # [0, π]

        # Ray direction in 3D
        ray_x = np.sin(theta) * np.sin(phi)
        ray_y = np.cos(phi)
        ray_z = np.cos(theta) * np.sin(phi)

        # Project onto source camera plane (pinhole model)
        f = src_width / (2.0 * np.tan(np.radians(self.src_hfov / 2.0)))
        cx, cy = src_width / 2.0, src_height / 2.0

        sx = f * ray_x / np.maximum(ray_z, 1e-6) + cx
        sy = f * ray_y / np.maximum(ray_z, 1e-6) + cy

        # Clamp to valid source region
        sx = np.clip(sx, 0, src_width - 1)
        sy = np.clip(sy, 0, src_height - 1)

        self._mesh = (sx.astype(np.float32), sy.astype(np.float32))

    def map_stereo_pair(
        self, left_frame: np.ndarray, right_frame: np.ndarray
    ) -> np.ndarray:
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
