"""
Stage 3 — Equirectangular Projection
=====================================

Map a planar stereo pair onto a 180° hemisphere using equirectangular
projection. The output is a 3840 × 1920 equirectangular frame ready for
VR180 playback.

Geometry
--------
Equirectangular projection maps spherical coordinates (θ, φ) to planar
coordinates (u, v) linearly:

.. math::

   u = \\frac{\\theta}{2\\pi} \\cdot W    \\quad
   v = \\frac{\\pi - \\phi}{\\pi} \\cdot H

For VR180 the horizontal field of view is 180° (π rad), so the mapping
covers only the front hemisphere.

Coordinate System
-----------------
::

   Spherical (θ, φ):  θ ∈ [-π/2, π/2]  (horizontal, ±90°)
                       φ ∈ [0, π]       (vertical, top-to-bottom)

   Equirect UV:        u ∈ [0, W)       rightward
                       v ∈ [0, H)       downward

   W = 3840, H = 1920  (2:1 aspect)

Mesh Warp
---------
A fixed grid mesh is pre-computed for the projection, mapping each
equirectangular output pixel to its source location in the planar stereo
frame. This is applied via OpenCV remap for speed.

::

           Source (Planar)              Destination (Equirect 180°)
       ┌─────────────────┐            ╭─────────────────────╮
       │                 │            │    φ=0 (top pole)   │
       │  Original       │   warp     │    ┌───────────┐    │
       │  + Stereo Shift │  ──────►   │    │ 180° view │    │
       │                 │            │    └───────────┘    │
       └─────────────────┘            │   φ=π (bottom pole) │
                                      ╰─────────────────────╯
                                         3840 × 1920

Configuration
-------------
+-------------------+----------+--------------------------------------------+
| Parameter         | Default  | Description                                |
+===================+==========+============================================+
| ``output_width``  | 3840     | Equirectangular frame width (pixels)       |
+-------------------+----------+--------------------------------------------+
| ``output_height`` | 1920     | Equirectangular frame height (pixels)      |
+-------------------+----------+--------------------------------------------+
| ``hfov``          | 180      | Horizontal field of view (degrees)         |
+-------------------+----------+--------------------------------------------+
| ``vfov``          | 180      | Vertical field of view (degrees)           |
+-------------------+----------+--------------------------------------------+
| ``interpolation`` | Lanczos  | ``"lanczos"``, ``"cubic"``, ``"linear"``   |
+-------------------+----------+--------------------------------------------+

References
----------
- Google VR180 specification: https://vr.google.com/vr180/
- Equirectangular projection: https://wiki.panotools.org/Equirectangular_Projection
- OpenCV: Fisheye / omnidirectional camera models
"""

import numpy as np
from typing import Optional, Tuple


class EquirectangularMapper:
    """Map planar stereo views to VR180 equirectangular format.

    Pre-computes the mesh grid on first call and caches it for
    efficient per-frame mapping.
    """

    def __init__(
        self,
        output_width: int = 3840,
        output_height: int = 1920,
        hfov: float = 180.0,
        vfov: float = 180.0,
        interpolation: str = "lanczos",
    ):
        self.output_width = output_width
        self.output_height = output_height
        self.hfov = hfov
        self.vfov = vfov
        self.interpolation = interpolation
        self._mesh: Optional[Tuple[np.ndarray, np.ndarray]] = None

    def _build_mesh(self, src_width: int, src_height: int):
        """Pre-compute the equirectangular → planar mapping mesh.

        For each output pixel (u, v):
          1. Convert to spherical coordinates (θ, φ)
          2. Project to planar camera coordinates (x, y)
          3. Map to source pixel (sx, sy)
        """
        W_out, H_out = self.output_width, self.output_height
        W_src, H_src = src_width, src_height

        # Output pixel grid
        u, v = np.meshgrid(np.arange(W_out), np.arange(H_out))
        u = u.astype(np.float32)
        v = v.astype(np.float32)

        # Spherical coordinates (radians)
        # θ: [-hfov/2, +hfov/2]  →  [-π/2, π/2]
        # φ: [top, bottom]       →  [π, 0]
        theta = (u / W_out - 0.5) * np.radians(self.hfov)
        phi = (1.0 - v / H_out) * np.radians(self.vfov)

        # Spherical → planar pinhole projection
        # Assuming a standard perspective camera looking along +Z
        f = W_src / (2.0 * np.tan(np.radians(min(self.hfov, 120.0) / 2.0)))
        cx, cy = W_src / 2.0, H_src / 2.0

        # Ray direction in camera space
        ray_x = np.sin(theta) * np.sin(phi)
        ray_y = np.cos(phi)
        ray_z = np.cos(theta) * np.sin(phi)

        # Perspective projection
        sx = f * ray_x / np.maximum(ray_z, 1e-6) + cx
        sy = f * ray_y / np.maximum(ray_z, 1e-6) + cy

        # Clamp to valid source region
        sx = np.clip(sx, 0, W_src - 1)
        sy = np.clip(sy, 0, H_src - 1)

        self._mesh = (sx, sy)

    def _get_interp_flag(self):
        import cv2
        flags = {
            "lanczos": cv2.INTER_LANCZOS4,
            "cubic": cv2.INTER_CUBIC,
            "linear": cv2.INTER_LINEAR,
        }
        return flags.get(self.interpolation, cv2.INTER_LANCZOS4)

    def map(self, frame: np.ndarray) -> np.ndarray:
        """Map a planar frame to equirectangular VR180.

        Args:
            frame: Input planar image (H, W, 3), uint8.
                   Can be a side-by-side stereo pair (2×W, H) or mono.

        Returns:
            Equirectangular frame (self.output_height, self.output_width, 3), uint8.
        """
        import cv2

        H_src, W_src = frame.shape[:2]

        if self._mesh is None:
            self._build_mesh(W_src, H_src)

        sx, sy = self._mesh
        interp = self._get_interp_flag()

        equirect = cv2.remap(
            frame, sx, sy, interp,
            borderMode=cv2.BORDER_REPLICATE,
        )

        return equirect

    def map_stereo_pair(
        self, left_frame: np.ndarray, right_frame: np.ndarray
    ) -> np.ndarray:
        """Map left+right views into a SBS equirectangular frame.

        Each view is independently mapped to a half-width equirect,
        then concatenated side-by-side.

        Args:
            left_frame: Left eye planar image (H, W, 3)
            right_frame: Right eye planar image (H, W, 3)

        Returns:
            SBS equirect frame (H_out, W_out*2, 3)
        """
        left_eq = self.map(left_frame)
        right_eq = self.map(right_frame)
        return np.concatenate([left_eq, right_eq], axis=1)

    def reset_mesh(self):
        """Force mesh rebuild on next map call."""
        self._mesh = None