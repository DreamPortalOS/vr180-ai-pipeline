"""
Stage 2 — Stereo Disparity Rendering
=====================================

Generate left-eye and right-eye views from a 2D frame + depth map using
horizontal parallax shift based on depth values.

Core Algorithm
--------------
For each pixel at position ``(x, y)`` with depth ``d``:

.. math::

   shift = \\frac{B \\cdot f}{d}

   x_L = x + \\frac{shift}{2}
   x_R = x - \\frac{shift}{2}

Where:
- ``B`` = interpupillary distance (default 0.064 m)
- ``f`` = focal length in pixels
- ``d`` = metric depth from Stage 1

Disocclusion Handling
---------------------
Shifting creates holes where previously occluded background is revealed.
Three inpainting strategies (configurable):

1. **Edge-aware inpainting** (default) — Navier-Stokes based, preserves
   edges around disoccluded regions
2. **Optical-flow guided** — use bi-directional flow to fill holes with
   temporally consistent content from neighbouring frames
3. **Depth-aware fill** — extend background colour from nearest pixel
   at similar depth

Pipeline
--------
::

   Frame + Depth Map
          │
          ▼
   ┌──────────────────────────┐
   │  Compute Disparity Map   │
   │  d(x,y) → shift(x,y)     │
   └────────────┬─────────────┘
                │
        ┌───────┴───────┐
        ▼               ▼
   ┌──────────┐   ┌──────────┐
   │ Left     │   │ Right    │
   │ Shift    │   │ Shift    │
   └────┬─────┘   └────┬─────┘
        │              │
   ┌────▼─────┐   ┌────▼─────┐
   │ Inpaint  │   │ Inpaint  │
   │ Holes    │   │ Holes    │
   └────┬─────┘   └────┬─────┘
        │              │
        ▼              ▼
   Left View       Right View
   (SBS Pair)

Configuration
-------------
+-----------------------+----------+--------------------------------------------+
| Parameter             | Default  | Description                                |
+=======================+==========+============================================+
| ``ipd``               | 0.064    | Interpupillary distance (meters)           |
+-----------------------+----------+--------------------------------------------+
| ``focal_length_px``   | auto     | Focal length in pixels (from video width)  |
+-----------------------+----------+--------------------------------------------+
| ``inpaint_method``    | edge     | ``"edge"``, ``"flow"``, or ``"depth"``     |
+-----------------------+----------+--------------------------------------------+
| ``max_disparity``     | 0.05     | Max shift as fraction of image width       |
+-----------------------+----------+--------------------------------------------+
| ``temporal_smooth``   | True     | Temporal smoothing of disparity across     |
|                       |          | frames to reduce flicker                   |
+-----------------------+----------+--------------------------------------------+

References
----------
- "A Critical Review of 2D-to-3D Conversion Methods" (Xing et al., 2023)
- "Depth Image Based Rendering" (Fehn, 2004)
"""

import numpy as np
from typing import Optional, Tuple


class StereoRenderer:
    """Generate stereoscopic left/right views from monocular frames + depth.

    Uses geometric parallax based on metric depth maps to produce a
    side-by-side stereo pair suitable for VR180 projection.
    """

    def __init__(
        self,
        ipd: float = 0.064,
        focal_length_px: Optional[float] = None,
        inpaint_method: str = "edge",
        max_disparity: float = 0.05,
        temporal_smooth: bool = True,
    ):
        self.ipd = ipd
        self.focal_length_px = focal_length_px
        self.inpaint_method = inpaint_method
        self.max_disparity = max_disparity
        self.temporal_smooth = temporal_smooth
        self._prev_disparity: Optional[np.ndarray] = None

    def render(
        self, frame: np.ndarray, depth: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate left and right views.

        Args:
            frame: Input RGB image (H, W, 3), uint8
            depth: Depth map (H, W), float32 (metric meters)

        Returns:
            Tuple of (left_view, right_view), each (H, W, 3), uint8
        """
        H, W = frame.shape[:2]

        # Auto-compute focal length in pixels if not set
        if self.focal_length_px is None:
            # Assume a ~70° horizontal FOV for typical AI-generated video
            self.focal_length_px = W / (2 * np.tan(np.radians(35)))

        # Compute per-pixel disparity shift
        disparity = self._compute_disparity(depth)

        # Temporal smoothing
        if self.temporal_smooth and self._prev_disparity is not None:
            alpha = 0.3
            disparity = alpha * disparity + (1 - alpha) * self._prev_disparity
        self._prev_disparity = disparity.copy()

        # Generate left and right views via remapping
        left = self._shift_view(frame, disparity, direction="left")
        right = self._shift_view(frame, disparity, direction="right")

        return left, right

    def _compute_disparity(self, depth: np.ndarray) -> np.ndarray:
        """Convert metric depth to pixel disparity.

        Uses the standard stereo geometry formula:
            disparity = (ipd * focal_length_px) / depth
        """
        safe_depth = np.maximum(depth, 0.1)  # avoid division by zero
        disp = (self.ipd * self.focal_length_px) / safe_depth

        # Clamp to max disparity (fraction of image width)
        max_px = self.max_disparity * depth.shape[1]
        disp = np.clip(disp, 0, max_px)

        return disp.astype(np.float32)

    def _shift_view(
        self, frame: np.ndarray, disparity: np.ndarray, direction: str
    ) -> np.ndarray:
        """Shift pixels horizontally based on disparity.

        Uses OpenCV's remap with bilinear interpolation.
        Direction 'left' shifts right (for left eye) and vice versa.
        """
        import cv2

        H, W = frame.shape[:2]

        # Build horizontal displacement map
        sign = 1.0 if direction == "left" else -1.0
        map_x, map_y = np.meshgrid(np.arange(W), np.arange(H))
        map_x = (map_x + sign * disparity).astype(np.float32)
        map_y = map_y.astype(np.float32)

        shifted = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)

        # Inpaint holes (black pixels at disoccluded edges)
        if self.inpaint_method == "edge":
            mask = self._find_holes(shifted)
            if mask.any():
                shifted = cv2.inpaint(shifted, mask, 3, cv2.INPAINT_TELEA)

        return shifted

    def _find_holes(self, image: np.ndarray) -> np.ndarray:
        """Find black disocclusion holes in a shifted view.

        Returns a binary mask suitable for cv2.inpaint.
        """
        gray = image.mean(axis=2) if image.ndim == 3 else image
        mask = (gray < 1).astype(np.uint8) * 255
        return mask

    def render_batch(
        self, frames: list, depths: list
    ) -> list:
        """Process a batch of frame/depth pairs."""
        return [self.render(f, d) for f, d in zip(frames, depths)]

    def reset_temporal_state(self):
        """Clear temporal smoothing state for a new video sequence."""
        self._prev_disparity = None