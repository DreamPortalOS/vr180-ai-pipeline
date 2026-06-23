"""
Stage 2 — Stereo Disparity Rendering
=====================================

Generate left-eye and right-eye views from a 2D frame + depth map using
horizontal parallax shift based on depth values.

Core Algorithm
--------------
For each pixel at position (x, y) with depth d:

    shift = (ipd * focal_length_px) / d
    x_L = x + shift / 2   (left eye — shift right)
    x_R = x - shift / 2   (right eye — shift left)
"""


import numpy as np


class StereoRenderer:
    """Generate stereoscopic left/right views from monocular frames + depth.

    Uses geometric parallax based on depth maps to produce a
    side-by-side stereo pair suitable for VR180 projection.
    """

    def __init__(
        self,
        ipd: float = 0.064,           # Interpupillary distance in meters
        focal_length_px: float | None = None,
        max_disparity: float = 0.05,   # Max shift as fraction of image width
        temporal_smooth: bool = True,
        convergence: float = 0.3,      # Convergence plane depth (fraction of max depth)
    ):
        self.ipd = ipd
        self.focal_length_px = focal_length_px
        self.max_disparity = max_disparity
        self.temporal_smooth = temporal_smooth
        self.convergence = convergence
        self._prev_disparity: np.ndarray | None = None

    def render(
        self, frame: np.ndarray, depth: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate left and right views.

        Args:
            frame: Input RGB image (H, W, 3), uint8
            depth: Depth map (H, W), float32 (normalized [0,1] or metric)

        Returns:
            Tuple of (left_view, right_view), each (H, W, 3), uint8
        """
        import cv2

        H, W = frame.shape[:2]

        # Auto-compute focal length in pixels (use local variable to avoid side effects)
        focal_length_px = self.focal_length_px
        if focal_length_px is None:
            # Assume ~70° horizontal FOV
            focal_length_px = W / (2 * np.tan(np.radians(35)))

        # Compute per-pixel disparity shift
        disparity = self._compute_disparity(depth, focal_length_px)

        # Temporal smoothing
        if self.temporal_smooth and self._prev_disparity is not None:
            alpha = 0.3
            disparity = alpha * disparity + (1 - alpha) * self._prev_disparity
        self._prev_disparity = disparity.copy()

        # Build remap grids
        grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
        grid_x = grid_x.astype(np.float32)
        grid_y = grid_y.astype(np.float32)

        # Left eye: shift right (positive x direction)
        left_x = grid_x + disparity
        left_view = cv2.remap(frame, left_x, grid_y, cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)

        # Right eye: shift left (negative x direction)
        right_x = grid_x - disparity
        right_view = cv2.remap(frame, right_x, grid_y, cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)

        # Inpaint disocclusion holes
        left_view = self._inpaint_holes(left_view)
        right_view = self._inpaint_holes(right_view)

        return left_view, right_view

    def _compute_disparity(self, depth: np.ndarray, focal_length_px: float) -> np.ndarray:
        """Convert depth to pixel disparity.

        Formula: disparity = (ipd * focal_length_px) / depth
        Closer objects get larger disparity (more 3D pop-out).
        """
        # Normalize depth to a meaningful range
        d_min, d_max = depth.min(), depth.max()
        depth_norm = (depth - d_min) / (d_max - d_min) if d_max > d_min else np.zeros_like(depth)

        # Convergence plane: objects at convergence depth have zero disparity
        # Objects closer than convergence pop out (positive disparity)
        # Objects farther recede (negative disparity)
        d_conv = self.convergence
        depth_rel = d_conv - depth_norm  # positive = closer than convergence

        # Compute disparity
        max_px = self.max_disparity * depth.shape[1]
        disp = depth_rel * max_px * 2  # Scale to use full range

        return np.clip(disp, -max_px, max_px).astype(np.float32)

    def _inpaint_holes(self, image: np.ndarray) -> np.ndarray:
        """Find and inpaint disocclusion holes (black/zero strips at edges).

        Uses edge-aware detection: only detects holes near image borders
        to avoid false positives on legitimately dark regions in the scene.
        """
        import cv2

        # Detect black regions (holes from shifting)
        gray = image.mean(axis=2)
        raw_mask = (gray < 1).astype(np.uint8)

        # Only consider holes within a border region (where disocclusion occurs)
        # This prevents false positives on legitimately dark scene content
        _H, W = raw_mask.shape
        border_width = max(int(W * 0.05), 5)  # 5% of width or at least 5px
        border_mask = np.zeros_like(raw_mask)
        border_mask[:border_width, :] = 1        # top
        border_mask[-border_width:, :] = 1       # bottom
        border_mask[:, :border_width] = 1        # left
        border_mask[:, -border_width:] = 1       # right

        # Combine: only inpaint dark pixels near borders
        mask = (raw_mask & border_mask).astype(np.uint8) * 255

        if mask.sum() > 0:
            # Dilate mask slightly to catch edge pixels
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
            image = cv2.inpaint(image, mask, 5, cv2.INPAINT_TELEA)

        return image

    def render_batch(
        self, frames: list, depths: list
    ) -> list:
        """Process a batch of frame/depth pairs."""
        return [self.render(f, d) for f, d in zip(frames, depths, strict=False)]

    def reset_temporal_state(self):
        """Clear temporal smoothing state for a new video sequence."""
        self._prev_disparity = None
