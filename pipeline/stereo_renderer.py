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
        ipd: float = 0.064,  # Interpupillary distance in meters
        focal_length_px: float | None = None,
        max_disparity: float = 0.02,  # Max shift as fraction of image width (~0.02 for comfortable VR180)
        temporal_smooth: bool = True,
        convergence: float = 0.3,  # Convergence plane depth (fraction of max depth)
        src_hfov: float | None = None,  # Source horizontal FOV in degrees (None = auto from projection)
    ):
        self.ipd = ipd
        self.focal_length_px = focal_length_px
        self.max_disparity = max_disparity
        self.temporal_smooth = temporal_smooth
        self.convergence = convergence
        self.src_hfov = src_hfov
        self._prev_disparity: np.ndarray | None = None

    def render(self, frame: np.ndarray, depth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Generate left and right views.

        Args:
            frame: Input RGB image (H, W, 3), uint8
            depth: Depth map (H, W), float32 (normalized [0,1] or metric)

        Returns:
            Tuple of (left_view, right_view), each (H, W, 3), uint8
        """
        import cv2

        H, W = frame.shape[:2]

        # Auto-compute focal length in pixels from source hfov (or legacy 70° default)
        focal_length_px = self.focal_length_px
        if focal_length_px is None:
            hfov_deg = self.src_hfov if self.src_hfov is not None else 70.0
            focal_length_px = W / (2 * np.tan(np.radians(hfov_deg / 2)))

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
        left_view = cv2.remap(frame, left_x, grid_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        # Right eye: shift left (negative x direction)
        right_x = grid_x - disparity
        right_view = cv2.remap(frame, right_x, grid_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        # Inpaint disocclusion holes — pass remap coords so holes along object
        # edges (not just image borders) are detected and repaired.
        left_view = self._inpaint_holes(left_view, map_x=left_x, map_y=grid_y)
        right_view = self._inpaint_holes(right_view, map_x=right_x, map_y=grid_y)

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

    def _inpaint_holes(
        self, image: np.ndarray, map_x: np.ndarray | None = None, map_y: np.ndarray | None = None
    ) -> np.ndarray:
        """Find and inpaint disocclusion holes.

        Two detection paths:

        1. ``map_x``/``map_y`` given (from ``render``): any pixel whose source
           coordinate falls **outside** the image is a disocclusion hole — mask
           it.  This catches holes **anywhere** in the frame, including along
           object edges (the bug that produced "transparent" spots on the drone
           body and crystal).
        2. No maps given (legacy callers): fall back to black-pixel detection
           with a border-restricted mask (original behaviour).

        The border-only mask was the historical bug: it left disocclusion
        along object contours untouched, which the stereo warp then filled
        with ``BORDER_REPLICATE`` smearing — the "transparent" artefact the
        owner reported.
        """
        import cv2

        H, W = image.shape[:2]

        if map_x is not None and map_y is not None:
            # Path 1: precise — pixels that map outside the source image.
            outside = (map_x < 0) | (map_x >= W) | (map_y < 0) | (map_y >= H)
            mask = outside.astype(np.uint8) * 255
        else:
            # Path 2: legacy — only border dark pixels.
            gray = image.mean(axis=2)
            raw_mask = (gray < 1).astype(np.uint8)
            border_width = max(int(W * 0.05), 5)
            border_mask = np.zeros_like(raw_mask)
            border_mask[:border_width, :] = 1
            border_mask[-border_width:, :] = 1
            border_mask[:, :border_width] = 1
            border_mask[:, -border_width:] = 1
            mask = (raw_mask & border_mask).astype(np.uint8) * 255

        if mask.sum() > 0:
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
            image = cv2.inpaint(image, mask, 5, cv2.INPAINT_TELEA)

        return image

    def render_batch(self, frames: list, depths: list) -> list:
        """Process a batch of frame/depth pairs."""
        return [self.render(f, d) for f, d in zip(frames, depths, strict=False)]

    def render_sequence_chunked(
        self,
        frames: list,
        depths: list,
        *,
        chunk_size: int | None = None,
        overlap: int = 0,
    ) -> list:
        """Render a frame/depth sequence in memory-bounded chunks (V-4, #37).

        Identical output to ``[self.render(f, d) for f, d in zip(frames,
        depths)]`` — chunking is a memory optimisation only.  Because the same
        ``StereoRenderer`` instance drives every chunk, the temporal state
        (``_prev_disparity``) is **continuous** across chunk boundaries, so the
        result is bit-exact with ``overlap=0``.  ``overlap`` is accepted (and
        ignored for correctness — it only costs recompute) so callers that
        also feed other finite-memory stages can pass one uniform value.

        Peak memory is proportional to ``chunk_size``, not to the clip length:
        only one chunk's frames/depths are in RAM at a time.

        Args:
            frames: Left/mono source frames.
            depths: Matching depth maps (same length).
            chunk_size: Frames per chunk (default :func:`default_chunk_size`).
            overlap: Warmup frames per chunk (default 0 — sufficient here).

        Returns:
            List of ``(left, right)`` pairs, same length/order as input.
        """
        from pipeline.chunked_processor import process_in_chunks

        pairs = list(zip(frames, depths, strict=True))

        def _process_chunk(chunk_pairs, warm_offset, emit_offset):
            outs = [self.render(f, d) for f, d in chunk_pairs]
            return iter(outs[warm_offset:])

        return list(process_in_chunks(pairs, _process_chunk, chunk_size=chunk_size, overlap=overlap))

    def reset_temporal_state(self):
        """Clear temporal smoothing state for a new video sequence."""
        self._prev_disparity = None
