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

Occlusion handling (G-8, #355)
------------------------------
A *backward* remap indexed by the destination pixel's own depth (the pre-#355
behaviour) never produces a hole away from the image border: every output pixel
samples somewhere inside the source.  What it produces instead is a silhouette
that does not move while its content does — so a band of width ``≈ disparity``
around every depth discontinuity carries foreground colour in one eye and
background colour in the other.  That band is the "extra edge" the owner saw,
and its width grows with disparity, which is why it survived the #354
full-frame inpainting fix.

The renderer therefore *forward*-warps (splats) each eye with a z-buffer so the
silhouette actually moves.  The pixels left empty behind a near object are the
true disocclusions; they are filled from the **background side** of the
contour (the neighbour with the smaller disparity), never from the foreground.
"""

import logging

import numpy as np

log = logging.getLogger(__name__)


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
        occlusion_aware: bool = True,  # G-8 (#355): z-buffered forward warp + background-side fill
        edge_align: bool = True,  # G-8 (#355): snap disparity edges to image edges (guided filter)
    ):
        self.ipd = ipd
        self.focal_length_px = focal_length_px
        self.max_disparity = max_disparity
        self.temporal_smooth = temporal_smooth
        self.convergence = convergence
        self.src_hfov = src_hfov
        self.occlusion_aware = occlusion_aware
        self.edge_align = edge_align
        self._prev_disparity: np.ndarray | None = None
        #: Fraction of pixels filled as disocclusion in the last :meth:`render`,
        #: as ``{"left": float, "right": float}`` (G-8 #355 — also logged).
        self.last_fill_ratio: dict[str, float] = {"left": 0.0, "right": 0.0}

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

        # G-8 (#355): snap disparity edges onto image edges before warping, so
        # the silhouette we move matches the silhouette the viewer sees.  A
        # depth map that is "fatter" than the object drags a ring of background
        # along with the foreground — visible as the same halo.
        if self.edge_align:
            disparity = self._align_disparity_edges(disparity, frame)

        # Temporal smoothing
        if self.temporal_smooth and self._prev_disparity is not None:
            alpha = 0.3
            disparity = alpha * disparity + (1 - alpha) * self._prev_disparity
        self._prev_disparity = disparity.copy()

        if self.occlusion_aware:
            left_view, left_fill = self._warp_eye(frame, disparity, sign=+1)
            right_view, right_fill = self._warp_eye(frame, disparity, sign=-1)
            self.last_fill_ratio = {"left": left_fill, "right": right_fill}
            if left_fill or right_fill:
                log.debug(
                    "disocclusion fill: left %.3f%% / right %.3f%% of pixels",
                    left_fill * 100.0,
                    right_fill * 100.0,
                )
            return left_view, right_view

        # Legacy backward-remap path (pre-#355), kept for A/B comparison.
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

    def _align_disparity_edges(self, disparity: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """Snap disparity edges onto image edges (G-8, #355).

        He et al.'s guided filter with the source frame's luma as the guide.
        Where the guide is flat the result is a local mean (harmless
        smoothing); where the guide has an edge the result keeps that edge.
        A depth map whose silhouette is *fatter* than the object therefore
        stops dragging a ring of background along with the foreground — that
        ring is one of the two things the owner saw as an "extra edge".

        Built from ``cv2.boxFilter`` alone: ``cv2.ximgproc.guidedFilter``
        lives in opencv-contrib, which this project does not depend on.
        """
        import cv2

        H, W = disparity.shape[:2]
        radius = max(2, round(min(H, W) * 0.01))
        ksize = radius * 2 + 1
        if ksize >= min(H, W):  # image too small to filter meaningfully
            return disparity.astype(np.float32)

        if frame.ndim == 3 and frame.shape[2] == 3:
            guide = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)
        else:
            guide = frame.reshape(H, W, -1).mean(axis=2).astype(np.float32)
        guide /= 255.0
        src = disparity.astype(np.float32)

        def box(x: np.ndarray) -> np.ndarray:
            return cv2.boxFilter(x, -1, (ksize, ksize), normalize=True, borderType=cv2.BORDER_REFLECT)

        mean_g, mean_d = box(guide), box(src)
        var_g = box(guide * guide) - mean_g * mean_g
        cov_gd = box(guide * src) - mean_g * mean_d

        # Guide is in [0, 1]; eps sets the luma contrast below which the
        # filter smooths rather than preserves (~1% of full range).
        a = cov_gd / (var_g + 1e-4)
        b = mean_d - a * mean_g
        return (box(a) * guide + box(b)).astype(np.float32)

    def _warp_eye(self, frame: np.ndarray, disparity: np.ndarray, sign: int) -> tuple[np.ndarray, float]:
        """Forward-warp (splat) one eye with a z-buffer, then fill from the background.

        ``sign=+1`` renders the left eye, ``sign=-1`` the right, matching the
        legacy backward-remap convention (``left_x = x + disparity``): a source
        pixel at ``x`` lands at ``x - sign * disparity``.

        Unlike a backward remap, a forward splat actually *moves* the
        silhouette, so the band around a depth discontinuity becomes an
        explicit hole instead of silently carrying foreground colour into the
        other eye.  Each hole is then filled from the **background side** —
        the horizontal neighbour with the *smaller* disparity — so the
        revealed area gets background, never a smear of the foreground.

        Returns:
            ``(view, fill_ratio)`` where ``fill_ratio`` is the fraction of
            output pixels that were holes.
        """
        H, W = frame.shape[:2]
        src3 = frame.reshape(H, W, -1)
        C = src3.shape[2]

        cols = np.arange(W, dtype=np.float32)[None, :]
        dest = np.rint(cols - sign * disparity).astype(np.int64)
        inside = (dest >= 0) & (dest < W)

        row_base = (np.arange(H, dtype=np.int64) * W)[:, None]
        flat_dest = row_base + np.clip(dest, 0, W - 1)

        # z-buffer — larger disparity means nearer, so the nearest source wins.
        zbuf = np.full(H * W, -np.inf, dtype=np.float32)
        np.maximum.at(zbuf, flat_dest[inside], disparity[inside])

        # A source pixel draws iff it owns its destination's z-buffer value.
        wins = inside & (disparity >= zbuf[flat_dest])
        src_idx = np.flatnonzero(wins)
        dst_idx = flat_dest.reshape(-1)[src_idx]

        out = np.zeros((H * W, C), dtype=frame.dtype)
        out[dst_idx] = src3.reshape(H * W, C)[src_idx]
        drawn = np.zeros(H * W, dtype=bool)
        drawn[dst_idx] = True

        drawn2 = drawn.reshape(H, W)
        holes = ~drawn2
        fill_ratio = float(holes.sum()) / float(H * W)
        if not fill_ratio:
            return out.reshape(frame.shape), 0.0

        out3 = out.reshape(H, W, C)
        zb = zbuf.reshape(H, W)
        idx = np.broadcast_to(np.arange(W, dtype=np.int64), (H, W))

        # Nearest drawn column at or to the left / right of every pixel.
        left_i = np.maximum.accumulate(np.where(drawn2, idx, -1), axis=1)
        right_i = np.minimum.accumulate(np.where(drawn2, idx, W)[:, ::-1], axis=1)[:, ::-1]
        has_left, has_right = left_i >= 0, right_i < W
        lc, rc = np.clip(left_i, 0, W - 1), np.clip(right_i, 0, W - 1)

        rows = np.broadcast_to(np.arange(H, dtype=np.int64)[:, None], (H, W))
        # The background side is the neighbour with the smaller disparity.
        take_left = has_left & (~has_right | (zb[rows, lc] <= zb[rows, rc]))
        pick = np.where(take_left, lc, rc)

        usable = holes & (has_left | has_right)
        out3[usable] = out3[rows[usable], pick[usable]]

        # A row nothing was drawn into keeps its un-warped source pixels.
        dead = holes & ~(has_left | has_right)
        if dead.any():
            out3[dead] = src3[dead]

        return out3.reshape(frame.shape), fill_ratio

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
