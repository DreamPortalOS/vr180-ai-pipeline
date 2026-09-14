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
contour, never from the foreground.

.. _disparity-sign:

Which way is "near"?
--------------------
Both depth backends in this repo (Depth-Anything V2 through
:class:`~pipeline.depth_estimator.DepthEstimator`, DepthCrafter through
:mod:`pipeline.depth_crafter`) emit *inverse* depth normalised to ``[0, 1]``:
**larger value = nearer**.  :meth:`StereoRenderer._compute_disparity` then
computes ``(convergence - depth) * …``, so in the field it returns

    **smaller (more negative) disparity = nearer.**

That is the geometrically correct signed disparity: a near surface comes out
*crossed* (it sits further right in the left eye than in the right), which is
what puts it in front of the screen plane.  ``nearness = -disparity`` is
therefore the depth order used by the z-buffer, by the background-side fill,
and by the gradient limiter — pinned by
``test_near_object_renders_with_crossed_disparity``.
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
        edge_align_radius: float = 0.005,  # G-8 (#355): guided-filter radius / short side — see _align_disparity_edges
        gradient_limit: float | None = 0.9,  # G-8 (#355): max |d disparity / dx| per eye (None = off)
    ):
        self.ipd = ipd
        self.focal_length_px = focal_length_px
        self.max_disparity = max_disparity
        self.temporal_smooth = temporal_smooth
        self.convergence = convergence
        self.src_hfov = src_hfov
        self.occlusion_aware = occlusion_aware
        self.edge_align = edge_align
        self.edge_align_radius = edge_align_radius
        self.gradient_limit = gradient_limit
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

        # G-8 (#355): bound the horizontal disparity gradient.  A depth cliff
        # of Δ px is Δ px of content that exists in neither eye; bounding the
        # gradient turns that tear into a Δ/g-wide ramp of *real, stretched*
        # pixels, which both eyes still share.  Applied before the temporal EMA
        # so the smoothed field stays gradient-limited too (a convex
        # combination of g-Lipschitz fields is g-Lipschitz).
        if self.gradient_limit is not None:
            disparity = self._limit_disparity_gradient(disparity, self.gradient_limit)

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
        """Convert depth to signed pixel disparity.

        *depth* is the backends' normalised **inverse** depth (larger = nearer),
        so the returned field is *smaller = nearer* — see :ref:`disparity-sign`.
        Objects nearer than the convergence plane come out negative (crossed,
        in front of the screen), objects beyond it positive.
        """
        # Normalize depth to a meaningful range
        d_min, d_max = depth.min(), depth.max()
        depth_norm = (depth - d_min) / (d_max - d_min) if d_max > d_min else np.zeros_like(depth)

        # Convergence plane: objects at convergence depth have zero disparity
        # Objects closer than convergence pop out (positive disparity)
        # Objects farther recede (negative disparity)
        d_conv = self.convergence
        depth_rel = d_conv - depth_norm  # negative = nearer than convergence

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

        **Radius.**  ``edge_align_radius`` is a fraction of the short side, and
        it is a two-sided cost, so the default is measured rather than guessed:

        * A radius of ``r`` can only drag a depth edge that is misaligned by up
          to roughly ``r / 2`` px (pinned by
          ``test_edge_align_radius_bounds_how_far_an_edge_can_be_dragged``), so
          too small a radius simply does nothing.
        * Everywhere the guide is *flat* the filter is a plain local mean, so
          too large a radius flattens the far field's real structure.  Measured
          on the drone clip's frame 120 (``gen_1x1_4k_drone.mp4``, DepthCrafter
          depth, 2880² equirect per eye), raising this from 0.005 to 0.02 costs
          **+143%** inter-eye block-match error in the far field (2.6 → 7.8 px
          MAE) — it trades away real background 3D to make the contour ratio
          look better, which is exactly what #355 rules out.

        0.005 (14 px at 2880) is the largest radius that left the far field
        within the card's 10% budget.
        """
        import cv2

        H, W = disparity.shape[:2]
        radius = max(2, round(min(H, W) * self.edge_align_radius))
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

    @staticmethod
    def _limit_disparity_gradient(disparity: np.ndarray, limit: float) -> np.ndarray:
        """Bound ``|∂disparity/∂x|`` by *limit* (G-8, #355).

        Returns the **largest** *limit*-Lipschitz field that is everywhere ≤
        the input: the min-plus erosion by the cone ``k ↦ limit·|k|``.
        Equivalently, it dilates ``nearness = -disparity`` (see
        :ref:`disparity-sign`).  Three properties make this the right tool:

        * Where the field already satisfies the bound the output is bit-exact
          with the input.  This is not a blur — undisturbed geometry, and in
          particular the entire far field, is untouched.
        * It only ever pulls *towards* the viewer, so a foreground object keeps
          its full pop-out.  What changes is a ``Δ/limit``-wide band of
          background beside each contour, ramped in instead of stepped.
        * The result is still bounded by the input's own range, so it can never
          push disparity past the ``max_disparity`` comfort clip.

        Why bound the gradient at all: the per-eye mapping is
        ``x ↦ x ∓ disparity(x)``, whose local scale factor is ``1 ∓ d'(x)`` —
        *with opposite signs in the two eyes*.  Where ``|d'| > 1`` the mapping
        stops being monotone and a band of width ``≈Δ`` exists in neither eye;
        no amount of inpainting makes the eyes agree about invented pixels.
        That band is what an opaque body being locally see-through looks like.

        The default ``limit`` (0.9) is therefore chosen just under 1: every
        destination column stays within one pixel of some source pixel, so the
        sub-pixel splat in :meth:`_warp_eye` covers the frame with real,
        stretched content and there is nothing left to invent.  Pushing it
        lower also shrinks the inter-eye width ratio ``(1+d')/(1-d')``, but at
        the cost of a proportionally wider ramp, so it is left as a knob rather
        than made the default.

        The cost is honest and local: background within ``Δ/limit`` px of a
        contour sits at an intermediate depth rather than its true one.

        Args:
            disparity: Per-eye disparity field, ``(H, W)`` float.
            limit: Max change in disparity per pixel of x.  ``≤ 0`` is a no-op
                (an unbounded-gradient request).

        Returns:
            The gradient-limited field, float32.
        """
        out = -disparity.astype(np.float32)  # work on nearness; negated back below
        if limit <= 0:
            return disparity.astype(np.float32, copy=True)

        W = out.shape[1]
        # A cone of radius R is reachable by composing cones of radius
        # 1, 2, 4, …  (max-plus: cone_a ∘ cone_b = cone_{a+b}), so the whole
        # envelope costs O(log W) full-array passes instead of O(W) column
        # steps.  R only has to span the largest cliff the field can hold.
        span = float(out.max() - out.min())
        reach = min(W - 1, int(np.ceil(span / limit)) if span > 0 else 0)
        step = 1
        while step <= reach:
            drop = limit * step
            shifted_left = np.empty_like(out)  # neighbour `step` px to the right
            shifted_left[:, : W - step] = out[:, step:] - drop
            shifted_left[:, W - step :] = -np.inf
            shifted_right = np.empty_like(out)  # neighbour `step` px to the left
            shifted_right[:, step:] = out[:, : W - step] - drop
            shifted_right[:, :step] = -np.inf
            out = np.maximum(out, np.maximum(shifted_left, shifted_right))
            step *= 2
        return -out

    def _warp_eye(self, frame: np.ndarray, disparity: np.ndarray, sign: int) -> tuple[np.ndarray, float]:
        """Forward-warp (splat) one eye with a z-buffer, then fill from the background.

        ``sign=+1`` renders the left eye, ``sign=-1`` the right, matching the
        legacy backward-remap convention (``left_x = x + disparity``): a source
        pixel at ``x`` lands at ``x - sign * disparity``.

        Unlike a backward remap, a forward splat actually *moves* the
        silhouette, so the band around a depth discontinuity becomes an
        explicit hole instead of silently carrying foreground colour into the
        other eye.  Each hole is then filled from the **background side** —
        the horizontal neighbour that is *farther*, i.e. the one with the
        *larger* disparity (see :ref:`disparity-sign`) — so the revealed area
        gets background, never a smear of the foreground.

        The sign matters twice, and both ways round it is exactly the artefact
        this card exists to remove: an inverted z-buffer lets the background
        paint over the object (holes *in* an opaque body), and an inverted fill
        paints the foreground colour into the revealed band (the halo).

        Returns:
            ``(view, fill_ratio)`` where ``fill_ratio`` is the fraction of
            output pixels that were holes.
        """
        H, W = frame.shape[:2]
        src3 = frame.reshape(H, W, -1)
        C = src3.shape[2]
        n = H * W

        cols = np.arange(W, dtype=np.float32)[None, :]
        xs = np.broadcast_to(np.arange(W, dtype=np.int64), (H, W))
        row_base = (np.arange(H, dtype=np.int64) * W)[:, None]

        # Sub-pixel splat: a source pixel lands at a fractional destination and
        # is shared between the two columns straddling it (a tent kernel).
        # Rounding to the nearest column instead — the obvious implementation —
        # costs up to half a pixel of shift *per eye*, in opposite directions,
        # which is a stereo error the viewer sees on fine texture.
        dest_f = cols - sign * disparity
        base = np.floor(dest_f).astype(np.int64)
        frac = (dest_f - base).astype(np.float32)

        taps = []
        for k, weight in ((0, 1.0 - frac), (1, frac)):
            col = base + k
            live = (col >= 0) & (col < W) & (weight > 0)
            taps.append((row_base + np.clip(col, 0, W - 1), live, weight))

        # z-buffer on nearness = -disparity: the *smallest* disparity is the
        # nearest surface, so it is the one that survives a collision.
        zbuf = np.full(n, np.inf, dtype=np.float32)
        for flat, live, _ in taps:
            np.minimum.at(zbuf, flat[live], disparity[live])

        # Which source column owns each destination.  Ties are pixels of the
        # same surface, so either answer is right.
        winner = np.full(n, -(W + 2), dtype=np.int64)
        for flat, live, _ in taps:
            owns = live & (disparity <= zbuf[flat])
            winner[flat[owns]] = xs[owns]

        # Accumulate the tent, but only from the surface that won: neighbours
        # within one column of the winner are the same surface, anything
        # further away is the occluded one and must not bleed through.
        acc = np.zeros((n, C), dtype=np.float32)
        wsum = np.zeros(n, dtype=np.float32)
        for flat, live, weight in taps:
            sel = live & (np.abs(xs - winner[flat]) <= 1)
            f, w = flat[sel], weight[sel]
            wsum += np.bincount(f, w, minlength=n)
            for c in range(C):
                acc[:, c] += np.bincount(f, w * src3[:, :, c][sel], minlength=n)

        drawn = wsum > 0
        out = np.zeros((n, C), dtype=frame.dtype)
        np.divide(acc, wsum[:, None], out=acc, where=drawn[:, None])
        values = acc[drawn]
        if np.issubdtype(frame.dtype, np.integer):
            lo, hi = np.iinfo(frame.dtype).min, np.iinfo(frame.dtype).max
            values = np.rint(values).clip(lo, hi)
        out[drawn] = values.astype(frame.dtype)

        drawn2 = drawn.reshape(H, W)
        holes = ~drawn2
        fill_ratio = float(holes.sum()) / float(n)
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
        # The background side is the *farther* neighbour — the larger disparity.
        take_left = has_left & (~has_right | (zb[rows, lc] >= zb[rows, rc]))
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
