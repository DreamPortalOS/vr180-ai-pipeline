#!/usr/bin/env python3
"""AI Temporal Outpainter for VR180 Boundary Regions (Phase 3 R&D).

Fills missing VR180 equirectangular boundary regions using optical flow-based
temporal consistency. When a flat 2D video is projected onto a VR180 sphere,
the extreme left/right edges and poles lack content. This module iteratively
outpaints those regions by propagating information from adjacent frames using
dense optical flow (Farneback).

Key features:
- Frame boundary detection for missing VR180 regions
- Optical flow-based temporal consistency engine
- Iterative outpainting with convergence detection
- Quality metrics (SSIM, PSNR) for validation

Usage:
    from pipeline.research.temporal_outpainter import TemporalOutpainter
    outpainter = TemporalOutpainter()
    result_frames = outpainter.outpaint(frames, mask)
"""

import logging
import math
from dataclasses import dataclass

import cv2
import numpy as np

log = logging.getLogger("temporal-outpainter")


@dataclass
class OutpaintQualityMetrics:
    """Quality metrics for outpainting validation."""
    ssim: float
    psnr: float
    coverage_pct: float  # percentage of mask filled
    iterations_used: int
    converged: bool


class TemporalOutpainter:
    """Optical flow-based temporal outpainter for VR180 boundary regions.
    
    Algorithm:
    1. Detect boundary regions (poles, extreme edges) that need outpainting
    2. For each frame, compute dense optical flow (Farneback) to adjacent frames
    3. Warp adjacent frames' boundary content using the flow field
    4. Blend warped contributions using a distance-weighted average
    5. Iterate until convergence (pixel change < threshold)
    
    Args:
        max_iterations: Maximum outpainting iterations per frame
        convergence_threshold: Pixel change threshold for convergence (0-255)
        flow_pyramid_scale: Optical flow pyramid scale (smaller = faster, less accurate)
        pole_angle_deg: Angular extent of pole regions to outpaint
        edge_margin_pct: Percentage of horizontal FOV to treat as edge region
        temporal_window: Number of adjacent frames to sample from (each side)
    """

    def __init__(
        self,
        max_iterations: int = 5,
        convergence_threshold: float = 1.5,
        flow_pyramid_scale: float = 0.5,
        pole_angle_deg: float = 30.0,
        edge_margin_pct: float = 0.05,
        temporal_window: int = 3,
    ):
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.flow_pyramid_scale = flow_pyramid_scale
        self.pole_angle_deg = pole_angle_deg
        self.edge_margin_pct = edge_margin_pct
        self.temporal_window = temporal_window

    def detect_boundary_mask(self, frame: np.ndarray) -> np.ndarray:
        """Generate a binary mask of VR180 boundary regions that need outpainting.
        
        In equirectangular projection, the poles (top/bottom) and extreme
        left/right edges lack content when projecting from a flat 2D source.
        
        Args:
            frame: RGB equirectangular frame (H, W, 3)
            
        Returns:
            Binary mask (H, W) where 255 = needs outpainting, 0 = original content
        """
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        # Pole regions: top and bottom bands
        # In equirect, latitude ranges from -90° (bottom) to +90° (top)
        # pole_angle_deg maps to a fraction of the vertical extent
        pole_frac = self.pole_angle_deg / 180.0
        pole_h = int(h * pole_frac / 2)  # each pole

        # Top pole
        mask[:pole_h, :] = 255
        # Bottom pole
        mask[h - pole_h:, :] = 255

        # Edge regions: extreme left/right margins
        edge_w = int(w * self.edge_margin_pct)
        if edge_w > 0:
            mask[:, :edge_w] = 255
            mask[:, w - edge_w:] = 255

        return mask

    def compute_optical_flow(
        self, frame1: np.ndarray, frame2: np.ndarray
    ) -> np.ndarray:
        """Compute dense optical flow between two grayscale frames.
        
        Uses Farneback algorithm for dense flow estimation.
        
        Args:
            frame1: First frame (RGB or grayscale)
            frame2: Second frame (RGB or grayscale)
            
        Returns:
            Flow array (H, W, 2) with dx, dy per pixel
        """
        if len(frame1.shape) == 3:
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        else:
            gray1 = frame1
        if len(frame2.shape) == 3:
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
        else:
            gray2 = frame2

        # Downscale for speed if configured
        if self.flow_pyramid_scale < 1.0:
            h, w = gray1.shape[:2]
            new_w = int(w * self.flow_pyramid_scale)
            new_h = int(h * self.flow_pyramid_scale)
            gray1 = cv2.resize(gray1, (new_w, new_h))
            gray2 = cv2.resize(gray2, (new_w, new_h))

        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

        # Upscale flow back to original resolution if downscaled
        if self.flow_pyramid_scale < 1.0:
            h, w = frame1.shape[:2] if len(frame1.shape) == 3 else frame1.shape
            flow = cv2.resize(flow, (w, h))
            # Scale flow vectors proportionally
            flow[:, :, 0] /= self.flow_pyramid_scale
            flow[:, :, 1] /= self.flow_pyramid_scale

        return flow

    def warp_frame_by_flow(
        self, frame: np.ndarray, flow: np.ndarray
    ) -> np.ndarray:
        """Warp a frame using optical flow field.
        
        Args:
            frame: RGB frame (H, W, 3)
            flow: Optical flow (H, W, 2) with dx, dy
            
        Returns:
            Warped frame
        """
        h, w = frame.shape[:2]
        # Create coordinate grids
        x_coords = np.arange(w, dtype=np.float32)
        y_coords = np.arange(h, dtype=np.float32)
        x_grid, y_grid = np.meshgrid(x_coords, y_coords)

        # Apply flow: new position = current + flow
        map_x = x_grid + flow[:, :, 0]
        map_y = y_grid + flow[:, :, 1]

        warped = cv2.remap(
            frame, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        return warped

    def _compute_distance_weights(
        self, h: int, w: int, mask: np.ndarray
    ) -> np.ndarray:
        """Compute distance-based blending weights from mask boundary.
        
        Pixels closer to the original content get higher weight when blending
        outpainted contributions.
        
        Args:
            h: Frame height
            w: Frame width
            mask: Binary boundary mask
            
        Returns:
            Weight map (H, W) float32, range [0, 1]
        """
        # Distance transform from mask boundary (inward)
        inv_mask = 255 - mask
        dist = cv2.distanceTransform(inv_mask, cv2.DIST_L2, 5)

        # Normalize to [0, 1]
        dmax = dist.max()
        if dmax > 0:
            weights = dist / dmax
        else:
            weights = np.ones((h, w), dtype=np.float32)

        # Invert: pixels near boundary get high weight
        weights = 1.0 - weights
        return weights.astype(np.float32)

    def outpaint_frame(
        self,
        target_frame: np.ndarray,
        context_frames: list[np.ndarray],
        mask: np.ndarray,
    ) -> tuple[np.ndarray, OutpaintQualityMetrics]:
        """Outpaint boundary regions of a single frame using temporal context.
        
        Args:
            target_frame: The frame to outpaint (RGB)
            context_frames: List of adjacent frames for temporal propagation
            mask: Binary mask of regions to fill
            
        Returns:
            Tuple of (outpainted_frame, quality_metrics)
        """
        h, w = target_frame.shape[:2]
        result = target_frame.copy()
        mask_bool = mask > 0

        if not context_frames or not np.any(mask_bool):
            metrics = OutpaintQualityMetrics(
                ssim=1.0, psnr=float('inf'),
                coverage_pct=0.0, iterations_used=0, converged=True,
            )
            return result, metrics

        # Pre-compute flow fields from each context frame to target
        flow_fields = []
        for ctx in context_frames:
            flow = self.compute_optical_flow(ctx, target_frame)
            flow_fields.append(flow)

        # Distance weights for blending
        blend_weights = self._compute_distance_weights(h, w, mask)

        # Iterative outpainting
        prev_result = result.copy()
        converged = False
        iterations_used = 0

        for iteration in range(self.max_iterations):
            # Accumulate warped contributions
            accumulator = np.zeros((h, w, 3), dtype=np.float64)
            weight_sum = np.zeros((h, w), dtype=np.float64)

            for ctx, flow in zip(context_frames, flow_fields):
                # Warp context frame's content into target's coordinate space
                warped = self.warp_frame_by_flow(ctx, flow)

                # Only use the contribution in masked regions
                contribution_weight = blend_weights.copy()
                contribution_weight[~mask_bool] = 0

                for c in range(3):
                    accumulator[:, :, c] += warped[:, :, c].astype(np.float64) * contribution_weight
                weight_sum += contribution_weight

            # Normalize
            weight_sum[weight_sum == 0] = 1.0
            filled = np.zeros_like(result)
            for c in range(3):
                filled[:, :, c] = (accumulator[:, :, c] / weight_sum).astype(np.uint8)

            # Apply only to masked regions
            result[mask_bool] = filled[mask_bool]

            # Check convergence
            diff = np.abs(
                result[mask_bool].astype(np.float64) -
                prev_result[mask_bool].astype(np.float64)
            )
            mean_change = diff.mean() if diff.size > 0 else 0.0
            iterations_used = iteration + 1

            if mean_change < self.convergence_threshold:
                converged = True
                log.info(
                    f"  Converged at iteration {iterations_used} "
                    f"(mean change: {mean_change:.2f})"
                )
                break

            prev_result = result.copy()

            # Update flow fields with refined result for next iteration
            if iteration < self.max_iterations - 1:
                flow_fields = []
                for ctx in context_frames:
                    flow = self.compute_optical_flow(ctx, result)
                    flow_fields.append(flow)

        # Compute quality metrics
        coverage = float(np.sum(mask_bool)) / (h * w) * 100
        psnr_val = self._compute_psnr(target_frame, result, mask_bool)
        ssim_val = self._compute_ssim(target_frame, result, mask_bool)

        metrics = OutpaintQualityMetrics(
            ssim=ssim_val,
            psnr=psnr_val,
            coverage_pct=coverage,
            iterations_used=iterations_used,
            converged=converged,
        )

        return result, metrics

    def outpaint(
        self,
        frames: list[np.ndarray],
        mask: np.ndarray = None,
    ) -> tuple[list[np.ndarray], list[OutpaintQualityMetrics]]:
        """Outpaint boundary regions for an entire sequence of frames.
        
        Args:
            frames: List of RGB equirectangular frames
            mask: Optional pre-computed boundary mask. If None, auto-detected.
            
        Returns:
            Tuple of (outpainted_frames, per_frame_metrics)
        """
        if not frames:
            return [], []

        if mask is None:
            mask = self.detect_boundary_mask(frames[0])
            log.info(
                f"Auto-detected boundary mask: "
                f"{np.sum(mask > 0) / mask.size * 100:.1f}% of frame"
            )

        outpainted = []
        all_metrics = []
        n = len(frames)

        for i, frame in enumerate(frames):
            # Collect temporal context frames
            context_indices = []
            for offset in range(-self.temporal_window, self.temporal_window + 1):
                j = i + offset
                if 0 <= j < n and j != i:
                    context_indices.append(j)

            context_frames = [frames[j] for j in context_indices]

            log.info(
                f"Outpainting frame {i}/{n} "
                f"(context: {len(context_frames)} frames)"
            )

            result, metrics = self.outpaint_frame(frame, context_frames, mask)
            outpainted.append(result)
            all_metrics.append(metrics)

        # Summary
        avg_ssim = np.mean([m.ssim for m in all_metrics])
        avg_psnr = np.mean([m.psnr for m in all_metrics if m.psnr != float('inf')])
        converged_count = sum(1 for m in all_metrics if m.converged)
        log.info(
            f"Outpainting complete: {converged_count}/{n} converged, "
            f"avg SSIM={avg_ssim:.4f}, avg PSNR={avg_psnr:.2f}dB"
        )

        return outpainted, all_metrics

    def _compute_psnr(
        self, original: np.ndarray, result: np.ndarray, mask: np.ndarray
    ) -> float:
        """Compute PSNR between original and result in non-masked regions.
        
        Args:
            original: Original frame
            result: Outpainted frame
            mask: Boolean mask (True = outpainted region)
            
        Returns:
            PSNR in dB
        """
        # Only compare in non-outpainted regions (original content preserved)
        inv_mask = ~mask
        if not np.any(inv_mask):
            return float('inf')

        diff = (
            original[inv_mask].astype(np.float64) -
            result[inv_mask].astype(np.float64)
        )
        mse = np.mean(diff ** 2)
        if mse == 0:
            return float('inf')
        return 10 * math.log10(255.0 ** 2 / mse)

    def _compute_ssim(
        self, original: np.ndarray, result: np.ndarray, mask: np.ndarray
    ) -> float:
        """Compute a simplified SSIM between original and result in non-masked regions.
        
        Uses block-based SSIM computation on the non-outpainted content
        to verify the outpainting didn't corrupt original pixels.
        
        Args:
            original: Original frame (RGB)
            result: Outpainted frame (RGB)
            mask: Boolean mask
            
        Returns:
            SSIM value (0 to 1)
        """
        # Compare only non-masked regions using grayscale
        gray_orig = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY).astype(np.float64)
        gray_result = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY).astype(np.float64)

        inv_mask = ~mask
        if not np.any(inv_mask):
            return 1.0

        mu_o = gray_orig[inv_mask].mean()
        mu_r = gray_result[inv_mask].mean()
        sigma_o_sq = gray_orig[inv_mask].var()
        sigma_r_sq = gray_result[inv_mask].var()
        sigma_or = np.mean(
            (gray_orig[inv_mask] - mu_o) * (gray_result[inv_mask] - mu_r)
        )

        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2

        numerator = (2 * mu_o * mu_r + c1) * (2 * sigma_or + c2)
        denominator = (mu_o ** 2 + mu_r ** 2 + c1) * (sigma_o_sq + sigma_r_sq + c2)

        return float(numerator / denominator) if denominator != 0 else 1.0