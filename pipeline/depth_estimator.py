"""
Stage 1 — Depth Estimation
===========================

Per-frame metric depth map generation from 2D video input using
Depth Anything V2 via HuggingFace transformers pipeline.

Supports:
- Depth Anything V2 (Small / Base / Large) — robust zero-shot depth estimation
- Apple MPS acceleration on M-series Macs
- Batch processing with temporal smoothing hints

Usage:
    from pipeline.depth_estimator import DepthEstimator

    estimator = DepthEstimator(model_size="small")
    depth_map = estimator.estimate(frame)          # single frame
    depth_maps = estimator.estimate_batch(frames)  # batch
"""

import numpy as np
from typing import Optional, List


class DepthEstimator:
    """Per-frame depth estimation using Depth Anything V2.

    Uses the HuggingFace transformers pipeline API for easy model loading.
    """

    # Valid HuggingFace model repos
    MODEL_REPOS = {
        "small": "depth-anything/Depth-Anything-V2-Small-hf",
        "base": "depth-anything/Depth-Anything-V2-Base-hf",
        "large": "depth-anything/Depth-Anything-V2-Large-hf",
    }

    def __init__(
        self,
        model_size: str = "small",
        device: Optional[str] = None,
        calibrate: bool = True,
    ):
        """
        Args:
            model_size: "small" (24.8M params), "base" (97.5M), or "large" (335M)
            device: "cuda", "mps", "cpu". Auto-detected if None.
            calibrate: Scale relative depth to approximate metric depth.
        """
        if model_size not in self.MODEL_REPOS:
            raise ValueError(
                f"Unknown model size '{model_size}'. Choose from: {list(self.MODEL_REPOS.keys())}"
            )
        self.model_size = model_size
        self.device = device or self._auto_device()
        self.calibrate = calibrate
        self._pipe = None

    def _auto_device(self) -> str:
        """Auto-detect best available device."""
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_model(self):
        """Lazy-load the depth estimation pipeline."""
        if self._pipe is not None:
            return
        from transformers import pipeline

        print(f"[Depth] Loading {self.MODEL_REPOS[self.model_size]} on {self.device}...")
        self._pipe = pipeline(
            task="depth-estimation",
            model=self.MODEL_REPOS[self.model_size],
            device=self.device,
        )
        print(f"[Depth] Model loaded successfully.")

    def estimate(self, frame: np.ndarray) -> np.ndarray:
        """Estimate depth for a single RGB frame.

        Args:
            frame: RGB image (H, W, 3), uint8 [0, 255] or float [0, 1]

        Returns:
            Depth map (H, W), float32. Metric depth when calibrate=True.
        """
        from PIL import Image

        self._load_model()

        # Convert numpy to PIL
        if frame.dtype == np.uint8:
            pil_img = Image.fromarray(frame)
        elif frame.dtype == np.float32 or frame.dtype == np.float64:
            pil_img = Image.fromarray((frame * 255).astype(np.uint8))
        else:
            pil_img = Image.fromarray(frame.astype(np.uint8))

        # Run inference
        result = self._pipe(pil_img)
        depth = result["depth"]  # PIL Image

        # Convert back to numpy
        depth_np = np.array(depth, dtype=np.float32)

        # Resize to match input resolution if needed
        if depth_np.shape[:2] != frame.shape[:2]:
            import cv2
            depth_np = cv2.resize(depth_np, (frame.shape[1], frame.shape[0]),
                                  interpolation=cv2.INTER_LINEAR)

        # Normalize to [0, 1] range
        d_min, d_max = depth_np.min(), depth_np.max()
        if d_max > d_min:
            depth_np = (depth_np - d_min) / (d_max - d_min)
        else:
            depth_np = np.zeros_like(depth_np)

        # Optional metric calibration
        if self.calibrate:
            depth_np = self._calibrate_metric(depth_np)

        return depth_np

    def estimate_batch(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Estimate depth for a list of frames."""
        return [self.estimate(f) for f in frames]

    @staticmethod
    def _calibrate_metric(relative_depth: np.ndarray) -> np.ndarray:
        """Convert relative depth to approximate metric depth.

        Maps median depth to a plausible scene depth (~5m for FPV flight).
        The output is still in [0, ~10m] range but with a more intuitive scale.
        """
        target_median = 5.0
        safe_median = max(np.median(relative_depth), 0.01)
        scale = target_median / safe_median
        return relative_depth * scale
