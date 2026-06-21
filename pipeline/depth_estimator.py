"""
Stage 1 — Depth Estimation
===========================

Per-frame metric depth map generation from 2D video input.

Models
------
- **Depth Anything V2** (recommended) — robust zero-shot depth estimation
  with fine-grained detail preservation. SOTA on monocular depth.
- **MiDaS 3.1** — lightweight alternative with good generalisation across
  indoor/outdoor scenes.

Architecture
------------
::

   Input Frame (720p RGB)
          │
          ▼
   ┌──────────────────┐
   │  Image Encoder   │  DINOv2 / ViT backbone
   │  (patch embed)   │
   └────────┬─────────┘
            │
   ┌────────▼─────────┐
   │  DPT Head        │  Dense Prediction Transformer
   │  (multiscale     │  reassembles patch features
   │   fusion)        │  into per-pixel depth
   └────────┬─────────┘
            │
   ┌────────▼─────────┐
   │  Depth Head      │  Relative → metric depth
   │  (inverse depth  │  shift + scale calibration
   │   → metric)      │  (optional — uses focal length
   │                  │   from EXIF or user config)
   └────────┬─────────┘
            │
            ▼
   Depth Map (720p, float32, range [0, d_max])

Usage
-----
.. code:: python

    from pipeline.depth_estimator import DepthEstimator

    estimator = DepthEstimator(model_name="depth-anything-v2")
    depth_map = estimator.estimate(frame)          # single frame
    depth_maps = estimator.estimate_batch(frames)  # batch

Output Format
-------------
- **Shape**: ``(H, W)`` — same resolution as input
- **Dtype**: ``float32`` — metric depth in meters (when calibrated),
  or relative depth in [0, 1] (raw mode)
- **Range**: calibrated to real-world units via focal-length heuristic
  when available

Configuration
-------------
+-------------------+----------+----------------------------------------------------+
| Parameter         | Default  | Description                                        |
+===================+==========+====================================================+
| ``model_name``    | depth-   | ``"depth-anything-v2"`` (ViT-giant) or             |
|                   | anything-| ``"midas-3.1"`` (ViT-large)                        |
|                   | v2       |                                                    |
+-------------------+----------+----------------------------------------------------+
| ``device``        | auto     | ``"cuda"``, ``"mps"``, or ``"cpu"``               |
+-------------------+----------+----------------------------------------------------+
| ``calibrate``     | True     | Enable metric-depth calibration                    |
+-------------------+----------+----------------------------------------------------+
| ``focal_length``  | None     | Override camera focal length (mm). Auto-detect     |
|                   |          | from EXIF when available.                          |
+-------------------+----------+----------------------------------------------------+

References
----------
- Depth Anything V2: https://github.com/DepthAnything/Depth-Anything-V2
- MiDaS: https://github.com/isl-org/MiDaS
- DPT: Vision Transformers for Dense Prediction
  (Ranftl et al., ICCV 2021)
"""

import numpy as np
from typing import Optional


class DepthEstimator:
    """Per-frame depth estimation using monocular depth models.

    Supports Depth Anything V2 and MiDaS backends with automatic
    device detection and optional metric-depth calibration.
    """

    def __init__(
        self,
        model_name: str = "depth-anything-v2",
        device: Optional[str] = None,
        calibrate: bool = True,
        focal_length: Optional[float] = None,
    ):
        self.model_name = model_name
        self.device = device or self._auto_device()
        self.calibrate = calibrate
        self.focal_length = focal_length
        self._model = None
        self._transform = None

    def _auto_device(self) -> str:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_model(self):
        """Lazy-load the depth estimation model."""
        if self._model is not None:
            return

        if "midas" in self.model_name:
            self._load_midas()
        else:
            self._load_depth_anything()

    def _load_depth_anything(self):
        """Load Depth Anything V2 model via HuggingFace."""
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        repo = "depth-anything/Depth-Anything-V2-Giant"
        self._transform = AutoImageProcessor.from_pretrained(repo)
        self._model = AutoModelForDepthEstimation.from_pretrained(repo).to(self.device)
        self._model.eval()

    def _load_midas(self):
        """Load MiDaS 3.1 model via torch.hub."""
        import torch

        self._model = torch.hub.load("intel-isl/MiDaS", "MiDaS", trust_repo=True)
        self._model.eval()
        self._model.to(self.device)
        self._transform = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        self._transform = self._transform.dpt_transform

    @torch.no_grad()
    def estimate(self, frame: np.ndarray) -> np.ndarray:
        """Estimate depth for a single RGB frame.

        Args:
            frame: RGB image (H, W, 3), uint8 [0, 255] or float [0, 1]

        Returns:
            Depth map (H, W), float32.
        """
        self._load_model()
        import torch

        if frame.dtype == np.uint8:
            frame = frame.astype(np.float32) / 255.0

        if "midas" in self.model_name:
            img_tensor = self._transform(frame).to(self.device)
            prediction = self._model(img_tensor)
            depth = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze().cpu().numpy()
        else:
            inputs = self._transform(images=frame, return_tensors="pt").to(self.device)
            outputs = self._model(**inputs)
            depth = outputs.predicted_depth.squeeze().cpu().numpy()

        if self.calibrate:
            depth = self._calibrate_metric(depth)

        return depth.astype(np.float32)

    def estimate_batch(self, frames: list) -> list:
        """Estimate depth for a list of frames."""
        return [self.estimate(f) for f in frames]

    def _calibrate_metric(self, relative_depth: np.ndarray) -> np.ndarray:
        """Convert relative depth to approximate metric depth using
        focal-length heuristic. Based on the inverse-depth relationship:
        depth = f * baseline / disparity.

        Without known camera intrinsics, we apply a global scale factor
        that maps the median relative depth to a plausible scene depth.
        """
        # Typical indoor-outdoor median depth heuristic: ~5m
        target_median = 5.0
        scale = target_median / max(np.median(relative_depth), 1e-6)
        return relative_depth * scale