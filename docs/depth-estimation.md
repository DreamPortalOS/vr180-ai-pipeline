# Stage 1: Depth Estimation

## Goal

Generate a dense, per-pixel depth map for every frame of the input video.
This serves as the geometric basis for stereoscopic view generation.

## Model Selection

### Depth Anything V2 (Recommended)

| Property | Value |
|----------|-------|
| Architecture | DINOv2 ViT-giant → DPT head |
| Resolution | Any (best at ~518px resize) |
| Precision | FP16/FP32 |
| Zero-shot | Excellent across domains |
| Detail | Fine-grained, preserves edges |
| Speed | ~2 fps on RTX 4090 (720p) |

**Depth Anything V2** uses a DINOv2 vision transformer backbone
(ViT-giant, 1.1B params) with a Dense Prediction Transformer (DPT)
head. It is trained on a large-scale dataset of ~62M images, giving
it strong zero-shot generalisation.

### MiDaS 3.1 (Lightweight Alternative)

| Property | Value |
|----------|-------|
| Architecture | ViT-L → DPT head |
| Resolution | 384px (recommended) |
| Params | ~345M |
| Speed | ~15 fps on RTX 4090 |
| Quality | Good, less fine detail |

MiDaS is faster and lighter, suitable for real-time preview or when
GPU memory is constrained.

## Algorithm

### 1. Preprocessing

Each frame is resized to the model's preferred input size while
preserving aspect ratio (padding to square for ViT):

```
Frame (720×1280) → Resize short side → Pad to square → Normalise
```

### 2. Forward Pass

The encoder extracts multi-scale patch features, which the DPT head
reassembles into a single-channel depth map:

```
Patch Embed ▸ Transformer Blocks (×40) ▸ Reassemble ▸ Fusion ▸ Inverse Depth
```

### 3. Depth Calibration

Relative depth (inverse depth, range [0, 1]) is converted to
approximate metric depth using a focal-length heuristic:

```
depth_metric = scale_factor / depth_relative
```

Where `scale_factor` is derived from the camera's focal length
(estimated from video metadata or user-provided).

## Output

```
Depth Map:
  Shape:  (H, W) — same resolution as input
  Dtype:  float32
  Range:  [0.1, ~50] meters (metric, calibrated)
  Format: NumPy .npy file
```

### Visualisation

For debugging, depth maps can be inverted and colour-mapped:

```python
import cv2
import numpy as np

depth = np.load("depth_000000.npy")
# Normalise to [0, 255] for visualisation
depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
depth_colored = cv2.applyColorMap(depth_norm.astype(np.uint8), cv2.COLORMAP_TURBO)
cv2.imwrite("depth_vis.png", depth_colored)
```

## Performance Optimisation

- **Batch inference**: process multiple frames per forward pass
- **FP16**: half-precision reduces VRAM and increases throughput
- **Temporal consistency**: output depth should be temporally smoothed
  (framewise processing may cause flicker)

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Depth is pure black | Model not loaded | Check model download |
| Depth is noisy/grainy | Low input quality | Pre-denoise input frames |
| Depth is flat (no variation) | Compression artefacts | Increase video bitrate |
| Slow processing | Large model on CPU | Use GPU, FP16, or MiDaS |
