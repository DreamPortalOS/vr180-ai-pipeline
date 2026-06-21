# Stage 2: Stereo Disparity Rendering

## Goal

Convert a 2D frame (with its depth map) into a stereoscopic pair:
one image for the left eye, one for the right eye. The depth drives a
horizontal parallax shift that simulates human binocular vision.

## How It Works

### Parallax Shift Formula

For each pixel at `(x, y)` with metric depth `d`:

```
shift_px = (IPD × focal_length) / d

Left eye:   x_L = x + shift_px / 2
Right eye:  x_R = x - shift_px / 2
```

Where:
- **IPD** (Interpupillary Distance): 0.064 m (average human)
- **focal_length**: computed from image width and assumed ~70° HFOV
- **d**: metric depth from Stage 1

### Depth → Disparity Mapping

```
Near objects (d ≈ 0.3 m)  → large shift (~40 px)  → strong 3D
Mid distance  (d ≈ 5 m)   → medium shift (~2 px)  → natural 3D
Far objects  (d ≈ 50 m+)  → minimal shift (~0 px) → depth plane
```

### Disocclusion Inpainting

Shifting creates holes at disoccluded edges (background revealed from
behind a foreground object). Three strategies:

| Method | Quality | Speed | Description |
|--------|---------|-------|-------------|
| **Edge** (default) | Good | Fast | Navier-Stokes fluid dynamics inpainting (OpenCV TELEA) |
| **Depth-fill** | Better | Medium | Extend nearest pixel at similar depth |
| **Flow-guided** | Best | Slow | Bidirectional optical flow from neighbouring frames |

## Zero Parallax Plane

The **zero parallax plane** (ZPP) is the depth at which pixels have
zero horizontal shift. Objects at this depth appear at screen depth
(neither in front nor behind the display). By default, the ZPP is at
infinity (distant objects appear at screen depth). This can be adjusted.

## Key Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| IPD | 0.064 m | Stereo strength — larger = stronger 3D, risk of eye strain |
| Max disparity | 5% width | Clamp to prevent excessive divergence |
| Temporal smooth | On | Prevent flicker from frame-to-frame depth variation |

## Output

```
Left/Right Views:
  Shape:  (H, W, 3) — same resolution as input
  Dtype:  uint8 [0, 255] RGB
  Format: PNG image files (left_*.png, right_*.png)
```

## Visual Quality Guidelines

- **Avoid hyper-stereo**: excessive IPD/IPD-like settings cause
  miniaturisation (the "Gulliver effect")
- **Foreground objects should pop**, mid-ground should be natural,
  background should recede gently
- **Horizontal-only disparity**: no vertical misalignment (causes
  headaches in VR)
