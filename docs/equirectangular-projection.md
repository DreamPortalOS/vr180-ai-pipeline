# Stage 3: Equirectangular Projection

## Goal

Map the planar stereo pair onto a 180° hemisphere using equirectangular
projection, producing a 3840 × 1920 frame per eye (7680 × 1920 SBS).

## Why Equirectangular?

VR headsets internally render equirectangular textures mapped to a
sphere. By pre-warping frames into this format, we eliminate the need
for real-time distortion in the headset. Google's VR180 spec mandates
equirectangular projection.

## Mapping Geometry

### Coordinate Systems

```
Spherical (θ, φ):
  θ  ∈ [-90°, +90°]  Horizontal (yaw), 0° = centre
  φ  ∈ [0°, 180°]     Vertical (pitch), 0° = top

Equirectangular (u, v):
  u  ∈ [0, W)         Horizontal pixel
  v  ∈ [0, H)         Vertical pixel

Conversion:
  u = W × (θ / 180° + 0.5)
  v = H × (1.0 - φ / 180°)
```

### Inverse Mapping (Output → Input)

For each output pixel `(u, v)`:

1. Convert to spherical angles `(θ, φ)`
2. Project ray onto a pinhole camera plane
3. Sample source pixel (bilinear/Lanczos)

The mesh is **pre-computed once** per video resolution, then applied
per-frame via OpenCV `remap()` for maximum throughput.

## Resolution Considerations

| Output | Pixels | Per-eye | Use Case |
|--------|--------|---------|----------|
| 3840×1920 | 7.4 MP | 3840×1920 | VR180 target (recommended) |
| 2880×1440 | 4.1 MP | 2880×1440 | Mobile/performance VR |
| 1920×960 | 1.8 MP | 1920×960 | Preview / testing |

The 2:1 aspect ratio (width = 2× height) is mandatory for
equirectangular projection.

## Field of View

- **HFOV = 180°**: Captures exactly the front hemisphere (VR180 standard)
- **VFOV = 180°**: Full vertical coverage (zenith to nadir)

With 180° horizontal FOV, pixels at the extreme left/right edges
correspond to θ = ±90° (directly to the sides).

## Quality Notes

- Lanczos interpolation (default) preserves sharpness at edges
- Source frame should have sufficient resolution to avoid blur
- Pole regions (top/bottom of equirect) are visually compressed
  — common to apply Gaussian blur there for artefact reduction
