# Architecture Overview

## System Design

The VR180 AI Pipeline is a modular, stage-based system for converting
2D AI-generated videos into VR180 immersive format. Each stage is
independently runnable, allowing iterative development and debugging.

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Input: 2D Video                             │
│                     720p, 16:9, 24-60 fps                           │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Stage 1:  DepthEstimator                                           │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │  Depth Anything V2 / MiDaS 3.1                                  ││
│  │   ┌──────────┐  ┌──────────┐  ┌─────────────┐                  ││
│  │   │ ViT/DINO │→│ DPT Head │→│ Calibration │ → Depth Map (H,W) ││
│  │   │ Encoder  │  │ Fusion   │  │ (metric)    │                  ││
│  │   └──────────┘  └──────────┘  └─────────────┘                  ││
│  └─────────────────────────────────────────────────────────────────┘│
│  Output: Depth maps (H, W) float32                                  │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Stage 2:  StereoRenderer                                           │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │  Stereo Disparity                                                ││
│  │                                                                    │
│  │  Frame + Depth ──► shift = B·f / d ──► Remap ──► Left + Right  ││
│  │                                          │                       ││
│  │                                     Inpaint Holes                ││
│  │                                      (edge/flow/depth)           ││
│  └─────────────────────────────────────────────────────────────────┘│
│  Output: Left / Right eye views (pair)                               │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Stage 3:  EquirectangularMapper                                    │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │  Spherical Projection                                            ││
│  │                                                                    │
│  │  Left/Right ──► θ,φ → UV ──► Mesh Warp ──► 3840×1920           ││
│  │                                    (OpenCV remap)                ││
│  └─────────────────────────────────────────────────────────────────┘│
│  Output: SBS equirectangular frames (7680×1920)                      │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Stage 4:  VRMetadataEmbedder                                       │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │  MP4 Encoding + VR Tags                                          ││
│  │                                                                    │
│  │  ┌────────────┐  ┌─────────────────┐  ┌──────────────────────┐  ││
│  │  │ H.264/H.265│  │ SphericalVideo  │  │ Camera Motion        │  ││
│  │  │ Encode     │  │ V2 XML          │  │ Metadata Track       │  ││
│  │  └────────────┘  └─────────────────┘  └──────────────────────┘  ││
│  └─────────────────────────────────────────────────────────────────┘│
│  Output: VR180 MP4 (playable on Quest, Apple Vision Pro)             │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
                   Output: VR180 Video
            Side-by-Side · Equirectangular · 60fps
```

## Data Flow

### Frame Pipeline

Each frame flows through the stages sequentially:

```
┌──────┐    ┌──────┐    ┌──────┐    ┌──────┐    ┌──────┐
│Frame │───►│Depth │───►│Stereo│───►│Equir │───►│Meta  │
│Reader│    │Est   │    │Render│    │Map   │    │Embed │
└──────┘    └──────┘    └──────┘    └──────┘    └──────┘
  720p       720p        720px2      3840x       3840x
  16:9       float32     RGB        1920        1920x2
                                                     SBS
```

### Parallel Processing Opportunities

| Stages | Parallelisation Strategy          |
|--------|-----------------------------------|
| Depth  | Batch inference on GPU            |
| Stereo | Per-frame independent             |
| Equir  | Pre-computed mesh, per-frame fast |
| Meta   | Encoder-level parallelism         |

## Component Interaction

### Module Dependencies

```
pipeline/
├── depth_estimator.py      ← torch, transformers
├── stereo_renderer.py      ← numpy, cv2
├── equirectangular_mapper.py  ← numpy, cv2
└── vr_metadata.py          ← ffmpeg subprocess
```

### Data Contracts

| Between               | Format                        | Shape          |
|-----------------------|-------------------------------|----------------|
| Reader → Depth        | np.ndarray uint8 RGB          | (H, W, 3)      |
| Depth → Stereo        | np.ndarray float32 depth      | (H, W)         |
| Stereo → Equirect     | np.ndarray uint8 RGB (×2)     | (H, W, 3) ×2   |
| Equirect → Metadata   | np.ndarray uint8 RGB (SBS)    | (H, 2W, 3)     |

## Memory Management

For 720p input → 3840×1920 VR180 output:

| Stage    | Memory per Frame   | Notes                        |
|----------|--------------------|------------------------------|
| Depth    | ~1.5 + 0.6 GB     | Model weights + frame        |
| Stereo   | ~30 MB             | All intermediate buffers     |
| Equirect | ~35 MB             | Mesh + frame                 |
| Metadata | ~100 MB            | Encoder buffer               |

A batch pipeline processing 30 frames at once would need ~4-6 GB VRAM.

## Error Handling Strategy

- Each stage validates its input format before processing
- Missing depth files in stereo stage raise clear FileNotFoundError
- FFmpeg errors in metadata stage surface the full stderr log
- CLI `--max-frames` for quick iteration on partial clips

## Extension Points

### Additional Depth Models
Extend `DepthEstimator._load_model()` to support new backends
(e.g., ZoeDepth, DPT-Large)

### Alternative Inpainting
Add new inpaint methods in `StereoRenderer.__init__` and implement
in `_shift_view`

### Custom Projections
Override `EquirectangularMapper._build_mesh()` for non-standard
projections (e.g., 360° dual-fisheye)
