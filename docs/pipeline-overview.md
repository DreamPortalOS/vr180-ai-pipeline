# Pipeline Overview

## What This Pipeline Does

Converts AI-generated 2D videos (typically 720p, 16:9, 24-60 fps) into
**VR180 format** — a stereoscopic 180° equirectangular video playable
on Meta Quest, Apple Vision Pro, and other VR headsets.

## Why VR180?

VR180 (half-sphere, front-facing) is the preferred format for:

- **Cinematic VR** — immersive without requiring 360° head turning
- **AI video** — most generative models produce forward-facing content
- **Comfort** — reduces neck strain vs. 360° video
- **Resolution efficiency** — doubles effective pixel density vs. 360°

## Pipeline at a Glance

```
                     ┌─ Spherical ─┐
                     │  Coordinate │
                     │   Mapping   │
                     └──────┬──────┘
                            │
┌──────────┐  ┌────────┐  ┌─▼───────┐  ┌────────────┐  ┌─────────────┐
│ 2D Video │─►│ Depth  │─►│ Stereo  │─►│ Equirect  │─►│ VR Metadata │─► VR180!
│ (720p)   │  │  Map   │  │ Pair    │  │  180°     │  │   Embed     │
└──────────┘  └────────┘  └─────────┘  └───────────┘  └─────────────┘
```

## When to Use This Pipeline

| Scenario | Fit |
|----------|-----|
| AI-generated music videos | 🟢 Great — controlled camera works well |
| Text-to-video clips | 🟢 Great — add depth to flat renders |
| Real-world 2D footage | 🟡 Good — depth estimation quality varies |
| Fast-moving sports | 🟡 Moderate — motion blur challenges depth |
| 360° / stereo native footage | 🔴 Wrong tool — already immersive |

## Output Specifications

| Property          | Value                        |
|-------------------|------------------------------|
| Resolution        | 7680 × 1920 (SBS)            |
| Aspect Ratio      | 4:1 (2× 3840×1920)           |
| Projection        | Equirectangular (180°)       |
| Stereo Format     | Side-by-side                 |
| Frame Rate        | 60 fps (configurable)        |
| Codec             | H.264 / H.265                |
| Container         | MP4 with SphericalVideo V2   |
| VR Headset Support| Quest, Vision Pro, PCVR      |
