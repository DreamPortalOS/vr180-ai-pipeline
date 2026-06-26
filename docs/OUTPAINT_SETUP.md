# Outpainter Deployment Guide (AI Mode)

**Status:** Placeholder — lead fills actual model paths and inference commands.

This document describes how to deploy an AI outpainting backend for the `--outpaint ai` mode in
`run_pipeline.py`.  The gradient mode (`--outpaint gradient`) requires no setup — it works with OpenCV only.

---

## Architecture

```
run_pipeline.py → Outpainter(mode="ai", ai_backend=SDInpaintBackend)
                                                │
                                          ┌─────┴──────┐
                                          │  diffusers   │  (Hugging Face pipeline)
                                          │  torch        │
                                          └─────┬──────┘
                                                │
                                       SDXL/Flux inpaint model
                                       (local checkpoint or HF hub)
```

The `SDInpaintBackend` class in `pipeline/outpainter.py` loads a
`StableDiffusionInpaintPipeline` from `diffusers` and runs per-frame outpainting
guided by the black-boundary mask.

---

## Prerequisites

- Python 3.10+
- `pip install diffusers torch transformers accelerate xformers`
- GPU with 8 GB+ VRAM (16 GB recommended for SDXL inpaint)

## Recommended model

| Model | Quality | VRAM | Notes |
|---|---|---|---|
| `stabilityai/stable-diffusion-2-inpainting` | Good | 6 GB | Default, works out of the box |
| `diffusers/stable-diffusion-xl-1.0-inpainting-0.1` | Excellent | 12 GB | Better sky/cloud fill |
| Custom fine-tune | Best | Varies | For specific VR180 environments |

---

## Usage

### Default model (auto-download from Hugging Face)

```bash
python scripts/run_pipeline.py --input video.mp4 --outpaint ai
```

This will download `stabilityai/stable-diffusion-2-inpainting` on first run (≈6 GB).

### Custom model path

```python
# In your wrapper script:
from pipeline.outpainter import SDInpaintBackend, Outpainter

backend = SDInpaintBackend(model_path="path/to/sdxl-inpaint", device="cuda")
outpainter = Outpainter(mode="ai", ai_backend=backend)
```

---

## TODO (lead)

- [ ] Test SDXL inpaint output quality on equirectangular VR180 frames.
- [ ] Decide whether to run per-frame or only on keyframes + interpolation.
- [ ] Optimize: crop mask region to reduce inference area (optional).
- [ ] Add Seedance / FLUX inpaint as alternative `AIOutpaintBackend` subclass.
- [ ] Fill in exact CLI inference commands if using external script instead of diffusers.
