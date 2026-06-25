# StereoCrafter Setup Guide

> **StereoCrafter** (by Tencent) is a video diffusion model that produces
> clean stereoscopic left/right views from a video + per-frame depth maps.
> It uses depth-guided forward splatting to detect disocclusion regions,
> then a video diffusion model inpaints those regions with temporally
> consistent content — eliminating the ghosting/smear artifacts of
> simple depth-shift rendering.

---

## Requirements

| Component      | Requirement                                    |
|----------------|------------------------------------------------|
| GPU            | NVIDIA GPU with 16 GB+ VRAM (24 GB recommended) |
| CUDA           | CUDA 11.8+                                     |
| OS             | Linux (Ubuntu 22.04+) or Windows (WSL2)         |
| Python         | 3.10+                                          |
| Disk           | ~30 GB (code + checkpoints + training data)    |

---

## 1. Clone the Repository

```bash
git clone https://github.com/Tencent/StereoCrafter.git
cd StereoCrafter
```

---

## 2. Install Dependencies

```bash
# Create a dedicated conda environment
conda create -n stereocrafter python=3.10 -y
conda activate stereocrafter

# Install PyTorch (CUDA 11.8)
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# Install StereoCrafter requirements
pip install -r requirements.txt

# Install xformers for memory-efficient attention
pip install xformers==0.0.22
```

---

## 3. Download Pretrained Checkpoints

> **TODO (lead):** Fill in the exact commands and URLs for downloading the
> StereoCrafter checkpoint(s). The original repo typically provides one or
> more of the following:

- **StereoCrafter** — main model for L/R generation from video + depth
- **DepthCrafter** — (already handled by `DepthCrafterEstimator`) depth
  estimation used as input to StereoCrafter

Place checkpoints in:

```
checkpoints/
├── stereocrafter.ckpt
├── ...
```

Expected download commands (to be filled by lead):

```bash
# Placeholder — replace with actual URLs from Tencent/StereoCrafter
wget -P checkpoints/ https://huggingface.co/Tencent/StereoCrafter/resolve/main/stereocrafter.ckpt
```

---

## 4. Verify Inference

```bash
# Run a quick test with a short sample video
python run.py \
    --video sample.mp4 \
    --depth_dir ./depth_maps/ \
    --output_left left_out.mp4 \
    --output_right right_out.mp4 \
    --max_resolution 512 \
    --checkpoint_dir ./checkpoints/
```

> **TODO (lead):** Fill in the *exact* CLI flags expected by Tencent's
> inference script, and any additional arguments (e.g., `--num_frames`,
> `--overlap`).

---

## 5. Integration with vr180-ai-pipeline

Once StereoCrafter is deployed and the repo path is known, tell the pipeline
to use it:

```bash
# Via environment variables
export STEREOCRAFTER_REPO_DIR=/path/to/StereoCrafter
export CUDA_VISIBLE_DEVICES=0

python scripts/run_pipeline.py \
    --input video.mp4 \
    --output vr180.mp4 \
    --depth-model depthcrafter \
    --stereo-model stereocrafter
```

Or via CLI flags:

```bash
python scripts/run_pipeline.py \
    --input video.mp4 \
    --output vr180.mp4 \
    --depth-model depthcrafter \
    --stereo-model stereocrafter \
    --stereocrafter-repo-dir /path/to/StereoCrafter \
    --stereocrafter-python ~/miniconda3/envs/stereocrafter/bin/python \
    --stereocrafter-max-res 768
```

---

## Optional: Running Without StereoCrafter Repository

If the repository is not deployed, `CLIBackend` will raise a clear error:

```
RuntimeError: StereoCrafter repository not found at: /path/to/StereoCrafter
  Clone the repo:
    git clone https://github.com/Tencent/StereoCrafter.git
```

The fallback stereo model (`--stereo-model default`) does **not** require
StereoCrafter — it uses the existing `StereoRenderer` (depth-shift + simple
inpaint).

---

## Troubleshooting

| Symptom                     | Likely Cause                         | Fix                                                |
|-----------------------------|--------------------------------------|----------------------------------------------------|
| `CUDA out of memory`        | Resolution too high                   | Reduce `--stereocrafter-max-res` (e.g., 512→384)  |
| `No module named '...'`     | StereoCrafter deps not installed      | `pip install -r requirements.txt` in stereo env    |
| `FileNotFoundError: python` | Wrong Python path                     | Set `--stereocrafter-python` to conda env python   |
| Subprocess non-zero exit    | StereoCrafter internal error          | Run StereoCrafter directly to isolate              |
| Output files not created    | Inference script changed output paths | Update `_find_inference_script` / script arguments |

---

## Architecture Notes

Pipeline integration (for developers):

```
Input Video ──► DepthCrafter ──► Depth Maps ──► StereoCrafter ──► L/R Videos
    │                                              │
    │                                              │
    └── stereo_crafter.py                          │
        │                                          │
        ├── StereoCrafterRenderer (CUDA guard)     │
        │   └── render_video(input, depth, ...)    │
        │                                          │
        └── CLIBackend                             │
            └── subprocess(run.py, ...) ────────────┘
```

The `StereoCrafterRenderer` class handles:
1. **CUDA guard**: raises clear error if no GPU
2. **Input assembly**: writes frames + depth maps to temp files
3. **Subprocess delegation**: calls actual inference script
4. **Output verification**: checks L/R videos were created
5. **Cleanup**: temp files managed by tempdir

For testing, inject a `MockStereoCrafterBackend` to bypass all model
dependencies.
