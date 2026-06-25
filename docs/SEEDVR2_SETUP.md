# SeedVR2 Video Upscaling — Deployment Guide

_Goal: upscale the 720p/1080p source clip to 1440p+ **before** VR180/fulldome conversion.
SeedVR2 (ByteDance, ICLR2026) is the 2026 SOTA temporal video super-res — best-in-class on
short, AI-generated, compressed clips (exactly our input)._

This guide documents the **CLI backend** (the only path verified on a 12 GB RTX 4070S).
It uses the ComfyUI node's `inference_cli.py` — no ComfyUI server required.

> **Compatibility:** Windows 11 + 12 GB VRAM verified.  Python 3.10–3.12.

---

## 1. Clone ComfyUI + SeedVR2 Node

The node ships `inference_cli.py` which does everything: automatic model download (with resume),
OOM-retry, tiled VAE decode, and direct MP4 output.

```powershell
# 1. ComfyUI base (optional but provides model dir structure)
git clone https://github.com/comfyanonymous/ComfyUI.git
# or just create the model dir manually:
# mkdir -p ComfyUI/models/SEEDVR2

# 2. SeedVR2 custom node (the CLI script lives here)
cd ComfyUI/custom_nodes
git clone https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git
cd ComfyUI-SeedVR2_VideoUpscaler
```

**Directory layout after cloning:**
```
ComfyUI/
├── custom_nodes/
│   └── ComfyUI-SeedVR2_VideoUpscaler/
│       ├── inference_cli.py          ← the CLI entry point
│       ├── src/                      ← internal modules (imported via cwd)
│       └── ...
└── models/
    └── SEEDVR2/                      ← model files go here
        └── seedvr2_ema_3b_fp8_e4m3fn.safetensors
```

---

## 2. Download the Model

The node's auto-downloader **can** fetch models, but it lacks resume — a dropped connection
restarts from zero.  For reliable downloads use `curl` with `-C -` (resume):

```powershell
# From the node's HF mirror (preferred — matches the node's expected filename)
curl -L -C - -o ComfyUI/models/SEEDVR2/seedvr2_ema_3b_fp8_e4m3fn.safetensors ^
  https://huggingface.co/numz/SeedVR2_comfyUI/resolve/main/seedvr2_ema_3b_fp8_e4m3fn.safetensors

# Alternative: official ByteDance repo (but filename differs — you may need a symlink)
# curl -L -C - -o ComfyUI/models/SEEDVR2/seedvr2_ema_3b_fp8_e4m3fn.safetensors ^
#   https://huggingface.co/ByteDance-Seed/SeedVR2-3B/resolve/main/seedvr2_ema_3b_fp16.safetensors
```

> **Filenames matter.** `inference_cli.py` defaults to `DEFAULT_DIT = "seedvr2_ema_3b_fp8_e4m3fn.safetensors"`.
> If you use a different file, pass `--dit <filename>` to the CLI.

**Model choice for 12 GB:**

| Model file | VRAM | Quality | Recommended |
|---|---|---|---|
| `seedvr2_ema_3b_fp8_e4m3fn.safetensors` | ~10 GB | Good | **Default** — fits 12 GB with headroom |
| `seedvr2_ema_3b_fp16.safetensors` | ~14 GB | Best | OOMs on 12 GB; need 16 GB+ |
| `seedvr2_ema_3b-Q4_K_M.gguf` | ~8 GB | Fair | Use only if fp8 OOMs |

---

## 3. Verify the CLI Works

From the **node directory** (`ComfyUI/custom_nodes/ComfyUI-SeedVR2_VideoUpscaler`):

```powershell
python inference_cli.py --help
```

You should see flags like `--vae_decode_tiled`, `--batch_size`, `--resolution`, etc.
If you get `ModuleNotFoundError: No module named 'src'`, you're running from the wrong
directory — `cwd` must be the node directory.

---

## 4. Run Upscaling (Manual Test)

```powershell
cd ComfyUI/custom_nodes/ComfyUI-SeedVR2_VideoUpscaler

python inference_cli.py "D:\path\to\input.mp4" ^
  --output "D:\path\to\output.mp4" ^
  --resolution 1440 ^
  --batch_size 5 ^
  --vae_decode_tiled ^
  --vae_decode_tile_size 512 ^
  --vae_decode_tile_overlap 64 ^
  --output_format mp4 ^
  --model_dir "D:\ComfyUI\models\SEEDVR2"
```

**12 GB parameters explained:**

| Flag | Value | Why |
|---|---|---|
| `--resolution` | `1440` | Short-side target. 1080p × 2 = 2160 (too big), 1440 is the sweet spot |
| `--batch_size` | `5` | Must be `4n+1`.  5 fits in 12 GB; 9 OOMs on long clips |
| `--vae_decode_tiled` | *(on)* | **Required** — without it VAE decode OOMs on 12 GB |
| `--vae_decode_tile_size` | `512` | Smaller tiles = less VRAM, slightly slower |
| `--vae_decode_tile_overlap` | `64` | Tile overlap to hide seams |

**Performance:** ~60 s/frame at 1080p → 1440p.  A 100-frame 4-second clip takes ~100 minutes.
Upscaling to 2160p is proportionally slower (~3×).  This is an **offline tool** — start it and
walk away.

---

## 5. Integrate with Pipeline

Set environment variables (or pass CLI flags):

```powershell
$env:SEEDVR2_NODE_DIR = "D:\ComfyUI\custom_nodes\ComfyUI-SeedVR2_VideoUpscaler"
$env:SEEDVR2_MODEL_DIR = "D:\ComfyUI\models\SEEDVR2"
$env:SEEDVR2_RESOLUTION = "1440"
```

Then run:

```powershell
python scripts/run_pipeline.py ^
  --input video\googlegemini.mp4 ^
  --output video\googlegemini_vr180.mp4 ^
  --video-upscale seedvr2 ^
  --seedvr2-node-dir "%SEEDVR2_NODE_DIR%" ^
  --seedvr2-model-dir "%SEEDVR2_MODEL_DIR%" ^
  --seedvr2-resolution 1440 ^
  --src-hfov 150 --codec h265 --crf 16
```

If env vars are set, you can omit the `--seedvr2-*` flags entirely:

```powershell
python scripts/run_pipeline.py --input video.mp4 --output vr180.mp4 --video-upscale seedvr2
```

---

## 6. Troubleshooting

### "inference_cli.py not found"
→ You cloned the wrong repo or the node directory is incomplete.
```powershell
dir /b D:\ComfyUI\custom_nodes\ComfyUI-SeedVR2_VideoUpscaler\*.py
```
Should show `inference_cli.py`.  If not, re-clone:
```powershell
git -C D:\ComfyUI\custom_nodes\ComfyUI-SeedVR2_VideoUpscaler pull
```

### "No module named 'src'"
→ You are not running from the node directory.  `cwd` **must** be the node directory.
The pipeline's `CLIBackend` sets this automatically.

### "CUDA out of memory"
→ Reduce `--batch_size` to 1 or use `--vae_decode_tiled` (which is on by default).
If still OOM, switch to the GGUF quant model and add `--dit seedvr2_ema_3b-Q4_K_M.gguf`.

### Model download fails / slow
→ The auto-downloader in `inference_cli.py` uses `requests` without resume.
**Pre-download the model** manually with `curl -C -` (see Step 2 above) and place it in
`models/SEEDVR2/`.  The CLI will find it there and skip download.

### Very slow (~10 min per frame)
→ You might be CPU-bound (not using CUDA).  Verify with `nvidia-smi` that the GPU is utilised.
→ If piped through WSL2, GPU passthrough may not work.  Run natively on Windows.

---

## Environment Variable Reference

| Variable | Corresponding `--seedvr2-*` flag | Default |
|---|---|---|
| `SEEDVR2_NODE_DIR` | `--seedvr2-node-dir` | *(required)* |
| `SEEDVR2_PYTHON` | `--seedvr2-python` | `python` |
| `SEEDVR2_MODEL_DIR` | `--seedvr2-model-dir` | `<node_dir>/../../models/SEEDVR2` |
| `SEEDVR2_VAE_TILE_SIZE` | *(constructor param)* | `512` |
| `SEEDVR2_RESOLUTION` | `--seedvr2-resolution` | `1440` |

---

## Deprecated: ComfyUI Server Path

The old ComfyUIBackend (HTTP API) is preserved in `pipeline/video_upscaler.py` but is
**not recommended**.  The CLI backend (`CLIBackend`) is simpler, faster to set up, and
matches the exact command that was verified on the 12 GB test machine.
