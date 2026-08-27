# SeedVR2 Video Upscaling — In-Repo Deployment

_Goal: upscale the 720p/1080p source clip to 1440p+ **before** VR180/fulldome conversion.
SeedVR2 (ByteDance, ICLR2026) is the 2026 SOTA temporal video super-res — best-in-class on
short, AI-generated, compressed clips (exactly our input)._

As of 2026-08-28 the legacy `D:/ComfyUI` install was removed. SeedVR2 now lives
**inside this repo** (everything gitignored) and is picked up automatically by
`pipeline/video_upscaler.CLIBackend` via in-repo default paths. One command deploys it.

> **Compatibility:** Windows 11 + RTX 4070 SUPER (12 GB) verified. Python 3.10–3.12. CUDA-only.

---

## 1. One-Command Bootstrap

From the repo root, on the Windows host with the GPU:

```powershell
python scripts/setup_seedvr2.py
```

That's it. The script does everything, idempotently (re-running only fills missing pieces):

| Step | What | Time |
|---|---|---|
| 1 | `git clone` the SeedVR2 node repo into `third_party/seedvr2_videoupscaler/` (or `git pull` if present) | ~30 s |
| 2 | Build a **dedicated** venv *inside* the node dir (never the project-root venv), install `torch==2.6.0` on cu124 + the node's requirements | ~5 min |
| 3 | Download the two model weights to `models/SEEDVR2/` | ~15 min (4 GB) |
| 4 | Self-check: `inference_cli.py --help` | ~5 s |
| 5 | Print the env-var values (optional — in-repo defaults are used automatically) | |

**Options:**

```powershell
python scripts/setup_seedvr2.py --skip-model       # weights already on disk (>1 GB)
python scripts/setup_seedvr2.py --skip-deps        # venv + pip install already done
python scripts/setup_seedvr2.py --dry-run          # print planned steps, zero I/O
python scripts/setup_seedvr2.py --pip-mirror https://pypi.tuna.tsinghua.edu.cn/simple
```

The `--pip-mirror` flag forwards `-i <url>` to pip (useful from mainland China:
[Tsinghua](https://pypi.tuna.tsinghua.edu.cn/simple) / [Aliyun](https://mirrors.aliyun.com/pypi/simple/)).
`torch` is **always** installed from the official PyTorch cu124 index (never mirrored),
because the PyTorch wheels are index-specific.

---

## 2. In-Repo Layout

Everything below is gitignored (`models/`, `third_party/`, `.venv/`):

```
<repo>/
├── third_party/
│   └── seedvr2_videoupscaler/     ← git clone of numz/ComfyUI-SeedVR2_VideoUpscaler
│       ├── inference_cli.py       ← the CLI entry point
│       ├── src/                   ← internal modules
│       └── .venv/                 ← dedicated venv (torch + node deps)
├── models/
│   └── SEEDVR2/                   ← model weights
│       ├── seedvr2_ema_3b_fp8_e4m3fn.safetensors   (≈3.2 GB)
│       └── ema_vae_fp16.safetensors                (≈0.5 GB)
└── pipeline/video_upscaler.py     ← CLIBackend uses these paths by default
```

The pipeline's `CLIBackend` resolves paths in this order (each layer only when the
previous is unset AND the path actually exists on disk):

1. Explicit `--seedvr2-*` flag / constructor argument
2. `SEEDVR2_NODE_DIR` / `SEEDVR2_PYTHON` / `SEEDVR2_MODEL_DIR` env vars
3. In-repo defaults: `third_party/seedvr2_videoupscaler`, its `.venv/python`, `models/SEEDVR2`
4. *(only for `model_dir`)* legacy ComfyUI layout `<node_dir>/../../models/SEEDVR2`

If none of the three layers resolve for `node_dir`, the constructor raises and points you
at `scripts/setup_seedvr2.py`.

---

## 3. Disk + VRAM Requirements

| Resource | Requirement | Notes |
|---|---|---|
| **Disk** | ~8 GB free | 4 GB models + ~3 GB venv + git clone |
| **VRAM** | 12 GB (RTX 4070 SUPER) | 16 GB+ recommended for faster throughput |
| **Python** | 3.10–3.12 | The host's python; a separate venv is created inside the node dir |
| **CUDA** | 12.4 drivers | torch 2.6.0 cu124 wheels |
| **ffmpeg** | on PATH | required by `inference_cli.py` + ffprobe |

**Model choice for 12 GB:**

| Model file | Size | VRAM | Quality | Recommended |
|---|---|---|---|---|
| `seedvr2_ema_3b_fp8_e4m3fn.safetensors` | **3.2 GB** | ~10 GB | Good | **Default** — fits 12 GB with headroom |
| `ema_vae_fp16.safetensors` | **0.5 GB** | ~1 GB | Good | **Required** for VAE encode/decode |
| `seedvr2_ema_3b_fp16.safetensors` | ~6 GB | ~14 GB | Best | OOMs on 12 GB; need 16 GB+ |

---

## 4. Run Upscaling

**You don't need to set any env vars** — the pipeline uses the in-repo defaults. Just:

```powershell
python scripts/run_pipeline.py ^
  --input video\source.mp4 ^
  --output video\upscaled_then_vr180.mp4 ^
  --video-upscale seedvr2 ^
  --seedvr2-resolution 1440 ^
  --src-hfov 150 --codec h265 --crf 16
```

To override the in-repo paths (e.g. weights on a faster NVMe), export the env vars the
bootstrap prints, or pass `--seedvr2-node-dir` / `--seedvr2-python` / `--seedvr2-model-dir`
to `run_pipeline.py`.

**12 GB parameters** (all set by `CLIBackend` automatically — verified on the RTX 4070S):

| Flag | Value | Why |
|---|---|---|
| `--resolution` | `1440` | Short-side target; 1080p×2=2160 is too big, 1440 is the sweet spot |
| `--batch_size` | `5` | Must be `4n+1`. 5 fits 12 GB; 9 OOMs on long clips |
| `--vae_decode_tiled` / `_encode_tiled` | on | **Required** — without it VAE encode/decode OOMs on 12 GB |
| `--vae_*_tile_size` | `512` | Smaller tiles = less VRAM, slightly slower |
| `--dit_offload_device` | `cpu` | Offload DIT transformer layers — saves ~2 GB |
| `--vae_offload_device` | `cpu` | Offload VAE — saves ~1 GB |

**Performance:** ~60 s/frame at 1080p → 1440p. A 100-frame 4-second clip takes ~100 minutes.
This is an **offline tool** — start it and walk away.

---

## 5. Troubleshooting

### `git clone` / `git pull` fails
Likely a network/proxy issue (common from mainland China). Set a proxy and re-run:

```powershell
git config --global http.proxy http://your-proxy:port
git config --global https.proxy http://your-proxy:port
python scripts/setup_seedvr2.py
```

### `pip install` fails / slow
Use a PyPI mirror for the CPU-side packages (torch still comes from the official cu124 index):

```powershell
python scripts/setup_seedvr2.py --pip-mirror https://pypi.tuna.tsinghua.edu.cn/simple
```

### Model download fails / slow
The bootstrap uses `hf_hub_download` first (with resume) and falls back to `curl -C -`
(resume-capable). If both fail, download manually and drop the files into `models/SEEDVR2/`:

```powershell
curl -L -C - -o models/SEEDVR2/seedvr2_ema_3b_fp8_e4m3fn.safetensors ^
  https://huggingface.co/numz/SeedVR2_comfyUI/resolve/main/seedvr2_ema_3b_fp8_e4m3fn.safetensors
curl -L -C - -o models/SEEDVR2/ema_vae_fp16.safetensors ^
  https://huggingface.co/numz/SeedVR2_comfyUI/resolve/main/ema_vae_fp16.safetensors
```

Re-running `python scripts/setup_seedvr2.py` will **skip** any file >1 GB.

### `CUDA out of memory`
Reduce `--batch_size` to 1, or pass `--seedvr2-resolution 1080`. The default config
(tiled VAE + CPU offload) already targets 12 GB.

### `CUDA is not available`
The upscaler is CUDA-only. Verify `nvidia-smi` shows the GPU and that the pytorch
installed in the **dedicated venv** (`third_party/seedvr2_videoupscaler/.venv/`) is the
CUDA build:

```powershell
third_party\seedvr2_videoupscaler\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
```

### `No module named 'src'` / `inference_cli.py` errors
You (or a script) is running `inference_cli.py` from the wrong directory. The pipeline's
`CLIBackend` sets `cwd` to the node dir automatically — you should never run the CLI by hand.
If the node dir is missing `inference_cli.py`, re-run the bootstrap.

---

## 6. Environment Variable Reference

| Variable | Corresponding `--seedvr2-*` flag | Default (in-repo) |
|---|---|---|
| `SEEDVR2_NODE_DIR` | `--seedvr2-node-dir` | `<repo>/third_party/seedvr2_videoupscaler` |
| `SEEDVR2_PYTHON` | `--seedvr2-python` | `<repo>/third_party/seedvr2_videoupscaler/.venv/python` |
| `SEEDVR2_MODEL_DIR` | `--seedvr2-model-dir` | `<repo>/models/SEEDVR2` |
| `SEEDVR2_VAE_TILE_SIZE` | *(constructor param)* | `512` |
| `SEEDVR2_RESOLUTION` | `--seedvr2-resolution` | `1440` |

---

## 7. CI Note

CI is Ubuntu + CPU-only + no GPU + no downloads. The bootstrap script is **never run**
in CI — it is tested with `--dry-run`, which asserts the planned step sequence with zero
I/O. The pipeline's `CLIBackend` is tested with mocked subprocess calls and fake node dirs
built with `tmp_path`.
