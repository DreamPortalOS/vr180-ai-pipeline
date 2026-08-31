# DepthCrafter Setup Guide

[Tencent/DepthCrafter](https://github.com/Tencent/DepthCrafter) provides
**temporally-consistent video depth estimation** — the cure for the
"flicker / left-and-right eye won't fuse" problem caused by per-frame depth
estimators (such as Depth-Anything V2).  It is the optional
`--depth-model depthcrafter` backend of `scripts/run_pipeline.py`; the
default depth backend remains Depth-Anything.

**CUDA-only.** Requires an NVIDIA GPU.  The target box is a Windows 12 GB
RTX 4070 SUPER.

---

## One-command bootstrap (recommended, G-6 style)

DepthCrafter is now managed **inside this repo** by an idempotent bootstrap
script — no separate `D:/DepthCrafter` install required:

```bash
python scripts/setup_depthcrafter.py
```

This creates (everything is gitignored):

```
third_party/DepthCrafter/   ← git clone of Tencent/DepthCrafter (with its own .venv)
models/DepthCrafter/        ← tencent/DepthCrafter weights (diffusers snapshot)
```

The script is idempotent.  Re-running only fills missing pieces.  Useful
switches:

```bash
python scripts/setup_depthcrafter.py --dry-run              # print planned steps, zero I/O
python scripts/setup_depthcrafter.py --skip-model           # weights already present
python scripts/setup_depthcrafter.py --skip-deps            # venv + pip already done
python scripts/setup_depthcrafter.py --repo-dir D:/DepthCrafter  # use an existing checkout
python scripts/setup_depthcrafter.py --pip-mirror https://pypi.tuna.tsinghua.edu.cn/simple
```

> **Why an in-repo venv, not the project venv.**  The dedicated venv
> (`third_party/DepthCrafter/.venv`) is *inside* the DepthCrafter checkout and
> is never the project-root venv.  It pins `torch==2.6.0` +
> `torchvision==0.21.0` on the official `cu124` index (the paired torchvision
> release — `torchvision==2.6.0` does not exist).
>
> **Node-deps install strategy.**  The upstream
> [`Tencent/DepthCrafter`](https://github.com/Tencent/DepthCrafter) repo
> declares its dependencies in **`pyproject.toml`**, but the bootstrap
> deliberately does **not** run `pip install -e .` against it.  Instead it
> installs a curated subset of the packages `run.py` actually imports
> (``fire diffusers transformers accelerate huggingface-hub mediapy decord``),
> exposed as the `RUNTIME_DEPS` constant in the script.  Reasons we do not
> follow the upstream pyproject wholesale:
>
> - **Editable install is broken.**  The upstream repo is a *flat-layout*
>   (`depthcrafter/` + `visualization/` top-level packages), so setuptools
>   rejects `pip install -e .` with "Multiple top-level packages discovered in
>   a flat-layout".  DepthCrafter is run as a script (`run.py`), never installed
>   as a package, so an editable install is unnecessary.
> - **Python / torch version conflicts.**  The upstream pyproject declares
>   `requires-python = ">=3.13"` and pins `torch>=2.7.1` / `torchvision>=0.22.1`.
>   Our dedicated venv runs **Python 3.12** and pins `torch==2.6.0` +
>   `torchvision==0.21.0` (cu124) in Step 2 of the bootstrap.  Following the
>   upstream declaration would either bump torch off the verified cu124 pairing
>   or fail outright on Python 3.12.
> - **Heavy demo/dev deps are excluded.**  `gradio`, `xformers`, `pytest` and
>   `matplotlib` from the upstream set are demo/development dependencies and
>   are intentionally left out of the 12 GB VRAM inference environment.
>
> The curated list keeps the venv aligned with our pinned cu124 torch pairing
> and with what `run.py` actually needs at import time.  If the list needs to
> grow (e.g. a new hard import in `run.py`), update `RUNTIME_DEPS` in
> `scripts/setup_depthcrafter.py` — torch/torchvision stay in Step 2 so a
> transitive dependency can never bump them.
>
> **Self-check is a hard gate.**  The final step runs `run.py --help` with the
> dedicated venv and treats any non-zero exit (or missing venv-python / missing
> `run.py`) as a fatal failure — the bootstrap exits 1 and prints the real
> stderr, so an unusable environment can never look like success.

### First inference run: ~10 GB auto-download

The bootstrap downloads only the `tencent/DepthCrafter` weights.  The base
model **`stabilityai/stable-video-diffusion-img2vid-xt`** is pulled
**automatically by diffusers on the first inference run** (about 10 GB in
total).  **Do not set `HF_ENDPOINT` to a mirror** — direct HuggingFace access
is required; the mirror path is unreliable for these large diffusers weights.

---

## Environment variables

The pipeline's `CLIBackend` (`pipeline/depth_crafter.py`) falls back to the
in-repo defaults when these env vars are unset.  In-repo paths are only
adopted when they actually exist on disk.  You generally **do not** need to
export anything — just run the bootstrap.

| Variable | Default (if unset) | Purpose |
|----------|--------------------|---------|
| `DEPTHCRAFTER_REPO_DIR` | `third_party/DepthCrafter` *(if exists)* | DepthCrafter checkout |
| `DEPTHCRAFTER_PYTHON` | in-repo venv python *(if exists)*, else `python` | Python for inference |
| `DEPTHCRAFTER_MODEL_DIR` | `models/DepthCrafter` *(if exists)* | Weight directory |
| `DEPTHCRAFTER_MAX_RES` | **`512`** (12 GB VRAM-safe) | Max short-side resolution |
| `DEPTHCRAFTER_PROCESS_LENGTH` | unset | Frames per chunk |
| `DEPTHCRAFTER_TARGET_FPS` | unset | Output FPS |

---

## 12 GB VRAM default

The official DepthCrafter default is `--max_res 1024`, but that blows the
12 GB buffer on the RTX 4070 SUPER.  The pipeline defaults to
**`DEPTHCRAFTER_MAX_RES=512`** — lead-verified safe floor.

Tradeoff: lower resolution → coarser depth → the stereoscopic effect is
slightly softer, but the depth is temporally smooth and the two eyes will
actually fuse.  If you have a 24 GB card, raise it:

```bash
export DEPTHCRAFTER_MAX_RES=768      # 24 GB card
# or pass it through the pipeline:
python scripts/run_pipeline.py ... --depthcrafter-max-res 768
```

---

## Running

With the bootstrap in place, the pipeline picks up DepthCrafter automatically:

```bash
python scripts/run_pipeline.py \
    --input video/src_720p_v2.mp4 \
    --output out/vr180.mp4 \
    --depth-model depthcrafter
```

### Pipeline flags

```
--depth-model {depth-anything,depthcrafter}   Depth backend (default: depth-anything)
--depthcrafter-repo-dir DIR                    Override repo path
--depthcrafter-python EXE                      Override python
--depthcrafter-checkpoint-dir DIR              Override model dir (aliased to model_dir)
--depthcrafter-max-res N                       Max short-side resolution
```

### Underlying CLI shape

The DepthCrafter `run.py` is a **fire-style** (positional first arg) CLI —
*not* argparse.  The pipeline builds this command (lead-verified 2026-08-31):

```bash
python run.py <video_path> --save_folder <dir> --max_res 512 --cpu_offload model \
    [--process_length N] [--target_fps N]
```

Output depth maps are written into `--save_folder` as `.npy` (preferred) or
`depth_*.png`, which the pipeline then loads as `(H, W)` float32 arrays.

---

## Existing external checkout (`D:/DepthCrafter`)

> **⚠️ Do not reuse.**  The pre-existing `D:/DepthCrafter` checkout ships with
> a `.venv` that has `torch 2.13.0+cpu` (no CUDA) and ABI-incompatible
> diffusers/transformers compiled against that torch.  Attempting to just
> swap in `torch 2.6.0+cu124` crashes with `WinError 127` on import.
>
> The bootstrap creates a **fresh, clean** venv inside the in-repo checkout
> instead.  If you really need to reuse an external clone, pass `--repo-dir`
> — the bootstrap will (re-)install a clean venv into it.

---

## Testing without a GPU

All unit tests in `tests/test_depth_crafter.py` and
`tests/test_setup_depthcrafter.py` are mock-based — zero network, zero
downloads, no CUDA.  Run with:

```bash
pytest tests/test_depth_crafter.py tests/test_setup_depthcrafter.py -v
```
