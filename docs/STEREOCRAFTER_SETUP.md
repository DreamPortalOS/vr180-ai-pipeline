# StereoCrafter Setup Guide (in-repo bootstrap)

> **StereoCrafter** (by Tencent) is a video diffusion model that produces
> clean stereoscopic left/right views from a video + per-frame depth maps.
> It uses depth-guided forward splatting to detect disocclusion regions,
> then a video diffusion model inpaints those regions with temporally
> consistent content — eliminating the ghosting/smear artifacts of
> simple depth-shift rendering.

This guide describes the **in-repo** deployment that ships with the
pipeline.  A single one-command bootstrap installs everything inside the
repo so the pipeline picks it up automatically — no separate
`D:/StereoCrafter` install, no Conda environment, no manual path wiring.

---

## 1. One-command bootstrap (recommended)

```bash
python scripts/setup_stereocrafter.py
```

The script is **idempotent** — re-running it only fills missing pieces.  It:

1. Clones `TencentARC/StereoCrafter` into `third_party/StereoCrafter/` (or
   `git pull`s if already present).
2. Builds a **dedicated** Python venv inside that checkout (never the
   project-root venv) and installs:
   - `torch==2.6.0` + `torchvision==0.21.0` from the stable **cu124**
     index (NOT nightly).
   - A curated subset of runtime deps that the inference entry point
     actually imports.  (We deliberately do **not** run `pip install -e .`
     against the upstream pyproject — it would bump the torch pins or add
     demo-only deps.)
3. Downloads `TencentARC/StereoCrafter` weights into
   `models/StereoCrafter/` via `huggingface_hub.snapshot_download`
   (skipped if already present).
4. Runs a self-check (`<inference_entry_point> --help`) so a broken env
   is caught early.

Created layout (everything is gitignored):

```
third_party/StereoCrafter/    ← TencentARC/StereoCrafter checkout (+ its own .venv)
models/StereoCrafter/         ← TencentARC/StereoCrafter weights
```

### Bootstrap options

```bash
python scripts/setup_stereocrafter.py --repo-dir D:/StereoCrafter   # use an existing checkout
python scripts/setup_stereocrafter.py --skip-model                  # weights already downloaded
python scripts/setup_stereocrafter.py --skip-deps                   # venv + pip already set up
python scripts/setup_stereocrafter.py --pip-mirror https://pypi.tuna.tsinghua.edu.cn/simple
python scripts/setup_stereocrafter.py --dry-run                     # print planned steps, no I/O
```

---

## 2. Requirements

| Component | Requirement |
|-----------|-------------|
| GPU       | NVIDIA GPU, **12 GB VRAM minimum** (24 GB recommended) |
| CUDA      | CUDA 12.4 (torch is pinned to the cu124 wheel index) |
| OS        | Windows / Linux / macOS are all supported by the bootstrap; **inference is CUDA-only** |
| Python    | 3.10+ (the dedicated venv is created by the bootstrap) |
| Disk      | ~20 GB (checkout + weights + venv) **+ ~10 GB for the SVD base** — auto-downloaded by diffusers on the first inference run, or pre-downloaded via `--svd-dir` |
| Network   | git + HF download (one-time); the bootstrap prints proxy hints if `git clone` fails |

---

## 3. Run it through the pipeline

The in-repo paths are the pipeline's **defaults**, so after the bootstrap
succeeds no environment variables are needed:

```bash
python scripts/run_pipeline.py \
    --input video.mp4 \
    --output vr180.mp4 \
    --depth-model depthcrafter \
    --stereo-model stereocrafter
```

> **Depth supply (streaming path, `--quality standard|high`):** StereoCrafter's
> in-repo forward-splat consumes the pipeline's own per-frame depth maps
> (issue #140). The streaming run feeds them like this (issue #143):
>
> * With `--depth-model depthcrafter` (recommended): the DepthCrafter stage's
>   real output dir is handed straight to `render_video` — temporally stable,
>   flicker-free depth.
> * Without it: the streaming pipeline **auto-emits** per-frame depth maps
>   with the per-frame estimator (Depth-Anything) into a temp dir and hands
>   that over, logging a WARNING. Functional, but not flicker-free — prefer
>   `--depth-model depthcrafter` for final renders.

You can override any of the defaults explicitly:

```bash
export STEREOCRAFTER_REPO_DIR="/path/to/StereoCrafter"
export STEREOCRAFTER_PYTHON="/path/to/StereoCrafter/.venv/.../python"
export STEREOCRAFTER_CKPT_DIR="/path/to/models/StereoCrafter"
export STEREOCRAFTER_MAX_RES="512"
# Windows PowerShell: $env:STEREOCRAFTER_REPO_DIR="<...>"
```

Or via CLI flags:

```bash
python scripts/run_pipeline.py \
    --input video.mp4 --output vr180.mp4 \
    --stereo-model stereocrafter \
    --stereocrafter-repo-dir /path/to/StereoCrafter \
    --stereocrafter-python /path/to/StereoCrafter/.venv/.../python \
    --stereocrafter-checkpoint-dir /path/to/models/StereoCrafter \
    --stereocrafter-max-res 512
```

### Env / constructor precedence

For each path, the resolution order is **constructor arg > env var >
in-repo default (only if it exists on disk)**; if none resolve, the
backend raises and points at `scripts/setup_stereocrafter.py`.

| Setting                 | Env var                        | Default (if env unset) |
|-------------------------|--------------------------------|------------------------|
| `repo_dir`              | `STEREOCRAFTER_REPO_DIR`       | `third_party/StereoCrafter` *(if exists)* |
| `python_exe`            | `STEREOCRAFTER_PYTHON`         | in-repo venv python *(if exists)*, else `python` |
| `checkpoint_dir`        | `STEREOCRAFTER_CKPT_DIR`       | `models/StereoCrafter` *(if exists)*, else `(<repo>)/checkpoints` — Stage 2 `--unet_path` |
| `pre_trained_path`      | `STEREOCRAFTER_SVD_PATH`       | `models/svd-img2vid-xt-1-1` *(if exists)*, else the HF id `stabilityai/stable-video-diffusion-img2vid-xt-1-1` — SVD base (Stage 2 `--pre_trained_path`) |

> **SVD base model (~10 GB) — first-run download.**  If no local copy of the
> SVD base exists, the backend passes the HF model id straight to diffusers,
> which downloads ~10 GB into the HF cache on the **first inference run**
> (same behaviour as DepthCrafter's auto-pull).  To front-load the download
> instead, run `python scripts/setup_stereocrafter.py --svd-dir models/svd-img2vid-xt-1-1`
> once, or set `STEREOCRAFTER_SVD_PATH` to an existing local snapshot.  A
> nonexistent local path is never passed — diffusers would treat it as an HF
> repo id and crash (issue #147).
| `max_resolution`        | `STEREOCRAFTER_MAX_RES`        | `512` |
| `max_disp`              | `STEREOCRAFTER_MAX_DISP`       | `20.0` — stereo baseline for the in-repo forward-splat |

> **Removed (issue #140):** `depthcrafter_unet_path` / `STEREOCRAFTER_DC_UNET_PATH`
> and the whole Stage-1 invocation.  The upstream Stage-1 `--unet_path` pointed at
> an *embedded* DepthCrafter this repo never deploys (see §5).  Depth now comes
> from the pipeline's own depth stage, so there is no Stage-1 UNet to configure.

---

## 4. VRAM / resolution tuning (12 GB target)

The default `--stereocrafter-max-res 512` is chosen to be safe on an **RTX
4070 SUPER 12 GB**.  Adjust to fit your GPU:

| Short-side (`--stereocrafter-max-res`) | Approx. VRAM | Use on |
|----------------------------------------|--------------|--------|
| 384                                    | ~10 GB       | tight 12 GB budgets / long clips |
| 512 (default)                          | ~12 GB       | RTX 4070 SUPER |
| 768                                    | ~16–18 GB    | RTX 3080 / 4080 class |
| 1024                                   | ~24 GB+      | A100 / RTX 4090 class |

A lower value scales the input down before inference (less VRAM, faster,
slightly softer output).  Set it with `--stereocrafter-max-res` or the
`STEREOCRAFTER_MAX_RES` env var.

> **Tip:** if you hit `CUDA out of memory`, drop `--stereocrafter-max-res`
> to the next tier down and re-run.  You do **not** need to rebuild the
> venv or redownload weights.

---

## 5. What the pipeline actually does with StereoCrafter (Stage 2 only — issue #140)

```
Input Video ──┐
              ├─► in-repo assembly (NO subprocess)        ← replaces upstream Stage 1
Pipeline's     │    (pipeline/stereo_crafter._write_splatting_grid_video)
own depth ────┘      input video + per-frame depth maps (depth_dir)
(--depth-model           │
  depthcrafter /         ▼
  depth-anything)    2×2 grid video (splatting_results.mp4)
                          │
                          ▼
                   Stage 2 subprocess: inpainting_inference.py   ← the disocclusion step
                        (video-diffusion-inpaints the disocclusion mask)
                        → <name>_sbs.mp4 (side-by-side stereo video)
                          │
                          ▼
                   SBS split → separate left.mp4 / right.mp4
```

**This repo drives only Stage 2.**  The upstream repo has **no `run.py`**; it
exposes two `fire`-style entry scripts at the repo root (both accept `--help`):

| Script | Stage | Used here? | Purpose |
|--------|-------|------------|---------|
| `depth_splatting_inference.py` | 1 | **NO** | Estimate per-frame depth (its own *embedded* DepthCrafter under `dependency/DepthCrafter/`), then forward-splat the left view to expose the disocclusion mask. |
| `inpainting_inference.py` | 2 | **YES** | The disocclusion-inpainting step this repo needs. Takes the 2×2 grid video and video-diffusion-inpaints the masked regions, writing `<video_name>_sbs.mp4` (left = original view, right = inpainted view) plus an anaglyph preview. Flags: `--pre_trained_path`, `--unet_path`, `--input_video_path`, `--save_dir`, `--frames_chunk`, `--overlap`, `--tile_num`. |

### Why Stage 1 is not run (and DepthCrafter is never embedded)

Upstream Stage 1 (`depth_splatting_inference.py`) hard-imports an **embedded**
copy of DepthCrafter at `dependency/DepthCrafter/depthcrafter/…`.  A stock
`TencentARC/StereoCrafter` checkout does **not** ship that copy, so running
Stage 1 as-is crashes immediately with:

```
ModuleNotFoundError: No module named 'dependency.DepthCrafter.depthcrafter'
```

Embedding DepthCrafter inside the StereoCrafter checkout would (a) duplicate
~3 GB of weights this repo already deploys for `--depth-model depthcrafter`,
and (b) conflict with the repo's own depth chain.  So Stage 1 is replaced by
two in-repo pieces (`pipeline/stereo_crafter.py`):

* **Depth** — the pipeline's own depth stage (`--depth-model depthcrafter` /
  `depth-anything`) writes `depth_*.npy` maps; `render_video` reads them from
  `depth_dir` (which **is** consumed now — it feeds the splat, not a
  subprocess).
* **Forward-splat** — a numpy port of upstream's `ForwardWarpStereo`
  (softmax-weighted splatting with an occlusion map) warps the left view to
  the right eye and derives the disocclusion mask.  The CUDA-only
  `Forward_Warp` extension upstream Stage 1 requires is deliberately not used.

The assembled **2×2 grid video** matches the layout `inpainting_inference.py`
actually parses (verified against the upstream source):

```
[ left        | depth_vis   ]      depth_vis is cosmetic — Stage 2 crops it away
[ mask        | warped_right]
```

`StereoCrafterRenderer` handles:
1. **CUDA guard** — raises a clear error if no GPU.
2. **Input assembly** — input video + the pipeline's own depth maps → the 2×2
   grid video (in-repo, no subprocess).
3. **Subprocess delegation** — runs Stage 2 only (fire-style flags).
4. **SBS split + output verification** — splits the SBS video into L/R files.

For testing, inject a `MockStereoCrafterBackend` to bypass all model
dependencies (see `tests/test_stereo_crafter.py`).

---

## 6. Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| `CUDA out of memory` | Resolution too high for your GPU | Lower `--stereocrafter-max-res` (e.g. 512 → 384) |
| `No known inference script found` | Repo checkout missing `inpainting_inference.py` (the only recognized entry — a stray `run.py`/`inference.py`/`depth_splatting_inference.py` is NOT accepted) | Re-run the bootstrap; or clone `TencentARC/StereoCrafter` into `third_party/StereoCrafter/` |
| `No module named 'dependency.DepthCrafter...'` | A Stage-1 (`depth_splatting_inference.py`) invocation — this repo **never** runs Stage 1 (issue #140). If you see this, you're on an old build or invoked the upstream script by hand | Pull the fix; run only via `--stereo-model stereocrafter`. Do **not** embed DepthCrafter into the StereoCrafter checkout |
| `No depth maps found in <depth_dir>` | Batch path: the pipeline's depth stage didn't run / wrote nothing — Stage-2 assembly needs the pipeline's own per-frame depth maps. (Streaming path: not expected — since issue #143 the streaming run either hands StereoCrafter the DepthCrafter stage's real output dir, or auto-emits per-frame depth maps itself.) | Batch: run the depth stage first (`--stage depth`), or use `--stereo-model default`. Streaming: pass `--depth-model depthcrafter` (recommended — flicker-free), or rely on the auto-emitted per-frame fallback (logged as a WARNING) |
| `No module named '...'` | Runtime dep missing from the dedicated venv | Add it to `RUNTIME_DEPS` in `scripts/setup_stereocrafter.py` and re-run |
| `FileNotFoundError: python` | Wrong venv python path | The bootstrap writes it in-repo; run the bootstrap, or set `STEREOCRAFTER_PYTHON` |
| Self-check fails | The inference entry point can't import | `python <venv>/python inpainting_inference.py --help` inside `third_party/StereoCrafter/` to see the real error |
| `git clone` fails | Network / proxy (common from mainland China) | Set `http(s).proxy`, or clone manually then pass `--repo-dir` |
| `snapshot_download` fails | HF access / network | Re-run the bootstrap; or clone `https://huggingface.co/TencentARC/StereoCrafter` into `models/StereoCrafter/` manually |
| Subprocess non-zero exit | StereoCrafter internal error | Run `inpainting_inference.py` directly inside `third_party/StereoCrafter/` to isolate the failure; check its stderr |
| `SBS output not found` | Stage 2 succeeded but did not write `<name>_sbs.mp4` where expected | Confirm `--save_dir` is writable and the inpainting step completed; check stderr of Stage 2 |

### If the bootstrap is not run / paths missing

`CLIBackend` raises an actionable error:

```
RuntimeError: No StereoCrafter repo/python/checkpoint paths were configured or found in-repo.
  Run the one-command bootstrap to deploy StereoCrafter inside the repo:
    python scripts/setup_stereocrafter.py
  See docs/STEREOCRAFTER_SETUP.md for disk/VRAM requirements and troubleshooting.
```

The fallback stereo model (`--stereo-model default`) does **not** require
StereoCrafter — it uses the existing `StereoRenderer` (depth-shift + simple
inpaint).  StereoCrafter is the upgrade path for cleaner disocclusion
inpainting (no ghosting/smear at object edges).
