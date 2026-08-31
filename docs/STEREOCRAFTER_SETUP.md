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
| Disk      | ~20 GB (checkout + weights + venv) |
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

| Setting             | Env var                   | Default (if env unset) |
|---------------------|---------------------------|------------------------|
| `repo_dir`          | `STEREOCRAFTER_REPO_DIR`  | `third_party/StereoCrafter` *(if exists)* |
| `python_exe`        | `STEREOCRAFTER_PYTHON`    | in-repo venv python *(if exists)*, else `python` |
| `checkpoint_dir`    | `STEREOCRAFTER_CKPT_DIR`  | `models/StereoCrafter` *(if exists)*, else `(<repo>)/checkpoints` |
| `max_resolution`    | `STEREOCRAFTER_MAX_RES`   | `512` |

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

## 5. What the pipeline actually does with StereoCrafter

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
            └── subprocess(<inference_entry_point>, ...) ───────────┘
```

`StereoCrafterRenderer` handles:
1. **CUDA guard** — raises a clear error if no GPU.
2. **Input assembly** — the pipeline stages frames + depth into a temp video.
3. **Subprocess delegation** — calls the repo's inference entry point.
4. **Output verification** — checks both L/R videos were created.

For testing, inject a `MockStereoCrafterBackend` to bypass all model
dependencies (see `tests/test_stereo_crafter.py`).

---

## 6. Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| `CUDA out of memory` | Resolution too high for your GPU | Lower `--stereocrafter-max-res` (e.g. 512 → 384) |
| `No module named '...'` | Runtime dep missing from the dedicated venv | Add it to `RUNTIME_DEPS` in `scripts/setup_stereocrafter.py` and re-run |
| `FileNotFoundError: python` | Wrong venv python path | The bootstrap writes it in-repo; run the bootstrap, or set `STEREOCRAFTER_PYTHON` |
| Self-check fails | The inference entry point can't import | `python <venv>/python inpainting_inference.py --help` inside `third_party/StereoCrafter/` to see the real error |
| `git clone` fails | Network / proxy (common from mainland China) | Set `http(s).proxy`, or clone manually then pass `--repo-dir` |
| `snapshot_download` fails | HF access / network | Re-run the bootstrap; or clone `https://huggingface.co/TencentARC/StereoCrafter` into `models/StereoCrafter/` manually |
| Subprocess non-zero exit | StereoCrafter internal error | Run the inference entry point directly to isolate; check `docs/STEREOCRAFTER_SETUP.md` |

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
