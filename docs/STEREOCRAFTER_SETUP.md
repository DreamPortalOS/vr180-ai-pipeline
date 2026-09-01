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
     actually imports.  `transformers`/`diffusers` are **pinned** to the
     combo upstream tested (`transformers==4.42.3` / `diffusers==0.29.2`) —
     this is the safetensors load fix; see §6.  (We deliberately do **not**
     run `pip install -e .` against the upstream pyproject — it would bump
     the torch pins or add demo-only deps.)
3. Downloads `TencentARC/StereoCrafter` weights into
   `models/StereoCrafter/` via `huggingface_hub.snapshot_download`
   (skipped if already present).
4. Pre-downloads the **SVD base model**
   (`stabilityai/stable-video-diffusion-img2vid-xt-1-1`) into
   `models/svd-img2vid-xt-1-1/`, **fp16 safetensors only (~5 GB, not the
   full ~10 GB repo)**.  This is a **gated** HF repo — the local HF token
   is read automatically and the account must have **accepted the license**
   (see §2.1).  If the account is not yet authorized, the step fails with a
   clear error naming the application page rather than a bare `403 OSError`.
   Loading the SVD base from this **local** dir is also what fixes the
   `pytorch_model.fp16.bin` load failure (§6, issue #155).  Skip with
   `--skip-svd` (NOT recommended — see §6).
5. Runs a self-check (`<inference_entry_point> --help`) so a broken env
   is caught early.

Created layout (everything is gitignored):

```
third_party/StereoCrafter/      ← TencentARC/StereoCrafter checkout (+ its own .venv)
models/StereoCrafter/           ← TencentARC/StereoCrafter weights
models/svd-img2vid-xt-1-1/      ← SVD base (fp16 safetensors, gated HF repo, ~5 GB)
```

### Bootstrap options

```bash
python scripts/setup_stereocrafter.py --repo-dir D:/StereoCrafter   # use an existing checkout
python scripts/setup_stereocrafter.py --skip-model                  # weights already downloaded
python scripts/setup_stereocrafter.py --skip-svd                     # SVD base already downloaded / defer to first run
python scripts/setup_stereocrafter.py --svd-dir D:/svd               # custom SVD target dir
python scripts/setup_stereocrafter.py --hf-token hf_...             # explicit HF token (else auto-read)
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
| Disk      | ~20 GB (checkout + weights + venv) **+ ~5 GB for the SVD base** — pre-downloaded (fp16 safetensors only) by the bootstrap into `models/svd-img2vid-xt-1-1/`; a **local** snapshot is the issue #155 safetensors load fix (§6); skip with `--skip-svd` to defer to the first inference run (NOT recommended) |
| HF access | An Hugging Face account token with **accepted license** for the gated SVD repo [`stabilityai/stable-video-diffusion-img2vid-xt-1-1`](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1) (see §2.1) |
| Network   | git + HF download (one-time); the bootstrap prints proxy hints if `git clone` fails |

### 2.1 SVD base model — gated HF repo (must accept license)

The StereoCrafter Stage-2 inpainting stage uses the SVD base model
[`stabilityai/stable-video-diffusion-img2vid-xt-1-1`](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1)
as its `--pre_trained_path`.  This is a **gated Hugging Face repo** —
the model's license must be accepted before it can be downloaded.

If the account behind your HF token has not accepted the license, the
first attempt to download (either by the bootstrap or by diffusers at
runtime) fails with a bare `403` / `OSError: ... gated repo ...` that
says nothing useful:

```
OSError: You are trying to access a gated repo.
  Cannot access gated repo for url
  https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1/resolve/main/...
  403 Client Error. Access to model ... is restricted and you are not in the authorized list.
```

**Fix (one-time):**

1. Open <https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1>
   in a browser, sign in with the **same** HF account whose token is on
   this machine, and accept the license agreement.  Approval is usually
   instant.
2. Re-run the bootstrap:
   ```bash
   python scripts/setup_stereocrafter.py
   ```
   The bootstrap now reads your local HF token automatically
   (`HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` env var, or
   `~/.cache/huggingface/token` / `huggingface-cli login`), passes it to
   `snapshot_download`, and pre-downloads the SVD base (fp16 safetensors
   only, ~5 GB) into `models/svd-img2vid-xt-1-1/`.  **No re-login is
   needed** — the token is already on disk; only the license acceptance is
   required.
3. If you pass a token explicitly:
   ```bash
   python scripts/setup_stereocrafter.py --hf-token hf_xxx
   ```

**If you decline to accept / cannot get authorization:** the SVD base
cannot be downloaded legally through any other channel
(non-official mirror weights are out of scope — see issue #150).  Your
only option is to skip the pre-download and rely on diffusers at runtime:

```bash
python scripts/setup_stereocrafter.py --skip-svd
```

When authorization fails at bootstrap time, the script does **not** print
a cryptic `403` — it prints the application page and the exact steps
above, so the fix is actionable on its face.

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
| `pre_trained_path`      | `STEREOCRAFTER_SVD_PATH`       | `models/svd-img2vid-xt-1-1` *(if exists)* — SVD base (Stage 2 `--pre_trained_path`); else the HF id `stabilityai/stable-video-diffusion-img2vid-xt-1-1` (remote resolution — NOT recommended, see §6) |

> **SVD base (~5 GB, fp16 safetensors) — pre-downloaded by the bootstrap.**
> The bootstrap pre-downloads the SVD base into `models/svd-img2vid-xt-1-1/`
> by default (it is a **gated** HF repo — see §2.1 for the one-time
> license-acceptance step).  Loading the base from that **local** dir is the
> fix for issue #155 (the `pytorch_model.fp16.bin` safetensors load failure —
> see §6): a local path lets transformers resolve `model.fp16.safetensors`
> via `os.path.isfile` instead of the error-prone remote `cached_file`.  If
> you ran the bootstrap with `--skip-svd` and no local copy exists, the
> backend falls back to the HF model id (remote resolution on the first
> inference run) and logs a WARNING.  To front-load the local copy later,
> re-run `python scripts/setup_stereocrafter.py` (the SVD step is idempotent),
> or set `STEREOCRAFTER_SVD_PATH` to an existing local snapshot.  A
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
| `snapshot_download` fails (StereoCrafter weights) | HF access / network | Re-run the bootstrap; or clone `https://huggingface.co/TencentARC/StereoCrafter` into `models/StereoCrafter/` manually |
| SVD download fails with `gated repo` / `403` / "not in the authorized list" | The HF account behind the local token has **not accepted the SVD model's license** — it is a gated repo (issue #150). The bootstrap prints the application page and steps automatically | Open <https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1>, sign in with the same HF account, accept the license (usually instant), then re-run `python scripts/setup_stereocrafter.py`. See §2.1 |
| SVD download fails, no token found | No HF token on disk (`~/.cache/huggingface/token`) and no `HF_TOKEN` env var | Run `huggingface-cli login` once, or pass `--hf-token hf_xxx` (see §2.1) |
| `Can't load the model ... pytorch_model.fp16.bin` (Stage 2 startup) | The SVD base is being resolved **remotely** (HF id), and the remote `cached_file` path tripped issue #155 — the repo ships ONLY safetensors, no `.bin`, so the remote-resolution fallback raises the misleading `.bin` error. **Not** an auth problem (that is the 403/gated row above). | Confirm the local snapshot exists at `models/svd-img2vid-xt-1-1/image_encoder/model.fp16.safetensors`; if not, re-run `python scripts/setup_stereocrafter.py` so the fp16 safetensors are pre-downloaded and the backend loads the local dir (it logs a WARNING if it falls back to the HF id).  If the snapshot is present but it still fails, the dedicated venv has an older `transformers` — re-run the bootstrap with `--skip-model --skip-svd` to reinstall the pinned `transformers==4.42.3`.  See §6a below. |
| Subprocess non-zero exit | StereoCrafter internal error | Run `inpainting_inference.py` directly inside `third_party/StereoCrafter/` to isolate the failure; check its stderr |
| `SBS output not found` | Stage 2 succeeded but did not write `<name>_sbs.mp4` where expected | Confirm `--save_dir` is writable and the inpainting step completed; check stderr of Stage 2 |

### 6a. The SVD base must load from a local directory (issue #155)

**Symptom:** Stage 2 startup dies with

```
OSError: Can't load the model for 'stabilityai/stable-video-diffusion-img2vid-xt-1-1'.
  ... make sure ... is the correct path to a directory containing a file named pytorch_model.fp16.bin.
  (raised from transformers/modeling_utils.py::_get_resolved_checkpoint_files)
```

**Root cause (lead-verified):** the HF repo ships **only safetensors**
(`image_encoder/model.fp16.safetensors`, `unet/diffusion_pytorch_model.fp16.safetensors`,
`vae/...`) — there is no `.bin` at all.  But `inpainting_inference.py` loads the
image_encoder / VAE with `variant="fp16"` and **no** `use_safetensors=True` flag.
When the path is an **HF repo id**, transformers resolves the weight file
remotely via `cached_file`; any non-`OSError` raised inside that call (a
transient auth / network glitch) gets re-wrapped as the generic "make sure
... pytorch_model.fp16.bin" error — a misleading tail that names `WEIGHTS_NAME`
(`pytorch_model.bin`) even though the loader actually tried `model.fp16.safetensors`
first.  So the error is a **load-path / version** problem, **not** an auth
problem (the 403 was already resolved in issue #150).

**The fix (two layers, both shipped here):**

1. **Local snapshot (the real fix).**  When `--pre_trained_path` points at a
   **local directory**, transformers' `_get_resolved_checkpoint_files` takes the
   local-folder branch, which checks `os.path.isfile(.../model.fp16.safetensors)`
   **first** (whenever `use_safetensors is not False`, the default `None`) — no
   `cached_file`, no remote resolution, no misleading wrap.  The bootstrap now
   **pre-downloads** that local snapshot by default (fp16 safetensors + configs
   only, ≈5 GB instead of the full ~10 GB repo) into
   `models/svd-img2vid-xt-1-1/`, and `pipeline.stereo_crafter` picks that local
   dir up automatically (issue #147 precedence: existing local dir > HF id).
2. **Pinned loader versions (defence-in-depth).**  `RUNTIME_DEPS` pins
   `transformers==4.42.3` / `diffusers==0.29.2` — the exact pair upstream
   `TencentARC/StereoCrafter`'s `requirements.txt` tested against.  This
   guarantees the local-folder safetensors-first resolution path is the one
   upstream validated; an unpinned `transformers` could drift to a 5.x whose API
   changes break the vendored `StableVideoDiffusionInpaintingPipeline`, or
   (older) a line that defaulted to `.bin`.  Re-run the bootstrap to pick up the
   pins (the venv is re-created/updated in Step 2).

**If you still hit it:** confirm
`models/svd-img2vid-xt-1-1/image_encoder/model.fp16.safetensors` exists (the
bootstrap's "already present" check keys on exactly those three fp16 safetensors).
If it does but Stage 2 still fails, the dedicated venv has an older `transformers`
— re-run the bootstrap with `--skip-model --skip-svd` (it reinstalls the pinned
`transformers==4.42.3`).  Do **not** answer this with `--skip-svd`: the local path
*is* the answer.

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
