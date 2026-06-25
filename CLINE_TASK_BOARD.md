# CLINE Task Board — VR180 AI Pipeline

**Repo:** `github.com/DreamPortalOS/vr180-ai-pipeline`
**FOCUS (2026-06-24):** the **2D → immersive video conversion workflow**, optimized for **clarity/resolution**.
Now **two delivery routes** off one shared pipeline (see `docs/SOLUTION_ARCHITECTURE.md`):
**Route 1 = 球幕/Fulldome (mono, no glasses)** · **Route 2 = VR180 (stereo, headset)**.
Platform layer archived on branch `archive/platform-layer` — do NOT rebuild it now.

**Coordination:** lead (Claude) uses isolated `git worktree`s; cline keeps `.clinerules` (branch from latest `main`, serial, pre-commit green, never `--no-verify`).

---

## 📮 ACTIVE DISPATCH — lead → cline (execute IN ORDER, one PR each, serial)

> **Local env is already provisioned by lead — do NOT reinstall torch.** The repo has a working `.venv`
> (Python 3.12, `torch 2.6.0+cu124` **CUDA** build verified on an RTX 4070 SUPER, `cuda=True`), ffmpeg is on
> PATH, and `Depth-Anything-V2-Small` is cached. First-run only: `.venv\Scripts\activate` then
> `pip install pre-commit && pre-commit install`. Branch each task from **latest `main`**. Never `--no-verify`.

### ▶ DISPATCH-1 = R-5 Fulldome renderer  (DO THIS FIRST — Route 1, active)
Branch `feat/R5-fulldome-mapper` from latest `main`.
- **New `pipeline/fulldome_mapper.py`** — `class FulldomeMapper`:
  - `__init__(self, dome_fov=180, coverage_h_fov=120, coverage_v_fov=None, output_size=4096, codec="h264", crf=18)`
  - `convert(self, input_path, output_path) -> str`: **ONE** ffmpeg pass over the whole video (NOT per-frame):
    `v360=input=flat:output=fisheye:ih_fov=<coverage_h_fov>:iv_fov=<coverage_v_fov>:h_fov=<dome_fov>:v_fov=<dome_fov>:w=<output_size>:h=<output_size>`
    When `coverage_v_fov is None`, derive it from the source aspect ratio. subprocess MUST use a list (no `shell=True`).
  - Validated recipe (lead ran it; produces a valid square-fisheye domemaster):
    `ffmpeg -i src.mp4 -vf "v360=input=flat:output=fisheye:ih_fov=120:iv_fov=75:h_fov=180:v_fov=180:w=2048:h=2048" out.mp4`
- **Modify `scripts/run_pipeline.py`**: add `--projection {vr180,fulldome}` (default `vr180`). When `fulldome`,
  **skip depth/stereo/equirect/metadata entirely** and call `FulldomeMapper`. Use `patch_file` (run_pipeline.py > 150 lines).
- **New `tests/test_fulldome_mapper.py`**: output square (w==h), fisheye, contains **NO** `sv3d`/`st3d` boxes.
- Acceptance: `--projection fulldome` → square fisheye domemaster; `--projection vr180` unchanged; `ruff check .` + `ruff format --check .` + `pytest` green.
- Do **NOT** touch depth/stereo/VR180 logic; do **NOT** add spherical metadata to fulldome.

### ▶ DISPATCH-2 = R-2 Comfort/geometry defaults  (after DISPATCH-1 merges)
Branch `feat/R2-comfort-defaults` from latest `main`. Bake validated defaults into the pipeline: square per-eye (1:1),
`max_disparity` default ≈ 0.02, a sensible `src_hfov` default with a docstring tradeoff note. Edit
`pipeline/equirectangular_mapper.py` + `pipeline/stereo_renderer.py` defaults; add tests asserting square-SBS layout
+ `sv3d`/`st3d` still injected. ruff + pytest green. Don't touch the fulldome branch.

### ▶ DISPATCH-3 = R-3 spatial_converter test coverage  (after DISPATCH-2 merges)
Branch `feat/R3-spatial-converter-tests` from latest `main`. New `tests/test_spatial_converter.py` covering the core
SBS / MV-HEVC conversion paths in `pipeline/spatial_converter.py` (coverage lost when `test_phase4` was archived).
**Tests only** — do NOT change the module under test. ruff + pytest green.

> Status (2026-06-25): **merged to main** — R-5 #17 · R-2 #22 · R-3 #23 · P-1 #24 · P-2 #25; lead fixes #19 (CUDA total_memory) + #21 (ffmpeg-stderr deadlock); plan/PRD #20. **NEXT = R-1 (DISPATCH-4)** — cline on `feat/R1-seedvr2-upscaler`.

### ▶ DISPATCH-4 = R-1 SeedVR2 source upscaler  (parallel-OK — branch from R-5's branch)
**Branch from `feat/R5-fulldome-mapper` (R-5's branch), NOT `main`.** R-1 and R-5 both edit
`scripts/run_pipeline.py`; per `.clinerules §6` branch the dependent task from the dependency's branch so they
develop in parallel without conflict (after #17 merges, R-1's PR shows only its own delta).
**The local SeedVR2 model is NOT deployed yet** (separate ComfyUI install — see `docs/SEEDVR2_SETUP.md`). Build the
wrapper + CLI + **mock** unit test only; do NOT download/run the real model — CI stays green via mock.
- **New `pipeline/video_upscaler.py`** — `class SeedVR2Upscaler`: CUDA-only (clear error on Mac/CPU);
  `batch_size` must be `4n+1` (1,5,9,13…) — validate/raise; `upscale(input_path, output_path, factor)` with an
  injectable/placeholder backend (real inference wired once the model is deployed).
- **`scripts/run_pipeline.py`** (patch — file > 150 lines): add `--video-upscale {none,seedvr2}` (default `none`) +
  `--video-upscale-factor`; when `seedvr2`, run as **Stage 0 before depth**; `none` = zero effect on the current flow.
- **New `tests/test_video_upscaler.py`** (mock model/subprocess): non-CUDA raises a clear error; `4n+1` validation;
  `--video-upscale none` leaves the pipeline unchanged.
- Acceptance: ruff + pytest green on Mac/CPU/CI (no CUDA or model needed). Don't touch other conversion modules.

---

## 🔁 SHARED (benefits both routes)

### R-1. SeedVR2 source super-resolution pre-stage  ← TOP PRIORITY
Root cause of "completely unclear" = the **720p source**. SeedVR2 (ByteDance, ICLR2026) SOTA temporal video SR.
Lead is delivering the 4070S deployment (`docs/SEEDVR2_SETUP.md`); cline wraps it into the pipeline:
- [ ] `pipeline/video_upscaler.py` — `SeedVR2Upscaler`; CUDA-only, clear Mac error; `batch_size`=`4n+1`.
- [ ] `scripts/run_pipeline.py`: `--video-upscale {none,seedvr2}` + `--video-upscale-factor`, Stage 0 before depth. Default `none`.
- [ ] Mock-based unit test (CI green on Mac, no CUDA/model).

### R-2. Bake geometry/comfort as pipeline DEFAULTS (validated in v2)
- [ ] Square per-eye (1:1) · `max_disparity` default ~0.02 · sensible `src_hfov` + doc tradeoff · test square-SBS + `sv3d`/`st3d`.

### R-3. Re-add `spatial_converter` test coverage (lost with archived test_phase4).

---

## 🟢 ROUTE 1 — Fulldome / 球幕影院 (mono, no glasses) — ▶ ACTIVE (user chose Route-1-first 2026-06-24)
No stereo → no ghosting, no nausea, max sharpness. Small build, reuses v360 machinery.
**Approach validated by lead** — single fast v360 pass (~4s for 10s clip, whole video at once, NOT per-frame):
```
ffmpeg -i src.mp4 -vf "v360=input=flat:output=fisheye:ih_fov=120:iv_fov=75:h_fov=180:v_fov=180:w=2048:h=2048" out.mp4
```
Output is a valid domemaster. `ih_fov/iv_fov` = source coverage → controls how much of the dome the flat clip fills
(low = a screen-like patch, high = fuller dome but more stretch). Corners outside the 180° circle are black (correct).

### R-5. Fulldome renderer (TOP — build now)
- [ ] `pipeline/fulldome_mapper.py` — wrap the validated v360 recipe. **Mono, no depth.** Params:
  `dome_fov` (default 180, up to 220), `coverage`/`ih_fov`+`iv_fov` (how much the source fills the dome, default
  ih_fov=120 iv_fov auto from aspect), output size (default 4096², configurable). **Single ffmpeg pass over the whole video.**
- [ ] `scripts/run_pipeline.py`: `--projection {vr180,fulldome}` (default vr180). Fulldome skips depth/stereo/spherical-metadata entirely.
- [ ] Test: output square, fisheye, no `sv3d`/`st3d`. Doc: preview in a fisheye-aware player; dome projector/warp questions user-pending.
- Note: pairs with SeedVR2 (R-1) for sharpness; the soft look is the 720p source, not the projection.

---

## 🔵 ROUTE 2 — VR180 stereo (headset) — harder, differentiator (mostly GPU, runs on 4070S)
Fixes the "重影没对上" + "抬头见边界". Backlog until SeedVR2 + fulldome land.
- StereoCrafter (clean disocclusion, no smear) · DepthCrafter (temporal depth, no shimmer) · 180° outpainting (fill FOV without stretch).

---

## ⚙️ Later
### R-4. Equirect performance — one batched `v360` pass instead of 2 ffmpeg subprocesses/frame (~10× faster).

## ⏸️ Paused (archived on `archive/platform-layer`)
Auth, Hermes/Feishu, DB, VideoGen integrations, web API, frontend, quota, commercialization. Revisit after conversion quality is nailed.
