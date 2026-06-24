# CLINE Task Board — VR180 AI Pipeline

**Repo:** `github.com/DreamPortalOS/vr180-ai-pipeline`
**FOCUS (2026-06-24):** ONE thing — the **2D → VR180 conversion workflow**, and specifically **resolution/clarity**.
Everything else is paused. Platform layer (`web/ db/ integrations/ notifications/ workers/ alembic/` + auth/quota/frontend)
is **archived** on branch `archive/platform-layer` — recover from there if needed, do NOT rebuild it now.

**Coordination:** lead (Claude) and cline share this working tree → lead now uses isolated `git worktree`s.
cline: keep to `.clinerules` (branch from latest `main`, serial, pre-commit green, never `--no-verify`).

---

## 🎯 Active queue (resolution first)

### R-1. SeedVR2 source super-resolution pre-stage  ← TOP PRIORITY
**Why:** root cause of "completely unclear" is the **720p source**. Mapping 720p into 2880²/eye upscales ~3–4×
with no real detail. SeedVR2 (ByteDance, ICLR2026) is the 2026 SOTA video SR — temporal-aware, best on
short AI-generated compressed clips (exactly our input). Runs on the user's **RTX 4070S / 3060 (12GB)**:
3B model needs ≥8GB, 12GB comfortable; GGUF quant available.

**Build (code on Mac, model runs on the GPU box):**
- [ ] `pipeline/video_upscaler.py` — `SeedVR2Upscaler` wrapping SeedVR2 inference
  (reference: `ByteDance-Seed/SeedVR` repo, or Cog wrapper `zsxkib/cog-ByteDance-Seed-SeedVR2`,
  or ComfyUI node `numz/ComfyUI-SeedVR2_VideoUpscaler`). Subprocess or import the inference entrypoint.
  - **CRITICAL:** `batch_size` must be `4n+1` (1,5,9,13,…) for temporal consistency — enforce/round.
  - Detect CUDA; on non-CUDA (Mac) raise a clear error: "SeedVR2 needs a CUDA GPU — run this stage on the 4070S".
- [ ] Wire into `scripts/run_pipeline.py`: `--video-upscale {none,seedvr2}` + `--video-upscale-factor {2,4}`,
  running on the SOURCE video BEFORE depth (Stage 0). Default `none`.
- [ ] Mock-based unit test (no model/CUDA needed) so CI stays green on Mac.
- [ ] `docs/SEEDVR2_SETUP.md` — exact 4070S setup: clone repo, download `seedvr2_ema_3b_fp16` weights, CUDA deps, run cmd.
**Acceptance:** `pytest` green on Mac (mock); a documented one-liner upscales the 720p source →
~2880p on the 4070S, then `run_pipeline` produces a visibly sharper VR180.

### R-2. Bake M1a geometry/comfort as pipeline DEFAULTS
Validated interactively (delivered v2). Make them defaults, not ad-hoc flags:
- [ ] Per-eye **square** (1:1) output (e.g. 2880²/eye → SBS 5760×2880) — fixes the vertically-squished look.
- [ ] Comfortable `max_disparity` default ~0.02 (was 0.05) so eyes fuse — reduces ghosting/double-image.
- [ ] Sensible `src_hfov` default + doc the fill-vs-stretch tradeoff.
- [ ] Test asserting default output is square-per-eye SBS with valid `sv3d`/`st3d`.

### R-3. Re-add `spatial_converter` test coverage
`tests/test_phase4.py` (which covered `pipeline/spatial_converter.py`) was archived with the platform layer.
- [ ] Add focused `tests/test_spatial_converter.py` (MV-HEVC / SBS metadata box assertions).

### R-4. Equirect performance (after R-1/R-2)
Mapper spawns 2 ffmpeg subprocesses/frame (~1.5–3s/frame). Move to one batched `v360` pass or in-process mapping.

---

## ⏸️ Paused (archived on `archive/platform-layer`)
Auth (PR #8), Hermes/Feishu (PR #9), DB persistence, VideoGen integrations (Kling/Seedance/Veo), web API,
frontend, quota, commercialization. Re-introduce only after conversion quality is nailed.

## 🧊 Backlog (quality, needs research/GPU — see `docs/ROADMAP.md` M1c/M1d)
DepthCrafter (temporal depth, less shimmer) · StereoCrafter (clean disocclusions, less ghosting) · 180° generative outpainting (fill FOV without stretch, kills the "boundary when looking up").
