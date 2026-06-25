# CLINE Task Board — VR180 AI Pipeline

**Repo:** `github.com/DreamPortalOS/vr180-ai-pipeline`
**FOCUS (2026-06-24):** the **2D → immersive video conversion workflow**, optimized for **clarity/resolution**.
Now **two delivery routes** off one shared pipeline (see `docs/SOLUTION_ARCHITECTURE.md`):
**Route 1 = 球幕/Fulldome (mono, no glasses)** · **Route 2 = VR180 (stereo, headset)**.
Platform layer archived on branch `archive/platform-layer` — do NOT rebuild it now.

**Coordination:** lead (Claude) uses isolated `git worktree`s; cline keeps `.clinerules` (branch from latest `main`, serial, pre-commit green, never `--no-verify`).

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

## 🟢 ROUTE 1 — Fulldome / 球幕影院 (mono, no glasses) — recommended next
No stereo → no ghosting, no nausea, max sharpness. Small build, reuses v360 machinery.

### R-5. Fulldome renderer
- [ ] `pipeline/fulldome_mapper.py` — map upscaled 2D → **domemaster** (circular fisheye, azimuthal-equidistant,
  square, 180° default / configurable to 200°+) via ffmpeg `v360=input=flat:output=fisheye`. **Mono, no depth.**
- [ ] `scripts/run_pipeline.py`: `--projection {vr180,fulldome}` (default vr180). Fulldome skips depth/stereo/spherical-metadata.
- [ ] Output square 4K² (configurable to 8K²); h265. Test: output is square, fisheye, no `sv3d`/`st3d`.
- [ ] Doc: how to preview (fisheye-aware player) + note dome-software/projector questions are user-pending.

---

## 🔵 ROUTE 2 — VR180 stereo (headset) — harder, differentiator (mostly GPU, runs on 4070S)
Fixes the "重影没对上" + "抬头见边界". Backlog until SeedVR2 + fulldome land.
- StereoCrafter (clean disocclusion, no smear) · DepthCrafter (temporal depth, no shimmer) · 180° outpainting (fill FOV without stretch).

---

## ⚙️ Later
### R-4. Equirect performance — one batched `v360` pass instead of 2 ffmpeg subprocesses/frame (~10× faster).

## ⏸️ Paused (archived on `archive/platform-layer`)
Auth, Hermes/Feishu, DB, VideoGen integrations, web API, frontend, quota, commercialization. Revisit after conversion quality is nailed.
