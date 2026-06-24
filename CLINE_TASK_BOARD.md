# CLINE Task Board — VR180 AI Pipeline

**Repo:** `github.com/DreamPortalOS/vr180-ai-pipeline` (migrated 2026-06-24)
**Open PRs:** #9 `feat/hermes-agent` (✅ CI green), #8 `feat/t2-auth-v2` (❌ CI red — fix pushed, waiting for CI re-run)

---

## ✅ Completed

### T-A. Fix PR #8 (T2 auth) CI failure — `passlib` missing ✅
**Result:** 48/48 tests pass. Committed + pushed to feat/t2-auth-v2 (commit 1456851). CI will re-run.
- [x] `git checkout feat/t2-auth-v2 && git pull`
- [x] Add `passlib[bcrypt]` to `requirements.txt` AND `pyproject.toml` deps
- [x] Pin `bcrypt>=4.0.0,<5.0.0` for passlib compatibility (bcrypt 5.x broke passlib)
- [x] `pip install -r requirements.txt` then `pytest tests/test_auth.py tests/test_web_api.py` — **48 passed**

### Hermes Notification Agent (2026-06-24) — PR #9 ✅ CI GREEN, confirmed
- [x] `notifications/feishu.py` — Feishu card builder + `send_vr180_notification()` via `httpx`
- [x] `scripts/watch_and_notify.py` — CLI with `--once`, `--file`, `--watch-dir` modes
- [x] Ruff clean + pre-commit hooks passing; branch pushed → PR #9 (lint+test SUCCESS)

### Housekeeping (2026-06-24)
- [x] Deleted stale local/remote merged branches; pruned stale remote-tracking refs
- [x] Removed `__pycache__/` + `.DS_Store`; archived `OVERNIGHT_RD_REPORT.md` → `docs/archive/`
- [x] Rewrote `README.md`; updated this board

---

## ▶️ NOW: T-B — Fix the 15 test failures on `main`
**Execute T-A → T-B → T-C strictly in order.** After each: tests green → commit → push → then
`git checkout main && git pull` before the next. Do NOT work two tasks in parallel.

### T-B. Fix the 15 test failures on `main` — In Progress
Run `PYTHONPATH=. pytest -q` to reproduce. Two distinct groups:
1. **`test_integrations.py` (13 fail)** — `@pytest.mark.asyncio` reported as *Unknown mark*,
   so the async Kling/Seedance/Veo provider tests are collected but never awaited.
   - [ ] Ensure `pytest-asyncio` is installed AND declared in deps
   - [ ] Set `asyncio_mode = "auto"` under `[tool.pytest.ini_options]` in `pyproject.toml`
   - [ ] Re-run until all 13 pass
2. **`test_phase1_optimizations.py::test_build_ffmpeg_cmd[_h265]` (2 fail)** — caused by the
   **uncommitted WIP** in `pipeline/streaming_pipeline.py` (VideoToolbox hardware-encoder
   support: hw codec map + bitrate-instead-of-CRF, which reordered the ffmpeg arg list).
   - [ ] DEFAULT DECISION: **keep the VideoToolbox feature** (it's a real speedup) but make it
     non-breaking — ensure the default `h264`/`h265` path emits the SAME arg order as before, then
     update `test_build_ffmpeg_cmd` + `_h265` to assert the new list. If that gets messy, fall back to
     `git checkout -- pipeline/streaming_pipeline.py` to drop the WIP entirely.
   - [ ] `pytest tests/test_phase1_optimizations.py` green

---

## ⏭️  Next After T-B

### T-C. Fix the broken VR180 metadata pipeline (found 2026-06-24)
**Why:** A pipeline run produced an SBS file with **zero** spherical/stereo metadata.
`pipeline/spherical_injector.py::inject_spherical_metadata` tried Google `spatialmedia`
(not installed) then fell back to `ffmpeg -metadata:s:v spherical-video=...`, which ffmpeg
silently drops — no `sv3d`/`st3d` box is written, so headsets can't recognize VR180.
The hand-rolled ISOBMFF builders in `spherical_injector.py` (`_build_svv3d`/`_build_svproj`/
`_build_svmi`) and the EOF-append boxes in `spatial_converter.py` use **invalid box names**
and are **never wired in** — dead, misleading code.
**Fix already verified manually:** installing Google spatial-media
(`pip install "git+https://github.com/google/spatial-media.git#egg=spatialmedia"`) makes the
existing `python3 -m spatialmedia -i -2 -s left-right -p equirectangular` path inject correct
`sv3d`+`st3d` (Stereo Mode 2) boxes. Pipeline already produces correct 7680×1920 SBS frames.
**Action:**
- [ ] Add the spatial-media git dependency to `requirements.txt` (+ note in README)
- [ ] Make the ffmpeg fallback FAIL LOUDLY (raise) instead of pretending success
- [ ] Delete the dead/incorrect ISOBMFF builders in `spherical_injector.py` and the EOF-append
      blocks in `spatial_converter.py` (or replace with a real, tested implementation)
- [ ] Add a regression test: run injector on a tiny clip, assert `sv3d`+`st3d` bytes present
      and `spatialmedia <file>` reads back `Stereo Mode: 2`

### T-D. Phase Q — Quality / Stabilization (needs remote GPU) — backlog
DepthCrafter (temporal depth), StereoCrafter (hi-fi stereo), edge AI outpainting, 8K upscale.

### T-E. Phase C — Frontend workflow platform — backlog
Web UI (ComfyUI-style): prompt → generate → convert → preview/download. Tiers: Convert / Generate / Studio.

### T-F. Commercialization — backlog
Pricing, API tiers, user management.
