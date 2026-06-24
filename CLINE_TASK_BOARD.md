# CLINE Task Board — VR180 AI Pipeline

**Current branch:** `main` (housekeeping + hermes-agent will be on own branch)

---

## ✅ Completed

### Housekeeping (2026-06-24)
- [x] Deleted stale local merged branches: `feat/t1-database`, `feat/t2-api-key-auth`, `feat/t2-auth-v2` (local only)
- [x] Deleted stale remote-tracking refs (`feat/prompt-builder-pr2`, `feat/t3-videogen` pruned)
- [x] Deleted stale remote branches: `chore/fix-hooks-and-t2-plan`, `chore/task-board-update`, `feat/t1-database`, `feat/t2-api-key-auth`
- [x] Removed `__pycache__/` and `.DS_Store` (excl `.venv`)
- [x] Archived `OVERNIGHT_RD_REPORT.md` → `docs/archive/`
- [x] Rewrote `README.md` as comprehensive project entry point with current state

### Hermes Notification Agent (2026-06-24)
- [x] `notifications/__init__.py` — package init
- [x] `notifications/feishu.py` — Feishu card builder + `send_vr180_notification()` via `httpx`
- [x] `scripts/watch_and_notify.py` — CLI with `--once`, `--file`, `--watch-dir` modes
- [x] Ruff clean (lint + format passed)

---

## ⏳ In Progress

### [Hermes] Feishu Notification Agent — PR
- [ ] Create branch `feat/hermes-agent`
- [ ] git add + commit
- [ ] git push + open PR
- [ ] Wait for CI green before marking done

---

## 📋 Next Development Priorities (ordered)

### 1. Fix T2 Auth PR #8 CI test failures
**Status:** ⏸️ Blocked — waiting for GitHub Actions logs to diagnose `pytest` failures
**Action:** Once CI logs available, read failures, fix code, push to existing PR
**See:** https://github.com/DreamPortalOS/vr180-ai-pipeline/pull/8

### 2. Fix pre-existing test failures
**Known broken tests:**
- `test_equirectangular_mapper.py` — shape assertion fails after direction fix
- `test_spherical_injector.py` — import error (`spherical_injector` refactored)
**Action:** branch from main, fix tests to match current code, push PR

### 3. Phase Q — Quality / Stabilization (needs remote GPU)
- DepthCrafter (temporal-consistent depth)
- StereoCrafter (high-fidelity stereo)
- Edge AI outpainting
- 8K upscale as pre-stage
**Dependency:** Remote GPU access (rental/sponsor)

### 4. Phase C — Frontend Workflow Platform
- Web UI (like ComfyUI) for prompt → generate → convert → preview/download
- Three-tier product: Convert / Generate / Studio

### 5. Commercialization
- Pricing, API tiers, user management
