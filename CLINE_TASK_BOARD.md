# 🎭 VR180 Studio — Overnight All-Phase Task Board

## Phase 0 — Bug Fixes & Core R&D

- [x] Task 1.1: Smart SBS Input Detection (auto-detect 4:1 SBS ratio, skip depth/stereo stages)
- [x] Task 1.2: Advanced VR180 Orientation Matrix Harness (`pipeline/research/orientation_matrix.py`)

## Phase 3 — AI Temporal Outpainting

- [x] Optical flow-based temporal outpainting (`pipeline/research/ai_outpainter.py`)
- [x] Boundary mask generation, flow computation, frame warping, Poisson blending
- [x] Quality metrics (PSNR/SSIM), multi-frame sequence processing

## Phase 3.5 — Web Infrastructure & API

- [x] FastAPI REST API (`web/app.py`) — health, task CRUD, file upload/download, results, quota
- [x] Task store (`web/task_store.py`) — in-memory CRUD with filtering & pagination
- [x] API schemas (`web/schemas.py`) — Pydantic models for all endpoints
- [x] Web API tests (`tests/test_web_api.py`) — 39 tests

## Phase 4 — Production Features

- [x] Quota system (`web/quota.py`) — user limits, usage tracking, persistence
- [x] Result persistence (`web/storage.py`) — file storage, metadata, cleanup
- [x] Spatial video converter (`pipeline/spatial_converter.py`) — MV-HEVC, SBS metadata injection
- [x] Phase 4 tests (`tests/test_phase4.py`) — 43 tests
- [x] Frontend SPA — `web/static/index.html`, `styles.css`, `app.js`
- [x] Wire frontend into FastAPI (static mount, SPA serving, v1 API endpoints)
- [x] Install python-multipart for file upload support
- [x] Full regression test: **195/195 passing**
- [ ] Commit & Push all Phase 4 work to GitHub