# CLINE TASK BOARD — Overnight All-Phase Epic

## Status Legend
- [ ] Not Started
- [x] Completed
- [~] In Progress

---

## PHASE 0: PREVIOUS PHASE 1 (Already Complete)
- [x] Device Detection (detect_best_device, get_device_info, resolve_device)
- [x] Streaming Pipeline (StreamingPipeline class, O(1) memory)
- [x] Tiled Upscaling (upscale_tiled with Gaussian/Linear blending)

---

## PHASE 2: VR180 ADVANCED CORRECTION & SBS DETECTION

### Task 1.1: Smart SBS Input Detection in run_pipeline.py
- [x] Implement auto-detection: if input width/height ratio ≈ 4:1 (e.g. 7680×1920), treat as existing SBS
- [x] When SBS detected, skip Stage 1 (Depth) and Stage 2 (Stereo), branch directly to Stage 3 (Equirect)
- [x] Add --force-sbs flag to manually override detection
- [x] Add tests for SBS detection logic — 7/7 passed

### Task 1.2: Advanced VR180 Orientation Matrix
- [x] Create pipeline/research/orientation_matrix.py
- [x] Programmatically test cv2.flip(img, 0), cv2.flip(img, 1), cv2.flip(img, -1) combinations
- [x] Generate ffmpeg transpose filter variations (0,1,2,3)
- [x] Build matrix of all orientation combinations
- [x] Output diagnostic video grid showing all variations — 17/17 tests passed

---

## PHASE 3: AI TEMPORAL OUTPAINTING (R&D)

### Task 3.1: Temporal Outpainter Module
- [x] Create pipeline/research/temporal_outpainter.py
- [x] Implement frame boundary detection for missing VR180 regions
- [x] Build optical flow-based temporal consistency engine
- [x] Implement iterative outpainting with convergence detection
- [x] Add quality metrics (SSIM, PSNR) for outpainting validation

### Task 3.2: Outpainter Integration Test
- [x] Create test harness for temporal outpainter
- [x] Generate synthetic test frames with known missing regions
- [x] Validate temporal consistency across frame sequences — 19/19 tests passed

---

## PHASE 4: WEB INFRASTRUCTURE & API

### Task 4.1: FastAPI Application & Task Store
- [x] Create web/__init__.py (package init)
- [x] Create web/task_store.py — Thread-safe in-memory task store with CRUD, status lifecycle, cancellation
- [x] Create web/schemas.py — Pydantic models for all request/response types
- [x] Create web/app.py — FastAPI app with health, task CRUD, list/pagination, cancel endpoints
- [x] Add fastapi, uvicorn, httpx to requirements.txt

### Task 4.2: API Tests
- [x] Create tests/test_web_api.py — 30 endpoint tests covering:
  - Health check (2 tests)
  - Task create with fields/metadata/validation (5 tests)
  - Task get by ID + 404 (2 tests)
  - Task list empty/create/filter/pagination (4 tests)
  - Task update status/completed/failed/404 (4 tests)
  - Task delete + 404 (2 tests)
  - Task cancel queued/processing/404 (3 tests)
  - TaskStore unit tests: create, list, lifecycle, cancel, delete, count, serialization (8 tests)
- [x] All 30 tests passed

---

## VERIFICATION
- [x] All new modules import successfully
- [x] All new tests pass (30/30 web API + 7 SBS + 17 orientation + 19 outpainter)
- [x] Full test suite passes (152/152, 0 regressions) — fixed 3 pre-existing failures
- [x] CLINE_TASK_BOARD.md kept updated throughout