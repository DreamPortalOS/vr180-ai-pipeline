"""FastAPI application for VR180 Studio.

Provides REST API endpoints for:
- Health check
- Task CRUD (create, read, list, update, delete, cancel)
- Quota management
- Result storage
- File upload/download
- Frontend SPA serving

Usage:
    uvicorn web.app:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from celery.result import AsyncResult
from db.engine import init_db
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from integrations.base import GenerationParams
from integrations.factory import get_provider, list_providers
from workers.celery_app import app as celery_app
from workers.convert_tasks import convert_to_vr180

from pipeline.prompt_builder import wrap_prompt_for_vr180
from web.schemas import (
    ErrorResponse,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    TaskCreateRequest,
    TaskListResponse,
    TaskResponse,
    TaskStatusEnum,
    TaskUpdateRequest,
)
from web.task_store_db import TaskStatus, TaskStoreDB

log = logging.getLogger("vr180-api")

# Global task store instance (DB-backed)
task_store = TaskStoreDB()

# Application start time for uptime calculation
_start_time: float = 0.0

# Paths
_STATIC_DIR = Path(__file__).parent / "static"
_UPLOAD_DIR = Path("data/uploads")
_OUTPUT_DIR = Path("data/outputs")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    global _start_time
    _start_time = time.monotonic()
    _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    init_db()
    log.info("VR180 Studio API started")
    yield
    log.info("VR180 Studio API shutting down")


app = FastAPI(
    title="VR180 Studio API",
    description="Production-grade VR180 video conversion pipeline API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


# ─── Frontend SPA ─────────────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_frontend():
    """Serve the frontend SPA."""
    index_path = _STATIC_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>VR180 Studio</h1><p>Frontend not built yet.</p>")


# ─── Health ───────────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        version="1.0.0",
        uptime_seconds=time.monotonic() - _start_time,
    )


@app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
async def health_check_v1():
    """Health check endpoint (v1 API prefix)."""
    return HealthResponse(
        status="ok",
        version="1.0.0",
        uptime_seconds=time.monotonic() - _start_time,
    )


# ─── Task CRUD ────────────────────────────────────────────────────────────────


@app.post(
    "/tasks",
    response_model=TaskResponse,
    status_code=201,
    tags=["Tasks"],
    responses={400: {"model": ErrorResponse}},
)
async def create_task(request: TaskCreateRequest):
    """Create a new VR180 conversion task.

    The task will be queued for processing. Use GET /tasks/{id} to poll status.
    """
    task = task_store.create_task(
        input_path=request.input_path,
        output_path=request.output_path,
        metadata=request.metadata,
    )
    return TaskResponse(**task.to_dict())


@app.post(
    "/api/v1/tasks",
    response_model=TaskResponse,
    status_code=201,
    tags=["Tasks"],
    responses={400: {"model": ErrorResponse}},
)
async def create_task_v1(
    file: UploadFile = File(...),
    output_format: str = Form("equirectangular"),
    resolution: str = Form("4k"),
    codec: str = Form("h265"),
    upscale: str = Form("true"),
    inject_metadata: str = Form("true"),
    x_user_id: str | None = Header(None, alias="X-User-Id"),
):
    """Create a new VR180 conversion task with file upload."""
    user_id = x_user_id or "default-user"
    task_id = str(uuid.uuid4())[:8]

    # Save uploaded file
    upload_path = _UPLOAD_DIR / f"{task_id}_{file.filename}"
    content = await file.read()
    upload_path.write_bytes(content)

    output_path = str(_OUTPUT_DIR / f"{task_id}_vr180.mp4")

    task = task_store.create_task(
        input_path=str(upload_path),
        output_path=output_path,
        metadata={
            "user_id": user_id,
            "output_format": output_format,
            "resolution": resolution,
            "codec": codec,
            "upscale": upscale == "true",
            "inject_metadata": inject_metadata == "true",
            "original_filename": file.filename,
            "file_size_bytes": len(content),
        },
    )

    # Dispatch async Celery conversion task
    celery_task = convert_to_vr180.apply_async(
        kwargs={
            "input_path": str(upload_path),
            "output_dir": str(_OUTPUT_DIR / task_id),
            "params": {
                "depth_model": "small",
                "output_format": output_format,
                "resolution": resolution,
                "codec": codec,
            },
        }
    )
    # Store Celery task id in metadata for progress tracking
    task_store.update_status(
        task.id,
        status=TaskStatus.PROCESSING,
        stage="queued",
    )
    task.metadata["celery_task_id"] = celery_task.id

    return TaskResponse(**task.to_dict())


@app.get(
    "/tasks",
    response_model=TaskListResponse,
    tags=["Tasks"],
)
async def list_tasks(
    status: TaskStatusEnum = None,
    limit: int = 50,
    offset: int = 0,
):
    """List all tasks with optional status filter and pagination."""
    store_status = TaskStatus(status.value) if status else None
    tasks = task_store.list_tasks(status=store_status, limit=limit, offset=offset)
    total = task_store.count_tasks(status=store_status)
    return TaskListResponse(
        tasks=[TaskResponse(**t.to_dict()) for t in tasks],
        total=total,
        limit=limit,
        offset=offset,
    )


@app.get(
    "/api/v1/tasks",
    response_model=TaskListResponse,
    tags=["Tasks"],
)
async def list_tasks_v1(
    status: TaskStatusEnum = None,
    limit: int = 50,
    offset: int = 0,
    x_user_id: str | None = Header(None, alias="X-User-Id"),
):
    """List all tasks with optional status filter and pagination (v1)."""
    store_status = TaskStatus(status.value) if status else None
    tasks = task_store.list_tasks(status=store_status, limit=limit, offset=offset)
    total = task_store.count_tasks(status=store_status)
    return TaskListResponse(
        tasks=[TaskResponse(**t.to_dict()) for t in tasks],
        total=total,
        limit=limit,
        offset=offset,
    )


@app.get(
    "/tasks/{task_id}",
    response_model=TaskResponse,
    tags=["Tasks"],
    responses={404: {"model": ErrorResponse}},
)
async def get_task(task_id: str):
    """Get a specific task by ID."""
    task = task_store.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    return TaskResponse(**task.to_dict())


@app.get(
    "/api/v1/tasks/{task_id}",
    response_model=TaskResponse,
    tags=["Tasks"],
    responses={404: {"model": ErrorResponse}},
)
async def get_task_v1(task_id: str):
    """Get a specific task by ID (v1)."""
    task = task_store.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    return TaskResponse(**task.to_dict())


@app.patch(
    "/tasks/{task_id}",
    response_model=TaskResponse,
    tags=["Tasks"],
    responses={404: {"model": ErrorResponse}},
)
async def update_task(task_id: str, request: TaskUpdateRequest):
    """Update a task's status, progress, or metadata.

    Used by internal pipeline workers to report progress.
    """
    store_status = TaskStatus(request.status.value)
    task = task_store.update_status(
        task_id,
        status=store_status,
        progress=request.progress,
        stage=request.stage,
        error=request.error,
        output_path=request.output_path,
    )
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    return TaskResponse(**task.to_dict())


@app.delete(
    "/tasks/{task_id}",
    status_code=204,
    tags=["Tasks"],
    responses={404: {"model": ErrorResponse}},
)
async def delete_task(task_id: str):
    """Delete a task from the store."""
    deleted = task_store.delete_task(task_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    return None


@app.delete(
    "/api/v1/tasks/{task_id}",
    status_code=204,
    tags=["Tasks"],
    responses={404: {"model": ErrorResponse}},
)
async def delete_task_v1(task_id: str):
    """Delete a task from the store (v1)."""
    deleted = task_store.delete_task(task_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    return None


@app.post(
    "/tasks/{task_id}/cancel",
    response_model=TaskResponse,
    tags=["Tasks"],
    responses={404: {"model": ErrorResponse}},
)
async def cancel_task(task_id: str):
    """Cancel a queued or processing task."""
    task = task_store.cancel_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    return TaskResponse(**task.to_dict())


@app.post(
    "/api/v1/tasks/{task_id}/cancel",
    response_model=TaskResponse,
    tags=["Tasks"],
    responses={404: {"model": ErrorResponse}},
)
async def cancel_task_v1(task_id: str):
    """Cancel a queued or processing task (v1)."""
    task = task_store.cancel_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    return TaskResponse(**task.to_dict())


@app.get(
    "/api/v1/tasks/{task_id}/download",
    tags=["Tasks"],
    responses={404: {"model": ErrorResponse}},
)
async def download_task_result(task_id: str):
    """Download the output file for a completed task."""
    task = task_store.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    if task.status != TaskStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Task not completed yet")
    output_path = task.output_path
    if not output_path or not Path(output_path).exists():
        raise HTTPException(status_code=404, detail="Output file not found")
    return FileResponse(
        path=output_path,
        media_type="video/mp4",
        filename=Path(output_path).name,
    )


# ─── Celery Progress ──────────────────────────────────────────────────────────


@app.get(
    "/api/v1/tasks/{task_id}/progress",
    tags=["Tasks"],
    responses={404: {"model": ErrorResponse}},
)
async def get_task_progress(task_id: str):
    """Get real-time progress of a Celery conversion task.

    Returns the Celery task state, progress percentage, and current stage.
    """
    task = task_store.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    celery_task_id = (task.metadata or {}).get("celery_task_id")
    if not celery_task_id:
        return {
            "state": task.status.value,
            "progress": task.progress,
            "stage": task.stage,
        }

    result = AsyncResult(celery_task_id, app=celery_app)
    info = result.info if isinstance(result.info, dict) else {}
    return {
        "state": result.state,
        "progress": info.get("progress", 0),
        "stage": info.get("stage", ""),
    }


# ─── Quota (stub) ─────────────────────────────────────────────────────────────


@app.get(
    "/api/v1/quota",
    tags=["Quota"],
)
async def get_quota_v1(
    x_user_id: str | None = Header(None, alias="X-User-Id"),
):
    """Get current user's quota."""
    user_id = x_user_id or "default-user"
    return {
        "user_id": user_id,
        "unlimited": True,
        "used": 0,
        "limit": 999,
        "remaining": 999,
    }


# ─── Results ──────────────────────────────────────────────────────────────────


@app.get(
    "/api/v1/results",
    tags=["Results"],
)
async def list_results_v1(
    limit: int = 50,
    offset: int = 0,
    x_user_id: str | None = Header(None, alias="X-User-Id"),
):
    """List completed results for a user."""
    completed_tasks = task_store.list_tasks(status=TaskStatus.COMPLETED, limit=limit, offset=offset)
    results = []
    for t in completed_tasks:
        meta = t.metadata or {}
        results.append(
            {
                "task_id": t.id,
                "filename": meta.get("original_filename", Path(t.output_path).name if t.output_path else t.id),
                "output_path": t.output_path,
                "created_at": t.created_at.isoformat() + "Z" if t.created_at else None,
                "file_size_bytes": Path(t.output_path).stat().st_size
                if t.output_path and Path(t.output_path).exists()
                else meta.get("file_size_bytes"),
                "output_format": meta.get("output_format", "equirectangular"),
                "resolution": meta.get("resolution", "4k"),
            }
        )
    return {"results": results, "total": len(results)}


@app.delete(
    "/api/v1/results/{task_id}",
    status_code=204,
    tags=["Results"],
    responses={404: {"model": ErrorResponse}},
)
async def delete_result_v1(task_id: str):
    """Delete a result and its output file."""
    task = task_store.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Result {task_id} not found")
    # Remove output file if it exists
    if task.output_path and Path(task.output_path).exists():
        Path(task.output_path).unlink()
    deleted = task_store.delete_task(task_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Result {task_id} not found")
    return None


# ─── Video Generation (external providers) ────────────────────────────────────


@app.post(
    "/api/v1/generate",
    response_model=GenerateResponse,
    status_code=201,
    tags=["Video Generation"],
    responses={
        400: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
    },
)
async def generate_video(request: GenerateRequest):
    """Generate a video using an external AI video generation provider.

    Wraps the user prompt with VR180 constraints via ``wrap_prompt_for_vr180``,
    submits to the chosen provider (kling / seedance / veo), and returns the
    external job ID for status polling via GET /api/v1/generate/{job_id}.
    """
    # 1. Wrap prompt with VR180 constraints
    try:
        wrapped = wrap_prompt_for_vr180(
            user_prompt=request.prompt,
            scene_type=request.scene_type,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None

    # 2. Build generation parameters
    params = GenerationParams(
        prompt=wrapped["positive"],
        negative_prompt=wrapped["negative"],
        duration_seconds=request.duration_seconds,
        resolution=request.resolution,
        fps=request.fps,
    )

    # 3. Instantiate provider and submit
    try:
        provider = get_provider(request.provider)
        job_id = await provider.submit(params)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    except Exception as exc:
        log.exception("Provider submission failed: %s", exc)
        raise HTTPException(
            status_code=502,
            detail=f"Provider submission failed: {exc}",
        ) from exc

    return GenerateResponse(
        job_id=job_id,
        provider=request.provider,
        prompt=wrapped["positive"],
        negative_prompt=wrapped["negative"],
        status="pending",
    )


@app.get(
    "/api/v1/generate/{job_id}",
    response_model=dict,
    tags=["Video Generation"],
    responses={404: {"model": ErrorResponse}},
)
async def poll_generation(
    job_id: str,
    provider: str = "kling",
):
    """Poll the status of a video generation job.

    Query parameter ``provider`` selects which provider to poll (default: kling).
    """
    try:
        prov = get_provider(provider)
        status = await prov.poll(job_id)
    except NotImplementedError:
        raise HTTPException(
            status_code=501,
            detail=f"Polling not yet implemented for provider '{provider}'",
        ) from None
    except Exception as exc:
        log.exception("Poll failed for job %s: %s", job_id, exc)
        raise HTTPException(
            status_code=502,
            detail=f"Poll failed: {exc}",
        ) from exc

    return {
        "job_id": status.job_id,
        "state": status.state.value,
        "progress": status.progress,
        "message": status.message,
        "output_url": status.output_url,
    }


@app.get(
    "/api/v1/providers",
    tags=["Video Generation"],
)
async def list_available_providers():
    """List all registered video generation providers."""
    return {
        "providers": list_providers(),
        "count": len(list_providers()),
    }
