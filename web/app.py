"""FastAPI application for VR180 Studio.

Provides REST API endpoints for:
- Health check
- Task CRUD (create, read, list, update, delete, cancel)
- File upload/download (future)

Usage:
    uvicorn web.app:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from web.schemas import (
    ErrorResponse,
    HealthResponse,
    TaskCreateRequest,
    TaskListResponse,
    TaskResponse,
    TaskUpdateRequest,
    TaskStatusEnum,
)
from web.task_store import TaskStore, TaskStatus

log = logging.getLogger("vr180-api")

# Global task store instance
task_store = TaskStore()

# Application start time for uptime calculation
_start_time: float = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    global _start_time
    _start_time = time.monotonic()
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


# ─── Health ───────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint."""
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