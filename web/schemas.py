"""Pydantic schemas for VR180 Studio API request/response models."""

from enum import Enum

from pydantic import BaseModel, Field


class TaskStatusEnum(str, Enum):
    """API task status enum (mirrors TaskStore.TaskStatus)."""

    queued = "queued"
    processing = "processing"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    version: str = "1.0.0"
    uptime_seconds: float


class TaskCreateRequest(BaseModel):
    """Request to create a new VR180 conversion task."""

    input_path: str = Field(..., description="Path to source video file")
    output_path: str | None = Field(None, description="Output file path (auto-generated if omitted)")
    metadata: dict | None = Field(default_factory=dict, description="Optional metadata")


class TaskResponse(BaseModel):
    """Full task state response."""

    id: str
    input_path: str
    output_path: str | None = None
    status: TaskStatusEnum
    progress: float = Field(0.0, ge=0.0, le=1.0)
    stage: str = "init"
    error: str | None = None
    created_at: str
    updated_at: str
    completed_at: str | None = None
    metadata: dict = Field(default_factory=dict)


class TaskListResponse(BaseModel):
    """Paginated task list response."""

    tasks: list[TaskResponse]
    total: int
    limit: int
    offset: int


class TaskUpdateRequest(BaseModel):
    """Request to update task status (used by internal workers)."""

    status: TaskStatusEnum
    progress: float | None = Field(None, ge=0.0, le=1.0)
    stage: str | None = None
    error: str | None = None
    output_path: str | None = None


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
    detail: str | None = None


class GenerateRequest(BaseModel):
    """Request to generate a video via an external provider."""

    prompt: str = Field(..., description="User creative prompt (wrapped with VR180 constraints)")
    scene_type: str = Field("fpv", description="Scene type: fpv / walkthrough / orbit / static")
    provider: str = Field("kling", description="Video generation provider: kling / seedance / veo")
    duration_seconds: int = Field(5, ge=1, le=30, description="Target video duration in seconds")
    resolution: str = Field("1080p", description="Output resolution")
    fps: int = Field(24, ge=1, le=60, description="Frames per second")


class GenerateResponse(BaseModel):
    """Response from a video generation request."""

    job_id: str = Field(..., description="Job ID from the external provider")
    provider: str = Field(..., description="Provider used for generation")
    prompt: str = Field(..., description="The full VR180-wrapped positive prompt")
    negative_prompt: str = Field(..., description="The VR180 negative prompt")
    status: str = Field("pending", description="Initial job status")
