"""In-memory task state store for VR180 pipeline jobs.

Tracks conversion tasks through their lifecycle: queued → processing → completed/failed.
Thread-safe for concurrent access from FastAPI async handlers.

Usage:
    from web.task_store import TaskStore, TaskStatus
    store = TaskStore()
    task = store.create_task(input_path="/path/to/video.mp4")
    store.update_status(task.id, TaskStatus.PROCESSING, progress=0.5)
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from threading import Lock

log = logging.getLogger("task-store")


class TaskStatus(str, Enum):
    """Pipeline task lifecycle states."""

    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PipelineTask:
    """Represents a single VR180 conversion task."""

    id: str
    input_path: str
    output_path: str | None = None
    status: TaskStatus = TaskStatus.QUEUED
    progress: float = 0.0  # 0.0 to 1.0
    stage: str = "init"  # current pipeline stage name
    error: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize task to JSON-compatible dict."""
        return {
            "id": self.id,
            "input_path": self.input_path,
            "output_path": self.output_path,
            "status": self.status.value,
            "progress": self.progress,
            "stage": self.stage,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "metadata": self.metadata,
        }


class TaskStore:
    """Thread-safe in-memory store for pipeline tasks.

    Provides CRUD operations on PipelineTask objects with locking
    for concurrent access from async FastAPI handlers and background workers.
    """

    def __init__(self):
        self._tasks: dict[str, PipelineTask] = {}
        self._lock = Lock()

    def create_task(
        self,
        input_path: str,
        output_path: str | None = None,
        metadata: dict | None = None,
    ) -> PipelineTask:
        """Create a new pipeline task in QUEUED state.

        Args:
            input_path: Path to the source video file
            output_path: Optional output path (auto-generated if None)
            metadata: Optional metadata dict (e.g., resolution, codec info)

        Returns:
            The newly created PipelineTask
        """
        task_id = str(uuid.uuid4())[:8]
        now = datetime.now(timezone.utc)
        task = PipelineTask(
            id=task_id,
            input_path=input_path,
            output_path=output_path,
            status=TaskStatus.QUEUED,
            progress=0.0,
            stage="init",
            created_at=now,
            updated_at=now,
            metadata=metadata or {},
        )
        with self._lock:
            self._tasks[task_id] = task
        log.info(f"Created task {task_id} for {input_path}")
        return task

    def get_task(self, task_id: str) -> PipelineTask | None:
        """Retrieve a task by ID.

        Args:
            task_id: The task identifier

        Returns:
            PipelineTask if found, None otherwise
        """
        with self._lock:
            return self._tasks.get(task_id)

    def list_tasks(
        self,
        status: TaskStatus | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[PipelineTask]:
        """List tasks with optional status filter and pagination.

        Args:
            status: Filter by task status (None = all)
            limit: Maximum number of tasks to return
            offset: Number of tasks to skip

        Returns:
            List of PipelineTask objects, sorted by creation time (newest first)
        """
        with self._lock:
            tasks = list(self._tasks.values())

        if status is not None:
            tasks = [t for t in tasks if t.status == status]

        tasks.sort(key=lambda t: t.created_at, reverse=True)
        return tasks[offset : offset + limit]

    def count_tasks(self, status: TaskStatus | None = None) -> int:
        """Count tasks, optionally filtered by status."""
        with self._lock:
            if status is None:
                return len(self._tasks)
            return sum(1 for t in self._tasks.values() if t.status == status)

    def update_status(
        self,
        task_id: str,
        status: TaskStatus,
        progress: float | None = None,
        stage: str | None = None,
        error: str | None = None,
        output_path: str | None = None,
    ) -> PipelineTask | None:
        """Update a task's status and optional fields.

        Args:
            task_id: The task identifier
            status: New status
            progress: Updated progress (0.0 to 1.0)
            stage: Current pipeline stage name
            error: Error message (for FAILED status)
            output_path: Output file path (for COMPLETED status)

        Returns:
            Updated PipelineTask, or None if task not found
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None

            task.status = status
            task.updated_at = datetime.now(timezone.utc)

            if progress is not None:
                task.progress = max(0.0, min(1.0, progress))
            if stage is not None:
                task.stage = stage
            if error is not None:
                task.error = error
            if output_path is not None:
                task.output_path = output_path

            if status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
                task.completed_at = datetime.now(timezone.utc)
                if status == TaskStatus.COMPLETED:
                    task.progress = 1.0

        log.info(f"Task {task_id} → {status.value} (progress={task.progress:.0%})")
        return task

    def delete_task(self, task_id: str) -> bool:
        """Delete a task from the store.

        Args:
            task_id: The task identifier

        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            if task_id in self._tasks:
                del self._tasks[task_id]
                log.info(f"Deleted task {task_id}")
                return True
            return False

    def cancel_task(self, task_id: str) -> PipelineTask | None:
        """Cancel a queued or processing task.

        Args:
            task_id: The task identifier

        Returns:
            Updated task, or None if not found
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            if task.status not in (TaskStatus.QUEUED, TaskStatus.PROCESSING):
                return task  # Can't cancel completed/failed tasks

        return self.update_status(task_id, TaskStatus.CANCELLED, stage="cancelled")
