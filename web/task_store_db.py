"""SQLAlchemy-backed task state store for VR180 pipeline jobs.

Drop-in replacement for the in-memory TaskStore. Persists tasks to SQLite/PostgreSQL
via SQLAlchemy 2.0 ORM. Thread-safe and compatible with FastAPI async handlers.

Usage:
    from web.task_store_db import TaskStoreDB
    store = TaskStoreDB()
    task = store.create_task(input_path="/path/to/video.mp4")
    store.update_status(task.id, TaskStatus.PROCESSING, progress=0.5)
"""

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from db.engine import SessionLocal
from db.models import ConversionTask as DBTask
from sqlalchemy.orm import Session

log = logging.getLogger("task-store-db")


class TaskStatus(str, Enum):
    """Pipeline task lifecycle states."""

    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PipelineTask:
    """Represents a single VR180 conversion task (in-memory view)."""

    id: str
    input_path: str
    output_path: str | None = None
    status: TaskStatus = TaskStatus.QUEUED
    progress: float = 0.0
    stage: str = "init"
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


def _db_to_dataclass(db_task: DBTask) -> PipelineTask:
    """Convert a DB model to the PipelineTask dataclass."""
    return PipelineTask(
        id=db_task.id,
        input_path=db_task.input_path,
        output_path=db_task.output_path,
        status=TaskStatus(db_task.status),
        progress=db_task.progress,
        stage=db_task.stage,
        error=db_task.error,
        created_at=db_task.created_at,
        updated_at=db_task.updated_at,
        completed_at=db_task.completed_at,
        metadata=json.loads(db_task.metadata_json) if db_task.metadata_json else {},
    )


class TaskStoreDB:
    """SQLAlchemy-backed store for pipeline tasks.

    Drop-in replacement for the in-memory TaskStore. All data is persisted
    to the database configured in db.engine.
    """

    def __init__(self, session_factory=None):
        """Initialize with optional custom session factory (for testing)."""
        self._session_factory = session_factory or SessionLocal

    def _get_session(self) -> Session:
        return self._session_factory()

    def create_task(
        self,
        input_path: str,
        output_path: str | None = None,
        metadata: dict | None = None,
    ) -> PipelineTask:
        """Create a new pipeline task in QUEUED state."""
        task_id = uuid.uuid4().hex[:8]
        now = datetime.now(timezone.utc)
        db_task = DBTask(
            id=task_id,
            input_path=input_path,
            output_path=output_path,
            status=TaskStatus.QUEUED.value,
            progress=0.0,
            stage="init",
            metadata_json=json.dumps(metadata or {}),
            created_at=now,
            updated_at=now,
        )
        db = self._get_session()
        try:
            db.add(db_task)
            db.commit()
            db.refresh(db_task)
            log.info(f"Created task {task_id} for {input_path}")
            return _db_to_dataclass(db_task)
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def get_task(self, task_id: str) -> PipelineTask | None:
        """Retrieve a task by ID."""
        db = self._get_session()
        try:
            db_task = db.query(DBTask).filter(DBTask.id == task_id).first()
            if db_task is None:
                return None
            return _db_to_dataclass(db_task)
        finally:
            db.close()

    def list_tasks(
        self,
        status: TaskStatus | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[PipelineTask]:
        """List tasks with optional status filter and pagination."""
        db = self._get_session()
        try:
            query = db.query(DBTask)
            if status is not None:
                query = query.filter(DBTask.status == status.value)
            query = query.order_by(DBTask.created_at.desc())
            query = query.offset(offset).limit(limit)
            return [_db_to_dataclass(t) for t in query.all()]
        finally:
            db.close()

    def count_tasks(self, status: TaskStatus | None = None) -> int:
        """Count tasks, optionally filtered by status."""
        db = self._get_session()
        try:
            query = db.query(DBTask)
            if status is not None:
                query = query.filter(DBTask.status == status.value)
            return query.count()
        finally:
            db.close()

    def update_status(
        self,
        task_id: str,
        status: TaskStatus,
        progress: float | None = None,
        stage: str | None = None,
        error: str | None = None,
        output_path: str | None = None,
    ) -> PipelineTask | None:
        """Update a task's status and optional fields."""
        db = self._get_session()
        try:
            db_task = db.query(DBTask).filter(DBTask.id == task_id).first()
            if db_task is None:
                return None

            db_task.status = status.value
            db_task.updated_at = datetime.now(timezone.utc)

            if progress is not None:
                db_task.progress = max(0.0, min(1.0, progress))
            if stage is not None:
                db_task.stage = stage
            if error is not None:
                db_task.error = error
            if output_path is not None:
                db_task.output_path = output_path

            if status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
                db_task.completed_at = datetime.now(timezone.utc)
                if status == TaskStatus.COMPLETED:
                    db_task.progress = 1.0

            db.commit()
            db.refresh(db_task)
            log.info(f"Task {task_id} → {status.value} (progress={db_task.progress:.0%})")
            return _db_to_dataclass(db_task)
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def delete_task(self, task_id: str) -> bool:
        """Delete a task from the store."""
        db = self._get_session()
        try:
            db_task = db.query(DBTask).filter(DBTask.id == task_id).first()
            if db_task is None:
                return False
            db.delete(db_task)
            db.commit()
            log.info(f"Deleted task {task_id}")
            return True
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def cancel_task(self, task_id: str) -> PipelineTask | None:
        """Cancel a queued or processing task."""
        task = self.get_task(task_id)
        if task is None:
            return None
        if task.status not in (TaskStatus.QUEUED, TaskStatus.PROCESSING):
            return task  # Can't cancel completed/failed tasks
        return self.update_status(task_id, TaskStatus.CANCELLED, stage="cancelled")
