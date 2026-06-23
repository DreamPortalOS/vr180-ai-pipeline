"""
Result Persistence & Storage for VR180 Studio.

Manages saving, retrieving, and listing processed VR180 video results.
SQLite-backed metadata store with filesystem-based video storage.

Usage:
    storage = ResultStorage(base_dir="data/results")
    result_id = storage.save_result(task_id="abc", input_path="in.mp4", output_path="out.mp4")
    result = storage.get_result(result_id)
    results = storage.list_results(user_id="user123")
"""

import contextlib
import json
import os
import shutil
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class StoredResult:
    """Metadata for a stored VR180 conversion result."""

    id: str
    task_id: str
    user_id: str
    input_filename: str
    output_path: str
    file_size_bytes: int
    duration_seconds: float
    width: int
    height: int
    fps: float
    codec: str
    stereoscopic_mode: str
    projection: str
    metadata_json: str
    created_at: str
    expires_at: str | None


class ResultNotFoundError(Exception):
    """Raised when a result is not found."""

    def __init__(self, result_id: str):
        self.result_id = result_id
        super().__init__(f"Result '{result_id}' not found")


class ResultStorage:
    """
    Thread-safe result storage with SQLite metadata and filesystem video storage.

    Features:
    - Store VR180 conversion results with rich metadata
    - List results with filtering and pagination
    - Automatic file cleanup on deletion
    - Configurable storage directory
    """

    DEFAULT_BASE_DIR = "data/results"

    def __init__(self, base_dir: str | None = None):
        self.base_dir = Path(base_dir or self.DEFAULT_BASE_DIR)
        self.db_path = self.base_dir / "results.db"
        self.videos_dir = self.base_dir / "videos"
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        """Initialize the SQLite database."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.videos_dir.mkdir(parents=True, exist_ok=True)

        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS results (
                    id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    user_id TEXT NOT NULL DEFAULT '',
                    input_filename TEXT NOT NULL DEFAULT '',
                    output_path TEXT NOT NULL DEFAULT '',
                    file_size_bytes INTEGER NOT NULL DEFAULT 0,
                    duration_seconds REAL NOT NULL DEFAULT 0,
                    width INTEGER NOT NULL DEFAULT 0,
                    height INTEGER NOT NULL DEFAULT 0,
                    fps REAL NOT NULL DEFAULT 0,
                    codec TEXT NOT NULL DEFAULT '',
                    stereoscopic_mode TEXT NOT NULL DEFAULT 'side-by-side',
                    projection TEXT NOT NULL DEFAULT 'equirectangular',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    expires_at TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_results_user_id
                ON results(user_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_results_task_id
                ON results(task_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_results_created_at
                ON results(created_at)
            """)
            conn.commit()

    def _get_conn(self) -> sqlite3.Connection:
        """Get a new SQLite connection."""
        return sqlite3.connect(str(self.db_path), timeout=10)

    def save_result(
        self,
        task_id: str,
        output_path: str,
        user_id: str = "",
        input_filename: str = "",
        file_size_bytes: int = 0,
        duration_seconds: float = 0.0,
        width: int = 0,
        height: int = 0,
        fps: float = 0.0,
        codec: str = "h264",
        stereoscopic_mode: str = "side-by-side",
        projection: str = "equirectangular",
        metadata: dict | None = None,
        copy_file: bool = True,
        expires_at: str | None = None,
    ) -> str:
        """
        Save a conversion result.

        Args:
            task_id: The task ID that produced this result.
            output_path: Path to the output video file.
            user_id: User who owns this result.
            input_filename: Original input filename.
            file_size_bytes: Size of the output file in bytes.
            duration_seconds: Video duration in seconds.
            width: Video width in pixels.
            height: Video height in pixels.
            fps: Frames per second.
            codec: Video codec (h264, h265, etc.).
            stereoscopic_mode: Stereo layout (side-by-side, top-bottom, mono).
            projection: Video projection (equirectangular, etc.).
            metadata: Additional metadata dict.
            copy_file: Whether to copy the output file to storage dir.
            expires_at: Optional expiration timestamp (ISO format).

        Returns:
            The result ID (UUID string).
        """
        result_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        # Determine storage path
        if copy_file and os.path.exists(output_path):
            ext = Path(output_path).suffix or ".mp4"
            stored_filename = f"{result_id}{ext}"
            stored_path = self.videos_dir / stored_filename
            shutil.copy2(output_path, stored_path)
            final_output_path = str(stored_path)
            file_size_bytes = file_size_bytes or os.path.getsize(stored_path)
        else:
            final_output_path = output_path

        metadata_json = json.dumps(metadata or {})

        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    """INSERT INTO results
                    (id, task_id, user_id, input_filename, output_path,
                     file_size_bytes, duration_seconds, width, height, fps,
                     codec, stereoscopic_mode, projection, metadata_json,
                     created_at, expires_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        result_id, task_id, user_id, input_filename,
                        final_output_path, file_size_bytes, duration_seconds,
                        width, height, fps, codec, stereoscopic_mode,
                        projection, metadata_json, now, expires_at,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

        return result_id

    def get_result(self, result_id: str) -> StoredResult:
        """Get a single result by ID. Raises ResultNotFoundError if not found."""
        with self._lock:
            conn = self._get_conn()
            try:
                row = conn.execute(
                    "SELECT * FROM results WHERE id = ?", (result_id,)
                ).fetchone()
                if not row:
                    raise ResultNotFoundError(result_id)
                return self._row_to_result(row)
            finally:
                conn.close()

    def list_results(
        self,
        user_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
        order_by: str = "created_at DESC",
    ) -> list[StoredResult]:
        """
        List results with optional user filtering and pagination.

        Args:
            user_id: Filter by user ID (None = all users).
            limit: Max results to return.
            offset: Pagination offset.
            order_by: SQL ORDER BY clause.

        Returns:
            List of StoredResult objects.
        """
        # Validate order_by to prevent SQL injection
        allowed_order = {
            "created_at DESC", "created_at ASC",
            "file_size_bytes DESC", "file_size_bytes ASC",
            "duration_seconds DESC", "duration_seconds ASC",
        }
        if order_by not in allowed_order:
            order_by = "created_at DESC"

        with self._lock:
            conn = self._get_conn()
            try:
                if user_id:
                    rows = conn.execute(
                        f"SELECT * FROM results WHERE user_id = ? ORDER BY {order_by} LIMIT ? OFFSET ?",
                        (user_id, limit, offset),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        f"SELECT * FROM results ORDER BY {order_by} LIMIT ? OFFSET ?",
                        (limit, offset),
                    ).fetchall()
                return [self._row_to_result(r) for r in rows]
            finally:
                conn.close()

    def count_results(self, user_id: str | None = None) -> int:
        """Count total results, optionally filtered by user."""
        with self._lock:
            conn = self._get_conn()
            try:
                if user_id:
                    row = conn.execute(
                        "SELECT COUNT(*) FROM results WHERE user_id = ?",
                        (user_id,),
                    ).fetchone()
                else:
                    row = conn.execute("SELECT COUNT(*) FROM results").fetchone()
                return row[0] if row else 0
            finally:
                conn.close()

    def delete_result(self, result_id: str, delete_file: bool = True) -> bool:
        """
        Delete a result by ID. Optionally delete the stored video file.

        Returns True if the result was found and deleted, False otherwise.
        """
        with self._lock:
            conn = self._get_conn()
            try:
                row = conn.execute(
                    "SELECT output_path FROM results WHERE id = ?", (result_id,)
                ).fetchone()
                if not row:
                    return False

                output_path = row[0]
                conn.execute("DELETE FROM results WHERE id = ?", (result_id,))
                conn.commit()

                # Delete the stored file
                if delete_file and output_path and os.path.exists(output_path):
                    with contextlib.suppress(OSError):
                        os.remove(output_path)

                return True
            finally:
                conn.close()

    def delete_by_task_id(self, task_id: str, delete_file: bool = True) -> int:
        """Delete all results for a task. Returns count of deleted results."""
        with self._lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    "SELECT id, output_path FROM results WHERE task_id = ?",
                    (task_id,),
                ).fetchall()

                for row in rows:
                    if delete_file and row[1] and os.path.exists(row[1]):
                        with contextlib.suppress(OSError):
                            os.remove(row[1])

                conn.execute(
                    "DELETE FROM results WHERE task_id = ?", (task_id,)
                )
                conn.commit()
                return len(rows)
            finally:
                conn.close()

    def cleanup_expired(self) -> int:
        """Delete all expired results. Returns count of deleted results."""
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    "SELECT id, output_path FROM results WHERE expires_at IS NOT NULL AND expires_at < ?",
                    (now,),
                ).fetchall()

                for row in rows:
                    if row[1] and os.path.exists(row[1]):
                        with contextlib.suppress(OSError):
                            os.remove(row[1])

                conn.execute(
                    "DELETE FROM results WHERE expires_at IS NOT NULL AND expires_at < ?",
                    (now,),
                )
                conn.commit()
                return len(rows)
            finally:
                conn.close()

    def get_total_storage_bytes(self) -> int:
        """Get total storage used by all stored result files."""
        with self._lock:
            conn = self._get_conn()
            try:
                row = conn.execute(
                    "SELECT COALESCE(SUM(file_size_bytes), 0) FROM results"
                ).fetchone()
                return row[0] if row else 0
            finally:
                conn.close()

    def _row_to_result(self, row: tuple) -> StoredResult:
        """Convert a database row to a StoredResult."""
        return StoredResult(
            id=row[0],
            task_id=row[1],
            user_id=row[2],
            input_filename=row[3],
            output_path=row[4],
            file_size_bytes=row[5],
            duration_seconds=row[6],
            width=row[7],
            height=row[8],
            fps=row[9],
            codec=row[10],
            stereoscopic_mode=row[11],
            projection=row[12],
            metadata_json=row[13],
            created_at=row[14],
            expires_at=row[15],
        )
