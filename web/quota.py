"""
Quota / Usage Limiting System for VR180 Studio.

Tracks per-user conversion usage with configurable limits.
SQLite-backed for persistence across server restarts.

Usage:
    quota = QuotaManager(max_free_conversions=3)
    quota.check_or_raise("user123")  # raises QuotaExceededError if over limit
    quota.record_usage("user123", task_id="abc", file_size_bytes=1024000)
"""

import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path


class QuotaExceededError(Exception):
    """Raised when a user has exceeded their conversion quota."""

    def __init__(self, user_id: str, used: int, limit: int):
        self.user_id = user_id
        self.used = used
        self.limit = limit
        remaining = max(0, limit - used)
        super().__init__(
            f"Quota exceeded for user '{user_id}': "
            f"{used}/{limit} conversions used, {remaining} remaining. "
            f"Upgrade to premium for unlimited conversions."
        )


class UserTier(str, Enum):
    FREE = "free"
    PREMIUM = "premium"
    ADMIN = "admin"


@dataclass
class UsageRecord:
    """A single usage record for a conversion task."""

    id: int
    user_id: str
    task_id: str
    file_size_bytes: int
    created_at: str
    tier: str


@dataclass
class UserQuota:
    """Quota status for a specific user."""

    user_id: str
    tier: str
    used: int
    limit: int
    remaining: int
    unlimited: bool


class QuotaManager:
    """
    Thread-safe, SQLite-backed quota manager.

    Supports per-user conversion limits with tier-based policies.
    Free tier: configurable limit (default 3 conversions).
    Premium/Admin tier: unlimited conversions.
    """

    DEFAULT_DB_PATH = "data/quota.db"

    def __init__(
        self,
        db_path: str | None = None,
        max_free_conversions: int = 3,
    ):
        self.db_path = db_path or self.DEFAULT_DB_PATH
        self.max_free_conversions = max_free_conversions
        self._lock = threading.RLock()
        self._tier_limits = {
            UserTier.FREE: max_free_conversions,
            UserTier.PREMIUM: -1,  # unlimited
            UserTier.ADMIN: -1,  # unlimited
        }
        self._init_db()

    def _init_db(self):
        """Initialize the SQLite database with required tables."""
        db_file = Path(self.db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)

        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    tier TEXT NOT NULL DEFAULT 'free',
                    created_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS usage_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    file_size_bytes INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_usage_user_id
                ON usage_records(user_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_usage_created_at
                ON usage_records(created_at)
            """)
            conn.commit()

    def _get_conn(self) -> sqlite3.Connection:
        """Get a new SQLite connection (each thread needs its own)."""
        return sqlite3.connect(self.db_path, timeout=10)

    def _ensure_user(self, conn: sqlite3.Connection, user_id: str) -> str:
        """Ensure user exists, create with FREE tier if not. Returns tier."""
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT OR IGNORE INTO users (user_id, tier, created_at) VALUES (?, ?, ?)",
            (user_id, UserTier.FREE.value, now),
        )
        row = conn.execute("SELECT tier FROM users WHERE user_id = ?", (user_id,)).fetchone()
        return row[0] if row else UserTier.FREE.value

    def get_usage_count(self, user_id: str) -> int:
        """Get the total number of conversions used by a user."""
        with self._lock:
            conn = self._get_conn()
            try:
                row = conn.execute(
                    "SELECT COUNT(*) FROM usage_records WHERE user_id = ?",
                    (user_id,),
                ).fetchone()
                return row[0] if row else 0
            finally:
                conn.close()

    def get_tier(self, user_id: str) -> str:
        """Get the user's current tier."""
        with self._lock:
            conn = self._get_conn()
            try:
                self._ensure_user(conn, user_id)
                row = conn.execute("SELECT tier FROM users WHERE user_id = ?", (user_id,)).fetchone()
                return row[0] if row else UserTier.FREE.value
            finally:
                conn.close()

    def set_tier(self, user_id: str, tier: UserTier):
        """Set a user's tier (e.g., upgrade to premium)."""
        with self._lock:
            conn = self._get_conn()
            try:
                now = datetime.now(timezone.utc).isoformat()
                conn.execute(
                    "INSERT OR REPLACE INTO users (user_id, tier, created_at) VALUES (?, ?, ?)",
                    (user_id, tier.value, now),
                )
                conn.commit()
            finally:
                conn.close()

    def get_limit(self, user_id: str) -> int:
        """Get the conversion limit for a user based on their tier. -1 = unlimited."""
        tier = self.get_tier(user_id)
        return self._tier_limits.get(tier, self.max_free_conversions)

    def get_quota(self, user_id: str) -> UserQuota:
        """Get the full quota status for a user."""
        tier = self.get_tier(user_id)
        used = self.get_usage_count(user_id)
        limit = self._tier_limits.get(tier, self.max_free_conversions)
        unlimited = limit < 0
        remaining = max(0, limit - used) if not unlimited else -1
        return UserQuota(
            user_id=user_id,
            tier=tier,
            used=used,
            limit=limit,
            remaining=remaining,
            unlimited=unlimited,
        )

    def check(self, user_id: str) -> bool:
        """Check if a user is within their quota. Returns True if allowed."""
        quota = self.get_quota(user_id)
        if quota.unlimited:
            return True
        return quota.used < quota.limit

    def check_or_raise(self, user_id: str):
        """
        Check quota and raise QuotaExceededError if over limit.
        Call this before starting a conversion task.
        """
        quota = self.get_quota(user_id)
        if not quota.unlimited and quota.used >= quota.limit:
            raise QuotaExceededError(
                user_id=user_id,
                used=quota.used,
                limit=quota.limit,
            )

    def record_usage(
        self,
        user_id: str,
        task_id: str,
        file_size_bytes: int = 0,
    ):
        """
        Record a successful conversion. Call after task completes.
        Raises QuotaExceededError if user is already over limit (race condition guard).
        """
        with self._lock:
            conn = self._get_conn()
            try:
                tier = self._ensure_user(conn, user_id)
                limit = self._tier_limits.get(tier, self.max_free_conversions)

                # Race condition guard: check count within lock
                if limit >= 0:
                    row = conn.execute(
                        "SELECT COUNT(*) FROM usage_records WHERE user_id = ?",
                        (user_id,),
                    ).fetchone()
                    current_count = row[0] if row else 0
                    if current_count >= limit:
                        raise QuotaExceededError(
                            user_id=user_id,
                            used=current_count,
                            limit=limit,
                        )

                now = datetime.now(timezone.utc).isoformat()
                conn.execute(
                    "INSERT INTO usage_records (user_id, task_id, file_size_bytes, created_at) VALUES (?, ?, ?, ?)",
                    (user_id, task_id, file_size_bytes, now),
                )
                conn.commit()
            finally:
                conn.close()

    def get_usage_history(
        self,
        user_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> list[UsageRecord]:
        """Get paginated usage history for a user."""
        with self._lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    "SELECT id, user_id, task_id, file_size_bytes, created_at "
                    "FROM usage_records WHERE user_id = ? "
                    "ORDER BY created_at DESC LIMIT ? OFFSET ?",
                    (user_id, limit, offset),
                ).fetchall()
                return [
                    UsageRecord(
                        id=r[0],
                        user_id=r[1],
                        task_id=r[2],
                        file_size_bytes=r[3],
                        created_at=r[4],
                        tier=self.get_tier(user_id),
                    )
                    for r in rows
                ]
            finally:
                conn.close()

    def reset_usage(self, user_id: str):
        """Reset a user's usage count (admin function)."""
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    "DELETE FROM usage_records WHERE user_id = ?",
                    (user_id,),
                )
                conn.commit()
            finally:
                conn.close()

    def get_total_usage(self) -> int:
        """Get total number of conversions across all users."""
        with self._lock:
            conn = self._get_conn()
            try:
                row = conn.execute("SELECT COUNT(*) FROM usage_records").fetchone()
                return row[0] if row else 0
            finally:
                conn.close()

    def get_total_storage_bytes(self) -> int:
        """Get total storage used across all conversions."""
        with self._lock:
            conn = self._get_conn()
            try:
                row = conn.execute("SELECT COALESCE(SUM(file_size_bytes), 0) FROM usage_records").fetchone()
                return row[0] if row else 0
            finally:
                conn.close()
