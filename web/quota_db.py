"""SQLAlchemy-backed Quota / Usage Limiting System for VR180 Studio.

Drop-in replacement for the raw sqlite QuotaManager. Uses SQLAlchemy 2.0 ORM
for persistence. Thread-safe and compatible with FastAPI async handlers.

Usage:
    from web.quota_db import QuotaManagerDB
    quota = QuotaManagerDB(max_free_conversions=3)
    quota.check_or_raise("user123")
    quota.record_usage("user123", task_id="abc", file_size_bytes=1024000)
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

from db.engine import SessionLocal
from db.models import UsageRecord as DBUsage
from db.models import User as DBUser
from sqlalchemy import func
from sqlalchemy.orm import Session

log = logging.getLogger("quota-db")


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
class UsageRecordData:
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


class QuotaManagerDB:
    """SQLAlchemy-backed quota manager.

    Drop-in replacement for the raw sqlite QuotaManager. Supports per-user
    conversion limits with tier-based policies.
    """

    def __init__(
        self,
        max_free_conversions: int = 3,
        session_factory=None,
    ):
        self.max_free_conversions = max_free_conversions
        self._session_factory = session_factory or SessionLocal
        self._tier_limits = {
            UserTier.FREE.value: max_free_conversions,
            UserTier.PREMIUM.value: -1,  # unlimited
            UserTier.ADMIN.value: -1,  # unlimited
        }

    def _get_session(self) -> Session:
        return self._session_factory()

    def _ensure_user(self, db: Session, user_id: str) -> str:
        """Ensure user exists, create with FREE tier if not. Returns tier."""
        user = db.query(DBUser).filter(DBUser.id == user_id).first()
        if user is None:
            user = DBUser(
                id=user_id,
                tier=UserTier.FREE.value,
                created_at=datetime.now(timezone.utc),
            )
            db.add(user)
            db.flush()
        return user.tier

    def get_usage_count(self, user_id: str) -> int:
        """Get the total number of conversions used by a user."""
        db = self._get_session()
        try:
            count = db.query(func.count(DBUsage.id)).filter(DBUsage.user_id == user_id).scalar()
            return count or 0
        finally:
            db.close()

    def get_tier(self, user_id: str) -> str:
        """Get the user's current tier."""
        db = self._get_session()
        try:
            tier = self._ensure_user(db, user_id)
            db.commit()
            return tier
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def set_tier(self, user_id: str, tier: UserTier):
        """Set a user's tier (e.g., upgrade to premium)."""
        db = self._get_session()
        try:
            user = db.query(DBUser).filter(DBUser.id == user_id).first()
            if user is None:
                user = DBUser(
                    id=user_id,
                    tier=tier.value,
                    created_at=datetime.now(timezone.utc),
                )
                db.add(user)
            else:
                user.tier = tier.value
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

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
        """Check quota and raise QuotaExceededError if over limit."""
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
        """Record a successful conversion. Call after task completes."""
        db = self._get_session()
        try:
            tier = self._ensure_user(db, user_id)
            limit = self._tier_limits.get(tier, self.max_free_conversions)

            # Race condition guard: check count within the same transaction
            if limit >= 0:
                current_count = db.query(func.count(DBUsage.id)).filter(DBUsage.user_id == user_id).scalar() or 0
                if current_count >= limit:
                    raise QuotaExceededError(
                        user_id=user_id,
                        used=current_count,
                        limit=limit,
                    )

            usage = DBUsage(
                user_id=user_id,
                task_id=task_id,
                file_size_bytes=file_size_bytes,
                created_at=datetime.now(timezone.utc),
            )
            db.add(usage)
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def get_usage_history(
        self,
        user_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> list[UsageRecordData]:
        """Get paginated usage history for a user."""
        db = self._get_session()
        try:
            tier = self._ensure_user(db, user_id)
            rows = (
                db.query(DBUsage)
                .filter(DBUsage.user_id == user_id)
                .order_by(DBUsage.created_at.desc())
                .offset(offset)
                .limit(limit)
                .all()
            )
            db.commit()
            return [
                UsageRecordData(
                    id=r.id,
                    user_id=r.user_id,
                    task_id=r.task_id,
                    file_size_bytes=r.file_size_bytes,
                    created_at=r.created_at.isoformat() if r.created_at else "",
                    tier=tier,
                )
                for r in rows
            ]
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def reset_usage(self, user_id: str):
        """Reset a user's usage count (admin function)."""
        db = self._get_session()
        try:
            db.query(DBUsage).filter(DBUsage.user_id == user_id).delete()
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def get_total_usage(self) -> int:
        """Get total number of conversions across all users."""
        db = self._get_session()
        try:
            count = db.query(func.count(DBUsage.id)).scalar()
            return count or 0
        finally:
            db.close()

    def get_total_storage_bytes(self) -> int:
        """Get total storage used across all conversions."""
        db = self._get_session()
        try:
            total = db.query(func.coalesce(func.sum(DBUsage.file_size_bytes), 0)).scalar()
            return total or 0
        finally:
            db.close()
