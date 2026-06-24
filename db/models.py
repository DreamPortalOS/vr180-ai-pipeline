"""SQLAlchemy ORM models for VR180 Studio.

Tables:
  - users: per-user account info and tier
  - api_keys: API key authentication (T2-ready)
  - conversion_tasks: persistent task state (replaces in-memory TaskStore)
  - usage_records: per-conversion usage tracking (replaces sqlite QuotaManager)
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, relationship


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _short_id() -> str:
    return uuid.uuid4().hex[:8]


class Base(DeclarativeBase):
    """Base class for all VR180 Studio ORM models."""

    pass


class User(Base):
    """User account — maps to a free/premium/admin tier."""

    __tablename__ = "users"

    id = Column(String(64), primary_key=True, default=_short_id)
    tier = Column(String(16), nullable=False, default="free")  # free | premium | admin
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)

    # relationships
    api_keys = relationship("APIKey", back_populates="user", cascade="all, delete-orphan")
    tasks = relationship("ConversionTask", back_populates="user", cascade="all, delete-orphan")
    usage_records = relationship("UsageRecord", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<User {self.id!r} tier={self.tier!r}>"


class APIKey(Base):
    """API keys for authentication (T2-ready)."""

    __tablename__ = "api_keys"

    id = Column(String(64), primary_key=True, default=_short_id)
    key = Column(String(128), unique=True, nullable=False, index=True)
    user_id = Column(String(64), ForeignKey("users.id"), nullable=False)
    name = Column(String(128), nullable=True)  # human-friendly label
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)

    # relationships
    user = relationship("User", back_populates="api_keys")

    def __repr__(self) -> str:
        return f"<APIKey {self.key[:8]}… user={self.user_id!r}>"


class ConversionTask(Base):
    """Persistent task state — replaces in-memory TaskStore."""

    __tablename__ = "conversion_tasks"

    id = Column(String(64), primary_key=True, default=_short_id)
    user_id = Column(String(64), ForeignKey("users.id"), nullable=True)
    input_path = Column(Text, nullable=False)
    output_path = Column(Text, nullable=True)
    status = Column(String(16), nullable=False, default="queued")
    progress = Column(Float, nullable=False, default=0.0)
    stage = Column(String(64), nullable=False, default="init")
    error = Column(Text, nullable=True)
    metadata_json = Column(Text, nullable=False, default="{}")
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    # relationships
    user = relationship("User", back_populates="tasks")

    __table_args__ = (
        Index("ix_tasks_status", "status"),
        Index("ix_tasks_user_id", "user_id"),
        Index("ix_tasks_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<ConversionTask {self.id!r} status={self.status!r}>"


class UsageRecord(Base):
    """Per-conversion usage tracking — replaces raw sqlite QuotaManager."""

    __tablename__ = "usage_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(64), ForeignKey("users.id"), nullable=False)
    task_id = Column(String(64), nullable=False)
    file_size_bytes = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)

    # relationships
    user = relationship("User", back_populates="usage_records")

    __table_args__ = (
        Index("ix_usage_user_id", "user_id"),
        Index("ix_usage_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<UsageRecord user={self.user_id!r} task={self.task_id!r}>"
