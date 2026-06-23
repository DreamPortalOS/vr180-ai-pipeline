"""SQLAlchemy ORM models for VR180 Studio."""

import datetime
import hashlib
import secrets

from sqlalchemy import Boolean, Column, DateTime, Integer, String
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Base class for all ORM models."""

    pass


class ApiKey(Base):
    """API key for authenticating write operations.

    The raw key is shown once at creation time. Only the SHA-256 hash
    is stored in the database.
    """

    __tablename__ = "api_keys"

    id = Column(Integer, primary_key=True, autoincrement=True)
    key_hash = Column(String(64), unique=True, nullable=False, index=True)
    name = Column(String(128), nullable=False, default="default")
    active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)

    # ── helpers ──────────────────────────────────────────────────────

    @staticmethod
    def hash_key(raw_key: str) -> str:
        """Return hex SHA-256 hash of *raw_key*."""
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    @classmethod
    def generate_key(cls, name: str = "default") -> tuple["ApiKey", str]:
        """Return (ApiKey instance, raw_key).  Caller must commit."""
        raw_key = f"vr180_{secrets.token_urlsafe(32)}"
        api_key = cls(key_hash=cls.hash_key(raw_key), name=name)
        return api_key, raw_key

    def verify(self, raw_key: str) -> bool:
        """Return True if *raw_key* matches this key's hash and is active."""
        if not self.active:
            return False
        return self.key_hash == self.hash_key(raw_key)
