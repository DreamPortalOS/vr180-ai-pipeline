"""SQLAlchemy engine and session factory for VR180 Studio.

Uses SQLite by default (file-based) with WAL mode for concurrent reads.
Can be overridden via environment variable DB_URL for PostgreSQL in production.

Usage:
    from db.engine import SessionLocal, init_db
    init_db()  # creates tables
    db = SessionLocal()
"""

import os
from collections.abc import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

# Default SQLite path — relative to project root
_DEFAULT_DB_PATH = os.environ.get("DB_URL", "sqlite:///data/vr180.db")

_engine = None
_SessionLocal = None


def _get_engine(url: str | None = None):
    """Get or create the global SQLAlchemy engine."""
    global _engine
    if _engine is not None and url is None:
        return _engine

    db_url = url or _DEFAULT_DB_PATH
    connect_args = {}

    # SQLite-specific configuration
    if db_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False

    _engine = create_engine(
        db_url,
        echo=False,
        connect_args=connect_args,
        pool_pre_ping=True,
    )

    # Enable WAL mode for SQLite (better concurrent read performance)
    if db_url.startswith("sqlite"):

        @event.listens_for(_engine, "connect")
        def _set_sqlite_pragma(dbapi_conn, _connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return _engine


def get_session_factory(url: str | None = None) -> sessionmaker[Session]:
    """Get or create the global session factory."""
    global _SessionLocal
    if _SessionLocal is not None and url is None:
        return _SessionLocal

    engine = _get_engine(url)
    _SessionLocal = sessionmaker(
        bind=engine,
        autocommit=False,
        autoflush=False,
        expire_on_commit=False,
    )
    return _SessionLocal


def SessionLocal(url: str | None = None) -> Session:  # noqa: N802
    """Create a new database session. Use as context manager or call .close()."""
    factory = get_session_factory(url)
    return factory()


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a DB session and closes it after use."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db(url: str | None = None):
    """Create all tables defined in db.models.Base.

    Call once at application startup.
    """
    from db.models import Base

    engine = _get_engine(url)
    Base.metadata.create_all(bind=engine)


def reset_engine():
    """Reset the global engine and session factory.

    Useful for testing with different database URLs.
    """
    global _engine, _SessionLocal
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionLocal = None
