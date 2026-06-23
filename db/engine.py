"""SQLAlchemy 2.0 engine and session configuration.

Uses DB_URL environment variable (default: sqlite:///./vr180.db).
Enables WAL mode for SQLite for better concurrent read performance.
"""

import os
from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

_DB_URL = os.getenv("DB_URL", "sqlite:///./vr180.db")

engine = create_engine(
    _DB_URL,
    echo=False,
    future=True,
    connect_args={"check_same_thread": False} if "sqlite" in _DB_URL else {},
)

# Enable WAL mode for SQLite
if "sqlite" in _DB_URL:

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_conn, connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, class_=Session)


def get_session() -> Generator[Session, None, None]:
    """Yield a SQLAlchemy session, closing it when done."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """Provide a transactional scope around a series of operations."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db(url: str | None = None) -> None:
    """Create all tables.  Optionally override DB_URL for testing."""
    if url:
        global engine, SessionLocal
        engine = create_engine(
            url,
            echo=False,
            future=True,
            connect_args={"check_same_thread": False} if "sqlite" in url else {},
        )
        if "sqlite" in url:

            @event.listens_for(engine, "connect")
            def _set_sqlite_pragma2(dbapi_conn, connection_record):
                cursor = dbapi_conn.cursor()
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()

        SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, class_=Session)
    from db.models import Base

    Base.metadata.create_all(bind=engine)


def reset_engine() -> None:
    """Reset engine to default DB_URL.  Used in test teardown."""
    global engine, SessionLocal
    default_url = os.getenv("DB_URL", "sqlite:///./vr180.db")
    engine = create_engine(
        default_url,
        echo=False,
        future=True,
        connect_args={"check_same_thread": False} if "sqlite" in default_url else {},
    )
    if "sqlite" in default_url:

        @event.listens_for(engine, "connect")
        def _set_sqlite_pragma3(dbapi_conn, connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, class_=Session)
