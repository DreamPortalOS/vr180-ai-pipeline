"""Shared test fixtures for database-backed tests."""

import pytest
from db.models import Base
from sqlalchemy import create_engine
from sqlalchemy.orm import Session


@pytest.fixture()
def db_engine(tmp_path):
    """Create a fresh SQLite database for each test."""
    db_url = f"sqlite:///{tmp_path}/test.db"
    engine = create_engine(db_url, echo=False, future=True, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    yield engine
    engine.dispose()


@pytest.fixture()
def db_session(db_engine):
    """Yield a transactional session that rolls back after each test."""
    connection = db_engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection)
    yield session
    session.close()
    transaction.rollback()
    connection.close()
