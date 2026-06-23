"""Tests for the SQLAlchemy database layer.

Covers:
  - DB engine initialization and table creation
  - ORM models (User, APIKey, ConversionTask, UsageRecord)
  - TaskStoreDB (CRUD, status transitions, pagination)
  - QuotaManagerDB (tier limits, usage tracking, quota enforcement)
"""

import pytest
from db.models import APIKey, Base, ConversionTask, UsageRecord, User
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from web.quota_db import QuotaExceededError, QuotaManagerDB, UserTier
from web.task_store_db import TaskStatus, TaskStoreDB

# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def db_engine():
    """Create an in-memory SQLite engine for testing."""
    engine = create_engine("sqlite:///:memory:", echo=False)

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_conn, _connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


@pytest.fixture()
def session_factory(db_engine):
    """Create a session factory bound to the test engine."""
    return sessionmaker(bind=db_engine, autocommit=False, autoflush=False, expire_on_commit=False)


@pytest.fixture()
def task_store(session_factory):
    """Create a TaskStoreDB backed by the test database."""
    return TaskStoreDB(session_factory=session_factory)


@pytest.fixture()
def quota_manager(session_factory):
    """Create a QuotaManagerDB backed by the test database."""
    return QuotaManagerDB(max_free_conversions=3, session_factory=session_factory)


# ─── ORM Model Tests ──────────────────────────────────────────────────────────


class TestORMModels:
    """Test SQLAlchemy ORM model creation and relationships."""

    def test_user_creation(self, session_factory):
        db = session_factory()
        try:
            user = User(id="u1", tier="free")
            db.add(user)
            db.commit()
            assert user.id == "u1"
            assert user.tier == "free"
            assert user.created_at is not None
        finally:
            db.close()

    def test_api_key_creation(self, session_factory):
        db = session_factory()
        try:
            user = User(id="u2", tier="premium")
            db.add(user)
            db.flush()
            key = APIKey(key="sk-test-12345678", user_id="u2", name="test-key")
            db.add(key)
            db.commit()
            assert key.is_active is True
            assert key.user.id == "u2"
        finally:
            db.close()

    def test_conversion_task_creation(self, session_factory):
        db = session_factory()
        try:
            task = ConversionTask(
                id="t1",
                input_path="/tmp/input.mp4",
                status="queued",
                metadata_json='{"codec": "h265"}',
            )
            db.add(task)
            db.commit()
            assert task.status == "queued"
            assert task.progress == 0.0
            assert task.stage == "init"
        finally:
            db.close()

    def test_usage_record_creation(self, session_factory):
        db = session_factory()
        try:
            user = User(id="u3", tier="free")
            db.add(user)
            db.flush()
            record = UsageRecord(user_id="u3", task_id="t1", file_size_bytes=1024)
            db.add(record)
            db.commit()
            assert record.id is not None
            assert record.file_size_bytes == 1024
        finally:
            db.close()

    def test_cascade_delete_user(self, session_factory):
        db = session_factory()
        try:
            user = User(id="u4", tier="free")
            db.add(user)
            db.flush()
            db.add(ConversionTask(id="t2", input_path="/tmp/a.mp4", user_id="u4"))
            db.add(UsageRecord(user_id="u4", task_id="t2", file_size_bytes=500))
            db.add(APIKey(key="sk-cascade-test", user_id="u4"))
            db.commit()

            db.delete(user)
            db.commit()

            assert db.query(ConversionTask).filter_by(user_id="u4").count() == 0
            assert db.query(UsageRecord).filter_by(user_id="u4").count() == 0
            assert db.query(APIKey).filter_by(user_id="u4").count() == 0
        finally:
            db.close()


# ─── TaskStoreDB Tests ────────────────────────────────────────────────────────


class TestTaskStoreDB:
    """Test the DB-backed TaskStore."""

    def test_create_and_get_task(self, task_store):
        task = task_store.create_task(
            input_path="/tmp/test.mp4",
            metadata={"codec": "h265"},
        )
        assert task.id is not None
        assert task.input_path == "/tmp/test.mp4"
        assert task.status == TaskStatus.QUEUED
        assert task.metadata == {"codec": "h265"}

        retrieved = task_store.get_task(task.id)
        assert retrieved is not None
        assert retrieved.id == task.id
        assert retrieved.input_path == "/tmp/test.mp4"

    def test_get_nonexistent_task(self, task_store):
        assert task_store.get_task("nonexistent") is None

    def test_update_status(self, task_store):
        task = task_store.create_task(input_path="/tmp/test.mp4")
        updated = task_store.update_status(
            task.id,
            TaskStatus.PROCESSING,
            progress=0.5,
            stage="depth_estimation",
        )
        assert updated is not None
        assert updated.status == TaskStatus.PROCESSING
        assert updated.progress == 0.5
        assert updated.stage == "depth_estimation"

    def test_update_to_completed_sets_progress_1(self, task_store):
        task = task_store.create_task(input_path="/tmp/test.mp4")
        updated = task_store.update_status(
            task.id,
            TaskStatus.COMPLETED,
            progress=0.8,
            output_path="/tmp/output.mp4",
        )
        assert updated.status == TaskStatus.COMPLETED
        assert updated.progress == 1.0
        assert updated.completed_at is not None
        assert updated.output_path == "/tmp/output.mp4"

    def test_update_to_failed_sets_completed_at(self, task_store):
        task = task_store.create_task(input_path="/tmp/test.mp4")
        updated = task_store.update_status(
            task.id,
            TaskStatus.FAILED,
            error="GPU out of memory",
        )
        assert updated.status == TaskStatus.FAILED
        assert updated.error == "GPU out of memory"
        assert updated.completed_at is not None

    def test_update_nonexistent_returns_none(self, task_store):
        result = task_store.update_status("nonexistent", TaskStatus.PROCESSING)
        assert result is None

    def test_list_tasks(self, task_store):
        for i in range(5):
            task_store.create_task(input_path=f"/tmp/test{i}.mp4")
        tasks = task_store.list_tasks()
        assert len(tasks) == 5

    def test_list_tasks_with_status_filter(self, task_store):
        t1 = task_store.create_task(input_path="/tmp/a.mp4")
        task_store.create_task(input_path="/tmp/b.mp4")
        task_store.update_status(t1.id, TaskStatus.PROCESSING)

        queued = task_store.list_tasks(status=TaskStatus.QUEUED)
        processing = task_store.list_tasks(status=TaskStatus.PROCESSING)
        assert len(queued) == 1
        assert len(processing) == 1

    def test_list_tasks_pagination(self, task_store):
        for i in range(10):
            task_store.create_task(input_path=f"/tmp/test{i}.mp4")
        page1 = task_store.list_tasks(limit=3, offset=0)
        page2 = task_store.list_tasks(limit=3, offset=3)
        assert len(page1) == 3
        assert len(page2) == 3
        assert page1[0].id != page2[0].id

    def test_count_tasks(self, task_store):
        for i in range(3):
            task_store.create_task(input_path=f"/tmp/test{i}.mp4")
        assert task_store.count_tasks() == 3
        assert task_store.count_tasks(status=TaskStatus.QUEUED) == 3
        assert task_store.count_tasks(status=TaskStatus.COMPLETED) == 0

    def test_delete_task(self, task_store):
        task = task_store.create_task(input_path="/tmp/test.mp4")
        assert task_store.delete_task(task.id) is True
        assert task_store.get_task(task.id) is None

    def test_delete_nonexistent_returns_false(self, task_store):
        assert task_store.delete_task("nonexistent") is False

    def test_cancel_queued_task(self, task_store):
        task = task_store.create_task(input_path="/tmp/test.mp4")
        cancelled = task_store.cancel_task(task.id)
        assert cancelled is not None
        assert cancelled.status == TaskStatus.CANCELLED

    def test_cancel_completed_task_is_noop(self, task_store):
        task = task_store.create_task(input_path="/tmp/test.mp4")
        task_store.update_status(task.id, TaskStatus.COMPLETED)
        result = task_store.cancel_task(task.id)
        assert result is not None
        assert result.status == TaskStatus.COMPLETED  # unchanged

    def test_cancel_nonexistent_returns_none(self, task_store):
        assert task_store.cancel_task("nonexistent") is None

    def test_task_to_dict(self, task_store):
        task = task_store.create_task(
            input_path="/tmp/test.mp4",
            metadata={"resolution": "4k"},
        )
        d = task.to_dict()
        assert d["id"] == task.id
        assert d["input_path"] == "/tmp/test.mp4"
        assert d["status"] == "queued"
        assert d["metadata"] == {"resolution": "4k"}
        assert "created_at" in d
        assert "updated_at" in d

    def test_tasks_persist_across_sessions(self, session_factory):
        """Verify tasks survive across different session instances."""
        store1 = TaskStoreDB(session_factory=session_factory)
        task = store1.create_task(input_path="/tmp/persist.mp4")
        task_id = task.id

        store2 = TaskStoreDB(session_factory=session_factory)
        retrieved = store2.get_task(task_id)
        assert retrieved is not None
        assert retrieved.input_path == "/tmp/persist.mp4"


# ─── QuotaManagerDB Tests ────────────────────────────────────────────────────


class TestQuotaManagerDB:
    """Test the DB-backed QuotaManager."""

    def test_new_user_is_free_tier(self, quota_manager):
        assert quota_manager.get_tier("user1") == "free"

    def test_free_user_limit(self, quota_manager):
        assert quota_manager.get_limit("user1") == 3

    def test_premium_user_unlimited(self, quota_manager):
        quota_manager.set_tier("user1", UserTier.PREMIUM)
        assert quota_manager.get_limit("user1") == -1

    def test_admin_user_unlimited(self, quota_manager):
        quota_manager.set_tier("user1", UserTier.ADMIN)
        assert quota_manager.get_limit("user1") == -1

    def test_record_and_count_usage(self, quota_manager):
        quota_manager.record_usage("user1", task_id="t1", file_size_bytes=1024)
        quota_manager.record_usage("user1", task_id="t2", file_size_bytes=2048)
        assert quota_manager.get_usage_count("user1") == 2

    def test_free_user_quota_enforcement(self, quota_manager):
        for i in range(3):
            quota_manager.record_usage("user1", task_id=f"t{i}")
        with pytest.raises(QuotaExceededError) as exc_info:
            quota_manager.record_usage("user1", task_id="t3")
        assert "user1" in str(exc_info.value)
        assert exc_info.value.used == 3
        assert exc_info.value.limit == 3

    def test_premium_user_no_limit(self, quota_manager):
        quota_manager.set_tier("user1", UserTier.PREMIUM)
        for i in range(100):
            quota_manager.record_usage("user1", task_id=f"t{i}")
        assert quota_manager.check("user1") is True

    def test_check_returns_true_when_within_quota(self, quota_manager):
        assert quota_manager.check("user1") is True

    def test_check_returns_false_when_over_quota(self, quota_manager):
        for i in range(3):
            quota_manager.record_usage("user1", task_id=f"t{i}")
        assert quota_manager.check("user1") is False

    def test_check_or_raise_when_over_quota(self, quota_manager):
        for i in range(3):
            quota_manager.record_usage("user1", task_id=f"t{i}")
        with pytest.raises(QuotaExceededError):
            quota_manager.check_or_raise("user1")

    def test_check_or_raise_passes_when_within_quota(self, quota_manager):
        quota_manager.record_usage("user1", task_id="t0")
        quota_manager.check_or_raise("user1")  # should not raise

    def test_get_quota(self, quota_manager):
        quota_manager.record_usage("user1", task_id="t0")
        q = quota_manager.get_quota("user1")
        assert q.user_id == "user1"
        assert q.tier == "free"
        assert q.used == 1
        assert q.limit == 3
        assert q.remaining == 2
        assert q.unlimited is False

    def test_get_quota_premium(self, quota_manager):
        quota_manager.set_tier("user1", UserTier.PREMIUM)
        q = quota_manager.get_quota("user1")
        assert q.unlimited is True
        assert q.remaining == -1

    def test_usage_history(self, quota_manager):
        quota_manager.set_tier("user1", UserTier.PREMIUM)
        for i in range(5):
            quota_manager.record_usage("user1", task_id=f"t{i}", file_size_bytes=i * 100)
        history = quota_manager.get_usage_history("user1", limit=3)
        assert len(history) == 3
        # Most recent first
        assert history[0].task_id == "t4"

    def test_usage_history_pagination(self, quota_manager):
        quota_manager.set_tier("user1", UserTier.PREMIUM)
        for i in range(5):
            quota_manager.record_usage("user1", task_id=f"t{i}")
        page2 = quota_manager.get_usage_history("user1", limit=2, offset=2)
        assert len(page2) == 2

    def test_reset_usage(self, quota_manager):
        for i in range(3):
            quota_manager.record_usage("user1", task_id=f"t{i}")
        quota_manager.reset_usage("user1")
        assert quota_manager.get_usage_count("user1") == 0

    def test_total_usage(self, quota_manager):
        quota_manager.record_usage("user1", task_id="t0")
        quota_manager.record_usage("user2", task_id="t1")
        assert quota_manager.get_total_usage() == 2

    def test_total_storage_bytes(self, quota_manager):
        quota_manager.record_usage("user1", task_id="t0", file_size_bytes=1024)
        quota_manager.record_usage("user2", task_id="t1", file_size_bytes=2048)
        assert quota_manager.get_total_storage_bytes() == 3072

    def test_set_tier_creates_user_if_not_exists(self, quota_manager):
        quota_manager.set_tier("new_user", UserTier.PREMIUM)
        assert quota_manager.get_tier("new_user") == "premium"

    def test_different_users_independent_quotas(self, quota_manager):
        for i in range(3):
            quota_manager.record_usage("user1", task_id=f"t{i}")
        # user2 should still have full quota
        assert quota_manager.check("user2") is True
        quota_manager.record_usage("user2", task_id="t0")
        assert quota_manager.get_usage_count("user2") == 1


# ─── Integration: engine.init_db ─────────────────────────────────────────────


class TestEngineInit:
    """Test db.engine initialization functions."""

    def test_init_db_creates_tables(self, tmp_path):
        from db.engine import init_db, reset_engine

        db_url = f"sqlite:///{tmp_path / 'test.db'}"
        try:
            init_db(url=db_url)
            # Verify tables exist by inserting a record
            engine = create_engine(db_url)
            with engine.connect() as conn:
                conn.execute(User.__table__.insert().values(id="test", tier="free"))
                conn.commit()
            engine.dispose()
        finally:
            reset_engine()

    def test_reset_engine(self):
        from db.engine import SessionLocal, init_db, reset_engine

        try:
            init_db(url="sqlite:///:memory:")
            db = SessionLocal(url="sqlite:///:memory:")
            db.close()
            reset_engine()
            # After reset, a new session should work
            init_db(url="sqlite:///:memory:")
            db = SessionLocal(url="sqlite:///:memory:")
            db.close()
        finally:
            reset_engine()
