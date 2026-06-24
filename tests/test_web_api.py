"""Tests for VR180 Studio Web API (Phase 4)."""

import pytest
from db.engine import SessionLocal
from db.models import APIKey, User
from fastapi.testclient import TestClient
from web.app import app
from web.auth import hash_key
from web.task_store import TaskStore
from web.task_store_db import TaskStatus

# Test API key used for auth-protected endpoints
TEST_API_KEY = "vr180_webapi_test_key_abc123"
TEST_KEY_HDR = {"X-API-Key": TEST_API_KEY}


def _seed_api_key():
    """Ensure a test API key exists in the DB."""
    db = SessionLocal()
    try:
        # Create a test user if not exists
        user = db.query(User).filter(User.id == "webapi-test-user").first()
        if user is None:
            user = User(id="webapi-test-user")
            db.add(user)
            db.flush()

        # Remove old keys for this user to avoid duplicates
        db.query(APIKey).filter(APIKey.user_id == user.id).delete()
        api_key = APIKey(
            key_hash=hash_key(TEST_API_KEY),
            name="webapi-test-key",
            user_id=user.id,
            is_active=True,
        )
        db.add(api_key)
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


@pytest.fixture(autouse=True)
def reset_store():
    """Reset task store before each test by deleting all tasks from DB."""
    try:
        from db.engine import SessionLocal
        from db.models import ConversionTask

        db = SessionLocal()
        try:
            db.query(ConversionTask).delete()
            db.commit()
        finally:
            db.close()
    except Exception:
        pass
    # Also ensure API key is seeded
    _seed_api_key()
    yield


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


# ─── Health ───────────────────────────────────────────────────────────────────


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_body(self, client):
        data = client.get("/health").json()
        assert data["status"] == "ok"
        assert data["version"] == "1.0.0"
        assert "uptime_seconds" in data
        assert data["uptime_seconds"] >= 0


# ─── Create Task ─────────────────────────────────────────────────────────────


class TestCreateTask:
    def test_create_returns_201(self, client):
        resp = client.post("/tasks", json={"input_path": "/tmp/test.mp4"}, headers=TEST_KEY_HDR)
        assert resp.status_code == 201

    def test_create_task_fields(self, client):
        data = client.post("/tasks", json={"input_path": "/tmp/test.mp4"}, headers=TEST_KEY_HDR).json()
        assert "id" in data
        assert data["input_path"] == "/tmp/test.mp4"
        assert data["status"] == "queued"
        assert data["progress"] == 0.0
        assert data["stage"] == "init"
        assert data["output_path"] is None
        assert data["error"] is None

    def test_create_with_output_path(self, client):
        data = client.post(
            "/tasks",
            json={
                "input_path": "/tmp/test.mp4",
                "output_path": "/tmp/out_vr180.mp4",
            },
            headers=TEST_KEY_HDR,
        ).json()
        assert data["output_path"] == "/tmp/out_vr180.mp4"

    def test_create_with_metadata(self, client):
        data = client.post(
            "/tasks",
            json={
                "input_path": "/tmp/test.mp4",
                "metadata": {"codec": "h264", "fps": 30},
            },
            headers=TEST_KEY_HDR,
        ).json()
        assert data["metadata"]["codec"] == "h264"
        assert data["metadata"]["fps"] == 30

    def test_create_missing_input_path(self, client):
        resp = client.post("/tasks", json={}, headers=TEST_KEY_HDR)
        assert resp.status_code == 422


# ─── Get Task ────────────────────────────────────────────────────────────────


class TestGetTask:
    def test_get_existing_task(self, client):
        create_resp = client.post("/tasks", json={"input_path": "/tmp/a.mp4"}, headers=TEST_KEY_HDR)
        task_id = create_resp.json()["id"]
        data = client.get(f"/tasks/{task_id}").json()
        assert data["id"] == task_id

    def test_get_nonexistent_task(self, client):
        resp = client.get("/tasks/nonexistent999")
        assert resp.status_code == 404


# ─── List Tasks ──────────────────────────────────────────────────────────────


class TestListTasks:
    def test_list_empty(self, client):
        data = client.get("/tasks").json()
        assert data["tasks"] == []
        assert data["total"] == 0

    def test_list_after_creating(self, client):
        client.post("/tasks", json={"input_path": "/tmp/a.mp4"}, headers=TEST_KEY_HDR)
        client.post("/tasks", json={"input_path": "/tmp/b.mp4"}, headers=TEST_KEY_HDR)
        data = client.get("/tasks").json()
        assert data["total"] == 2
        assert len(data["tasks"]) == 2

    def test_list_with_status_filter(self, client):
        create_resp = client.post("/tasks", json={"input_path": "/tmp/a.mp4"}, headers=TEST_KEY_HDR)
        task_id = create_resp.json()["id"]
        # Update one task to processing
        client.patch(
            f"/tasks/{task_id}",
            json={"status": "processing"},
            headers=TEST_KEY_HDR,
        )
        data = client.get("/tasks?status=queued").json()
        assert data["total"] == 0
        data = client.get("/tasks?status=processing").json()
        assert data["total"] == 1

    def test_list_pagination(self, client):
        for i in range(5):
            client.post(
                "/tasks",
                json={"input_path": f"/tmp/{i}.mp4"},
                headers=TEST_KEY_HDR,
            )
        data = client.get("/tasks?limit=2&offset=0").json()
        assert len(data["tasks"]) == 2
        assert data["total"] == 5


# ─── Update Task ─────────────────────────────────────────────────────────────


class TestUpdateTask:
    def test_update_status(self, client):
        create_resp = client.post("/tasks", json={"input_path": "/tmp/a.mp4"}, headers=TEST_KEY_HDR)
        task_id = create_resp.json()["id"]
        data = client.patch(
            f"/tasks/{task_id}",
            json={
                "status": "processing",
                "progress": 0.5,
                "stage": "depth_estimation",
            },
            headers=TEST_KEY_HDR,
        ).json()
        assert data["status"] == "processing"
        assert data["progress"] == 0.5
        assert data["stage"] == "depth_estimation"

    def test_update_nonexistent(self, client):
        resp = client.patch("/tasks/fake999", json={"status": "processing"}, headers=TEST_KEY_HDR)
        assert resp.status_code == 404

    def test_update_to_completed(self, client):
        create_resp = client.post("/tasks", json={"input_path": "/tmp/a.mp4"}, headers=TEST_KEY_HDR)
        task_id = create_resp.json()["id"]
        data = client.patch(
            f"/tasks/{task_id}",
            json={
                "status": "completed",
                "output_path": "/tmp/out.mp4",
            },
            headers=TEST_KEY_HDR,
        ).json()
        assert data["status"] == "completed"
        assert data["progress"] == 1.0
        assert data["output_path"] == "/tmp/out.mp4"
        assert data["completed_at"] is not None

    def test_update_to_failed_with_error(self, client):
        create_resp = client.post("/tasks", json={"input_path": "/tmp/a.mp4"}, headers=TEST_KEY_HDR)
        task_id = create_resp.json()["id"]
        data = client.patch(
            f"/tasks/{task_id}",
            json={
                "status": "failed",
                "error": "GPU out of memory",
            },
            headers=TEST_KEY_HDR,
        ).json()
        assert data["status"] == "failed"
        assert data["error"] == "GPU out of memory"


# ─── Delete Task ─────────────────────────────────────────────────────────────


class TestDeleteTask:
    def test_delete_existing(self, client):
        create_resp = client.post("/tasks", json={"input_path": "/tmp/a.mp4"}, headers=TEST_KEY_HDR)
        task_id = create_resp.json()["id"]
        resp = client.delete(f"/tasks/{task_id}", headers=TEST_KEY_HDR)
        assert resp.status_code == 204
        # Verify gone
        assert client.get(f"/tasks/{task_id}").status_code == 404

    def test_delete_nonexistent(self, client):
        resp = client.delete("/tasks/fake999", headers=TEST_KEY_HDR)
        assert resp.status_code == 404


# ─── Cancel Task ─────────────────────────────────────────────────────────────


class TestCancelTask:
    def test_cancel_queued(self, client):
        create_resp = client.post("/tasks", json={"input_path": "/tmp/a.mp4"}, headers=TEST_KEY_HDR)
        task_id = create_resp.json()["id"]
        data = client.post(f"/tasks/{task_id}/cancel", headers=TEST_KEY_HDR).json()
        assert data["status"] == "cancelled"

    def test_cancel_processing(self, client):
        create_resp = client.post("/tasks", json={"input_path": "/tmp/a.mp4"}, headers=TEST_KEY_HDR)
        task_id = create_resp.json()["id"]
        client.patch(
            f"/tasks/{task_id}",
            json={"status": "processing"},
            headers=TEST_KEY_HDR,
        )
        data = client.post(f"/tasks/{task_id}/cancel", headers=TEST_KEY_HDR).json()
        assert data["status"] == "cancelled"

    def test_cancel_nonexistent(self, client):
        resp = client.post("/tasks/fake999/cancel", headers=TEST_KEY_HDR)
        assert resp.status_code == 404


# ─── TaskStore Unit Tests ────────────────────────────────────────────────────


class TestTaskStore:
    def test_create_and_get(self):
        store = TaskStore()
        task = store.create_task(input_path="/tmp/v.mp4")
        assert task.status == TaskStatus.QUEUED
        retrieved = store.get_task(task.id)
        assert retrieved is not None
        assert retrieved.id == task.id

    def test_list_with_filter(self):
        store = TaskStore()
        t1 = store.create_task(input_path="/tmp/a.mp4")
        store.create_task(input_path="/tmp/b.mp4")
        store.update_status(t1.id, TaskStatus.PROCESSING)
        processing = store.list_tasks(status=TaskStatus.PROCESSING)
        assert len(processing) == 1
        assert processing[0].id == t1.id

    def test_update_status_lifecycle(self):
        store = TaskStore()
        task = store.create_task(input_path="/tmp/v.mp4")
        store.update_status(task.id, TaskStatus.PROCESSING, progress=0.5)
        store.update_status(task.id, TaskStatus.COMPLETED, progress=1.0)
        final = store.get_task(task.id)
        assert final.status == TaskStatus.COMPLETED
        assert final.completed_at is not None

    def test_cancel_task(self):
        store = TaskStore()
        task = store.create_task(input_path="/tmp/v.mp4")
        cancelled = store.cancel_task(task.id)
        assert cancelled.status == TaskStatus.CANCELLED

    def test_delete_task(self):
        store = TaskStore()
        task = store.create_task(input_path="/tmp/v.mp4")
        assert store.delete_task(task.id) is True
        assert store.get_task(task.id) is None
        assert store.delete_task(task.id) is False

    def test_count_tasks(self):
        store = TaskStore()
        assert store.count_tasks() == 0
        store.create_task(input_path="/tmp/a.mp4")
        store.create_task(input_path="/tmp/b.mp4")
        assert store.count_tasks() == 2

    def test_update_nonexistent_returns_none(self):
        store = TaskStore()
        result = store.update_status("fake", TaskStatus.PROCESSING)
        assert result is None

    def test_to_dict_serialization(self):
        store = TaskStore()
        task = store.create_task(input_path="/tmp/v.mp4", metadata={"fps": 60})
        d = task.to_dict()
        assert isinstance(d["id"], str)
        assert d["status"] == "queued"
        assert d["metadata"]["fps"] == 60
        assert "created_at" in d
        assert "updated_at" in d
