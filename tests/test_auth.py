"""Tests for API Key authentication (T2-REDO)."""

import pytest
from db.engine import SessionLocal
from db.models import APIKey, User
from fastapi.testclient import TestClient
from web.app import app
from web.auth import hash_key, verify_key

# ── helpers ────────────────────────────────────────────────────────────────────


def _seed_key(plain: str = "vr180_test_key_abcdef123456"):
    """Create a user + API key in the conftest-managed test DB."""
    db = SessionLocal()
    try:
        # Ensure the user exists (ignore if already present)
        existing = db.query(User).filter(User.id == "test-auth-user").first()
        if existing is None:
            user = User(id="test-auth-user")
            db.add(user)
            db.flush()
        else:
            user = existing

        api_key = APIKey(
            key_hash=hash_key(plain),
            name="test-key",
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
    return plain


def _clear_keys():
    """Remove all API keys from the test DB."""
    db = SessionLocal()
    try:
        db.query(APIKey).delete()
        db.commit()
    finally:
        db.close()


@pytest.fixture(autouse=True)
def _setup_db():
    """Ensure fresh DB + one active key per test."""
    _clear_keys()
    _seed_key()
    yield
    _clear_keys()


@pytest.fixture
def client():
    return TestClient(app)


# ── Unit: passlib helpers ─────────────────────────────────────────────────────


class TestHashVerify:
    def test_hash_and_verify_match(self):
        plain = "vr180_my_secret_key_xyz"
        h = hash_key(plain)
        assert verify_key(plain, h) is True

    def test_hash_and_verify_mismatch(self):
        h = hash_key("correct_key")
        assert verify_key("wrong_key", h) is False

    def test_hash_is_different_each_time(self):
        h1 = hash_key("same")
        h2 = hash_key("same")
        assert h1 != h2  # bcrypt salt ensures different hashes
        assert verify_key("same", h1) is True
        assert verify_key("same", h2) is True


# ── Integration: FastAPI dependency ────────────────────────────────────────────


class TestAuthHeader:
    """Verify that write endpoints reject requests without a valid key."""

    def test_create_task_missing_key(self, client):
        resp = client.post("/tasks", json={"input_path": "/tmp/test.mp4"})
        assert resp.status_code == 401
        assert resp.json()["detail"] == "Missing X-API-Key header"

    def test_create_task_invalid_key(self, client):
        resp = client.post(
            "/tasks",
            json={"input_path": "/tmp/test.mp4"},
            headers={"X-API-Key": "invalid_key"},
        )
        assert resp.status_code == 401
        assert resp.json()["detail"] == "Invalid or inactive API key"

    def test_create_task_valid_key(self, client):
        resp = client.post(
            "/tasks",
            json={"input_path": "/tmp/test.mp4"},
            headers={"X-API-Key": "vr180_test_key_abcdef123456"},
        )
        assert resp.status_code == 201

    def test_update_task_missing_key(self, client):
        resp = client.patch("/tasks/fake-id", json={"status": "processing"})
        assert resp.status_code == 401

    def test_delete_task_missing_key(self, client):
        resp = client.delete("/tasks/fake-id")
        assert resp.status_code == 401

    def test_cancel_task_missing_key(self, client):
        resp = client.post("/tasks/fake-id/cancel")
        assert resp.status_code == 401

    # ── Read endpoints should work WITHOUT auth ──

    def test_health_no_auth(self, client):
        assert client.get("/health").status_code == 200

    def test_list_tasks_no_auth(self, client):
        assert client.get("/tasks").status_code == 200

    def test_get_task_no_auth(self, client):
        assert client.get("/tasks/fake-id").status_code == 404  # not 401

    # ── Inactive key ──

    def test_inactive_key_rejected(self, client):
        """Set the seeded key to inactive, then verify it's rejected."""
        db = SessionLocal()
        try:
            key = db.query(APIKey).first()
            key.is_active = False
            db.commit()
        finally:
            db.close()

        resp = client.post(
            "/tasks",
            json={"input_path": "/tmp/test.mp4"},
            headers={"X-API-Key": "vr180_test_key_abcdef123456"},
        )
        assert resp.status_code == 401
        assert resp.json()["detail"] == "Invalid or inactive API key"

    # ── v1 endpoints ──

    def test_create_task_v1_missing_key(self, client):
        resp = client.post(
            "/api/v1/tasks",
            files={"file": ("test.mp4", b"fake video content")},
        )
        assert resp.status_code == 401

    def test_delete_v1_missing_key(self, client):
        resp = client.delete("/api/v1/tasks/fake-id")
        assert resp.status_code == 401

    def test_cancel_v1_missing_key(self, client):
        resp = client.post("/api/v1/tasks/fake-id/cancel")
        assert resp.status_code == 401

    def test_delete_result_v1_missing_key(self, client):
        resp = client.delete("/api/v1/results/fake-id")
        assert resp.status_code == 401

    def test_generate_video_missing_key(self, client):
        resp = client.post(
            "/api/v1/generate",
            json={"prompt": "a cat", "provider": "kling"},
        )
        assert resp.status_code == 401
