"""Tests for API key authentication."""

import pytest
from db.models import ApiKey, Base
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from web.auth import verify_api_key

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def db_engine(tmp_path):
    """Create a fresh SQLite database for each test."""
    db_url = f"sqlite:///{tmp_path}/test_auth.db"
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


@pytest.fixture()
def _test_session_local(db_engine):
    """Session factory bound to the test engine."""
    return sessionmaker(bind=db_engine, expire_on_commit=False, class_=Session)


@pytest.fixture()
def app_with_auth(db_engine, _test_session_local):
    """Create a minimal FastAPI app with the verify_api_key dependency."""
    _app = FastAPI()

    def _override_get_session():
        s = _test_session_local()
        try:
            yield s
        finally:
            s.close()

    from db.engine import get_session

    _app.dependency_overrides[get_session] = _override_get_session

    @_app.post("/protected", dependencies=[Depends(verify_api_key)])
    async def protected():
        return {"ok": True}

    @_app.get("/public")
    async def public():
        return {"ok": True}

    return _app


@pytest.fixture()
def api_key_obj(_test_session_local):
    """Create and persist an ApiKey directly via the test engine."""
    api_key, raw_key = ApiKey.generate_key(name="test-key")
    with _test_session_local() as s:
        s.add(api_key)
        s.commit()
    return api_key, raw_key


# ── Unit tests for ApiKey model ───────────────────────────────────────────────


class TestApiKeyModel:
    def test_generate_key_returns_raw_key(self):
        _api_key, raw_key = ApiKey.generate_key(name="unit")
        assert raw_key.startswith("vr180_")
        assert len(raw_key) > 10

    def test_hash_key_is_deterministic(self):
        h1 = ApiKey.hash_key("test123")
        h2 = ApiKey.hash_key("test123")
        assert h1 == h2

    def test_hash_key_differs_for_different_inputs(self):
        h1 = ApiKey.hash_key("test123")
        h2 = ApiKey.hash_key("test456")
        assert h1 != h2

    def test_verify_returns_true_for_matching_key(self, api_key_obj):
        api_key, raw_key = api_key_obj
        assert api_key.verify(raw_key) is True

    def test_verify_returns_false_for_wrong_key(self, api_key_obj):
        api_key, _ = api_key_obj
        assert api_key.verify("vr180_wrong") is False

    def test_verify_returns_false_when_inactive(self, api_key_obj):
        api_key, raw_key = api_key_obj
        api_key.active = False
        assert api_key.verify(raw_key) is False

    def test_key_stored_as_hash(self, api_key_obj, _test_session_local):
        _api_key, raw_key = api_key_obj
        # Raw key should NOT appear in the DB
        from sqlalchemy import select

        with _test_session_local() as s:
            result = s.execute(select(ApiKey))
            all_keys = result.scalars().all()
            for k in all_keys:
                assert raw_key not in k.key_hash or k.key_hash == ApiKey.hash_key(raw_key)


# ── Integration tests for verify_api_key dependency ──────────────────────────


class TestAuthDependency:
    def test_missing_header_returns_401(self, app_with_auth):
        client = TestClient(app_with_auth)
        resp = client.post("/protected")
        assert resp.status_code == 401
        assert "Missing X-API-Key" in resp.json()["detail"]

    def test_invalid_key_returns_401(self, app_with_auth):
        client = TestClient(app_with_auth)
        resp = client.post("/protected", headers={"X-API-Key": "bad_key"})
        assert resp.status_code == 401

    def test_valid_key_returns_200(self, app_with_auth, api_key_obj):
        _, raw_key = api_key_obj
        client = TestClient(app_with_auth)
        resp = client.post("/protected", headers={"X-API-Key": raw_key})
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}

    def test_inactive_key_returns_401(self, app_with_auth, api_key_obj, _test_session_local):
        api_key, raw_key = api_key_obj
        with _test_session_local() as s:
            obj = s.get(type(api_key), api_key.id)
            obj.active = False
            s.commit()
        client = TestClient(app_with_auth)
        resp = client.post("/protected", headers={"X-API-Key": raw_key})
        assert resp.status_code == 401

    def test_public_endpoint_no_auth(self, app_with_auth):
        client = TestClient(app_with_auth)
        resp = client.get("/public")
        assert resp.status_code == 200
