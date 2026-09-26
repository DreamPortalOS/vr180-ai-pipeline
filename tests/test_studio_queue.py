"""Server-side job queue + batch-shot submission (issue #419 队列与停止 / 只跑勾选镜头)."""

from __future__ import annotations

import time

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from studio.server import JobManager, create_app


@pytest.fixture()
def client(tmp_path):
    app = create_app(default_work_dir=str(tmp_path / "studio_work"))
    return TestClient(app)


def _shots(n: int) -> list[dict]:
    return [{"id": f"shot_{i + 1:02d}", "description": f"desc {i + 1}", "duration": 1} for i in range(n)]


def _wait_terminal(client: TestClient, job_ids: list[str], timeout: float = 60.0) -> dict:
    """Poll /api/jobs until every job reaches a terminal state."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        listed = {j["id"]: j for j in client.get("/api/jobs").json()}
        if all(listed[jid]["status"] in {"ok", "error", "cancelled"} for jid in job_ids):
            return listed
        time.sleep(0.1)
    raise AssertionError(f"jobs {job_ids} did not finish within {timeout}s")


def test_batch_shots_enqueues_one_job_per_shot(client, tmp_path) -> None:
    """3 checked shots => exactly 3 jobs submitted (只跑勾选镜头 acceptance)."""
    payload = {"shots": _shots(3), "work_dir": str(tmp_path / "batch")}
    res = client.post("/api/batch-shots", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert body["submitted"] == 3
    assert len(body["jobs"]) == 3
    job_ids = [j["job_id"] for j in body["jobs"]]
    assert len(set(job_ids)) == 3  # unique ids
    # The queue lists them.
    listed = client.get("/api/jobs").json()
    assert len(listed) == 3
    assert {j["id"] for j in listed} == set(job_ids)
    # Each enqueued job is a one-node project labelled with the shot id.
    assert [j["shot_id"] for j in body["jobs"]] == ["shot_01", "shot_02", "shot_03"]
    # Let the worker threads finish before tmp_path teardown (Windows file locks).
    final = _wait_terminal(client, job_ids)
    assert all(final[jid]["status"] == "ok" for jid in job_ids)


def test_batch_shots_rejects_empty_selection(client) -> None:
    res = client.post("/api/batch-shots", json={"shots": []})
    assert res.status_code == 400
    assert "no shots" in res.json()["detail"].lower()


def test_cancel_endpoint_404_for_unknown_job(client) -> None:
    res = client.post("/api/jobs/does-not-exist/cancel")
    assert res.status_code == 404


def test_stop_all_cancels_running_jobs(client, tmp_path) -> None:
    """Enqueue several jobs, then stop-all: every running job is flagged cancelled."""
    payload = {"shots": _shots(4), "work_dir": str(tmp_path / "batch2")}
    res = client.post("/api/batch-shots", json=payload)
    job_ids = [j["job_id"] for j in res.json()["jobs"]]
    stop = client.post("/api/jobs/stop-all").json()
    # At least the jobs we just enqueued were running when stop-all fired.
    assert stop["stopped"] >= 4
    final = _wait_terminal(client, job_ids)
    # A cancelled job (or one that finished first) lands in a terminal state.
    assert all(final[jid]["status"] in {"ok", "error", "cancelled"} for jid in job_ids)


def test_jobmanager_unit_create_cancel_list() -> None:
    jm = JobManager()
    j1 = jm.create("a", 3)
    j2 = jm.create("b", 5)
    assert {j["id"] for j in jm.list_dicts()} == {j1.id, j2.id}
    # cancel returns True for a running job and sets its event.
    assert jm.cancel(j1.id) is True
    assert j1.cancel_event.is_set()
    assert j1.cancelled is True
    # cancel of a finished job returns False.
    j2.finished_at = 1.0
    assert jm.cancel(j2.id) is False
    # get returns the live object.
    assert jm.get(j1.id) is j1
    assert jm.get("nope") is None


def test_jobmanager_cancel_all_counts_only_running() -> None:
    jm = JobManager()
    running = jm.create("r", 2)
    finished = jm.create("f", 2)
    finished.finished_at = 1.0
    n = jm.cancel_all()
    assert n == 1
    assert running.cancel_event.is_set()
