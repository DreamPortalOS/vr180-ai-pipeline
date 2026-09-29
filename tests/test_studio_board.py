"""分镜板 (manual storyboard board): text → image per shot, pick, shot → video, join."""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import pytest

from studio.board import (
    BOARD_ASPECTS,
    DOME_CONSTRAINT,
    board_prompt,
    board_to_stills,
    generate_shot_images,
    safe_shot_id,
)
from studio.nodes import get_node_class
from studio.nodes.image_gateway import ImageResult


class _FakeClient:
    def __init__(self, fail: int = 0) -> None:
        self.jobs = []
        self.fail = fail

    def generate_many(self, jobs, concurrency=3):
        self.jobs.extend(jobs)
        out = []
        for i, job in enumerate(jobs):
            if i < self.fail:
                out.append(ImageResult(job.key, None, "HTTPError: 502", 3))
                continue
            job.out_path.parent.mkdir(parents=True, exist_ok=True)
            job.out_path.write_bytes(b"\x89PNG fake")
            out.append(ImageResult(job.key, str(job.out_path), None, 1))
        return out


def test_board_prompt_prefixes_dome_constraint() -> None:
    text = board_prompt("sunken stone corridor", "dome", "teal volumetric light")
    assert text.startswith(DOME_CONSTRAINT.rstrip("."))
    assert "sunken stone corridor." in text and "teal volumetric light." in text
    assert "Fisheye" not in board_prompt("x", "16:9")


def test_board_prompt_rejects_empty() -> None:
    with pytest.raises(ValueError, match="分镜描述为空"):
        board_prompt("   ", "dome")


def test_safe_shot_id() -> None:
    assert safe_shot_id("../../etc/passwd") == "etc_passwd"
    assert safe_shot_id("") == "shot"
    assert safe_shot_id("S1 走廊") == "S1"


def test_generate_with_client_returns_candidates(tmp_path) -> None:
    client = _FakeClient(fail=1)
    res = generate_shot_images(prompt="corridor", aspect="dome", n=3, out_dir=tmp_path, shot_id="s1", client=client)
    assert res["provider"] == "gateway"
    assert len(res["images"]) == 2 and len(res["errors"]) == 1
    assert all(Path(p).parent == tmp_path for p in res["images"])
    assert client.jobs[0].prompt.startswith("Fisheye fulldome")


def test_generate_caps_variants(tmp_path) -> None:
    client = _FakeClient()
    res = generate_shot_images(prompt="x", aspect="16:9", n=99, out_dir=tmp_path, shot_id="s", client=client)
    assert len(res["images"]) == 4


def test_generate_without_gateway_writes_placeholders(tmp_path) -> None:
    res = generate_shot_images(prompt="x", aspect="dome", n=2, out_dir=tmp_path, shot_id="s", settings=None)
    assert res["provider"] == "mock"
    assert len(res["images"]) == 2 and all(Path(p).is_file() for p in res["images"])


def test_board_to_stills_skips_unpicked(tmp_path) -> None:
    img = tmp_path / "a.png"
    img.write_bytes(b"x")
    shots = [
        {"id": "s1", "prompt": "p1", "image": str(img), "duration": 3, "motion": "dolly_in"},
        {"id": "s2", "prompt": "p2"},
        {"id": "s3", "prompt": "p3", "image": str(img), "motion": "barrel_roll"},
    ]
    out = board_to_stills(shots, "dome")
    assert [s["id"] for s in out["shots"]] == ["s1", "s3"]
    assert out["pending"] == ["s2"]
    assert out["shots"][1]["motion"] == "static"
    assert out["summary"]["aspect_ratio"] == "1:1"


def test_board_node_emits_stills_and_errors_when_empty(tmp_path) -> None:
    node = get_node_class("storyboard.board")()
    img = tmp_path / "a.png"
    img.write_bytes(b"x")
    out = node.run(
        params={"aspect": "16:9", "shots": [{"id": "s1", "prompt": "p", "image": str(img)}]},
        inputs={},
        work_dir=str(tmp_path),
        node_id="n1",
    )
    assert out["stills"]["shots"][0]["image"] == str(img)
    assert out["meta"]["aspect"] == "16:9"
    with pytest.raises(ValueError, match="分镜板"):
        node.run(params={"shots": [{"id": "s1", "prompt": "p"}]}, inputs={}, work_dir=str(tmp_path), node_id="n1")


def test_board_aspects_cover_all_routes() -> None:
    assert set(BOARD_ASPECTS) == {"dome", "16:9", "2:1"}


pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from studio.server import create_app  # noqa: E402


@pytest.fixture()
def client(tmp_path):
    return TestClient(create_app(default_work_dir=str(tmp_path / "w")))


def test_board_images_endpoint_offline_placeholders(client, tmp_path) -> None:
    res = client.post("/api/board/images", json={"shot_id": "s1", "prompt": "corridor", "aspect": "dome", "n": 2})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["provider"] == "mock" and len(body["images"]) == 2
    assert all(Path(p).is_relative_to(tmp_path / "w") for p in body["images"])


def test_board_images_endpoint_validation(client) -> None:
    assert client.post("/api/board/images", json={"prompt": "x", "aspect": "4:3"}).status_code == 400
    assert client.post("/api/board/images", json={"prompt": " ", "aspect": "dome"}).status_code == 400


def test_board_concat_rejects_outside_paths(client, tmp_path) -> None:
    assert client.post("/api/board/concat", json={"videos": []}).status_code == 400
    outside = Path(__file__)
    assert client.post("/api/board/concat", json={"videos": [str(outside)]}).status_code == 403


def test_board_concat_joins_clips(client, tmp_path) -> None:
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not on PATH")
    import subprocess

    work = tmp_path / "w"
    clips = []
    for i, color in enumerate(("red", "blue")):
        p = work / f"c{i}.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i", f"color={color}:s=64x64:d=1:r=12", "-pix_fmt", "yuv420p", str(p)],
            capture_output=True,
            check=True,
        )
        clips.append(str(p))
    res = client.post("/api/board/concat", json={"videos": clips, "filename": "film"})
    assert res.status_code == 200, res.text
    job_id = res.json()["job_id"]
    deadline = time.time() + 60
    while time.time() < deadline:
        job = next(j for j in client.get("/api/jobs").json() if j["id"] == job_id)
        if job["status"] in {"ok", "error", "cancelled"}:
            break
        time.sleep(0.1)
    assert job["status"] == "ok", job
    out = job["report"]["results"]["n_board_concat"]["outputs"]["video"]
    assert out.endswith("film.mp4") and Path(out).is_file()
