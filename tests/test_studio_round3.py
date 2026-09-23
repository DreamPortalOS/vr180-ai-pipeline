"""Tests: VR180 apache backend, media files, settings API, G-11 constants."""

from __future__ import annotations

from pathlib import Path

import pytest

from studio.nodes.convert import Vr180ConvertNode
from studio.settings import StudioSettings


def test_vr180_apache_backend_param() -> None:
    import studio.nodes.convert as conv

    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))

        class R:
            returncode = 0
            stderr = ""
            stdout = ""

        # create output so node accepts
        out = Path(cmd[cmd.index("--output") + 1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"x")
        return R()

    import subprocess as sp

    orig = sp.run
    conv.subprocess.run = fake_run  # type: ignore[attr-defined]
    try:
        # need an input file
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "in.mp4"
            src.write_bytes(b"x")
            Vr180ConvertNode().run(
                params={"mode": "cli", "backend": "apache", "eye_size": 64},
                inputs={"video": str(src)},
                work_dir=td,
                node_id="vr",
            )
    finally:
        conv.subprocess.run = orig  # type: ignore[attr-defined]
    assert calls, "run_pipeline not invoked"
    cmd = calls[0]
    assert "--depth-model" in cmd
    assert cmd[cmd.index("--depth-model") + 1] == "depth-anything"
    assert cmd[cmd.index("--stereo-model") + 1] == "default"
    assert "--no-temporal" in cmd


def test_g11_constants_already_optimal() -> None:
    from scripts.check_source_quality import ANCHOR_TEXTURE_WINDOW

    assert ANCHOR_TEXTURE_WINDOW == 31


def test_settings_load_from_work_root(tmp_path, monkeypatch) -> None:
    for env in (
        "STUDIO_LITELLM_BASE_URL",
        "STUDIO_LITELLM_API_KEY",
        "STUDIO_LITELLM_MODEL",
        "OPENAI_API_KEY",
        "STUDIO_SETTINGS_FILE",
    ):
        monkeypatch.delenv(env, raising=False)
    path = tmp_path / "studio_settings.json"
    path.write_text(
        '{"litellm_base_url":"http://127.0.0.1:4000","litellm_model":"local-model","litellm_api_key":"sk-test"}',
        encoding="utf-8",
    )
    s = StudioSettings.load(path)
    assert s.litellm_base_url == "http://127.0.0.1:4000"
    assert s.litellm_model == "local-model"
    assert s.litellm_api_key == "sk-test"


def test_media_and_settings_api(tmp_path, monkeypatch) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from studio.server import create_app

    client = TestClient(create_app(default_work_dir=str(tmp_path / "w")))
    # settings get/post
    res = client.post(
        "/api/settings",
        json={"litellm_base_url": "http://127.0.0.1:4000", "litellm_model": "glm-5.2", "litellm_api_key": "sk-abc"},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["litellm_base_url"] == "http://127.0.0.1:4000"
    assert body["litellm_api_key_set"] is True
    assert body["litellm_api_key_masked"] != "sk-abc"

    # Force read from this app's work_root, not ambient STUDIO_SETTINGS_FILE
    monkeypatch.setenv("STUDIO_SETTINGS_FILE", str(tmp_path / "w" / "studio_settings.json"))
    res = client.get("/api/settings")
    assert res.status_code == 200
    assert res.json()["litellm_model"] == "glm-5.2"

    # media file inside work_root
    work = tmp_path / "w"
    media = work / "out.png"
    media.parent.mkdir(parents=True, exist_ok=True)
    media.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    res = client.get("/api/media", params={"path": str(media)})
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("image/")

    # reject path outside
    outside = Path(__file__).resolve()  # repo file
    res = client.get("/api/media", params={"path": str(outside)})
    assert res.status_code == 403
