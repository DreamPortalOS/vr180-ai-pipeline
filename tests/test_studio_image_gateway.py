"""#415: real storyboard stills through the LiteLLM gateway — all HTTP faked."""

from __future__ import annotations

import base64
import threading
from pathlib import Path

import pytest

from studio.nodes.image_gateway import (
    GatewayImageClient,
    ImageJob,
    build_still_prompt,
    size_for_aspect,
)
from studio.nodes.production import BatchStillNode

PNG = b"\x89PNG\r\n\x1a\nfake"


def _client(post, get=lambda url, timeout: PNG, **kw) -> GatewayImageClient:
    return GatewayImageClient("http://gw:9400", "sk-test", post=post, get=get, sleep=lambda s: None, **kw)


def test_url_reply_is_downloaded(tmp_path: Path) -> None:
    seen = {}

    def post(url, body, headers, timeout):
        seen.update(url=url, body=body, auth=headers["Authorization"])
        return {"data": [{"url": "http://img/1.png"}]}

    res = _client(post).generate(ImageJob("a", "prompt", tmp_path / "a.png"))
    assert res.error is None and Path(res.path).read_bytes() == PNG
    assert seen["url"] == "http://gw:9400/v1/images/generations"
    assert seen["body"]["model"] == "agnes-image-2.5-flash"
    assert seen["auth"] == "Bearer sk-test"


def test_b64_reply_is_decoded(tmp_path: Path) -> None:
    def post(url, body, headers, timeout):
        return {"data": [{"b64_json": base64.b64encode(PNG).decode()}]}

    res = _client(post).generate(ImageJob("a", "p", tmp_path / "a.png"))
    assert Path(res.path).read_bytes() == PNG


def test_base_url_already_ending_in_v1() -> None:
    c = GatewayImageClient("http://gw/v1/", "k", post=lambda *a: {}, sleep=lambda s: None)
    assert c.endpoint() == "http://gw/v1/images/generations"


def test_retries_then_succeeds(tmp_path: Path) -> None:
    calls = {"n": 0}

    def post(url, body, headers, timeout):
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("gateway reset")
        return {"data": [{"url": "u"}]}

    res = _client(post, retries=2).generate(ImageJob("a", "p", tmp_path / "a.png"))
    assert res.error is None and res.attempts == 3 and calls["n"] == 3


def test_gives_up_after_retries(tmp_path: Path) -> None:
    def post(url, body, headers, timeout):
        raise ConnectionError("down")

    res = _client(post, retries=2).generate(ImageJob("a", "p", tmp_path / "a.png"))
    assert res.path is None and "ConnectionError" in res.error and res.attempts == 3


def test_reply_without_image_is_an_error(tmp_path: Path) -> None:
    res = _client(lambda *a: {"data": [{}]}, retries=0).generate(ImageJob("a", "p", tmp_path / "a.png"))
    assert res.path is None and "no image" in res.error


def test_concurrency_is_capped(tmp_path: Path) -> None:
    lock = threading.Lock()
    state = {"now": 0, "peak": 0}
    gate = threading.Event()

    def post(url, body, headers, timeout):
        with lock:
            state["now"] += 1
            state["peak"] = max(state["peak"], state["now"])
        gate.wait(0.05)
        with lock:
            state["now"] -= 1
        return {"data": [{"url": "u"}]}

    jobs = [ImageJob(str(i), "p", tmp_path / f"{i}.png") for i in range(7)]
    results = _client(post).generate_many(jobs, concurrency=2)
    assert [r.key for r in results] == [str(i) for i in range(7)]
    assert state["peak"] <= 2


def test_missing_settings_are_reported() -> None:
    with pytest.raises(ValueError, match="litellm_base_url"):
        GatewayImageClient("", "k")
    with pytest.raises(ValueError, match="litellm_api_key"):
        GatewayImageClient("http://gw", "")


def test_size_for_aspect() -> None:
    assert size_for_aspect("1:1") == "1024x1024"
    assert size_for_aspect("16:9") == "1024x576"
    assert size_for_aspect("9:16") == "576x1024"
    assert size_for_aspect("bogus") == "1024x1024"


def test_prompt_carries_theme_style_and_negatives() -> None:
    p = build_still_prompt(
        {"description": "slow push into the gorge", "aspect_ratio": "1:1"},
        {"theme": "red rock canyon FPV", "style": "golden hour", "negative": "cuts, text"},
    )
    assert p.startswith("slow push into the gorge")
    for piece in ("red rock canyon FPV", "golden hour", "120 degree", "1:1 composition", "Avoid: cuts, text"):
        assert piece in p


class _FakeClient:
    def __init__(self, fail: set[str]) -> None:
        self.fail = fail
        self.jobs = []

    def generate_many(self, jobs, concurrency=3):
        from studio.nodes.image_gateway import ImageResult

        self.jobs = jobs
        out = []
        for j in jobs:
            if j.key in self.fail:
                out.append(ImageResult(j.key, None, "HTTPError: 500", 3))
            else:
                j.out_path.write_bytes(PNG)
                out.append(ImageResult(j.key, str(j.out_path), None, 1))
        return out


SHOTS = [{"id": "shot_01", "index": 0, "description": "rim"}, {"id": "shot_02", "index": 1, "description": "gorge"}]


def test_node_gateway_variants_and_partial_failure(tmp_path: Path) -> None:
    fake = _FakeClient(fail={"shot_02|1"})
    stills = BatchStillNode._gateway_stills(SHOTS, {"theme": "canyon"}, {"variants": 2}, tmp_path, client=fake)
    assert len(fake.jobs) == 4
    assert stills[0]["image"].endswith("shot_01_v1.png") and len(stills[0]["variants"]) == 2
    # first variant failed, second succeeded -> the card still has an image
    assert stills[1]["image"].endswith("shot_02_v2.png") and "error" not in stills[1]
    assert "canyon" in stills[0]["still_prompt"]


def test_node_marks_a_fully_failed_shot(tmp_path: Path) -> None:
    fake = _FakeClient(fail={"shot_02|1", "shot_02|2"})
    stills = BatchStillNode._gateway_stills(SHOTS, {}, {"variants": 2}, tmp_path, client=fake)
    assert stills[1]["image"] is None and "HTTPError" in stills[1]["error"]
    assert stills[0]["image"]


def test_node_run_fails_only_when_every_shot_failed(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        BatchStillNode,
        "_gateway_stills",
        staticmethod(lambda shots, summary, params, out_dir: [{**s, "image": None, "error": "x"} for s in shots]),
    )
    with pytest.raises(RuntimeError, match="every shot failed"):
        BatchStillNode().run(
            params={"provider": "gateway"},
            inputs={"shots": {"shots": SHOTS, "summary": {}}},
            work_dir=str(tmp_path),
            node_id="n",
        )


def test_mock_stays_the_default() -> None:
    schema = {p["name"]: p for p in BatchStillNode.param_schema()}
    assert schema["provider"]["default"] == "mock"
