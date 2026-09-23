"""Tests for scripts/segment_chain.py — chained multi-scene generation.

A *fake* provider is injected into :func:`segment_chain.run_chain` (no
network, no keys, no models, no real ffmpeg).  The fake writes tiny byte
blobs as its "video" and "last frame" artefacts and records every call so the
tests can assert the exact chaining behaviour — scene 0 as text-to-video,
every later scene as image-to-video seeded by ``scene_{N-1}_last.png``.

Budget-gate coverage (issue #328) points the usage ledger at ``tmp_path`` via
``VR180_LEDGER_PATH`` so a blocked scene can be staged deterministically
without touching the operator's real ``~/.vr180/usage_ledger.jsonl``.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
import scripts.segment_chain as sc

from integrations import usage_ledger
from integrations.base import GenerationResult, VideoGenProvider


class FakeChainProvider(VideoGenProvider):
    """Injected provider: records calls, returns local-path artefacts.

    Each call fabricates a ``GenerationResult`` whose ``video_url`` and
    ``content.last_frame`` point at real files under *scratch_dir* so
    ``run_chain``'s ``_save_asset`` copy step succeeds offline.  When
    *ledger_path* / *tokens_per_scene* are set, the fake also appends a usage
    record after each call — mirroring how a real provider (seedance) books
    spend, which the budget-gate test needs to stage the cap crossing.
    """

    def __init__(
        self,
        scratch_dir: Path,
        *,
        tokens_per_scene: int = 0,
        ledger_path: Path | None = None,
    ) -> None:
        super().__init__(api_key="fake")
        self.scratch_dir = scratch_dir
        self.scratch_dir.mkdir(parents=True, exist_ok=True)
        self.tokens_per_scene = tokens_per_scene
        self.ledger_path = ledger_path
        self.calls: list[dict] = []

    def _load_api_key(self) -> str:
        return "fake"

    def _make_result(self, image: str | None, prompt: str, duration: int) -> GenerationResult:
        index = len(self.calls) + 1
        video = self.scratch_dir / f"fake_{index}.mp4"
        video.write_bytes(b"fake-video-bytes")
        frame = self.scratch_dir / f"fake_{index}.png"
        frame.write_bytes(b"\x89PNGfake")

        self.calls.append({"image": image, "prompt": prompt, "duration": duration, "return_last_frame": True})

        if self.tokens_per_scene and self.ledger_path is not None:
            usage_ledger.record_generation(
                provider="fake",
                usage={
                    "completion_tokens": self.tokens_per_scene,
                    "total_tokens": self.tokens_per_scene,
                },
                task_id=f"fake-{index}",
                path=self.ledger_path,
            )

        return GenerationResult(
            video_url=str(video),
            provider=self.provider_name,
            job_id=f"fake-{index}",
            metadata={"status": "succeeded", "content": {"last_frame": str(frame)}},
        )

    def generate(
        self, prompt: str, duration: int = 5, aspect_ratio: str = "16:9", fps: int = 24, **kwargs
    ) -> GenerationResult:
        del aspect_ratio, fps, kwargs
        return self._make_result(None, prompt, duration)

    def generate_from_image(
        self, image_path: str, prompt: str = "", duration: int = 5, aspect_ratio: str = "16:9", **kwargs
    ) -> GenerationResult:
        del aspect_ratio, kwargs
        return self._make_result(image_path, prompt, duration)


def _write_plan(path: Path, scenes: list[dict]) -> Path:
    path.write_text(json.dumps(scenes), encoding="utf-8")
    return path


def _three_scenes() -> list[dict]:
    return [
        {"prompt": "fly over mountains", "duration": 2},
        {"prompt": "descend into the valley", "duration": 2},
        {"prompt": "sweep across the lake", "duration": 2},
    ]


class TestRunChain:
    def test_three_scene_chain(self, tmp_path: Path) -> None:
        """Scene 0 is text-to-video; scenes 1..2 seed from the prior last frame."""
        provider = FakeChainProvider(tmp_path / "provider")
        rc = sc.run_chain(_three_scenes(), tmp_path / "out", provider)

        assert rc == 0
        assert len(provider.calls) == 3

        assert provider.calls[0]["image"] is None
        assert provider.calls[1]["image"] == str(sc.scene_frame_path(tmp_path / "out", 0))
        assert provider.calls[2]["image"] == str(sc.scene_frame_path(tmp_path / "out", 1))
        assert all(call["return_last_frame"] is True for call in provider.calls)

        for idx in range(3):
            assert (tmp_path / "out" / f"scene_{idx}.mp4").exists()
            assert (tmp_path / "out" / f"scene_{idx}_last.png").exists()

    def test_first_scene_seed_image(self, tmp_path: Path) -> None:
        """A first scene with ``seed_image`` becomes image-to-video for scene 0."""
        provider = FakeChainProvider(tmp_path / "provider")
        plan = [{"prompt": "p0", "duration": 2, "seed_image": "first.png"}, {"prompt": "p1", "duration": 2}]
        rc = sc.run_chain(plan, tmp_path / "out", provider)

        assert rc == 0
        assert provider.calls[0]["image"] == "first.png"
        assert provider.calls[1]["image"] == str(sc.scene_frame_path(tmp_path / "out", 0))

    def test_resume_skips_existing(self, tmp_path: Path) -> None:
        """--resume with every scene already on disk makes zero provider calls."""
        out = tmp_path / "out"
        out.mkdir()
        for idx in range(3):
            sc.scene_video_path(out, idx).write_bytes(b"v")
            sc.scene_frame_path(out, idx).write_bytes(b"f")

        provider = FakeChainProvider(tmp_path / "provider")
        rc = sc.run_chain(_three_scenes(), out, provider, resume=True)

        assert rc == 0
        assert provider.calls == []

    def test_resume_partial_continues_chain(self, tmp_path: Path) -> None:
        """Scene 0 done on disk; scenes 1..2 regenerate using scene_0_last.png."""
        out = tmp_path / "out"
        out.mkdir()
        sc.scene_video_path(out, 0).write_bytes(b"v")
        sc.scene_frame_path(out, 0).write_bytes(b"f")

        provider = FakeChainProvider(tmp_path / "provider")
        rc = sc.run_chain(_three_scenes(), out, provider, resume=True)

        assert rc == 0
        assert len(provider.calls) == 2
        assert provider.calls[0]["image"] == str(sc.scene_frame_path(out, 0))
        assert provider.calls[1]["image"] == str(sc.scene_frame_path(out, 1))

    def test_dry_run_zero_calls(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """--dry-run prints params + estimate and never touches the provider."""
        provider = FakeChainProvider(tmp_path / "provider")
        rc = sc.run_chain(_three_scenes(), tmp_path / "out", provider, dry_run=True)

        assert rc == 0
        assert provider.calls == []
        assert not (tmp_path / "out" / "scene_0.mp4").exists()

        captured = capsys.readouterr().out
        assert "场景 0" in captured
        assert "预估" in captured
        assert "费用" in captured

    def test_budget_exceeded_stops_at_second_scene(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A token cap crossed after scene 0 blocks scene 1 and stops the chain."""
        ledger = tmp_path / "ledger.jsonl"
        monkeypatch.setenv("VR180_LEDGER_PATH", str(ledger))
        monkeypatch.setenv("VR180_BUDGET_CAP_TOKENS", "1000")
        # Pre-fill 900 tokens so scene 0 passes (90% -> warn) but its +100
        # recording pushes the cumulative to 1000, blocking scene 1.
        usage_ledger.record_generation(
            provider="seedance",
            usage={"completion_tokens": 900, "total_tokens": 900},
            task_id="pre-fill",
            path=ledger,
        )

        provider = FakeChainProvider(
            tmp_path / "provider",
            tokens_per_scene=100,
            ledger_path=ledger,
        )
        rc = sc.run_chain(
            _three_scenes(),
            tmp_path / "out",
            provider,
            budget_cap_tokens=1000,
        )

        assert rc == 2
        assert len(provider.calls) == 1
        # Scene 1 was never submitted, so its artefact is absent.
        assert not (tmp_path / "out" / "scene_1.mp4").exists()


class TestHelpers:
    def test_extract_last_frame_prefers_content_last_frame(self) -> None:
        result = GenerationResult(
            video_url="https://x/v.mp4",
            provider="seedance",
            metadata={
                "status": "succeeded",
                "content": {"video_url": "https://x/v.mp4", "last_frame": "https://x/f.png"},
            },
        )
        assert sc.extract_last_frame(result) == "https://x/f.png"

    def test_extract_last_frame_nested_outputs(self) -> None:
        result = GenerationResult(
            video_url="https://x/v.mp4",
            provider="seedance",
            metadata={"content": {"outputs": [{"image_url": "https://x/i.png"}]}},
        )
        assert sc.extract_last_frame(result) == "https://x/i.png"

    def test_extract_last_frame_missing_returns_none(self) -> None:
        result = GenerationResult(video_url="https://x/v.mp4", provider="seedance", metadata={})
        assert sc.extract_last_frame(result) is None

    def test_load_plan_valid(self, tmp_path: Path) -> None:
        plan_path = _write_plan(tmp_path / "plan.json", _three_scenes())
        assert sc.load_plan(plan_path) == _three_scenes()

    def test_load_plan_rejects_non_array(self, tmp_path: Path) -> None:
        plan_path = tmp_path / "plan.json"
        plan_path.write_text('{"prompt": "not an array"}', encoding="utf-8")
        with pytest.raises(ValueError):
            sc.load_plan(plan_path)

    def test_load_plan_rejects_missing_prompt(self, tmp_path: Path) -> None:
        plan_path = _write_plan(tmp_path / "plan.json", [{"duration": 2}])
        with pytest.raises(ValueError):
            sc.load_plan(plan_path)

    def test_load_plan_rejects_empty(self, tmp_path: Path) -> None:
        plan_path = _write_plan(tmp_path / "plan.json", [])
        with pytest.raises(ValueError):
            sc.load_plan(plan_path)

    def test_concat_writes_list_and_invokes_ffmpeg(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        out.mkdir()
        for idx in range(2):
            sc.scene_video_path(out, idx).write_bytes(b"v")

        with patch("scripts.segment_chain.subprocess.run") as run:
            run.return_value = subprocess.CompletedProcess([], 0, "", "")
            sc._concat(out, 2)

        list_path = out / "concat_list.txt"
        lines = list_path.read_text(encoding="utf-8").splitlines()
        assert lines == [
            f"file '{sc.scene_video_path(out, 0).as_posix()}'",
            f"file '{sc.scene_video_path(out, 1).as_posix()}'",
        ]

        cmd = run.call_args[0][0]
        assert cmd[0] == "ffmpeg"
        assert cmd[cmd.index("-f") + 1] == "concat"
        assert str(list_path) in cmd
        assert str(out / "chain.mp4") in cmd

    def test_concat_missing_scene_raises(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError):
            sc._concat(tmp_path / "out", 2)
