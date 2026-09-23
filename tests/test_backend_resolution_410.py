"""#410: worktree-aware DepthCrafter deploy root + sidecar records the backends that ran."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

from pipeline import depth_crafter as dc

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _make_worktree(tmp_path: Path) -> tuple[Path, Path]:
    main = tmp_path / "main"
    wt = main / ".claude" / "worktrees" / "wt1"
    (main / ".git" / "worktrees" / "wt1").mkdir(parents=True)
    wt.mkdir(parents=True)
    (wt / ".git").write_text(f"gitdir: {main / '.git' / 'worktrees' / 'wt1'}\n", encoding="utf-8")
    return main, wt


def test_main_checkout_root_parses_worktree_marker(tmp_path: Path) -> None:
    main, wt = _make_worktree(tmp_path)
    assert dc._main_checkout_root(wt) == main


def test_main_checkout_root_is_none_for_a_normal_checkout(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    assert dc._main_checkout_root(tmp_path) is None


def test_deploy_root_falls_back_to_main_checkout(tmp_path: Path) -> None:
    main, wt = _make_worktree(tmp_path)
    (main / "third_party" / "DepthCrafter").mkdir(parents=True)
    assert dc._deploy_root(wt) == main


def test_deploy_root_prefers_own_deployment(tmp_path: Path) -> None:
    main, wt = _make_worktree(tmp_path)
    (main / "third_party" / "DepthCrafter").mkdir(parents=True)
    (wt / "third_party" / "DepthCrafter").mkdir(parents=True)
    assert dc._deploy_root(wt) == wt


def test_deploy_root_stays_put_when_nothing_is_deployed(tmp_path: Path) -> None:
    _, wt = _make_worktree(tmp_path)
    assert dc._deploy_root(wt) == wt


@pytest.fixture
def run_pipeline_module():
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    try:
        import run_pipeline

        yield run_pipeline
    finally:
        sys.path.remove(str(PROJECT_ROOT / "scripts"))
        sys.modules.pop("run_pipeline", None)


def _capture_generation(run_pipeline_module, args, monkeypatch) -> dict:
    captured: dict = {}

    def fake_write_sidecar(path, *, immersive, generation):
        captured.update(generation)

    import pipeline.sidecar as sidecar

    monkeypatch.setattr(sidecar, "write_sidecar", fake_write_sidecar)
    run_pipeline_module._write_sidecar_from_args("out.mp4", "vr180", args)
    return captured


def test_sidecar_records_backends_that_ran(run_pipeline_module, monkeypatch) -> None:
    args = argparse.Namespace(
        output_width=2880,
        output_height=2880,
        preset="standalone",
        depth_backend_used="depth-anything",
        stereo_backend_used="default",
    )
    generation = _capture_generation(run_pipeline_module, args, monkeypatch)
    assert generation["depth_backend_used"] == "depth-anything"
    assert generation["stereo_backend_used"] == "default"


def test_sidecar_omits_backends_when_unknown(run_pipeline_module, monkeypatch) -> None:
    args = argparse.Namespace(output_width=2880, output_height=2880, preset="standalone")
    generation = _capture_generation(run_pipeline_module, args, monkeypatch)
    assert generation, "sidecar writer was not reached"
    assert "depth_backend_used" not in generation
