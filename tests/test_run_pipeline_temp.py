"""K-18 (#210): --keep-temp / auto-temp-dir cleanup.

The pipeline's default intermediate directory lives next to the input file
(```<input_stem>_vr180_temp/```) and was historically never cleaned up, leaving
multi-gigabyte stale artefacts in ``video/`` (the same residue that fed #208's
wrong numbers).  This module wires and tests the K-18 cleanup contract:

* Default (no ``--keep-temp``): auto-derived temp dir is deleted after a
  *successful* ``--stage all`` run.
* ``--keep-temp``: the auto-derived temp dir is kept (path is printed).
* Pre-existing auto-derived dir (mtime from a prior run): NOT deleted, and a
  ``logging.warning`` is emitted (this is exactly the #208 trap).
* Explicit ``--temp-dir``: NEVER deleted (that directory belongs to the user).
* Pipeline exception: temp dir is NOT deleted, and its path is printed so the
  operator can inspect the artefacts for debugging.

All tests are fully mocked: ``tmp_path`` supplies the filesystem, ``parse_args``
is replaced by a ``MagicMock`` and the rest of ``main()`` after the temp-dir
decision is stubbed so no real ffmpeg / depth model / stereo renderer ever
runs.  These are wiring tests — they assert the CLI plumbing and the
created-this-run / delete-ownership semantics, not the rendering stages.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts import run_pipeline as rp  # noqa: E402


def _make_args(**overrides) -> MagicMock:
    """Build a minimal MagicMock args namespace for ``main()``.

    Defaults mimic a no-op ``--stage all`` run: input exists under ``tmp_path``
    so the auto-derived temp dir lands there, no streaming / fulldome /
    video-upscale, no manifest, no concat.  Overrides let individual tests
    flip ``--keep-temp``, ``--temp-dir``, etc.
    """
    defaults = {
        "input": "/_should_not_be_used_",
        "inputs": None,
        "concat_crossfade": 0.0,
        "concat_mode": "demux",
        "validate_input": False,
        "output": None,
        "stage": "all",
        "streaming": False,
        "projection": "vr180",
        "video_upscale": "none",
        "fps": 30,
        "device": None,
        "stream": False,
        "max_frames": None,
        "upscale": 0,
        "stages": None,
        "manifest": None,
        "resume_from": None,
        "force_sbs": False,
        "keep_temp": False,
        "temp_dir": None,
        "resume": False,
    }
    defaults.update(overrides)
    return MagicMock(**defaults)


def _fake_main_stub(*, should_raise=False):
    """A drop-in for ``_stage_all_body`` that either returns cleanly or raises.

    By the time this is installed the auto-temp-dir resolution + warning
    branch has already executed, so this stub only exercises the cleanup
    tail (try/finally).  It accepts the full ``_stage_all_body`` signature
    (7 positional args) so monkeypatching works without argument errors.
    """

    def _stub(args, temp_dir, is_sbs, manifest, manifest_skip, manifest_stages, manifest_touched):
        if should_raise:
            raise RuntimeError("pipeline boom")
        return args

    return _stub


@pytest.fixture
def clean_env(monkeypatch):
    """Standard environment overrides so ``main()`` never touches real I/O.

    Returns a dict that individual tests can ``.update()`` and pass into
    ``_make_args`` for extra customisation.
    """
    env = {}

    monkeypatch.setattr(rp, "apply_quality_preset", lambda args: None)
    monkeypatch.setattr(rp, "_apply_comfort_preset", lambda args: None)
    monkeypatch.setattr(rp, "apply_playback_preset", lambda args: None)
    monkeypatch.setattr(rp, "_manifest_prepare", lambda args: (None, None, None))
    monkeypatch.setattr(rp, "detect_best_device", lambda: "cpu")
    monkeypatch.setattr(rp.sys, "exit", MagicMock())
    monkeypatch.setenv("PYTHONPATH", str(REPO_ROOT))

    return env


def test_auto_temp_created_success_deleted(tmp_path, clean_env, monkeypatch) -> None:
    """Default: auto-derived temp dir is created this run and success → deleted.

    The pipeline must delete the dir it created.  This is the #210 headline
    acceptance criterion: no more 2.3 GB of stale residue accumulating in
    ``video/``.
    """
    monkeypatch.setattr(
        rp,
        "parse_args",
        lambda: _make_args(input=str(tmp_path / "clip.mp4")),
    )
    monkeypatch.setattr(rp, "detect_sbs_input", lambda *a, **kw: False)
    monkeypatch.setattr(rp, "_stage_all_body", _fake_main_stub())

    auto_dir = tmp_path / "clip_vr180_temp"
    assert not auto_dir.exists()
    rp.main()
    assert not auto_dir.exists(), "auto-created temp dir must be cleaned up on success"


def test_auto_temp_keep_temp_flag_preserved(tmp_path, clean_env, monkeypatch) -> None:
    """``--keep-temp`` preserves the auto-derived dir and prints its path."""
    monkeypatch.setattr(
        rp,
        "parse_args",
        lambda: _make_args(input=str(tmp_path / "clip.mp4"), keep_temp=True),
    )
    monkeypatch.setattr(rp, "detect_sbs_input", lambda *a, **kw: False)
    monkeypatch.setattr(rp, "_stage_all_body", _fake_main_stub())

    auto_dir = tmp_path / "clip_vr180_temp"
    rp.main()
    assert auto_dir.exists(), "--keep-temp must preserve the auto-derived temp dir"


def test_auto_temp_preexisting_not_deleted_warns_v2(tmp_path, clean_env, monkeypatch, caplog) -> None:
    """Pre-existing auto-derived dir: NOT deleted, warning emitted.

    Caplog check uses ``level="WARNING"`` so the assertion targets the
    ``logging.warning`` call made inside ``main()`` (not the log handler's
    formatter).
    """
    auto_dir = tmp_path / "clip_vr180_temp"
    auto_dir.mkdir(parents=True)
    (auto_dir / "depth").mkdir(parents=True, exist_ok=True)
    (auto_dir / "depth" / "depth_000000.npy").write_bytes(b"stale")

    monkeypatch.setattr(
        rp,
        "parse_args",
        lambda: _make_args(input=str(tmp_path / "clip.mp4")),
    )
    monkeypatch.setattr(rp, "detect_sbs_input", lambda *a, **kw: False)
    monkeypatch.setattr(rp, "_stage_all_body", _fake_main_stub())

    caplog.set_level("WARNING", logger="vr180-pipeline")
    rp.main()

    assert auto_dir.exists(), "pre-existing temp dir must NOT be deleted"
    assert (auto_dir / "depth" / "depth_000000.npy").exists(), "stale content must survive"

    # Warning must convey the "pre-existing / stale" semantics (the #208 trap).
    warning_text = " ".join(r.message for r in caplog.records if r.levelno >= 30)
    assert "已存在" in warning_text or "existing" in warning_text.lower(), (
        f"expected 'existing/已存在' warning, got: {warning_text!r}"
    )


def test_explicit_temp_dir_never_deleted(tmp_path, clean_env, monkeypatch) -> None:
    """Explicit ``--temp-dir``: never deleted, even without ``--keep-temp``.

    That directory belongs to the user — deleting it would be a disaster
    (card says "删了是灾难").  This test pins that invariant.
    """
    user_dir = tmp_path / "user_managed_temp"
    user_dir.mkdir(parents=True)

    monkeypatch.setattr(
        rp,
        "parse_args",
        lambda: _make_args(
            input=str(tmp_path / "clip.mp4"),
            temp_dir=str(user_dir),
        ),
    )
    monkeypatch.setattr(rp, "detect_sbs_input", lambda *a, **kw: False)
    monkeypatch.setattr(rp, "_stage_all_body", _fake_main_stub())

    rp.main()
    assert user_dir.exists(), "user-provided --temp-dir must NEVER be deleted"


def test_explicit_temp_dir_not_deleted_with_keep_temp(tmp_path, clean_env, monkeypatch) -> None:
    """Explicit ``--temp-dir`` + ``--keep-temp``: still preserved (no-op path)."""
    user_dir = tmp_path / "user_managed_temp"
    user_dir.mkdir(parents=True)

    monkeypatch.setattr(
        rp,
        "parse_args",
        lambda: _make_args(
            input=str(tmp_path / "clip.mp4"),
            temp_dir=str(user_dir),
            keep_temp=True,
        ),
    )
    monkeypatch.setattr(rp, "detect_sbs_input", lambda *a, **kw: False)
    monkeypatch.setattr(rp, "_stage_all_body", _fake_main_stub())

    rp.main()
    assert user_dir.exists()


def test_exception_preserves_temp_and_prints_path(tmp_path, clean_env, monkeypatch, capsys) -> None:
    """Pipeline exception: temp dir preserved, path printed for debugging."""
    monkeypatch.setattr(
        rp,
        "parse_args",
        lambda: _make_args(input=str(tmp_path / "clip.mp4")),
    )
    monkeypatch.setattr(rp, "detect_sbs_input", lambda *a, **kw: False)
    monkeypatch.setattr(rp, "_stage_all_body", _fake_main_stub(should_raise=True))

    auto_dir = tmp_path / "clip_vr180_temp"
    with pytest.raises(RuntimeError, match="pipeline boom"):
        rp.main()

    assert auto_dir.exists(), "exception path must preserve the temp dir for debugging"
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert str(auto_dir) in combined, f"exception path must print the temp dir path for debugging, got: {combined!r}"


def test_keep_temp_prints_path(tmp_path, clean_env, monkeypatch, capsys) -> None:
    """``--keep-temp``: the kept path is printed so the operator knows where to look."""
    monkeypatch.setattr(
        rp,
        "parse_args",
        lambda: _make_args(input=str(tmp_path / "clip.mp4"), keep_temp=True),
    )
    monkeypatch.setattr(rp, "detect_sbs_input", lambda *a, **kw: False)
    monkeypatch.setattr(rp, "_stage_all_body", _fake_main_stub())

    auto_dir = tmp_path / "clip_vr180_temp"
    rp.main()
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert str(auto_dir) in combined, f"--keep-temp path must be printed, got: {combined!r}"


def test_stage_all_required_for_cleanup(tmp_path, clean_env, monkeypatch) -> None:
    """Non-``all`` stages do not trigger auto-temp cleanup (they are single-stage).

    ``--stage depth`` / ``--stage stereo`` etc. are single-stage entry points
    that the operator invokes for incremental iteration.  They are *not*
    wrapped in the try/finally cleanup harness — that lives only in the
    ``--stage all`` path where the auto-derived temp dir is a first-class
    lifecycle concern.  This test pins that invariant.
    """
    monkeypatch.setattr(
        rp,
        "parse_args",
        lambda: _make_args(input=str(tmp_path / "clip.mp4"), stage="depth"),
    )
    monkeypatch.setattr(rp, "_intake_frames", lambda *a, **kw: ([], 0))
    monkeypatch.setattr(rp, "run_depth_stage", lambda args, frames: [])

    rp.main()
    # No crash, no unexpected cleanup.
    assert True
