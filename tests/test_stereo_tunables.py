"""Tests for the Stage-2 throughput tunables (issue #217).

StereoCrafter Stage 2 (``inpainting_inference.py``) is the pipeline's heavy
bottleneck (≈66.7 s/frame on a 12 GB GPU).  Upstream exposes three
VRAM/throughput knobs — ``--frames_chunk`` / ``--overlap`` / ``--tile_num`` —
that decide how many frames are fed at once, how many overlap between chunks,
and how many spatial tiles each frame is split into.  Issue #217 wires these
out as opt-in constructor params + env vars on :class:`CLIBackend` (and
:class:`StereoCrafterRenderer`), with priority **explicit arg > env var >
None (flag omitted → upstream default, pre-#217 behaviour unchanged)**.

These tests inject a fake ``subprocess.run`` and assert the *command line*
only — no real inference, no model download, no GPU (mirrors the
``TestCLIBackendInference`` pattern in ``test_stereo_crafter.py``).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

from pipeline.stereo_crafter import CLIBackend

# ---------------------------------------------------------------------------
# Shared harness — build a backend + run render_video with everything
# subprocess/GPU-related stubbed, returning the captured Stage-2 command.
# ---------------------------------------------------------------------------


def _make_repo(tmp_path: Path) -> Path:
    """A repo dir with the real upstream Stage-2 entry (valid for construction)."""
    repo = tmp_path / "stereocrafter"
    repo.mkdir()
    (repo / "inpainting_inference.py").write_text("# stage 2")
    return repo


def _capture_stage2_cmd(backend: CLIBackend, tmp_path: Path) -> list[str]:
    """Run backend.render_video with stubs; return the captured Stage-2 argv.

    Every external dependency (CUDA guard, grid assembly, the subprocess
    itself, the SBS split) is stubbed so this runs on a bare CI box and never
    spawns a real process.  The Stage-2 SBS product is pre-created so the flow
    reaches the split step and returns normally.
    """
    depth_dir = tmp_path / "depth"
    depth_dir.mkdir()
    input_video = tmp_path / "input.mp4"
    input_video.write_text("")
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    (work_dir / "splatting_results_inpainting_results_sbs.mp4").write_text("fake sbs")

    captured: dict[str, list[str]] = {}

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=b"", stderr=b"")

    def _fake_split(sbs_path, left, right):
        Path(left).write_text("data")
        Path(right).write_text("data")

    with (
        patch("subprocess.run", side_effect=_fake_run),
        patch.object(CLIBackend, "_split_sbs_video", side_effect=_fake_split),
        patch("pipeline.stereo_crafter._assert_cuda", return_value=None),
        patch("pipeline.stereo_crafter.tempfile.mkdtemp", return_value=str(work_dir)),
        patch("pipeline.stereo_crafter._write_splatting_grid_video", return_value=None),
    ):
        backend.render_video(
            input_path=str(input_video),
            depth_dir=str(depth_dir),
            output_left=str(tmp_path / "left.mp4"),
            output_right=str(tmp_path / "right.mp4"),
        )
    return captured["cmd"]


def _flag_value(cmd: list[str], flag: str) -> str | None:
    """Return the value following *flag* in *cmd*, or None if absent."""
    if flag in cmd:
        return cmd[cmd.index(flag) + 1]
    return None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTunablesOmittedByDefault:
    """Regression: with no args and no env vars, none of the flags appear."""

    def test_no_tunables_no_flags(self, tmp_path):
        repo = _make_repo(tmp_path)
        with patch.dict("os.environ", {}, clear=True):
            backend = CLIBackend(repo_dir=str(repo))
        cmd = _capture_stage2_cmd(backend, tmp_path)

        assert "--frames_chunk" not in cmd
        assert "--overlap" not in cmd
        assert "--tile_num" not in cmd
        # The mandatory Stage-2 flags are still all present.
        for required in ("--pre_trained_path", "--unet_path", "--input_video_path", "--save_dir"):
            assert required in cmd

    def test_attrs_are_none_by_default(self, tmp_path):
        repo = _make_repo(tmp_path)
        with patch.dict("os.environ", {}, clear=True):
            backend = CLIBackend(repo_dir=str(repo))
        assert backend.frames_chunk is None
        assert backend.overlap is None
        assert backend.tile_num is None


class TestExplicitArgs:
    """Explicit constructor args → the corresponding flag + value appear."""

    def test_explicit_frames_chunk(self, tmp_path):
        repo = _make_repo(tmp_path)
        with patch.dict("os.environ", {}, clear=True):
            backend = CLIBackend(repo_dir=str(repo), frames_chunk=23)
        cmd = _capture_stage2_cmd(backend, tmp_path)
        assert _flag_value(cmd, "--frames_chunk") == "23"
        assert "--overlap" not in cmd
        assert "--tile_num" not in cmd

    def test_explicit_overlap(self, tmp_path):
        repo = _make_repo(tmp_path)
        with patch.dict("os.environ", {}, clear=True):
            backend = CLIBackend(repo_dir=str(repo), overlap=3)
        cmd = _capture_stage2_cmd(backend, tmp_path)
        assert _flag_value(cmd, "--overlap") == "3"
        assert "--frames_chunk" not in cmd
        assert "--tile_num" not in cmd

    def test_explicit_tile_num(self, tmp_path):
        repo = _make_repo(tmp_path)
        with patch.dict("os.environ", {}, clear=True):
            backend = CLIBackend(repo_dir=str(repo), tile_num=2)
        cmd = _capture_stage2_cmd(backend, tmp_path)
        assert _flag_value(cmd, "--tile_num") == "2"
        assert "--frames_chunk" not in cmd
        assert "--overlap" not in cmd

    def test_all_three_explicit(self, tmp_path):
        repo = _make_repo(tmp_path)
        with patch.dict("os.environ", {}, clear=True):
            backend = CLIBackend(repo_dir=str(repo), frames_chunk=16, overlap=4, tile_num=2)
        cmd = _capture_stage2_cmd(backend, tmp_path)
        assert _flag_value(cmd, "--frames_chunk") == "16"
        assert _flag_value(cmd, "--overlap") == "4"
        assert _flag_value(cmd, "--tile_num") == "2"


class TestEnvVarOverride:
    """Env vars alone → the flag + value appear (no constructor arg)."""

    def test_env_frames_chunk(self, tmp_path):
        repo = _make_repo(tmp_path)
        with patch.dict("os.environ", {"STEREOCRAFTER_FRAMES_CHUNK": "23"}, clear=True):
            backend = CLIBackend(repo_dir=str(repo))
        cmd = _capture_stage2_cmd(backend, tmp_path)
        assert _flag_value(cmd, "--frames_chunk") == "23"

    def test_env_overlap(self, tmp_path):
        repo = _make_repo(tmp_path)
        with patch.dict("os.environ", {"STEREOCRAFTER_OVERLAP": "3"}, clear=True):
            backend = CLIBackend(repo_dir=str(repo))
        cmd = _capture_stage2_cmd(backend, tmp_path)
        assert _flag_value(cmd, "--overlap") == "3"

    def test_env_tile_num(self, tmp_path):
        repo = _make_repo(tmp_path)
        with patch.dict("os.environ", {"STEREOCRAFTER_TILE_NUM": "1"}, clear=True):
            backend = CLIBackend(repo_dir=str(repo))
        cmd = _capture_stage2_cmd(backend, tmp_path)
        assert _flag_value(cmd, "--tile_num") == "1"

    def test_all_three_env(self, tmp_path):
        repo = _make_repo(tmp_path)
        env = {
            "STEREOCRAFTER_FRAMES_CHUNK": "16",
            "STEREOCRAFTER_OVERLAP": "4",
            "STEREOCRAFTER_TILE_NUM": "2",
        }
        with patch.dict("os.environ", env, clear=True):
            backend = CLIBackend(repo_dir=str(repo))
        cmd = _capture_stage2_cmd(backend, tmp_path)
        assert _flag_value(cmd, "--frames_chunk") == "16"
        assert _flag_value(cmd, "--overlap") == "4"
        assert _flag_value(cmd, "--tile_num") == "2"


class TestExplicitBeatsEnv:
    """Explicit arg > env var (priority rule from issue #217)."""

    def test_explicit_wins_over_env(self, tmp_path):
        repo = _make_repo(tmp_path)
        env = {
            "STEREOCRAFTER_FRAMES_CHUNK": "23",
            "STEREOCRAFTER_OVERLAP": "3",
            "STEREOCRAFTER_TILE_NUM": "1",
        }
        with patch.dict("os.environ", env, clear=True):
            backend = CLIBackend(repo_dir=str(repo), frames_chunk=16, overlap=4, tile_num=2)
        cmd = _capture_stage2_cmd(backend, tmp_path)
        # Explicit values, not the env values.
        assert _flag_value(cmd, "--frames_chunk") == "16"
        assert _flag_value(cmd, "--overlap") == "4"
        assert _flag_value(cmd, "--tile_num") == "2"

    def test_explicit_one_param_does_not_block_env_for_others(self, tmp_path):
        """Setting one explicit arg must not pull in the others from env.

        Each knob is independent: explicit frames_chunk + env overlap should
        yield frames_chunk from the arg and overlap from the env — and the
        third (unset both ways) stays absent.
        """
        repo = _make_repo(tmp_path)
        with patch.dict("os.environ", {"STEREOCRAFTER_OVERLAP": "3"}, clear=True):
            backend = CLIBackend(repo_dir=str(repo), frames_chunk=16)
        cmd = _capture_stage2_cmd(backend, tmp_path)
        assert _flag_value(cmd, "--frames_chunk") == "16"  # explicit
        assert _flag_value(cmd, "--overlap") == "3"  # env
        assert "--tile_num" not in cmd  # neither → omitted


class TestInvalidEnvVar:
    """An invalid (non-int) env value is warned + ignored, not fatal.

    Behaviour choice (issue #217 lets us pick): a typo like
    ``STEREOCRAFTER_FRAMES_CHUNK=abx`` must not crash a render that worked
    without it.  So the value is dropped (treated as unset → flag omitted →
    upstream default) and a WARNING is logged naming the env var.
    """

    def test_invalid_env_warns_and_omits_flag(self, tmp_path, caplog):
        import logging

        repo = _make_repo(tmp_path)
        with (
            patch.dict("os.environ", {"STEREOCRAFTER_FRAMES_CHUNK": "abc"}, clear=True),
            caplog.at_level(logging.WARNING, logger="pipeline.stereo_crafter"),
        ):
            backend = CLIBackend(repo_dir=str(repo))

        # Attr is None → flag will be omitted (upstream default).
        assert backend.frames_chunk is None
        # A warning was emitted naming the offending env var.
        assert any("STEREOCRAFTER_FRAMES_CHUNK" in r.message for r in caplog.records), caplog.text
        # And it must not have crashed the render.
        cmd = _capture_stage2_cmd(backend, tmp_path)
        assert "--frames_chunk" not in cmd

    def test_invalid_env_does_not_block_valid_env(self, tmp_path):
        """One bad env value must not poison the resolution of the others."""
        repo = _make_repo(tmp_path)
        env = {
            "STEREOCRAFTER_FRAMES_CHUNK": "abc",  # invalid → ignored
            "STEREOCRAFTER_OVERLAP": "3",  # valid → used
            "STEREOCRAFTER_TILE_NUM": "2",  # valid → used
        }
        with patch.dict("os.environ", env, clear=True):
            backend = CLIBackend(repo_dir=str(repo))
        assert backend.frames_chunk is None
        assert backend.overlap == 3
        assert backend.tile_num == 2


class TestSubprocessContract:
    """The subprocess call stays list-form, never shell=True (boundary lock)."""

    def test_subprocess_is_list_form_no_shell(self, tmp_path):
        repo = _make_repo(tmp_path)
        with patch.dict("os.environ", {}, clear=True):
            backend = CLIBackend(repo_dir=str(repo), frames_chunk=23, overlap=3, tile_num=1)

        depth_dir = tmp_path / "depth"
        depth_dir.mkdir()
        input_video = tmp_path / "input.mp4"
        input_video.write_text("")
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        (work_dir / "splatting_results_inpainting_results_sbs.mp4").write_text("fake sbs")

        seen_calls: list = []

        def _fake_run(cmd, **kwargs):
            seen_calls.append((cmd, kwargs))
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=b"", stderr=b"")

        with (
            patch("subprocess.run", side_effect=_fake_run),
            patch.object(CLIBackend, "_split_sbs_video", return_value=None),
            patch("pipeline.stereo_crafter._assert_cuda", return_value=None),
            patch("pipeline.stereo_crafter.tempfile.mkdtemp", return_value=str(work_dir)),
            patch("pipeline.stereo_crafter._write_splatting_grid_video", return_value=None),
        ):
            backend.render_video(
                input_path=str(input_video),
                depth_dir=str(depth_dir),
                output_left=str(tmp_path / "left.mp4"),
                output_right=str(tmp_path / "right.mp4"),
            )

        assert len(seen_calls) == 1
        cmd, kwargs = seen_calls[0]
        # cmd is a list of str (not a string, not shell-parsed).
        assert isinstance(cmd, list)
        assert all(isinstance(part, str) for part in cmd)
        # shell must never be True (boundary lock + CLAUDE.md).
        assert kwargs.get("shell") in (None, False)


class TestOOMHintMentionsTunables:
    """The OOM/failure message lists all three env var names (issue #217)."""

    def test_failure_message_lists_tunable_env_vars(self, tmp_path):
        repo = _make_repo(tmp_path)
        with patch.dict("os.environ", {}, clear=True):
            backend = CLIBackend(repo_dir=str(repo), frames_chunk=23, overlap=3, tile_num=1)

        depth_dir = tmp_path / "depth"
        depth_dir.mkdir()
        input_video = tmp_path / "input.mp4"
        input_video.write_text("")

        def _fake_run(cmd, **kwargs):
            kwargs["stderr"].write(b"CUDA out of memory\n")
            return subprocess.CompletedProcess(args=cmd, returncode=1)

        with (
            patch("subprocess.run", side_effect=_fake_run),
            patch("pipeline.stereo_crafter._assert_cuda", return_value=None),
            patch("pipeline.stereo_crafter._write_splatting_grid_video", return_value=None),
        ):
            try:
                backend.render_video(
                    input_path=str(input_video),
                    depth_dir=str(depth_dir),
                    output_left=str(tmp_path / "left.mp4"),
                    output_right=str(tmp_path / "right.mp4"),
                )
            except RuntimeError as exc:
                msg = str(exc)
            else:  # pragma: no cover - the run is rigged to fail
                raise AssertionError("expected RuntimeError for a failing Stage-2 subprocess")

        # The three new tunable env var names must appear in the OOM hint.
        assert "STEREOCRAFTER_FRAMES_CHUNK" in msg
        assert "STEREOCRAFTER_OVERLAP" in msg
        assert "STEREOCRAFTER_TILE_NUM" in msg
        # The pre-existing VRAM knob is still listed alongside them.
        assert "STEREOCRAFTER_MAX_RES" in msg

    def test_timeout_message_lists_tunable_env_vars(self, tmp_path):
        """The timeout branch also surfaces the tunables (issue #134 symmetry)."""
        repo = _make_repo(tmp_path)
        with patch.dict("os.environ", {"STEREOCRAFTER_TIMEOUT_SEC": "60"}, clear=True):
            backend = CLIBackend(repo_dir=str(repo), frames_chunk=23, overlap=3, tile_num=1)

        depth_dir = tmp_path / "depth"
        depth_dir.mkdir()
        input_video = tmp_path / "input.mp4"
        input_video.write_text("")

        def _fake_run(cmd, **kwargs):
            kwargs["stderr"].write(b"loading unet\nframe 10/500\n")
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs.get("timeout"))

        with (
            patch("subprocess.run", side_effect=_fake_run),
            patch("pipeline.stereo_crafter._assert_cuda", return_value=None),
            patch("pipeline.stereo_crafter._write_splatting_grid_video", return_value=None),
        ):
            try:
                backend.render_video(
                    input_path=str(input_video),
                    depth_dir=str(depth_dir),
                    output_left=str(tmp_path / "left.mp4"),
                    output_right=str(tmp_path / "right.mp4"),
                )
            except RuntimeError as exc:
                msg = str(exc)
            else:  # pragma: no cover - rigged to time out
                raise AssertionError("expected RuntimeError for a timed-out Stage-2 subprocess")

        assert "STEREOCRAFTER_FRAMES_CHUNK" in msg
        assert "STEREOCRAFTER_OVERLAP" in msg
        assert "STEREOCRAFTER_TILE_NUM" in msg
        assert "STEREOCRAFTER_MAX_RES" in msg
