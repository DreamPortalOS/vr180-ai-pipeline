"""Tests for DepthCrafterEstimator and its pluggable backend.

All tests are mock-based — no CUDA, no real model required.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pipeline.depth_crafter import (
    CLIBackend,
    DepthCrafterBackend,
    DepthCrafterEstimator,
    _assert_cuda,
)

# ---------------------------------------------------------------------------
# Mock backend for safe unit testing
# ---------------------------------------------------------------------------


class MockBackend(DepthCrafterBackend):
    """Mock backend that returns fake depth maps without any real inference."""

    def __init__(self, num_frames: int = 5, h: int = 480, w: int = 640) -> None:
        self.num_frames = num_frames
        self.h = h
        self.w = w
        self.last_input_path: str | None = None
        self.last_output_dir: str | None = None

    def estimate_video(
        self,
        input_path: str,
        output_dir: str,
    ) -> list[np.ndarray]:
        self.last_input_path = input_path
        self.last_output_dir = output_dir
        rng = np.random.default_rng(42)
        return [rng.random((self.h, self.w)).astype(np.float32) for _ in range(self.num_frames)]


# ---------------------------------------------------------------------------
# _assert_cuda
# ---------------------------------------------------------------------------


def test_assert_cuda_no_torch() -> None:
    """If torch is unimportable, _assert_cuda should raise."""
    # We mock by setting side effect on __import__
    orig_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__  # type: ignore[union-attr]

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("No torch")
        return orig_import(name, *args, **kwargs)

    with (
        patch("builtins.__import__", side_effect=fake_import),
        pytest.raises(RuntimeError, match="PyTorch is not installed"),
    ):
        _assert_cuda()


@patch("torch.cuda.is_available", return_value=False)
def test_assert_cuda_no_gpu(mock_is_avail: MagicMock) -> None:
    """If CUDA not available, _assert_cuda should raise."""
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        _assert_cuda()


@patch("torch.cuda.is_available", return_value=True)
def test_assert_cuda_ok(mock_is_avail: MagicMock) -> None:
    """If CUDA available, _assert_cuda should pass."""
    _assert_cuda()  # no exception


# ---------------------------------------------------------------------------
# DepthCrafterEstimator with mock backend
# ---------------------------------------------------------------------------


@patch("pipeline.depth_crafter._assert_cuda")  # bypass CUDA check
def test_estimator_with_mock_backend(mock_cuda: MagicMock) -> None:
    """DepthCrafterEstimator with MockBackend should return fake depths."""
    mock_backend = MockBackend(num_frames=3, h=108, w=192)
    estimator = DepthCrafterEstimator(backend=mock_backend)

    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
        tmp.write(b"fake video content")
        tmp.flush()
        depths = estimator.estimate_video(tmp.name)

    assert len(depths) == 3
    assert depths[0].shape == (108, 192)
    assert depths[0].dtype == np.float32


@patch("pipeline.depth_crafter._assert_cuda")
def test_estimator_input_not_found(mock_cuda: MagicMock) -> None:
    """Estimating depth for a non-existent file should raise FileNotFoundError."""
    mock_backend = MockBackend()
    estimator = DepthCrafterEstimator(backend=mock_backend)
    with pytest.raises(FileNotFoundError, match="not found"):
        estimator.estimate_video("/nonexistent/input.mp4")


# ---------------------------------------------------------------------------
# CLIBackend — path validation
# ---------------------------------------------------------------------------


@patch("pipeline.depth_crafter._assert_cuda")
def test_cli_backend_no_repo_dir(mock_cuda: MagicMock) -> None:
    """CLIBackend should raise if no repo_dir provided and no in-repo default exists."""
    with patch("pipeline.depth_crafter.INREPO_REPO_DIR") as mock_inrepo:
        mock_inrepo.is_dir.return_value = False
        with pytest.raises(RuntimeError, match="repository directory not specified"):
            CLIBackend(repo_dir=None)


@patch("pipeline.depth_crafter._assert_cuda")
def test_cli_backend_repo_not_found(mock_cuda: MagicMock) -> None:
    """CLIBackend should raise if repo_dir does not exist."""
    with pytest.raises(RuntimeError, match="repository not found"):
        CLIBackend(repo_dir="/nonexistent/depthcrafter_repo")


@patch("pipeline.depth_crafter._assert_cuda")
def test_cli_backend_no_inference_script(mock_cuda: MagicMock) -> None:
    """CLIBackend should raise if repo dir exists but no run.py entry point."""
    with tempfile.TemporaryDirectory() as tmpdir, pytest.raises(RuntimeError, match=r"run\.py not found"):
        CLIBackend(repo_dir=tmpdir)


@patch("pipeline.depth_crafter._assert_cuda")
def test_cli_backend_finds_inference_script(mock_cuda: MagicMock, tmp_path) -> None:
    """CLIBackend should find run.py in the repo dir.

    Hermetic: patch the in-repo default paths to non-existent tmp_path dirs
    so the python_exe default is guaranteed to be 'python' regardless of
    whether this host happens to have a real DepthCrafter checkout.
    """
    fake_inrepo_repo = tmp_path / "nope_depthcrafter"  # does NOT exist
    fake_inrepo_py = tmp_path / "nope_python"  # does NOT exist
    fake_inrepo_model = tmp_path / "nope_model"  # does NOT exist

    with (
        patch("pipeline.depth_crafter.INREPO_REPO_DIR", fake_inrepo_repo),
        patch("pipeline.depth_crafter.INREPO_PYTHON_EXE", fake_inrepo_py),
        patch("pipeline.depth_crafter.INREPO_MODEL_DIR", fake_inrepo_model),
        tempfile.TemporaryDirectory() as tmpdir,
    ):
        script_path = Path(tmpdir) / "run.py"
        script_path.write_text("print('ok')")

        backend = CLIBackend(repo_dir=tmpdir)
        assert backend.repo_dir == str(Path(tmpdir).resolve())
        assert backend.python_exe == "python"


# ---------------------------------------------------------------------------
# CLIBackend — subprocess execution
# ---------------------------------------------------------------------------


@patch("pipeline.depth_crafter._assert_cuda")
@patch("pipeline.depth_crafter.subprocess.run")
def test_cli_backend_subprocess_command(
    mock_run: MagicMock,
    mock_cuda: MagicMock,
) -> None:
    """CLIBackend should build the fire-style run.py command (lead-verified shape)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / "run.py"
        script_path.write_text("print('ok')")

        backend = CLIBackend(
            repo_dir=tmpdir,
            python_exe="python3",
            checkpoint_dir=str(Path(tmpdir) / "checkpoints"),
            max_resolution=768,
        )

        # Make subprocess.run return success with no depth files
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        with tempfile.TemporaryDirectory() as outdir, pytest.raises(RuntimeError, match="no depth files found"):
            backend.estimate_video(
                input_path=str(script_path),  # dummy path, just needs to exist
                output_dir=outdir,
            )

        # Verify the fire-style command structure:
        #   python3 run.py <video> --save_folder <dir> --max_res 768 --cpu_offload model
        assert mock_run.call_count == 1
        args, kwargs = mock_run.call_args
        cmd = args[0]
        assert cmd[0] == "python3"
        assert cmd[1] == str(script_path)
        # Positional video path (fire-style, not --video)
        assert cmd[2] == str(script_path.resolve())
        assert cmd[3] == "--save_folder"
        assert cmd[5] == "--max_res"
        assert cmd[6] == "768"
        assert "--cpu_offload" in cmd
        assert cmd[cmd.index("--cpu_offload") + 1] == "model"
        # No legacy argparse flags should be present.
        assert "--video" not in cmd
        assert "--output_dir" not in cmd
        assert "--max_resolution" not in cmd
        assert "--checkpoint_dir" not in cmd
        assert kwargs.get("cwd") == str(Path(tmpdir).resolve())
        assert kwargs.get("shell") in (None, False)  # never shell=True


@patch("pipeline.depth_crafter._assert_cuda")
@patch("pipeline.depth_crafter.subprocess.run")
def test_cli_backend_subprocess_failure(
    mock_run: MagicMock,
    mock_cuda: MagicMock,
) -> None:
    """Non-zero returncode should raise RuntimeError with stderr."""
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / "run.py"
        script_path.write_text("print('ok')")

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "CUDA out of memory"
        mock_run.return_value = mock_result

        backend = CLIBackend(repo_dir=tmpdir)
        with tempfile.TemporaryDirectory() as outdir, pytest.raises(RuntimeError, match="CUDA out of memory"):
            backend.estimate_video(input_path=str(script_path), output_dir=outdir)


# ---------------------------------------------------------------------------
# CLIBackend — subprocess output capture (issue #127)
# ---------------------------------------------------------------------------


@patch("pipeline.depth_crafter._assert_cuda")
@patch("pipeline.depth_crafter.subprocess.run")
def test_cli_backend_failure_error_contains_output_and_dir_listing(
    mock_run: MagicMock,
    mock_cuda: MagicMock,
) -> None:
    """Non-zero exit → RuntimeError with the subprocess output summary, command,
    cwd, and the real output-dir contents (issue #127)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "run.py").write_text("print('ok')")

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "depthcrafter exploded: unexpected kwarg --max_res"
        mock_run.return_value = mock_result

        backend = CLIBackend(repo_dir=tmpdir)
        with tempfile.TemporaryDirectory() as outdir:
            # Plant a stray artifact so the listing proves it shows real contents.
            (Path(outdir) / "leftover.mp4").write_text("x")
            with pytest.raises(RuntimeError) as exc_info:
                backend.estimate_video(input_path=str(Path(tmpdir) / "run.py"), output_dir=outdir)
        msg = str(exc_info.value)
        assert "unexpected kwarg --max_res" in msg  # stderr summary included
        assert "Command:" in msg
        assert "cwd:" in msg
        assert "leftover.mp4" in msg  # actual output-dir contents listed


@patch("pipeline.depth_crafter._assert_cuda")
@patch("pipeline.depth_crafter.subprocess.run")
def test_cli_backend_failure_logs_tail_at_error(
    mock_run: MagicMock,
    mock_cuda: MagicMock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Failed subprocess → its output tail is logged at ERROR (issue #127)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "run.py").write_text("print('ok')")

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = "loading weights done"
        mock_result.stderr = "boom: CUDA device-side assert"
        mock_run.return_value = mock_result

        backend = CLIBackend(repo_dir=tmpdir)
        with (
            tempfile.TemporaryDirectory() as outdir,
            caplog.at_level(logging.DEBUG, logger="pipeline.depth_crafter"),
            pytest.raises(RuntimeError),
        ):
            backend.estimate_video(input_path=str(Path(tmpdir) / "run.py"), output_dir=outdir)

    error_lines = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    assert any("CUDA device-side assert" in m for m in error_lines)
    assert any("loading weights done" in m for m in error_lines)


@patch("pipeline.depth_crafter._assert_cuda")
@patch("pipeline.depth_crafter.subprocess.run")
def test_cli_backend_success_logs_tail_at_debug_not_info(
    mock_run: MagicMock,
    mock_cuda: MagicMock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Successful subprocess → output tail logged at DEBUG only; default INFO
    logging must NOT be flooded with CLI output (issue #127)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "run.py").write_text("print('ok')")

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "processing frame 42/100"
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        backend = CLIBackend(repo_dir=tmpdir)

        # INFO level: CLI chatter must not appear in any record.
        outdir = Path(tmpdir) / "depth_output"
        outdir.mkdir()
        np.save(str(outdir / "depth_000000.npy"), np.zeros((4, 4), dtype=np.float32))
        with caplog.at_level(logging.INFO, logger="pipeline.depth_crafter"):
            backend.estimate_video(input_path=str(Path(tmpdir) / "run.py"), output_dir=str(outdir))
        assert not any("processing frame 42/100" in r.getMessage() for r in caplog.records)

        # DEBUG level: the captured tail IS available for troubleshooting.
        caplog.clear()
        outdir2 = Path(tmpdir) / "depth_output2"
        outdir2.mkdir()
        np.save(str(outdir2 / "depth_000000.npy"), np.zeros((4, 4), dtype=np.float32))
        with caplog.at_level(logging.DEBUG, logger="pipeline.depth_crafter"):
            backend.estimate_video(input_path=str(Path(tmpdir) / "run.py"), output_dir=str(outdir2))
        debug_msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG]
        assert any("processing frame 42/100" in m for m in debug_msgs)


@patch("pipeline.depth_crafter._assert_cuda")
@patch("pipeline.depth_crafter.subprocess.run")
def test_cli_backend_no_undrained_pipe(
    mock_run: MagicMock,
    mock_cuda: MagicMock,
) -> None:
    """stdout/stderr must go to a drained temp file, never an undrained PIPE
    (a 64 KB PIPE buffer deadlocks chatty inference CLIs — issue #127)."""
    import subprocess as sp

    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "run.py").write_text("print('ok')")

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        backend = CLIBackend(repo_dir=tmpdir)
        with tempfile.TemporaryDirectory() as outdir, pytest.raises(RuntimeError, match="no depth files found"):
            backend.estimate_video(input_path=str(Path(tmpdir) / "run.py"), output_dir=outdir)

        _, kwargs = mock_run.call_args
        assert kwargs.get("stdout") is not sp.PIPE
        assert kwargs.get("stderr") is not sp.PIPE
        assert kwargs.get("stdout") is not None  # drained file handle
        assert kwargs.get("stderr") is sp.STDOUT  # merged into the same file


@patch("pipeline.depth_crafter._assert_cuda")
@patch("pipeline.depth_crafter.subprocess.run")
def test_cli_backend_missing_depth_error_lists_dir_contents(
    mock_run: MagicMock,
    mock_cuda: MagicMock,
) -> None:
    """'no depth files' error now shows what the CLI actually wrote (issue #127 —
    this was the exact dead-end the lead hit while triaging #126)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "run.py").write_text("print('ok')")

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        backend = CLIBackend(repo_dir=tmpdir)
        with tempfile.TemporaryDirectory() as outdir:
            (Path(outdir) / "video_depth.mp4").write_text("x")  # CLI wrote an mp4
            with pytest.raises(RuntimeError) as exc_info:
                backend.estimate_video(input_path=str(Path(tmpdir) / "run.py"), output_dir=outdir)
        msg = str(exc_info.value)
        assert "no depth files found" in msg
        assert "video_depth.mp4" in msg


@patch("pipeline.depth_crafter._assert_cuda")
@patch("pipeline.depth_crafter.subprocess.run")
def test_cli_backend_subprocess_file_not_found(
    mock_run: MagicMock,
    mock_cuda: MagicMock,
) -> None:
    """FileNotFoundError on python_exe should be caught and re-raised."""
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / "run.py"
        script_path.write_text("print('ok')")

        mock_run.side_effect = FileNotFoundError("No such file")

        backend = CLIBackend(repo_dir=tmpdir, python_exe="nonexistent_python")
        with tempfile.TemporaryDirectory() as outdir, pytest.raises(RuntimeError, match="Python executable not found"):
            backend.estimate_video(input_path=str(script_path), output_dir=outdir)


@patch("pipeline.depth_crafter._assert_cuda")
@patch("pipeline.depth_crafter.subprocess.run")
def test_cli_backend_loads_npy_output(
    mock_run: MagicMock,
    mock_cuda: MagicMock,
) -> None:
    """CLIBackend should load .npy depth maps from output dir."""
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / "run.py"
        script_path.write_text("print('ok')")

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        # Write fake .npy depth files into output dir
        outdir_path = Path(tmpdir) / "depth_output"
        outdir_path.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(42)
        for i in range(3):
            np.save(str(outdir_path / f"depth_{i:06d}.npy"), rng.random((100, 200)).astype(np.float32))

        backend = CLIBackend(repo_dir=tmpdir)
        depths = backend.estimate_video(input_path=str(script_path), output_dir=str(outdir_path))
        assert len(depths) == 3
        assert depths[0].shape == (100, 200)
        assert depths[0].dtype == np.float32


@patch("pipeline.depth_crafter._assert_cuda")
@patch("pipeline.depth_crafter.subprocess.run")
def test_cli_backend_timeout(
    mock_run: MagicMock,
    mock_cuda: MagicMock,
) -> None:
    """subprocess.TimeoutExpired should be caught and re-raised."""
    import subprocess as sp

    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / "run.py"
        script_path.write_text("print('ok')")

        mock_run.side_effect = sp.TimeoutExpired(cmd="test", timeout=7200)

        backend = CLIBackend(repo_dir=tmpdir)
        with tempfile.TemporaryDirectory() as outdir, pytest.raises(RuntimeError, match="timed out after 2 hours"):
            backend.estimate_video(input_path=str(script_path), output_dir=outdir)


# ---------------------------------------------------------------------------
# CLIBackend — env var fallback
# ---------------------------------------------------------------------------


@patch("pipeline.depth_crafter._assert_cuda")
def test_cli_backend_env_vars(mock_cuda: MagicMock, tmp_path) -> None:
    """CLIBackend should read DEPTHCRAFTER_REPO_DIR from env.

    Hermetic: the in-repo python_exe must be neutralized so the assertion
    on python_exe is machine-independent.
    """
    fake_inrepo_py = tmp_path / "nope_python"  # does NOT exist
    fake_inrepo_repo = tmp_path / "nope_repo"
    fake_inrepo_model = tmp_path / "nope_model"

    with (
        tempfile.TemporaryDirectory() as tmpdir,
        patch.dict(os.environ, {"DEPTHCRAFTER_REPO_DIR": tmpdir}),
        patch("pipeline.depth_crafter.INREPO_REPO_DIR", fake_inrepo_repo),
        patch("pipeline.depth_crafter.INREPO_PYTHON_EXE", fake_inrepo_py),
        patch("pipeline.depth_crafter.INREPO_MODEL_DIR", fake_inrepo_model),
    ):
        script_path = Path(tmpdir) / "run.py"
        script_path.write_text("print('ok')")
        backend = CLIBackend()  # no argument — uses env var
        assert backend.repo_dir == str(Path(tmpdir).resolve())
        assert backend.python_exe == "python"


# ---------------------------------------------------------------------------
# CLIBackend — in-repo default path fallback (G-6 style)
# ---------------------------------------------------------------------------


def test_cli_backend_inrepo_defaults(tmp_path) -> None:
    """When no args/env are set but in-repo dirs exist, CLIBackend picks them up."""
    # Build a tmp_path-based in-repo layout (third_party/DepthCrafter + .venv + models).
    repo_dir = tmp_path / "third_party" / "DepthCrafter"
    repo_dir.mkdir(parents=True)
    (repo_dir / "run.py").write_text("print('ok')")

    venv_dir = repo_dir / ".venv"
    python_dir = venv_dir / "Scripts" if os.name == "nt" else venv_dir / "bin"
    python_dir.mkdir(parents=True)
    (python_dir / "python").write_text("# fake")

    model_dir = tmp_path / "models" / "DepthCrafter"
    model_dir.mkdir(parents=True)
    (model_dir / "readme").write_text("weights")

    with (
        patch("pipeline.depth_crafter._assert_cuda"),
        patch("pipeline.depth_crafter.INREPO_REPO_DIR", repo_dir),
        patch("pipeline.depth_crafter.INREPO_PYTHON_EXE", python_dir / "python"),
        patch("pipeline.depth_crafter.INREPO_MODEL_DIR", model_dir),
    ):
        backend = CLIBackend()  # no args, no env

    assert backend.repo_dir == str(repo_dir.resolve())
    assert backend.python_exe == str((python_dir / "python").resolve())
    assert backend.model_dir == str(model_dir.resolve())
    # 12 GB VRAM-safe default.
    assert backend.max_resolution == 512


def test_cli_backend_inrepo_python_absent_falls_to_python(tmp_path) -> None:
    """If the in-repo venv python is absent, python_exe falls back to 'python'."""
    repo_dir = tmp_path / "third_party" / "DepthCrafter"
    repo_dir.mkdir(parents=True)
    (repo_dir / "run.py").write_text("print('ok')")
    model_dir = tmp_path / "models" / "DepthCrafter"
    model_dir.mkdir(parents=True)

    with (
        patch("pipeline.depth_crafter._assert_cuda"),
        patch("pipeline.depth_crafter.INREPO_REPO_DIR", repo_dir),
        patch("pipeline.depth_crafter.INREPO_PYTHON_EXE") as mock_py,
        patch("pipeline.depth_crafter.INREPO_MODEL_DIR", model_dir),
    ):
        mock_py.is_file.return_value = False
        backend = CLIBackend()
    assert backend.python_exe == "python"


def test_cli_backend_inrepo_nonexistent_skipped(tmp_path) -> None:
    """If in-repo dirs are absent, CLIBackend falls through to the (required) env/args path."""
    repo_dir = tmp_path / "third_party" / "DepthCrafter"  # does NOT exist
    model_dir = tmp_path / "models" / "DepthCrafter"  # does NOT exist

    with (
        patch("pipeline.depth_crafter._assert_cuda"),
        patch("pipeline.depth_crafter.INREPO_REPO_DIR", repo_dir),
        patch("pipeline.depth_crafter.INREPO_PYTHON_EXE") as mock_py,
        patch("pipeline.depth_crafter.INREPO_MODEL_DIR", model_dir),
    ):
        mock_py.is_file.return_value = False
        # No in-repo default, no env → should raise the in-repo hint.
        with pytest.raises(RuntimeError, match=r"setup_depthcrafter\.py"):
            CLIBackend()


def test_cli_backend_default_max_res_512() -> None:
    """DEPTHCRAFTER_MAX_RES defaults to 512 (12 GB VRAM-safe) when unset."""
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "run.py").write_text("print('ok')")
        # Wipe any env override so the module default is exercised.
        with (
            patch.dict(os.environ, {"DEPTHCRAFTER_REPO_DIR": tmpdir}, clear=False),
            patch("pipeline.depth_crafter._assert_cuda"),
        ):
            backend = CLIBackend()
        assert backend.max_resolution == 512


@patch("pipeline.depth_crafter._assert_cuda")
@patch("pipeline.depth_crafter.subprocess.run")
def test_cli_backend_passes_process_length_and_target_fps(
    mock_run: MagicMock,
    mock_cuda: MagicMock,
) -> None:
    """Optional --process_length / --target_fps are forwarded when set, omitted when None."""
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "run.py").write_text("print('ok')")

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        backend = CLIBackend(
            repo_dir=tmpdir,
            max_resolution=512,
            process_length=64,
            target_fps=24,
        )

        with tempfile.TemporaryDirectory() as outdir, pytest.raises(RuntimeError, match="no depth files found"):
            backend.estimate_video(input_path=str(Path(tmpdir) / "run.py"), output_dir=outdir)

        cmd = mock_run.call_args[0][0]
        assert "--process_length" in cmd
        assert cmd[cmd.index("--process_length") + 1] == "64"
        assert "--target_fps" in cmd
        assert cmd[cmd.index("--target_fps") + 1] == "24"

        # With the knobs unset, they must be absent from the command.
        mock_run.reset_mock()
        backend2 = CLIBackend(repo_dir=tmpdir, max_resolution=512)
        with tempfile.TemporaryDirectory() as outdir, pytest.raises(RuntimeError, match="no depth files found"):
            backend2.estimate_video(input_path=str(Path(tmpdir) / "run.py"), output_dir=outdir)
        cmd2 = mock_run.call_args[0][0]
        assert "--process_length" not in cmd2
        assert "--target_fps" not in cmd2
