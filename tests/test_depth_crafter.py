"""Tests for DepthCrafterEstimator and its pluggable backend.

All tests are mock-based — no CUDA, no real model required.
"""

from __future__ import annotations

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
    """CLIBackend should raise if no repo_dir provided."""
    with pytest.raises(RuntimeError, match="repository directory not specified"):
        CLIBackend(repo_dir=None)


@patch("pipeline.depth_crafter._assert_cuda")
def test_cli_backend_repo_not_found(mock_cuda: MagicMock) -> None:
    """CLIBackend should raise if repo_dir does not exist."""
    with pytest.raises(RuntimeError, match="repository not found"):
        CLIBackend(repo_dir="/nonexistent/depthcrafter_repo")


@patch("pipeline.depth_crafter._assert_cuda")
def test_cli_backend_no_inference_script(mock_cuda: MagicMock) -> None:
    """CLIBackend should raise if repo dir exists but no inference script."""
    with tempfile.TemporaryDirectory() as tmpdir, pytest.raises(RuntimeError, match="No known inference script found"):
        CLIBackend(repo_dir=tmpdir)


@patch("pipeline.depth_crafter._assert_cuda")
def test_cli_backend_finds_inference_script(mock_cuda: MagicMock) -> None:
    """CLIBackend should find run.py in the repo dir."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a fake inference script
        script_path = Path(tmpdir) / "run.py"
        script_path.write_text("print('ok')")

        # We expect subprocess to fail because there's no video, but that's fine
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
    """CLIBackend should build correct subprocess command."""
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

        # Verify command structure
        assert mock_run.call_count == 1
        args, kwargs = mock_run.call_args
        cmd = args[0]
        assert cmd[0] == "python3"
        assert cmd[1] == str(script_path)
        assert cmd[2] == "--video"
        assert cmd[4] == "--output_dir"
        assert cmd[6] == "--max_resolution"
        assert cmd[7] == "768"
        assert cmd[8] == "--checkpoint_dir"
        assert cmd[9] == str(Path(tmpdir) / "checkpoints")
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
def test_cli_backend_env_vars(mock_cuda: MagicMock) -> None:
    """CLIBackend should read DEPTHCRAFTER_REPO_DIR from env."""
    with tempfile.TemporaryDirectory() as tmpdir, patch.dict(os.environ, {"DEPTHCRAFTER_REPO_DIR": tmpdir}):
        script_path = Path(tmpdir) / "run.py"
        script_path.write_text("print('ok')")
        backend = CLIBackend()  # no argument — uses env var
        assert backend.repo_dir == str(Path(tmpdir).resolve())
        assert backend.python_exe == "python"
