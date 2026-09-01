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
        self.last_target_size: tuple[int, int] | None = None

    def estimate_video(
        self,
        input_path: str,
        output_dir: str,
        target_size: tuple[int, int] | None = None,
    ) -> list[np.ndarray]:
        self.last_input_path = input_path
        self.last_output_dir = output_dir
        self.last_target_size = target_size
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
    """Non-zero returncode should raise RuntimeError with the stderr tail.

    Issue #127: stdout/stderr are drained to temp files (no PIPE), so the
    mock writes the child's stderr into the file the backend hands to
    ``subprocess.run``.  The exception must carry the stderr summary, the
    command, the cwd, and the real contents of the output dir.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / "run.py"
        script_path.write_text("print('ok')")

        outdir_path = Path(tmpdir) / "depth_out"

        def _fake_run(cmd, **kwargs):
            # The child crashes after writing a partial product + an error.
            outdir_path.mkdir(parents=True, exist_ok=True)
            (outdir_path / "clip_input.mp4").write_bytes(b"x")
            kwargs["stdout"].write(b"loading weights\n")
            kwargs["stderr"].write(b"Traceback (most recent call last):\nCUDA out of memory\n")
            return MagicMock(returncode=1)

        mock_run.side_effect = _fake_run

        backend = CLIBackend(repo_dir=tmpdir)
        with pytest.raises(RuntimeError) as excinfo:
            backend.estimate_video(input_path=str(script_path), output_dir=str(outdir_path))

        msg = str(excinfo.value)
        assert "CUDA out of memory" in msg  # stderr tail folded in
        assert "loading weights" in msg  # stdout tail folded in
        assert "Command:" in msg and "run.py" in msg
        assert f"cwd: {backend.repo_dir}" in msg
        # Real output-dir contents are listed so the operator sees what the
        # script actually produced before dying.
        assert "clip_input.mp4" in msg
        assert "Output dir contents:" in msg


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
    """CLIBackend should load .npy depth maps from output dir.

    NOTE: the output dir is wiped before inference, so the fake products must
    be written from inside the mocked subprocess call (side_effect), not
    before estimate_video() runs.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / "run.py"
        script_path.write_text("print('ok')")

        outdir_path = Path(tmpdir) / "depth_output"

        def _fake_run(cmd, **kwargs):
            outdir_path.mkdir(parents=True, exist_ok=True)
            rng = np.random.default_rng(42)
            for i in range(3):
                np.save(str(outdir_path / f"depth_{i:06d}.npy"), rng.random((100, 200)).astype(np.float32))
            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
            return result

        mock_run.side_effect = _fake_run

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


# ---------------------------------------------------------------------------
# Issue #126 — real upstream output is ``<stem>_depth.mp4``, not a .npy seq
# ---------------------------------------------------------------------------


def _write_gray_mp4(path: Path, num_frames: int, w: int = 64, h: int = 48, value: int = 128) -> None:
    """Synthesize a small grayscale mp4 (the shape of DepthCrafter's real output)."""
    import cv2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    wr = cv2.VideoWriter(str(path), fourcc, 24, (w, h))
    assert wr.isOpened(), f"cv2.VideoWriter could not create {path}"
    try:
        frame = np.full((h, w, 3), value, dtype=np.uint8)
        for _ in range(num_frames):
            wr.write(frame)
    finally:
        wr.release()


def _mock_success_with_products(mock_run: MagicMock, outdir_path: Path, stem: str, num_frames: int) -> None:
    """Make mock_run simulate a successful DepthCrafter run that writes mp4 products."""

    def _fake_run(cmd, **kwargs):
        outdir_path.mkdir(parents=True, exist_ok=True)
        _write_gray_mp4(outdir_path / f"{stem}_depth.mp4", num_frames=num_frames)
        (outdir_path / f"{stem}_input.mp4").write_bytes(b"x")  # decoy, must be ignored
        (outdir_path / f"{stem}_vis.mp4").write_bytes(b"x")  # decoy, must be ignored
        result = MagicMock()
        result.returncode = 0
        result.stdout = ""
        result.stderr = ""
        return result

    mock_run.side_effect = _fake_run


@patch("pipeline.depth_crafter._assert_cuda")
@patch("pipeline.depth_crafter.subprocess.run")
def test_cli_backend_loads_depth_mp4_output(
    mock_run: MagicMock,
    mock_cuda: MagicMock,
) -> None:
    """The real upstream run.py emits ``<stem>_depth.mp4`` — CLIBackend must decode it.

    Regression test for issue #126: previously the loader only looked for
    ``*.npy`` / ``depth_*.png`` and died with 'no depth files found'.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "repo"
        repo_dir.mkdir()
        (repo_dir / "run.py").write_text("print('ok')")
        input_video = Path(tmpdir) / "myclip.mp4"
        input_video.write_bytes(b"fake")
        outdir_path = Path(tmpdir) / "depth_out"

        _mock_success_with_products(mock_run, outdir_path, stem="myclip", num_frames=5)

        backend = CLIBackend(repo_dir=str(repo_dir))
        depths = backend.estimate_video(input_path=str(input_video), output_dir=str(outdir_path))

        assert len(depths) == 5
        assert depths[0].ndim == 2
        assert depths[0].dtype == np.float32
        # 8-bit grayscale video, normalized to [0, 1] — no histogram stretch.
        assert float(depths[0].min()) >= 0.0
        assert float(depths[0].max()) <= 1.0


@patch("pipeline.depth_crafter._assert_cuda")
@patch("pipeline.depth_crafter.subprocess.run")
def test_cli_backend_cleans_output_dir_before_run(
    mock_run: MagicMock,
    mock_cuda: MagicMock,
) -> None:
    """Pre-existing products must be wiped so stale files can't fake success.

    Regression test for issue #126 'fake success': a leftover .npy in the
    output dir was previously loaded as if it were the current run's product.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "repo"
        repo_dir.mkdir()
        (repo_dir / "run.py").write_text("print('ok')")
        input_video = Path(tmpdir) / "clip.mp4"
        input_video.write_bytes(b"fake")
        outdir_path = Path(tmpdir) / "depth_out"
        outdir_path.mkdir(parents=True)
        # Stale pollution from an earlier run — must NOT be picked up.
        np.save(str(outdir_path / "stale_leftover.npy"), np.zeros((4, 4), dtype=np.float32))
        (outdir_path / "clip_depth.mp4").write_bytes(b"stale bytes, not a video")

        _mock_success_with_products(mock_run, outdir_path, stem="clip", num_frames=3)

        backend = CLIBackend(repo_dir=str(repo_dir))
        depths = backend.estimate_video(input_path=str(input_video), output_dir=str(outdir_path))

        # Exactly the 3 fresh frames from this run's mp4 — the stale .npy
        # was wiped before inference, never loaded.
        assert len(depths) == 3
        # The dir now contains only this run's products.
        assert not (outdir_path / "stale_leftover.npy").exists()


@patch("pipeline.depth_crafter._assert_cuda")
@patch("pipeline.depth_crafter.subprocess.run")
def test_cli_backend_error_lists_dir_contents(
    mock_run: MagicMock,
    mock_cuda: MagicMock,
) -> None:
    """When no product form is found, the error must list what IS in the dir."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "repo"
        repo_dir.mkdir()
        (repo_dir / "run.py").write_text("print('ok')")
        input_video = Path(tmpdir) / "src_720p_v2.mp4"
        input_video.write_bytes(b"fake")
        outdir_path = Path(tmpdir) / "depth_out"

        def _fake_run(cmd, **kwargs):
            outdir_path.mkdir(parents=True, exist_ok=True)
            # The three mp4s the real upstream emits — but with a stem that
            # doesn't match, so no *_depth.mp4 is found for our stem...
            # Actually upstream DOES emit <stem>_depth.mp4; here we simulate
            # a future/renamed output so nothing matches:
            (outdir_path / "src_720p_v2_input.mp4").write_bytes(b"x")
            (outdir_path / "src_720p_v2_vis.mp4").write_bytes(b"x")
            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
            return result

        mock_run.side_effect = _fake_run

        backend = CLIBackend(repo_dir=str(repo_dir))
        with pytest.raises(RuntimeError) as excinfo:
            backend.estimate_video(input_path=str(input_video), output_dir=str(outdir_path))
        msg = str(excinfo.value)
        assert "no depth files found" in msg
        assert "src_720p_v2_input.mp4" in msg
        assert "src_720p_v2_vis.mp4" in msg


# ---------------------------------------------------------------------------
# Issue #130 — depth maps must be resized back to the SOURCE frame size
# ---------------------------------------------------------------------------
# DepthCrafter internally down-samples to ``--max_res`` (short side), so its
# decoded depth maps come back at the *model* resolution (e.g. 256×512) while
# the source frames are 720×1280.  Downstream stages operate at source size,
# so CLIBackend must resize before returning.


def _write_color_mp4(path: Path, num_frames: int, w: int, h: int) -> None:
    """Synthesize a small colour mp4 to stand in for the SOURCE video (probe target)."""
    import cv2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    wr = cv2.VideoWriter(str(path), fourcc, 24, (w, h))
    assert wr.isOpened(), f"cv2.VideoWriter could not create {path}"
    try:
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        for _ in range(num_frames):
            wr.write(frame)
    finally:
        wr.release()


def _mock_run_with_small_depth_mp4(mock_run: MagicMock, outdir_path: Path, stem: str, num_frames: int) -> None:
    """Simulate a successful run whose *_depth.mp4 is at MODEL resolution (256×512)."""

    def _fake_run(cmd, **kwargs):
        outdir_path.mkdir(parents=True, exist_ok=True)
        # Model-resolution depth product: 256×512 (the issue-#130 case).
        _write_gray_mp4(outdir_path / f"{stem}_depth.mp4", num_frames=num_frames, w=512, h=256)
        result = MagicMock()
        result.returncode = 0
        result.stdout = ""
        result.stderr = ""
        return result

    mock_run.side_effect = _fake_run


def _make_repo(tmpdir: str) -> Path:
    repo_dir = Path(tmpdir) / "repo"
    repo_dir.mkdir()
    (repo_dir / "run.py").write_text("print('ok')")
    return repo_dir


@patch("pipeline.depth_crafter._assert_cuda")
@patch("pipeline.depth_crafter.subprocess.run")
def test_cli_backend_resizes_depths_to_caller_target_size(
    mock_run: MagicMock,
    mock_cuda: MagicMock,
) -> None:
    """121 model-res (256×512) depths + declared source 720×1280 → all (720, 1280).

    Regression test for issue #130: without the resize, the stereo/EMA stage
    died with ``operands could not be broadcast together with shapes
    (720,1280) (256,512)``.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = _make_repo(tmpdir)
        input_video = Path(tmpdir) / "src_720p.mp4"
        input_video.write_bytes(b"fake")  # probe will fail; target_size must win
        outdir_path = Path(tmpdir) / "depth_out"

        _mock_run_with_small_depth_mp4(mock_run, outdir_path, stem="src_720p", num_frames=121)

        backend = CLIBackend(repo_dir=str(repo_dir))
        depths = backend.estimate_video(
            input_path=str(input_video),
            output_dir=str(outdir_path),
            target_size=(720, 1280),
        )

        assert len(depths) == 121
        for d in depths:
            assert d.shape == (720, 1280)
            assert d.dtype == np.float32


@patch("pipeline.depth_crafter._assert_cuda")
@patch("pipeline.depth_crafter.subprocess.run")
def test_cli_backend_resizes_depths_to_probed_source_size(
    mock_run: MagicMock,
    mock_cuda: MagicMock,
) -> None:
    """Without target_size, the source size is probed from the input video (cv2)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = _make_repo(tmpdir)
        input_video = Path(tmpdir) / "real_src.mp4"
        # Source video is 720×1280 (w=1280, h=720) — probeable.
        _write_color_mp4(input_video, num_frames=2, w=1280, h=720)
        outdir_path = Path(tmpdir) / "depth_out"

        _mock_run_with_small_depth_mp4(mock_run, outdir_path, stem="real_src", num_frames=4)

        backend = CLIBackend(repo_dir=str(repo_dir))
        depths = backend.estimate_video(input_path=str(input_video), output_dir=str(outdir_path))

        assert len(depths) == 4
        for d in depths:
            assert d.shape == (720, 1280)


@patch("pipeline.depth_crafter._assert_cuda")
@patch("pipeline.depth_crafter.subprocess.run")
def test_cli_backend_target_size_wins_over_probe_with_log(
    mock_run: MagicMock,
    mock_cuda: MagicMock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When caller size and probed size disagree, the caller's wins and a line is logged."""
    import logging

    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = _make_repo(tmpdir)
        input_video = Path(tmpdir) / "real_src.mp4"
        _write_color_mp4(input_video, num_frames=2, w=1280, h=720)  # probe → (720, 1280)
        outdir_path = Path(tmpdir) / "depth_out"

        _mock_run_with_small_depth_mp4(mock_run, outdir_path, stem="real_src", num_frames=3)

        backend = CLIBackend(repo_dir=str(repo_dir))
        with caplog.at_level(logging.WARNING, logger="pipeline.depth_crafter"):
            depths = backend.estimate_video(
                input_path=str(input_video),
                output_dir=str(outdir_path),
                target_size=(360, 640),  # disagrees with the probe on purpose
            )

        for d in depths:
            assert d.shape == (360, 640)
        assert any("disagrees with probed source size" in r.message for r in caplog.records)


@patch("pipeline.depth_crafter._assert_cuda")
@patch("pipeline.depth_crafter.subprocess.run")
def test_cli_backend_resize_is_linear_not_nearest(
    mock_run: MagicMock,
    mock_cuda: MagicMock,
) -> None:
    """Depth is continuous — resizing must INTERPOLATE, not quantize to source levels."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = _make_repo(tmpdir)
        input_video = Path(tmpdir) / "src.mp4"
        input_video.write_bytes(b"fake")
        outdir_path = Path(tmpdir) / "depth_out"

        # 2×2 gradient depth video at model res; upsample 4× to 8×8.  With
        # NEAREST the output would contain only the 4 source values; LINEAR
        # must introduce intermediate values.
        def _fake_run(cmd, **kwargs):
            outdir_path.mkdir(parents=True, exist_ok=True)
            import cv2

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            wr = cv2.VideoWriter(str(outdir_path / "src_depth.mp4"), fourcc, 24, (2, 2))
            assert wr.isOpened()
            # 2×2: values 0, 85, 170, 255 (all channels equal → grayscale).
            frame = np.array([[[0, 0, 0], [85, 85, 85]], [[170, 170, 170], [255, 255, 255]]], dtype=np.uint8)
            wr.write(frame)
            wr.release()
            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
            return result

        mock_run.side_effect = _fake_run

        backend = CLIBackend(repo_dir=str(repo_dir))
        depths = backend.estimate_video(
            input_path=str(input_video),
            output_dir=str(outdir_path),
            target_size=(8, 8),
        )

        assert depths[0].shape == (8, 8)
        unique = np.unique(depths[0])
        # NEAREST would yield ≤4 unique values; LINEAR interpolation yields more.
        assert unique.size > 4, f"expected interpolated values, got only {unique}"


@patch("pipeline.depth_crafter._assert_cuda")
@patch("pipeline.depth_crafter.subprocess.run")
def test_cli_backend_depth_value_range_matches_depth_anything(
    mock_run: MagicMock,
    mock_cuda: MagicMock,
) -> None:
    """Returned depths are float32 normalized to [0, 1] — the Depth-Anything convention.

    Depth-Anything normalizes to [0, 1] (pipeline/depth_estimator.py), so
    DepthCrafter's mp4 path (8-bit gray → /255) and .npy fallback must both
    land in the same range, or a single --comfort preset would mean two
    different things on the two backends.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = _make_repo(tmpdir)
        input_video = Path(tmpdir) / "src.mp4"
        input_video.write_bytes(b"fake")
        outdir_path = Path(tmpdir) / "depth_out"

        def _fake_run(cmd, **kwargs):
            outdir_path.mkdir(parents=True, exist_ok=True)
            _write_gray_mp4(outdir_path / "src_depth.mp4", num_frames=3, w=64, h=48)
            # Alternate-backend product form: raw .npy already in [0, 1].
            rng = np.random.default_rng(7)
            np.save(str(outdir_path / "alt_depth.npy"), rng.random((48, 64)).astype(np.float32))
            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
            return result

        mock_run.side_effect = _fake_run

        backend = CLIBackend(repo_dir=str(repo_dir))
        depths = backend.estimate_video(
            input_path=str(input_video),
            output_dir=str(outdir_path),
            target_size=(720, 1280),
        )

        for d in depths:
            assert d.dtype == np.float32
            assert float(d.min()) >= 0.0
            assert float(d.max()) <= 1.0


@patch("pipeline.depth_crafter._assert_cuda")
@patch("pipeline.depth_crafter.subprocess.run")
def test_cli_backend_npy_fallback_also_resized(
    mock_run: MagicMock,
    mock_cuda: MagicMock,
) -> None:
    """The resize applies to every product form, not just the mp4 path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = _make_repo(tmpdir)
        input_video = Path(tmpdir) / "src.mp4"
        input_video.write_bytes(b"fake")
        outdir_path = Path(tmpdir) / "depth_out"

        def _fake_run(cmd, **kwargs):
            outdir_path.mkdir(parents=True, exist_ok=True)
            rng = np.random.default_rng(42)
            for i in range(3):
                np.save(str(outdir_path / f"depth_{i:06d}.npy"), rng.random((256, 512)).astype(np.float32))
            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
            return result

        mock_run.side_effect = _fake_run

        backend = CLIBackend(repo_dir=str(repo_dir))
        depths = backend.estimate_video(
            input_path=str(input_video),
            output_dir=str(outdir_path),
            target_size=(720, 1280),
        )

        assert len(depths) == 3
        for d in depths:
            assert d.shape == (720, 1280)


@patch("pipeline.depth_crafter._assert_cuda")
def test_estimator_forwards_target_size_to_backend(mock_cuda: MagicMock) -> None:
    """DepthCrafterEstimator must forward target_size to the injected backend."""
    mock_backend = MockBackend(num_frames=2, h=256, w=512)
    estimator = DepthCrafterEstimator(backend=mock_backend)

    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
        tmp.write(b"fake video content")
        tmp.flush()
        estimator.estimate_video(tmp.name, target_size=(720, 1280))

    assert mock_backend.last_target_size == (720, 1280)


# ---------------------------------------------------------------------------
# Issue #127 — subprocess stdout/stderr must never be swallowed
# ---------------------------------------------------------------------------


@patch("pipeline.depth_crafter._assert_cuda")
@patch("pipeline.depth_crafter.subprocess.run")
def test_cli_backend_drains_output_to_files_not_pipes(
    mock_run: MagicMock,
    mock_cuda: MagicMock,
) -> None:
    """Deadlock guard: stdout/stderr are drained to files, never an undrained PIPE.

    A PIPE whose 64 KB buffer fills without a reader deadlocks the pipeline
    (see streaming_pipeline.py issues #21/#45/#49).  The backend must hand
    ``subprocess.run`` real file objects and must not use ``capture_output``.
    """
    import subprocess as sp

    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = _make_repo(tmpdir)
        input_video = Path(tmpdir) / "src.mp4"
        input_video.write_bytes(b"fake")
        outdir_path = Path(tmpdir) / "depth_out"

        captured: dict = {}

        def _fake_run(cmd, **kwargs):
            captured["capture_output"] = kwargs.get("capture_output")
            stdout = kwargs.get("stdout")
            stderr = kwargs.get("stderr")
            # Real file objects — not PIPE, not DEVNULL.  Filenos are probed
            # NOW, while the child is "running": the backend closes the temp
            # files after subprocess.run returns, so a post-hoc fileno()
            # would raise ValueError on the closed file.
            captured["stdout_is_pipe_or_devnull"] = stdout in (sp.PIPE, sp.DEVNULL)
            captured["stderr_is_pipe_or_devnull"] = stderr in (sp.PIPE, sp.DEVNULL)
            captured["stdout_fileno"] = stdout.fileno() if stdout is not None else None
            captured["stderr_fileno"] = stderr.fileno() if stderr is not None else None
            outdir_path.mkdir(parents=True, exist_ok=True)
            _write_gray_mp4(outdir_path / "src_depth.mp4", num_frames=2)
            return MagicMock(returncode=0)

        mock_run.side_effect = _fake_run

        backend = CLIBackend(repo_dir=str(repo_dir))
        backend.estimate_video(
            input_path=str(input_video),
            output_dir=str(outdir_path),
            target_size=(48, 64),
        )

        assert captured["capture_output"] in (None, False)
        assert captured["stdout_is_pipe_or_devnull"] is False
        assert captured["stderr_is_pipe_or_devnull"] is False
        assert captured["stdout_fileno"] is not None and captured["stdout_fileno"] >= 0
        assert captured["stderr_fileno"] is not None and captured["stderr_fileno"] >= 0


@patch("pipeline.depth_crafter._assert_cuda")
@patch("pipeline.depth_crafter.subprocess.run")
def test_cli_backend_success_output_not_spammed_at_info(
    mock_run: MagicMock,
    mock_cuda: MagicMock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Successful runs log the subprocess tail at DEBUG only (issue #127).

    With the logger at INFO, none of the child's chatty stdout/stderr may
    reach the log; at DEBUG the tail becomes visible for troubleshooting.
    """
    import logging

    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = _make_repo(tmpdir)
        input_video = Path(tmpdir) / "src.mp4"
        input_video.write_bytes(b"fake")
        outdir_path = Path(tmpdir) / "depth_out"

        def _fake_run(cmd, **kwargs):
            outdir_path.mkdir(parents=True, exist_ok=True)
            _write_gray_mp4(outdir_path / "src_depth.mp4", num_frames=2)
            kwargs["stdout"].write(b"noisy-child-stdout\n")
            kwargs["stderr"].write(b"noisy-child-stderr\n")
            return MagicMock(returncode=0)

        mock_run.side_effect = _fake_run

        backend = CLIBackend(repo_dir=str(repo_dir))
        with caplog.at_level(logging.INFO, logger="pipeline.depth_crafter"):
            backend.estimate_video(
                input_path=str(input_video),
                output_dir=str(outdir_path),
                target_size=(48, 64),
            )
        assert "noisy-child-stdout" not in caplog.text
        assert "noisy-child-stderr" not in caplog.text
        # The command line itself is still logged at INFO (existing behaviour).
        assert "DepthCrafter CLIBackend command:" in caplog.text

        # At DEBUG the tail IS available.
        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger="pipeline.depth_crafter"):
            backend.estimate_video(
                input_path=str(input_video),
                output_dir=str(outdir_path),
                target_size=(48, 64),
            )
        assert "noisy-child-stdout" in caplog.text
        assert "noisy-child-stderr" in caplog.text


@patch("pipeline.depth_crafter._assert_cuda")
@patch("pipeline.depth_crafter.subprocess.run")
def test_cli_backend_failure_logs_tail_at_error(
    mock_run: MagicMock,
    mock_cuda: MagicMock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Failure logs the last ~40 lines of stdout/stderr at ERROR level."""
    import logging

    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = _make_repo(tmpdir)
        input_video = Path(tmpdir) / "src.mp4"
        input_video.write_bytes(b"fake")
        outdir_path = Path(tmpdir) / "depth_out"

        def _fake_run(cmd, **kwargs):
            outdir_path.mkdir(parents=True, exist_ok=True)
            kwargs["stdout"].write(b"out-line\n" * 100)
            kwargs["stderr"].write(b"err-line\n" * 99 + b"fatal: boom\n")
            return MagicMock(returncode=2)

        mock_run.side_effect = _fake_run

        backend = CLIBackend(repo_dir=str(repo_dir))
        with caplog.at_level(logging.ERROR, logger="pipeline.depth_crafter"), pytest.raises(RuntimeError):
            backend.estimate_video(input_path=str(input_video), output_dir=str(outdir_path))

        error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert error_records, "expected an ERROR-level record with the subprocess tail"
        # The last stderr line must be in the ERROR log; lines beyond the
        # 40-line tail window must NOT be (the log stays bounded).
        assert any("fatal: boom" in r.message for r in error_records)
        # 100 stdout lines emitted, tail is 40 → fewer than 60 out-lines logged.
        out_lines_logged = caplog.text.count("out-line")
        assert 0 < out_lines_logged <= 45
