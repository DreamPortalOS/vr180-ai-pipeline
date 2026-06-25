"""Tests for StereoCrafter renderer.

These tests use mock backends so they pass on any platform (no CUDA or
model deployment needed).  They verify:
- CUDA-only guard raises clear error when CUDA is absent
- Backend construction and path validation
- Interface contract of StereoCrafterBackend
- ``scripts/run_pipeline.py --stereo-model default`` behavior unchanged
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from pipeline.stereo_crafter import StereoCrafterBackend

# ---------------------------------------------------------------------------
# Determine if we're on a CI / non-CUDA machine
# ---------------------------------------------------------------------------

_HAS_CUDA: bool = False
try:
    import torch

    _HAS_CUDA = torch.cuda.is_available()  # type: ignore[attr-defined]
except (ImportError, AssertionError):
    pass

# ---------------------------------------------------------------------------
# Mock backend (inherits from ABC for correct type-checking)
# ---------------------------------------------------------------------------


class MockStereoCrafterBackend(StereoCrafterBackend):
    """In-memory mock that returns synthetic L/R video paths."""

    def __init__(self, fail_on_call: bool = False, **kwargs) -> None:
        self.fail_on_call = fail_on_call
        self.kwargs = kwargs
        self.call_count = 0

    def render_video(
        self,
        input_path: str,
        depth_dir: str,
        output_left: str,
        output_right: str,
    ) -> tuple[str, str]:
        self.call_count += 1
        if self.fail_on_call:
            raise RuntimeError("Mock StereoCrafter backend failed (intentional test error).")

        # Create dummy output files
        for p in (output_left, output_right):
            Path(p).parent.mkdir(parents=True, exist_ok=True)
            Path(p).write_text("mock_video_data")

        return output_left, output_right


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCudaGuard:
    """Verify the CUDA guard raises clear, actionable errors."""

    def test_non_cuda_raises(self):
        """On non-CUDA systems, _assert_cuda should raise."""
        from pipeline.stereo_crafter import _assert_cuda

        if _HAS_CUDA:
            pytest.skip("CUDA is available — skipping non-CUDA error test")

        with pytest.raises(RuntimeError) as exc_info:
            _assert_cuda()
        msg = str(exc_info.value)
        assert "CUDA is not available" in msg
        assert "STEREOCRAFTER_SETUP.md" in msg


class TestMockBackendContract:
    """Verify the mock backend satisfies the abstract interface."""

    def test_mock_backend_signature(self):
        """Mock backend can be called as a StereoCrafterBackend."""
        backend = MockStereoCrafterBackend()
        import inspect

        sig = inspect.signature(backend.render_video)
        param_names = list(sig.parameters.keys())
        assert "input_path" in param_names
        assert "depth_dir" in param_names
        assert "output_left" in param_names
        assert "output_right" in param_names

    def test_mock_backend_call(self, tmp_path):
        """Mock backend returns the expected L/R paths."""
        backend = MockStereoCrafterBackend()
        left = str(tmp_path / "left.mp4")
        right = str(tmp_path / "right.mp4")

        result_l, result_r = backend.render_video(
            input_path=str(tmp_path / "input.mp4"),
            depth_dir=str(tmp_path / "depth"),
            output_left=left,
            output_right=right,
        )
        assert result_l == left
        assert result_r == right
        assert Path(left).exists()
        assert Path(right).exists()

    def test_mock_backend_failure(self, tmp_path):
        """Mock backend raises on fail_on_call."""
        backend = MockStereoCrafterBackend(fail_on_call=True)
        with pytest.raises(RuntimeError, match="Mock StereoCrafter backend failed"):
            backend.render_video(
                input_path=str(tmp_path / "input.mp4"),
                depth_dir=str(tmp_path / "depth"),
                output_left=str(tmp_path / "left.mp4"),
                output_right=str(tmp_path / "right.mp4"),
            )


class TestStereoCrafterRenderer:
    """Integration-style tests using the mock backend."""

    def test_renderer_with_mock_backend(self, tmp_path):
        """StereoCrafterRenderer with a mock backend works end-to-end."""
        from pipeline.stereo_crafter import StereoCrafterRenderer

        if not _HAS_CUDA:
            pytest.skip("CUDA not available — cannot instantiate StereoCrafterRenderer")

        backend = MockStereoCrafterBackend()
        renderer = StereoCrafterRenderer(backend=backend)

        input_video = tmp_path / "input.mp4"
        input_video.write_text("fake_video")

        depth_dir = tmp_path / "depth"
        depth_dir.mkdir()

        left_out = str(tmp_path / "left.mp4")
        right_out = str(tmp_path / "right.mp4")

        result_l, result_r = renderer.render_video(
            input_path=str(input_video),
            depth_dir=str(depth_dir),
            output_left=left_out,
            output_right=right_out,
        )
        assert result_l == left_out
        assert result_r == right_out
        assert backend.call_count == 1

    def test_renderer_missing_input(self, tmp_path):
        """RenderVideo raises FileNotFoundError when input doesn't exist."""
        from pipeline.stereo_crafter import StereoCrafterRenderer

        if not _HAS_CUDA:
            pytest.skip("CUDA not available — cannot instantiate StereoCrafterRenderer")

        backend = MockStereoCrafterBackend()
        renderer = StereoCrafterRenderer(backend=backend)

        with pytest.raises(FileNotFoundError, match="Input video not found"):
            renderer.render_video(
                input_path=str(tmp_path / "nonexistent.mp4"),
                depth_dir=str(tmp_path / "depth"),
            )

    def test_renderer_missing_depth_dir(self, tmp_path):
        """RenderVideo raises NotADirectoryError when depth_dir doesn't exist."""
        from pipeline.stereo_crafter import StereoCrafterRenderer

        if not _HAS_CUDA:
            pytest.skip("CUDA not available — cannot instantiate StereoCrafterRenderer")

        backend = MockStereoCrafterBackend()
        renderer = StereoCrafterRenderer(backend=backend)

        input_video = tmp_path / "input.mp4"
        input_video.write_text("fake_video")

        with pytest.raises(NotADirectoryError):
            renderer.render_video(
                input_path=str(input_video),
                depth_dir=str(tmp_path / "nonexistent_depth"),
            )

    def test_backend_unavailable_error_message(self):
        """CLIBackend without repo_dir raises clear, actionable error."""
        from pipeline.stereo_crafter import CLIBackend

        with pytest.raises(RuntimeError) as exc_info:
            CLIBackend(repo_dir=None)
        msg = str(exc_info.value)
        assert "STEREOCRAFTER_REPO_DIR" in msg
        assert "STEREOCRAFTER_SETUP.md" in msg


class TestPipelineIntegration:
    """Verify that --stereo-model default leaves the pipeline unchanged."""

    def test_default_stereo_model_arg(self):
        """--stereo-model default should parse without error."""
        from scripts.run_pipeline import parse_args

        args = parse_args(["--input", "dummy.mp4", "--stereo-model", "default"])
        assert args.stereo_model == "default"

    def test_stereocrafter_model_arg(self):
        """--stereo-model stereocrafter should parse without error."""
        from scripts.run_pipeline import parse_args

        args = parse_args(["--input", "dummy.mp4", "--stereo-model", "stereocrafter"])
        assert args.stereo_model == "stereocrafter"

    def test_stereocrafter_extra_args_parsed(self):
        """StereoCrafter-specific args should parse without error."""
        from scripts.run_pipeline import parse_args

        test_argv = [
            "--input",
            "dummy.mp4",
            "--stereo-model",
            "stereocrafter",
            "--stereocrafter-repo-dir",
            "/fake/path",
            "--stereocrafter-python",
            "python3",
            "--stereocrafter-max-res",
            "768",
        ]
        args = parse_args(test_argv)
        assert args.stereocrafter_repo_dir == "/fake/path"
        assert args.stereocrafter_python == "python3"
        assert args.stereocrafter_max_res == 768


class TestCLIBackendConstruction:
    """Tests for CLIBackend path validation and construction."""

    def test_cli_backend_requires_repo_dir(self):
        """CLIBackend raises error when no repo_dir is provided."""
        from pipeline.stereo_crafter import CLIBackend

        with patch.dict(os.environ, {}, clear=True), pytest.raises(RuntimeError) as exc_info:
            CLIBackend()
        msg = str(exc_info.value)
        assert "STEREOCRAFTER_REPO_DIR" in msg

    def test_cli_backend_missing_repo_raises(self, tmp_path):
        """CLIBackend raises on non-existent repo_dir."""
        from pipeline.stereo_crafter import CLIBackend

        fake_repo = str(tmp_path / "nonexistent")
        with pytest.raises(RuntimeError) as exc_info:
            CLIBackend(repo_dir=fake_repo)
        msg = str(exc_info.value)
        assert "StereoCrafter repository not found" in msg
        assert "git clone" in msg

    def test_cli_backend_env_variable(self):
        """CLIBackend reads repo_dir from STEREOCRAFTER_REPO_DIR env var."""
        from pipeline.stereo_crafter import CLIBackend

        with (
            patch.dict(os.environ, {"STEREOCRAFTER_REPO_DIR": "/env/path"}),
            patch.object(CLIBackend, "_validate_paths", return_value=None),
        ):
            backend = CLIBackend()
            assert backend.repo_dir == str(Path("/env/path").resolve())

    def test_cli_backend_finds_no_script(self, tmp_path):
        """CLIBackend raises when repo dir has no inference script."""
        from pipeline.stereo_crafter import CLIBackend

        repo = tmp_path / "stereocrafter"
        repo.mkdir()

        with pytest.raises(RuntimeError) as exc_info:
            CLIBackend(repo_dir=str(repo))
        msg = str(exc_info.value)
        assert "No known inference script found" in msg


class TestCLIBackendInference:
    """Tests for the actual subprocess invocation (subprocess mocked)."""

    def test_render_video_success(self, tmp_path):
        """Successful subprocess returns L/R paths."""
        from pipeline.stereo_crafter import CLIBackend

        repo = tmp_path / "stereocrafter"
        repo.mkdir()
        script = repo / "run.py"
        script.write_text("")

        backend = CLIBackend(repo_dir=str(repo))

        depth_dir = tmp_path / "depth"
        depth_dir.mkdir()
        input_video = tmp_path / "input.mp4"
        input_video.write_text("")
        left_out = str(tmp_path / "left_out.mp4")
        right_out = str(tmp_path / "right_out.mp4")

        Path(left_out).write_text("data")
        Path(right_out).write_text("data")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

            with patch("pipeline.stereo_crafter._assert_cuda", return_value=None):
                result_l, result_r = backend.render_video(
                    input_path=str(input_video),
                    depth_dir=str(depth_dir),
                    output_left=left_out,
                    output_right=right_out,
                )

        assert result_l == left_out
        assert result_r == right_out
        mock_run.assert_called_once()

    def test_render_video_subprocess_failure(self, tmp_path):
        """Subprocess non-zero exit raises clear RuntimeError."""
        from pipeline.stereo_crafter import CLIBackend

        repo = tmp_path / "stereocrafter"
        repo.mkdir()
        script = repo / "run.py"
        script.write_text("")

        backend = CLIBackend(repo_dir=str(repo))

        depth_dir = tmp_path / "depth"
        depth_dir.mkdir()
        input_video = tmp_path / "input.mp4"
        input_video.write_text("")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=1, stdout="", stderr="CUDA OOM error"
            )

            with (
                patch("pipeline.stereo_crafter._assert_cuda", return_value=None),
                pytest.raises(RuntimeError) as exc_info,
            ):
                backend.render_video(
                    input_path=str(input_video),
                    depth_dir=str(depth_dir),
                    output_left=str(tmp_path / "left.mp4"),
                    output_right=str(tmp_path / "right.mp4"),
                )
        msg = str(exc_info.value)
        assert "StereoCrafter inference failed" in msg
        assert "CUDA OOM" in msg

    def test_render_video_missing_outputs(self, tmp_path):
        """Subprocess succeeds but outputs not created -> clear error."""
        from pipeline.stereo_crafter import CLIBackend

        repo = tmp_path / "stereocrafter"
        repo.mkdir()
        script = repo / "run.py"
        script.write_text("")

        backend = CLIBackend(repo_dir=str(repo))

        depth_dir = tmp_path / "depth"
        depth_dir.mkdir()
        input_video = tmp_path / "input.mp4"
        input_video.write_text("")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

            with (
                patch("pipeline.stereo_crafter._assert_cuda", return_value=None),
                pytest.raises(RuntimeError) as exc_info,
            ):
                backend.render_video(
                    input_path=str(input_video),
                    depth_dir=str(depth_dir),
                    output_left=str(tmp_path / "missing_left.mp4"),
                    output_right=str(tmp_path / "missing_right.mp4"),
                )
        msg = str(exc_info.value)
        assert "output video(s) not found" in msg
