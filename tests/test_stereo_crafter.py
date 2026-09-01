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
from types import SimpleNamespace
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

    def test_backend_unavailable_error_message(self, tmp_path, monkeypatch):
        """CLIBackend without repo_dir raises clear, actionable error.

        Hermetic: the in-repo default repo/python/checkpoint paths are
        monkeypatched to non-existent tmp_path dirs so the test is independent
        of whether this host actually has a deployed StereoCrafter checkout
        (mirror of the I-1.3 DepthCrafter fix — otherwise this host, which
        has a real checkout, hits a different branch and the assertion fails).
        """
        from pipeline import stereo_crafter as sc
        from pipeline.stereo_crafter import CLIBackend

        absent_repo = tmp_path / "third_party" / "StereoCrafter_no_exist"
        absent_py = tmp_path / "venv_no_exist" / "python"
        absent_ckpt = tmp_path / "models" / "StereoCrafter_no_exist"
        monkeypatch.setattr(sc, "INREPO_REPO_DIR", absent_repo, raising=True)
        monkeypatch.setattr(sc, "INREPO_PYTHON_EXE", absent_py, raising=True)
        monkeypatch.setattr(sc, "INREPO_CKPT_DIR", absent_ckpt, raising=True)

        with patch.dict(os.environ, {}, clear=True), pytest.raises(RuntimeError) as exc_info:
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

    def test_cli_backend_requires_repo_dir(self, tmp_path, monkeypatch):
        """CLIBackend raises error when no repo_dir is provided.

        Hermetic: in-repo default paths are neutralized so this is independent
        of host deployment state (I-1.3 pattern).
        """
        from pipeline import stereo_crafter as sc
        from pipeline.stereo_crafter import CLIBackend

        absent_repo = tmp_path / "third_party" / "StereoCrafter_no_exist"
        absent_py = tmp_path / "venv_no_exist" / "python"
        absent_ckpt = tmp_path / "models" / "StereoCrafter_no_exist"
        monkeypatch.setattr(sc, "INREPO_REPO_DIR", absent_repo, raising=True)
        monkeypatch.setattr(sc, "INREPO_PYTHON_EXE", absent_py, raising=True)
        monkeypatch.setattr(sc, "INREPO_CKPT_DIR", absent_ckpt, raising=True)

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

    def test_cli_backend_finds_inpainting_entry(self, tmp_path):
        """CLIBackend accepts the real upstream inpainting_inference.py entry.

        Acceptance criterion: a tmp_path repo with inpainting_inference.py
        must be found by the candidate list (Stage 2 / disocclusion step).
        """
        from pipeline.stereo_crafter import CLIBackend

        repo = tmp_path / "stereocrafter"
        repo.mkdir()
        (repo / "inpainting_inference.py").write_text("# fake stage-2 entry")

        backend = CLIBackend(repo_dir=str(repo))
        assert backend._find_inference_script("inpainting_inference.py") == str(
            (repo / "inpainting_inference.py").resolve()
        )

    def test_cli_backend_finds_splatting_entry(self, tmp_path):
        """CLIBackend accepts the real upstream depth_splatting_inference.py.

        Stage 1 (depth splatting) — the upstream stage run before inpainting.
        """
        from pipeline.stereo_crafter import CLIBackend

        repo = tmp_path / "stereocrafter"
        repo.mkdir()
        (repo / "depth_splatting_inference.py").write_text("# fake stage-1 entry")

        backend = CLIBackend(repo_dir=str(repo))
        assert backend._find_inference_script("depth_splatting_inference.py") == str(
            (repo / "depth_splatting_inference.py").resolve()
        )

    def test_cli_backend_rejects_legacy_inference_py_name(self, tmp_path):
        """A repo with only the legacy ``inference.py`` name is NOT accepted.

        Acceptance criterion: the old fictional name ``inference.py`` must not
        be misrecognized as a valid entry — upstream has no such file.
        """
        from pipeline.stereo_crafter import CLIBackend

        repo = tmp_path / "stereocrafter"
        repo.mkdir()
        (repo / "inference.py").write_text("# legacy fictional name — must NOT match")
        (repo / "run.py").write_text("# also not an upstream entry")

        with pytest.raises(RuntimeError) as exc_info:
            CLIBackend(repo_dir=str(repo))
        msg = str(exc_info.value)
        assert "No known inference script found" in msg


class TestCLIBackendInRepoFallback:
    """Tests for the in-repo default path fallback (G-6 / I-2 pattern).

    When env vars are unset, CLIBackend should adopt the in-repo paths
    (third_party/StereoCrafter / its .venv python / models/StereoCrafter)
    ONLY when they exist on disk.  These tests build the dirs with tmp_path
    and monkeypatch the module-level constants so nothing touches the real repo.
    """

    @pytest.fixture
    def inrepo_sandbox(self, tmp_path, monkeypatch):
        """Create in-repo StereoCrafter layout under tmp_path and point constants at it."""
        repo = tmp_path / "third_party" / "StereoCrafter"
        repo.mkdir(parents=True)
        # Real upstream entry (Stage 2 / inpainting), NOT the fictional run.py.
        (repo / "inpainting_inference.py").write_text("# fake stage-2 entry")

        venv_dir = repo / ".venv"
        venv_dir.mkdir()
        python_exe = venv_dir / (Path("Scripts") / "python.exe" if os.name == "nt" else Path("bin") / "python")
        python_exe.parent.mkdir(parents=True, exist_ok=True)
        python_exe.write_text("# fake python")

        model_dir = tmp_path / "models" / "StereoCrafter"
        model_dir.mkdir(parents=True)
        (model_dir / "weights.bin").write_text("w")

        from pipeline import stereo_crafter as sc

        monkeypatch.setattr(sc, "INREPO_REPO_DIR", repo, raising=True)
        monkeypatch.setattr(sc, "INREPO_PYTHON_EXE", python_exe, raising=True)
        monkeypatch.setattr(sc, "INREPO_CKPT_DIR", model_dir, raising=True)
        return SimpleNamespace(repo=repo, python_exe=python_exe, model_dir=model_dir)

    def test_fallback_to_inrepo_paths_when_env_unset(self, tmp_path, inrepo_sandbox, monkeypatch):
        """With no env and no constructor args, in-repo paths are adopted."""
        from pipeline.stereo_crafter import CLIBackend

        with patch.dict(os.environ, {}, clear=True):
            backend = CLIBackend()
        assert backend.repo_dir == str(inrepo_sandbox.repo)
        assert backend.python_exe == str(inrepo_sandbox.python_exe)
        assert backend.checkpoint_dir == str(inrepo_sandbox.model_dir)

    def test_env_vars_take_precedence_over_inrepo(self, tmp_path, inrepo_sandbox, monkeypatch):
        """Env vars win over in-repo defaults even when in-repo paths exist."""
        from pipeline.stereo_crafter import CLIBackend

        explicit_repo = tmp_path / "custom_repo"
        explicit_repo.mkdir()
        (explicit_repo / "inpainting_inference.py").write_text("# fake stage-2 entry")
        explicit_ckpt = tmp_path / "custom_ckpt"
        explicit_ckpt.mkdir()
        (explicit_ckpt / "w.bin").write_text("w")

        with patch.dict(
            os.environ,
            {
                "STEREOCRAFTER_REPO_DIR": str(explicit_repo),
                "STEREOCRAFTER_CKPT_DIR": str(explicit_ckpt),
            },
            clear=True,
        ):
            backend = CLIBackend()
        assert backend.repo_dir == str(Path(explicit_repo).resolve())
        assert backend.checkpoint_dir == str(Path(explicit_ckpt).resolve())

    def test_inrepo_fallback_ignored_when_paths_missing(self, tmp_path, monkeypatch):
        """If in-repo dirs don't exist, the fallback is skipped and the error points to the bootstrap."""
        from pipeline.stereo_crafter import CLIBackend

        absent = tmp_path / "third_party" / "StereoCrafter_no_exist"
        absent_py = tmp_path / "venv_no_exist" / "python"
        absent_ckpt = tmp_path / "models" / "StereoCrafter_no_exist"

        from pipeline import stereo_crafter as sc

        monkeypatch.setattr(sc, "INREPO_REPO_DIR", absent, raising=True)
        monkeypatch.setattr(sc, "INREPO_PYTHON_EXE", absent_py, raising=True)
        monkeypatch.setattr(sc, "INREPO_CKPT_DIR", absent_ckpt, raising=True)

        with (
            patch.dict(os.environ, {}, clear=True),
            pytest.raises(RuntimeError) as exc_info,
        ):
            CLIBackend()
        msg = str(exc_info.value)
        assert "setup_stereocrafter.py" in msg
        assert "STEREOCRAFTER_SETUP.md" in msg

    def test_max_resolution_default_is_12gb_safe(self, tmp_path, inrepo_sandbox):
        """Default max_resolution is the 12 GB-safe 512 (overridable via env/arg)."""
        from pipeline.stereo_crafter import CLIBackend

        with patch.dict(os.environ, {}, clear=True):
            backend = CLIBackend()
        assert backend.max_resolution == 512

        with patch.dict(os.environ, {"STEREOCRAFTER_MAX_RES": "768"}, clear=True):
            backend = CLIBackend()
        assert backend.max_resolution == 768

        with patch.dict(os.environ, {}, clear=True):
            backend = CLIBackend(max_resolution=384)
        assert backend.max_resolution == 384


class TestCLIBackendInference:
    """Tests for the actual subprocess invocation (subprocess mocked).

    The real StereoCrafter flow is two fire-style stages (splatting then
    inpainting) plus an SBS-split post-step.  These tests stub ``subprocess.run``
    and ``_split_sbs_video`` so no real inference or video I/O happens; they
    verify the command structure, call order, and error handling.
    """

    @staticmethod
    def _make_repo(tmp_path):
        """A repo dir with both real upstream entry scripts (valid for construction)."""
        repo = tmp_path / "stereocrafter"
        repo.mkdir()
        (repo / "inpainting_inference.py").write_text("# stage 2")
        (repo / "depth_splatting_inference.py").write_text("# stage 1")
        return repo

    def test_render_video_success(self, tmp_path):
        """Both stages succeed; backend returns the L/R paths from the split step."""
        from pipeline.stereo_crafter import CLIBackend

        repo = self._make_repo(tmp_path)
        backend = CLIBackend(repo_dir=str(repo))

        depth_dir = tmp_path / "depth"
        depth_dir.mkdir()
        input_video = tmp_path / "input.mp4"
        input_video.write_text("")
        left_out = str(tmp_path / "left_out.mp4")
        right_out = str(tmp_path / "right_out.mp4")

        # Pin the work dir so we can pre-create the SBS output the backend looks for.
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        sbs_name = "splatting_results_inpainting_results_sbs.mp4"
        (work_dir / sbs_name).write_text("fake sbs")

        with (
            patch("subprocess.run") as mock_run,
            patch.object(CLIBackend, "_split_sbs_video", return_value=None) as mock_split,
            patch("pipeline.stereo_crafter._assert_cuda", return_value=None),
            patch("pipeline.stereo_crafter.tempfile.mkdtemp", return_value=str(work_dir)),
        ):
            mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

            # _split_sbs_video is responsible for materializing the outputs.
            def _fake_split(sbs_path, left, right):
                Path(left).write_text("data")
                Path(right).write_text("data")

            mock_split.side_effect = _fake_split

            result_l, result_r = backend.render_video(
                input_path=str(input_video),
                depth_dir=str(depth_dir),
                output_left=left_out,
                output_right=right_out,
            )

        assert result_l == left_out
        assert result_r == right_out
        # Two stages → two subprocess calls, in stage order.
        assert mock_run.call_count == 2
        mock_split.assert_called_once()
        # Stage 1 must run before Stage 2 (splatting script first).
        first_cmd = mock_run.call_args_list[0].args[0]
        second_cmd = mock_run.call_args_list[1].args[0]
        assert first_cmd[1].endswith("depth_splatting_inference.py")
        assert second_cmd[1].endswith("inpainting_inference.py")

    def test_render_video_success_output_not_spammed_at_info(self, tmp_path, caplog):
        """Successful runs log the subprocess tail at DEBUG only (issue #127).

        With the logger at INFO, none of the child's chatty stdout/stderr may
        reach the log; at DEBUG the tail becomes visible.
        """
        import logging

        from pipeline.stereo_crafter import CLIBackend

        repo = self._make_repo(tmp_path)
        backend = CLIBackend(repo_dir=str(repo))

        depth_dir = tmp_path / "depth"
        depth_dir.mkdir()
        input_video = tmp_path / "input.mp4"
        input_video.write_text("")
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        (work_dir / "splatting_results_inpainting_results_sbs.mp4").write_text("fake sbs")

        def _fake_run(cmd, **kwargs):
            kwargs["stdout"].write(b"noisy-child-stdout\n")
            kwargs["stderr"].write(b"noisy-child-stderr\n")
            return subprocess.CompletedProcess(args=cmd, returncode=0)

        def _fake_split(sbs_path, left, right):
            Path(left).write_text("data")
            Path(right).write_text("data")

        with (
            patch("subprocess.run", side_effect=_fake_run),
            patch.object(CLIBackend, "_split_sbs_video", side_effect=_fake_split),
            patch("pipeline.stereo_crafter._assert_cuda", return_value=None),
            patch("pipeline.stereo_crafter.tempfile.mkdtemp", return_value=str(work_dir)),
            caplog.at_level(logging.INFO, logger="pipeline.stereo_crafter"),
        ):
            backend.render_video(
                input_path=str(input_video),
                depth_dir=str(depth_dir),
                output_left=str(tmp_path / "left.mp4"),
                output_right=str(tmp_path / "right.mp4"),
            )

        text = caplog.text
        assert "noisy-child-stdout" not in text
        assert "noisy-child-stderr" not in text
        # The command line itself is still logged at INFO (existing behaviour).
        assert "StereoCrafter CLIBackend" in text

        # At DEBUG the tail IS available for troubleshooting.
        caplog.clear()
        with (
            patch("subprocess.run", side_effect=_fake_run),
            patch.object(CLIBackend, "_split_sbs_video", side_effect=_fake_split),
            patch("pipeline.stereo_crafter._assert_cuda", return_value=None),
            patch("pipeline.stereo_crafter.tempfile.mkdtemp", return_value=str(work_dir)),
            caplog.at_level(logging.DEBUG, logger="pipeline.stereo_crafter"),
        ):
            backend.render_video(
                input_path=str(input_video),
                depth_dir=str(depth_dir),
                output_left=str(tmp_path / "left.mp4"),
                output_right=str(tmp_path / "right.mp4"),
            )
        assert "noisy-child-stdout" in caplog.text
        assert "noisy-child-stderr" in caplog.text

    def test_render_video_command_uses_real_fire_flags(self, tmp_path):
        """Commands use the real upstream fire-style flags (not the old fictional ones)."""
        from pipeline.stereo_crafter import CLIBackend

        repo = self._make_repo(tmp_path)
        backend = CLIBackend(
            repo_dir=str(repo),
            python_exe="python3",
            checkpoint_dir=str(tmp_path / "sc_unet"),
            pre_trained_path=str(tmp_path / "svd"),
            depthcrafter_unet_path=str(tmp_path / "dc_unet"),
        )

        depth_dir = tmp_path / "depth"
        depth_dir.mkdir()
        input_video = tmp_path / "input.mp4"
        input_video.write_text("")

        # Pin the work dir and pre-create the SBS file so the flow reaches the split.
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        (work_dir / "splatting_results_inpainting_results_sbs.mp4").write_text("fake sbs")

        with (
            patch("subprocess.run") as mock_run,
            patch.object(CLIBackend, "_split_sbs_video", return_value=None),
            patch("pipeline.stereo_crafter._assert_cuda", return_value=None),
            patch("pipeline.stereo_crafter.tempfile.mkdtemp", return_value=str(work_dir)),
        ):
            mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
            backend.render_video(
                input_path=str(input_video),
                depth_dir=str(depth_dir),
                output_left=str(tmp_path / "left.mp4"),
                output_right=str(tmp_path / "right.mp4"),
            )

        cmd1 = mock_run.call_args_list[0].args[0]
        cmd2 = mock_run.call_args_list[1].args[0]
        # Stage 1 real flags.
        assert "--pre_trained_path" in cmd1
        assert "--unet_path" in cmd1
        assert "--input_video_path" in cmd1
        assert "--output_video_path" in cmd1
        assert cmd1[0] == "python3"
        # Stage 2 real flags.
        assert "--pre_trained_path" in cmd2
        assert "--unet_path" in cmd2
        assert "--input_video_path" in cmd2
        assert "--save_dir" in cmd2
        # The fictional flags must NOT appear.
        for fictional in (
            "--video",
            "--depth_dir",
            "--output_left",
            "--output_right",
            "--max_resolution",
            "--checkpoint_dir",
        ):
            assert fictional not in cmd1
            assert fictional not in cmd2
        # shell=True must never be used.
        for call in mock_run.call_args_list:
            assert call.kwargs.get("shell") in (None, False)

    def test_render_video_subprocess_failure(self, tmp_path):
        """A non-zero exit in a stage raises a clear RuntimeError naming the stage.

        Issue #127: stdout/stderr are drained to temp files (no PIPE), so the
        mock writes the child's stderr into the file the backend hands to
        ``subprocess.run`` — the message must contain that stderr tail plus
        the command, cwd, and output-dir contents.
        """
        from pipeline.stereo_crafter import CLIBackend

        repo = self._make_repo(tmp_path)
        backend = CLIBackend(repo_dir=str(repo))

        depth_dir = tmp_path / "depth"
        depth_dir.mkdir()
        input_video = tmp_path / "input.mp4"
        input_video.write_text("")

        def _fake_run(cmd, **kwargs):
            # Simulate the child writing to its drained stderr/stdout files.
            kwargs["stdout"].write(b"stage log line\n")
            kwargs["stderr"].write(b"loading unet\nCUDA OOM error\n")
            return subprocess.CompletedProcess(args=cmd, returncode=1)

        with (
            patch("subprocess.run", side_effect=_fake_run) as mock_run,
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
        assert "failed" in msg
        assert "CUDA OOM" in msg
        # Issue #127: command, cwd, and the real output-dir contents are in the message.
        assert "Stage 1 (depth splatting)" in msg
        assert "depth_splatting_inference.py" in msg
        assert f"cwd: {backend.repo_dir}" in msg
        assert "Output dir contents:" in msg
        assert mock_run.call_count == 1  # stage 1 failed, stage 2 never ran

    def test_render_video_missing_sbs_output(self, tmp_path):
        """Both stages succeed but the SBS file is absent -> clear error."""
        from pipeline.stereo_crafter import CLIBackend

        repo = self._make_repo(tmp_path)
        backend = CLIBackend(repo_dir=str(repo))

        depth_dir = tmp_path / "depth"
        depth_dir.mkdir()
        input_video = tmp_path / "input.mp4"
        input_video.write_text("")

        with patch("subprocess.run") as mock_run, patch("pipeline.stereo_crafter._assert_cuda", return_value=None):
            mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

            with pytest.raises(RuntimeError) as exc_info:
                backend.render_video(
                    input_path=str(input_video),
                    depth_dir=str(depth_dir),
                    output_left=str(tmp_path / "missing_left.mp4"),
                    output_right=str(tmp_path / "missing_right.mp4"),
                )
        msg = str(exc_info.value)
        assert "SBS output not found" in msg
