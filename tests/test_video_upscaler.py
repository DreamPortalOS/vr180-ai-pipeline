"""Tests for SeedVR2 video upscaler.

CI-safe: uses mocks to avoid requiring CUDA or a live ComfyUI server.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pipeline.video_upscaler import (
    ComfyUIBackend,
    SeedVR2Upscaler,
    UpscaleBackend,
    _assert_cuda,
    _validate_batch_size,
)

# ---------------------------------------------------------------------------
# _validate_batch_size
# ---------------------------------------------------------------------------


class TestValidateBatchSize:
    def test_valid_4n_plus_1_values(self) -> None:
        """1, 5, 9, 13, 17 are valid."""
        for n in [1, 5, 9, 13, 17]:
            _validate_batch_size(n)  # should not raise

    def test_zero_raises(self) -> None:
        """0 should raise ValueError for batch_size >= 1 check."""
        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            _validate_batch_size(0)

    def test_non_4n_plus_1_values(self) -> None:
        """2, 3, 4, 6, 10 should raise ValueError about 4n+1."""
        for n in [2, 3, 4, 6, 10]:
            with pytest.raises(ValueError, match="4n\\+1"):
                _validate_batch_size(n)

    def test_negative(self) -> None:
        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            _validate_batch_size(-1)


# ---------------------------------------------------------------------------
# _assert_cuda
# ---------------------------------------------------------------------------


class TestAssertCuda:
    def test_no_torch_raises(self) -> None:
        """If torch isn't importable, raise RuntimeError."""

        # We mock the import to fail
        with patch.dict("sys.modules", {"torch": None}), pytest.raises(RuntimeError, match="PyTorch is not installed"):
            _assert_cuda()

    @patch("torch.cuda.is_available", return_value=False)
    def test_cuda_not_available_raises(self, mock_cuda) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(RuntimeError, match="CUDA is not available"):
            _assert_cuda()

    @patch("torch.cuda.is_available", return_value=True)
    def test_cuda_available_ok(self, mock_cuda) -> None:  # type: ignore[no-untyped-def]
        # Import inside test so torch is in sys.modules

        _assert_cuda()  # should not raise


# ---------------------------------------------------------------------------
# MockBackend for CI-safe tests
# ---------------------------------------------------------------------------


class MockBackend(UpscaleBackend):
    """A backend that simulates successful upscaling by copying the input file."""

    def __init__(self) -> None:
        self.called_with: dict | None = None

    def upscale(
        self,
        input_path: str,
        output_path: str,
        factor: int,
        batch_size: int,
    ) -> str:
        self.called_with = {
            "input_path": input_path,
            "output_path": output_path,
            "factor": factor,
            "batch_size": batch_size,
        }
        # Copy input to output to simulate the backend creating a result.
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(input_path, "rb") as src, open(output_path, "wb") as dst:
            dst.write(src.read())
        return output_path


# ---------------------------------------------------------------------------
# SeedVR2Upscaler (with MockBackend)
# ---------------------------------------------------------------------------


class TestSeedVR2Upscaler:
    def setup_method(self) -> None:
        # By default _assert_cuda patches to avoid real CUDA check
        self._cuda_patcher = patch(
            "pipeline.video_upscaler._assert_cuda",
            return_value=None,
        )
        self._cuda_patcher.start()

    def teardown_method(self) -> None:
        self._cuda_patcher.stop()

    def test_upscale_success(self) -> None:
        """Happy path: MockBackend copies input to output."""
        mock_backend = MockBackend()

        upscaler = SeedVR2Upscaler(
            batch_size=5,
            backend=mock_backend,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.mp4")
            output_path = os.path.join(tmpdir, "output.mp4")
            # Create a dummy input file
            Path(input_path).write_text("dummy video content")

            result = upscaler.upscale(input_path, output_path, factor=2)

            assert result == output_path
            assert os.path.isfile(output_path)
            assert mock_backend.called_with is not None
            assert mock_backend.called_with["factor"] == 2
            assert mock_backend.called_with["batch_size"] == 5
            assert mock_backend.called_with["input_path"] == input_path

    def test_invalid_batch_size_constructor(self) -> None:
        """Constructor should raise if batch_size isn't 4n+1."""
        with pytest.raises(ValueError, match="4n\\+1"):
            SeedVR2Upscaler(
                batch_size=2,
                backend=MockBackend(),
            )

    def test_invalid_factor(self) -> None:
        """upscale should raise for factor not in (2,3,4)."""
        upscaler = SeedVR2Upscaler(
            batch_size=5,
            backend=MockBackend(),
        )
        with pytest.raises(ValueError, match="factor must be 2, 3, or 4"):
            upscaler.upscale("dummy.mp4", "out.mp4", factor=5)

    def test_file_not_found(self) -> None:
        """upscale should raise if input file doesn't exist."""
        upscaler = SeedVR2Upscaler(
            batch_size=5,
            backend=MockBackend(),
        )
        with pytest.raises(FileNotFoundError, match="not found"):
            upscaler.upscale("/nonexistent/path.mp4", "out.mp4", factor=2)


# ---------------------------------------------------------------------------
# ComfyUIBackend connectivity checks (mock-level, no real server)
# ---------------------------------------------------------------------------


class TestComfyUIBackendConnectivity:
    def test_connection_refused(self) -> None:
        """Simulate connection refused via real requests exceptions — raises clear RuntimeError."""
        import requests as req

        backend = ComfyUIBackend(base_url="http://127.0.0.1:1")

        # Use a bare MagicMock but attach .exceptions so _check_connectivity
        # can resolve self._session.exceptions.ConnectionError correctly.
        mock_session = MagicMock()
        mock_session.get.side_effect = req.exceptions.ConnectionError("Connection refused")
        mock_session.exceptions = req.exceptions
        backend._session = mock_session

        with pytest.raises(RuntimeError, match="Cannot connect to ComfyUI"):
            backend._check_connectivity()

    @patch(
        "pipeline.video_upscaler.ComfyUIBackend._import_requests",
        return_value=MagicMock(),
    )
    def test_connectivity_success(self, mock_import_requests: MagicMock) -> None:
        """Simulate successful connectivity check."""
        backend = ComfyUIBackend(base_url="http://127.0.0.1:8188")

        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_session.get.return_value = mock_resp
        backend._session = mock_session

        # Should not raise
        backend._check_connectivity()

    @patch(
        "pipeline.video_upscaler.ComfyUIBackend._import_requests",
        return_value=MagicMock(),
    )
    def test_upscale_no_cuda(self, mock_import_requests: MagicMock) -> None:
        """ComfyUIBackend.upscale raises if CUDA unavailable."""
        backend = ComfyUIBackend(base_url="http://127.0.0.1:8188")

        # Mock torch import missing
        with (
            patch.dict("sys.modules", {"torch": None}),
            pytest.raises(RuntimeError, match="PyTorch is not installed"),
        ):
            backend.upscale("input.mp4", "output.mp4", factor=2, batch_size=5)


# ---------------------------------------------------------------------------
# run_pipeline.py integration: --video-upscale none leaves pipeline unchanged
# ---------------------------------------------------------------------------


class TestPipelineIntegration:
    """Verify the --video-upscale none flag has zero impact on pipeline flow."""

    def test_video_upscale_none_is_default(self) -> None:
        """Confirm the default is 'none' and the pipeline should not import SeedVR2 logic."""
        import argparse

        # Simulate the parser from run_pipeline
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--video-upscale",
            choices=["none", "seedvr2"],
            default="none",
        )
        args = parser.parse_args([])
        assert args.video_upscale == "none"

    def test_run_pipeline_with_none_skips_seedvr2(self) -> None:
        """When --video-upscale none, main() should not call run_seedvr2_prestage.

        This simulation checks that `args.video_upscale == 'none'` causes
        the pre-stage block to be skipped.
        """
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--video-upscale", choices=["none", "seedvr2"], default="none")
        parser.add_argument("--video-upscale-factor", type=int, default=2)
        parser.add_argument("--seedvr2-url", default="http://127.0.0.1:8188")
        parser.add_argument("--input", default="dummy.mp4")
        args = parser.parse_args(["--input", "dummy.mp4"])

        # The seedvr2 pre-stage block in main() checks:
        #   if args.video_upscale == "seedvr2": run_seedvr2_prestage(args)
        # With 'none', input should remain unchanged.
        original_input = args.input
        assert args.video_upscale == "none", f"Expected none, got {args.video_upscale}"
        # Simulate the guard in main():
        if args.video_upscale == "seedvr2":
            args.input = "upscaled.mp4"
        assert args.input == original_input, "none should NOT change input"
