"""DepthCrafter — temporally-consistent video depth estimation.

Provides a DepthCrafterEstimator that delegates to Tencent/DepthCrafter
inference scripts via a pluggable backend (default: CLIBackend).
CUDA-only; raises clear errors on CPU/Mac builds.

DepthCrafter processes an entire video clip at once, producing temporally
smooth depth maps that eliminate the flickering / ghosting artifacts of
per-frame depth estimators (such as Depth-Anything V2).

Usage::

    from pipeline.depth_crafter import DepthCrafterEstimator

    estimator = DepthCrafterEstimator()
    depths = estimator.estimate_video("input.mp4", "output_depth_dir/")

Reference:
    https://github.com/Tencent/DepthCrafter
"""

from __future__ import annotations

import logging
import os
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CUDA guard
# ---------------------------------------------------------------------------


def _assert_cuda() -> None:
    """Raise RuntimeError if CUDA is not available."""
    try:
        import torch
    except ImportError:
        raise RuntimeError("PyTorch is not installed. DepthCrafter requires PyTorch with CUDA.") from None
    if not torch.cuda.is_available():  # type: ignore[attr-defined]
        raise RuntimeError(
            "CUDA is not available — cannot run DepthCrafterEstimator.\n"
            "This depth estimator requires an NVIDIA GPU with CUDA support.\n"
            "See docs/DEPTHCRAFTER_SETUP.md for setup instructions."
        )


# ---------------------------------------------------------------------------
# Abstract backend
# ---------------------------------------------------------------------------


class DepthCrafterBackend(ABC):
    """Pluggable backend for the actual DepthCrafter inference call."""

    @abstractmethod
    def estimate_video(
        self,
        input_path: str,
        output_dir: str,
    ) -> list[np.ndarray]:
        """Run depth estimation on *input_path* and return depth maps.

        Returns a list of (H, W) float32 depth maps, one per frame.
        The backend is responsible for saving intermediate results to
        *output_dir* as needed.
        """
        ...


# ---------------------------------------------------------------------------
# CLI backend
# ---------------------------------------------------------------------------


class CLIBackend(DepthCrafterBackend):
    """Backend that runs DepthCrafter's inference script as a subprocess.

    Spawns the DepthCrafter repository's inference script — no server
    required.  All paths can be set via constructor arguments or
    environment variables:

    ========================== ========================= ===============================
    Constructor param          Env var                   Default
    ========================== ========================= ===============================
    ``repo_dir``               ``DEPTHCRAFTER_REPO_DIR`` ``(none, required)``
    ``python_exe``             ``DEPTHCRAFTER_PYTHON``   ``python``
    ``checkpoint_dir``         ``DEPTHCRAFTER_CKPT_DIR`` ``(repo_dir)/checkpoints``
    ``max_resolution``         ``DEPTHCRAFTER_MAX_RES``  ``1024``
    ========================== ========================= ===============================
    """

    def __init__(
        self,
        repo_dir: str | None = None,
        python_exe: str | None = None,
        checkpoint_dir: str | None = None,
        max_resolution: int | None = None,
    ) -> None:
        # repo_dir: required (env fallback)
        _repo_dir = repo_dir or os.environ.get("DEPTHCRAFTER_REPO_DIR")
        if not _repo_dir:
            raise RuntimeError(
                "DepthCrafter repository directory not specified.\n"
                "Set --depthcrafter-repo-dir or the DEPTHCRAFTER_REPO_DIR "
                "environment variable.\n"
                "See docs/DEPTHCRAFTER_SETUP.md for setup instructions."
            )
        self.repo_dir: str = str(Path(_repo_dir).resolve())

        # python_exe
        self.python_exe = python_exe or os.environ.get("DEPTHCRAFTER_PYTHON", "python")

        # checkpoint_dir
        if checkpoint_dir:
            self.checkpoint_dir = str(Path(checkpoint_dir).resolve())
        elif os.environ.get("DEPTHCRAFTER_CKPT_DIR"):
            self.checkpoint_dir = str(Path(os.environ["DEPTHCRAFTER_CKPT_DIR"]).resolve())
        else:
            self.checkpoint_dir = str(Path(self.repo_dir) / "checkpoints")

        # max resolution for inference (short side)
        self.max_resolution = max_resolution or int(os.environ.get("DEPTHCRAFTER_MAX_RES", "1024"))

        # Verify paths
        self._validate_paths()

    # ------------------------------------------------------------------
    # Path validation
    # ------------------------------------------------------------------
    def _validate_paths(self) -> None:
        """Check critical paths exist. Does NOT require checkpoints —
        they can be downloaded by the user later."""
        issues: list[str] = []

        repo = Path(self.repo_dir)
        if not repo.is_dir():
            issues.append(
                f"DepthCrafter repository not found at: {self.repo_dir}\n"
                f"  Clone the repo:\n"
                f"    git clone https://github.com/Tencent/DepthCrafter.git\n"
                f"  See docs/DEPTHCRAFTER_SETUP.md for details."
            )

        # Look for a known inference entry point
        if repo.is_dir():
            candidates = [
                repo / "run.py",
                repo / "inference.py",
                repo / "scripts" / "inference.py",
                repo / "depthcrafter" / "inference.py",
            ]
            found = any(c.is_file() for c in candidates)
            if not found:
                issues.append(
                    f"No known inference script found in {self.repo_dir}.\n"
                    f"  Expected one of: {', '.join(str(c.relative_to(repo)) for c in candidates)}\n"
                    f"  See docs/DEPTHCRAFTER_SETUP.md for the required file layout."
                )

        if issues:
            raise RuntimeError("DepthCrafter setup is incomplete:\n" + "\n".join(f"  \u2022 {i}" for i in issues))

    def _find_inference_script(self) -> str:
        """Return the path to the first known inference entry point."""
        repo = Path(self.repo_dir)
        candidates = [
            repo / "run.py",
            repo / "inference.py",
            repo / "scripts" / "inference.py",
            repo / "depthcrafter" / "inference.py",
        ]
        for c in candidates:
            if c.is_file():
                return str(c)
        raise RuntimeError(f"No known inference script found in {self.repo_dir}. See docs/DEPTHCRAFTER_SETUP.md.")

    # ------------------------------------------------------------------
    # Main inference method
    # ------------------------------------------------------------------
    def estimate_video(
        self,
        input_path: str,
        output_dir: str,
    ) -> list[np.ndarray]:
        _assert_cuda()

        # Ensure output dir exists
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        inference_script = self._find_inference_script()

        # Build command list (NO shell=True for security)
        cmd: list[str] = [
            self.python_exe,
            inference_script,
            "--video",
            input_path,
            "--output_dir",
            output_dir,
            "--max_resolution",
            str(self.max_resolution),
            "--checkpoint_dir",
            self.checkpoint_dir,
        ]

        log.info("DepthCrafter CLIBackend command: %s", " ".join(cmd))
        log.info("DepthCrafter CLIBackend cwd: %s", self.repo_dir)

        try:
            result = subprocess.run(
                cmd,
                cwd=self.repo_dir,
                capture_output=True,
                text=True,
                timeout=7200,  # 2 hours max
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Python executable not found: {self.python_exe}. "
                f"Set DEPTHCRAFTER_PYTHON or --depthcrafter-python to the correct path."
            ) from exc
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                "DepthCrafter inference timed out after 2 hours. The video may be too long or the GPU too slow."
            ) from None

        if result.returncode != 0:
            stderr = result.stderr.strip()
            raise RuntimeError(
                f"DepthCrafter inference failed (exit code {result.returncode}):\n"
                f"  {stderr}\n"
                f"  Command: {' '.join(cmd)}\n"
                f"  See docs/DEPTHCRAFTER_SETUP.md for troubleshooting."
            )

        # Load depth maps from output dir
        depths: list[np.ndarray] = []
        npy_files = sorted(out_path.glob("*.npy"))
        png_files = sorted(out_path.glob("depth_*.png"))

        if npy_files:
            for f in npy_files:
                depths.append(np.load(str(f)))
        elif png_files:
            import cv2

            for f in png_files:
                img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
                if img is not None:
                    depths.append(img.astype(np.float32) / 255.0)
        else:
            raise RuntimeError(
                f"DepthCrafter finished but no depth files found in {output_dir}.\n"
                f"  Check the inference script output format in docs/DEPTHCRAFTER_SETUP.md."
            )

        log.info("DepthCrafter: loaded %d depth maps from %s", len(depths), output_dir)
        return depths


# ---------------------------------------------------------------------------
# DepthCrafterEstimator (front-facing class)
# ---------------------------------------------------------------------------


class DepthCrafterEstimator:
    """Wrapper for DepthCrafter temporally-consistent video depth estimation.

    CUDA-only.  Processes an entire video clip at once to produce
    temporally smooth depth maps.

    The *backend* argument allows injecting a different backend (e.g.
    for testing).  Defaults to :class:`CLIBackend`.
    """

    def __init__(
        self,
        backend: DepthCrafterBackend | None = None,
        repo_dir: str | None = None,
        python_exe: str | None = None,
        checkpoint_dir: str | None = None,
        max_resolution: int | None = None,
    ) -> None:
        _assert_cuda()

        if backend is not None:
            self.backend = backend
        else:
            self.backend = CLIBackend(
                repo_dir=repo_dir,
                python_exe=python_exe,
                checkpoint_dir=checkpoint_dir,
                max_resolution=max_resolution,
            )

    def estimate_video(
        self,
        input_path: str,
        output_dir: str | None = None,
    ) -> list[np.ndarray]:
        """Estimate temporally-consistent depth for an entire video.

        Args:
            input_path: Path to input video file.
            output_dir: Directory to save intermediate depth outputs.
                If None, a temporary directory is created.

        Returns:
            List of (H, W) float32 depth maps, one per frame.
        """
        import tempfile

        resolved_output = output_dir or tempfile.mkdtemp(prefix="depthcrafter_")

        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"Input video not found: {input_path}")

        log.info(
            "DepthCrafterEstimator: %s \u2192 %s",
            input_path,
            resolved_output,
        )

        return self.backend.estimate_video(
            input_path=input_path,
            output_dir=resolved_output,
        )
