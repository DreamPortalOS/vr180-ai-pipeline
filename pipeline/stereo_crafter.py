"""StereoCrafter — depth-aware stereo video generation with disocclusion inpainting.

Provides a StereoCrafterRenderer that delegates to Tencent/StereoCrafter
inference scripts via a pluggable backend (default: CLIBackend).
CUDA-only; raises clear errors on CPU/Mac builds.

StereoCrafter (Tencent) uses depth-guided forward splatting + video diffusion
inpainting to produce clean stereoscopic left/right views without the
ghosting/smear artifacts of simple depth-based shifting.

Usage::

    from pipeline.stereo_crafter import StereoCrafterRenderer

    renderer = StereoCrafterRenderer()
    left_video, right_video = renderer.render_video("input.mp4", depth_maps)

Reference:
    https://github.com/Tencent/StereoCrafter
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CUDA guard
# ---------------------------------------------------------------------------


def _assert_cuda() -> None:
    """Raise RuntimeError if CUDA is not available."""
    try:
        import torch
    except ImportError:
        raise RuntimeError("PyTorch is not installed. StereoCrafter requires PyTorch with CUDA.") from None
    if not torch.cuda.is_available():  # type: ignore[attr-defined]
        raise RuntimeError(
            "CUDA is not available — cannot run StereoCrafterRenderer.\n"
            "This stereo renderer requires an NVIDIA GPU with CUDA support.\n"
            "See docs/STEREOCRAFTER_SETUP.md for setup instructions."
        )


# ---------------------------------------------------------------------------
# Abstract backend
# ---------------------------------------------------------------------------


class StereoCrafterBackend(ABC):
    """Pluggable backend for the actual StereoCrafter inference call."""

    @abstractmethod
    def render_video(
        self,
        input_path: str,
        depth_dir: str,
        output_left: str,
        output_right: str,
    ) -> tuple[str, str]:
        """Run StereoCrafter inference and return paths to L/R videos.

        Args:
            input_path: Path to the input video file.
            depth_dir: Directory containing depth maps (as .npy or .png).
            output_left: Desired path for the left-eye output video.
            output_right: Desired path for the right-eye output video.

        Returns:
            Tuple of (left_video_path, right_video_path).
        """
        ...


# ---------------------------------------------------------------------------
# CLI backend
# ---------------------------------------------------------------------------


class CLIBackend(StereoCrafterBackend):
    """Backend that runs StereoCrafter's inference script as a subprocess.

    Spawns the StereoCrafter repository's inference script — no server
    required.  All paths can be set via constructor arguments or
    environment variables:

    ========================== ========================= ===============================
    Constructor param          Env var                   Default
    ========================== ========================= ===============================
    ``repo_dir``               ``STEREOCRAFTER_REPO_DIR`` ``(none, required)``
    ``python_exe``             ``STEREOCRAFTER_PYTHON``   ``python``
    ``checkpoint_dir``         ``STEREOCRAFTER_CKPT_DIR`` ``(repo_dir)/checkpoints``
    ``max_resolution``         ``STEREOCRAFTER_MAX_RES``  ``1024``
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
        _repo_dir = repo_dir or os.environ.get("STEREOCRAFTER_REPO_DIR")
        if not _repo_dir:
            raise RuntimeError(
                "StereoCrafter repository directory not specified.\n"
                "Set --stereocrafter-repo-dir or the STEREOCRAFTER_REPO_DIR "
                "environment variable.\n"
                "See docs/STEREOCRAFTER_SETUP.md for setup instructions."
            )
        self.repo_dir: str = str(Path(_repo_dir).resolve())

        # python_exe
        self.python_exe = python_exe or os.environ.get("STEREOCRAFTER_PYTHON", "python")

        # checkpoint_dir
        if checkpoint_dir:
            self.checkpoint_dir = str(Path(checkpoint_dir).resolve())
        elif os.environ.get("STEREOCRAFTER_CKPT_DIR"):
            self.checkpoint_dir = str(Path(os.environ["STEREOCRAFTER_CKPT_DIR"]).resolve())
        else:
            self.checkpoint_dir = str(Path(self.repo_dir) / "checkpoints")

        # max resolution for inference (short side)
        self.max_resolution = max_resolution or int(os.environ.get("STEREOCRAFTER_MAX_RES", "1024"))

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
                f"StereoCrafter repository not found at: {self.repo_dir}\n"
                f"  Clone the repo:\n"
                f"    git clone https://github.com/Tencent/StereoCrafter.git\n"
                f"  See docs/STEREOCRAFTER_SETUP.md for details."
            )

        # Look for a known inference entry point
        if repo.is_dir():
            candidates = [
                repo / "run.py",
                repo / "inference.py",
                repo / "scripts" / "inference.py",
                repo / "stereocrafter" / "inference.py",
            ]
            found = any(c.is_file() for c in candidates)
            if not found:
                issues.append(
                    f"No known inference script found in {self.repo_dir}.\n"
                    f"  Expected one of: {', '.join(str(c.relative_to(repo)) for c in candidates)}\n"
                    f"  See docs/STEREOCRAFTER_SETUP.md for the required file layout."
                )

        if issues:
            raise RuntimeError("StereoCrafter setup is incomplete:\n" + "\n".join(f"  \u2022 {i}" for i in issues))

    def _find_inference_script(self) -> str:
        """Return the path to the first known inference entry point."""
        repo = Path(self.repo_dir)
        candidates = [
            repo / "run.py",
            repo / "inference.py",
            repo / "scripts" / "inference.py",
            repo / "stereocrafter" / "inference.py",
        ]
        for c in candidates:
            if c.is_file():
                return str(c)
        raise RuntimeError(f"No known inference script found in {self.repo_dir}. See docs/STEREOCRAFTER_SETUP.md.")

    # ------------------------------------------------------------------
    # Main inference method
    # ------------------------------------------------------------------
    def render_video(
        self,
        input_path: str,
        depth_dir: str,
        output_left: str,
        output_right: str,
    ) -> tuple[str, str]:
        _assert_cuda()

        inference_script = self._find_inference_script()

        # Build command list (NO shell=True for security)
        cmd: list[str] = [
            self.python_exe,
            inference_script,
            "--video",
            input_path,
            "--depth_dir",
            depth_dir,
            "--output_left",
            output_left,
            "--output_right",
            output_right,
            "--max_resolution",
            str(self.max_resolution),
            "--checkpoint_dir",
            self.checkpoint_dir,
        ]

        log.info("StereoCrafter CLIBackend command: %s", " ".join(cmd))
        log.info("StereoCrafter CLIBackend cwd: %s", self.repo_dir)

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
                f"Set STEREOCRAFTER_PYTHON or --stereocrafter-python "
                f"to the correct path."
            ) from exc
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                "StereoCrafter inference timed out after 2 hours. The video may be too long or the GPU too slow."
            ) from None

        if result.returncode != 0:
            stderr = result.stderr.strip()
            raise RuntimeError(
                f"StereoCrafter inference failed (exit code {result.returncode}):\n"
                f"  {stderr}\n"
                f"  Command: {' '.join(cmd)}\n"
                f"  See docs/STEREOCRAFTER_SETUP.md for troubleshooting."
            )

        # Verify outputs exist
        left_path = Path(output_left)
        right_path = Path(output_right)
        missing: list[str] = []
        if not left_path.is_file():
            missing.append(str(left_path))
        if not right_path.is_file():
            missing.append(str(right_path))
        if missing:
            raise RuntimeError(
                f"StereoCrafter finished but output video(s) not found:\n"
                f"  {', '.join(missing)}\n"
                f"  Check the inference script output format "
                f"in docs/STEREOCRAFTER_SETUP.md."
            )

        log.info(
            "StereoCrafter: L → %s, R → %s",
            str(left_path),
            str(right_path),
        )
        return str(left_path), str(right_path)


# ---------------------------------------------------------------------------
# StereoCrafterRenderer (front-facing class)
# ---------------------------------------------------------------------------


class StereoCrafterRenderer:
    """Wrapper for StereoCrafter depth-aware stereo video generation.

    CUDA-only.  Processes an entire video clip with depth maps to produce
    clean stereoscopic left/right views with disocclusion inpainting.

    The *backend* argument allows injecting a different backend (e.g.
    for testing).  Defaults to :class:`CLIBackend`.
    """

    def __init__(
        self,
        backend: StereoCrafterBackend | None = None,
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

    def render_video(
        self,
        input_path: str,
        depth_dir: str | None = None,
        output_left: str | None = None,
        output_right: str | None = None,
    ) -> tuple[str, str]:
        """Generate stereoscopic L/R videos with disocclusion inpainting.

        Args:
            input_path: Path to input video file.
            depth_dir: Directory containing per-frame depth maps (as .npy or .png).
                If None, a temporary directory is created and must be populated
                by the caller before calling this method.
            output_left: Desired path for the left-eye output video.
                If None, a temp path is generated.
            output_right: Desired path for the right-eye output video.
                If None, a temp path is generated.

        Returns:
            Tuple of (left_video_path, right_video_path).
        """
        resolved_depth = depth_dir or tempfile.mkdtemp(prefix="stereocrafter_depth_")
        resolved_left = output_left or tempfile.mktemp(suffix=".mp4", prefix="stereocrafter_left_")
        resolved_right = output_right or tempfile.mktemp(suffix=".mp4", prefix="stereocrafter_right_")

        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"Input video not found: {input_path}")

        if not os.path.isdir(resolved_depth):
            raise NotADirectoryError(
                f"Depth directory not found: {resolved_depth}. "
                f"Provide a valid --depth-dir or ensure depth maps are saved."
            )

        log.info(
            "StereoCrafterRenderer: %s + depth/ → %s | %s",
            input_path,
            resolved_left,
            resolved_right,
        )

        return self.backend.render_video(
            input_path=input_path,
            depth_dir=resolved_depth,
            output_left=resolved_left,
            output_right=resolved_right,
        )
