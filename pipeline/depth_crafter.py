"""DepthCrafter — temporally-consistent video depth estimation.

Provides a DepthCrafterEstimator that delegates to Tencent/DepthCrafter
inference scripts via a pluggable backend (default: CLIBackend).
CUDA-only; raises clear errors on CPU/Mac builds.

DepthCrafter processes an entire video clip at once, producing temporally
smooth depth maps that eliminate the flickering / ghosting artifacts of
per-frame depth estimators (such as Depth-Anything V2).  It is the
``--depth-model depthcrafter`` option of :mod:`scripts.run_pipeline`; the
default depth backend remains Depth-Anything.

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
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# How many tail lines of captured subprocess output to log on success/failure.
# Issue #127: the CLI's own output is the only clue to the real failure cause
# (e.g. the CLI wrote an .mp4 where the pipeline expected .npy), so keep enough
# tail lines to show it — but not the full multi-hour log.
_SUCCESS_LOG_LINES = 20
_FAILURE_LOG_LINES = 40


def _tail_lines(fh, max_lines: int) -> list[str]:
    """Return the last *max_lines* non-empty lines of a captured-output handle.

    Reads at most the last 256 KB so a multi-GB stdout does not get loaded
    into memory; unreadable handles produce a placeholder line instead of
    raising (the caller is already on an error path).
    """
    try:
        fh.flush()
        fh.seek(0, os.SEEK_END)
        size = fh.tell()
        fh.seek(max(0, size - 256 * 1024))
        text = fh.read().decode("utf-8", errors="replace")
    except (OSError, ValueError, AttributeError):
        return ["(captured output unreadable)"]
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    return lines[-max_lines:] if lines else ["(no output captured)"]


def _combined_output(fh, proc, max_lines: int) -> list[str]:
    """Tail lines of the capture file, falling back to the CompletedProcess.

    Real runs write into the capture file; mocked ``subprocess.run`` never
    does, so when the file is empty use the ``stdout``/``stderr`` attributes
    the mock populated instead (keeps existing tests meaningful).
    """
    lines = _tail_lines(fh, max_lines)
    if lines != ["(no output captured)"]:
        return lines
    parts: list[str] = []
    for stream_name in ("stdout", "stderr"):
        stream = getattr(proc, stream_name, None)
        if isinstance(stream, str) and stream.strip():
            parts.extend(ln.rstrip() for ln in stream.splitlines() if ln.strip())
    return parts[-max_lines:] if parts else lines


def _dir_listing(dir_path: str) -> str:
    """One-line-per-entry listing of *dir_path* for failure diagnostics.

    Issue #127: when the CLI "succeeds" but writes the wrong artifact (mp4
    instead of npy), the pipeline's generic error gives no hint — the actual
    directory contents are the fastest way to see what really landed there.
    """
    try:
        entries = sorted(os.listdir(dir_path))
    except OSError as exc:
        return f"    (cannot list directory: {exc})"
    if not entries:
        return "    (empty)"
    return "\n".join(f"    {e}" for e in entries[:50]) + ("\n    ..." if len(entries) > 50 else "")


def _run_subprocess_captured(
    cmd: list[str],
    *,
    cwd: str,
    timeout: int,
    label: str,
    output_dir: str | None = None,
) -> None:
    """Run *cmd* with stdout+stderr captured to a temp file; raise on failure.

    stdout/stderr go to an OS-drained temp file (the same pattern
    ``pipeline/streaming_pipeline.py`` uses for ffmpeg): an undrained
    ``PIPE`` fills its 64 KB buffer and deadlocks a chatty inference CLI,
    while ``DEVNULL`` would hide the very output this helper exists to show.

    On success the last ~20 combined lines are logged at DEBUG (no INFO spam).
    On failure the last ~40 lines are logged at ERROR and summarized into the
    raised ``RuntimeError`` along with the command, cwd, and the actual
    contents of *output_dir*.
    """
    log.info("%s command: %s", label, " ".join(cmd))
    log.info("%s cwd: %s", label, cwd)

    with tempfile.TemporaryFile(prefix="vr180-cli-capture-") as capture:
        try:
            proc = subprocess.run(
                cmd,
                cwd=cwd,
                stdout=capture,
                stderr=subprocess.STDOUT,
                timeout=timeout,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Python executable not found: {cmd[0]}. "
                f"Set the appropriate env var / CLI flag to the correct path.\n"
                f"  Command: {' '.join(cmd)}\n"
                f"  cwd: {cwd}"
            ) from exc
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"{label} timed out after {timeout // 3600} hours. "
                f"The video may be too long or the GPU too slow.\n"
                f"  Command: {' '.join(cmd)}\n"
                f"  cwd: {cwd}"
            ) from None

        if proc.returncode == 0:
            for line in _combined_output(capture, proc, _SUCCESS_LOG_LINES):
                log.debug("%s | %s", label, line)
            return

        tail = _combined_output(capture, proc, _FAILURE_LOG_LINES)
        for line in tail:
            log.error("%s | %s", label, line)
        summary = "\n".join(f"  {ln}" for ln in tail[-10:])
        msg = (
            f"{label} failed (exit code {proc.returncode}):\n"
            f"  --- subprocess output (tail) ---\n"
            f"{summary}\n"
            f"  Command: {' '.join(cmd)}\n"
            f"  cwd: {cwd}"
        )
        if output_dir is not None:
            msg += f"\n  Output dir: {output_dir}\n  Contents:\n{_dir_listing(output_dir)}"
        raise RuntimeError(msg)


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
# In-repo default paths (managed by scripts/setup_depthcrafter.py)
# ---------------------------------------------------------------------------
# The repo root is two parents up from this file (pipeline/depth_crafter.py).
_REPO_ROOT = Path(__file__).resolve().parent.parent
INREPO_REPO_DIR = _REPO_ROOT / "third_party" / "DepthCrafter"
INREPO_PYTHON_EXE = (
    INREPO_REPO_DIR / ".venv" / (Path("Scripts") / "python.exe" if os.name == "nt" else Path("bin") / "python")
)
INREPO_MODEL_DIR = _REPO_ROOT / "models" / "DepthCrafter"

# 12 GB VRAM-safe default for the short-side resolution cap.  The official
# default is 1024, but that blows the 12 GB buffer on an RTX 4070 SUPER;
# 512 is the lead-verified safe floor.  Tune via DEPTHCRAFTER_MAX_RES.
_DEFAULT_MAX_RES = 512


def _inrepo_env_hint() -> str:
    """Text appended to errors when no DepthCrafter paths were configured/found."""
    return (
        "No DepthCrafter repo/python/model paths were configured or found in-repo.\n"
        "  Run the one-command bootstrap to deploy DepthCrafter inside the repo:\n"
        "    python scripts/setup_depthcrafter.py\n"
        "  See docs/DEPTHCRAFTER_SETUP.md for disk/VRAM requirements and troubleshooting."
    )


# ---------------------------------------------------------------------------
# CLI backend
# ---------------------------------------------------------------------------


class CLIBackend(DepthCrafterBackend):
    """Backend that runs DepthCrafter's inference script as a subprocess.

    Spawns the DepthCrafter repository's ``run.py`` (fire-style CLI) — no
    server required.  All paths can be set via constructor arguments or
    environment variables, falling back to the in-repo defaults set up by
    :mod:`scripts.setup_depthcrafter`.  In-repo paths are only adopted when
    they actually exist on disk; otherwise the constructor raises and points
    at the bootstrap script.

    ========================== =============================== ===========================
    Constructor param          Env var                         Default (if env unset)
    ========================== =============================== ===========================
    ``repo_dir``               ``DEPTHCRAFTER_REPO_DIR``       in-repo ``third_party/DepthCrafter`` *(if exists)*
    ``python_exe``             ``DEPTHCRAFTER_PYTHON``         in-repo venv python *(if exists)*, else ``python``
    ``model_dir``              ``DEPTHCRAFTER_MODEL_DIR``      in-repo ``models/DepthCrafter`` *(if exists)*
    ``max_resolution``         ``DEPTHCRAFTER_MAX_RES``       ``512`` (12 GB VRAM-safe)
    ``process_length``         ``DEPTHCRAFTER_PROCESS_LENGTH`` unset (let the model decide)
    ``target_fps``             ``DEPTHCRAFTER_TARGET_FPS``     unset (source FPS)
    ========================== =============================== ===========================

    The underlying ``run.py`` CLI shape (lead-verified 2026-08-31) is
    fire-style, not argparse::

        python run.py <video_path> --save_folder <dir> --max_res 512 --cpu_offload model
                                    [--process_length N] [--target_fps N]
    """

    def __init__(
        self,
        repo_dir: str | None = None,
        python_exe: str | None = None,
        # Legacy compat: the old CLIBackend exposed ``checkpoint_dir``; modern
        # DepthCrafter weights live in a HF snapshot dir, so this aliases to
        # ``model_dir``.  Kept so the run_pipeline.py wiring doesn't break.
        checkpoint_dir: str | None = None,
        model_dir: str | None = None,
        max_resolution: int | None = None,
        process_length: int | None = None,
        target_fps: int | None = None,
    ) -> None:
        # repo_dir: explicit > env > in-repo default (only if it exists on disk)
        _repo_dir = repo_dir or os.environ.get("DEPTHCRAFTER_REPO_DIR")
        if not _repo_dir and INREPO_REPO_DIR.is_dir():
            _repo_dir = str(INREPO_REPO_DIR)
        if not _repo_dir:
            raise RuntimeError(
                "DepthCrafter repository directory not specified.\n"
                "Set --depthcrafter-repo-dir or the DEPTHCRAFTER_REPO_DIR "
                "environment variable, or run the in-repo bootstrap:\n"
                f"  {_inrepo_env_hint()}"
            )
        self.repo_dir: str = str(Path(_repo_dir).resolve())

        # python_exe: explicit > env > in-repo venv python (only if exists) > "python"
        if python_exe:
            self.python_exe = python_exe
        elif os.environ.get("DEPTHCRAFTER_PYTHON"):
            self.python_exe = os.environ["DEPTHCRAFTER_PYTHON"]
        elif INREPO_PYTHON_EXE.is_file():
            self.python_exe = str(INREPO_PYTHON_EXE)
        else:
            self.python_exe = "python"

        # model_dir: explicit > legacy checkpoint_dir compat > env > in-repo default
        _model_dir = model_dir or checkpoint_dir or os.environ.get("DEPTHCRAFTER_MODEL_DIR")
        if not _model_dir and os.environ.get("DEPTHCRAFTER_CKPT_DIR"):
            _model_dir = os.environ["DEPTHCRAFTER_CKPT_DIR"]
        if _model_dir:
            self.model_dir = str(Path(_model_dir).resolve())
        elif INREPO_MODEL_DIR.is_dir():
            self.model_dir = str(INREPO_MODEL_DIR)
        else:
            self.model_dir = str(Path(self.repo_dir) / "checkpoints")

        # Max resolution for inference (short side) — 12 GB VRAM-safe default.
        self.max_resolution = max_resolution or int(os.environ.get("DEPTHCRAFTER_MAX_RES", str(_DEFAULT_MAX_RES)))

        # Optional runtime tuning knobs forwarded to run.py (fire-style).  0 → unset.
        _pl = process_length or int(os.environ.get("DEPTHCRAFTER_PROCESS_LENGTH", "0")) or None
        self.process_length: int | None = _pl
        _tfps = target_fps or int(os.environ.get("DEPTHCRAFTER_TARGET_FPS", "0")) or None
        self.target_fps: int | None = _tfps

        # Verify paths
        self._validate_paths()

    # ------------------------------------------------------------------
    # Path validation
    # ------------------------------------------------------------------
    def _validate_paths(self) -> None:
        """Check critical paths exist.  Does NOT require model weights —
        they can be downloaded by the user later (or by the bootstrap)."""
        issues: list[str] = []

        repo = Path(self.repo_dir)
        if not repo.is_dir():
            issues.append(
                f"DepthCrafter repository not found at: {self.repo_dir}\n"
                f"  Run the one-command bootstrap to deploy it in-repo:\n"
                f"    python scripts/setup_depthcrafter.py\n"
                f"  See docs/DEPTHCRAFTER_SETUP.md for details."
            )

        # Look for the fire-style CLI entry point (run.py).
        if repo.is_dir() and not (repo / "run.py").is_file():
            issues.append(
                f"run.py not found at {repo / 'run.py'}\n"
                f"  The repo directory exists but may be incomplete.  Re-run the bootstrap:\n"
                f"    python scripts/setup_depthcrafter.py"
            )

        if issues:
            raise RuntimeError("DepthCrafter setup is incomplete:\n" + "\n".join(f"  • {i}" for i in issues))

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

        # The subprocess runs with cwd=repo_dir, so relative caller paths would
        # resolve against the DepthCrafter checkout, not the repo (input then
        # "not found"; output would land inside repo_dir). Absolutize both.
        input_path = str(Path(input_path).resolve())
        output_dir = str(out_path.resolve())

        run_script = str(Path(self.repo_dir) / "run.py")

        # Build command list (NO shell=True for security).
        # run.py is fire-style: positional video_path, --save_folder, --max_res,
        # --cpu_offload model, plus optional --process_length / --target_fps.
        cmd: list[str] = [
            self.python_exe,
            run_script,
            input_path,
            "--save_folder",
            output_dir,
            "--max_res",
            str(self.max_resolution),
            "--cpu_offload",
            "model",
        ]
        if self.process_length is not None:
            cmd.extend(["--process_length", str(self.process_length)])
        if self.target_fps is not None:
            cmd.extend(["--target_fps", str(self.target_fps)])

        _run_subprocess_captured(
            cmd,
            cwd=self.repo_dir,
            timeout=7200,  # 2 hours max
            label="DepthCrafter CLIBackend",
            output_dir=output_dir,
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
                f"  Contents of the output dir:\n{_dir_listing(output_dir)}\n"
                f"  The CLI exited 0 — re-run with logging at DEBUG to see its captured output tail.\n"
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
            "DepthCrafterEstimator: %s → %s",
            input_path,
            resolved_output,
        )

        return self.backend.estimate_video(
            input_path=input_path,
            output_dir=resolved_output,
        )
