"""StereoCrafter — depth-aware stereo video generation with disocclusion inpainting.

Provides a StereoCrafterRenderer that delegates to TencentARC/StereoCrafter
inference scripts via a pluggable backend (default: CLIBackend).
CUDA-only; raises clear errors on CPU/Mac builds.

StereoCrafter (TencentARC) uses depth-guided forward splatting + video diffusion
inpainting to produce clean stereoscopic left/right views without the
ghosting/smear artifacts of simple depth-based shifting.

The upstream repo exposes **two** fire-style entry scripts (no ``run.py``):

* ``depth_splatting_inference.py`` (Stage 1) — runs DepthCrafter internally to
  estimate per-frame depth, then forward-splats the left view to produce a
  *splatting* video whose right half holds the disocclusion mask.
* ``inpainting_inference.py`` (Stage 2 — the disocclusion-inpainting step this
  repo needs) — takes the Stage-1 splatting video and video-diffusion-inpaints
  the disocclusion regions, writing a side-by-side (SBS) stereoscopic video
  (left = splatted view, right = inpainted view).

The canonical invocation order is documented in the repo's ``run_inference.sh``
(Stage 1 → Stage 2).  :class:`CLIBackend` reproduces that two-stage flow and
then splits the resulting SBS video into the separate left/right files the
pipeline expects.

Usage::

    from pipeline.stereo_crafter import StereoCrafterRenderer

    renderer = StereoCrafterRenderer()
    left_video, right_video = renderer.render_video("input.mp4", depth_maps)

Reference:
    https://github.com/TencentARC/StereoCrafter
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path

log = logging.getLogger(__name__)

# How many tail lines of captured subprocess output to log on success/failure.
# Issue #127: the CLI's own output is the only clue to the real failure cause,
# so keep enough tail lines to show it — but not the full multi-hour log.
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
    """One-line-per-entry listing of *dir_path* for failure diagnostics."""
    try:
        entries = sorted(os.listdir(dir_path))
    except OSError as exc:
        return f"    (cannot list directory: {exc})"
    if not entries:
        return "    (empty)"
    return "\n".join(f"    {e}" for e in entries[:50]) + ("\n    ..." if len(entries) > 50 else "")


# ---------------------------------------------------------------------------
# In-repo default paths (managed by scripts/setup_stereocrafter.py)
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent

# StereoCrafter checkout lives under third_party/ (gitignored).
INREPO_REPO_DIR = _REPO_ROOT / "third_party" / "StereoCrafter"
# Dedicated venv python created inside the checkout (never the project venv).
INREPO_PYTHON_EXE = (
    INREPO_REPO_DIR / ".venv" / (Path("Scripts") / "python.exe" if os.name == "nt" else Path("bin") / "python")
)
# Model weights under models/StereoCrafter (models/ is gitignored).
INREPO_CKPT_DIR = _REPO_ROOT / "models" / "StereoCrafter"

# 12 GB VRAM (RTX 4070 SUPER) safe defaults.  512 short-side is the sweet spot;
# bump to 768/1024 on larger GPUs via --stereocrafter-max-res or the env var.
DEFAULT_MAX_RESOLUTION = 512

# Upstream TencentARC/StereoCrafter has NO ``run.py``.  Its root-level fire-style
# entry scripts (verified against the actual checkout, 2026-09-01) are:
#
#   depth_splatting_inference.py  — Stage 1: depth-guided forward splatting
#   inpainting_inference.py      — Stage 2: disocclusion inpainting (this repo's step)
#
# Both ``from fire import Fire`` and accept ``--help``.  Listed in call order;
# Stage 2 (the disocclusion-inpainting step the pipeline needs) is the primary
# entry point — but it consumes Stage 1's splatting video as its input, so the
# backend drives both in sequence.
INFERENCE_SCRIPT_STAGE1 = "depth_splatting_inference.py"
INFERENCE_SCRIPT_STAGE2 = "inpainting_inference.py"
# Candidate entry scripts the backend will look for, in priority order.  Only the
# two real upstream names are recognized; legacy guesses (``run.py``,
# ``inference.py``, ``scripts/inference.py``) are deliberately absent so a
# mislabeled checkout is not silently accepted.
INFERENCE_SCRIPT_CANDIDATES = [
    INFERENCE_SCRIPT_STAGE2,  # Stage 2 (inpainting) — primary entry this repo drives
    INFERENCE_SCRIPT_STAGE1,  # Stage 1 (splatting)  — upstream stage, run first
]


def _inrepo_env_hint() -> str:
    """Text appended to errors when no StereoCrafter paths were configured/found."""
    return (
        "No StereoCrafter repo/python/checkpoint paths were configured or found in-repo.\n"
        "  Set --stereocrafter-repo-dir / STEREOCRAFTER_REPO_DIR to your checkout, or\n"
        "  run the one-command bootstrap to deploy StereoCrafter inside the repo:\n"
        "    python scripts/setup_stereocrafter.py\n"
        "  See docs/STEREOCRAFTER_SETUP.md for disk/VRAM requirements and troubleshooting."
    )


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
    environment variables, with in-repo defaults (set by
    :mod:`scripts/setup_stereocrafter.py`) adopted only when the paths
    actually exist on disk:

    ========================== =============================== =============================================
    Constructor param          Env var                         Default (if env unset)
    ========================== =============================== =============================================
    ``repo_dir``               ``STEREOCRAFTER_REPO_DIR``      in-repo ``third_party/StereoCrafter`` *(if exists)*
    ``python_exe``             ``STEREOCRAFTER_PYTHON``        in-repo venv python *(if exists)*, else ``python``
    ``checkpoint_dir``         ``STEREOCRAFTER_CKPT_DIR``      in-repo ``models/StereoCrafter`` *(if exists)*, else ``(repo_dir)/checkpoints``
    ``pre_trained_path``       ``STEREOCRAFTER_SVD_PATH``      ``(repo_dir)/weights/stable-video-diffusion-img2vid-xt-1-1``
    ``depthcrafter_unet_path`` ``STEREOCRAFTER_DC_UNET_PATH``  ``(repo_dir)/weights/DepthCrafter``
    ``max_resolution``         ``STEREOCRAFTER_MAX_RES``       ``512`` (12 GB VRAM safe)
    ========================== =============================== =============================================

    ``checkpoint_dir`` is the StereoCrafter UNet dir (Stage 2 ``--unet_path``);
    ``depthcrafter_unet_path`` is the DepthCrafter UNet dir (Stage 1
    ``--unet_path``); ``pre_trained_path`` is the SVD base model dir used as
    ``--pre_trained_path`` by both stages.  See the upstream ``run_inference.sh``
    for the canonical two-stage call order.

    If none of repo_dir / env / in-repo resolve, the constructor raises and
    points to ``scripts/setup_stereocrafter.py``.
    """

    def __init__(
        self,
        repo_dir: str | None = None,
        python_exe: str | None = None,
        checkpoint_dir: str | None = None,
        pre_trained_path: str | None = None,
        depthcrafter_unet_path: str | None = None,
        max_resolution: int | None = None,
    ) -> None:
        # repo_dir: explicit > env > in-repo default (only if it exists on disk)
        _repo_dir = repo_dir or os.environ.get("STEREOCRAFTER_REPO_DIR")
        if not _repo_dir and INREPO_REPO_DIR.is_dir():
            _repo_dir = str(INREPO_REPO_DIR)
        if not _repo_dir:
            raise RuntimeError(_inrepo_env_hint())
        self.repo_dir: str = str(Path(_repo_dir).resolve())

        # python_exe: explicit > env > in-repo venv python (only if exists) > "python"
        if python_exe:
            self.python_exe = python_exe
        elif os.environ.get("STEREOCRAFTER_PYTHON"):
            self.python_exe = os.environ["STEREOCRAFTER_PYTHON"]
        elif INREPO_PYTHON_EXE.is_file():
            self.python_exe = str(INREPO_PYTHON_EXE)
        else:
            self.python_exe = "python"

        # checkpoint_dir (StereoCrafter UNet, Stage 2 --unet_path):
        #   explicit > env > in-repo default > (repo_dir)/checkpoints
        if checkpoint_dir:
            self.checkpoint_dir = str(Path(checkpoint_dir).resolve())
        elif os.environ.get("STEREOCRAFTER_CKPT_DIR"):
            self.checkpoint_dir = str(Path(os.environ["STEREOCRAFTER_CKPT_DIR"]).resolve())
        elif INREPO_CKPT_DIR.is_dir():
            self.checkpoint_dir = str(INREPO_CKPT_DIR)
        else:
            self.checkpoint_dir = str(Path(self.repo_dir) / "checkpoints")

        # pre_trained_path (SVD base model, --pre_trained_path for both stages):
        #   explicit > env > (repo_dir)/weights/stable-video-diffusion-img2vid-xt-1-1
        if pre_trained_path:
            self.pre_trained_path = str(Path(pre_trained_path).resolve())
        elif os.environ.get("STEREOCRAFTER_SVD_PATH"):
            self.pre_trained_path = str(Path(os.environ["STEREOCRAFTER_SVD_PATH"]).resolve())
        else:
            self.pre_trained_path = str(Path(self.repo_dir) / "weights" / "stable-video-diffusion-img2vid-xt-1-1")

        # depthcrafter_unet_path (DepthCrafter UNet, Stage 1 --unet_path):
        #   explicit > env > (repo_dir)/weights/DepthCrafter
        if depthcrafter_unet_path:
            self.depthcrafter_unet_path = str(Path(depthcrafter_unet_path).resolve())
        elif os.environ.get("STEREOCRAFTER_DC_UNET_PATH"):
            self.depthcrafter_unet_path = str(Path(os.environ["STEREOCRAFTER_DC_UNET_PATH"]).resolve())
        else:
            self.depthcrafter_unet_path = str(Path(self.repo_dir) / "weights" / "DepthCrafter")

        # max resolution for inference (short side); 512 is the 12 GB VRAM safe default.
        self.max_resolution = max_resolution or int(
            os.environ.get("STEREOCRAFTER_MAX_RES", str(DEFAULT_MAX_RESOLUTION))
        )

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
                f"  Run the one-command bootstrap to deploy it in-repo:\n"
                f"    python scripts/setup_stereocrafter.py\n"
                f"  Or clone the repo manually:\n"
                f"    git clone https://github.com/TencentARC/StereoCrafter.git\n"
                f"  See docs/STEREOCRAFTER_SETUP.md for details."
            )

        # Look for a known inference entry point.  Only the real upstream names
        # are recognized (inpainting_inference.py / depth_splatting_inference.py);
        # a stray inference.py / run.py is NOT accepted as a substitute.
        if repo.is_dir():
            candidates = [repo / name for name in INFERENCE_SCRIPT_CANDIDATES]
            found = any(c.is_file() for c in candidates)
            if not found:
                issues.append(
                    f"No known inference script found in {self.repo_dir}.\n"
                    f"  Expected one of: {', '.join(INFERENCE_SCRIPT_CANDIDATES)}\n"
                    f"  See docs/STEREOCRAFTER_SETUP.md for the required file layout."
                )

        if issues:
            raise RuntimeError("StereoCrafter setup is incomplete:\n" + "\n".join(f"  \u2022 {i}" for i in issues))

    def _find_inference_script(self, name: str | None = None) -> str:
        """Return the path to an inference entry point in the repo.

        With *name* given, resolve that specific entry (raises if absent).
        Without *name*, return the first candidate that exists on disk
        (Stage 2 / inpainting is preferred).
        """
        repo = Path(self.repo_dir)
        names = [name] if name else list(INFERENCE_SCRIPT_CANDIDATES)
        for n in names:
            candidate = repo / n
            if candidate.is_file():
                return str(candidate)
        raise RuntimeError(
            f"No known inference script found in {self.repo_dir}. "
            f"Expected one of: {', '.join(INFERENCE_SCRIPT_CANDIDATES)}. "
            f"See docs/STEREOCRAFTER_SETUP.md."
        )

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

        # StereoCrafter upstream is a *two-stage* fire-style pipeline (see the
        # repo's run_inference.sh).  The repo needs the disocclusion-inpainting
        # step (Stage 2 / inpainting_inference.py), but that script consumes
        # Stage 1's splatting video as its input, so the backend drives both:
        #
        #   Stage 1  depth_splatting_inference.py
        #     --pre_trained_path <SVD base>  --unet_path <DepthCrafter unet>
        #     --input_video_path <in>       --output_video_path <splat>
        #     [--max_disp 20.0] [--process_length -1] [--batch_size 10]
        #
        #   Stage 2  inpainting_inference.py        ← the disocclusion step
        #     --pre_trained_path <SVD base>  --unet_path <StereoCrafter unet>
        #     --input_video_path <splat>    --save_dir <dir>
        #     [--frames_chunk 23] [--overlap 3] [--tile_num 1]
        #
        # Stage 2 writes a single side-by-side video (<name>_sbs.mp4); the
        # pipeline contract wants separate left/right files, so the backend
        # splits the SBS frame into L/R afterwards.
        #
        # NOTE: StereoCrafter runs its OWN internal depth estimation in Stage 1
        # (DepthCrafterDemo) — the caller-supplied *depth_dir* is therefore not
        # forwarded to the subprocess.  It is accepted for interface
        # compatibility with the pipeline (which always passes it) and must
        # already exist as a directory; StereoCrafter simply does not consume
        # the external DepthCrafter depth maps.

        if depth_dir and not os.path.isdir(depth_dir):
            raise NotADirectoryError(
                f"Depth directory not found: {depth_dir}. Provide a valid --depth-dir or ensure depth maps are saved."
            )

        # Absolutize caller paths: the subprocess runs with cwd=repo_dir, so
        # relative paths would resolve against the StereoCrafter checkout.
        abs_input = str(Path(input_path).resolve())
        work_dir = Path(tempfile.mkdtemp(prefix="stereocrafter_work_"))
        splat_video = str((work_dir / "splatting_results.mp4").resolve())
        sbs_dir = str(work_dir.resolve())

        # --- Stage 1: depth splatting -------------------------------------
        stage1_script = self._find_inference_script(INFERENCE_SCRIPT_STAGE1)
        cmd1: list[str] = [
            self.python_exe,
            stage1_script,
            "--pre_trained_path",
            self.pre_trained_path,
            "--unet_path",
            self.depthcrafter_unet_path,
            "--input_video_path",
            abs_input,
            "--output_video_path",
            splat_video,
        ]
        self._run_subprocess(cmd1, label="Stage 1 (depth splatting)", output_dir=str(work_dir))

        # --- Stage 2: disocclusion inpainting -----------------------------
        stage2_script = self._find_inference_script(INFERENCE_SCRIPT_STAGE2)
        cmd2: list[str] = [
            self.python_exe,
            stage2_script,
            "--pre_trained_path",
            self.pre_trained_path,
            "--unet_path",
            self.checkpoint_dir,
            "--input_video_path",
            splat_video,
            "--save_dir",
            sbs_dir,
        ]
        self._run_subprocess(cmd2, label="Stage 2 (disocclusion inpainting)", output_dir=sbs_dir)

        # --- Split the SBS output into separate L/R videos ---------------
        # inpainting_inference.py writes <video_name>_sbs.mp4 in save_dir, where
        # video_name = basename(input).replace(".mp4","").replace("_splatting_results","")
        #                + "_inpainting_results".
        video_name = (
            Path(splat_video).name.replace(".mp4", "").replace("_splatting_results", "") + "_inpainting_results"
        )
        sbs_path = Path(sbs_dir) / f"{video_name}_sbs.mp4"
        if not sbs_path.is_file():
            raise RuntimeError(
                f"StereoCrafter finished but SBS output not found:\n"
                f"  {sbs_path}\n"
                f"  Contents of the output dir {sbs_dir}:\n{_dir_listing(sbs_dir)}\n"
                f"  Both stages exited 0 — re-run with logging at DEBUG to see their captured output tails.\n"
                f"  Check the inpainting_inference.py output format "
                f"in docs/STEREOCRAFTER_SETUP.md."
            )

        self._split_sbs_video(str(sbs_path), output_left, output_right)

        log.info(
            "StereoCrafter: L → %s, R → %s",
            output_left,
            output_right,
        )
        return output_left, output_right

    # ------------------------------------------------------------------
    # Subprocess helpers
    # ------------------------------------------------------------------
    def _run_subprocess(self, cmd: list[str], *, label: str, output_dir: str | None = None) -> None:
        """Run a StereoCrafter subprocess stage, raising on failure.

        The command is always a list (never ``shell=True``).  *label* names
        the stage for error messages; *output_dir* (if given) is listed in the
        failure message so the real on-disk artifacts are visible.

        stdout/stderr are captured to an OS-drained temp file (issue #127, the
        same pattern ``pipeline/streaming_pipeline.py`` uses for ffmpeg): an
        undrained ``PIPE`` fills its 64 KB buffer and deadlocks a chatty
        inference CLI, while ``DEVNULL`` hides the failure cause.  On success
        the last ~20 combined lines are logged at DEBUG (no INFO spam); on
        failure the last ~40 lines are logged at ERROR and summarized into the
        raised ``RuntimeError`` together with the command and cwd.
        """
        full_label = f"StereoCrafter CLIBackend {label}"
        log.info("%s command: %s", full_label, " ".join(cmd))
        log.info("%s cwd: %s", full_label, self.repo_dir)

        with tempfile.TemporaryFile(prefix="vr180-cli-capture-") as capture:
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=self.repo_dir,
                    stdout=capture,
                    stderr=subprocess.STDOUT,
                    timeout=7200,  # 2 hours max
                )
            except FileNotFoundError as exc:
                raise RuntimeError(
                    f"Python executable not found: {self.python_exe}. "
                    f"Set STEREOCRAFTER_PYTHON or --stereocrafter-python "
                    f"to the correct path.\n"
                    f"  Command: {' '.join(cmd)}\n"
                    f"  cwd: {self.repo_dir}"
                ) from exc
            except subprocess.TimeoutExpired:
                raise RuntimeError(
                    "StereoCrafter inference timed out after 2 hours. "
                    "The video may be too long or the GPU too slow.\n"
                    f"  Command: {' '.join(cmd)}\n"
                    f"  cwd: {self.repo_dir}"
                ) from None

            if proc.returncode == 0:
                for line in _combined_output(capture, proc, _SUCCESS_LOG_LINES):
                    log.debug("%s | %s", full_label, line)
                return

            tail = _combined_output(capture, proc, _FAILURE_LOG_LINES)
            for line in tail:
                log.error("%s | %s", full_label, line)
            summary = "\n".join(f"  {ln}" for ln in tail[-10:])
            msg = (
                f"StereoCrafter {label} failed (exit code {proc.returncode}):\n"
                f"  --- subprocess output (tail) ---\n"
                f"{summary}\n"
                f"  Command: {' '.join(cmd)}\n"
                f"  cwd: {self.repo_dir}"
            )
            if output_dir is not None:
                msg += f"\n  Output dir: {output_dir}\n  Contents:\n{_dir_listing(output_dir)}"
            msg += "\n  See docs/STEREOCRAFTER_SETUP.md for troubleshooting."
            raise RuntimeError(msg)

    def _split_sbs_video(self, sbs_path: str, output_left: str, output_right: str) -> None:
        """Split a StereoCrafter SBS video into separate left/right files.

        Uses OpenCV (already a pipeline dependency) to read each SBS frame,
        halve it vertically into left/right, and write two output videos with
        the source fps / codec.
        """
        import cv2

        cap = cv2.VideoCapture(sbs_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open StereoCrafter SBS video for splitting: {sbs_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if frame_w == 0 or frame_h == 0:
            cap.release()
            raise RuntimeError(f"Could not read SBS video dimensions from {sbs_path}")
        half_w = frame_w // 2

        Path(output_left).parent.mkdir(parents=True, exist_ok=True)
        Path(output_right).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        w_left = cv2.VideoWriter(output_left, fourcc, fps, (half_w, frame_h))
        w_right = cv2.VideoWriter(output_right, fourcc, fps, (half_w, frame_h))

        missing: list[str] = []
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                w_left.write(frame[:, :half_w])
                w_right.write(frame[:, half_w:])
        finally:
            cap.release()
            w_left.release()
            w_right.release()

        if not Path(output_left).is_file():
            missing.append(output_left)
        if not Path(output_right).is_file():
            missing.append(output_right)
        if missing:
            raise RuntimeError(
                f"StereoCrafter SBS split finished but output video(s) not found:\n"
                f"  {', '.join(missing)}\n"
                f"  Check the SBS output format in docs/STEREOCRAFTER_SETUP.md."
            )


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
        pre_trained_path: str | None = None,
        depthcrafter_unet_path: str | None = None,
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
                pre_trained_path=pre_trained_path,
                depthcrafter_unet_path=depthcrafter_unet_path,
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
