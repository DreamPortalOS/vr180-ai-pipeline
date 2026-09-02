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

import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# How many lines of subprocess output to surface on success (DEBUG) /
# failure (ERROR + exception summary) — issue #127.
_OUTPUT_TAIL_SUCCESS_LINES = 20
_OUTPUT_TAIL_FAILURE_LINES = 40


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
# Subprocess output helpers (issue #127)
# ---------------------------------------------------------------------------


def _read_tail_lines(fileobj, max_lines: int) -> str:
    """Read back the whole capture file and return its last *max_lines* lines.

    The file is binary (it received raw subprocess bytes); decode leniently.
    Returns a parenthesized placeholder when nothing was captured, so logs
    and exception messages never show a confusing blank block.
    """
    try:
        fileobj.flush()
        fileobj.seek(0)
        text = fileobj.read().decode("utf-8", errors="replace")
    except (OSError, ValueError):
        return "(output unreadable)"
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return "(no output)"
    return "\n".join(lines[-max_lines:])


def _indent(text: str, prefix: str = "    ") -> str:
    """Indent every line of *text* (for readable exception blocks)."""
    return "\n".join(prefix + ln for ln in text.splitlines())


def _dir_listing_block(output_dir: str) -> str:
    """Format the real contents of *output_dir* for an error message."""
    try:
        entries = sorted(p.name for p in Path(output_dir).iterdir())
    except OSError:
        entries = ["(directory unreadable)"]
    if not entries:
        entries = ["(empty)"]
    listing = "\n".join(f"      - {name}" for name in entries)
    return f"  Output dir contents:\n{listing}\n"


def _find_depth_mp4(out_path: Path, stem: str | None = None) -> Path | None:
    """Locate the depth video in *out_path* (exact stem match first)."""
    if stem:
        exact = out_path / f"{stem}_depth.mp4"
        if exact.is_file():
            return exact
    matches = sorted(out_path.glob("*_depth.mp4"))
    return matches[0] if matches else None


def _load_depths_from_mp4(mp4_path: Path) -> list[np.ndarray]:
    """Decode an 8-bit grayscale depth video into float32 [0, 1] frames."""
    import cv2

    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise RuntimeError(f"DepthCrafter produced {mp4_path} but it could not be opened as a video.")
    depths: list[np.ndarray] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            depths.append(gray.astype(np.float32) / 255.0)
    finally:
        cap.release()
    if not depths:
        raise RuntimeError(f"DepthCrafter produced {mp4_path} but no frames could be decoded from it.")
    return depths


def load_depth_maps_from_dir(depth_dir: str, stem: str | None = None) -> list[np.ndarray]:
    """Load per-frame depth maps from *depth_dir*, mp4 first.

    This is the single shared depth-product reader (issue #145): both the
    DepthCrafter backend (producer side) and the StereoCrafter backend
    (consumer side) go through this function, so the mp4-vs-npy/png gap of
    issue #126 cannot be fixed on one side and forgotten on the other.

    Preference order:
      1. ``<stem>_depth.mp4`` (or any ``*_depth.mp4``) — DepthCrafter's real
         upstream output: an 8-bit grayscale visualization video, decoded and
         normalized to [0, 1] float32.
      2. ``depth_*.npy`` / ``*.npy`` sequence (pipeline checkpoints / legacy).
      3. ``depth_*.png`` / ``*.png`` sequence (legacy / alternate backends,
         e.g. hand-seeded Depth-Anything visualisations).
    Raises RuntimeError listing the actual dir contents if none are found.
    """
    out_path = Path(depth_dir)
    depth_mp4 = _find_depth_mp4(out_path, stem)
    if depth_mp4 is not None:
        return _load_depths_from_mp4(depth_mp4)

    npy_files = sorted(out_path.glob("depth_*.npy")) or sorted(out_path.glob("*.npy"))
    if npy_files:
        return [np.load(str(f)) for f in npy_files]

    png_files = sorted(out_path.glob("depth_*.png")) or sorted(out_path.glob("*.png"))
    if png_files:
        import cv2

        imgs: list[np.ndarray] = []
        for f in png_files:
            img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
            if img is not None:
                imgs.append(img.astype(np.float32) / 255.0)
        if imgs:
            return imgs

    # Nothing found — list the actual contents so the next person can
    # see at a glance what the depth stage really produced (issue #133).
    contents = sorted(p.name for p in out_path.iterdir()) or ["(empty)"]
    listing = "\n".join(f"    - {name}" for name in contents)
    raise RuntimeError(
        f"No depth maps found in {depth_dir}.\n"
        f"  Looked for: *_depth.mp4, *.npy, depth_*.png / *.png (stem={stem!r}).\n"
        f"  Actual directory contents:\n{listing}"
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
        target_size: tuple[int, int] | None = None,
    ) -> list[np.ndarray]:
        """Run depth estimation on *input_path* and return depth maps.

        Returns a list of (H, W) float32 depth maps, one per frame, resized to
        the source frame size (``target_size`` if given, else probed from
        *input_path*).  The backend is responsible for saving intermediate
        results to *output_dir* as needed.
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

# Default subprocess timeout (seconds) — the historical hard-coded 2 hours.
# Override via DEPTHCRAFTER_TIMEOUT_SEC (issue #134).
_DEFAULT_TIMEOUT_SEC = 7200


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
    (subprocess timeout)       ``DEPTHCRAFTER_TIMEOUT_SEC``    ``7200`` (2 hours)
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

        # Subprocess timeout in seconds (issue #134: was a hard-coded 2 hours).
        self.timeout_sec: int = int(os.environ.get("DEPTHCRAFTER_TIMEOUT_SEC", str(_DEFAULT_TIMEOUT_SEC)))

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
        target_size: tuple[int, int] | None = None,
    ) -> list[np.ndarray]:
        _assert_cuda()

        # Ensure output dir exists and is EMPTY.  Reusing a dirty directory is
        # a real footgun: leftover .npy files from an earlier run get loaded
        # as if they were this run's product (the "fake success" reported in
        # issue #126), silently defeating A/B comparisons.
        out_path = Path(output_dir)
        self._clean_output_dir(out_path)

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

        self._run_subprocess(cmd, output_dir=output_dir)

        # Load depth maps from output dir.  The real upstream run.py emits
        # ``<stem>_depth.mp4`` (an 8-bit grayscale video of the depth maps),
        # NOT a .npy sequence — see docs/DEPTHCRAFTER_SETUP.md.  Older/alternate
        # backends may still emit .npy or depth_*.png sequences, so keep those
        # fallbacks.
        stem = Path(input_path).stem
        depths = self._load_depths(out_path, stem)

        # Issue #130: DepthCrafter down-samples the input to ``--max_res``
        # (short side) before inference, so the decoded depth maps come back
        # at the *model* resolution (e.g. 256×512 for a 720×1280 source).
        # Downstream stages (stereo render / EMA smoothing) operate at the
        # source frame size, so resize back now — mirroring the Depth-Anything
        # path in pipeline/depth_estimator.py.
        depths = self._resize_depths_to_source(depths, input_path, target_size)

        log.info("DepthCrafter: loaded %d depth maps from %s", len(depths), output_dir)
        return depths

    # ------------------------------------------------------------------
    # Subprocess invocation (issue #127: never swallow stdout/stderr)
    # ------------------------------------------------------------------
    def _run_subprocess(self, cmd: list[str], *, output_dir: str) -> None:
        """Run the DepthCrafter inference script, surfacing its output.

        stdout/stderr go to temp files (the same drained-file pattern used
        for ffmpeg in ``pipeline/streaming_pipeline.py``): an undrained PIPE
        deadlocks once its 64 KB buffer fills, and DEVNULL hides the very
        error the operator needs.  On success the last few lines are logged
        at DEBUG (no INFO-level spam); on failure the tail is logged at
        ERROR and folded into the raised exception together with the
        command, cwd, and the real contents of *output_dir*.
        """
        log.info("DepthCrafter CLIBackend command: %s", " ".join(cmd))
        log.info("DepthCrafter CLIBackend cwd: %s", self.repo_dir)

        # Temp files are OS-drained at no cost — no PIPE, no deadlock.
        # Closed below, not at this scope's exit, hence no `with`.
        out_file = tempfile.TemporaryFile(prefix="depthcrafter-stdout-")  # noqa: SIM115
        err_file = tempfile.TemporaryFile(prefix="depthcrafter-stderr-")  # noqa: SIM115
        try:
            try:
                result = subprocess.run(
                    cmd,
                    cwd=self.repo_dir,
                    stdout=out_file,
                    stderr=err_file,
                    timeout=self.timeout_sec,
                    check=False,
                )
            except FileNotFoundError as exc:
                raise RuntimeError(
                    f"Python executable not found: {self.python_exe}. "
                    f"Set DEPTHCRAFTER_PYTHON or --depthcrafter-python to the correct path."
                ) from exc
            except subprocess.TimeoutExpired:
                # Issue #134: the timeout branch must carry the same diagnostic
                # context as the non-zero-exit branch (issue #127) — command,
                # cwd, key params, the output tail produced before the kill,
                # and the real contents of the output dir.
                stdout_tail = _read_tail_lines(out_file, _OUTPUT_TAIL_FAILURE_LINES)
                stderr_tail = _read_tail_lines(err_file, _OUTPUT_TAIL_FAILURE_LINES)
                raise RuntimeError(
                    f"DepthCrafter inference timed out after {self.timeout_sec} seconds "
                    f"(configured via DEPTHCRAFTER_TIMEOUT_SEC; default {_DEFAULT_TIMEOUT_SEC}).\n"
                    f"  The video may be too long or the GPU too slow — raise the timeout or\n"
                    f"  lower the workload (e.g. DEPTHCRAFTER_MAX_RES, currently {self.max_resolution}).\n"
                    f"  Command: {' '.join(cmd)}\n"
                    f"  cwd: {self.repo_dir}\n"
                    f"  max_res: {self.max_resolution} | process_length: {self.process_length} | "
                    f"target_fps: {self.target_fps}\n"
                    f"  Output dir: {output_dir}\n"
                    f"{_dir_listing_block(output_dir)}"
                    f"  --- stdout (last {_OUTPUT_TAIL_FAILURE_LINES} lines before timeout) ---\n"
                    f"{_indent(stdout_tail)}\n"
                    f"  --- stderr (last {_OUTPUT_TAIL_FAILURE_LINES} lines before timeout) ---\n"
                    f"{_indent(stderr_tail)}\n"
                    f"  See docs/DEPTHCRAFTER_SETUP.md for troubleshooting."
                ) from None

            returncode = result.returncode
            stdout_tail = _read_tail_lines(out_file, _OUTPUT_TAIL_FAILURE_LINES)
            stderr_tail = _read_tail_lines(err_file, _OUTPUT_TAIL_FAILURE_LINES)

            if returncode != 0:
                log.error(
                    "DepthCrafter inference failed (exit code %s).\n"
                    "--- stdout (last %d lines) ---\n%s\n"
                    "--- stderr (last %d lines) ---\n%s",
                    returncode,
                    _OUTPUT_TAIL_FAILURE_LINES,
                    stdout_tail,
                    _OUTPUT_TAIL_FAILURE_LINES,
                    stderr_tail,
                )
                raise RuntimeError(
                    f"DepthCrafter inference failed (exit code {returncode}).\n"
                    f"  Command: {' '.join(cmd)}\n"
                    f"  cwd: {self.repo_dir}\n"
                    f"  Output dir: {output_dir}\n"
                    f"{_dir_listing_block(output_dir)}"
                    f"  --- stdout (last {_OUTPUT_TAIL_FAILURE_LINES} lines) ---\n"
                    f"{_indent(stdout_tail)}\n"
                    f"  --- stderr (last {_OUTPUT_TAIL_FAILURE_LINES} lines) ---\n"
                    f"{_indent(stderr_tail)}\n"
                    f"  See docs/DEPTHCRAFTER_SETUP.md for troubleshooting."
                )

            # Success: tail at DEBUG so the default INFO level stays quiet.
            log.debug(
                "DepthCrafter subprocess finished.\n"
                "--- stdout (last %d lines) ---\n%s\n"
                "--- stderr (last %d lines) ---\n%s",
                _OUTPUT_TAIL_SUCCESS_LINES,
                _read_tail_lines(out_file, _OUTPUT_TAIL_SUCCESS_LINES),
                _OUTPUT_TAIL_SUCCESS_LINES,
                _read_tail_lines(err_file, _OUTPUT_TAIL_SUCCESS_LINES),
            )
        finally:
            out_file.close()
            err_file.close()

    # ------------------------------------------------------------------
    # Output-dir hygiene
    # ------------------------------------------------------------------
    @staticmethod
    def _clean_output_dir(out_path: Path) -> None:
        """Create *out_path* if missing, then remove any pre-existing contents.

        Only depth artifacts are ever written here by this backend, and the
        dir is caller-provided (typically a temp dir), so a full wipe is the
        safe way to guarantee the loaded frames come from THIS run.
        """
        out_path.mkdir(parents=True, exist_ok=True)
        for entry in sorted(out_path.iterdir()):
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry)
            else:
                entry.unlink()

    # ------------------------------------------------------------------
    # Output loading
    # ------------------------------------------------------------------
    def _load_depths(self, out_path: Path, stem: str) -> list[np.ndarray]:
        """Load per-frame depth maps from *out_path*, mp4 first.

        Delegates to the shared :func:`load_depth_maps_from_dir` (issue #145:
        the same reader is used by the StereoCrafter consumer side, so the
        mp4-vs-npy/png gap of issue #126 cannot be fixed twice).
        """
        try:
            return load_depth_maps_from_dir(str(out_path), stem=stem)
        except RuntimeError as exc:
            raise RuntimeError(
                f"DepthCrafter finished but {exc}\n"
                f"  Check the inference script output format in docs/DEPTHCRAFTER_SETUP.md."
            ) from None

    # ------------------------------------------------------------------
    # Resize back to source frame size (issue #130)
    # ------------------------------------------------------------------
    def _resize_depths_to_source(
        self,
        depths: list[np.ndarray],
        input_path: str,
        target_size: tuple[int, int] | None,
    ) -> list[np.ndarray]:
        """Resize each depth map to the source frame size ``(h, w)``.

        ``target_size`` (caller-provided) wins; when absent the size is probed
        from *input_path* with cv2.  If both are available but disagree, the
        caller's value is used and one line is logged.
        """
        if not depths:
            return depths

        probed = self._probe_video_size(input_path)
        if target_size is not None:
            if probed is not None and probed != target_size:
                log.warning(
                    "DepthCrafter: target_size %s disagrees with probed source size %s — using target_size",
                    target_size,
                    probed,
                )
            h, w = target_size
        elif probed is not None:
            h, w = probed
        else:
            log.warning(
                "DepthCrafter: no target_size given and source size could not be probed "
                "from %s — returning model-resolution depths (%s)",
                input_path,
                depths[0].shape,
            )
            return depths

        import cv2

        resized: list[np.ndarray] = []
        for d in depths:
            if d.shape[:2] == (h, w):
                resized.append(d)
            else:
                # Depth is a continuous quantity — INTER_LINEAR, not NEAREST.
                resized.append(cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR))
        if resized and resized[0].shape[:2] != depths[0].shape[:2]:
            log.info(
                "DepthCrafter: resized %d depth maps from %s to (%d, %d) (source frame size)",
                len(resized),
                depths[0].shape[:2],
                h,
                w,
            )
        return resized

    @staticmethod
    def _probe_video_size(input_path: str) -> tuple[int, int] | None:
        """Return ``(h, w)`` of *input_path* via cv2, or None if unprobeable."""
        try:
            import cv2
        except ImportError:
            return None
        try:
            cap = cv2.VideoCapture(str(input_path))
            if not cap.isOpened():
                return None
            try:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            finally:
                cap.release()
        except Exception:  # pragma: no cover - defensive, never mask inference result
            return None
        if h <= 0 or w <= 0:
            return None
        return (h, w)


# ---------------------------------------------------------------------------
# Depth-product content cache (I-8a, issue #182)
# ---------------------------------------------------------------------------
#
# Cache key = sha256( content_fingerprint || "depthcrafter" || canonical_json(params) )
#
# ``content_fingerprint`` is sha256 over ``str(file_size)`` + the first 4 MB of
# the file + the last 4 MB of the file — NOT the whole file.  A full-file hash
# on a long VR180 clip (minutes of video, hundreds of MB) would itself take
# minutes, defeating the point of a cache whose lookup must be cheap enough to
# do *before* launching the multi-gigabyte subprocess.  Head + tail is a cheap
# stable content proxy: two genuinely different videos are astronomically
# unlikely to share both their first and last 4 MB *and* their exact size, so a
# truncated/extended copy of the same content (different size) cannot collide.
# For files ≤ 4 MB the head read already covers the whole file, so the tail
# read is skipped (reading it would double-hash the same bytes and break the
# stable-hash contract for small inputs).
#
# The key is **content-only, path/name-agnostic** (lead decision): the same
# bytes at a different filesystem path MUST hit.  Path, name and mtime are
# deliberately excluded — copying the source to a new location (or re-rendering
# the same bytes) must reuse the cached depth product.

_DEPTH_CACHE_NAME = "depth"
_DEPTH_CACHE_META = "meta.json"  # reuse the #121 schema, do not invent another
_FINGERPRINT_CHUNK = 4 * 1024 * 1024  # 4 MB head/tail sample


def _fingerprint_file(path: str) -> str | None:
    """sha256 over ``file_size + first 4MB + last 4MB`` of *path*.

    Returns ``None`` when the file cannot be read (locked, unreadable, gone) so
    the caller can treat the cache as a miss and fall through to inference —
    the cache is an *optimisation*, never a correctness gate.  On Windows a
    ``NamedTemporaryFile(delete=True)`` holds an exclusive share lock, so
    ``open(..., "rb")`` raises ``PermissionError``; that is caught here.
    """
    h = hashlib.sha256()
    try:
        size = os.path.getsize(path)
        h.update(str(size).encode("utf-8"))
        with open(path, "rb") as f:
            head = f.read(_FINGERPRINT_CHUNK)
            h.update(head)
            # Only read the tail when the head didn't already cover the whole
            # file; otherwise this would double-hash the same bytes (breaking
            # the stable-hash contract for ≤4 MB inputs).
            if size > _FINGERPRINT_CHUNK:
                f.seek(max(0, size - _FINGERPRINT_CHUNK))
                h.update(f.read(_FINGERPRINT_CHUNK))
    except OSError as exc:
        log.debug("[cache] could not fingerprint %s (%s) — treating as miss", path, exc)
        return None
    return h.hexdigest()


def _backend_cache_params(backend: object) -> dict[str, object]:
    """The output-affecting params pulled off *backend* for the cache key.

    All reads are safe ``getattr(..., default)`` so a pluggable backend that
    does not expose a given knob (e.g. the test ``MockBackend`` has no
    ``max_resolution``) contributes a fixed placeholder, not an exception.  A
    missing knob is recorded as ``None`` — two backends that both omit it then
    hash the same, which is correct (neither can change the output via a knob
    it doesn't have).  Issue #182 round-1 rejection: ``backend.max_resolution``
    was read unconditionally and broke every existing mock backend that lacked
    that attribute; this function makes every backend-derived value optional.
    """
    return {
        "max_res": getattr(backend, "max_resolution", None),
        "process_length": getattr(backend, "process_length", None),
        "target_fps": getattr(backend, "target_fps", None),
    }


def _compute_cache_key(
    input_path: str,
    backend: object,
    *,
    model_name: str = "depthcrafter",
) -> str | None:
    """Content-keyed cache key for a depth run, or ``None`` if unhashable.

    ``None`` (file unreadable) means "cannot key → behave as a cache miss".
    """
    fingerprint = _fingerprint_file(input_path)
    if fingerprint is None:
        return None
    params = _backend_cache_params(backend)
    canonical = json.dumps(params, sort_keys=True, separators=(",", ":"))
    key_input = f"{fingerprint}|{model_name}|{canonical}"
    return hashlib.sha256(key_input.encode("utf-8")).hexdigest()


def _cache_dir_for(key: str, cache_dir: Path | None) -> Path:
    """Resolve the on-disk cache entry dir for *key*."""
    base = cache_dir if cache_dir is not None else (_REPO_ROOT / "models" / ".cache" / _DEPTH_CACHE_NAME)
    return Path(base) / key


def _cache_entry_valid(cached_dir: Path, meta: dict) -> bool:
    """True if *cached_dir* holds a usable product set matching *meta*.

    A valid entry has at least one ``depth_*.npy`` map and its ``meta.json``
    frame count matches the number of ``.npy`` files on disk — a half-written
    / tampered entry (wrong frame count, no maps) is a miss, not a silent
    reuse of stale or partial product.
    """
    npy_files = list(cached_dir.glob("depth_*.npy"))
    if not npy_files:
        return False
    meta_frames = meta.get("num_frames")
    if isinstance(meta_frames, int):
        return meta_frames == len(npy_files)
    return True


def _load_cached_depths(cached_dir: Path) -> list[np.ndarray]:
    """Load the cached ``depth_*.npy`` sequence (sorted, oldest-index-first)."""
    npy_files = sorted(cached_dir.glob("depth_*.npy"))
    return [np.load(str(f)) for f in npy_files]


def _persist_depths_to_cache(
    depths: list[np.ndarray],
    cached_dir: Path,
    *,
    backend: object,
    model_name: str,
) -> None:
    """Write *depths* as ``depth_*.npy`` + ``meta.json`` into *cached_dir*.

    ``meta.json`` reuses the I-6 (#121) schema (depth_model / num_frames /
    max_res / process_length / target_fps / model_size / timestamp) — do not
    invent a parallel schema.  Backend-derived fields use the same safe
    ``getattr`` defaults as the key so a backend without a knob records
    ``None`` consistently on both sides.
    """
    cached_dir.mkdir(parents=True, exist_ok=True)
    # Wipe any stale partial entry first so a half-written previous run cannot
    # bleed into this one (mirrors CLIBackend._clean_output_dir discipline).
    for entry in list(cached_dir.iterdir()):
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry)
        else:
            entry.unlink()
    for i, depth in enumerate(depths):
        np.save(str(cached_dir / f"depth_{i:06d}.npy"), depth)
    meta: dict[str, object] = {
        "depth_model": model_name,
        "num_frames": len(depths),
        "model_size": getattr(backend, "model_size", None),
        "max_res": getattr(backend, "max_resolution", None),
        "process_length": getattr(backend, "process_length", None),
        "target_fps": getattr(backend, "target_fps", None),
        "temporal_smoothing": getattr(backend, "temporal_smoothing", 0.0) or 0.0,
    }
    # Use a fixed timezone-naive timestamp from os.time-likes rather than
    # datetime.now() so the entry is deterministic per-run; meta is for
    # provenance comparison, not a cache key, so monotonicity is enough.
    try:
        import time

        meta["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
    except Exception:  # pragma: no cover - defensive, never block caching
        meta["timestamp"] = "unknown"
    meta_path = cached_dir / _DEPTH_CACHE_META
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    log.info("[cache] stored %d depth maps → %s", len(depths), cached_dir)


# ---------------------------------------------------------------------------
# DepthCrafterEstimator (front-facing class)
# ---------------------------------------------------------------------------


class DepthCrafterEstimator:
    """Wrapper for DepthCrafter temporally-consistent video depth estimation.

    CUDA-only.  Processes an entire video clip at once to produce
    temporally smooth depth maps.

    The *backend* argument allows injecting a different backend (e.g.
    for testing).  Defaults to :class:`CLIBackend`.

    Cache (I-8a, issue #182): depth products are cached by a content-keyed
    fingerprint (file size + first/last 4 MB sha256) together with the model
    name and output-affecting params, so a second run of the same input + same
    params reuses the cached depth maps and **never starts the subprocess**.
    Disable with ``use_cache=False``; redirect with ``cache_dir``.
    """

    def __init__(
        self,
        backend: DepthCrafterBackend | None = None,
        repo_dir: str | None = None,
        python_exe: str | None = None,
        checkpoint_dir: str | None = None,
        max_resolution: int | None = None,
        use_cache: bool = True,
        cache_dir: Path | None = None,
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
        self.use_cache = use_cache
        # cache_dir=None → default models/.cache/depth (resolved lazily so the
        # estimator is constructible without write access to that dir).
        self.cache_dir = cache_dir

    def estimate_video(
        self,
        input_path: str,
        output_dir: str | None = None,
        target_size: tuple[int, int] | None = None,
    ) -> list[np.ndarray]:
        """Estimate temporally-consistent depth for an entire video.

        Args:
            input_path: Path to input video file.
            output_dir: Directory to save intermediate depth outputs.
                If None, a temporary directory is created.
            target_size: Optional ``(h, w)`` of the source frames.  Depth maps
                are resized to this before being returned (issue #130); when
                None the backend probes the input video for its size.

        Returns:
            List of (H, W) float32 depth maps, one per frame, at the source
            frame size, normalized to [0, 1] (same convention as the
            Depth-Anything backend in ``pipeline/depth_estimator.py``).

        Cache (issue #182): before launching the backend, the content-keyed
        cache is checked.  A hit returns the cached ``depth_*.npy`` maps and
        the backend (subprocess) is never invoked.  A miss runs the backend,
        then persists the returned maps as ``depth_*.npy`` + ``meta.json``
        (reusing the #121 schema) into the cache dir for next time.
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

        # --- Cache lookup (I-8a, issue #182) ---------------------------------
        # The cache is an optimisation, never a correctness gate: any failure
        # to key / read / write it is swallowed and we fall through to a full
        # backend run.  ``use_cache=False`` skips the lookup entirely so the
        # caller can force a recompute (e.g. a "refresh depth" flag).
        if self.use_cache:
            key = _compute_cache_key(input_path, self.backend)
            if key is not None:
                cached_dir = _cache_dir_for(key, self.cache_dir)
                meta_path = cached_dir / _DEPTH_CACHE_META
                if meta_path.is_file():
                    try:
                        with open(meta_path, encoding="utf-8") as f:
                            meta = json.load(f)
                    except (OSError, json.JSONDecodeError) as exc:
                        log.debug("[cache] meta unreadable at %s (%s) — miss", cached_dir, exc)
                        meta = {}
                    if _cache_entry_valid(cached_dir, meta):
                        log.info("[cache] hit %s", key[:8])
                        return _load_cached_depths(cached_dir)
                    log.info("[cache] miss (stale/partial) %s — recomputing", key[:8])
                else:
                    log.info("[cache] miss %s — recomputing", key[:8])

        # --- Full inference --------------------------------------------------
        depths = self.backend.estimate_video(
            input_path=input_path,
            output_dir=resolved_output,
            target_size=target_size,
        )

        # --- Cache persist ---------------------------------------------------
        # Only persist when we have a key AND real depth maps.  A ``None`` key
        # (file unreadable / lock held) means we cannot reliably key the entry,
        # so skip the write — the next run will simply miss again, never
        # silently reuse a product stored under a key derived from partial data.
        if self.use_cache and depths:
            key = _compute_cache_key(input_path, self.backend)
            if key is not None:
                cached_dir = _cache_dir_for(key, self.cache_dir)
                try:
                    _persist_depths_to_cache(
                        depths,
                        cached_dir,
                        backend=self.backend,
                        model_name="depthcrafter",
                    )
                except OSError as exc:
                    # Cache write failure must never fail the run — the depths
                    # are already in hand; the caller gets them back regardless.
                    log.warning("[cache] could not persist entry at %s (%s)", cached_dir, exc)

        return depths
