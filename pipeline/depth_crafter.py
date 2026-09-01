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
# Issue #182 — depth-product cache (content-keyed, reproducible)
# ---------------------------------------------------------------------------

#: Size of the head/tail slices used in the video content fingerprint (4 MB).
#: Reading only the head + tail (not the full file) is the deliberate design:
#: a depth model's output depends on the *content*, and for long VR180 /
#: domemaster clips the full-file sha256 would take minutes to compute and
#: dominate the warm-up cost, defeating the cache's whole point.  The head
#: + tail hash is a cheap, stable proxy: two bytes-identical clips collide,
#: and in practice a content change anywhere near the boundaries trips the
#: fingerprint.  File size is also hashed, so a truncated copy of the same
#: content cannot collide with the full clip.
_DEPTH_CACHE_SLICE_BYTES = 4 * 1024 * 1024


# The repo root is two parents up from this file (pipeline/depth_crafter.py).
# Defined here so the cache default path can reference it; the in-repo
# default paths below (INREPO_REPO_DIR etc.) keep their own derivation for
# clarity and test-monkeypatchability.
_REPO_ROOT = Path(__file__).resolve().parent.parent


_DEFAULT_CACHE_ROOT = _REPO_ROOT / "models" / ".cache" / "depth"


def _video_content_fingerprint(input_path: str) -> tuple[str, int]:
    """Return ``(sha256_hex, file_size_bytes)`` of *input_path*'s content.

    The digest is computed over the **file size** and the **first and last
    4 MB** of the file — never the full file.  See the module-level note on
    ``_DEPTH_CACHE_SLICE_BYTES`` for why the full file is skipped.

    Raises :class:`OSError` if the file cannot be opened for reading (e.g.
    the path is held under an incompatible share lock on Windows), so the
    caller can treat it as a cache miss and let inference proceed.

    The size is returned alongside the hex digest so the tuple is human-
    inspectable in logs without re-deriving the length from the digest.
    """
    h = hashlib.sha256()
    size = os.path.getsize(input_path)
    h.update(str(size).encode("utf-8"))
    with open(input_path, "rb") as f:
        head = f.read(_DEPTH_CACHE_SLICE_BYTES)
        h.update(head)
        # Read the tail only when the file is larger than one slice; for
        # tiny files the head already covered the entire content, so
        # re-reading it would double-hash the bytes and break reproducibility
        # of a full-file digest for small inputs.
        if size > _DEPTH_CACHE_SLICE_BYTES:
            tail_size = min(size, _DEPTH_CACHE_SLICE_BYTES)
            f.seek(max(0, size - tail_size))
            h.update(f.read(tail_size))
    return h.hexdigest(), int(size)


def _cache_params(backend: DepthCrafterBackend) -> dict:
    """Extract the key params from *backend* that affect depth output.

    ``max_resolution`` (short-side resolution cap), ``process_length``
    (temporal window), and ``target_fps`` (frame-rate remapping) all change
    what the model produces.  Path-like fields (repo_dir, python_exe,
    model_dir) are deliberately excluded: they describe *where* the code
    runs, not *what* it computes.
    """
    params: dict = {"max_res": backend.max_resolution}
    if backend.process_length is not None:
        params["process_length"] = backend.process_length
    if backend.target_fps is not None:
        params["target_fps"] = backend.target_fps
    return params


def compute_cache_key(
    input_path: str,
    backend: DepthCrafterBackend,
) -> str:
    """Compute the depth cache key for this input + backend config.

    The key is ``sha256(fingerprint_hex || model_name || params_json)``, so
    the same clip hashed against different resolution / window / fps settings
    yields distinct cache entries.  File names and paths never participate —
    the same content at a different path must hit the same cache (lead
    decision, issue #182).
    """
    fingerprint_hex, _ = _video_content_fingerprint(input_path)
    params = _cache_params(backend)
    params_bytes = json.dumps(params, sort_keys=True, separators=(",", ":")).encode("utf-8")
    model_name = "depthcrafter"
    digest = hashlib.sha256()
    digest.update(fingerprint_hex.encode("utf-8"))
    digest.update(model_name.encode("utf-8"))
    digest.update(params_bytes)
    return digest.hexdigest()


def _default_cache_dir(cache_dir: Path | None) -> Path:
    """Resolve ``cache_dir`` to the repo-local depth cache root."""
    if cache_dir is not None:
        return cache_dir
    return _DEFAULT_CACHE_ROOT


def _cache_dir_for_key(cache_root: Path, key: str) -> Path:
    return cache_root / key


def _write_depth_meta(depth_dir: Path, *, num_frames: int, params: dict) -> None:
    """Write ``meta.json`` into *depth_dir* using the I-6 / #121 structure.

    Reuses the field names and semantics established in ``scripts.run_pipeline``
    so depth-stability provenance reporting and resume-safety validation
    keep working across the cache layer without a parallel meta schema.
    """
    meta = {
        "depth_model": "depthcrafter",
        "num_frames": int(num_frames),
        "max_res": params.get("max_res"),
        "process_length": params.get("process_length"),
        "target_fps": params.get("target_fps"),
        "temporal_smoothing": 0.0,
        "model_size": None,
    }
    import datetime

    meta["timestamp"] = datetime.datetime.now().isoformat(timespec="seconds")
    meta_path = depth_dir / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def _copy_dir_contents(src: Path, dst: Path) -> None:
    """Copy every entry from *src* into *dst* (flat, no nesting)."""
    dst.mkdir(parents=True, exist_ok=True)
    for entry in src.iterdir():
        target = dst / entry.name
        if entry.is_file() or entry.is_symlink():
            shutil.copy2(entry, target)
        else:
            shutil.copytree(entry, target, dirs_exist_ok=True)


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
# DepthCrafterEstimator (front-facing class)
# ---------------------------------------------------------------------------


class DepthCrafterEstimator:
    """Wrapper for DepthCrafter temporally-consistent video depth estimation.

    CUDA-only.  Processes an entire video clip at once to produce
    temporally smooth depth maps.

    The *backend* argument allows injecting a different backend (e.g.
    for testing).  Defaults to :class:`CLIBackend`.

    Issue #182 cache control:

    - ``use_cache`` (``True`` by default) enables the content-keyed depth
      product cache.  Passing ``False`` forces a fresh inference even when
      a matching cache entry exists.
    - ``cache_dir`` selects the cache root (``models/.cache/depth/``
      relative to the repo when ``None``).  Each input+params pair maps to
      a ``<sha256>/`` subdirectory whose contents are copied into the
      requested output dir on a hit.
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
        self.cache_dir = _default_cache_dir(cache_dir)

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
        """
        import tempfile

        resolved_output = Path(output_dir or tempfile.mkdtemp(prefix="depthcrafter_"))

        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"Input video not found: {input_path}")

        log.info(
            "DepthCrafterEstimator: %s → %s",
            input_path,
            resolved_output,
        )

        # Issue #182: cache is resolved against the backend's params, not the
        # front-facing output_dir.  Compute the key up front so the hit path
        # can short-circuit before any subprocess is spawned.  The key
        # depends on being able to read the input file; if that fails (e.g.
        # the path is held under an incompatible share lock on Windows, or
        # the file vanished between the existence check and now), treat it as
        # a cache miss and let inference proceed — the cache is an
        # optimization, never a correctness gate.
        key = None
        if self.use_cache:
            try:
                key = compute_cache_key(input_path, self.backend)
            except OSError:
                key = None

        cached_dir: Path | None = None
        if key is not None:
            cached_dir = _cache_dir_for_key(self.cache_dir, key)
            if _cache_entry_valid(cached_dir):
                log.info("[cache] hit %s", key[:8])
                _copy_dir_contents(cached_dir, resolved_output)
                # Copy the cache's meta.json into the run's output dir too,
                # so downstream meta readers (run_pipeline, depth_stability)
                # see the provenance on the resolved output.
                src_meta = cached_dir / "meta.json"
                if src_meta.is_file():
                    shutil.copy2(src_meta, resolved_output / "meta.json")
                return load_depth_maps_from_dir(str(resolved_output))

        depths = self.backend.estimate_video(
            input_path=input_path,
            output_dir=str(resolved_output),
            target_size=target_size,
        )

        # Issue #182: on a miss, persist the products into the cache so the
        # next identical request is a hit.  This runs *after* the backend has
        # confirmed the products are loadable (``depths`` is non-empty), so
        # we never cache a partial / failed run.
        if self.use_cache and key is not None and cached_dir is not None and depths:
            _persist_cache_hit(
                cache_dir=cached_dir,
                source_dir=resolved_output,
                depths=depths,
                params=_cache_params(self.backend),
            )
            # Also drop meta.json into the run's output dir so downstream
            # consumers (stereo stage, depth_stability report) can read
            # provenance without needing to know about the cache layer.
            _write_depth_meta(
                resolved_output,
                num_frames=len(depths),
                params=_cache_params(self.backend),
            )

        return depths


def _cache_entry_valid(cached_dir: Path) -> bool:
    """True if *cached_dir* holds both a meta.json and at least one depth product."""
    if not cached_dir.is_dir():
        return False
    if not (cached_dir / "meta.json").is_file():
        return False
    # Presence of any mp4 / npy / png indicates a real product, not just an
    # empty dir left by a crashed run.  ``list(...)`` is intentional: a
    # glob iterator is truthy regardless of its contents, so ``any()`` on a
    # bare generator would always short-circuit on the first (empty) glob.
    return any(
        list(cached_dir.glob("*_depth.mp4"))
        or list(cached_dir.glob("depth_*.npy"))
        or list(cached_dir.glob("*.npy"))
        or list(cached_dir.glob("depth_*.png"))
        or list(cached_dir.glob("*.png"))
    )


def _persist_cache_hit(
    *,
    cache_dir: Path,
    source_dir: Path,
    depths: list[np.ndarray],
    params: dict,
) -> None:
    """Copy the just-produced depth products into the cache and write meta.json."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Wipe any stale cache contents from a previous interrupted run so a
    # partial product cannot silently satisfy a future hit check.
    for entry in sorted(cache_dir.iterdir()):
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry)
        else:
            entry.unlink()
    _copy_dir_contents(source_dir, cache_dir)
    _write_depth_meta(
        cache_dir,
        num_frames=len(depths),
        params=params,
    )
    log.info("[cache] miss persisted %d frames under %s", len(depths), cache_dir)
