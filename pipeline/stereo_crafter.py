"""StereoCrafter — depth-aware stereo video generation with disocclusion inpainting.

Provides a StereoCrafterRenderer that delegates to TencentARC/StereoCrafter's
``inpainting_inference.py`` via a pluggable backend (default: CLIBackend).
CUDA-only; raises clear errors on CPU/Mac builds.

StereoCrafter (TencentARC) uses depth-guided forward splatting + video diffusion
inpainting to produce clean stereoscopic left/right views without the
ghosting/smear artifacts of simple depth-based shifting.

The upstream repo exposes **two** fire-style entry scripts (no ``run.py``):

* ``depth_splatting_inference.py`` (Stage 1) — runs an **embedded** copy of
  DepthCrafter under ``dependency/DepthCrafter/`` to estimate per-frame depth,
  then forward-splats the left view to produce the *splatting* grid video.
* ``inpainting_inference.py`` (Stage 2 — the disocclusion-inpainting step this
  repo needs) — takes the Stage-1 grid video and video-diffusion-inpaints the
  disocclusion regions, writing a side-by-side (SBS) stereoscopic video
  (left = original view, right = inpainted view).

**This repo drives Stage 2 only** (issue #140).  Stage 1 is *not* run: its
embedded DepthCrafter would duplicate this repo's own depth chain
(``--depth-model depthcrafter`` / ``depth-anything``) and its 3 GB of weights,
and it hard-crashes on a stock checkout (``No module named
'dependency.DepthCrafter.depthcrafter'``).  Instead :class:`CLIBackend`
assembles the Stage-2 input grid itself — the pipeline's own per-frame depth
maps (``depth_dir``) + an in-repo forward-splat of the input video — and feeds
it to ``inpainting_inference.py``.  See ``docs/STEREOCRAFTER_SETUP.md``.

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

# How many lines of subprocess output to surface on success (DEBUG) /
# failure (ERROR + exception summary) — issue #127.
_OUTPUT_TAIL_SUCCESS_LINES = 20
_OUTPUT_TAIL_FAILURE_LINES = 40


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


# ---------------------------------------------------------------------------
# Stage-2 input assembly (issue #140 — replaces upstream Stage 1)
# ---------------------------------------------------------------------------


def _load_video_frames_rgb(video_path: str) -> tuple[list, float]:
    """Read a video into a list of RGB uint8 frames + fps (OpenCV, CPU)."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video for splat assembly: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames: list = []
    try:
        while True:
            ret, bgr = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"Input video has no readable frames: {video_path}")
    return frames, fps


def _load_depth_maps(depth_dir: str, num_frames: int) -> list:
    """Load per-frame depth maps from *depth_dir* as float32 arrays in [0, 1].

    Consumes whatever the pipeline's depth stage produced — DepthCrafter's
    real ``*_depth.mp4`` grayscale video, ``depth_*.npy`` checkpoints, or
    ``depth_*.png`` visualisations — via the **shared** reader
    :func:`pipeline.depth_crafter.load_depth_maps_from_dir` (issue #145: the
    mp4 consumer-side gap of issue #126 must be fixed in exactly one place).
    Each map is resized to the video frame size by the caller.  Maps are
    min-max normalised per-frame to [0, 1] — the same convention upstream
    Stage 1 applies to its DepthCrafter output.
    """
    from pipeline.depth_crafter import load_depth_maps_from_dir

    try:
        depths = load_depth_maps_from_dir(depth_dir)
    except RuntimeError as exc:
        raise RuntimeError(
            f"{exc}\n"
            f"  Run the depth stage first (--stage depth, or drop --stereo-model stereocrafter\n"
            f"  so the default depth-shift renderer is used)."
        ) from None

    if len(depths) < num_frames:
        raise RuntimeError(
            f"Depth dir {depth_dir} has {len(depths)} map(s) but the input video has "
            f"{num_frames} frame(s) — the depth checkpoint is truncated/stale.  Re-run the "
            f"depth stage so every frame has a depth map."
        )
    return depths[:num_frames]


def _forward_splat(frame_rgb, depth, max_disp: float) -> tuple:
    """Forward-splat the left view to the right eye; return (warped, occlusion).

    In-repo numpy port of upstream ``ForwardWarpStereo`` (softmax-weighted
    forward splatting, ``occlu_map=True``) — the CUDA-only ``Forward_Warp``
    extension upstream Stage 1 requires is deliberately NOT used here (issue
    #140).  *depth* is min-max normalised to [0, 1] per frame (near = 1),
    matching the convention upstream applies to its DepthCrafter output.

    Returns ``(warped_rgb, occlusion_mask)`` — both ``(H, W, 3)`` float32 in
    [0, 1]; the mask is 1.0 where the right eye is disoccluded (needs
    inpainting).
    """
    import numpy as np

    h, w = frame_rgb.shape[:2]
    disp = (depth.astype(np.float32) * 2.0 - 1.0) * max_disp  # (H, W), px

    # Weights: upstream uses 1.414 ** (disp - disp.min()) to avoid overflow.
    weights = np.power(1.414, disp - disp.min()).astype(np.float32)

    # flow = -disp (right eye looks left); splat each source pixel onto the
    # target grid and accumulate softmax weights (vectorised over pixels).
    grid_x = np.arange(w, dtype=np.float32)[None, :].repeat(h, axis=0)
    target_x = np.clip(np.rint(grid_x - disp).astype(np.int64), 0, w - 1)
    # Flat 1D target index per source pixel (row-major: row * w + col) — the
    # accumulators below are 1D length h*w, so a bare reshape(-1) of the 2D
    # (row, col) target would collapse every row onto the same indices.
    rows = np.arange(h, dtype=np.int64)[:, None].repeat(w, axis=1)
    flat_t = (rows * w + target_x).reshape(-1)
    flat_w = weights.reshape(-1)

    warped = np.empty_like(frame_rgb, dtype=np.float32)
    for c in range(3):
        num = np.zeros(h * w, dtype=np.float64)
        np.add.at(num, flat_t, (frame_rgb[..., c].astype(np.float32) * weights).reshape(-1))
        den = np.zeros(h * w, dtype=np.float64)
        np.add.at(den, flat_t, flat_w)
        den_safe = np.where(den > 1e-6, den, 1.0)
        warped[..., c] = (num / den_safe).reshape(h, w)

    # Occlusion: splat a ones map; pixels nothing lands on are disoccluded.
    occ_num = np.zeros(h * w, dtype=np.float64)
    np.add.at(occ_num, flat_t, flat_w)
    occlusion = (occ_num.reshape(h, w) <= 1e-6).astype(np.float32)
    occlusion = np.repeat(occlusion[..., None], 3, axis=-1)

    return warped, occlusion


def _colorize_depth(depth) -> object:
    """Rainbow-colourised depth visualisation (top-right grid quadrant).

    Cosmetic only — upstream Stage 2 crops this quadrant away.  Mirrors the
    turbo-style colouring of upstream's ``vis_sequence_depth``.
    """
    import cv2
    import numpy as np

    vis = cv2.applyColorMap((np.clip(depth, 0.0, 1.0) * 255).astype(np.uint8), cv2.COLORMAP_RAINBOW)
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def _write_splatting_grid_video(
    input_path: str,
    depth_dir: str,
    output_path: str,
    max_disp: float,
) -> None:
    """Assemble the Stage-2 (``inpainting_inference.py``) input grid video.

    The upstream Stage-2 script consumes a 2×2 grid video of shape
    ``(2H, 2W, 3)`` whose quadrants are (verified against the actual
    ``inpainting_inference.py`` source, 2026-09-01)::

        [ left        | depth_vis   ]
        [ mask        | warped_right]

    Upstream produces this grid in Stage 1 with an embedded DepthCrafter +
    a CUDA forward-splat kernel.  This repo replaces that step (issue #140):
    depth comes from the pipeline's own depth stage (*depth_dir*) and the
    forward-splat is the in-repo numpy port in :func:`_forward_splat`.
    """
    import cv2
    import numpy as np

    frames, fps = _load_video_frames_rgb(input_path)
    depths = _load_depth_maps(depth_dir, len(frames))

    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w * 2, h * 2),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open splatting grid video for writing: {output_path}")

    log.info(
        "StereoCrafter: assembling Stage-2 grid video (%d frames, %dx%d quadrants) → %s",
        len(frames),
        w,
        h,
        output_path,
    )
    try:
        for frame_rgb, depth in zip(frames, depths, strict=True):
            depth = np.asarray(depth, dtype=np.float32)
            if depth.ndim == 3:
                depth = depth.mean(axis=-1)
            if depth.shape != (h, w):
                depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
            d_min, d_max = float(depth.min()), float(depth.max())
            depth = (depth - d_min) / (d_max - d_min) if d_max > d_min else np.zeros_like(depth)

            frame_f = frame_rgb.astype(np.float32) / 255.0
            warped, mask = _forward_splat(frame_f, depth, max_disp)
            depth_vis = _colorize_depth(depth)

            top = np.concatenate([frame_f, depth_vis], axis=1)
            bottom = np.concatenate([mask, warped], axis=1)
            grid = np.concatenate([top, bottom], axis=0)
            grid_uint8 = np.clip(grid * 255.0, 0, 255).astype(np.uint8)
            writer.write(cv2.cvtColor(grid_uint8, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


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
# Optional locally pre-downloaded SVD base (Stage 2 --pre_trained_path).
INREPO_SVD_DIR = _REPO_ROOT / "models" / "svd-img2vid-xt-1-1"
# HF model id for the SVD base — passed through to diffusers (which downloads
# it on first run) when no local copy exists.  Never pass a *nonexistent*
# local path here: diffusers/transformers would treat the path string as a
# repo id and crash with "Repo id must use alphanumeric chars" (issue #147).
SVD_HF_MODEL_ID = "stabilityai/stable-video-diffusion-img2vid-xt-1-1"

# 12 GB VRAM (RTX 4070 SUPER) safe defaults.  512 short-side is the sweet spot;
# bump to 768/1024 on larger GPUs via --stereocrafter-max-res or the env var.
DEFAULT_MAX_RESOLUTION = 512

# Default subprocess timeout for the inpainting stage (seconds) — the
# historical hard-coded 2 hours.  Override via STEREOCRAFTER_TIMEOUT_SEC
# (issue #134).
DEFAULT_TIMEOUT_SEC = 7200

# Stereo baseline used when forward-splatting the right eye (issue #140).
# Mirrors the upstream Stage-1 default: disp = (depth * 2 - 1) * MAX_DISP,
# so near pixels shift up to +20 px and far pixels down to -20 px.
DEFAULT_MAX_DISP = 20.0

# Upstream TencentARC/StereoCrafter has NO ``run.py``.  Its root-level fire-style
# entry scripts (verified against the actual checkout, 2026-09-01) are
# ``depth_splatting_inference.py`` (Stage 1) and ``inpainting_inference.py``
# (Stage 2).  This repo drives **Stage 2 only** (issue #140): Stage 1 imports
# an *embedded* DepthCrafter (``dependency.DepthCrafter.depthcrafter``) that a
# stock checkout does not ship — and this repo deliberately never embeds it
# (the pipeline's own ``--depth-model`` chain already produces the depth, and
# the forward-splat is assembled in-repo).  The recognized entry is therefore
# the inpainting script alone.
INFERENCE_SCRIPT = "inpainting_inference.py"
# Candidate entry scripts the backend will look for, in priority order.  Only
# the real upstream Stage-2 name is recognized; legacy guesses (``run.py``,
# ``inference.py``, ``scripts/inference.py``) and Stage 1
# (``depth_splatting_inference.py``) are deliberately absent so a mislabeled
# checkout is not silently accepted.
INFERENCE_SCRIPT_CANDIDATES = [
    INFERENCE_SCRIPT,  # Stage 2 (inpainting) — the only entry this repo drives
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
    """Pluggable backend for the StereoCrafter Stage-2 inpainting call."""

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
            depth_dir: Directory with the pipeline's own per-frame depth maps
                (``depth_*.npy`` or ``*.png``) — consumed by the in-repo
                forward-splat assembly (issue #140).
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

    ===================== =============================== =============================================
    Constructor param     Env var                         Default (if env unset)
    ===================== =============================== =============================================
    ``repo_dir``          ``STEREOCRAFTER_REPO_DIR``      in-repo ``third_party/StereoCrafter`` *(if exists)*
    ``python_exe``        ``STEREOCRAFTER_PYTHON``        in-repo venv python *(if exists)*, else ``python``
    ``checkpoint_dir``    ``STEREOCRAFTER_CKPT_DIR``      in-repo ``models/StereoCrafter`` *(if exists)*, else ``(repo_dir)/checkpoints``
    ``pre_trained_path``  ``STEREOCRAFTER_SVD_PATH``      in-repo ``models/svd-img2vid-xt-1-1`` *(if exists)*, else the HF id ``stabilityai/stable-video-diffusion-img2vid-xt-1-1`` (diffusers downloads on first run)
    ``max_resolution``    ``STEREOCRAFTER_MAX_RES``       ``512`` (12 GB VRAM safe)
    ``max_disp``          ``STEREOCRAFTER_MAX_DISP``      ``20.0`` (stereo baseline, upstream Stage-1 default)
    (stage timeout)       ``STEREOCRAFTER_TIMEOUT_SEC``   ``7200`` (2 hours)
    ===================== =============================== =============================================

    ``checkpoint_dir`` is the StereoCrafter UNet dir (``--unet_path``) and
    ``pre_trained_path`` the SVD base model dir (``--pre_trained_path``), both
    for the Stage-2 ``inpainting_inference.py`` — the only upstream script this
    repo drives (issue #140).  The upstream Stage-1 ``--unet_path`` (an embedded
    DepthCrafter) is gone: depth comes from the pipeline's own depth stage and
    the forward-splat is assembled in-repo.

    If none of repo_dir / env / in-repo resolve, the constructor raises and
    points to ``scripts/setup_stereocrafter.py``.
    """

    def __init__(
        self,
        repo_dir: str | None = None,
        python_exe: str | None = None,
        checkpoint_dir: str | None = None,
        pre_trained_path: str | None = None,
        max_resolution: int | None = None,
        max_disp: float | None = None,
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

        # pre_trained_path (SVD base model, --pre_trained_path for Stage 2):
        #   explicit > env > in-repo models/svd-img2vid-xt-1-1 (if exists) >
        #   HF model id (remote resolution on first run — NOT the recommended path).
        #
        # Issue #147: NEVER default to a nonexistent local path — diffusers /
        # transformers would treat the path string as an HF repo id and crash
        # with "Repo id must use alphanumeric chars".
        #
        # Issue #150: the SVD repo is GATED — the HF account behind the local
        # token must have accepted the license or any download 403s.
        #
        # Issue #155: prefer a LOCAL dir.  The repo ships ONLY safetensors
        # (no .bin); loading the HF id resolves the weight file remotely via
        # transformers' ``cached_file``, which wraps any non-OSError (auth /
        # network glitch) as the misleading "make sure ...
        # pytorch_model.fp16.bin" error.  A local snapshot lets the
        # local-folder branch resolve ``model.fp16.safetensors`` via
        # ``os.path.isfile``.  The bootstrap pre-downloads that local
        # snapshot (fp16 safetensors only) by default.
        if pre_trained_path:
            self.pre_trained_path = str(Path(pre_trained_path).resolve())
        elif os.environ.get("STEREOCRAFTER_SVD_PATH"):
            self.pre_trained_path = str(Path(os.environ["STEREOCRAFTER_SVD_PATH"]).resolve())
        elif INREPO_SVD_DIR.is_dir():
            self.pre_trained_path = str(INREPO_SVD_DIR)
        else:
            self.pre_trained_path = SVD_HF_MODEL_ID
            log.warning(
                "StereoCrafter: SVD base not found locally at %s — falling back to HF model id %r "
                "for remote resolution on first run.  THIS PATH TRIPPED ISSUE #155 (the misleading "
                "'pytorch_model.fp16.bin' safetensors load failure: the repo ships ONLY safetensors, "
                "no .bin, so the remote cached_file fallback raises that error), AND it is a GATED "
                "repo — the HF account behind the local token must have accepted the license at "
                "https://huggingface.co/%s or the runtime download will 403 (issue #150).  A LOCAL "
                "snapshot is strongly preferred.  Pre-download it (fp16 safetensors only, ≈5 GB) by "
                "re-running 'python scripts/setup_stereocrafter.py' (it pre-downloads the SVD base "
                "by default), or set STEREOCRAFTER_SVD_PATH to an existing local snapshot.",
                INREPO_SVD_DIR,
                SVD_HF_MODEL_ID,
                SVD_HF_MODEL_ID,
            )

        # max resolution for inference (short side); 512 is the 12 GB VRAM safe default.
        self.max_resolution = max_resolution or int(
            os.environ.get("STEREOCRAFTER_MAX_RES", str(DEFAULT_MAX_RESOLUTION))
        )

        # Stereo baseline for the in-repo forward-splat (px); mirrors the
        # upstream Stage-1 --max_disp default.
        self.max_disp: float = float(
            max_disp if max_disp is not None else os.environ.get("STEREOCRAFTER_MAX_DISP", str(DEFAULT_MAX_DISP))
        )

        # Subprocess timeout in seconds (issue #134: was a hard-coded 2 hours).
        self.timeout_sec: int = int(os.environ.get("STEREOCRAFTER_TIMEOUT_SEC", str(DEFAULT_TIMEOUT_SEC)))

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

        # Look for the Stage-2 inference entry point.  Only the real upstream
        # name (inpainting_inference.py) is recognized; a stray inference.py /
        # run.py / depth_splatting_inference.py is NOT accepted as a substitute.
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
        """Return the path to the Stage-2 inference entry point in the repo.

        With *name* given, resolve that specific entry (raises if absent).
        Without *name*, return the first candidate that exists on disk (the
        inpainting script).
        """
        repo = Path(self.repo_dir)
        names = [name] if name else list(INFERENCE_SCRIPT_CANDIDATES)
        for n in names:
            candidate = repo / n
            if candidate.is_file():
                return str(candidate)
        raise RuntimeError(
            f"No known inference script found in {self.repo_dir}. "
            f"Expected: {', '.join(INFERENCE_SCRIPT_CANDIDATES)}. "
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

        # Issue #140: this repo drives Stage 2 (inpainting_inference.py) ONLY.
        # Upstream Stage 1 (depth_splatting_inference.py) embeds its own
        # DepthCrafter under dependency/DepthCrafter/ — a stock checkout does
        # not ship it, so Stage 1 hard-crashes (No module named
        # 'dependency.DepthCrafter.depthcrafter'), and this repo deliberately
        # never embeds it (duplicate 3 GB weights + a conflicting depth chain).
        # Instead the backend assembles the Stage-2 input itself:
        #
        #   1. In-repo assembly (no subprocess): input video + the pipeline's
        #      own per-frame depth maps (*depth_dir*) are forward-splatted into
        #      the 2x2 grid video Stage 2 consumes:
        #          [ left | depth_vis ]
        #          [ mask | warped    ]
        #   2. Stage 2  inpainting_inference.py   ← the disocclusion step
        #        --pre_trained_path <SVD base>  --unet_path <StereoCrafter unet>
        #        --input_video_path <grid>     --save_dir <dir>
        #        [--frames_chunk 23] [--overlap 3] [--tile_num 1]
        #
        # Stage 2 writes a single side-by-side video (<name>_sbs.mp4); the
        # pipeline contract wants separate left/right files, so the backend
        # splits the SBS frame into L/R afterwards.

        if not depth_dir or not os.path.isdir(depth_dir):
            raise NotADirectoryError(
                f"Depth directory not found: {depth_dir}. "
                f"Run the depth stage first (--stage depth) so per-frame depth maps exist."
            )

        # Absolutize caller paths: the subprocess runs with cwd=repo_dir, so
        # relative paths would resolve against the StereoCrafter checkout.
        abs_input = str(Path(input_path).resolve())
        work_dir = Path(tempfile.mkdtemp(prefix="stereocrafter_work_"))
        splat_video = str((work_dir / "splatting_results.mp4").resolve())
        sbs_dir = str(work_dir.resolve())

        # --- Assemble the Stage-2 input grid (in-repo; replaces upstream
        #     Stage 1).  Consumes the pipeline's own depth maps via depth_dir.
        _write_splatting_grid_video(abs_input, depth_dir, splat_video, self.max_disp)

        # --- Stage 2: disocclusion inpainting -----------------------------
        stage2_script = self._find_inference_script(INFERENCE_SCRIPT)
        cmd: list[str] = [
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
        self._run_subprocess(cmd, label="Stage 2 (disocclusion inpainting)", output_dir=sbs_dir)

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
    def _run_subprocess(self, cmd: list[str], *, label: str, output_dir: str) -> None:
        """Run a StereoCrafter subprocess stage, raising on failure.

        The command is always a list (never ``shell=True``).  *label* names
        the stage for error messages; *output_dir* is the stage's output
        directory, listed verbatim in failure messages.

        stdout/stderr go to temp files (the same drained-file pattern used
        for ffmpeg in ``pipeline/streaming_pipeline.py``): an undrained PIPE
        deadlocks once its 64 KB buffer fills, and DEVNULL hides the very
        error the operator needs.  On success the last few lines are logged
        at DEBUG (no INFO-level spam); on failure the tail is logged at
        ERROR and folded into the raised exception (issue #127).
        """
        log.info("StereoCrafter CLIBackend %s command: %s", label, " ".join(cmd))
        log.info("StereoCrafter CLIBackend cwd: %s", self.repo_dir)

        # Temp files are OS-drained at no cost — no PIPE, no deadlock.
        # Closed below, not at this scope's exit, hence no `with`.
        out_file = tempfile.TemporaryFile(prefix="stereocrafter-stdout-")  # noqa: SIM115
        err_file = tempfile.TemporaryFile(prefix="stereocrafter-stderr-")  # noqa: SIM115
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
                    f"Set STEREOCRAFTER_PYTHON or --stereocrafter-python "
                    f"to the correct path."
                ) from exc
            except subprocess.TimeoutExpired:
                # Issue #134: the timeout branch must carry the same diagnostic
                # context as the non-zero-exit branch (issue #127) — command,
                # cwd, key params, the output tail produced before the kill,
                # and the real contents of the output dir.
                stdout_tail = _read_tail_lines(out_file, _OUTPUT_TAIL_FAILURE_LINES)
                stderr_tail = _read_tail_lines(err_file, _OUTPUT_TAIL_FAILURE_LINES)
                raise RuntimeError(
                    f"StereoCrafter {label} timed out after {self.timeout_sec} seconds "
                    f"(configured via STEREOCRAFTER_TIMEOUT_SEC; default {DEFAULT_TIMEOUT_SEC}).\n"
                    f"  The video may be too long or the GPU too slow — raise the timeout or\n"
                    f"  lower the workload (e.g. STEREOCRAFTER_MAX_RES, currently {self.max_resolution}).\n"
                    f"  Command: {' '.join(cmd)}\n"
                    f"  cwd: {self.repo_dir}\n"
                    f"  Output dir: {output_dir}\n"
                    f"{_dir_listing_block(output_dir)}"
                    f"  --- stdout (last {_OUTPUT_TAIL_FAILURE_LINES} lines before timeout) ---\n"
                    f"{_indent(stdout_tail)}\n"
                    f"  --- stderr (last {_OUTPUT_TAIL_FAILURE_LINES} lines before timeout) ---\n"
                    f"{_indent(stderr_tail)}\n"
                    f"  See docs/STEREOCRAFTER_SETUP.md for troubleshooting."
                ) from None

            if result.returncode != 0:
                stdout_tail = _read_tail_lines(out_file, _OUTPUT_TAIL_FAILURE_LINES)
                stderr_tail = _read_tail_lines(err_file, _OUTPUT_TAIL_FAILURE_LINES)
                log.error(
                    "StereoCrafter %s failed (exit code %s).\n"
                    "--- stdout (last %d lines) ---\n%s\n"
                    "--- stderr (last %d lines) ---\n%s",
                    label,
                    result.returncode,
                    _OUTPUT_TAIL_FAILURE_LINES,
                    stdout_tail,
                    _OUTPUT_TAIL_FAILURE_LINES,
                    stderr_tail,
                )
                raise RuntimeError(
                    f"StereoCrafter {label} failed (exit code {result.returncode}).\n"
                    f"  Command: {' '.join(cmd)}\n"
                    f"  cwd: {self.repo_dir}\n"
                    f"  Output dir: {output_dir}\n"
                    f"{_dir_listing_block(output_dir)}"
                    f"  --- stdout (last {_OUTPUT_TAIL_FAILURE_LINES} lines) ---\n"
                    f"{_indent(stdout_tail)}\n"
                    f"  --- stderr (last {_OUTPUT_TAIL_FAILURE_LINES} lines) ---\n"
                    f"{_indent(stderr_tail)}\n"
                    f"  See docs/STEREOCRAFTER_SETUP.md for troubleshooting."
                )

            # Success: tail at DEBUG so the default INFO level stays quiet.
            log.debug(
                "StereoCrafter %s finished.\n--- stdout (last %d lines) ---\n%s\n--- stderr (last %d lines) ---\n%s",
                label,
                _OUTPUT_TAIL_SUCCESS_LINES,
                _read_tail_lines(out_file, _OUTPUT_TAIL_SUCCESS_LINES),
                _OUTPUT_TAIL_SUCCESS_LINES,
                _read_tail_lines(err_file, _OUTPUT_TAIL_SUCCESS_LINES),
            )
        finally:
            out_file.close()
            err_file.close()

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
        max_resolution: int | None = None,
        max_disp: float | None = None,
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
                max_resolution=max_resolution,
                max_disp=max_disp,
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
            depth_dir: Directory with the pipeline's own per-frame depth maps
                (``depth_*.npy`` or ``*.png``), consumed by the in-repo
                forward-splat assembly.  If None, a temporary directory is
                created and must be populated by the caller before calling
                this method.
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
