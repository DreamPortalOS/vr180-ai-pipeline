#!/usr/bin/env python3
"""One-command, in-repo StereoCrafter bootstrap (CUDA-only, disocclusion inpainting).

Deploys TencentARC/StereoCrafter **inside** this repo so the pipeline's
:mod:`pipeline.stereo_crafter` ``CLIBackend`` can pick it up automatically via
in-repo default paths — no separate ``D:/StereoCrafter`` install required.

Layout created (everything is gitignored):

    third_party/StereoCrafter/         ← git clone of TencentARC/StereoCrafter (incl. its own .venv)
    models/StereoCrafter/              ← TencentARC/StereoCrafter weights (hf snapshot)
    models/svd-img2vid-xt-1-1/         ← SVD base model (gated HF repo, ~10 GB)

Steps (idempotent — re-running only fills missing pieces):

    1. Clone TencentARC/StereoCrafter (or ``git pull`` if present).
       ``--repo-dir`` can point at an existing checkout instead (e.g. ``D:/StereoCrafter``).
    2. Build a **dedicated** venv inside the node dir (never the project-root venv):
       - torch==2.6.0 + torchvision==0.21.0 on the official cu124 index (stable, NOT nightly).
       - Node runtime deps: a curated subset (``RUNTIME_DEPS``) of the packages
         the inference entry point actually imports.  See docs/STEREOCRAFTER_SETUP.md
         for the rationale (no ``pip install -e .``, torch pins preserved).
    3. Download ``TencentARC/StereoCrafter`` weights into ``models/StereoCrafter/`` via
       ``huggingface_hub.snapshot_download`` (existing download is skipped).
    4. Pre-download the **SVD base model** (``stabilityai/stable-video-diffusion-img2vid-xt-1-1``,
       ~10 GB) into ``models/svd-img2vid-xt-1-1/``.  This is a **gated** HF repo — the
       local HF token is read automatically and the account **must have accepted the
       model's license** (see :data:`_SVD_REPO_URL`).  A 403 from a non-authorized
       account produces a clear error pointing at the application page rather than a
       bare ``OSError`` (issue #150).  Skip with ``--skip-svd``.
    5. Self-check: run the repo's inference entry point with ``--help`` via the
       dedicated venv.

The script never touches the project-root venv and never downloads anything in CI
(the ``--dry-run`` flag is used by tests to assert the step sequence with zero I/O).

Usage::

    python scripts/setup_stereocrafter.py
    python scripts/setup_stereocrafter.py --repo-dir D:/StereoCrafter   # existing checkout
    python scripts/setup_stereocrafter.py --skip-model                  # weights already downloaded
    python scripts/setup_stereocrafter.py --skip-svd                    # SVD base already downloaded / defer to first run
    python scripts/setup_stereocrafter.py --skip-deps                   # venv + pip install already done
    python scripts/setup_stereocrafter.py --svd-dir D:/svd              # custom SVD target dir
    python scripts/setup_stereocrafter.py --hf-token hf_...             # explicit HF token (else auto-read)
    python scripts/setup_stereocrafter.py --pip-mirror https://pypi.tuna.tsinghua.edu.cn/simple
    python scripts/setup_stereocrafter.py --pip-timeout 7200            # raise the per-pip wall-clock cap
    python scripts/setup_stereocrafter.py --dry-run                     # print planned steps, no side effects
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

log = logging.getLogger("setup-stereocrafter")

# ---------------------------------------------------------------------------
# Repo root: scripts/setup_stereocrafter.py lives at <repo>/scripts/, so repo
# root is two parents up.  The CLIBackend in pipeline/stereo_crafter.py uses
# the same convention (parent.parent of the pipeline/ package).
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent

NODE_REPO_URL = "https://github.com/TencentARC/StereoCrafter.git"
INREPO_NODE_DIR = REPO_ROOT / "third_party" / "StereoCrafter"
INREPO_MODEL_DIR = REPO_ROOT / "models" / "StereoCrafter"

# Dedicated venv lives *inside* the node dir (isolated from the project venv).
INREPO_VENV_DIR = INREPO_NODE_DIR / ".venv"
INREPO_PYTHON = INREPO_VENV_DIR / (Path("Scripts") / "python.exe" if os.name == "nt" else Path("bin") / "python")

# Stable cu124 torch — paired torchvision release (torchvision==2.6.0 does not exist).
TORCH_VERSION = "2.6.0"
TORCHVISION_VERSION = "0.21.0"
TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu124"

# ---------------------------------------------------------------------------
# pip subprocess timeouts (issue #162).
#
# The bootstrap used to hard-code a 1200s (20 min) wall-clock cap on the two
# pip installs in Step 2.  That is too tight when pip has to *downgrade*
# (e.g. transformers 5.16 → 4.42.3 from issue #155) on a slow/unstable network
# (common from mainland China): 20 min is not enough to re-resolve the dep
# tree + re-download, the subprocess hits ``TimeoutExpired`` and the whole
# bootstrap dies mid-install — leaving the env half-installed and blocking the
# idempotent re-run that is the only way to recover.
#
# The cap is now configurable and looser by default.  ``--pip-timeout`` (CLI)
# overrides the env var; the env var overrides the default.  We also pass pip
# its own per-connection ``--timeout`` so a single stalled TCP read does not
# silently hang for the full wall-clock allowance.
# ---------------------------------------------------------------------------
# Default wall-clock cap for a single ``pip install`` subprocess (seconds).
DEFAULT_PIP_TIMEOUT = 3600  # 1 hour — covers a downgrade + re-download on slow links
# pip's own per-connection socket timeout (seconds).  ``--timeout`` is a pip
# flag (NOT a subprocess kwarg) — it bounds each individual HTTP read so a
# stalled mirror connection fails fast instead of burning the full allowance.
PIP_CONNECT_TIMEOUT = 120
# Environment variable that overrides DEFAULT_PIP_TIMEOUT without touching the
# CLI.  Shared across all three setup scripts so the operator can set it once.
# The ``SETUP_`` prefix avoids colliding with pip's own ``PIP_TIMEOUT`` env var
# (which controls pip's per-connection timeout and would be double-counted).
PIP_TIMEOUT_ENV_VAR = "SETUP_PIP_TIMEOUT"

# Curated runtime deps for the StereoCrafter inference entry point.
#
# Deliberately NOT a full ``pip install -e .`` of the upstream pyproject:
#   1. StereoCrafter is run as a script (an inference entry point), so it never
#      needs to be installed as an importable package — the curated list is all
#      the CLI needs.
#   2. Upstream pyproject files frequently pin torch / add demo-only deps
#      (gradio, xformers) that would either bump the torch cu124 pairing or
#      blow out the install.  torch / torchvision are intentionally NOT here —
#      they are pinned in Step 2 against the cu124 index so a transitive dep
#      can never bump them.
# If inpainting_inference.py / depth_splatting_inference.py start importing
# something new, add it here rather than switching to ``pip install -e .``.
#
# Verified against the two upstream entry points' top-level imports (the only
# two scripts this bootstrap self-checks):
#   inpainting_inference.py:        os, cv2, numpy, fire, torch, decord,
#                                   transformers, diffusers (+ local pipeline)
#   depth_splatting_inference.py:   gc, os, cv2, numpy, torch, torch.nn,
#                                   torchvision.io, diffusers, fire, decord
#                                   (+ vendored dependency/ & Forward_Warp)
# → both import ``decord``; numpy ships transitively via torch/diffusers.
#
# Pinned versions of the model loaders, matched to the combo upstream
# TencentARC/StereoCrafter tested against (its requirements.txt pins
# transformers==4.42.3 / diffusers==0.29.2).  WHY pinning matters here
# (issue #155): the SVD base repo ships ONLY safetensors (no .bin at all),
# and inpainting_inference.py loads the image_encoder / vae with
# ``variant="fp16"`` but WITHOUT ``use_safetensors=True``.  Whether the loader
# then picks ``model.fp16.safetensors`` over ``pytorch_model.fp16.bin`` is
# governed by the local-folder branch of transformers'
# _get_resolved_checkpoint_files — which prefers safetensors when
# ``use_safetensors is not False`` (the default, None) in 4.42.3+.  An
# unpinned ``transformers`` could resolve to a 5.x that changed the vendored
# pipeline's API surface, or an older line that defaulted to .bin; pinning to
# the upstream-tested pair guarantees the safetensors-first path is exactly
# the one upstream validated.  See docs/STEREOCRAFTER_SETUP.md §6.
TRANSFORMERS_PIN = "==4.42.3"
DIFFUSERS_PIN = "==0.29.2"

RUNTIME_DEPS: tuple[str, ...] = (
    f"diffusers{DIFFUSERS_PIN}",  # SD/SVD-based video diffusion backbone used for inpainting
    f"transformers{TRANSFORMERS_PIN}",  # pulled by diffusers models — pinned (issue #155, safetensors)
    "accelerate",  # used by diffusers model loaders
    "huggingface-hub",  # weight download / caching
    "opencv-python",  # video I/O (cv2)
    "einops",  # tensor reshapes used by the video diffusion blocks
    "ftfy",  # string cleaning (common in HF model code)
    "fire",  # inpainting_inference.py / depth_splatting_inference.py fire-style CLI
    "decord",  # video loader (VideoReader) — both entry scripts import it
)

# HuggingFace repo for the StereoCrafter weights.
_MODEL_REPO_ID = "TencentARC/StereoCrafter"

# SVD base model (Stage 2 --pre_trained_path).  Pre-downloaded by default into
# INREPO_SVD_DIR so the first inference run does not have to fetch ~10 GB.  When
# the local copy is missing the pipeline instead passes this HF id straight to
# diffusers, which downloads it on the first inference run (issue #147).
#
# IMPORTANT (issue #150): this is a **gated** HF repo — the account behind the
# local HF token must have accepted the model's license agreement, otherwise
# snapshot_download / diffusers raise a 403 ``OSError: ... gated repo``.  The
# bootstrap surfaces that case with a clear error pointing at the application
# page (see :func:`_svd_gated_error`) instead of a bare OSError.
_SVD_REPO_ID = "stabilityai/stable-video-diffusion-img2vid-xt-1-1"
# The HF web page where the license is accepted (used in error messages).
_SVD_REPO_URL = f"https://huggingface.co/{_SVD_REPO_ID}"
INREPO_SVD_DIR = REPO_ROOT / "models" / "svd-img2vid-xt-1-1"

# Where huggingface_hub stores the login token on disk (used as a fallback when
# the ``huggingface_hub`` package is importable but exposes no token helper, or
# when the caller passes none).  Mirrors ``huggingface_hub.constants`` so we do
# not hard-couple this setup script to the package's internals.
_HF_TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN")

# Selective snapshot patterns for the SVD base (issue #155).  The repo ships
# ONLY safetensors (no .bin); we fetch just the **fp16** safetensors + the
# configs every Stage-2 loader (CLIPVisionModelWithProjection / VAE / the
# vendored SVD pipeline) resolves, skipping the fp32 variants + the unused
# full-pipeline aggregate weight.  This cuts the download from ~10 GB to ~5 GB
# AND guarantees the local snapshot has exactly the files the local-folder
# resolver looks for (see :func:`_has_svd_fp16_safetensors`).
_SVD_FP16_ALLOW_PATTERNS: tuple[str, ...] = (
    "*.json",  # model_index.json + every subfolder's config.json
    "image_encoder/model.fp16.safetensors",
    "unet/diffusion_pytorch_model.fp16.safetensors",
    "vae/diffusion_pytorch_model.fp16.safetensors",
    "scheduler/scheduler_config.json",
    "feature_extractor/preprocessor_config.json",  # pipeline __init__ reads it
)
# Belt-and-braces: never fetch .bin (there are none upstream, but the intent
# stays legible if the allow list is loosened later).  Also exclude the fp32
# safetensors variants explicitly — only fp16 is wanted.
_SVD_IGNORE_PATTERNS: tuple[str, ...] = (
    "*.bin",
    "image_encoder/model.safetensors",  # fp32 variant — not needed
    "unet/diffusion_pytorch_model.safetensors",  # fp32 variant
    "vae/diffusion_pytorch_model.safetensors",  # fp32 variant
    "svd_xt_1_1.safetensors",  # full-pipeline aggregate (we load subfolders)
)


# ---------------------------------------------------------------------------
# pip-timeout resolver (issue #162).
#
# Precedence: explicit CLI value (--pip-timeout) > env var (SETUP_PIP_TIMEOUT)
# > DEFAULT_PIP_TIMEOUT (3600s).  An explicit CLI value wins even if the env
# var is also set, so a one-off ``--pip-timeout 7200`` on the command line is
# never silently overridden by a stale env var the operator forgot about.
# ---------------------------------------------------------------------------


def resolve_pip_timeout(cli_value: int | None = None) -> int:
    """Resolve the wall-clock timeout (seconds) for a single pip subprocess.

    *cli_value* is the ``--pip-timeout`` argument (``None`` if absent on the
    CLI).  Falls back to :data:`PIP_TIMEOUT_ENV_VAR`, then
    :data:`DEFAULT_PIP_TIMEOUT`.  A non-integer env value is ignored with a
    warning rather than crashing the bootstrap (it is a convenience override,
    not a hard requirement).
    """
    if cli_value is not None:
        return cli_value
    env_val = os.environ.get(PIP_TIMEOUT_ENV_VAR)
    if env_val:
        try:
            return int(env_val)
        except ValueError:
            log.warning(
                "Ignoring non-integer %s=%r — falling back to default %ss.",
                PIP_TIMEOUT_ENV_VAR,
                env_val,
                DEFAULT_PIP_TIMEOUT,
            )
    return DEFAULT_PIP_TIMEOUT


def _has_svd_fp16_safetensors(model_dir: Path) -> bool:
    """Return True if *model_dir* has the fp16 safetensors the Stage-2 loaders
    resolve on the local-folder branch (issue #155).

    Mirrors what transformers' ``_get_resolved_checkpoint_files`` looks for
    with ``variant="fp16"`` and ``use_safetensors`` left at its default
    (None → "is not False"): ``<subfolder>/model.fp16.safetensors`` (image_encoder)
    and ``<subfolder>/diffusion_pytorch_model.fp16.safetensors`` (unet / vae).
    A dir missing the fp16 image_encoder safetensors is NOT considered ready —
    that is exactly the file the upstream loaders need.
    """
    needed = (
        model_dir / "image_encoder" / "model.fp16.safetensors",
        model_dir / "unet" / "diffusion_pytorch_model.fp16.safetensors",
        model_dir / "vae" / "diffusion_pytorch_model.fp16.safetensors",
    )
    return all(p.is_file() for p in needed)


# ---------------------------------------------------------------------------
# Runner: either records steps (dry-run) or executes them.
# ---------------------------------------------------------------------------


class DryRunBuffer:
    """Accumulates step descriptions in dry-run mode; no I/O is performed."""

    def __init__(self) -> None:
        self.steps: list[str] = []

    def record(self, message: str) -> None:
        self.steps.append(message)


def run_step(
    cmd: list[str],
    cwd: str | None = None,
    timeout: int | None = None,
    *,
    dry_run: bool,
    buffer: DryRunBuffer,
    label: str | None = None,
) -> None:
    """Print *cmd*, then either record it (dry-run) or execute it.

    A ``subprocess.TimeoutExpired`` is caught and re-raised as a
    :class:`RuntimeError` carrying an actionable hint (issue #162) — pip's
    wall-clock timeout must NOT be a bare, uncaught traceback that kills the
    bootstrap with no guidance.  The hint tells the operator to re-run
    (idempotent — completed steps are skipped) or raise the allowance, and
    mentions ``--pip-mirror`` for the slow-network (mainland China) case.
    """
    label = label or " ".join(cmd)
    if dry_run:
        buffer.record(label)
        return
    log.info("▶ %s", label)
    try:
        subprocess.check_call(cmd, cwd=cwd, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise _pip_timeout_error(exc, timeout=timeout, label=label) from exc


def _pip_timeout_error(
    exc: subprocess.TimeoutExpired,
    *,
    timeout: int | None,
    label: str,
) -> RuntimeError:
    """Wrap a pip ``TimeoutExpired`` with an actionable hint (issue #162).

    The raw ``TimeoutExpired`` traceback says only "timed out after N seconds"
    and exits the whole bootstrap — the operator is left with a half-installed
    venv and no idea what to do next.  This surfaces three concrete recovery
    moves (idempotent re-run, ``--pip-mirror``, raise ``--pip-timeout``) so the
    failure is recoverable on its face.
    """
    secs = timeout if timeout is not None else DEFAULT_PIP_TIMEOUT
    return RuntimeError(
        f"{label} timed out after {secs}s — this is almost always a "
        "slow or unstable network connection (common from mainland China).\n"
        "  Recovery (the bootstrap is idempotent — completed steps are skipped):\n"
        "    1. Just re-run the same command.  pip resumes from its cache and\n"
        "       the venv is not recreated, so a slow install finishes across\n"
        "       two runs instead of dying on one.\n"
        "    2. If you have not already, route pip through a mirror:\n"
        "         python scripts/setup_stereocrafter.py --pip-mirror "
        "https://pypi.tuna.tsinghua.edu.cn/simple\n"
        f"    3. Raise the wall-clock allowance (e.g. --pip-timeout 7200, or\n"
        f"       set the {PIP_TIMEOUT_ENV_VAR} env var).\n"
        f"  Underlying error: {exc}"
    )


# ---------------------------------------------------------------------------
# Step 1: clone / pull the node repo (or use --repo-dir)
# ---------------------------------------------------------------------------


def _effective_node_dir(explicit_repo_dir: str | None) -> Path:
    """Resolve the node dir: explicit --repo-dir > in-repo default."""
    if explicit_repo_dir:
        return Path(explicit_repo_dir)
    return INREPO_NODE_DIR


def ensure_node_repo(
    explicit_repo_dir: str | None,
    *,
    dry_run: bool,
    buffer: DryRunBuffer,
) -> None:
    """Clone TencentARC/StereoCrafter, ``git pull`` if present, or use --repo-dir."""
    node_dir = _effective_node_dir(explicit_repo_dir)

    if explicit_repo_dir:
        if node_dir.is_dir() and _is_git_dir(node_dir):
            log.info("Existing repo-dir %s — pulling latest...", node_dir)
            run_step(
                ["git", "pull"],
                cwd=str(node_dir),
                dry_run=dry_run,
                buffer=buffer,
                label=f"git pull (in {node_dir})",
            )
            return
        if not node_dir.is_dir():
            log.warning("--repo-dir %s does not exist — will create it via git clone.", node_dir)

    if node_dir.is_dir():
        if not _is_git_dir(node_dir):
            log.info("Node dir exists but is not a git checkout — re-cloning.")
        else:
            log.info("Node dir already exists at %s — pulling latest...", node_dir)
            run_step(
                ["git", "pull"],
                cwd=str(node_dir),
                dry_run=dry_run,
                buffer=buffer,
                label=f"git pull (in {node_dir})",
            )
            return

    third_party_dir = REPO_ROOT / "third_party"
    clone_cmd = ["git", "clone", NODE_REPO_URL, str(node_dir)]
    clone_label = f"git clone {NODE_REPO_URL} {node_dir}"
    if dry_run:
        buffer.record(clone_label)
        return
    third_party_dir.mkdir(parents=True, exist_ok=True)
    log.info("▶ %s", clone_label)
    try:
        subprocess.check_call(clone_cmd, timeout=600)
    except subprocess.CalledProcessError as exc:
        _proxy_hint("git clone")
        raise exc

    log.info("Node repo cloned to %s", node_dir)


def _is_git_dir(path: Path) -> bool:
    return (path / ".git").exists()


def _proxy_hint(command: str) -> None:
    log.warning(
        "▸ %s failed — likely a network/proxy issue (common from mainland China).\n"
        "  Try setting your git/http proxy and re-run, e.g.:\n"
        "      git config --global http.proxy http://your-proxy:port\n"
        "      git config --global https.proxy http://your-proxy:port\n"
        "  Or clone manually:\n"
        "      git clone %s <path>",
        command,
        NODE_REPO_URL,
    )


# ---------------------------------------------------------------------------
# Step 2: dedicated venv + pip install
# ---------------------------------------------------------------------------


def _pip_mirror_args(pip_mirror: str | None) -> list[str]:
    """If a PyPI mirror is given, emit ``-i <url>``.  torch still uses --index-url."""
    if pip_mirror:
        return ["-i", pip_mirror]
    return []


def _venv_python_for(node_dir: Path) -> Path:
    """The venv python path that belongs to a given node dir."""
    venv_dir = node_dir / ".venv"
    return venv_dir / (Path("Scripts") / "python.exe" if os.name == "nt" else Path("bin") / "python")


def ensure_venv_and_deps(
    explicit_repo_dir: str | None,
    pip_mirror: str | None,
    *,
    pip_timeout: int,
    dry_run: bool,
    buffer: DryRunBuffer,
) -> None:
    """Create the dedicated venv and install torch (cu124) + the repo's deps.

    *pip_timeout* is the wall-clock cap (seconds) for each pip subprocess
    (issue #162 — ``--pip-timeout`` / ``SETUP_PIP_TIMEOUT``; default 3600s).
    Both pip commands also get pip's own per-connection ``--timeout
    {PIP_CONNECT_TIMEOUT}`` so a stalled connection fails fast instead of
    hanging for the full wall-clock allowance.
    """
    node_dir = _effective_node_dir(explicit_repo_dir)
    venv_dir = node_dir / ".venv"
    python_exe = _venv_python_for(node_dir)

    if not dry_run:
        if not python_exe.is_file():
            venv_cmd = [sys.executable, "-m", "venv", str(venv_dir)]
            log.info("Creating dedicated venv at %s (this may take a moment)...", venv_dir)
            subprocess.check_call(venv_cmd, timeout=300)
        else:
            log.info("Dedicated venv already exists at %s — re-installing runtime deps.", venv_dir)

    if dry_run:
        run_step(
            [sys.executable, "-m", "venv", str(venv_dir)],
            dry_run=True,
            buffer=buffer,
            label=f"{sys.executable} -m venv {venv_dir}",
        )

    cmd_torch = [
        str(python_exe),
        "-m",
        "pip",
        "install",
        "--retries",
        "10",
        "--timeout",
        str(PIP_CONNECT_TIMEOUT),
        f"torch=={TORCH_VERSION}",
        f"torchvision=={TORCHVISION_VERSION}",
        "--index-url",
        TORCH_INDEX_URL,
        *_pip_mirror_args(pip_mirror),
    ]
    run_step(cmd_torch, dry_run=dry_run, buffer=buffer, timeout=pip_timeout)

    base_cmd = [
        str(python_exe),
        "-m",
        "pip",
        "install",
        "--retries",
        "10",
        "--timeout",
        str(PIP_CONNECT_TIMEOUT),
    ]
    mirror_args = _pip_mirror_args(pip_mirror)
    cmd_node = [*base_cmd, *RUNTIME_DEPS, *mirror_args]
    label_node = f"pip install {' '.join(RUNTIME_DEPS)}"
    run_step(cmd_node, dry_run=dry_run, buffer=buffer, timeout=pip_timeout, label=label_node)


# ---------------------------------------------------------------------------
# Step 3: model weights via snapshot_download
# ---------------------------------------------------------------------------


def _model_dir_for(explicit_repo_dir: str | None) -> Path:
    """Where the TencentARC/StereoCrafter weights land. Always models/StereoCrafter."""
    return INREPO_MODEL_DIR


def download_models(
    explicit_repo_dir: str | None,
    skip_model: bool,
    *,
    dry_run: bool,
    buffer: DryRunBuffer,
) -> None:
    """Download TencentARC/StereoCrafter via snapshot_download; skip if already present."""
    if skip_model:
        log.info("--skip-model: model download skipped")
        return

    model_dir = _model_dir_for(explicit_repo_dir)

    label = f"snapshot_download {_MODEL_REPO_ID} → {model_dir}"
    if dry_run:
        buffer.record(label)
        return

    if model_dir.is_dir() and _has_snapshot_files(model_dir):
        log.info("Model snapshot already present at %s — skipping.", model_dir)
        return

    model_dir.mkdir(parents=True, exist_ok=True)
    log.info("▶ %s", label)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        log.warning("huggingface_hub not installed in this environment. Install with: pip install huggingface_hub")
        return

    try:
        snapshot_download(
            repo_id=_MODEL_REPO_ID,
            local_dir=str(model_dir),
            local_dir_use_symlinks=False,
        )
    except Exception as exc:
        log.warning("snapshot_download failed for %s: %s", _MODEL_REPO_ID, exc)
        log.warning("You can try again later, or clone the HF repo manually into %s.", model_dir)


def _has_snapshot_files(model_dir: Path) -> bool:
    """Return True if the snapshot dir contains any files (non-empty = downloaded)."""
    return any(child.is_file() for child in model_dir.iterdir())


# ---------------------------------------------------------------------------
# HF token + gated-repo handling (issue #150)
# ---------------------------------------------------------------------------


def _read_hf_token(explicit_token: str | None = None) -> str | None:
    """Resolve the HF access token to pass to ``snapshot_download``.

    Precedence: explicit arg (``--hf-token``) > the standard env vars
    (``HF_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN`` / ``HUGGINGFACE_TOKEN``) >
    ``huggingface_hub``'s own stored token (``HfFolder.get_token()``).  The
    ``huggingface_hub`` package is imported lazily so this stays importable
    in CI (where it is not installed); in that case we still consult the env
    vars as a fallback.  Returns ``None`` if no token is available.
    """
    if explicit_token:
        return explicit_token

    for var in _HF_TOKEN_ENV_VARS:
        value = os.environ.get(var)
        if value:
            return value

    try:
        from huggingface_hub import HfFolder

        return HfFolder.get_token()  # type: ignore[no-any-return]
    except ImportError:
        return None
    except Exception:  # pragma: no cover — defensive: never let token lookup crash setup
        return None


def _is_gated_repo_error(exc: BaseException) -> bool:
    """Return True if *exc* is the HF "gated repo" 401/403 access error.

    ``huggingface_hub`` raises this as an ``OSError`` whose message contains
    "gated repo" (e.g. ``OSError: You are trying to access a gated repo.``).
    We match on the message rather than the exception type so this is robust
    across ``huggingface_hub`` versions that may raise a subclass.
    """
    text = f"{exc}"
    text_lower = text.lower()
    if "gated repo" in text_lower or "restricted and you are not in the authorized list" in text_lower:
        return True
    # Bare HTTP 401/403 without the phrase — flag it if the text also mentions
    # the specific gated model so we do not mislabel an unrelated 403.
    return ("403" in text or "401" in text) and _SVD_REPO_ID in text_lower


def _svd_gated_error(exc: BaseException) -> RuntimeError:
    """Build the actionable error for the SVD gated-repo 403 (issue #150).

    The raw ``OSError: ... gated repo ... 403`` tells the user nothing about
    *why* or *what to do*.  This wraps it with the model page where the
    license must be accepted and notes the token requirement, so the failure
    is actionable on its face.
    """
    return RuntimeError(
        f"The SVD base model {_SVD_REPO_ID!r} is a gated Hugging Face repo and "
        f"the account behind the local HF token is not authorized to access it.\n"
        f"  {exc}\n"
        f"  → Open this page in a browser, sign in with the SAME Hugging Face "
        f"account whose token is on this machine, and accept the license:\n"
        f"      {_SVD_REPO_URL}\n"
        f"  Approval is usually instant.  The token is already on disk "
        f"(~/.cache/huggingface/token); after accepting, re-run this "
        f"bootstrap — no token re-login needed.\n"
        f"  To pass a token explicitly instead: --hf-token hf_xxx  "
        f"(or set the {', '.join(_HF_TOKEN_ENV_VARS)} env var)."
    )


def download_svd_base(
    svd_dir: str | None,
    *,
    skip_svd: bool = False,
    hf_token: str | None = None,
    dry_run: bool,
    buffer: DryRunBuffer,
) -> None:
    """Pre-download the SVD base model (Stage 2 ``--pre_trained_path``) into *target*.

    Runs by default (issue #150): the SVD base is a **gated** HF repo, so the
    first inference run's diffusers auto-download can fail with a bare
    ``OSError: ... gated repo ... 403`` that says nothing actionable.
    Front-loading the download here lets us read the local HF token, accept
    the license up front, and surface a clear error (with the application
    page) if the account is not yet authorized.

    It is ALSO the fix for issue #155 (the safetensors load failure).  The
    repo ships **only safetensors** (no ``.bin``); ``inpainting_inference.py``
    loads the image_encoder/vae with ``variant="fp16"`` and NO
    ``use_safetensors`` flag.  When the path is an HF repo *id*, transformers
    resolves the weight file remotely via ``cached_file`` — any non-``OSError``
    raised inside that call (auth glitch, transient network) gets re-wrapped
    as the misleading "make sure ... pytorch_model.fp16.bin" error.  A
    **local directory** dodges that: the local-folder branch of the resolver
    checks ``os.path.isfile(.../model.fp16.safetensors)`` first (when
    ``use_safetensors is not False``, the default), so a local snapshot loads
    cleanly.  That is why this step fetches the **fp16 safetensors only**
    (≈5 GB, not the full ~10 GB repo) into ``models/svd-img2vid-xt-1-1`` — and
    the pipeline picks that local dir up automatically (issue #147
    precedence), never relying on the remote ``cached_file`` path.

    ``--svd-dir`` overrides the target directory; ``--skip-svd`` skips the
    step entirely (the pipeline then passes the HF model id to diffusers,
    which downloads it on the first run — NOT recommended, see issue #155).
    """
    if skip_svd:
        log.info("--skip-svd: SVD base pre-download skipped")
        return

    target = Path(svd_dir) if svd_dir is not None else INREPO_SVD_DIR

    label = f"snapshot_download {_SVD_REPO_ID} → {target} (fp16 safetensors only, ≈5 GB)"
    if dry_run:
        buffer.record(label)
        return

    if target.is_dir() and _has_svd_fp16_safetensors(target):
        log.info("SVD base (fp16 safetensors) already present at %s — skipping.", target)
        return

    target.mkdir(parents=True, exist_ok=True)
    log.info("▶ %s (gated repo, needs HF token; fp16 safetensors only, ≈5 GB)", label)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        log.warning("huggingface_hub not installed in this environment. Install with: pip install huggingface_hub")
        return

    token = _read_hf_token(hf_token)
    if token is None:
        log.warning(
            "No HF token found (no --hf-token, no %s env var, no stored "
            "huggingface_hub login).  The SVD base %r is a GATED repo — a "
            "download without an authorized token will 403.  If you have not "
            "done so, accept the license at %s and run "
            "``huggingface-cli login`` (or pass --hf-token).",
            " / ".join(_HF_TOKEN_ENV_VARS),
            _SVD_REPO_ID,
            _SVD_REPO_URL,
        )

    try:
        snapshot_download(
            repo_id=_SVD_REPO_ID,
            local_dir=str(target),
            local_dir_use_symlinks=False,
            token=token,
            allow_patterns=_SVD_FP16_ALLOW_PATTERNS,
            ignore_patterns=_SVD_IGNORE_PATTERNS,
        )
    except Exception as exc:
        if _is_gated_repo_error(exc):
            # Gated repo: raise an actionable error naming the application page
            # rather than letting the bare 403 OSError through (issue #150).
            raise _svd_gated_error(exc) from exc
        log.warning("snapshot_download failed for %s: %s", _SVD_REPO_ID, exc)
        log.warning("You can try again later, or clone the HF repo manually into %s.", target)


# ---------------------------------------------------------------------------
# Step 4: self-check
# ---------------------------------------------------------------------------


def _find_inference_script(node_dir: Path) -> Path | None:
    """Return the path to the first known StereoCrafter inference entry point.

    The upstream ``TencentARC/StereoCrafter`` repo has no ``run.py`` — its two
    root-level ``fire``-style entry points are ``inpainting_inference.py``
    (Stage 2, the active entry ``run_inference.sh`` calls) and
    ``depth_splatting_inference.py`` (Stage 1).  Both support ``--help``.
    """
    candidates = [
        node_dir / "inpainting_inference.py",
        node_dir / "depth_splatting_inference.py",
        node_dir / "run.py",  # legacy fallback in case upstream re-adds it
        node_dir / "inference.py",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def self_check(
    explicit_repo_dir: str | None,
    *,
    dry_run: bool,
    buffer: DryRunBuffer,
) -> None:
    """Run the repo's inference entry point with ``--help`` via the dedicated venv.

    A missing venv python or a missing inference script is fatal — the
    environment is not usable if the CLI can't even import.  A non-zero exit
    from ``--help`` surfaces the raw stderr so the caller sees the real cause.
    """
    node_dir = _effective_node_dir(explicit_repo_dir)
    python_exe = _venv_python_for(node_dir)

    cli_script = _find_inference_script(node_dir) if not dry_run else None

    script_name = str(Path("inpainting_inference.py")) if cli_script is None else cli_script.name
    cmd = [str(python_exe), script_name, "--help"]
    label = f"{_cmd_line(cmd)}  (self-check)"
    if dry_run:
        buffer.record(label)
        return

    if not python_exe.is_file():
        raise RuntimeError(
            f"Dedicated venv python not found at {python_exe} — self-check cannot run. "
            "Re-run the bootstrap to (re)create the venv."
        )
    if cli_script is None:
        raise RuntimeError(
            f"No known inference script (inpainting_inference.py / "
            f"depth_splatting_inference.py / run.py / inference.py) found under {node_dir} "
            "— self-check cannot run. "
            "The StereoCrafter checkout may be incomplete; re-run the bootstrap."
        )

    log.info("▶ %s", label)
    try:
        result = subprocess.run(cmd, cwd=str(node_dir), capture_output=True, text=True, timeout=60)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Dedicated venv python not found at {python_exe}. Re-run the bootstrap.") from exc

    if result.returncode == 0:
        log.info("✓ Self-check passed — inference entry point loads cleanly.")
        return

    stderr = result.stderr.strip() or result.stdout.strip() or "<no output>"
    indented = "\n".join("    " + line for line in stderr[:800].splitlines())
    raise RuntimeError(
        f"Self-check FAILED: the inference entry point returned exit code {result.returncode}.\n  stderr:\n{indented}"
    )


def _cmd_line(cmd: list[str]) -> str:
    return " ".join(cmd)


# ---------------------------------------------------------------------------
# Final report + env-var hints
# ---------------------------------------------------------------------------


def print_summary(explicit_repo_dir: str | None) -> None:
    node_dir = _effective_node_dir(explicit_repo_dir)
    python_exe = _venv_python_for(node_dir)
    model_dir = _model_dir_for(explicit_repo_dir)
    svd_dir = INREPO_SVD_DIR

    log.info("")
    log.info("═" * 60)
    log.info("✓ StereoCrafter in-repo bootstrap complete.")
    log.info("═" * 60)
    log.info(
        "  repo_dir   = %s",
        node_dir,
    )
    log.info(
        "  python     = %s",
        python_exe,
    )
    log.info(
        "  model_dir  = %s",
        model_dir,
    )
    log.info(
        "  svd_dir    = %s",
        svd_dir,
    )
    log.info("")
    log.info(
        "These are the repo's DEFAULT paths, so the pipeline picks them up "
        "automatically.  You do NOT need to export them unless you want to "
        "override the defaults.  To export explicitly:"
    )
    log.info("")
    log.info('  export STEREOCRAFTER_REPO_DIR="%s"', node_dir)
    log.info('  export STEREOCRAFTER_PYTHON="%s"', python_exe)
    log.info('  export STEREOCRAFTER_CKPT_DIR="%s"', model_dir)
    log.info('  export STEREOCRAFTER_SVD_PATH="%s"', svd_dir)
    log.info('  #  Windows PowerShell: $env:STEREOCRAFTER_REPO_DIR="<...>"')
    log.info("")
    log.info("First inference run uses ~12 GB VRAM at the default resolution (512).")
    log.info("Tune with --stereocrafter-max-res (lower = less VRAM). See docs/STEREOCRAFTER_SETUP.md.")
    log.info("")
    log.info("Then run the pipeline with clean disocclusion inpainting:")
    log.info("  python scripts/run_pipeline.py --input video.mp4 --output vr180.mp4 --stereo-model stereocrafter")
    log.info("═" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deploy StereoCrafter inside this repo (CUDA-only, disocclusion inpainting). Idempotent.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/setup_stereocrafter.py              # full bootstrap\n"
            "  python scripts/setup_stereocrafter.py --skip-model # weights already present\n"
            "  python scripts/setup_stereocrafter.py --dry-run    # print planned steps only\n"
            "\n"
            "See docs/STEREOCRAFTER_SETUP.md for disk/VRAM requirements and troubleshooting.\n"
        ),
    )
    parser.add_argument(
        "--repo-dir",
        default=None,
        help=(
            "Path to an existing StereoCrafter checkout (e.g. D:/StereoCrafter). "
            "Defaults to the in-repo third_party/StereoCrafter directory."
        ),
    )
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="Skip model weight download (already downloaded).",
    )
    parser.add_argument(
        "--svd-dir",
        default=None,
        help=(
            "Override the SVD base model target directory.  Defaults to the "
            f"in-repo models/svd-img2vid-xt-1-1.  The SVD base "
            f"({_SVD_REPO_ID}, ~10 GB) is pre-downloaded by default (issue "
            "#150) — it is a GATED HF repo, so the local HF token is read "
            "automatically and the account must have accepted the license."
        ),
    )
    parser.add_argument(
        "--skip-svd",
        action="store_true",
        help=(
            "Skip the SVD base pre-download.  The pipeline will then pass the "
            "HF model id to diffusers, which downloads it on the first run "
            "(the historical behaviour — but a gated-repo 403 at runtime is "
            "less actionable than at bootstrap time)."
        ),
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help=(
            "Explicit Hugging Face access token for the gated SVD repo.  If "
            "omitted, the token is auto-read from the standard env vars "
            f"({', '.join(_HF_TOKEN_ENV_VARS)}) or huggingface_hub's stored "
            "login."
        ),
    )
    parser.add_argument(
        "--skip-deps",
        action="store_true",
        help="Skip venv creation and pip install (already set up).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the steps that would be executed and exit 0. Performs NO I/O.",
    )
    parser.add_argument(
        "--pip-mirror",
        default=None,
        help=(
            "Optional PyPI mirror URL passed as ``-i`` to pip "
            "(e.g. https://pypi.tuna.tsinghua.edu.cn/simple). "
            "torch is ALWAYS installed from the official cu124 index."
        ),
    )
    parser.add_argument(
        "--pip-timeout",
        type=int,
        default=None,
        help=(
            "Wall-clock timeout (seconds) for each pip install subprocess. "
            f"Default {DEFAULT_PIP_TIMEOUT}s ({DEFAULT_PIP_TIMEOUT // 60} min) — "
            "a downgrade + re-download on a slow network can exceed the old "
            "20-min cap (issue #162).  The {PIP_TIMEOUT_ENV_VAR} env var "
            "overrides the default; the CLI overrides the env var."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    args = parse_args(argv)

    buffer = DryRunBuffer()

    log.info("StereoCrafter in-repo bootstrap (repo=%s)", REPO_ROOT)
    if args.dry_run:
        log.info("[dry-run] no side effects will be performed")

    try:
        log.info("\n── Step 1/5: repo ──")
        ensure_node_repo(args.repo_dir, dry_run=args.dry_run, buffer=buffer)

        log.info("\n── Step 2/5: dedicated venv + pip install ──")
        if args.skip_deps:
            log.info("--skip-deps: venv + pip install skipped")
        else:
            ensure_venv_and_deps(
                args.repo_dir,
                args.pip_mirror,
                pip_timeout=resolve_pip_timeout(args.pip_timeout),
                dry_run=args.dry_run,
                buffer=buffer,
            )

        log.info("\n── Step 3/5: StereoCrafter weights ──")
        download_models(args.repo_dir, args.skip_model, dry_run=args.dry_run, buffer=buffer)

        log.info("\n── Step 4/5: SVD base model (gated HF repo) ──")
        download_svd_base(
            args.svd_dir,
            skip_svd=args.skip_svd,
            hf_token=args.hf_token,
            dry_run=args.dry_run,
            buffer=buffer,
        )

        log.info("\n── Step 5/5: self-check ──")
        self_check(args.repo_dir, dry_run=args.dry_run, buffer=buffer)
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        log.error("Bootstrap FAILED at: %s", exc)
        sys.exit(1)

    if args.dry_run:
        log.info("\n[dry-run] planned steps:")
        for i, step in enumerate(buffer.steps, start=1):
            log.info("  %d. %s", i, step)
    else:
        print_summary(args.repo_dir)


if __name__ == "__main__":
    main()
