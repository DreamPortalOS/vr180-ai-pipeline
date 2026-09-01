#!/usr/bin/env python3
"""One-command, in-repo DepthCrafter bootstrap (CUDA-only, temporally-consistent depth).

Deploys Tencent/DepthCrafter **inside** this repo so the pipeline's
:mod:`pipeline.depth_crafter` ``CLIBackend`` can pick it up automatically via
in-repo default paths — no separate ``D:/DepthCrafter`` install required.

Layout created (everything is gitignored):

    third_party/DepthCrafter/          ← git clone of Tencent/DepthCrafter (incl. its own .venv)
    models/DepthCrafter/               ← tencent/DepthCrafter weights (diffusers snapshot)

Steps (idempotent — re-running only fills missing pieces):

    1. Clone Tencent/DepthCrafter (or ``git pull`` if present).
       ``--repo-dir`` can point at an existing checkout instead (e.g. ``D:/DepthCrafter``).
    2. Build a **dedicated** venv inside the node dir (never the project-root venv):
       - torch==2.6.0 + torchvision==0.21.0 on the official cu124 index (stable, NOT nightly).
       - Node runtime deps: a curated subset (``RUNTIME_DEPS``) of the packages
         ``run.py`` actually imports.  We deliberately do NOT run ``pip install
         -e .`` against the upstream pyproject — it is a flat-layout (setuptools
         rejects it with "Multiple top-level packages"), declares
         ``requires-python >= 3.13`` and pins torch>=2.7.1/torchvision>=0.22.1,
         which would conflict with our Python 3.12 + pinned cu124 torch.
         See docs/DEPTHCRAFTER_SETUP.md for the full rationale.
    3. Download ``tencent/DepthCrafter`` weights into ``models/DepthCrafter/`` via
       ``huggingface_hub.snapshot_download`` (existing download is skipped).
       The base model ``stabilityai/stable-video-diffusion-img2vid-xt`` is pulled
       automatically by diffusers on first inference run (~10 GB total; see the doc).
    4. Self-check: run ``run.py --help`` with the dedicated venv.

The script never touches the project-root venv and never downloads anything in CI
(the ``--dry-run`` flag is used by tests to assert the step sequence with zero I/O).

Usage::

    python scripts/setup_depthcrafter.py
    python scripts/setup_depthcrafter.py --repo-dir D:/DepthCrafter   # existing checkout
    python scripts/setup_depthcrafter.py --skip-model                  # weights already downloaded
    python scripts/setup_depthcrafter.py --skip-deps                   # venv + pip install already done
    python scripts/setup_depthcrafter.py --pip-mirror https://pypi.tuna.tsinghua.edu.cn/simple
    python scripts/setup_depthcrafter.py --dry-run                     # print planned steps, no side effects
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

log = logging.getLogger("setup-depthcrafter")

# ---------------------------------------------------------------------------
# Repo root: scripts/setup_depthcrafter.py lives at <repo>/scripts/, so repo
# root is two parents up.  The CLIBackend in pipeline/depth_crafter.py uses the
# same convention (parent.parent of the pipeline/ package).
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent

NODE_REPO_URL = "https://github.com/Tencent/DepthCrafter.git"
INREPO_NODE_DIR = REPO_ROOT / "third_party" / "DepthCrafter"
INREPO_MODEL_DIR = REPO_ROOT / "models" / "DepthCrafter"

# Dedicated venv lives *inside* the node dir (isolated from the project venv).
INREPO_VENV_DIR = INREPO_NODE_DIR / ".venv"
INREPO_PYTHON = INREPO_VENV_DIR / (Path("Scripts") / "python.exe" if os.name == "nt" else Path("bin") / "python")

# Stable cu124 torch — paired torchvision release (torchvision==2.6.0 does not exist).
TORCH_VERSION = "2.6.0"
TORCHVISION_VERSION = "0.21.0"
TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu124"

# Default subprocess timeout for the pip install steps (issue #166, mirroring
# the #165 fix on setup_stereocrafter.py).  pip degrades + dependency-tree
# resolution can take ~20 minutes on mainland-China networks, exceeding the old
# 1200s ceiling and aborting the whole bootstrap.  Bumped to 1 hour; override
# via --pip-timeout or the SETUP_PIP_TIMEOUT env var.  The pip *connection*
# timeout (--timeout 120) is separate: it governs a single socket operation,
# not the whole install.
DEFAULT_PIP_TIMEOUT: int = 3600

# pip's own single-connection timeout (--timeout N) used to improve success on
# weak networks (issue #166, mirroring #165).  Distinct from the
# subprocess-level timeout (DEFAULT_PIP_TIMEOUT above): that one bounds the
# whole install; this one bounds a single socket handshake/read.
PIP_CONNECT_TIMEOUT: int = 120

# Curated runtime deps that run.py actually imports.
#
# Deliberately NOT a full install of the upstream pyproject.toml.  Reasons:
#   1. The upstream repo is a flat-layout (``depthcrafter/`` + ``visualization/``
#      top-level packages), so ``pip install -e .`` fails with
#      "Multiple top-level packages discovered in a flat-layout".  DepthCrafter
#      is run as a script (``run.py``), so it never needs to be installed as a
#      package — the curated list is all the CLI needs.
#   2. The upstream pyproject declares ``requires-python = ">=3.13"`` and pins
#      ``torch>=2.7.1``, ``torchvision>=0.22.1``, ``xformers``, ``gradio`` and
#      ``decord``.  Our dedicated venv runs Python 3.12 and pins
#      ``torch==2.6.0`` / ``torchvision==0.21.0`` (cu124) in Step 2 — following
#      the upstream declaration wholesale would either bump torch (breaking the
#      cu124 pairing) or fail outright on Python 3.12.
#   3. ``gradio`` / ``xformers`` / ``pytest`` / ``matplotlib`` are demo/dev
#      dependencies and are intentionally left out; ``decord`` is present as a
#      hard video-loader dependency that run.py may require at load time.
#
# torch / torchvision are intentionally NOT here — they are pinned in Step 2
# against the cu124 index so they can never be bumped by a transitive dep.
RUNTIME_DEPS: tuple[str, ...] = (
    "fire",  # run.py's fire-style CLI
    "diffusers",  # DepthCrafter pipeline weights + inference
    "transformers",  # pulled by diffusers models
    "accelerate",  # used by diffusers SVD loader
    "huggingface-hub",  # weight download / caching
    "mediapy",  # video I/O (numpy-media)
    "decord",  # video loader (used at load time)
)

# HuggingFace repo for the diffusers weights.
_MODEL_REPO_ID = "tencent/DepthCrafter"
# The base SVD model is pulled automatically by diffusers on first inference run.
_BASE_MODEL_REPO_ID = "stabilityai/stable-video-diffusion-img2vid-xt"


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
    """Print *cmd*, then either record it (dry-run) or execute it."""
    label = label or " ".join(cmd)
    if dry_run:
        buffer.record(label)
        return
    log.info("▶ %s", label)
    subprocess.check_call(cmd, cwd=cwd, timeout=timeout)


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
    """Clone the DepthCrafter repo, ``git pull`` if present, or use --repo-dir."""
    node_dir = _effective_node_dir(explicit_repo_dir)

    # If an existing checkout is pointed at, treat it as authoritative (pull if it's a git dir).
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

    # In-repo path: clone/pull as usual.
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
    dry_run: bool,
    buffer: DryRunBuffer,
    pip_timeout: int = DEFAULT_PIP_TIMEOUT,
) -> None:
    """Create the dedicated venv and install torch (cu124) + the repo's deps.

    *pip_timeout* is the subprocess-level timeout (issue #166, mirroring #165):
    the maximum wall time allowed for each pip install.  Defaults to
    DEFAULT_PIP_TIMEOUT (3600s) so all existing call sites remain valid
    without change.  Override via ``--pip-timeout`` on the CLI or the
    SETUP_PIP_TIMEOUT env var (CLI wins > env > default).
    """
    node_dir = _effective_node_dir(explicit_repo_dir)
    venv_dir = node_dir / ".venv"
    python_exe = _venv_python_for(node_dir)

    # In real mode, create the venv first if missing (the pip installs use its python).
    if not dry_run:
        if not python_exe.is_file():
            venv_cmd = [sys.executable, "-m", "venv", str(venv_dir)]
            log.info("Creating dedicated venv at %s (this may take a moment)...", venv_dir)
            subprocess.check_call(venv_cmd, timeout=300)
        else:
            log.info("Dedicated venv already exists at %s — re-installing runtime deps.", venv_dir)

    # In dry-run mode, always plan the venv-creation step (it would be needed if the venv is absent).
    if dry_run:
        run_step(
            [sys.executable, "-m", "venv", str(venv_dir)],
            dry_run=True,
            buffer=buffer,
            label=f"{sys.executable} -m venv {venv_dir}",
        )

    # (a) torch + torchvision on the stable cu124 index.
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

    # (b) the curated runtime deps for run.py.
    #
    # See the RUNTIME_DEPS constant above for why we install a curated subset
    # instead of ``pip install -e .`` from the upstream pyproject.  Notably:
    # no -e (DepthCrafter is run as a script, never installed as a package),
    # and no torch/torchvision (those are pinned in step (a) so a transitive
    # dep can never bump them off the cu124 pairing).
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
    """Where the tencent/DepthCrafter weights land.  --repo-dir always uses models/DepthCrafter."""
    return INREPO_MODEL_DIR


def download_models(
    explicit_repo_dir: str | None,
    skip_model: bool,
    *,
    dry_run: bool,
    buffer: DryRunBuffer,
) -> None:
    """Download tencent/DepthCrafter via snapshot_download; skip if already present."""
    if skip_model:
        log.info("--skip-model: model download skipped")
        return

    model_dir = _model_dir_for(explicit_repo_dir)

    label = (
        f"snapshot_download {_MODEL_REPO_ID} → {model_dir}"
        f"  (base {_BASE_MODEL_REPO_ID} auto-downloaded by diffusers on first run)"
    )
    if dry_run:
        buffer.record(label)
        return

    if model_dir.is_dir() and _has_snapshot_files(model_dir):
        log.info("Model snapshot already present at %s — skipping.", model_dir)
        return

    model_dir.mkdir(parents=True, exist_ok=True)
    log.info("▶ %s", label)
    log.info(
        "  Note: the base model %s is NOT downloaded here — it is pulled automatically",
        _BASE_MODEL_REPO_ID,
    )
    log.info(
        "  by diffusers on the first inference run (~10 GB total). Do not set HF_ENDPOINT"
        " mirror (direct HF access is required)."
    )

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
# Step 4: self-check
# ---------------------------------------------------------------------------


def self_check(
    explicit_repo_dir: str | None,
    *,
    dry_run: bool,
    buffer: DryRunBuffer,
) -> None:
    """Run ``run.py --help`` with the dedicated venv.

    A non-zero exit, a missing venv python, or a missing ``run.py`` is fatal —
    the environment is not usable if the CLI can't even import.  The raw stderr
    is surfaced so the caller (and the user) can see the real reason.
    """
    node_dir = _effective_node_dir(explicit_repo_dir)
    python_exe = _venv_python_for(node_dir)
    cli_script = node_dir / "run.py"

    cmd = [str(python_exe), "run.py", "--help"]
    label = f"{_cmd_line(cmd)}  (self-check)"
    if dry_run:
        buffer.record(label)
        return

    if not python_exe.is_file():
        raise RuntimeError(
            f"Dedicated venv python not found at {python_exe} — self-check cannot run. "
            "Re-run the bootstrap to (re)create the venv."
        )
    if not cli_script.is_file():
        raise RuntimeError(
            f"run.py not found at {cli_script} — self-check cannot run. "
            "The DepthCrafter checkout is incomplete; re-run the bootstrap."
        )

    log.info("▶ %s", label)
    try:
        result = subprocess.run(cmd, cwd=str(node_dir), capture_output=True, text=True, timeout=60)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Dedicated venv python not found at {python_exe}. Re-run the bootstrap.") from exc

    if result.returncode == 0:
        log.info("✓ Self-check passed — run.py loads cleanly.")
        return

    # Non-zero: the environment is broken. Surface the real stderr and fail hard.
    stderr = result.stderr.strip() or result.stdout.strip() or "<no output>"
    indented = "\n".join("    " + line for line in stderr[:800].splitlines())
    raise RuntimeError(
        f"Self-check FAILED: run.py --help returned exit code {result.returncode}.\n  stderr:\n{indented}"
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

    log.info("")
    log.info("═" * 60)
    log.info("✓ DepthCrafter in-repo bootstrap complete.")
    log.info("═" * 60)
    log.info(
        "  repo_dir  = %s",
        node_dir,
    )
    log.info(
        "  python    = %s",
        python_exe,
    )
    log.info(
        "  model_dir = %s",
        model_dir,
    )
    log.info("")
    log.info(
        "These are the repo's DEFAULT paths, so the pipeline picks them up "
        "automatically.  You do NOT need to export them unless you want to "
        "override the defaults.  To export explicitly:"
    )
    log.info("")
    log.info('  export DEPTHCRAFTER_REPO_DIR="%s"', node_dir)
    log.info('  export DEPTHCRAFTER_PYTHON="%s"', python_exe)
    log.info('  export DEPTHCRAFTER_MODEL_DIR="%s"', model_dir)
    log.info('  #  Windows PowerShell: $env:DEPTHCRAFTER_REPO_DIR="<...>"')
    log.info("")
    log.info("First inference run will auto-download the base model (~10 GB):")
    log.info("  stabilityai/stable-video-diffusion-img2vid-xt")
    log.info("")
    log.info("Then run the pipeline with temporally-consistent depth:")
    log.info("  python scripts/run_pipeline.py --input video.mp4 --output vr180.mp4 --depth-model depthcrafter")
    log.info("═" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deploy DepthCrafter inside this repo (CUDA-only, temporally-consistent depth). Idempotent.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/setup_depthcrafter.py              # full bootstrap\n"
            "  python scripts/setup_depthcrafter.py --skip-model # weights already present\n"
            "  python scripts/setup_depthcrafter.py --dry-run    # print planned steps only\n"
            "\n"
            "See docs/DEPTHCRAFTER_SETUP.md for disk/VRAM requirements and troubleshooting.\n"
        ),
    )
    parser.add_argument(
        "--repo-dir",
        default=None,
        help=(
            "Path to an existing DepthCrafter checkout (e.g. D:/DepthCrafter). "
            "Defaults to the in-repo third_party/DepthCrafter directory."
        ),
    )
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="Skip model weight download (already downloaded).",
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
        default=None,
        type=int,
        metavar="SECONDS",
        help=(
            f"Subprocess timeout for pip install in seconds.  Defaults to "
            f"{DEFAULT_PIP_TIMEOUT}s (issue #166); override via the "
            f"SETUP_PIP_TIMEOUT env var.  CLI arg wins > env var > default."
        ),
    )
    return parser.parse_args(argv)


def _resolve_pip_timeout(cli_value: int | None) -> int:
    """Resolve the pip subprocess timeout with the required precedence.

    Order: --pip-timeout (CLI) > SETUP_PIP_TIMEOUT (env) > DEFAULT_PIP_TIMEOUT.
    Mirrors the #165 implementation on setup_stereocrafter.py so all three
    setup scripts share the same naming and precedence.
    """
    if cli_value is not None:
        return cli_value
    env_value = os.environ.get("SETUP_PIP_TIMEOUT")
    if env_value is not None:
        try:
            return int(env_value)
        except ValueError:
            log.warning(
                "SETUP_PIP_TIMEOUT=%r is not a valid integer — falling back to "
                "the default (%ds).  Pass a plain integer or use --pip-timeout.",
                env_value,
                DEFAULT_PIP_TIMEOUT,
            )
    return DEFAULT_PIP_TIMEOUT


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    args = parse_args(argv)

    buffer = DryRunBuffer()
    pip_timeout = _resolve_pip_timeout(args.pip_timeout)

    log.info("DepthCrafter in-repo bootstrap (repo=%s)", REPO_ROOT)
    if args.dry_run:
        log.info("[dry-run] no side effects will be performed")

    try:
        # Step 1 — node repo
        log.info("\n── Step 1/4: repo ──")
        ensure_node_repo(args.repo_dir, dry_run=args.dry_run, buffer=buffer)

        # Step 2 — venv + deps
        log.info("\n── Step 2/4: dedicated venv + pip install ──")
        if args.skip_deps:
            log.info("--skip-deps: venv + pip install skipped")
        else:
            ensure_venv_and_deps(
                args.repo_dir,
                args.pip_mirror,
                dry_run=args.dry_run,
                buffer=buffer,
                pip_timeout=pip_timeout,
            )

        # Step 3 — models
        log.info("\n── Step 3/4: model weights ──")
        download_models(args.repo_dir, args.skip_model, dry_run=args.dry_run, buffer=buffer)

        # Step 4 — self-check
        log.info("\n── Step 4/4: self-check ──")
        self_check(args.repo_dir, dry_run=args.dry_run, buffer=buffer)
    except subprocess.TimeoutExpired as exc:
        log.warning(
            "▸ pip install timed out after %ss. 网络慢可加 --pip-mirror <url> 或直接重跑（已完成步骤会跳过）.",
            exc.timeout,
        )
        log.error("Bootstrap FAILED at: %s", exc)
        sys.exit(1)
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
