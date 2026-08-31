#!/usr/bin/env python3
"""One-command, in-repo StereoCrafter bootstrap (CUDA-only, disocclusion inpainting).

Deploys TencentARC/StereoCrafter **inside** this repo so the pipeline's
:mod:`pipeline.stereo_crafter` ``CLIBackend`` can pick it up automatically via
in-repo default paths — no separate ``D:/StereoCrafter`` install required.

Layout created (everything is gitignored):

    third_party/StereoCrafter/         ← git clone of TencentARC/StereoCrafter (incl. its own .venv)
    models/StereoCrafter/              ← TencentARC/StereoCrafter weights (hf snapshot)

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
    4. Self-check: run the repo's inference entry point with ``--help`` via the
       dedicated venv.

The script never touches the project-root venv and never downloads anything in CI
(the ``--dry-run`` flag is used by tests to assert the step sequence with zero I/O).

Usage::

    python scripts/setup_stereocrafter.py
    python scripts/setup_stereocrafter.py --repo-dir D:/StereoCrafter   # existing checkout
    python scripts/setup_stereocrafter.py --skip-model                  # weights already downloaded
    python scripts/setup_stereocrafter.py --skip-deps                   # venv + pip install already done
    python scripts/setup_stereocrafter.py --pip-mirror https://pypi.tuna.tsinghua.edu.cn/simple
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
RUNTIME_DEPS: tuple[str, ...] = (
    "diffusers",  # SD/SVD-based video diffusion backbone used for inpainting
    "transformers",  # pulled by diffusers models
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
    dry_run: bool,
    buffer: DryRunBuffer,
) -> None:
    """Create the dedicated venv and install torch (cu124) + the repo's deps."""
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
        f"torch=={TORCH_VERSION}",
        f"torchvision=={TORCHVISION_VERSION}",
        "--index-url",
        TORCH_INDEX_URL,
        *_pip_mirror_args(pip_mirror),
    ]
    run_step(cmd_torch, dry_run=dry_run, buffer=buffer, timeout=1200)

    base_cmd = [str(python_exe), "-m", "pip", "install", "--retries", "10"]
    mirror_args = _pip_mirror_args(pip_mirror)
    cmd_node = [*base_cmd, *RUNTIME_DEPS, *mirror_args]
    label_node = f"pip install {' '.join(RUNTIME_DEPS)}"
    run_step(cmd_node, dry_run=dry_run, buffer=buffer, timeout=1200, label=label_node)


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

    log.info("")
    log.info("═" * 60)
    log.info("✓ StereoCrafter in-repo bootstrap complete.")
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
    log.info('  export STEREOCRAFTER_REPO_DIR="%s"', node_dir)
    log.info('  export STEREOCRAFTER_PYTHON="%s"', python_exe)
    log.info('  export STEREOCRAFTER_CKPT_DIR="%s"', model_dir)
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
        log.info("\n── Step 1/4: repo ──")
        ensure_node_repo(args.repo_dir, dry_run=args.dry_run, buffer=buffer)

        log.info("\n── Step 2/4: dedicated venv + pip install ──")
        if args.skip_deps:
            log.info("--skip-deps: venv + pip install skipped")
        else:
            ensure_venv_and_deps(args.repo_dir, args.pip_mirror, dry_run=args.dry_run, buffer=buffer)

        log.info("\n── Step 3/4: model weights ──")
        download_models(args.repo_dir, args.skip_model, dry_run=args.dry_run, buffer=buffer)

        log.info("\n── Step 4/4: self-check ──")
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
