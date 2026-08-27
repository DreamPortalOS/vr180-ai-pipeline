#!/usr/bin/env python3
"""One-command, in-repo SeedVR2 bootstrap (ComfyUI-free standalone).

Deploys the SeedVR2 video upscaler **inside** this repo so the pipeline's
:mod:`pipeline.video_upscaler` ``CLIBackend`` can pick it up automatically via
in-repo default paths — no separate ``D:/ComfyUI`` install required.

Layout created (everything is gitignored):

    third_party/seedvr2_videoupscaler/   ← git clone of the node repo (incl. its own .venv)
    models/SEEDVR2/                      ← model weights

Steps (idempotent — re-running only fills missing pieces):

    1. Clone numz/ComfyUI-SeedVR2_VideoUpscaler (or ``git pull`` if present).
    2. Build a **dedicated** venv inside the node dir (never the project-root venv):
       - torch==2.6.0 + torchvision on the official cu124 index (stable, NOT nightly).
       - ``pip install -r requirements.txt``.
    3. Download the two required model weights into models/SEEDVR2/ (skip if >1 GB).
    4. Self-check: run ``inference_cli.py --help`` with the dedicated venv.
    5. Print the three env vars you can export (optional, since in-repo defaults work).

The script never touches the project-root venv and never downloads anything in CI
(the ``--dry-run`` flag is used by tests to assert the step sequence with zero I/O).

Usage::

    python scripts/setup_seedvr2.py
    python scripts/setup_seedvr2.py --skip-model      # weights already downloaded
    python scripts/setup_seedvr2.py --skip-deps       # venv + pip install already done
    python scripts/setup_seedvr2.py --pip-mirror https://pypi.tuna.tsinghua.edu.cn/simple
    python scripts/setup_seedvr2.py --dry-run         # print planned steps, no side effects
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

log = logging.getLogger("setup-seedvr2")

# ---------------------------------------------------------------------------
# Repo root: scripts/setup_seedvr2.py lives at <repo>/scripts/, so repo root
# is two parents up.  The CLIBackend in pipeline/video_upscaler.py uses the
# same convention (parent.parent of the pipeline/ package).
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent

NODE_REPO_URL = "https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git"
INREPO_NODE_DIR = REPO_ROOT / "third_party" / "seedvr2_videoupscaler"
INREPO_MODEL_DIR = REPO_ROOT / "models" / "SEEDVR2"

# Dedicated venv lives *inside* the node dir (isolated from the project venv).
INREPO_VENV_DIR = INREPO_NODE_DIR / ".venv"
INREPO_PYTHON = INREPO_VENV_DIR / (Path("Scripts") / "python.exe" if os.name == "nt" else Path("bin") / "python")

# Stable cu124 torch — the README's nightly cu130 is pinned off in favour of
# the release wheel, which is what the 12 GB RTX 4070S target needs.
TORCH_VERSION = "2.6.0"
TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu124"

# Model weights: numz/SeedVR2_comfyUI on HuggingFace.  Sizes are lead-verified.
_MODEL_REPO_ID = "numz/SeedVR2_comfyUI"
_MODEL_DIT = "seedvr2_ema_3b_fp8_e4m3fn.safetensors"  # ~3.2 GB
_MODEL_VAE = "ema_vae_fp16.safetensors"  # ~0.5 GB
_MODELS = [
    (_MODEL_DIT, "3.2 GB"),
    (_MODEL_VAE, "0.5 GB"),
]

# 1 GB skip threshold for existing models (avoids re-downloading huge files).
_MODEL_SKIP_SIZE = 1 * 1024 * 1024 * 1024


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
# Step 1: clone / pull the node repo
# ---------------------------------------------------------------------------


def ensure_node_repo(*, dry_run: bool, buffer: DryRunBuffer) -> None:
    """Clone the SeedVR2 node repo, or ``git pull`` if it already exists."""
    if INREPO_NODE_DIR.is_dir():
        if not _is_git_dir(INREPO_NODE_DIR):
            log.info("Node dir exists but is not a git checkout — re-cloning.")
        else:
            log.info("Node dir already exists at %s — pulling latest...", INREPO_NODE_DIR)
            run_step(
                ["git", "pull"],
                cwd=str(INREPO_NODE_DIR),
                dry_run=dry_run,
                buffer=buffer,
                label=f"git pull (in {INREPO_NODE_DIR})",
            )
            return

    third_party_dir = REPO_ROOT / "third_party"
    clone_cmd = ["git", "clone", NODE_REPO_URL, str(INREPO_NODE_DIR)]
    clone_label = f"git clone {NODE_REPO_URL} {INREPO_NODE_DIR}"
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

    log.info("Node repo cloned to %s", INREPO_NODE_DIR)


def _is_git_dir(path: Path) -> bool:
    return (path / ".git").exists()


def _proxy_hint(command: str) -> None:
    log.warning(
        "▸ %s failed — likely a network/proxy issue (common from mainland China).\n"
        "  Try setting your git/http proxy and re-run, e.g.:\n"
        "      git config --global http.proxy http://your-proxy:port\n"
        "      git config --global https.proxy http://your-proxy:port\n"
        "  Or clone manually:\n"
        "      git clone %s %s",
        command,
        NODE_REPO_URL,
        INREPO_NODE_DIR,
    )


# ---------------------------------------------------------------------------
# Step 2: dedicated venv + pip install
# ---------------------------------------------------------------------------


def _pip_mirror_args(pip_mirror: str | None) -> list[str]:
    """If a PyPI mirror is given, emit ``-i <url>``.  torch still uses --index-url."""
    if pip_mirror:
        return ["-i", pip_mirror]
    return []


def ensure_venv_and_deps(pip_mirror: str | None, *, dry_run: bool, buffer: DryRunBuffer) -> None:
    """Create the dedicated venv and install torch (cu124) + the node's requirements."""
    # In dry-run mode we always plan every step regardless of on-disk state, so
    # the recorded sequence is stable. In real mode, create the venv first if
    # missing (the torch/reqs pip installs use the venv's python).
    if not dry_run and not INREPO_PYTHON.is_file():
        venv_cmd = [sys.executable, "-m", "venv", str(INREPO_VENV_DIR)]
        log.info("Creating dedicated venv at %s (this may take a moment)...", INREPO_VENV_DIR)
        subprocess.check_call(venv_cmd, timeout=300)
    elif not dry_run and INREPO_PYTHON.is_file():
        log.info("Dedicated venv already exists at %s — re-installing requirements to be sure.", INREPO_VENV_DIR)

    if dry_run and not INREPO_PYTHON.is_file():
        # Plan the venv-creation step (only when it would actually be needed).
        run_step(
            [sys.executable, "-m", "venv", str(INREPO_VENV_DIR)],
            dry_run=True,
            buffer=buffer,
            label=f"{sys.executable} -m venv {INREPO_VENV_DIR}",
        )

    # (a) torch + torchvision on the stable cu124 index.
    cmd_torch = [
        str(INREPO_PYTHON),
        "-m",
        "pip",
        "install",
        "--retries",
        "10",
        f"torch=={TORCH_VERSION}",
        f"torchvision=={TORCH_VERSION}",
        "--index-url",
        TORCH_INDEX_URL,
        *_pip_mirror_args(pip_mirror),
    ]
    run_step(cmd_torch, dry_run=dry_run, buffer=buffer, timeout=1200)

    # (b) the node's own requirements.txt.
    requirements_txt = INREPO_NODE_DIR / "requirements.txt"
    if not dry_run and not requirements_txt.is_file():
        log.warning("requirements.txt not found at %s — skipping pip install of node deps.", requirements_txt)
        return

    cmd_reqs = [
        str(INREPO_PYTHON),
        "-m",
        "pip",
        "install",
        "--retries",
        "10",
        "-r",
        str(requirements_txt),
        *_pip_mirror_args(pip_mirror),
    ]
    run_step(cmd_reqs, dry_run=dry_run, buffer=buffer, timeout=1200)


# ---------------------------------------------------------------------------
# Step 3: model download
# ---------------------------------------------------------------------------


def _model_skip_existing(path: Path) -> bool:
    """Return True if *path* already exists and is bigger than the skip threshold."""
    return path.is_file() and path.stat().st_size > _MODEL_SKIP_SIZE


def _model_url(filename: str) -> str:
    return f"https://huggingface.co/{_MODEL_REPO_ID}/resolve/main/{filename}"


def _download_with_curl(filename: str, *, dry_run: bool, buffer: DryRunBuffer) -> None:
    """Resume-capable download: ``curl -L -C - -o <path> <url>`` (subprocess list form)."""
    url = _model_url(filename)
    out_path = INREPO_MODEL_DIR / filename
    cmd = ["curl", "-L", "-C", "-", "-f", "-o", str(out_path), url]
    label = f"curl -L -C - {url} → {out_path}"
    if dry_run:
        buffer.record(label)
        return
    INREPO_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    log.info("▶ %s", label)
    subprocess.check_call(cmd, timeout=3600)


def _download_with_hf_hub(filename: str, *, dry_run: bool, buffer: DryRunBuffer) -> None:
    """Preferred path: ``hf_hub_download`` into the in-repo model dir."""
    label = f"hf_hub_download {_MODEL_REPO_ID} {filename} → {INREPO_MODEL_DIR}"
    if dry_run:
        buffer.record(label)
        return

    INREPO_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        log.warning(
            "huggingface_hub not installed in this environment. "
            "Falling back to curl resume download. "
            "Install with: pip install huggingface_hub"
        )
        _download_with_curl(filename, dry_run=False, buffer=buffer)
        return

    log.info("▶ %s", label)
    out_path = Path(
        hf_hub_download(
            repo_id=_MODEL_REPO_ID,
            filename=filename,
            local_dir=str(INREPO_MODEL_DIR),
            local_dir_use_symlinks=False,
        )
    )
    log.info("Downloaded %s → %s", filename, out_path)


def download_models(skip_model: bool, *, dry_run: bool, buffer: DryRunBuffer) -> None:
    """Download the two required model weights; skip if already >1 GB on disk."""
    if skip_model:
        log.info("--skip-model: model download skipped")
        return

    if dry_run:
        INREPO_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for filename, human_size in _MODELS:
        target = INREPO_MODEL_DIR / filename
        if not dry_run and _model_skip_existing(target):
            log.info("Model already present (>1 GB): %s — skipping.", target)
            continue
        log.info("Downloading %s (~%s)…", filename, human_size)
        try:
            _download_with_hf_hub(filename, dry_run=dry_run, buffer=buffer)
        except Exception as exc:
            log.warning(
                "hf_hub_download failed for %s (%s) — falling back to curl resume.",
                filename,
                exc,
            )
            _download_with_curl(filename, dry_run=dry_run, buffer=buffer)


# ---------------------------------------------------------------------------
# Step 4: self-check
# ---------------------------------------------------------------------------


def self_check(*, dry_run: bool, buffer: DryRunBuffer) -> None:
    """Run ``inference_cli.py --help`` with the dedicated venv (exit 0 = good)."""
    if not dry_run and not INREPO_PYTHON.is_file():
        log.warning("Dedicated venv python not found at %s — skipping self-check.", INREPO_PYTHON)
        return
    cli_script = INREPO_NODE_DIR / "inference_cli.py"
    if not dry_run and not cli_script.is_file():
        log.warning("inference_cli.py not found — skipping self-check.")
        return

    cmd = [str(INREPO_PYTHON), "inference_cli.py", "--help"]
    label = f"{_cmd_line(cmd)}  (self-check)"
    if dry_run:
        buffer.record(label)
        return

    log.info("▶ %s", label)
    try:
        result = subprocess.run(cmd, cwd=str(INREPO_NODE_DIR), capture_output=True, text=True, timeout=60)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Dedicated venv python not found at {INREPO_PYTHON}. Re-run the bootstrap.") from exc

    if result.returncode == 0:
        log.info("✓ Self-check passed — inference_cli.py loads cleanly.")
    else:
        log.warning(
            "Self-check returned exit code %s (may be non-fatal).\n  stderr: %s",
            result.returncode,
            result.stderr.strip()[:500],
        )


def _cmd_line(cmd: list[str]) -> str:
    return " ".join(cmd)


# ---------------------------------------------------------------------------
# Final report + env-var hints
# ---------------------------------------------------------------------------


def print_summary() -> None:
    log.info("")
    log.info("═" * 60)
    log.info("✓ SeedVR2 in-repo bootstrap complete.")
    log.info("═" * 60)
    log.info(
        "  node_dir   = %s",
        INREPO_NODE_DIR,
    )
    log.info(
        "  python_exe = %s",
        INREPO_PYTHON,
    )
    log.info(
        "  model_dir  = %s",
        INREPO_MODEL_DIR,
    )
    log.info("")
    log.info(
        "These are the repo's DEFAULT paths, so the pipeline picks them up "
        "automatically.  You do NOT need to export them unless you want to "
        "override the defaults.  To export explicitly:"
    )
    log.info("")
    log.info('  export SEEDVR2_NODE_DIR="%s"', INREPO_NODE_DIR)
    log.info('  export SEEDVR2_PYTHON="%s"', INREPO_PYTHON)
    log.info('  export SEEDVR2_MODEL_DIR="%s"', INREPO_MODEL_DIR)
    log.info('  #  Windows PowerShell: $env:SEEDVR2_NODE_DIR="<...>"')
    log.info("")
    log.info("Then run the pipeline with:")
    log.info("  python scripts/run_pipeline.py --input video.mp4 --output vr180.mp4 --video-upscale seedvr2")
    log.info("═" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deploy SeedVR2 inside this repo (ComfyUI-free standalone). Idempotent.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/setup_seedvr2.py              # full bootstrap\n"
            "  python scripts/setup_seedvr2.py --skip-model # weights already present\n"
            "  python scripts/setup_seedvr2.py --dry-run    # print planned steps only\n"
            "\n"
            "See docs/SEEDVR2_SETUP.md for disk/VRAM requirements and troubleshooting.\n"
        ),
    )
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="Skip model weight download (already downloaded, >1 GB).",
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

    log.info("SeedVR2 in-repo bootstrap (repo=%s)", REPO_ROOT)
    if args.dry_run:
        log.info("[dry-run] no side effects will be performed")

    # Step 1 — node repo
    log.info("\n── Step 1/4: node repo ──")
    ensure_node_repo(dry_run=args.dry_run, buffer=buffer)

    # Step 2 — venv + deps
    log.info("\n── Step 2/4: dedicated venv + pip install ──")
    if args.skip_deps:
        log.info("--skip-deps: venv + pip install skipped")
    else:
        ensure_venv_and_deps(args.pip_mirror, dry_run=args.dry_run, buffer=buffer)

    # Step 3 — models
    log.info("\n── Step 3/4: model weights ──")
    download_models(args.skip_model, dry_run=args.dry_run, buffer=buffer)

    # Step 4 — self-check
    log.info("\n── Step 4/4: self-check ──")
    self_check(dry_run=args.dry_run, buffer=buffer)

    if args.dry_run:
        log.info("\n[dry-run] planned steps:")
        for i, step in enumerate(buffer.steps, start=1):
            log.info("  %d. %s", i, step)
    else:
        print_summary()


if __name__ == "__main__":
    main()
