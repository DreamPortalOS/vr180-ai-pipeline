#!/usr/bin/env python3
"""Automate SeedVR2 setup steps that can be done without a GUI.

Action items:
  1. Download ByteDance-Seed/SeedVR2-3B model(s) from HuggingFace Hub
     into ComfyUI's models/seedvr2 directory (or a custom path).
  2. git clone numz/ComfyUI-SeedVR2_VideoUpscaler into ComfyUI's
     custom_nodes directory.

ComfyUI *itself* is NOT installed by this script — the user must first
install ComfyUI manually (portable Windows build).  See docs/SEEDVR2_SETUP.md
for the full instructions.

Usage:
    python scripts/setup_seedvr2.py                     # uses defaults below
    python scripts/setup_seedvr2.py --comfy-dir D:/ComfyUI
    python scripts/setup_seedvr2.py --model-dir D:/ComfyUI/models/seedvr2
    python scripts/setup_seedvr2.py --skip-model         # already downloaded
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("setup-seedvr2")

# ---------------------------------------------------------------------------
# Default paths (Windows-centric, override via --comfy-dir)
# ---------------------------------------------------------------------------

_DEFAULT_COMFY_DIR = os.environ.get(
    "COMFYUI_DIR",
    str(Path.home() / "ComfyUI"),
)

_MODEL_REPO_ID = "ByteDance-Seed/SeedVR2-3B"
_NODE_REPO_URL = "https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(cmd: list[str], cwd: str | None = None) -> None:
    log.info("Running: %s", " ".join(cmd))
    subprocess.check_call(cmd, cwd=cwd)


# ---------------------------------------------------------------------------
# Model download
# ---------------------------------------------------------------------------


def download_model(model_dir: str) -> None:
    """Download ByteDance-Seed/SeedVR2-3B model into *model_dir*."""
    os.makedirs(model_dir, exist_ok=True)
    log.info("Downloading %s → %s", _MODEL_REPO_ID, model_dir)

    # Check if huggingface_hub is available
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        log.warning("huggingface_hub not installed. Install it with: pip install huggingface_hub")
        log.info("Falling back to manual curl/git-lfs instructions.")
        log.info(
            "  Open: https://huggingface.co/%s\n"
            "  Download seedvr2_ema_3b_fp16.safetensors (or seedvr2_ema_3b-Q4_K_M.gguf)\n"
            "  and place in: %s",
            _MODEL_REPO_ID,
            model_dir,
        )
        return

    snapshot_download(
        repo_id=_MODEL_REPO_ID,
        local_dir=model_dir,
        local_dir_use_symlinks=False,
        ignore_patterns=["*.md", "*.txt"],
    )
    log.info("Model downloaded to %s", model_dir)


# ---------------------------------------------------------------------------
# Custom node clone
# ---------------------------------------------------------------------------


def install_custom_node(comfy_dir: str) -> None:
    """git clone the SeedVR2 custom node into ComfyUI/custom_nodes/."""
    custom_nodes_dir = os.path.join(comfy_dir, "custom_nodes")
    node_dir = os.path.join(custom_nodes_dir, "ComfyUI-SeedVR2_VideoUpscaler")

    if os.path.isdir(node_dir):
        log.info("Custom node already exists at %s — pulling latest...", node_dir)
        _run(["git", "pull"], cwd=node_dir)
        return

    os.makedirs(custom_nodes_dir, exist_ok=True)
    log.info("Cloning %s → %s", _NODE_REPO_URL, node_dir)
    _run(
        ["git", "clone", _NODE_REPO_URL, node_dir],
    )
    log.info("Custom node installed at %s", node_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automate SeedVR2 assets setup (models + custom node).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python scripts/setup_seedvr2.py --comfy-dir D:/ComfyUI\n"
            "\n"
            "For the full manual installation guide see docs/SEEDVR2_SETUP.md.\n"
        ),
    )
    parser.add_argument(
        "--comfy-dir",
        default=_DEFAULT_COMFY_DIR,
        help=f"ComfyUI root directory (default: {_DEFAULT_COMFY_DIR})",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help=(
            "SeedVR2 model directory (default: <comfy-dir>/models/seedvr2; "
            "or ComfyUI-Manager layout <comfy-dir>/models/LLM/seedvr2 if preferred)"
        ),
    )
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="Skip model download (already downloaded)",
    )
    parser.add_argument(
        "--skip-node",
        action="store_true",
        help="Skip custom node clone (already installed)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Verify ComfyUI dir exists (even minimally)
    comfy_dir = args.comfy_dir
    if not os.path.isdir(comfy_dir):
        log.warning(
            "ComfyUI directory not found at '%s'.\n"
            "Please install ComfyUI first (portable Windows build):\n"
            "  https://github.com/comfyanonymous/ComfyUI\n"
            "Then re-run this script with --comfy-dir pointing to it.\n"
            "Proceeding anyway (model/node downloads will still work).",
            comfy_dir,
        )

    # Determine model directory
    model_dir = args.model_dir or os.path.join(comfy_dir, "models", "seedvr2")

    if not args.skip_model:
        download_model(model_dir)
    else:
        log.info("--skip-model: model download skipped")

    if not args.skip_node:
        install_custom_node(comfy_dir)
    else:
        log.info("--skip-node: custom node clone skipped")

    log.info("")
    log.info("═" * 50)
    log.info("Setup complete!  Next steps (manual):")
    log.info("  1. Start ComfyUI (run main.py or the portable .bat)")
    log.info("  2. Verify the SeedVR2 custom node appears in the UI")
    log.info("  3. Load / create a SeedVR2 workflow in ComfyUI")
    log.info("  4. Run the pipeline with: --video-upscale seedvr2")
    log.info("═" * 50)


if __name__ == "__main__":
    main()
