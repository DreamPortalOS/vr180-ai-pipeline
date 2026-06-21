#!/usr/bin/env python3
"""Download model weights for VR180 pipeline.

Usage:
    python scripts/download_models.py                     # All models
    python scripts/download_models.py --model depth       # Depth only
    python scripts/download_models.py --model midas       # MiDaS only
    python scripts/download_models.py --model all         # All
"""

import argparse
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("download-models")

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download model weights for VR180 pipeline"
    )
    parser.add_argument(
        "--model", "-m",
        default="all",
        choices=["all", "depth", "midas"],
        help="Which model to download (default: all)"
    )
    parser.add_argument(
        "--output-dir",
        default=MODELS_DIR,
        help="Directory to store models (default: models/)"
    )
    return parser.parse_args()


def download_depth_anything(output_dir: str):
    """Download Depth Anything V2 (Giant variant) from HuggingFace."""
    log.info("Downloading Depth Anything V2 (ViT-giant)...")
    try:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        repo = "depth-anything/Depth-Anything-V2-Giant"
        log.info(f"  Loading {repo}...")
        processor = AutoImageProcessor.from_pretrained(repo)
        model = AutoModelForDepthEstimation.from_pretrained(repo)

        save_dir = os.path.join(output_dir, "depth-anything-v2")
        os.makedirs(save_dir, exist_ok=True)
        processor.save_pretrained(save_dir)
        model.save_pretrained(save_dir)
        log.info(f"  ✅ Saved to {save_dir}")
    except ImportError:
        log.warning("  transformers not installed. Run: pip install transformers accelerate")
        raise
    except Exception as e:
        log.error(f"  ❌ Failed: {e}")
        raise


def download_midas(output_dir: str):
    """Download MiDaS 3.1 via torch.hub."""
    log.info("Downloading MiDaS 3.1...")
    try:
        import torch

        model = torch.hub.load("intel-isl/MiDaS", "MiDaS", trust_repo=True)
        save_dir = os.path.join(output_dir, "midas-3.1")
        os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(save_dir, "midas_v3.1.pth"))
        log.info(f"  ✅ Saved to {save_dir}")
    except Exception as e:
        log.error(f"  ❌ Failed: {e}")
        raise


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.model in ("all", "depth"):
        download_depth_anything(args.output_dir)

    if args.model in ("all", "midas"):
        download_midas(args.output_dir)

    log.info("Download complete!")


if __name__ == "__main__":
    main()