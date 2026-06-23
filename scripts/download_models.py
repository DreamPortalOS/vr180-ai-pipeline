"""Download Depth Anything V2 model weights for VR180 pipeline.

Usage:
    python scripts/download_models.py                     # All models
    python scripts/download_models.py --model depth       # Depth only
    python scripts/download_models.py --model all         # All
"""

import argparse
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("download-models")

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

# HuggingFace repos for Depth Anything V2
MODEL_REPOS = {
    "small": "depth-anything/Depth-Anything-V2-Small-hf",
    "base": "depth-anything/Depth-Anything-V2-Base-hf",
    "large": "depth-anything/Depth-Anything-V2-Large-hf",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Download model weights for VR180 pipeline")
    parser.add_argument(
        "--model",
        "-m",
        default="small",
        choices=["all", "small", "base", "large"],
        help="Which model to download (default: small)",
    )
    parser.add_argument("--output-dir", default=MODELS_DIR, help="Directory to store models (default: models/)")
    return parser.parse_args()


def download_model(size: str, output_dir: str):
    """Download a Depth Anything V2 model from HuggingFace.

    Uses transformers to download and cache the model, then saves
    it to the specified output directory for offline use.
    """
    repo = MODEL_REPOS[size]
    log.info(f"Downloading Depth Anything V2 ({size}) from {repo}...")
    try:
        from transformers import pipeline

        # Create the pipeline to trigger download and caching
        pipe = pipeline(
            task="depth-estimation",
            model=repo,
        )

        # Save model to the specified output directory
        save_dir = os.path.join(output_dir, f"depth-anything-v2-{size}")
        os.makedirs(save_dir, exist_ok=True)

        # Save the model and processor for offline use
        pipe.model.save_pretrained(save_dir)
        pipe.processor.save_pretrained(save_dir)
        log.info(f"  ✅ Model saved to {save_dir}")

    except Exception as e:
        log.error(f"  ❌ Failed: {e}")
        raise


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.model == "all":
        for size in MODEL_REPOS:
            download_model(size, args.output_dir)
    else:
        download_model(args.model, args.output_dir)

    log.info("Download complete!")


if __name__ == "__main__":
    main()
