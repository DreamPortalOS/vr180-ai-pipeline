"""CLI for generating videos via external providers (Kling / Seedance / Veo).

Usage::

    python -m scripts.generate "fly over mountains" --provider kling
    python -m scripts.generate "walkthrough of a temple" --provider seedance --target-aware --scene walkthrough
    python -m scripts.generate "dome flyover" --provider veo --duration 8 --aspect-ratio 16:9
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import httpx
from integrations.factory import get_provider, list_providers

log = logging.getLogger(__name__)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "video")


def _ensure_output_dir() -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR


def _download_video(url: str, out_path: str) -> str:
    """Download a video from *url* to *out_path* and return the path."""
    log.info("Downloading video from %s -> %s", url, out_path)
    with httpx.Client(timeout=300, follow_redirects=True) as client:
        resp = client.get(url)
        resp.raise_for_status()
        with open(out_path, "wb") as f:
            f.write(resp.content)
    log.info("Downloaded %d bytes", len(resp.content))
    return out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a video using an external AI provider.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Available providers: " + ", ".join(list_providers()) + "\n\n"
            "Environment variables:\n"
            "  KLING_API_KEY      API key for Kling\n"
            "  SEEDANCE_API_KEY   API key for Seedance\n"
            "  VEO_API_KEY        API key for Veo (Vertex AI)\n"
            "  GCP_PROJECT_ID     GCP project (Veo, defaults to 'my-project')\n"
        ),
    )
    parser.add_argument("prompt", type=str, help="Text description of the desired video.")
    parser.add_argument(
        "--provider",
        "-p",
        type=str,
        default="kling",
        choices=list_providers(),
        help="Video generation provider (default: kling).",
    )
    parser.add_argument(
        "--duration",
        "-d",
        type=int,
        default=5,
        help="Target duration in seconds (default: 5).",
    )
    parser.add_argument(
        "--aspect-ratio",
        "-a",
        type=str,
        default="16:9",
        help='Aspect ratio (default: "16:9").',
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=24,
        help="Target frame rate (default: 24).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file path (default: video/{provider}_{timestamp}.mp4).",
    )
    parser.add_argument(
        "--target-aware",
        action="store_true",
        help="Wrap the prompt through prompt_builder for target-aware VR180 constraints.",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="fpv",
        choices=["fpv", "walkthrough", "orbit", "static"],
        help="Scene type for prompt wrapping (default: fpv). Only used with --target-aware.",
    )
    parser.add_argument(
        "--list-providers",
        action="store_true",
        help="List available providers and exit.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.list_providers:
        print("Available providers:", ", ".join(list_providers()))
        return 0

    # Optional prompt wrapping
    prompt = args.prompt
    negative_prompt: str | None = None
    if args.target_aware:
        try:
            from pipeline.prompt_builder import wrap_prompt

            wrapped = wrap_prompt(prompt, scene_type=args.scene, target="vr180_flight")
            prompt = wrapped["positive"]
            negative_prompt = wrapped.get("negative")
            log.info(
                "Target-aware prompt wrapping applied (scene=%s). Positive length=%d, negative length=%d",
                args.scene,
                len(prompt),
                len(negative_prompt or ""),
            )
        except ImportError:
            log.warning("prompt_builder not available; using raw prompt")
        except Exception as exc:
            log.warning("Prompt wrapping failed: %s; using raw prompt", exc)

    # Get provider
    try:
        provider = get_provider(args.provider)
    except ValueError as exc:
        log.error("Provider error: %s", exc)
        return 1

    # Generate
    log.info(
        "Generating with provider=%s, duration=%d, aspect_ratio=%s, fps=%d",
        args.provider,
        args.duration,
        args.aspect_ratio,
        args.fps,
    )
    kwargs: dict[str, str | int | float] = {}
    if negative_prompt:
        kwargs["negative_prompt"] = negative_prompt

    try:
        result = provider.generate(
            prompt=prompt,
            duration=args.duration,
            aspect_ratio=args.aspect_ratio,
            fps=args.fps,
            **kwargs,
        )
    except RuntimeError as exc:
        log.error("Generation failed: %s", exc)
        return 1

    # Determine output path
    if args.output:
        out_path = args.output
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(
            _ensure_output_dir(),
            f"{args.provider}_{timestamp}.mp4",
        )

    # Download
    try:
        _download_video(result.video_url, out_path)
    except Exception as exc:
        log.error("Download failed: %s", exc)
        log.info("Video URL (download manually): %s", result.video_url)
        return 1

    print(f"\n✅ Video saved to: {out_path}")
    print(f"   Provider: {result.provider}")
    if result.job_id:
        print(f"   Job ID:   {result.job_id}")
    print(f"   URL:      {result.video_url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
