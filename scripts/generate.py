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
    """Copy the video at *url* to *out_path* and return the path.

    *url* is normally an ``http(s)`` URL, in which case the bytes are
    fetched over the network.  Some providers (notably the local mock
    provider) return a local file path as ``video_url``; in that case the
    file is copied rather than fetched.  This keeps the full chain runnable
    on CI without keys or network access.
    """
    if _is_local_path(url):
        log.info("Copying local video %s -> %s", url, out_path)
        if not os.path.exists(url):
            raise RuntimeError(f"Video file not found: {url}")
        with open(url, "rb") as src, open(out_path, "wb") as dst:
            dst.write(src.read())
        log.info("Copied %d bytes", os.path.getsize(out_path))
        return out_path
    log.info("Downloading video from %s -> %s", url, out_path)
    with httpx.Client(timeout=300, follow_redirects=True) as client:
        resp = client.get(url)
        resp.raise_for_status()
        with open(out_path, "wb") as f:
            f.write(resp.content)
    log.info("Downloaded %d bytes", len(resp.content))
    return out_path


def _is_local_path(value: str) -> bool:
    """Return True if *value* looks like a local filesystem path."""
    return not (value.startswith("http://") or value.startswith("https://"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a video using an external AI provider.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Available providers: " + ", ".join(list_providers()) + "\n\n"
            "Environment variables:\n"
            "  KLING_API_KEY      API key for Kling\n"
            "  ARK_API_KEY        API key for Seedance (Volcengine Ark)\n"
            "  VEO_API_KEY        API key for Veo (Vertex AI)\n"
            "  GCP_PROJECT_ID     GCP project (Veo, defaults to 'my-project')\n"
        ),
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default="",
        help="Text description of the desired video. Optional when --image is given.",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path or URL to a starting image; switches to image-to-video (generate_from_image).",
    )
    parser.add_argument(
        "--provider",
        "-p",
        type=str,
        default="kling",
        choices=list_providers(),
        help="Video generation provider (default: kling). Use 'mock' to run fully offline.",
    )
    parser.add_argument(
        "--duration",
        "-d",
        type=int,
        default=5,
        help="Target duration in seconds (default: 5).",
    )
    parser.add_argument(
        "--gen-resolution",
        default="480p",
        choices=["480p", "720p", "1080p"],
        help="Generation resolution tier (default: 480p — quota discipline). "
        "Higher tiers consume more quota; a reminder is logged when >480p is selected.",
    )
    parser.add_argument(
        "--gen-ratio",
        default="adaptive",
        help="Generation aspect ratio passed through to the provider (default: adaptive). "
        "Seedance accepts e.g. adaptive/16:9/9:16/1:1.",
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
    parser.add_argument(
        "--template",
        type=str,
        default=None,
        metavar="KEY",
        help="Render the prompt body from a prompt_library template (e.g. slow_dolly_in). "
        "Mutually exclusive with the positional prompt argument. "
        "Use --list-templates to see available keys.",
    )
    parser.add_argument(
        "--template-field",
        action="append",
        default=None,
        metavar="NAME=VALUE",
        help="Fill a template placeholder. Repeatable. Required when --template is given; "
        "each value must be NAME=VALUE. Ignored without --template.",
    )
    parser.add_argument(
        "--list-templates",
        action="store_true",
        help="List available prompt_library templates (key, summary, placeholders, recommended duration) and exit.",
    )
    return parser


def _parse_template_fields(raw: list[str] | None) -> dict[str, str]:
    """Parse repeated ``--template-field NAME=VALUE`` into a dict.

    Each entry must contain exactly one ``=``.  An entry missing the ``=``
    separator is reported by name so the user knows which one is malformed.
    """
    fields: dict[str, str] = {}
    if not raw:
        return fields
    for item in raw:
        if "=" not in item:
            raise ValueError(f"--template-field expects NAME=VALUE, got {item!r} (missing '=')")
        name, _, value = item.partition("=")
        name = name.strip()
        if not name:
            raise ValueError(f"--template-field expects NAME=VALUE, got {item!r} (empty name)")
        fields[name] = value
    return fields


def _list_templates() -> int:
    """Print every prompt_library template and return exit code 0."""
    from pipeline.prompt_library import list_templates

    print("Available prompt templates:")
    for t in list_templates():
        placeholders = ", ".join(t.placeholders) if t.placeholders else "(none)"
        print(f"  {t.key}")
        print(f"      summary : {t.summary}")
        print(f"      motion  : {t.camera_motion}")
        print(f"      fields  : {placeholders}")
        print(f"      duration: {t.recommended_duration:g}s")
    return 0


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

    if args.list_templates:
        return _list_templates()

    # --template and the positional prompt are mutually exclusive; at least
    # one of them (or --image) must supply the prompt text.
    if args.template and args.prompt:
        log.error("--template and a positional prompt are mutually exclusive.")
        return 2

    prompt = args.prompt
    negative_prompt: str | None = None

    # Render the prompt body from a template first (if requested), *then*
    # apply --target-aware wrapping. The two layers are composable, not
    # alternative: the template decides "what to say", wrap_prompt decides
    # "what target-projection suffix to append".
    if args.template:
        try:
            fields = _parse_template_fields(args.template_field)
        except ValueError as exc:
            log.error("%s", exc)
            return 2
        try:
            from pipeline.prompt_library import (
                PromptLibraryError,
                get_template,
                render,
            )

            tmpl = get_template(args.template)
            prompt = render(args.template, **fields)
            log.info(
                "使用模板 %s（镜头运动 %s，推荐时长 %gs）",
                args.template,
                tmpl.camera_motion,
                tmpl.recommended_duration,
            )
        except KeyError as exc:
            # Unknown template key — message already lists available keys.
            log.error("%s", exc)
            return 2
        except PromptLibraryError as exc:
            # Missing / extra placeholder — surface the message verbatim
            # (it names the offending fields) instead of swallowing it.
            log.error("%s", exc)
            return 2

    # Optional prompt wrapping
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

    kwargs: dict[str, str | int | float] = {}
    if negative_prompt:
        kwargs["negative_prompt"] = negative_prompt
    # H-2: generation-tier passthrough. Seedance reads resolution/ratio from
    # the request body; other providers ignore unknown kwargs. Default stays
    # 480p/adaptive so the quota discipline is unchanged.
    kwargs["resolution"] = args.gen_resolution
    kwargs["ratio"] = args.gen_ratio
    if args.gen_resolution != "480p":
        log.warning("⚠️  高档位（%s）消耗更多额度，请确认后再继续。", args.gen_resolution)

    # Generate — text-to-video or image-to-video
    if args.image:
        if not prompt:
            log.warning("--image given without a prompt; using empty prompt")
        log.info(
            "Image-to-video with provider=%s, image=%s, duration=%d, aspect_ratio=%s, fps=%d",
            args.provider,
            args.image,
            args.duration,
            args.aspect_ratio,
            args.fps,
        )
        try:
            result = provider.generate_from_image(
                image_path=args.image,
                prompt=prompt,
                duration=args.duration,
                aspect_ratio=args.aspect_ratio,
                fps=args.fps,
                **kwargs,
            )
        except NotImplementedError as exc:
            log.error("Provider %s does not support image-to-video: %s", args.provider, exc)
            return 1
        except RuntimeError as exc:
            log.error("Generation failed: %s", exc)
            return 1
    else:
        if not prompt:
            log.error("A prompt (or --image) is required.")
            return 2
        log.info(
            "Generating with provider=%s, duration=%d, aspect_ratio=%s, fps=%d",
            args.provider,
            args.duration,
            args.aspect_ratio,
            args.fps,
        )
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
