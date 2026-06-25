#!/usr/bin/env python3
"""Prompt Lab — offline batch prompt variant generator.

Generates prompt variants across output targets × scene types
by calling pipeline.prompt_builder.wrap_prompt() for each combination.
Results are collected into a JSON manifest for offline comparison/versioning.

No external API, network, or GPU required — purely local prompt construction.

Usage:
    python scripts/prompt_lab.py --prompt "A dragon flying through clouds"
    python scripts/prompt_lab.py --prompt "..." --targets vr180_flight,fulldome_180
    python scripts/prompt_lab.py --prompt "..." --scenes fpv,orbit --out my_manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pipeline.prompt_builder import wrap_prompt

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_TARGETS = ["vr180_flight", "fulldome_180", "vr360_dome"]
DEFAULT_SCENES = ["fpv", "orbit", "static"]

# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def generate_manifest(
    prompt: str,
    targets: list[str] | None = None,
    scenes: list[str] | None = None,
) -> list[dict[str, str]]:
    """Generate prompt variants for every target × scene combination.

    Parameters
    ----------
    prompt : str
        Original creative prompt text (must be non-empty).
    targets : list[str] | None
        Output targets to iterate over.
        Defaults to all three supported targets.
    scenes : list[str] | None
        Scene types to iterate over.
        Defaults to ``fpv``, ``orbit``, ``static``.

    Returns
    -------
    list[dict[str, str]]
        Each entry contains keys:
        ``target``, ``scene``, ``positive``, ``negative``, ``notes``.

    Raises
    ------
    ValueError
        If *prompt* is empty or whitespace-only.
    """
    if not prompt or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")

    _targets = targets or DEFAULT_TARGETS
    _scenes = scenes or DEFAULT_SCENES

    manifest: list[dict[str, str]] = []
    for target in _targets:
        for scene in _scenes:
            result = wrap_prompt(prompt, scene_type=scene, target=target)
            manifest.append(
                {
                    "target": target,
                    "scene": scene,
                    "positive": result["positive"],
                    "negative": result["negative"],
                    "notes": result.get("notes", ""),
                }
            )
    return manifest


# ---------------------------------------------------------------------------
# Terminal table
# ---------------------------------------------------------------------------


def _print_table(manifest: list[dict[str, str]]) -> None:
    """Print a compact human-readable table of prompt variants."""
    col_target = 18
    col_scene = 12
    col_positive_preview = 80

    header = f"{'Target':<{col_target}} {'Scene':<{col_scene}} {'Positive (first 80 chars)':<{col_positive_preview}}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for item in manifest:
        preview = item["positive"][:col_positive_preview]
        print(f"{item['target']:<{col_target}} {item['scene']:<{col_scene}} {preview:<{col_positive_preview}}")
    print(sep)


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Generate batch prompt variants for offline comparison.",
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="Original creative prompt text (required).",
    )
    parser.add_argument(
        "--targets",
        default=",".join(DEFAULT_TARGETS),
        help=(f"Comma-separated output targets. Default: {','.join(DEFAULT_TARGETS)}"),
    )
    parser.add_argument(
        "--scenes",
        default=",".join(DEFAULT_SCENES),
        help=(f"Comma-separated scene types. Default: {','.join(DEFAULT_SCENES)}"),
    )
    parser.add_argument(
        "--out",
        default="prompt_lab_manifest.json",
        help="Output manifest JSON path. Default: prompt_lab_manifest.json",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns exit code (0 = success)."""
    args = parse_args(argv)

    prompt = args.prompt
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]

    try:
        manifest = generate_manifest(prompt, targets=targets, scenes=scenes)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    out_path = Path(args.out)
    out_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nGenerated {len(manifest)} prompt variants → {out_path}")
    _print_table(manifest)

    return 0


if __name__ == "__main__":
    sys.exit(main())
