#!/usr/bin/env python3
"""Stereo parameter A/B sweep tool — render VR180 variants across a disparity × convergence grid.

Renders one short clip per parameter combination so Quest-side A/B scoring
turns "guessing parameters" into "choosing parameters".  Each variant is
written to ``<outdir>/sweep_d{disparity}_c{convergence}.mp4`` and a Markdown
manifest (``<outdir>/variants.md``) lists every file with empty scoring
columns for headset-side evaluation.

The render step is injectable (``render_variant`` callable) so unit tests run
without GPU, models, or ffmpeg.  The default runner reuses the existing
``scripts.run_pipeline`` stage functions — no conversion logic is duplicated.

Usage:
    python scripts/stereo_sweep.py --input video.mp4 --outdir out/sweep1
    python scripts/stereo_sweep.py -i video.mp4 -o out/sweep1 --limit-seconds 10
    python scripts/stereo_sweep.py -i video.mp4 -o out/sweep1 \\
        --disparities 0.02,0.04,0.06 --convergences near,mid
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from pipeline.comfort_presets import COMFORT_PRESETS

log = logging.getLogger("stereo-sweep")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_DISPARITIES: tuple[float, ...] = (0.02, 0.04, 0.06, 0.08)
DEFAULT_CONVERGENCES: tuple[str, ...] = ("near", "mid", "far")

# Named convergence presets → StereoRenderer ``convergence`` (fraction of
# normalized depth where the zero-parallax plane sits).  Larger = zero plane
# farther away = more of the scene pops out of the screen.
CONVERGENCE_PRESETS: dict[str, float] = {
    "near": 0.15,
    "mid": 0.30,
    "far": 0.50,
}

SCORE_COLUMNS: tuple[str, ...] = ("清晰度", "重影", "立体感", "舒适度")


def build_comfort_grid() -> list[dict]:
    """I-3 (#88): default sweep grid = the three owner-tuned comfort presets.

    Each comfort preset is one row of the grid (not a Cartesian product), so
    a default sweep renders exactly three variants — ``safe``, ``balanced``,
    ``strong`` — each encoding the matched max_disparity / convergence pair
    the operator should A/B in the headset.  ``convergence_name`` holds the
    preset label so it survives into the filename (``sweep_d0.035_cbalanced``)
    and the variants.md manifest.  ``temporal_smooth`` is True in every tier
    (see pipeline.comfort_presets).
    """
    out: list[dict] = []
    for name, cfg in COMFORT_PRESETS.items():
        out.append(
            {
                "max_disparity": float(cfg["max_disparity"]),
                "convergence_name": name,
                "convergence": float(cfg["convergence"]),
                "temporal_smooth": bool(cfg["temporal_smooth"]),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Grid / naming / manifest — pure functions (unit-testable, no I/O)
# ---------------------------------------------------------------------------


def build_grid(
    disparities: Sequence[float] | None = None,
    convergences: Sequence[str] | None = None,
) -> list[dict]:
    """Build the Cartesian product of disparity × convergence presets.

    Args:
        disparities: ``max_disparity`` values (fraction of image width).
            Defaults to ``DEFAULT_DISPARITIES``.
        convergences: named convergence presets.  Defaults to
            ``DEFAULT_CONVERGENCES``.  Unknown names raise ``ValueError``.

    Returns:
        List of variant dicts with keys ``max_disparity`` (float),
        ``convergence_name`` (str) and ``convergence`` (float preset value),
        ordered by disparity (outer) then convergence (inner).

    Raises:
        ValueError: if any disparity is not positive, or a convergence name
            is not in ``CONVERGENCE_PRESETS``.
    """
    disps = list(disparities) if disparities is not None else list(DEFAULT_DISPARITIES)
    convs = list(convergences) if convergences is not None else list(DEFAULT_CONVERGENCES)

    if not disps:
        raise ValueError("disparities must contain at least one value")
    if not convs:
        raise ValueError("convergences must contain at least one value")

    for d in disps:
        if d <= 0:
            raise ValueError(f"max_disparity must be positive, got {d}")
    unknown = [c for c in convs if c not in CONVERGENCE_PRESETS]
    if unknown:
        raise ValueError(f"unknown convergence preset(s): {unknown} — choose from {sorted(CONVERGENCE_PRESETS)}")

    return [
        {
            "max_disparity": d,
            "convergence_name": c,
            "convergence": CONVERGENCE_PRESETS[c],
        }
        for d, c in itertools.product(disps, convs)
    ]


def variant_filename(variant: dict) -> str:
    """Encode a variant's parameters into its output filename.

    Format: ``sweep_d{disparity}_c{convergence}.mp4`` — e.g.
    ``sweep_d0.04_cmid.mp4``.
    """
    return f"sweep_d{variant['max_disparity']:g}_c{variant['convergence_name']}.mp4"


def render_variants_md(
    variants: Sequence[dict],
    source: str,
    limit_seconds: float,
) -> str:
    """Render the Markdown manifest table with empty scoring columns.

    Args:
        variants: variant dicts as produced by :func:`build_grid`, each with
            an added ``filename`` key.
        source: source video path (recorded in the header).
        limit_seconds: clip length used for the sweep (recorded in the header).

    Returns:
        Markdown document as a string.
    """
    lines = [
        "# Stereo Sweep Variants",
        "",
        f"- Source: `{source}`",
        f"- Clip length: first {limit_seconds:g} s",
        f"- Variants: {len(variants)}",
        "",
        "| 文件 | max_disparity | convergence | " + " | ".join(SCORE_COLUMNS) + " | 备注 |",
        "| --- | --- | --- | " + " | ".join("---" for _ in SCORE_COLUMNS) + " | --- |",
    ]
    for v in variants:
        empty_scores = " | ".join("" for _ in SCORE_COLUMNS)
        lines.append(
            f"| {v['filename']} | {v['max_disparity']:g} | {v['convergence_name']} ({v['convergence']:g}) "
            f"| {empty_scores} |  |"
        )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Render runner — reuses scripts.run_pipeline stage functions
# ---------------------------------------------------------------------------


def prepare_pipeline_args(input_path: str, output_path: str, variant: dict):
    """Build fully-resolved run_pipeline args for one sweep variant.

    ``parse_args`` alone is not a complete contract: the --quality presets
    (#34) leave ``output_width``/``output_height``/``bitrate`` as ``None``
    for ``apply_quality_preset`` to fill in (``run_pipeline.main`` does this
    before any stage runs). Calling the stage functions with the unresolved
    ``None`` values crashes in the equirect stage, so resolve here too.
    """
    from scripts import run_pipeline as rp

    args = rp.parse_args(["--input", input_path, "--output", output_path])
    rp.apply_quality_preset(args)
    args.max_disparity = variant["max_disparity"]
    return args


def run_pipeline_variant(
    input_path: str,
    output_path: str,
    variant: dict,
    limit_seconds: float,
    fps: int | None = None,
) -> str:
    """Render one sweep variant via the existing pipeline stage functions.

    Reads the first ``limit_seconds`` of the source, runs depth → stereo →
    equirect → metadata with the variant's ``max_disparity`` / ``convergence``
    injected into the stereo renderer, and writes an SBS mp4.

    Heavy imports are deferred so the sweep module (grid/manifest logic) stays
    importable on machines without cv2/torch (e.g. CI running pure unit tests).
    """
    import cv2
    from scripts import run_pipeline as rp

    args = prepare_pipeline_args(input_path, output_path, variant)
    if args.device is None:
        from pipeline.device_utils import detect_best_device

        args.device = detect_best_device()

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    src_fps = src_fps if src_fps and src_fps > 0 else 30.0
    args.fps = fps or round(src_fps)

    max_frames = max(1, round(limit_seconds * src_fps))
    frames = list(rp.read_frames(input_path, max_frames=max_frames))
    if not frames:
        raise RuntimeError(f"No frames read from {input_path}")
    log.info(
        "Variant %s: %d frames (%.1f s @ %.1f fps)", variant_filename(variant), len(frames), limit_seconds, src_fps
    )

    depths = rp.run_depth_stage(args, frames)

    # Inject the variant's convergence preset into the default renderer.
    if args.stereo_model == "default":
        from pipeline.stereo_renderer import StereoRenderer

        renderer = StereoRenderer(
            ipd=args.ipd,
            max_disparity=args.max_disparity,
            temporal_smooth=not args.no_temporal,
            convergence=variant["convergence"],
        )
        left_frames, right_frames = [], []
        for frame, depth in zip(frames, depths, strict=False):
            left, right = renderer.render(frame, depth)
            left_frames.append(left)
            right_frames.append(right)
    else:
        log.warning(
            "stereo-model=%s has no convergence parameter — grid degenerates to disparity only for this variant",
            args.stereo_model,
        )
        left_frames, right_frames = rp.run_stereo_stage(args, frames, depths)

    sbs_frames = rp.run_equirect_stage(args, left_frames, right_frames)
    result = rp.run_metadata_stage(args, sbs_frames)
    return str(result)


# ---------------------------------------------------------------------------
# Sweep orchestration
# ---------------------------------------------------------------------------


def run_sweep(
    input_path: str,
    outdir: str | Path,
    limit_seconds: float = 5.0,
    disparities: Sequence[float] | None = None,
    convergences: Sequence[str] | None = None,
    render_variant: Callable[[str, str, dict, float], str] | None = None,
    fps: int | None = None,
    *,
    comfort_presets: bool | None = None,
) -> list[dict]:
    """Render every grid variant and write the manifest.

    Args:
        input_path: source video path.
        outdir: output directory (created if missing).
        limit_seconds: only convert the first N seconds of the source.
        disparities / convergences: grid axes (see :func:`build_grid`).  When
            both are ``None`` and *comfort_presets* is not ``False``, the grid
            defaults to the three I-3 comfort presets via
            :func:`build_comfort_grid`.
        render_variant: injectable render callable with signature
            ``(input_path, output_path, variant, limit_seconds) -> str``.
            Defaults to :func:`run_pipeline_variant`.  Tests inject a fake
            runner so no real conversion/ffmpeg runs.
        fps: optional output fps override passed to the default runner.
        comfort_presets: if ``True`` (or left ``None`` with no custom axes),
            sweep the three comfort presets instead of the legacy
            disparities×convergences Cartesian product.  ``False`` forces the
            legacy product even when no custom axes are given.

    Returns:
        The variant list (with ``filename`` filled in) that was written to
        ``variants.md``.
    """
    runner = render_variant or run_pipeline_variant
    if comfort_presets is False:
        variants = build_grid(disparities=disparities, convergences=convergences)
    elif disparities is None and convergences is None:
        variants = build_comfort_grid()
    else:
        variants = build_grid(disparities=disparities, convergences=convergences)

    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    for v in variants:
        filename = variant_filename(v)
        v["filename"] = filename
        dest = out_path / filename
        log.info("=== Rendering %s (d=%g, c=%s) ===", filename, v["max_disparity"], v["convergence_name"])
        runner(input_path, str(dest), v, limit_seconds)

    manifest_path = out_path / "variants.md"
    manifest_path.write_text(
        render_variants_md(variants, source=str(input_path), limit_seconds=limit_seconds),
        encoding="utf-8",
    )
    log.info("✅ Sweep complete: %d variants + manifest → %s", len(variants), manifest_path)
    return variants


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.  Accept optional *argv* for testing."""
    parser = argparse.ArgumentParser(
        description="Stereo parameter A/B sweep — render one VR180 variant per "
        "max_disparity × convergence combination for Quest-side scoring.",
    )
    parser.add_argument("--input", "-i", required=True, help="Source 2D video file (MP4, MOV, etc.)")
    parser.add_argument("--outdir", "-o", required=True, help="Output directory for variants + variants.md")
    parser.add_argument(
        "--limit-seconds",
        type=float,
        default=5.0,
        help="Only convert the first N seconds of the source (default: 5)",
    )
    parser.add_argument(
        "--disparities",
        default=None,
        help=(
            "Comma-separated max_disparity values for a custom grid "
            "(e.g. 0.02,0.04,0.06). When omitted the default grid sweeps the "
            "three --comfort presets (safe/balanced/strong) instead."
        ),
    )
    parser.add_argument(
        "--convergences",
        default=None,
        help=(
            f"Comma-separated convergence presets from {sorted(CONVERGENCE_PRESETS)} "
            "for a custom grid (e.g. near,mid,far). Requires --disparities; "
            "when both are omitted the default grid sweeps the three comfort presets."
        ),
    )
    parser.add_argument("--fps", type=int, default=None, help="Output fps (default: inherit from source)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.  Returns exit code (0 = success)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args(argv)

    disparities: list[float] | None = None
    convergences: list[str] | None = None
    if args.disparities is not None or args.convergences is not None:
        # A custom grid axis was given → use the legacy disparities×convergences
        # product (build_grid validates convergence names). A missing axis falls
        # back to the module default so e.g. ``--convergences bogus`` still
        # surfaces the "unknown convergence" validation error rather than a
        # usage message.
        try:
            disparities = [float(x) for x in (args.disparities or "").split(",") if x.strip()] or list(
                DEFAULT_DISPARITIES
            )
        except ValueError:
            print(f"Error: --disparities must be comma-separated numbers, got {args.disparities!r}", file=sys.stderr)
            return 1
        convergences = [c.strip() for c in (args.convergences or "").split(",") if c.strip()] or list(
            DEFAULT_CONVERGENCES
        )

    try:
        run_sweep(
            input_path=args.input,
            outdir=args.outdir,
            limit_seconds=args.limit_seconds,
            disparities=disparities,
            convergences=convergences,
            fps=args.fps,
        )
    except (ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
