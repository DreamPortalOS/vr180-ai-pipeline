#!/usr/bin/env python3
"""VR180 Pipeline CLI — convert 2D AI video to VR180 immersive format.

Usage:
    # Full pipeline
    python scripts/run_pipeline.py --input video.mp4 --output vr180.mp4

    # With temporal smoothing + H.265
    python scripts/run_pipeline.py --input video.mp4 --output vr180.mp4 \
        --model-size base --codec h265 --fps 30 --temporal-smoothing 0.3

    # With pixel upscaling
    python scripts/run_pipeline.py --input video.mp4 --output vr180.mp4 --upscale 2

    # Validate input format
    python scripts/run_pipeline.py --input video.mp4 --validate-input

    # Individual stages with temp dir
    python scripts/run_pipeline.py --input video.mp4 --stage depth
    python scripts/run_pipeline.py --input video.mp4 --stage stereo --temp-dir frames/
    python scripts/run_pipeline.py --input video.mp4 --stage equirect
    python scripts/run_pipeline.py --input video.mp4 --stage metadata --output vr180.mp4
"""

import argparse
import json
import logging
import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from pipeline.depth_crafter import DepthCrafterEstimator
from pipeline.depth_estimator import DepthEstimator
from pipeline.device_utils import detect_best_device, resolve_device
from pipeline.equirectangular_mapper import EquirectangularMapper
from pipeline.fulldome_mapper import FulldomeMapper
from pipeline.stereo_renderer import StereoRenderer
from pipeline.streaming_pipeline import StreamingPipeline
from pipeline.upscaler import PixelUpscaler
from pipeline.video_upscaler import SeedVR2Upscaler
from pipeline.vr_metadata import VRMetadataEmbedder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("vr180-pipeline")


def parse_args():
    parser = argparse.ArgumentParser(description="2D AI Video → VR180 Conversion Pipeline")
    parser.add_argument("--input", "-i", required=True, help="Input video file (MP4, MOV, etc.)")
    parser.add_argument("--output", "-o", default=None, help="Output VR180 video path")
    parser.add_argument(
        "--stage",
        "-s",
        choices=["all", "depth", "stereo", "equirect", "metadata"],
        default="all",
        help="Pipeline stage to run (default: all)",
    )
    parser.add_argument(
        "--model-size", default="small", choices=["small", "base", "large"], help="Depth Anything V2 model size"
    )
    parser.add_argument("--device", default=None, help="Compute device (cuda, mps, cpu)")
    parser.add_argument("--ipd", type=float, default=0.064, help="Interpupillary distance in meters")
    parser.add_argument("--max-disparity", type=float, default=0.05, help="Max disparity as fraction of image width")
    parser.add_argument("--codec", choices=["h264", "h265"], default="h264", help="Output video codec")
    parser.add_argument("--crf", type=int, default=23, help="Constant rate factor")
    parser.add_argument("--fps", type=int, default=None, help="Output frame rate (default: inherit from source video)")
    parser.add_argument("--output-width", type=int, default=3840, help="Equirectangular output width per eye")
    parser.add_argument("--output-height", type=int, default=1920, help="Equirectangular output height per eye")
    parser.add_argument("--src-hfov", type=float, default=70.0, help="Source camera horizontal FOV (degrees)")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit number of frames (for testing)")
    parser.add_argument("--no-temporal", action="store_true", help="Disable temporal smoothing")
    parser.add_argument("--temp-dir", default=None, help="Directory for intermediate files")
    parser.add_argument("--no-ffmpeg-v360", action="store_true", help="Disable ffmpeg v360, use OpenCV fallback")
    parser.add_argument(
        "--no-flip", action="store_true", help="Disable vertical flip (default: flip on for VR headset)"
    )

    # New: temporal smoothing
    parser.add_argument(
        "--temporal-smoothing", type=float, default=0.0, help="Temporal EMA alpha for depth smoothing (0=off, 0.3-0.5)"
    )
    parser.add_argument(
        "--stereo-smoothing", type=float, default=0.0, help="Temporal EMA alpha for stereo shift (0=off)"
    )
    parser.add_argument(
        "--baseline", type=int, default=0, help="Override stereo baseline shift in pixels (0=use IPD-based)"
    )

    # New: pixel upscaling
    parser.add_argument("--upscale", type=int, default=0, choices=[0, 2, 4], help="Upscale factor (0=off, 2=2×, 4=4×)")
    parser.add_argument("--upscale-model", default=None, help="Real-ESRGAN model name (auto if omitted)")
    parser.add_argument(
        "--upscale-ffmpeg", action="store_true", help="Use ffmpeg/OpenCV lanczos upscale instead of Real-ESRGAN"
    )

    # New: output encoding options
    parser.add_argument("--bitrate", default=None, help="Target bitrate (e.g., 50M). Overrides CRF if set.")
    parser.add_argument("--hardware-encoder", action="store_true", help="Use hardware encoder (VideoToolbox)")

    # New: input validation
    parser.add_argument(
        "--validate-input", action="store_true", help="Validate input video format and print recommendations"
    )

    # New: checkpoint/resume
    parser.add_argument("--resume", action="store_true", help="Resume from last completed checkpoint stage")

    # Phase 1: Streaming pipeline (PRD §7.2)
    parser.add_argument(
        "--streaming", action="store_true", help="Use streaming pipeline (O(1) memory, pipes to ffmpeg)"
    )

    # Phase 1: Tiled upscaling (PRD §7.4)
    parser.add_argument("--tiled-upscale", action="store_true", help="Use tiled upscaling for large frames (8K-safe)")
    parser.add_argument("--tile-size", type=int, default=512, help="Tile size for tiled upscaling (default: 512)")

    # Phase 2: Smart SBS detection (Task 1.1)
    parser.add_argument(
        "--force-sbs", action="store_true", help="Force treat input as SBS stereo (skip depth/stereo stages)"
    )

    # R-5: Fulldome projection
    parser.add_argument(
        "--projection",
        choices=["vr180", "fulldome"],
        default="vr180",
        help="Output projection: vr180 (stereo spherical) or fulldome (mono fisheye, default vr180)",
    )
    parser.add_argument(
        "--dome-fov", type=float, default=180.0, help="Fulldome fisheye FOV in degrees (default 180, max ~220)"
    )
    parser.add_argument(
        "--dome-coverage-h", type=float, default=120.0, help="Fulldome source horizontal coverage FOV (default 120)"
    )
    parser.add_argument(
        "--dome-coverage-v", type=float, default=None, help="Fulldome source vertical coverage FOV (auto if omitted)"
    )
    parser.add_argument(
        "--dome-size", type=int, default=4096, help="Fulldome output square size in pixels (default 4096)"
    )

    # Depth model selection
    parser.add_argument(
        "--depth-model",
        choices=["depth-anything", "depthcrafter"],
        default="depth-anything",
        help="Depth estimation backend: depth-anything (per-frame, default) or "
        "depthcrafter (temporally-consistent video depth, CUDA-only)",
    )
    parser.add_argument(
        "--depthcrafter-repo-dir",
        default=None,
        help="DepthCrafter repository directory (or env DEPTHCRAFTER_REPO_DIR)",
    )
    parser.add_argument(
        "--depthcrafter-python",
        default=None,
        help="Python executable for DepthCrafter inference (or env DEPTHCRAFTER_PYTHON)",
    )
    parser.add_argument(
        "--depthcrafter-checkpoint-dir",
        default=None,
        help="DepthCrafter checkpoint directory (or env DEPTHCRAFTER_CKPT_DIR)",
    )
    parser.add_argument(
        "--depthcrafter-max-res",
        type=int,
        default=None,
        help="Max resolution (short side) for DepthCrafter inference (or env DEPTHCRAFTER_MAX_RES)",
    )

    # R-1: SeedVR2 video upscaling pre-stage
    parser.add_argument(
        "--video-upscale",
        choices=["none", "seedvr2"],
        default="none",
        help="Video upscaling method: none (skip) or seedvr2 (SeedVR2, Stage 0) (default: none)",
    )
    parser.add_argument(
        "--video-upscale-factor",
        type=int,
        default=2,
        choices=[2, 3, 4],
        help="SeedVR2 upscaling factor (default: 2)",
    )
    # [Deprecated] ComfyUI URL — use --seedvr2-node-dir for CLI backend
    parser.add_argument(
        "--seedvr2-url",
        default="http://127.0.0.1:8188",
        help="[Deprecated — use --seedvr2-node-dir] ComfyUI server URL (default: http://127.0.0.1:8188)",
    )

    # SeedVR2 CLI backend params
    parser.add_argument(
        "--seedvr2-node-dir",
        default=None,
        help="SeedVR2 custom node directory (contains inference_cli.py). Can also set SEEDVR2_NODE_DIR env var.",
    )
    parser.add_argument(
        "--seedvr2-python",
        default=None,
        help="Python executable for inference_cli.py (default: python). Can also set SEEDVR2_PYTHON env var.",
    )
    parser.add_argument(
        "--seedvr2-model-dir",
        default=None,
        help="SeedVR2 model .safetensors directory. "
        "Can also set SEEDVR2_MODEL_DIR env var (default: <node_dir>/../../models/SEEDVR2).",
    )
    parser.add_argument(
        "--seedvr2-resolution",
        type=int,
        default=None,
        help="Output short-side resolution. Auto from source height × factor if 0. "
        "Can also set SEEDVR2_RESOLUTION env var (default: 1440).",
    )

    return parser.parse_args()


def read_frames(video_path: str, max_frames: int | None = None):
    """Yield RGB frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames:
        total = min(total, max_frames)

    log.info(
        f"Video: {fps:.2f} fps, {total} frames, "
        f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
        f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
    )

    count = 0
    while count < total:
        ret, frame = cap.read()
        if not ret:
            break
        yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        count += 1

    cap.release()


def get_output_path(args, suffix=".mp4"):
    """Get output path or generate default."""
    if args.output:
        return args.output
    stem = Path(args.input).stem
    return f"{stem}_vr180{suffix}"


def get_temp_dir(args, subdir=None):
    """Get or create temp directory for intermediate files.

    If --temp-dir is specified, uses that directory.
    Otherwise, uses a default directory next to the input file
    for consistent cross-stage access.
    """
    if args.temp_dir:
        base = Path(args.temp_dir)
    else:
        # Default: use a directory next to the input file for stage persistence
        input_stem = Path(args.input).stem
        base = Path(args.input).parent / f"{input_stem}_vr180_temp"
    path = base / subdir if subdir else base
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def run_depth_stage(args, frames):
    """Stage 1: Estimate depth for all frames."""
    log.info("=== Stage 1: Depth Estimation ===")

    # DepthCrafter mode — process entire video at once (temporally consistent)
    if args.depth_model == "depthcrafter":
        log.info("Using DepthCrafter for temporally-consistent video depth estimation")
        out_dir = get_temp_dir(args, "depth")
        estimator = DepthCrafterEstimator(
            repo_dir=args.depthcrafter_repo_dir,
            python_exe=args.depthcrafter_python,
            checkpoint_dir=args.depthcrafter_checkpoint_dir,
            max_resolution=args.depthcrafter_max_res,
        )
        depths = estimator.estimate_video(
            input_path=args.input,
            output_dir=out_dir,
        )
        # Save individual depth maps for downstream stages
        for i, depth in enumerate(depths):
            dmax = float(np.nanmax(depth))
            depth_vis = (depth / dmax * 255).astype(np.uint8) if dmax > 0 else depth.astype(np.uint8)
            cv2.imwrite(
                os.path.join(out_dir, f"depth_{i:06d}.png"),
                cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO),
            )
            np.save(os.path.join(out_dir, f"depth_{i:06d}.npy"), depth)

        log.info(f"Depth maps (DepthCrafter) saved to {out_dir}/")
        return depths

    # Default: Depth-Anything V2 per-frame
    estimator = DepthEstimator(
        model_size=args.model_size,
        device=args.device,
        calibrate=True,
    )

    out_dir = get_temp_dir(args, "depth")
    depths = []
    prev_depth = None
    temporal_alpha = args.temporal_smoothing if args.temporal_smoothing > 0 else None

    for i, frame in enumerate(tqdm(frames, desc="Estimating depth")):
        depth = estimator.estimate(frame)

        # Pipeline-level temporal smoothing (EMA)
        if temporal_alpha and prev_depth is not None:
            depth = temporal_alpha * depth + (1 - temporal_alpha) * prev_depth
        prev_depth = depth

        depths.append(depth)
        # Save depth map for inspection
        dmax = float(np.nanmax(depth))
        depth_vis = (depth / dmax * 255).astype(np.uint8) if dmax > 0 else depth.astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, f"depth_{i:06d}.png"), cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO))
        np.save(os.path.join(out_dir, f"depth_{i:06d}.npy"), depth)

    log.info(f"Depth maps saved to {out_dir}/")
    return depths


def run_stereo_stage(args, frames, depths):
    """Stage 2: Generate stereo left/right views."""
    log.info("=== Stage 2: Stereo Disparity Rendering ===")

    renderer = StereoRenderer(
        ipd=args.ipd,
        max_disparity=args.max_disparity,
        temporal_smooth=not args.no_temporal,
    )

    left_dir = get_temp_dir(args, "left")
    right_dir = get_temp_dir(args, "right")

    left_frames, right_frames = [], []
    for i, (frame, depth) in enumerate(
        tqdm(zip(frames, depths, strict=False), desc="Rendering stereo", total=len(frames))
    ):
        left, right = renderer.render(frame, depth)
        left_frames.append(left)
        right_frames.append(right)

        # Save intermediate files
        cv2.imwrite(os.path.join(left_dir, f"left_{i:06d}.png"), cv2.cvtColor(left, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(right_dir, f"right_{i:06d}.png"), cv2.cvtColor(right, cv2.COLOR_RGB2BGR))

    log.info(f"Stereo views: {len(left_frames)} frames each")
    return left_frames, right_frames


def run_equirect_stage(args, left_frames, right_frames):
    """Stage 3: Map stereo views to equirectangular."""
    log.info("=== Stage 3: Equirectangular Projection ===")

    mapper = EquirectangularMapper(
        output_width=args.output_width,
        output_height=args.output_height,
        src_hfov=args.src_hfov,
        use_ffmpeg=not args.no_ffmpeg_v360,
    )

    out_dir = get_temp_dir(args, "equirect")
    sbs_frames = []
    for i, (left, right) in enumerate(
        tqdm(zip(left_frames, right_frames, strict=False), desc="Mapping to equirect", total=len(left_frames))
    ):
        sbs = mapper.map_stereo_pair(left, right)
        sbs_frames.append(sbs)
        cv2.imwrite(os.path.join(out_dir, f"equirect_{i:06d}.png"), cv2.cvtColor(sbs, cv2.COLOR_RGB2BGR))

    log.info(
        f"Generated {len(sbs_frames)} equirectangular SBS frames ({sbs_frames[0].shape[1]}×{sbs_frames[0].shape[0]})"
    )
    return sbs_frames


def run_metadata_stage(args, sbs_frames):
    """Stage 4: Encode video with VR metadata."""
    log.info("=== Stage 4: VR Metadata Embedding ===")

    embedder = VRMetadataEmbedder(
        codec=args.codec,
        crf=args.crf,
        fps=args.fps,
    )

    output_path = get_output_path(args)

    H, W = sbs_frames[0].shape[:2]
    log.info(f"Encoding {len(sbs_frames)} frames ({W}×{H}) → {output_path}")
    result = embedder.embed_single_frame_batch(
        sbs_frames,
        output_path,
        width=W,
        height=H,
    )
    return result


def run_seedvr2_prestage(args) -> str:
    """Stage 0: SeedVR2 video upscaling (runs on the whole video file before frame loading).

    Upscales the input video via SeedVR2 inference_cli.py (CLI backend),
    saves the result to a temp path, and returns the path to the upscaled
    video.  The caller replaces args.input with this path so all downstream
    stages see the higher-resolution source.
    """
    log.info("=== Stage 0: SeedVR2 Video Upscaling (%d×) ===", args.video_upscale_factor)

    temp_dir = get_temp_dir(args)
    stem = Path(args.input).stem
    upscaled_path = os.path.join(temp_dir, f"{stem}_seedvr2_{args.video_upscale_factor}x.mp4")

    upscaler = SeedVR2Upscaler(
        batch_size=5,
        node_dir=args.seedvr2_node_dir,
        python_exe=args.seedvr2_python,
        model_dir=args.seedvr2_model_dir,
        resolution=args.seedvr2_resolution,
    )

    log.info("SeedVR2: %s → %s (factor=%d)", args.input, upscaled_path, args.video_upscale_factor)
    result = upscaler.upscale(
        input_path=args.input,
        output_path=upscaled_path,
        factor=args.video_upscale_factor,
    )
    log.info("SeedVR2 upscale complete → %s", result)
    return result


def run_upscale_stage(args, frames):
    """Stage 0: Pixel upscaling (optional)."""
    log.info(f"=== Stage 0: Pixel Upscaling ({args.upscale}×) ===")

    if args.upscale_ffmpeg:
        log.info("Using OpenCV lanczos upscale (fallback)")
        upscaled = []
        for frame in tqdm(frames, desc="Upscaling (lanczos)"):
            h, w = frame.shape[:2]
            result = cv2.resize(frame, (w * args.upscale, h * args.upscale), interpolation=cv2.INTER_LANCZOS4)
            upscaled.append(result)
        return upscaled

    try:
        upscaler = PixelUpscaler(
            scale=args.upscale,
            model_name=args.upscale_model,
            device=args.device,
        )
    except ImportError:
        log.warning("realesrgan not installed, falling back to OpenCV lanczos")
        upscaled = []
        for frame in tqdm(frames, desc="Upscaling (lanczos)"):
            h, w = frame.shape[:2]
            result = cv2.resize(frame, (w * args.upscale, h * args.upscale), interpolation=cv2.INTER_LANCZOS4)
            upscaled.append(result)
        return upscaled

    upscaled = []
    use_tiled = getattr(args, "tiled_upscale", False)
    tile_size = getattr(args, "tile_size", 512)

    for frame in tqdm(frames, desc=f"Upscaling ({args.upscale}× Real-ESRGAN{' tiled' if use_tiled else ''})"):
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if use_tiled:
            result_bgr = upscaler.upscale_tiled(
                frame_bgr,
                tile_size=tile_size,
                progress_callback=None,
            )
        else:
            result_bgr = upscaler.upscale_frame(frame_bgr)
        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
        upscaled.append(result_rgb)

    log.info(
        f"Upscaled {len(upscaled)} frames: "
        f"{frames[0].shape[1]}×{frames[0].shape[0]} → "
        f"{upscaled[0].shape[1]}×{upscaled[0].shape[0]}"
    )
    return upscaled


def validate_input_format(input_path: str):
    """Validate input video format and print VR180 recommendations."""
    import json
    import subprocess

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"❌ Cannot open: {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total / fps if fps > 0 else 0
    cap.release()

    # Get codec info via ffprobe
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,pix_fmt,bit_rate,profile",
                "-show_entries",
                "format=format_name,bit_rate",
                "-of",
                "json",
                input_path,
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        info = json.loads(result.stdout) if result.returncode == 0 else {}
    except Exception:
        info = {}

    codec = info.get("streams", [{}])[0].get("codec_name", "unknown")
    pix_fmt = info.get("streams", [{}])[0].get("pix_fmt", "unknown")
    bitrate = info.get("format", {}).get("bit_rate", "unknown")
    fmt = info.get("format", {}).get("format_name", "unknown")

    print("=" * 60)
    print("INPUT VIDEO FORMAT ANALYSIS")
    print("=" * 60)
    print(f"  File:      {os.path.basename(input_path)}")
    print(f"  Format:    {fmt}")
    print(f"  Codec:     {codec}")
    print(f"  Pixel fmt: {pix_fmt}")
    print(f"  Resolution: {w}×{h}")
    print(f"  FPS:       {fps:.2f}")
    print(f"  Duration:  {duration:.2f}s ({total} frames)")
    print(f"  Bitrate:   {int(bitrate) // 1000 if bitrate != 'unknown' else '?'} kbps")
    print()

    score = 0
    issues = []
    recommendations = []

    # Resolution
    if w >= 1920 and h >= 1080:
        score += 2
        print("  ✅ Resolution: Good (≥1080p)")
    elif w >= 1280:
        score += 1
        issues.append("Resolution is 720p — consider 1080p+ input")
        recommendations.append("Re-record at 1080p+ or use --upscale 2")
    else:
        issues.append(f"Resolution {w}×{h} is low")
        recommendations.append("Use --upscale 2 or --upscale 4 to compensate")

    # Codec
    if codec in ("h264", "hevc", "h265"):
        score += 2
        print(f"  ✅ Codec: {codec} (recommended)")
    elif codec in ("prores", "dnxhd"):
        score += 2
        print(f"  ✅ Codec: {codec} (professional quality)")
    else:
        score += 1
        issues.append(f"Codec '{codec}' may cause quality loss")
        recommendations.append("Transcode to H.264 or H.265 first")

    # FPS
    if 24 <= fps <= 60:
        score += 2
        print(f"  ✅ FPS: {fps:.0f} (good for VR)")
    elif fps > 60:
        score += 1
        issues.append(f"FPS {fps:.0f} is very high — will increase processing time")
        recommendations.append("Consider --fps 30 to reduce processing")
    else:
        score += 1
        issues.append(f"FPS {fps:.0f} is low — may cause motion sickness in VR")
        recommendations.append("Record at 24-60fps")

    # Bitrate
    if bitrate != "unknown":
        br_mbps = int(bitrate) / 1_000_000
        if br_mbps >= 20:
            score += 2
            print(f"  ✅ Bitrate: {br_mbps:.1f} Mbps (high quality)")
        elif br_mbps >= 8:
            score += 1
            print(f"  ⚠️  Bitrate: {br_mbps:.1f} Mbps (moderate)")
            recommendations.append("Use higher bitrate source if available")
        else:
            issues.append(f"Bitrate {br_mbps:.1f} Mbps is very low")

    # Duration
    if duration <= 120:
        print(f"  ✅ Duration: {duration:.0f}s (manageable)")
    else:
        print(f"  ⚠️  Duration: {duration:.0f}s (long — will take significant time)")

    print()
    print(f"  INPUT QUALITY SCORE: {score}/8")
    print()

    if issues:
        print("  ISSUES:")
        for issue in issues:
            print(f"    ⚠️  {issue}")
        print()

    if recommendations:
        print("  RECOMMENDATIONS:")
        for rec in recommendations:
            print(f"    💡 {rec}")
        print()

    print("  OUTPUT RESOLUTION GUIDE:")
    if w <= 1280:
        print(f"    Input {w}×{h} → Upscale 4× → SBS output (--upscale 4)")
    elif w <= 1920:
        print(f"    Input {w}×{h} → Upscale 2× → SBS output (--upscale 2)")
    else:
        print(f"    Input {w}×{h} → Direct → SBS output (no upscale needed)")

    print("=" * 60)


def save_checkpoint(temp_dir: str, stage: str, info: dict | None = None):
    """Save a checkpoint file indicating which stage completed."""
    checkpoint_path = os.path.join(temp_dir, "checkpoint.json")
    data = {"last_completed_stage": stage}
    if info:
        data.update(info)
    with open(checkpoint_path, "w") as f:
        json.dump(data, f, indent=2)
    log.info(f"💾 Checkpoint saved: {stage}")


def load_checkpoint(temp_dir: str):
    """Load checkpoint info. Returns dict or None."""
    checkpoint_path = os.path.join(temp_dir, "checkpoint.json")
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            return json.load(f)
    return None


STAGE_ORDER = ["upscale", "depth", "stereo", "equirect", "metadata"]
STAGE_ORDER_SBS = ["upscale", "equirect", "metadata"]  # Skip depth & stereo for SBS input


def detect_sbs_input(video_path: str, force_sbs: bool = False) -> bool:
    """Detect if input video is already a Side-by-Side (SBS) stereo frame.

    Detection logic:
    - If --force-sbs is set, always return True
    - If width/height ratio >= 3.5:1 (e.g., 7680×1920 = 4:1), treat as SBS

    Args:
        video_path: Path to input video file
        force_sbs: Manual override flag

    Returns:
        True if input should be treated as SBS stereo
    """
    if force_sbs:
        log.info("🔒 --force-sbs flag set: treating input as SBS stereo")
        return True

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if h == 0:
        return False

    ratio = w / h
    is_sbs = ratio >= 3.5

    if is_sbs:
        log.info(
            f"🔍 SBS auto-detection: {w}×{h} (ratio {ratio:.2f}:1) → SBS stereo detected! Skipping depth/stereo stages."
        )
    else:
        log.info(f"🔍 SBS auto-detection: {w}×{h} (ratio {ratio:.2f}:1) → Standard 2D input. Running full pipeline.")

    return is_sbs


def get_resume_start_stage(temp_dir: str):
    """Determine which stage to resume from based on checkpoint."""
    ckpt = load_checkpoint(temp_dir)
    if not ckpt:
        return 0  # start from beginning
    last = ckpt.get("last_completed_stage", "")
    if last in STAGE_ORDER:
        idx = STAGE_ORDER.index(last) + 1
        if idx < len(STAGE_ORDER):
            log.info(f"📂 Resuming after stage '{last}' → starting '{STAGE_ORDER[idx]}'")
            return idx
    return 0


def main():
    args = parse_args()

    # R-1: SeedVR2 pre-stage — upscale the input video file before any frame loading
    if args.video_upscale == "seedvr2":
        original_input = args.input
        args.input = run_seedvr2_prestage(args)
        log.info("SeedVR2 pre-stage: input replaced %s → %s", original_input, args.input)

    # Auto-detect device if not specified
    if args.device is None:
        args.device = detect_best_device()
    else:
        args.device = resolve_device(args.device)

    # Handle --validate-input mode
    if args.validate_input:
        validate_input_format(args.input)
        return

    # Inherit output fps from source video unless explicitly overridden.
    # Prevents speed-up/duration-mismatch when source != 30fps.
    if args.fps is None:
        cap = cv2.VideoCapture(args.input)
        src_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        args.fps = round(src_fps) if src_fps and src_fps > 0 else 30
        log.info(f"📹 Output fps inherited from source: {args.fps}")

    # Streaming pipeline mode (PRD §7.2)
    if args.streaming and args.stage == "all":
        log.info("🚀 Streaming pipeline mode (O(1) memory)")
        pipeline = StreamingPipeline(
            model_size=args.model_size,
            device=args.device,
            ipd=args.ipd,
            max_disparity=args.max_disparity,
            output_width=args.output_width,
            output_height=args.output_height,
            src_hfov=args.src_hfov,
            codec=args.codec,
            crf=args.crf,
            fps=args.fps,
        )
        output = get_output_path(args)
        result = pipeline.process_stream(args.input, output, max_frames=args.max_frames)
        log.info(f"✅ Streaming pipeline complete → {result}")
        return

    # R-5: Fulldome projection mode — skip all depth/stereo/equirect/metadata
    if args.projection == "fulldome":
        log.info("🌐 Fulldome projection mode — bypassing depth/stereo/equirect/metadata stages")
        mapper = FulldomeMapper(
            dome_fov=args.dome_fov,
            coverage_h_fov=args.dome_coverage_h,
            coverage_v_fov=args.dome_coverage_v,
            output_size=args.dome_size,
            codec=args.codec,
            crf=args.crf,
        )
        output = get_output_path(args, suffix="_dome.mp4")
        result = mapper.convert(args.input, output)
        log.info(f"✅ Fulldome conversion complete → {result}")
        return

    if args.stage == "all":
        # Smart SBS detection: if input is already SBS, skip depth/stereo
        is_sbs = detect_sbs_input(args.input, force_sbs=args.force_sbs)

        temp_dir = get_temp_dir(args)

        # Determine resume point
        start_idx = 0
        if args.resume:
            start_idx = get_resume_start_stage(temp_dir)

        # Use SBS stage order if input is already stereo
        base_order = STAGE_ORDER_SBS if is_sbs else STAGE_ORDER
        need_frames = start_idx == 0
        stages_to_run = base_order[start_idx:] if start_idx > 0 else base_order

        # Filter: only run upscale if --upscale is set
        if "upscale" in stages_to_run and args.upscale == 0:
            stages_to_run = [s for s in stages_to_run if s != "upscale"]

        # Load frames if needed
        frames = None
        if need_frames or "depth" in stages_to_run:
            frames = list(read_frames(args.input, args.max_frames))
            log.info(f"Loaded {len(frames)} frames")

            if frames:
                H, W = frames[0].shape[:2]
                mem_mb = len(frames) * H * W * 3 / (1024 * 1024)
                if mem_mb > 1024:
                    log.warning(
                        f"⚠️  Frame buffer uses ~{mem_mb:.0f} MB in RAM. "
                        f"For large videos, consider using --max-frames or --temp-dir."
                    )

        # Run stages sequentially with checkpointing
        depths = None
        left_frames, right_frames = None, None
        sbs_frames = None
        output = None

        for stage in stages_to_run:
            if stage == "upscale":
                frames = run_upscale_stage(args, frames)
                save_checkpoint(temp_dir, "upscale")

            elif stage == "depth":
                if frames is None:
                    frames = list(read_frames(args.input, args.max_frames))
                depths = run_depth_stage(args, frames)
                save_checkpoint(temp_dir, "depth", {"num_frames": len(depths)})

            elif stage == "stereo":
                if depths is None:
                    # Load depth maps from disk
                    depth_dir = get_temp_dir(args, "depth")
                    import glob

                    depth_files = sorted(glob.glob(os.path.join(depth_dir, "*.npy")))
                    depths = [np.load(f) for f in depth_files]
                    log.info(f"📂 Loaded {len(depths)} depth maps from checkpoint")
                if frames is None:
                    frames = list(read_frames(args.input, args.max_frames))
                left_frames, right_frames = run_stereo_stage(args, frames, depths)
                save_checkpoint(temp_dir, "stereo", {"num_frames": len(left_frames)})

            elif stage == "equirect":
                if is_sbs and left_frames is None:
                    # SBS input: split each frame into left/right halves
                    log.info("🔲 SBS input detected — splitting frames into left/right")
                    if frames is None:
                        frames = list(read_frames(args.input, args.max_frames))
                    left_frames, right_frames = [], []
                    for frame in frames:
                        _h, w = frame.shape[:2]
                        mid = w // 2
                        left_frames.append(frame[:, :mid, :])
                        right_frames.append(frame[:, mid:, :])
                    log.info(
                        f"  Split {len(frames)} SBS frames: "
                        f"{frames[0].shape[1]}×{frames[0].shape[0]} → "
                        f"left/right {left_frames[0].shape[1]}×{left_frames[0].shape[0]}"
                    )
                elif left_frames is None:
                    # Standard input: load from checkpoint
                    import glob

                    left_dir = get_temp_dir(args, "left")
                    right_dir = get_temp_dir(args, "right")
                    left_files = sorted(glob.glob(os.path.join(left_dir, "*.png")))
                    right_files = sorted(glob.glob(os.path.join(right_dir, "*.png")))
                    left_frames = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in left_files]
                    right_frames = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in right_files]
                    log.info(f"📂 Loaded {len(left_frames)} stereo frames from checkpoint")
                sbs_frames = run_equirect_stage(args, left_frames, right_frames)
                save_checkpoint(temp_dir, "equirect", {"num_frames": len(sbs_frames)})

            elif stage == "metadata":
                if sbs_frames is None:
                    import glob

                    eq_dir = get_temp_dir(args, "equirect")
                    files = sorted(glob.glob(os.path.join(eq_dir, "*.png")))
                    sbs_frames = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in files]
                    log.info(f"📂 Loaded {len(sbs_frames)} equirect frames from checkpoint")
                output = run_metadata_stage(args, sbs_frames)

        if output:
            log.info(f"✅ Pipeline complete → {output}")

    elif args.stage == "depth":
        frames = list(read_frames(args.input, args.max_frames))
        run_depth_stage(args, frames)

    elif args.stage == "stereo":
        frames = list(read_frames(args.input, args.max_frames))
        depth_dir = get_temp_dir(args, "depth")
        depths = []
        for i in range(len(frames)):
            d = np.load(os.path.join(depth_dir, f"depth_{i:06d}.npy"))
            depths.append(d)
        run_stereo_stage(args, frames, depths)

    elif args.stage == "equirect":
        left_dir = get_temp_dir(args, "left")
        right_dir = get_temp_dir(args, "right")
        import glob

        left_files = sorted(glob.glob(os.path.join(left_dir, "*.png")))
        right_files = sorted(glob.glob(os.path.join(right_dir, "*.png")))
        left_frames = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in left_files]
        right_frames = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in right_files]
        run_equirect_stage(args, left_frames, right_frames)

    elif args.stage == "metadata":
        eq_dir = get_temp_dir(args, "equirect")
        import glob

        files = sorted(glob.glob(os.path.join(eq_dir, "*.png")))
        frames = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in files]
        run_metadata_stage(args, frames)


if __name__ == "__main__":
    main()
