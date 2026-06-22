#!/usr/bin/env python3
"""VR180 Pipeline CLI — convert 2D AI video to VR180 immersive format.

Usage:
    # Full pipeline
    python scripts/run_pipeline.py --input video.mp4 --output vr180.mp4

    # With config
    python scripts/run_pipeline.py --input video.mp4 --output vr180.mp4 \
        --model-size base --codec h265 --fps 30

    # Individual stages with temp dir
    python scripts/run_pipeline.py --input video.mp4 --stage depth
    python scripts/run_pipeline.py --input video.mp4 --stage stereo --temp-dir frames/
    python scripts/run_pipeline.py --input video.mp4 --stage equirect
    python scripts/run_pipeline.py --input video.mp4 --stage metadata --output vr180.mp4
"""

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from pipeline.depth_estimator import DepthEstimator
from pipeline.stereo_renderer import StereoRenderer
from pipeline.equirectangular_mapper import EquirectangularMapper
from pipeline.vr_metadata import VRMetadataEmbedder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("vr180-pipeline")


def parse_args():
    parser = argparse.ArgumentParser(
        description="2D AI Video → VR180 Conversion Pipeline"
    )
    parser.add_argument("--input", "-i", required=True,
                        help="Input video file (MP4, MOV, etc.)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output VR180 video path")
    parser.add_argument("--stage", "-s",
                        choices=["all", "depth", "stereo", "equirect", "metadata"],
                        default="all",
                        help="Pipeline stage to run (default: all)")
    parser.add_argument("--model-size",
                        default="small",
                        choices=["small", "base", "large"],
                        help="Depth Anything V2 model size")
    parser.add_argument("--device", default=None,
                        help="Compute device (cuda, mps, cpu)")
    parser.add_argument("--ipd", type=float, default=0.064,
                        help="Interpupillary distance in meters")
    parser.add_argument("--max-disparity", type=float, default=0.05,
                        help="Max disparity as fraction of image width")
    parser.add_argument("--codec", choices=["h264", "h265"], default="h264",
                        help="Output video codec")
    parser.add_argument("--crf", type=int, default=23,
                        help="Constant rate factor")
    parser.add_argument("--fps", type=int, default=30,
                        help="Output frame rate")
    parser.add_argument("--output-width", type=int, default=3840,
                        help="Equirectangular output width per eye")
    parser.add_argument("--output-height", type=int, default=1920,
                        help="Equirectangular output height per eye")
    parser.add_argument("--src-hfov", type=float, default=70.0,
                        help="Source camera horizontal FOV (degrees)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Limit number of frames (for testing)")
    parser.add_argument("--no-temporal", action="store_true",
                        help="Disable temporal smoothing")
    parser.add_argument("--temp-dir", default=None,
                        help="Directory for intermediate files")
    parser.add_argument("--no-ffmpeg-v360", action="store_true",
                        help="Disable ffmpeg v360, use OpenCV fallback")
    return parser.parse_args()


def read_frames(video_path: str, max_frames: int = None):
    """Yield RGB frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames:
        total = min(total, max_frames)

    log.info(f"Video: {fps:.2f} fps, {total} frames, "
             f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
             f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

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
    """Get or create temp directory for intermediate files."""
    if args.temp_dir:
        base = Path(args.temp_dir)
    else:
        base = Path(tempfile.mkdtemp(prefix="vr180_"))
    if subdir:
        path = base / subdir
    else:
        path = base
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def run_depth_stage(args, frames):
    """Stage 1: Estimate depth for all frames."""
    log.info("=== Stage 1: Depth Estimation ===")
    estimator = DepthEstimator(
        model_size=args.model_size,
        device=args.device,
    )

    out_dir = get_temp_dir(args, "depth")
    depths = []
    for i, frame in enumerate(tqdm(frames, desc="Estimating depth")):
        depth = estimator.estimate(frame)
        depths.append(depth)
        # Save depth map for inspection
        depth_vis = (depth / depth.max() * 255).astype(np.uint8) if depth.max() > 0 else depth.astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, f"depth_{i:06d}.png"),
                    cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO))
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
        tqdm(zip(frames, depths), desc="Rendering stereo", total=len(frames))
    ):
        left, right = renderer.render(frame, depth)
        left_frames.append(left)
        right_frames.append(right)

        # Save intermediate files
        cv2.imwrite(os.path.join(left_dir, f"left_{i:06d}.png"),
                    cv2.cvtColor(left, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(right_dir, f"right_{i:06d}.png"),
                    cv2.cvtColor(right, cv2.COLOR_RGB2BGR))

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
        tqdm(zip(left_frames, right_frames),
             desc="Mapping to equirect", total=len(left_frames))
    ):
        sbs = mapper.map_stereo_pair(left, right)
        sbs_frames.append(sbs)
        cv2.imwrite(os.path.join(out_dir, f"equirect_{i:06d}.png"),
                    cv2.cvtColor(sbs, cv2.COLOR_RGB2BGR))

    log.info(f"Generated {len(sbs_frames)} equirectangular SBS frames "
             f"({sbs_frames[0].shape[1]}×{sbs_frames[0].shape[0]})")
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
        sbs_frames, output_path, width=W, height=H,
    )
    return result


def main():
    args = parse_args()

    if args.stage == "all":
        # Read frames once, pass through all stages
        frames = list(read_frames(args.input, args.max_frames))
        log.info(f"Loaded {len(frames)} frames")

        depths = run_depth_stage(args, frames)
        left_frames, right_frames = run_stereo_stage(args, frames, depths)
        sbs_frames = run_equirect_stage(args, left_frames, right_frames)
        output = run_metadata_stage(args, sbs_frames)
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
