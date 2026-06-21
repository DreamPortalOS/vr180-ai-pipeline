#!/usr/bin/env python3
"""VR180 Pipeline CLI — convert 2D AI video to VR180 immersive format.

Usage:
    # Full pipeline
    python scripts/run_pipeline.py --input video.mp4 --output vr180.mp4

    # Individual stages
    python scripts/run_pipeline.py --input video.mp4 --stage depth --output depth/
    python scripts/run_pipeline.py --input video.mp4 --stage stereo --output stereo/
    python scripts/run_pipeline.py --input video.mp4 --stage equirect --output sphere.mp4
    python scripts/run_pipeline.py --input video.mp4 --stage metadata --output vr180.mp4

    # With configuration
    python scripts/run_pipeline.py --input video.mp4 --output vr180.mp4 \\
        --depth-model depth-anything-v2 --ipd 0.064 \\
        --codec h265 --crf 20 --fps 60
"""

import argparse
import logging
import os
import sys

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
    parser.add_argument(
        "--input", "-i", required=True,
        help="Input video file (MP4, MOV, etc.)"
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Output path (video file for full pipeline, directory for depth/stage)"
    )
    parser.add_argument(
        "--stage", "-s",
        choices=["all", "depth", "stereo", "equirect", "metadata"],
        default="all",
        help="Pipeline stage to run (default: all)"
    )
    parser.add_argument(
        "--depth-model",
        default="depth-anything-v2",
        choices=["depth-anything-v2", "midas-3.1"],
        help="Depth estimation model (default: depth-anything-v2)"
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Compute device (cuda, mps, cpu). Auto-detected if omitted."
    )
    parser.add_argument(
        "--ipd", type=float, default=0.064,
        help="Interpupillary distance in meters (default: 0.064)"
    )
    parser.add_argument(
        "--max-disparity", type=float, default=0.05,
        help="Max disparity as fraction of image width (default: 0.05)"
    )
    parser.add_argument(
        "--codec", choices=["h264", "h265"], default="h264",
        help="Output video codec (default: h264)"
    )
    parser.add_argument(
        "--crf", type=int, default=23,
        help="Constant rate factor (default: 23)"
    )
    parser.add_argument(
        "--fps", type=int, default=60,
        help="Output frame rate (default: 60)"
    )
    parser.add_argument(
        "--output-width", type=int, default=3840,
        help="Equirectangular output width (default: 3840)"
    )
    parser.add_argument(
        "--output-height", type=int, default=1920,
        help="Equirectangular output height (default: 1920)"
    )
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Limit number of frames to process (for testing)"
    )
    parser.add_argument(
        "--no-temporal", action="store_true",
        help="Disable temporal smoothing in disparity"
    )
    return parser.parse_args()


def read_frames(video_path: str, max_frames: int = None):
    """Yield frames from a video file."""
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


def run_depth_stage(args):
    """Stage 1: Estimate depth for all frames and save."""
    log.info("=== Stage 1: Depth Estimation ===")
    estimator = DepthEstimator(
        model_name=args.depth_model,
        device=args.device,
    )

    os.makedirs(args.output, exist_ok=True)
    frames = list(read_frames(args.input, args.max_frames))

    for i, frame in enumerate(tqdm(frames, desc="Estimating depth")):
        depth = estimator.estimate(frame)
        np.save(os.path.join(args.output, f"depth_{i:06d}.npy"), depth)

    log.info(f"Depth maps saved to {args.output}/")
    return frames


def run_stereo_stage(args, frames=None, depths=None):
    """Stage 2: Generate stereo views."""
    log.info("=== Stage 2: Stereo Disparity Rendering ===")

    if frames is None:
        frames = list(read_frames(args.input, args.max_frames))

    if depths is None:
        depth_dir = args.output
        depths = []
        for i in range(len(frames)):
            d = np.load(os.path.join(depth_dir, f"depth_{i:06d}.npy"))
            depths.append(d)

    renderer = StereoRenderer(
        ipd=args.ipd,
        max_disparity=args.max_disparity,
        temporal_smooth=not args.no_temporal,
    )

    os.makedirs(os.path.join(args.output, "left"), exist_ok=True)
    os.makedirs(os.path.join(args.output, "right"), exist_ok=True)

    for i, (frame, depth) in enumerate(
        tqdm(zip(frames, depths), desc="Rendering stereo", total=len(frames))
    ):
        left, right = renderer.render(frame, depth)
        cv2.imwrite(
            os.path.join(args.output, "left", f"left_{i:06d}.png"),
            cv2.cvtColor(left, cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(
            os.path.join(args.output, "right", f"right_{i:06d}.png"),
            cv2.cvtColor(right, cv2.COLOR_RGB2BGR),
        )

    log.info(f"Stereo views saved to {args.output}/left/ and {args.output}/right/")
    return frames, depths


def run_equirect_stage(args, frames=None):
    """Stage 3: Map stereo views to equirectangular."""
    log.info("=== Stage 3: Equirectangular Projection ===")

    mapper = EquirectangularMapper(
        output_width=args.output_width,
        output_height=args.output_height,
    )

    if frames is None:
        frames = list(read_frames(args.input, args.max_frames))

    stereo_dir = args.output
    equirect_frames = []

    for i in tqdm(range(len(frames)), desc="Mapping to equirect"):
        left = cv2.imread(
            os.path.join(stereo_dir, "left", f"left_{i:06d}.png")
        )
        right = cv2.imread(
            os.path.join(stereo_dir, "right", f"right_{i:06d}.png")
        )
        if left is None or right is None:
            log.warning(f"Missing frame pair {i}, skipping")
            continue

        left = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
        right = cv2.cvtColor(right, cv2.COLOR_BGR2RGB)

        sbs = mapper.map_stereo_pair(left, right)
        equirect_frames.append(sbs)

    log.info(f"Generated {len(equirect_frames)} equirectangular frames")
    return equirect_frames


def run_metadata_stage(args, frames=None):
    """Stage 4: Encode video with VR metadata."""
    log.info("=== Stage 4: VR Metadata Embedding ===")

    embedder = VRMetadataEmbedder(
        codec=args.codec,
        crf=args.crf,
        fps=args.fps,
    )

    if frames is None:
        log.info("Reading frames from directory...")
        frames = []
        # Read from output directory SBS PNGs
        import glob
        files = sorted(glob.glob(os.path.join(args.output, "equirect_*.png")))
        for f in tqdm(files, desc="Loading frames"):
            img = cv2.imread(f)
            if img is not None:
                frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    log.info(f"Encoding {len(frames)} frames → {args.output}")
    result = embedder.embed_single_frame_batch(
        frames, args.output,
        width=frames[0].shape[1] if frames else 7680,
        height=frames[0].shape[0] if frames else 1920,
    )
    return result


def main():
    args = parse_args()

    if args.stage == "all":
        # Full pipeline
        frames = run_depth_stage(args)
        frames, depths = run_stereo_stage(args, frames=frames)
        equirect_frames = run_equirect_stage(args, frames=frames)
        run_metadata_stage(args, frames=equirect_frames)
        log.info(f"✅ Pipeline complete → {args.output}")
    elif args.stage == "depth":
        run_depth_stage(args)
    elif args.stage == "stereo":
        run_stereo_stage(args)
    elif args.stage == "equirect":
        run_equirect_stage(args)
    elif args.stage == "metadata":
        run_metadata_stage(args)


if __name__ == "__main__":
    main()
