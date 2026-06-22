#!/usr/bin/env python3
"""
Multi-Hypothesis Inversion Matrix Test
=======================================

Generates 4 test clips with different flip combinations to determine
the correct vertical orientation for Quest 3 playback.

Combinations:
1. Original (no flip)
2. cv2.flip (vertical flip) on each frame
3. ffmpeg vflip filter
4. ffmpeg v360 pitch=180 (rotate 180 degrees around horizontal axis)

Usage:
    python pipeline/research/test_inversion_matrix.py [input_video]

Defaults to video/testfpv_vr180.mp4 if no input provided.
Output videos saved to video/inversion_tests/
"""

import cv2
import numpy as np
import subprocess
import os
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("inversion-test")

# Create output directory
OUTPUT_DIR = Path(__file__).parent.parent.parent / "video" / "inversion_tests"
OUTPUT_DIR.mkdir(exist_ok=True)

def get_video_info(video_path):
    """Extract video metadata using ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", str(video_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    import json
    data = json.loads(result.stdout)
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            return {
                "width": int(stream.get("width", 0)),
                "height": int(stream.get("height", 0)),
                "duration": float(stream.get("duration", 0)),
                "fps": eval(stream.get("r_frame_rate", "30/1"))
            }
    return None

def apply_cv2_flip(input_path, output_path, flip_code=0):
    """Apply vertical flip using OpenCV (frame-by-frame)."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        log.error(f"Cannot open {input_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        flipped = cv2.flip(frame, flip_code)
        out.write(flipped)
        frame_count += 1
        if frame_count % 30 == 0:
            log.info(f"Processed {frame_count} frames...")

    cap.release()
    out.release()
    log.info(f"cv2.flip (code={flip_code}) applied: {frame_count} frames -> {output_path}")
    return True

def apply_ffmpeg_vflip(input_path, output_path):
    """Apply vertical flip using ffmpeg vflip filter."""
    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-vf", "vflip",
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-an", str(output_path)
    ]
    log.info(f"Running ffmpeg vflip: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        log.info(f"ffmpeg vflip applied -> {output_path}")
        return True
    else:
        log.error(f"ffmpeg vflip failed: {result.stderr[-500:]}")
        return False

def apply_ffmpeg_v360_pitch(input_path, output_path, pitch=180):
    """Apply v360 pitch rotation using ffmpeg."""
    # v360 filter with pitch=180 rotates the equirectangular projection
    # vertically, effectively flipping the top and bottom.
    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-vf", f"v360=e:e:pitch={pitch}",
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-an", str(output_path)
    ]
    log.info(f"Running ffmpeg v360 (pitch={pitch}): {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        log.info(f"ffmpeg v360 pitch={pitch} applied -> {output_path}")
        return True
    else:
        log.error(f"ffmpeg v360 failed: {result.stderr[-500:]}")
        return False

def apply_ffmpeg_v360_roll(input_path, output_path, roll=180):
    """Apply v360 roll rotation using ffmpeg."""
    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-vf", f"v360=e:e:roll={roll}",
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-an", str(output_path)
    ]
    log.info(f"Running ffmpeg v360 (roll={roll}): {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        log.info(f"ffmpeg v360 roll={roll} applied -> {output_path}")
        return True
    else:
        log.error(f"ffmpeg v360 roll failed: {result.stderr[-500:]}")
        return False

def main():
    input_video = sys.argv[1] if len(sys.argv) > 1 else "video/testfpv_vr180.mp4"
    if not os.path.exists(input_video):
        log.error(f"Input video not found: {input_video}")
        sys.exit(1)

    log.info(f"Input video: {input_video}")
    info = get_video_info(input_video)
    if info:
        log.info(f"Video info: {info['width']}x{info['height']} @ {info['fps']:.2f}fps, {info['duration']:.2f}s")
    else:
        log.warning("Could not read video info via ffprobe")

    # Define test matrix
    tests = [
        ("01_original", None, "No transformation (baseline)"),
        ("02_cv2_flip_vertical", lambda i, o: apply_cv2_flip(i, o, flip_code=0), "cv2.flip(frame, 0) - vertical flip around x-axis"),
        ("03_ffmpeg_vflip", apply_ffmpeg_vflip, "ffmpeg vflip filter"),
        ("04_ffmpeg_v360_pitch180", lambda i, o: apply_ffmpeg_v360_pitch(i, o, pitch=180), "ffmpeg v360=e:e:pitch=180"),
        ("05_ffmpeg_v360_roll180", lambda i, o: apply_ffmpeg_v360_roll(i, o, roll=180), "ffmpeg v360=e:e:roll=180"),
    ]

    # Generate test clips
    for name, transform_func, description in tests:
        output_path = OUTPUT_DIR / f"{name}.mp4"
        log.info(f"\n{'='*60}")
        log.info(f"Test: {name}")
        log.info(f"Description: {description}")

        if transform_func is None:
            # Copy original
            import shutil
            shutil.copy2(input_video, output_path)
            log.info(f"Copied original -> {output_path}")
        else:
            success = transform_func(input_video, output_path)
            if not success:
                log.error(f"Failed to generate {name}")

    log.info(f"\n{'='*60}")
    log.info(f"All test clips generated in: {OUTPUT_DIR}")
    log.info("Transfer these to Quest 3 and test playback orientation.")
    log.info("Expected correct orientation: right-side-up when viewed in VR.")

    # Generate summary document
    summary = OUTPUT_DIR / "README.md"
    with open(summary, "w") as f:
        f.write("# Inversion Matrix Test Results\n\n")
        f.write(f"Generated from: `{input_video}`\n\n")
        f.write("| # | File | Transformation | Description |\n")
        f.write("|---|------|----------------|-------------|\n")
        for i, (name, _, desc) in enumerate(tests, 1):
            f.write(f"| {i} | `{name}.mp4` | {name[3:]} | {desc} |\n")
        f.write("\n## Test Instructions\n")
        f.write("1. Copy all MP4 files to Quest 3\n")
        f.write("2. Open each in a VR video player (e.g., Skybox, Pigasus)\n")
        f.write("3. Note which video appears correctly right-side-up\n")
        f.write("4. The winning transformation will be integrated into the pipeline\n")

    log.info(f"Summary written to {summary}")

if __name__ == "__main__":
    main()