#!/usr/bin/env python3
"""VR180 Orientation Matrix Diagnostic Harness (Task 1.2).

Programmatically tests all combinations of:
- cv2.flip(img, 0)  — vertical flip (upside down)
- cv2.flip(img, 1)  — horizontal flip (mirror)
- cv2.flip(img, -1) — both flips (180° rotation)
- ffmpeg transpose filters (0,1,2,3)
- No-op (identity)

Generates a diagnostic video grid showing all orientation variations
for a given input frame, so the correct VR180 orientation can be identified.

Usage:
    python -m pipeline.research.orientation_matrix --input video.mp4 --output orientation_grid.mp4
    python -m pipeline.research.orientation_matrix --input video.mp4 --frame 50 --output diag.png
"""

import argparse
import logging
import os
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("orientation-matrix")

# ── Orientation definitions ──────────────────────────────────────────────────

FLIP_LABELS = {
    "none": "Original",
    "vflip": "Vertical Flip (cv2.flip 0)",
    "hflip": "Horizontal Flip (cv2.flip 1)",
    "both": "Both Flips (cv2.flip -1)",
}

TRANSPOSE_LABELS = {
    "none": "No Transpose",
    "t0": "transpose=0 (90° CW)",
    "t1": "transpose=1 (90° CCW + hflip)",
    "t2": "transpose=2 (90° CW + hflip)",
    "t3": "transpose=3 (90° CCW)",
}

COMBO_LABELS = {}  # populated at runtime


def apply_flip(frame: np.ndarray, flip_type: str) -> np.ndarray:
    """Apply OpenCV flip to a frame.

    Args:
        frame: RGB image (H, W, 3)
        flip_type: one of 'none', 'vflip', 'hflip', 'both'

    Returns:
        Flipped frame
    """
    if flip_type == "none":
        return frame.copy()
    elif flip_type == "vflip":
        return cv2.flip(frame, 0)
    elif flip_type == "hflip":
        return cv2.flip(frame, 1)
    elif flip_type == "both":
        return cv2.flip(frame, -1)
    else:
        raise ValueError(f"Unknown flip type: {flip_type}")


def apply_transpose(frame: np.ndarray, transpose_type: str) -> np.ndarray:
    """Apply ffmpeg-style transpose operation using OpenCV.

    Transpose codes (matching ffmpeg):
        0 = 90° counter-clockwise and vertical flip (equivalent to rot90 + vflip)
        1 = 90° clockwise
        2 = 90° counter-clockwise
        3 = 90° clockwise and vertical flip

    Args:
        frame: RGB image (H, W, 3)
        transpose_type: one of 'none', 't0', 't1', 't2', 't3'

    Returns:
        Transposed frame
    """
    if transpose_type == "none":
        return frame.copy()
    elif transpose_type == "t0":
        # ffmpeg transpose=0: 90° CCW + vflip = rotate 90° CW
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif transpose_type == "t1":
        # ffmpeg transpose=1: 90° CW
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif transpose_type == "t2":
        # ffmpeg transpose=2: 90° CCW
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif transpose_type == "t3":
        # ffmpeg transpose=3: 90° CW + vflip = rotate 90° CCW
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        raise ValueError(f"Unknown transpose type: {transpose_type}")


def generate_orientation_matrix(
    frame: np.ndarray,
    flip_types: list[str] | None = None,
    transpose_types: list[str] | None = None,
) -> tuple[np.ndarray, dict]:
    """Generate a grid of all orientation combinations.

    Args:
        frame: RGB input frame
        flip_types: list of flip operations to test
        transpose_types: list of transpose operations to test

    Returns:
        Tuple of (grid_image, combo_map) where combo_map maps
        (row, col) → (label, transformed_frame)
    """
    if flip_types is None:
        flip_types = ["none", "vflip", "hflip", "both"]
    if transpose_types is None:
        transpose_types = ["none", "t0", "t1", "t2", "t3"]

    n_rows = len(transpose_types)
    n_cols = len(flip_types)

    # Normalize all tiles to same size (use original frame size)
    h, w = frame.shape[:2]
    tile_h, tile_w = h, w

    # Add padding for labels
    label_h = 40
    grid_h = n_rows * (tile_h + label_h)
    grid_w = n_cols * tile_w

    grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    combo_map = {}

    for row, t_type in enumerate(transpose_types):
        for col, f_type in enumerate(flip_types):
            # Apply transforms: flip first, then transpose
            flipped = apply_flip(frame, f_type)
            transposed = apply_transpose(flipped, t_type)

            # Resize to tile size if dimensions changed
            if transposed.shape[:2] != (tile_h, tile_w):
                transposed = cv2.resize(transposed, (tile_w, tile_h))

            y_offset = row * (tile_h + label_h)
            x_offset = col * tile_w

            grid[y_offset : y_offset + tile_h, x_offset : x_offset + tile_w] = transposed

            # Draw label
            label = f"{FLIP_LABELS[f_type]} + {TRANSPOSE_LABELS[t_type]}"
            short_label = f"{f_type}+{t_type}"
            cv2.putText(
                grid,
                short_label,
                (x_offset + 5, y_offset + tile_h + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            combo_map[(row, col)] = {
                "label": label,
                "short": short_label,
                "flip": f_type,
                "transpose": t_type,
                "frame": transposed,
            }

            log.info(f"  [{row},{col}] {short_label}: {label}")

    return grid, combo_map


def save_individual_tiles(combo_map: dict, output_dir: str):
    """Save each orientation variant as an individual image for inspection."""
    os.makedirs(output_dir, exist_ok=True)
    for (row, col), info in combo_map.items():
        fname = f"orient_r{row}_c{col}_{info['short']}.png"
        path = os.path.join(output_dir, fname)
        bgr = cv2.cvtColor(info["frame"], cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, bgr)
    log.info(f"Saved {len(combo_map)} tile images to {output_dir}/")


def generate_ffmpeg_filter_map(combo_map: dict) -> str:
    """Generate equivalent ffmpeg filter strings for each orientation combo.

    Returns a human-readable report of ffmpeg equivalents.
    """
    lines = ["# FFmpeg Filter Equivalents", ""]
    for (row, col), info in combo_map.items():
        flip = info["flip"]
        transpose = info["transpose"]
        filters = []
        if flip == "vflip":
            filters.append("vflip")
        elif flip == "hflip":
            filters.append("hflip")
        elif flip == "both":
            filters.append("hflip,vflip")

        if transpose == "t0":
            filters.append("transpose=0")
        elif transpose == "t1":
            filters.append("transpose=1")
        elif transpose == "t2":
            filters.append("transpose=2")
        elif transpose == "t3":
            filters.append("transpose=3")

        filter_str = ",".join(filters) if filters else "(no filter)"
        lines.append(f'  [{row},{col}] {info["label"]}: -vf "{filter_str}"')

    return "\n".join(lines)


def run_orientation_matrix(
    input_path: str,
    output_path: str | None = None,
    frame_idx: int = 0,
    save_tiles: bool = True,
) -> str:
    """Main entry: generate orientation diagnostic grid from a video.

    Args:
        input_path: Path to input video
        output_path: Output path for grid image (PNG) or video
        frame_idx: Which frame to extract (0-indexed)
        save_tiles: Whether to save individual tile images

    Returns:
        Path to output grid image
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    # Seek to desired frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame_bgr = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(f"Cannot read frame {frame_idx} from {input_path}")

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    log.info(f"Extracted frame {frame_idx}: {frame_rgb.shape[1]}×{frame_rgb.shape[0]}")

    # Generate matrix
    log.info("Generating orientation matrix (4 flips × 5 transposes = 20 combinations)...")
    grid, combo_map = generate_orientation_matrix(frame_rgb)

    # Save grid
    if output_path is None:
        stem = Path(input_path).stem
        output_path = f"{stem}_orientation_grid.png"

    grid_bgr = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, grid_bgr)
    log.info(f"Orientation grid saved to {output_path}")

    # Save individual tiles
    if save_tiles:
        tile_dir = str(Path(output_path).parent / "orientation_tiles")
        save_individual_tiles(combo_map, tile_dir)

    # Print ffmpeg equivalents
    ffmpeg_report = generate_ffmpeg_filter_map(combo_map)
    report_path = str(Path(output_path).with_suffix(".txt"))
    with open(report_path, "w") as f:
        f.write(ffmpeg_report)
    log.info(f"FFmpeg filter report saved to {report_path}")
    print(ffmpeg_report)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="VR180 Orientation Matrix Diagnostic Tool")
    parser.add_argument("--input", "-i", required=True, help="Input video file")
    parser.add_argument("--output", "-o", default=None, help="Output grid image path")
    parser.add_argument("--frame", type=int, default=0, help="Frame index to extract (default: 0)")
    parser.add_argument("--no-tiles", action="store_true", help="Skip saving individual tile images")

    args = parser.parse_args()
    run_orientation_matrix(
        args.input,
        output_path=args.output,
        frame_idx=args.frame,
        save_tiles=not args.no_tiles,
    )


if __name__ == "__main__":
    main()
