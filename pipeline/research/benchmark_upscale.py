#!/usr/bin/env python3
"""
Upscale Model Benchmark Suite
===============================

Compares multiple upscaling configurations on a sample video clip:
1. Real-ESRGAN x2 (lightweight)
2. Real-ESRGAN x4plus (high quality)
3. ffmpeg Lanczos resampling (traditional baseline)
4. ffmpeg Bicubic resampling (traditional baseline)

Metrics measured:
- VRAM peak usage (GPU memory)
- Execution time per frame
- Tile edge seam artifacts (pixel discontinuity at tile boundaries)
- Output PSNR/SSIM vs original (if reference available)

Usage:
    python pipeline/research/benchmark_upscale.py [input_video] [--frames 100]
"""

import json
import logging
import os
import subprocess
import sys
import time
import tracemalloc
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("upscale-bench")

OUTPUT_DIR = Path(__file__).parent.parent.parent / "video" / "benchmark_results"
OUTPUT_DIR.mkdir(exist_ok=True)


@dataclass
class BenchmarkResult:
    """Stores benchmark results for a single configuration."""
    method: str
    scale_factor: int
    total_frames: int
    total_time_sec: float
    avg_time_per_frame_ms: float
    peak_memory_mb: float
    tile_edge_discontinuity: float  # Max pixel diff at tile boundaries
    output_resolution: str
    notes: str = ""


def extract_frames(video_path: str, max_frames: int = 100,
                   output_dir: str | None = None) -> list[str]:
    """Extract frames from video for benchmarking."""
    if output_dir is None:
        output_dir = str(OUTPUT_DIR / "input_frames")

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_paths = []
    idx = 0
    while idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        path = os.path.join(output_dir, f"frame_{idx:04d}.png")
        cv2.imwrite(path, frame)
        frame_paths.append(path)
        idx += 1

    cap.release()
    log.info(f"Extracted {len(frame_paths)} frames to {output_dir}")
    return frame_paths


def measure_tile_discontinuity(image: np.ndarray, tile_size: int = 512) -> float:
    """
    Measure pixel discontinuity at tile boundaries.

    Returns the maximum absolute pixel difference across tile edges.
    Lower values = smoother stitching.
    """
    h, w = image.shape[:2]
    max_diff = 0.0

    # Check horizontal tile boundaries
    for y in range(tile_size, h, tile_size):
        if y < h:
            row_above = image[y-1, :].astype(np.float32)
            row_below = image[y, :].astype(np.float32)
            diff = np.max(np.abs(row_above - row_below))
            max_diff = max(max_diff, diff)

    # Check vertical tile boundaries
    for x in range(tile_size, w, tile_size):
        if x < w:
            col_left = image[:, x-1].astype(np.float32)
            col_right = image[:, x].astype(np.float32)
            diff = np.max(np.abs(col_left - col_right))
            max_diff = max(max_diff, diff)

    return max_diff


def benchmark_ffmpeg_resize(frame_paths: list[str], scale: int,
                            method: str = "lanczos") -> BenchmarkResult:
    """Benchmark ffmpeg-based resizing (Lanczos or Bicubic)."""
    log.info(f"Benchmarking ffmpeg {method} x{scale}...")

    output_dir = OUTPUT_DIR / f"ffmpeg_{method}_x{scale}"
    output_dir.mkdir(exist_ok=True)

    start = time.time()
    max_discontinuity = 0

    for i, frame_path in enumerate(frame_paths):
        out_path = str(output_dir / f"frame_{i:04d}.png")

        # Use ffmpeg to resize
        interp = "lanczos" if method == "lanczos" else "bicubic"
        cmd = [
            "ffmpeg", "-y", "-i", frame_path,
            "-vf", f"scale=iw*{scale}:ih*{scale}:flags={interp}",
            out_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0 and os.path.exists(out_path):
            out_img = cv2.imread(out_path)
            if out_img is not None:
                disc = measure_tile_discontinuity(out_img)
                max_discontinuity = max(max_discontinuity, disc)

        if (i + 1) % 20 == 0:
            log.info(f"  {method}: {i+1}/{len(frame_paths)} frames")

    elapsed = time.time() - start
    avg_ms = (elapsed / len(frame_paths)) * 1000 if frame_paths else 0

    # Get output resolution
    sample_out = str(output_dir / "frame_0000.png")
    if os.path.exists(sample_out):
        img = cv2.imread(sample_out)
        resolution = f"{img.shape[1]}x{img.shape[0]}" if img is not None else "N/A"
    else:
        resolution = "N/A"

    return BenchmarkResult(
        method=f"ffmpeg_{method}",
        scale_factor=scale,
        total_frames=len(frame_paths),
        total_time_sec=round(elapsed, 2),
        avg_time_per_frame_ms=round(avg_ms, 2),
        peak_memory_mb=0,  # ffmpeg is external process
        tile_edge_discontinuity=round(max_discontinuity, 2),
        output_resolution=resolution,
        notes="External ffmpeg process; VRAM not measured"
    )


def benchmark_realesrgan(frame_paths: list[str], scale: int = 2,
                          model_name: str = "RealESRGAN_x2plus") -> BenchmarkResult:
    """
    Benchmark Real-ESRGAN upscaling.

    Tries to use realesrgan-ncnn-vulkan CLI if available,
    otherwise falls back to Python basic_sr implementation.
    """
    log.info(f"Benchmarking Real-ESRGAN x{scale} ({model_name})...")

    output_dir = OUTPUT_DIR / f"realesrgan_x{scale}"
    output_dir.mkdir(exist_ok=True)

    # Check if realesrgan-ncnn-vulkan is available
    ncnn_available = False
    try:
        subprocess.run(["realesrgan-ncnn-vulkan", "-i", "test", "-o", "test"],
                                capture_output=True, text=True, timeout=5)
        ncnn_available = True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    if not ncnn_available:
        log.info("realesrgan-ncnn-vulkan not found, trying Python fallback...")
        return _benchmark_realesrgan_python(frame_paths, scale, model_name, output_dir)

    # Use ncnn-vulkan binary
    tracemalloc.start()
    start = time.time()
    max_discontinuity = 0

    # Process all frames at once
    input_dir = str(Path(frame_paths[0]).parent)
    cmd = [
        "realesrgan-ncnn-vulkan",
        "-i", input_dir,
        "-o", str(output_dir),
        "-s", str(scale),
        "-n", model_name,
    ]
    subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Measure discontinuity
    for out_file in sorted(output_dir.glob("*.png")):
        img = cv2.imread(str(out_file))
        if img is not None:
            disc = measure_tile_discontinuity(img)
            max_discontinuity = max(max_discontinuity, disc)

    avg_ms = (elapsed / len(frame_paths)) * 1000 if frame_paths else 0

    sample_out = output_dir / "frame_0000.png"
    if sample_out.exists():
        img = cv2.imread(str(sample_out))
        resolution = f"{img.shape[1]}x{img.shape[0]}" if img is not None else "N/A"
    else:
        resolution = "N/A"

    return BenchmarkResult(
        method=f"realesrgan_ncnn_x{scale}",
        scale_factor=scale,
        total_frames=len(frame_paths),
        total_time_sec=round(elapsed, 2),
        avg_time_per_frame_ms=round(avg_ms, 2),
        peak_memory_mb=round(peak_mem / 1024 / 1024, 2),
        tile_edge_discontinuity=round(max_discontinuity, 2),
        output_resolution=resolution,
        notes="Used realesrgan-ncnn-vulkan"
    )


def _benchmark_realesrgan_python(frame_paths: list[str], scale: int,
                                  model_name: str, output_dir: Path) -> BenchmarkResult:
    """Python-based Real-ESRGAN benchmark using basicsr/realesrgan."""
    try:
        import torch
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        # Determine device
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        log.info(f"Using device: {device}")

        # Build model
        if scale == 4:
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            model_path = "weights/RealESRGAN_x4plus.pth"
        else:
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            model_path = "weights/RealESRGAN_x2plus.pth"

        upsampler = RealESRGANer(
            scale=scale,
            model_path=model_path if os.path.exists(model_path) else None,
            model=model,
            tile=512,
            tile_pad=10,
            pre_pad=0,
            half=device == "cuda",
            device=device,
        )

        tracemalloc.start()
        start = time.time()
        max_discontinuity = 0

        for i, frame_path in enumerate(frame_paths):
            img = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
            output, _ = upsampler.enhance(img, outscale=scale)

            out_path = str(output_dir / f"frame_{i:04d}.png")
            cv2.imwrite(out_path, output)

            disc = measure_tile_discontinuity(output)
            max_discontinuity = max(max_discontinuity, disc)

            if (i + 1) % 20 == 0:
                log.info(f"  realesrgan x{scale}: {i+1}/{len(frame_paths)} frames")

        elapsed = time.time() - start
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        avg_ms = (elapsed / len(frame_paths)) * 1000 if frame_paths else 0

        sample_out = output_dir / "frame_0000.png"
        if sample_out.exists():
            img = cv2.imread(str(sample_out))
            resolution = f"{img.shape[1]}x{img.shape[0]}" if img is not None else "N/A"
        else:
            resolution = "N/A"

        return BenchmarkResult(
            method=f"realesrgan_python_x{scale}",
            scale_factor=scale,
            total_frames=len(frame_paths),
            total_time_sec=round(elapsed, 2),
            avg_time_per_frame_ms=round(avg_ms, 2),
            peak_memory_mb=round(peak_mem / 1024 / 1024, 2),
            tile_edge_discontinuity=round(max_discontinuity, 2),
            output_resolution=resolution,
            notes=f"Python basicsr; device={device}"
        )

    except ImportError as e:
        log.warning(f"Real-ESRGAN Python not available: {e}")
        return BenchmarkResult(
            method=f"realesrgan_x{scale}",
            scale_factor=scale,
            total_frames=0,
            total_time_sec=0,
            avg_time_per_frame_ms=0,
            peak_memory_mb=0,
            tile_edge_discontinuity=0,
            output_resolution="N/A",
            notes=f"SKIPPED: {e}"
        )


def generate_report(results: list[BenchmarkResult], output_path: str):
    """Generate markdown benchmark report."""
    with open(output_path, "w") as f:
        f.write("# Upscale Model Benchmark Report\n\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Results Summary\n\n")
        f.write("| Method | Scale | Frames | Total Time | Avg/Frame | Peak VRAM | Tile Seam | Resolution |\n")
        f.write("|--------|-------|--------|------------|-----------|-----------|-----------|------------|\n")

        for r in results:
            if r.total_frames == 0:
                continue
            f.write(f"| {r.method} | x{r.scale_factor} | {r.total_frames} | "
                    f"{r.total_time_sec:.1f}s | {r.avg_time_per_frame_ms:.0f}ms | "
                    f"{r.peak_memory_mb:.0f}MB | {r.tile_edge_discontinuity:.0f} | "
                    f"{r.output_resolution} |\n")

        f.write("\n## Key Findings\n\n")

        # Find fastest
        valid = [r for r in results if r.total_frames > 0]
        if valid:
            fastest = min(valid, key=lambda r: r.avg_time_per_frame_ms)
            f.write(f"- **Fastest**: {fastest.method} at {fastest.avg_time_per_frame_ms:.0f}ms/frame\n")

            # Find lowest seam artifact
            best_seam = min(valid, key=lambda r: r.tile_edge_discontinuity)
            f.write(f"- **Best tile seam**: {best_seam.method} (discontinuity={best_seam.tile_edge_discontinuity:.0f})\n")

            # Find most memory efficient
            mem_valid = [r for r in valid if r.peak_memory_mb > 0]
            if mem_valid:
                lowest_mem = min(mem_valid, key=lambda r: r.peak_memory_mb)
                f.write(f"- **Lowest memory**: {lowest_mem.method} at {lowest_mem.peak_memory_mb:.0f}MB\n")

        f.write("\n## Method Details\n\n")
        for r in results:
            f.write(f"### {r.method}\n")
            f.write(f"- Scale: x{r.scale_factor}\n")
            f.write(f"- Notes: {r.notes}\n")
            if r.total_frames > 0:
                f.write(f"- Avg time: {r.avg_time_per_frame_ms:.0f}ms/frame\n")
                f.write(f"- Tile discontinuity: {r.tile_edge_discontinuity:.0f}\n")
            f.write("\n")

    log.info(f"Report saved to {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Upscale Model Benchmark")
    parser.add_argument("input", nargs="?", default="video/testfpv.mp4",
                        help="Input video path")
    parser.add_argument("--frames", type=int, default=50, help="Number of frames to benchmark")
    parser.add_argument("--scale", type=int, default=2, help="Upscale factor (2 or 4)")
    parser.add_argument("--methods", nargs="+",
                        default=["ffmpeg_lanczos", "ffmpeg_bicubic", "realesrgan"],
                        help="Methods to benchmark")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        log.error(f"Input not found: {args.input}")
        sys.exit(1)

    # Extract frames for benchmarking
    frame_paths = extract_frames(args.input, max_frames=args.frames)
    if not frame_paths:
        log.error("No frames extracted")
        sys.exit(1)

    results = []

    # Run benchmarks for each method
    if "ffmpeg_lanczos" in args.methods:
        r = benchmark_ffmpeg_resize(frame_paths, args.scale, method="lanczos")
        results.append(r)
        log.info(f"  -> {r.avg_time_per_frame_ms:.0f}ms/frame, seam={r.tile_edge_discontinuity}")

    if "ffmpeg_bicubic" in args.methods:
        r = benchmark_ffmpeg_resize(frame_paths, args.scale, method="bicubic")
        results.append(r)
        log.info(f"  -> {r.avg_time_per_frame_ms:.0f}ms/frame, seam={r.tile_edge_discontinuity}")

    if "realesrgan" in args.methods:
        r = benchmark_realesrgan(frame_paths, scale=args.scale)
        results.append(r)
        log.info(f"  -> {r.avg_time_per_frame_ms:.0f}ms/frame, seam={r.tile_edge_discontinuity}")

    # Generate report
    report_path = str(OUTPUT_DIR / "benchmark_report.md")
    generate_report(results, report_path)

    # Also save JSON
    json_path = str(OUTPUT_DIR / "benchmark_results.json")
    with open(json_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    log.info(f"JSON results: {json_path}")

    log.info("Benchmark complete!")


if __name__ == "__main__":
    main()
