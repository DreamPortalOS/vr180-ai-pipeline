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
import datetime
import json
import logging
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

# K-15 (#205): let this script run directly (``python scripts/run_pipeline.py``)
# without the caller having to set PYTHONPATH — put the repo root on sys.path
# before importing the ``pipeline`` package.  Idempotent (no duplicate entries)
# and a no-op when PYTHONPATH already points here.  The imports below are now
# module-level-but-post-statement, so each is silenced for E402 (repo
# convention, see tests/test_batch_runner.py etc.) rather than being
# reshuffled — order is load-bearing for the existing call graph.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from tqdm import tqdm  # noqa: E402

from pipeline.comfort_presets import COMFORT_PRESETS, DEFAULT_COMFORT, resolve_comfort  # noqa: E402
from pipeline.depth_crafter import DepthCrafterEstimator  # noqa: E402
from pipeline.depth_estimator import DepthEstimator  # noqa: E402
from pipeline.device_utils import detect_best_device, resolve_device  # noqa: E402
from pipeline.equirectangular_mapper import EquirectangularMapper  # noqa: E402
from pipeline.fulldome_mapper import FulldomeMapper  # noqa: E402
from pipeline.outpainter import Outpainter  # noqa: E402
from pipeline.playback_presets import (  # noqa: E402
    DEFAULT_PLAYBACK,
    PLAYBACK_PRESETS,
    resolve_playback,
)
from pipeline.segment_concat import (  # noqa: E402
    ConcatError,
    ConcatSegment,
    concat_segments,
)
from pipeline.stereo_crafter import StereoCrafterRenderer  # noqa: E402
from pipeline.stereo_renderer import StereoRenderer  # noqa: E402
from pipeline.streaming_pipeline import (  # noqa: E402
    DEFAULT_QUALITY,
    StreamingPipeline,
    resolve_quality,
    scaled_bitrate_mbps,
)
from pipeline.upscaler import PixelUpscaler  # noqa: E402
from pipeline.video_upscaler import SeedVR2Upscaler  # noqa: E402
from pipeline.vr_metadata import VRMetadataEmbedder  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("vr180-pipeline")


def parse_args(argv: list[str] | None = None):
    """Parse CLI arguments.  Accept optional *argv* for testing."""
    parser = argparse.ArgumentParser(description="2D AI Video → VR180 Conversion Pipeline")

    # C-1b (#191): single input vs multi-segment concat are mutually exclusive
    # entry modes.  Exactly one must be provided — the mutual-exclusion group
    # enforces that ``--input`` and ``--inputs`` are not both given; the "at
    # least one" check lives in main() so the error message can name both
    # flags rather than argparse's terse per-group message.
    _input_group = parser.add_mutually_exclusive_group()
    _input_group.add_argument(
        "--input",
        "-i",
        required=False,
        help="Input video file (MP4, MOV, etc.). Mutually exclusive with --inputs.",
    )
    _input_group.add_argument(
        "--inputs",
        nargs="+",
        default=None,
        help=(
            "C-1b: one or more pre-generated flat (pre-VR180) clips, given in "
            "the order they should appear.  They are concatenated into a single "
            "intermediate file (see --temp-dir) and that intermediate is fed to "
            "the existing VR180 pipeline unchanged.  Mutually exclusive with --input/-i."
        ),
    )
    # --concat-crossfade / --concat-mode configure the concat pre-stage; they
    # are NOT entry-mode selectors, so they must NOT be in the mutual-exclusion
    # group (they must be passable alongside --inputs).  They are no-ops when
    # --inputs is not given.
    parser.add_argument(
        "--concat-crossfade",
        type=float,
        default=0.0,
        help="C-1b: xfade transition length in seconds between concatenated segments (default 0.0 = hard cut).",
    )
    parser.add_argument(
        "--concat-mode",
        choices=["demux", "filter"],
        default="demux",
        help="C-1b: concat engine (demux = lossless -c copy, filter = re-encode; default demux).",
    )
    parser.add_argument("--output", "-o", default=None, help="Output VR180 video path")
    parser.add_argument(
        "--stage",
        "-s",
        choices=["all", "depth", "stereo", "equirect", "outpaint", "metadata"],
        default="all",
        help="Pipeline stage to run (default: all)",
    )
    parser.add_argument(
        "--model-size", default="small", choices=["small", "base", "large"], help="Depth Anything V2 model size"
    )
    parser.add_argument("--device", default=None, help="Compute device (cuda, mps, cpu)")
    parser.add_argument("--ipd", type=float, default=0.064, help="Interpupillary distance in meters")

    # I-3 (#88): comfort presets — owner-tuned low-parallax tiers.  A preset is
    # a starting point: an explicit --max-disparity / --convergence /
    # --no-temporal always wins over the preset (see resolve_comfort).  The
    # three flags below use None as a sentinel so the preset can tell "user did
    # not set this" from "user explicitly chose 0".
    parser.add_argument(
        "--comfort",
        choices=sorted(COMFORT_PRESETS),
        default=DEFAULT_COMFORT,
        help=(
            f"Comfort preset for stereo parallax strength (default: {DEFAULT_COMFORT}). "
            f"safe = low disparity, far convergence — '久看不晕', weak 3D; "
            f"balanced = middle ground, survives shaky depth; "
            f"strong = max pop, needs solid depth (DepthCrafter/StereoCrafter). "
            "Explicit --max-disparity/--convergence/--no-temporal override the preset."
        ),
    )
    parser.add_argument(
        "--max-disparity",
        type=float,
        default=None,
        help=(
            "Max disparity as fraction of image width. Overrides the --comfort preset's value; "
            "when omitted the preset fills it in (default balanced = 0.035)."
        ),
    )
    parser.add_argument(
        "--convergence",
        type=float,
        default=None,
        help=(
            "StereoRenderer convergence plane (fraction of normalised depth where zero parallax "
            "sits; larger = zero plane farther away = more of the scene recedes). Overrides "
            "--comfort preset (default balanced = 0.35)."
        ),
    )
    # --codec / --crf default to None (sentinel) so D-2's --preset can tell
    # "user did not set this" from "user explicitly chose h264/crf 23".  The
    # legacy h264/23 defaults are back-filled in apply_playback_preset so the
    # resolved values the encode stages see are unchanged when no --preset is
    # given (source = passthrough).
    parser.add_argument(
        "--codec",
        choices=["h264", "h265"],
        default=None,
        help="Output video codec (default: h264; h265 when --preset pcvr/standalone)",
    )
    parser.add_argument(
        "--crf", type=int, default=None, help="Constant rate factor (default: 23; 18 pcvr / 23 standalone)"
    )
    parser.add_argument(
        "--gop",
        type=int,
        default=None,
        help="Fixed GOP length in frames (1s seek-precision default when --preset "
        "pcvr/standalone is set). 0 = let ffmpeg pick. Explicit value wins over preset.",
    )
    # D-2 (#79): downstream playback presets — PCVR / Quest standalone / source.
    # A preset is a starting point: an explicit --codec / --crf / --bitrate / --gop
    # always wins (see resolve_playback).  ``source`` (default) fills in nothing,
    # so pre-D-2 behaviour is bit-exact unless the operator opts in.
    parser.add_argument(
        "--preset",
        choices=sorted(PLAYBACK_PRESETS),
        default=DEFAULT_PLAYBACK,
        help=(
            f"Downstream playback preset (default: {DEFAULT_PLAYBACK}). "
            "pcvr = HEVC CRF 18, 1s GOP, IDR, +faststart (PC NVDEC, quality first); "
            "standalone = HEVC CRF 23, 1s GOP, IDR, +faststart (Quest 3 power/bitrate); "
            "source = passthrough (no encode tuning — keeps --codec/--crf as-is). "
            "Explicit --codec/--crf/--bitrate/--gop override the preset."
        ),
    )
    parser.add_argument("--fps", type=int, default=None, help="Output frame rate (default: inherit from source video)")
    parser.add_argument(
        "--quality",
        choices=["preview", "standard", "high"],
        default=DEFAULT_QUALITY,
        help="Quality preset for per-eye output resolution and memory path: "
        "preview = 1920²/eye, legacy fast path (lowest RAM, for iteration); "
        "standard = 2880²/eye, streaming (O(1) RAM) — default; "
        "high = 3840²/eye, streaming (O(1) RAM, sharpest on Quest-class HMDs). "
        "Explicit --output-width/--output-height override the preset resolution. "
        "Output bitrate scales with pixel area (capped by --max-bitrate).",
    )
    parser.add_argument(
        "--max-bitrate",
        type=float,
        default=200.0,
        help="Cap for the pixel-area-scaled output bitrate in Mbps (default: 200)",
    )
    parser.add_argument("--output-width", type=int, default=None, help="Equirectangular output width per eye")
    parser.add_argument("--output-height", type=int, default=None, help="Equirectangular output height per eye")
    parser.add_argument("--src-hfov", type=float, default=70.0, help="Source camera horizontal FOV (degrees)")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit number of frames (for testing)")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="V-4: batch depth/stereo stage chunk size in frames. When set, the "
        "non-streaming depth and stereo stages process frames in memory-bounded "
        "chunks (peak RAM ∝ chunk_size, not clip length) instead of buffering the "
        "whole sequence. Default None = legacy whole-sequence buffering. The "
        "streaming pipeline (O(1) memory) is unaffected.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="V-4: warmup frames replayed at each chunk boundary to rebuild "
        "temporal state. 0 (default) is exact for stateless stages and for "
        "stages whose state persists across chunks (depth estimator, stereo "
        "renderer). Use >=1 only for finite-window temporal filters.",
    )
    parser.add_argument("--no-temporal", action="store_true", help="Disable temporal smoothing")
    parser.add_argument("--temp-dir", default=None, help="Directory for intermediate files")
    # K-18 (#210): --keep-temp.  Default (False) means the auto-derived
    # intermediate directory (<input_stem>_vr180_temp/) is deleted after a
    # successful --stage all run, so video/ no longer accumulates multi-GB
    # stale residue (the same residue that fed #208's wrong numbers).  Only
    # the auto-derived dir is cleaned up: a user-supplied --temp-dir is NEVER
    # deleted (that directory belongs to the user), and a pre-existing
    # auto-derived dir is NOT deleted either (a loud warning surfaces the
    # stale-artefact risk instead).  On exception the temp dir is preserved
    # and its path is printed for debugging.  --keep-temp preserves the
    # auto-derived dir and prints its path.
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        default=False,
        help=(
            "K-18: keep the auto-derived intermediate directory after the run "
            "and print its path.  Default (False) deletes the auto-derived "
            "directory after a successful --stage all run so stale residue "
            "does not accumulate.  A user-supplied --temp-dir is never "
            "deleted regardless of this flag."
        ),
    )
    parser.add_argument("--no-ffmpeg-v360", action="store_true", help="Disable ffmpeg v360, use OpenCV fallback")
    parser.add_argument(
        "--no-equirect-batched", action="store_true", help="Disable batched equirect mapping (revert to per-frame)"
    )
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
    parser.add_argument(
        "--hw-encoder",
        choices=["auto", "on", "off"],
        default="auto",
        help="NVENC hardware encoding for the streaming pipeline: "
        "auto (default) = probe NVENC with a tiny test encode, fall back to software if unavailable; "
        "on = force NVENC without probing (user takes the risk); "
        "off = software encoding only (libx264/libx265).",
    )

    # New: input validation
    parser.add_argument(
        "--validate-input", action="store_true", help="Validate input video format and print recommendations"
    )

    # New: checkpoint/resume
    parser.add_argument("--resume", action="store_true", help="Resume from last completed checkpoint stage")

    # V-3: cross-machine staged pipeline + job manifest (issue #36)
    parser.add_argument(
        "--stages",
        default=None,
        help="Comma-separated stage subset to run: upscale,depth,stereo,project,encode "
        "(default: all — behaviour identical to running without this flag)",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Write/update a job manifest JSON at this path after the run (cross-machine relay)",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        metavar="MANIFEST",
        help="Resume from a job manifest: completed stages are hash-validated, then skipped",
    )
    parser.add_argument(
        "--machine",
        default=None,
        help="Machine label recorded into the manifest (e.g. win-cuda, mac-mps)",
    )

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
        help="Output projection: vr180 (stereo spherical) or fulldome (mono fisheye, default vr180). "
        "⚠️ fulldome 球幕/桌面用，不适用于 VR 头显——头显请用 vr180",
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

    # Stereo model selection
    parser.add_argument(
        "--stereo-model",
        choices=["default", "stereocrafter"],
        default="default",
        help="Stereo rendering backend: default (depth-shift + inpaint, default) or "
        "stereocrafter (Tencent StereoCrafter, CUDA-only, cleaner disocclusion)",
    )
    parser.add_argument(
        "--stereocrafter-repo-dir",
        default=None,
        help="StereoCrafter repository directory (or env STEREOCRAFTER_REPO_DIR)",
    )
    parser.add_argument(
        "--stereocrafter-python",
        default=None,
        help="Python executable for StereoCrafter inference (or env STEREOCRAFTER_PYTHON)",
    )
    parser.add_argument(
        "--stereocrafter-checkpoint-dir",
        default=None,
        help="StereoCrafter checkpoint directory (or env STEREOCRAFTER_CKPT_DIR)",
    )
    parser.add_argument(
        "--stereocrafter-max-res",
        type=int,
        default=None,
        help="Max resolution (short side) for StereoCrafter inference (or env STEREOCRAFTER_MAX_RES)",
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

    # H-1: audio passthrough (issue #73)
    parser.add_argument(
        "--copy-audio-from",
        default=None,
        metavar="PATH",
        help="H-1: attach an audio track from this source file to the final VR180 "
        "output (lossless -c copy remux). The sv3d/st3d VR metadata is preserved. "
        "Implied audio source is the input video when omitted; pass an explicit path "
        "to attach a different track.",
    )

    # R-6: 180° Outpaint fill
    parser.add_argument(
        "--outpaint",
        choices=["none", "gradient", "ai"],
        default="none",
        help="Outpaint black boundary regions in equirect frames: "
        "none (skip, default), gradient (OpenCV-based), or ai (SDXL inpaint, requires backend deployment)",
    )
    parser.add_argument(
        "--outpaint-mask-threshold",
        type=int,
        default=10,
        help="Pixel brightness threshold for black boundary detection (default: 10)",
    )
    parser.add_argument(
        "--outpaint-mask-top-ratio",
        type=float,
        default=0.25,
        help="Fraction of height scanned from top for black boundaries (default: 0.25)",
    )
    parser.add_argument(
        "--outpaint-mask-bottom-ratio",
        type=float,
        default=0.25,
        help="Fraction of height scanned from bottom for black boundaries (default: 0.25)",
    )

    return parser.parse_args(argv)


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


def _intake_frames(video_path: str, max_frames: int | None, lazy: bool):
    """Load source frames for the batch pipeline.

    Returns ``(frames, total)`` where ``total`` is the (capped) frame count.

    When ``lazy=True`` (V-4.1a, only when ``--chunk-size`` is active), ``frames``
    is a generator — ``read_frames`` is returned directly so the full clip is
    never materialised in RAM.  Downstream chunked stages (depth, via
    ``process_in_chunks``) stream through it with a bounded circular buffer.
    ``total`` still comes from the cheap ``CAP_PROP_FRAME_COUNT`` metadata so
    progress bars can show a known count without materialising any frames.

    When ``lazy=False`` the legacy behaviour is preserved exactly: frames are
    materialised into a list and a large-buffer warning is emitted if RAM use
    exceeds 1 GB.  This keeps the non-``--chunk-size`` path bit-for-bit
    unchanged.
    """
    frames_gen = read_frames(video_path, max_frames)

    # Peek the total off the generator's metadata is not directly exposed, so
    # pull the first frame to materialise the cv2 header logging in read_frames
    # AND obtain a reliable count.  read_frames logs fps/total/size from
    # CAP_PROP_* metadata (no full materialisation).  To get ``total`` without
    # consuming frames we probe the same metadata cheaply:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames:
        total = min(total, max_frames)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if lazy:
        return frames_gen, total

    frames = list(frames_gen)
    log.info("Loaded %d frames", len(frames))
    if frames and h and w:
        mem_mb = len(frames) * h * w * 3 / (1024 * 1024)
        if mem_mb > 1024:
            log.warning(
                "⚠️  Frame buffer uses ~%.0f MB in RAM. For large videos, consider using --max-frames or --temp-dir.",
                mem_mb,
            )
    return frames, total


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


# ---------------------------------------------------------------------------
# K-18 (#210): auto-derived temp-dir lifecycle
# ---------------------------------------------------------------------------


def _resolve_temp_dir_lifecycle(args) -> dict:
    """Resolve the temp dir for a ``--stage all`` run and return its lifecycle metadata.

    Returns a dict with:

    * ``temp_dir`` (``Path``): the resolved auto-derived temp dir path (the
      one that ``get_temp_dir`` would produce with ``subdir=None``).
    * ``auto`` (``bool``): True when the dir is auto-derived
      (``<input_stem>_vr180_temp``), False when ``--temp-dir`` was given.
    * ``newly_created`` (``bool``): True when the auto-derived dir did NOT
      exist before this run and we created it here — i.e. *this run* owns
      it and is allowed to clean it up on success.
    * ``pre_existed`` (``bool``): True when the auto-derived dir already
      existed before this run — another run / operator owns it and we must
      NOT delete it (but we must warn: this is the #208 stale-artefact trap).

    The actual directory creation happens here (the lifecycle owner), so
    ``get_temp_dir`` (which still exists for subdirectory children like
    ``depth/`` / ``left/`` / ``right/``) does not double-create or race
    with the pre-existence check.  ``main()`` calls this once at the top
    of the ``--stage all`` block and then wraps the stage loop in
    try/finally using the returned dict.

    ``--temp-dir`` (explicit) → ``auto=False``: user owns the directory,
    never delete.  Auto-derived + not-existed → ``auto=True``,
    ``newly_created=True``: this run owns it, delete on success.
    Auto-derived + pre-existed → ``auto=True``, ``pre_existed=True``:
    warn and do NOT delete.
    """
    keep = bool(getattr(args, "keep_temp", False))
    if getattr(args, "temp_dir", None):
        # User-supplied --temp-dir: this run does not own the directory,
        # so we never delete it (that would be a disaster — the user may
        # have other state in there).
        base = Path(args.temp_dir)
        base.mkdir(parents=True, exist_ok=True)
        return {
            "temp_dir": base,
            "auto": False,
            "newly_created": False,
            "pre_existed": False,
            "keep": keep,
        }

    input_stem = Path(args.input).stem
    base = Path(args.input).parent / f"{input_stem}_vr180_temp"
    pre_existed = base.exists()

    if pre_existed:
        # K-18 / #208 trap: the dir already exists from a prior run or
        # another process.  Do NOT delete it (we don't own it), and
        # loudly warn the operator that it may contain stale artefacts
        # (exactly the residue that fed #208's wrong comparison numbers).
        log.warning(
            "⚠️  发现已存在的中间目录 %s（可能含旧产物）— 本次不会清理它。 "
            "如需全新运行请手动删除后再启动，或显式指定 --temp-dir 隔离产物。",
            base,
        )
        return {
            "temp_dir": base,
            "auto": True,
            "newly_created": False,
            "pre_existed": True,
            "keep": keep,
        }

    # Fresh auto-derived dir: create it and record that this run owns it.
    base.mkdir(parents=True, exist_ok=False)
    if keep:
        log.info("💾 --keep-temp: 中间目录将保留 → %s", base)
    else:
        log.info("💾 中间目录 → %s（运行结束后将自动清理）", base)
    return {
        "temp_dir": base,
        "auto": True,
        "newly_created": True,
        "pre_existed": False,
        "keep": keep,
    }


def _cleanup_temp_dir_if_owned(lifecycle: dict) -> None:
    """Clean up the auto-derived temp dir if this run owns it and it succeeded.

    Only deletes when all three hold:

    * ``lifecycle["auto"]`` is True (auto-derived, not user-supplied).
    * ``lifecycle["newly_created"]`` is True (this run created it).
    * ``lifecycle["keep"]`` is False (``--keep-temp`` was not given).

    Uses ``shutil.rmtree`` for recursive removal; silent on
    ``FileNotFoundError`` (race: already gone) — never raises.
    """
    if not lifecycle.get("auto"):
        return
    if not lifecycle.get("newly_created"):
        return
    if lifecycle.get("keep"):
        return
    base = lifecycle["temp_dir"]
    try:
        shutil.rmtree(base)
        log.info("🧹 已清理中间目录 %s", base)
    except FileNotFoundError:
        # Already gone (race or manual delete) — nothing to do.
        return


def _print_temp_dir_path(lifecycle: dict) -> None:
    """Print the temp dir path on --keep-temp or on exception."""
    if not lifecycle.get("auto"):
        return
    print(f"💾 中间目录: {lifecycle['temp_dir']}")


def build_depth_backend(args, *, fallback: bool = True):
    """I-5 (#120): build the requested ``--depth-model`` backend for injection.

    Returns ``(estimator, backend_name)``.  When ``--depth-model`` is the
    default ``depth-anything`` this returns ``(None, "depth-anything")`` — the
    caller falls through to the built-in Depth-Anything estimator (the
    pre-I-5 default, so behaviour is unchanged without the flag).

    When ``--depth-model depthcrafter`` is requested, a
    :class:`DepthCrafterEstimator` is constructed and returned for injection
    into either the batch depth stage or the streaming pipeline.  This factory
    is the **single** place the DepthCrafter constructor argument assembly
    lives — both :func:`run_depth_stage` (batch path) and the streaming branch
    in :func:`main` call it, so the construction logic is never duplicated.

    ``fallback`` selects the I-5 unavailable-backend policy:

    * ``fallback=True`` (streaming path): if the backend cannot be constructed
      (CUDA missing, repo not deployed), a loud WARNING is logged and the
      pipeline **falls back to Depth-Anything** — a requested-but-unavailable
      advanced backend never runs silently with the wrong model, it always
      announces the fallback.  The stream still produces output.
    * ``fallback=False`` (batch depth stage): the pre-I-5 hard-fail contract
      is preserved — the constructor's RuntimeError propagates.  The batch
      depth stage is a dedicated, explicitly-invoked step, so a silent degrade
      would waste an operator's run; failing loudly is the accepted behaviour.
    """
    if getattr(args, "depth_model", "depth-anything") != "depthcrafter":
        return None, "depth-anything"
    try:
        estimator = DepthCrafterEstimator(
            repo_dir=args.depthcrafter_repo_dir,
            python_exe=args.depthcrafter_python,
            checkpoint_dir=args.depthcrafter_checkpoint_dir,
            max_resolution=args.depthcrafter_max_res,
        )
    except (RuntimeError, OSError) as exc:
        if not fallback:
            raise
        log.warning(
            "⚠️  --depth-model depthcrafter requested but the DepthCrafter backend "
            "is unavailable — FALLING BACK to depth-anything (per-frame). "
            "The '治晕' temporal-consistency benefit will NOT apply this run.\n"
            "  Reason: %s\n"
            "  Deploy with: python scripts/setup_depthcrafter.py  (needs CUDA)",
            exc,
        )
        return None, "depth-anything"
    log.info("Using DepthCrafter for temporally-consistent video depth estimation")
    return estimator, "depthcrafter"


def build_stereo_backend(args, *, fallback: bool = True):
    """I-5 (#120): build the requested ``--stereo-model`` backend for injection.

    Returns ``(renderer, backend_name)``.  When ``--stereo-model`` is the
    default ``default`` this returns ``(None, "default")`` — the caller falls
    through to the built-in per-frame :class:`StereoRenderer` (the pre-I-5
    default, so behaviour is unchanged without the flag).

    When ``--stereo-model stereocrafter`` is requested, a
    :class:`StereoCrafterRenderer` is constructed and returned for injection.
    This factory is the **single** place the StereoCrafter constructor argument
    assembly lives — both :func:`run_stereo_stage` (batch path) and the
    streaming branch in :func:`main` call it, so the construction logic is
    never duplicated.

    ``fallback`` selects the I-5 unavailable-backend policy (see
    :func:`build_depth_backend`): ``True`` (streaming) logs a WARNING and
    falls back to the default StereoRenderer; ``False`` (batch stereo stage)
    preserves the pre-I-5 hard-fail contract.
    """
    if getattr(args, "stereo_model", "default") != "stereocrafter":
        return None, "default"
    try:
        renderer = StereoCrafterRenderer(
            repo_dir=args.stereocrafter_repo_dir,
            python_exe=args.stereocrafter_python,
            checkpoint_dir=args.stereocrafter_checkpoint_dir,
            max_resolution=args.stereocrafter_max_res,
        )
    except (RuntimeError, OSError) as exc:
        if not fallback:
            raise
        log.warning(
            "⚠️  --stereo-model stereocrafter requested but the StereoCrafter backend "
            "is unavailable — FALLING BACK to the default depth-shift StereoRenderer. "
            "The clean-disocclusion '治晕' benefit will NOT apply this run.\n"
            "  Reason: %s\n"
            "  Deploy with: python scripts/setup_stereocrafter.py  (needs CUDA)",
            exc,
        )
        return None, "default"
    log.info("Using StereoCrafter for depth-aware stereo with disocclusion inpainting")
    return renderer, "stereocrafter"


# ---------------------------------------------------------------------------
# I-6 (#121): model-scoped depth checkpoint dir + meta.json
# ---------------------------------------------------------------------------

#: Depth checkpoint meta file name written into ``get_depth_dir(args)``.
DEPTH_META_NAME = "meta.json"


def get_depth_dir(args) -> str:
    """Depth checkpoint directory, scoped by depth model (I-6, #121).

    Returns ``<temp>/depth/<depth_model>/`` so ``--depth-model depth-anything``
    and ``--depth-model depthcrafter`` write to **non-overlapping** directories
    and cannot silently clobber / cross-reuse each other's ``depth_XXXXXX.npy``.

    Before #121 every model wrote into the shared ``<temp>/depth/`` directory,
    so an A/B run or a resume could silently mix two models' depth products.
    """
    depth_model = getattr(args, "depth_model", "depth-anything")
    return get_temp_dir(args, os.path.join("depth", depth_model))


def _depth_meta_params(args, num_frames: int) -> dict[str, Any]:
    """The parameter snapshot persisted as ``meta.json`` in the depth dir.

    On resume this is compared to the *current* invocation's params: a mismatch
    (different model / max_res / model_size / frame count) means the cached
    depth maps are not the ones the operator is asking for — recompute, never
    silently reuse (issue #121 resume-safety requirement).
    """

    def _safe(val: Any) -> Any:
        # In tests ``args`` may be a MagicMock whose un-set attributes are
        # auto-Mocks (not JSON-serialisable).  Only keep plain scalar values.
        return val if isinstance(val, (str, int, float, bool)) or val is None else None

    params: dict[str, Any] = {
        "depth_model": _safe(getattr(args, "depth_model", "depth-anything")) or "depth-anything",
        "num_frames": int(num_frames),
        "model_size": _safe(getattr(args, "model_size", None)),
        "max_res": _safe(getattr(args, "depthcrafter_max_res", None)),
        "temporal_smoothing": _safe(getattr(args, "temporal_smoothing", 0.0)) or 0.0,
    }
    return params


def save_depth_meta(args, num_frames: int, depth_dir: str | None = None) -> None:
    """Write ``meta.json`` (model + key params + timestamp) into the depth dir.

    Idempotent — every depth run overwrites it so the meta always reflects the
    maps that are actually on disk.
    """
    depth_dir = depth_dir or get_depth_dir(args)
    meta = _depth_meta_params(args, num_frames)
    meta["timestamp"] = datetime.datetime.now().isoformat(timespec="seconds")
    meta_path = os.path.join(depth_dir, DEPTH_META_NAME)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    log.info("💾 Depth meta saved: %s (model=%s, frames=%d)", meta_path, meta["depth_model"], num_frames)


def load_depth_meta(depth_dir: str | Path) -> dict | None:
    """Read ``meta.json`` from a depth dir; ``None`` if absent."""
    meta_path = os.path.join(str(depth_dir), DEPTH_META_NAME)
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)


def depth_meta_matches(args, meta: dict | None) -> bool:
    """True if *meta*'s model + params match the current invocation (I-6).

    Compares the identity-relevant fields (model, depthcrafter max_res,
    model_size, temporal smoothing).  ``meta=None`` (no meta file) is treated
    as a mismatch so a pre-#121 cache is never silently reused.  ``num_frames``
    is deliberately NOT compared here — it is checked against the actual npy
    count on disk by :func:`load_depth_checkpoint` (a tampered meta that lies
    about the frame count would otherwise pass by construction).
    """
    if not meta:
        return False
    want = _depth_meta_params(args, meta.get("num_frames", 0))
    keys = ("depth_model", "model_size", "max_res", "temporal_smoothing")
    return all(meta.get(k) == want.get(k) for k in keys)


def load_depth_checkpoint(args, depth_dir: str | None = None) -> list[np.ndarray] | None:
    """Load cached ``depth_*.npy`` maps with I-6 (#121) meta validation.

    resume safety: when the depth dir's ``meta.json`` exists and its model /
    params **do not match** the current invocation (operator switched
    ``--depth-model``, changed ``--depthcrafter-max-res`` / ``--model-size`` /
    temporal smoothing), the cache is stale — return ``None`` so the caller
    **recomputes** fresh; we never silently reuse a different model's maps.

    A dir **without** ``meta.json`` (a pre-#121 or hand-seeded cache, e.g. the
    V-3 manifest flow's stubbed depth stage) cannot be validated — it is loaded
    with a loud "source 未知" warning rather than silently trusted, preserving
    pre-#121 resume behaviour while making the unknown provenance visible.
    """
    import glob

    depth_dir = depth_dir or get_depth_dir(args)
    meta = load_depth_meta(depth_dir)
    depth_files = sorted(glob.glob(os.path.join(depth_dir, "depth_*.npy")))
    # No cached maps at all is NOT a stale-cache signal — it just means the
    # depth stage produced nothing (e.g. a V-3 manifest flow whose stubbed
    # depth stage wrote no npy).  Return an empty list so the caller does not
    # mistake it for the meta-mismatch "recompute" case below.
    if not depth_files:
        return []

    if meta is None:
        log.warning(
            "⚠️  Depth checkpoint at %s has no meta.json (source 未知) — loading "
            "without model/params validation; re-run --stage depth to attach provenance.",
            depth_dir,
        )
        depths = [np.load(f) for f in depth_files]
        log.info("📂 Loaded %d depth maps from checkpoint (source 未知)", len(depths))
        return depths

    if not depth_meta_matches(args, meta):
        log.warning(
            "⚠️  Depth checkpoint at %s is stale or from a different model/params "
            "(meta: %s) — will recompute depth, not silently reuse.",
            depth_dir,
            meta,
        )
        return None
    # A meta that lies about the frame count (or a truncated cache) is stale:
    # require the npy count to agree with the meta's num_frames.
    if meta.get("num_frames") is not None and int(meta["num_frames"]) != len(depth_files):
        log.warning(
            "⚠️  Depth checkpoint at %s has %d npy files but meta claims %d frames — "
            "will recompute depth, not silently reuse.",
            depth_dir,
            len(depth_files),
            meta["num_frames"],
        )
        return None
    depths = [np.load(f) for f in depth_files]
    log.info("📂 Loaded %d depth maps from checkpoint (model=%s)", len(depths), meta.get("depth_model"))
    return depths


def run_depth_stage(args, frames):
    """Stage 1: Estimate depth for all frames."""
    log.info("=== Stage 1: Depth Estimation ===")

    # DepthCrafter mode — process entire video at once (temporally consistent)
    if args.depth_model == "depthcrafter":
        # I-5 (#120): construction is shared with the streaming path via
        # build_depth_backend (no duplication).  The batch depth stage keeps
        # the pre-I-5 hard-fail contract (fallback=False): it is a dedicated,
        # explicitly-invoked step, so a silent degrade would waste a run.
        # I-6 (#121): the output dir is model-scoped (get_depth_dir) so two
        # depth models never share a checkpoint directory.
        out_dir = get_depth_dir(args)
        estimator, _ = build_depth_backend(args, fallback=False)
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

        save_depth_meta(args, len(depths), out_dir)
        log.info(f"Depth maps (DepthCrafter) saved to {out_dir}/")
        return depths

    # Default: Depth-Anything V2 per-frame
    estimator = DepthEstimator(
        model_size=args.model_size,
        device=args.device,
        calibrate=True,
    )

    out_dir = get_depth_dir(args)
    depths = []
    prev_depth = None
    temporal_alpha = args.temporal_smoothing if args.temporal_smoothing > 0 else None

    # V-4 (#37): optional chunked processing.  When --chunk-size is set, frames
    # are processed in memory-bounded chunks via process_in_chunks.  Depth-Anything
    # is stateless so chunking is bit-exact with overlap=0; the pipeline-level
    # EMA (prev_depth) is carried across chunks by processing the chunk stream
    # sequentially below, so its state is continuous and the output matches the
    # whole-sequence run exactly.
    chunk_size = getattr(args, "chunk_size", None)

    def _emit_depth(i, frame):
        nonlocal prev_depth
        depth = estimator.estimate(frame)
        if temporal_alpha and prev_depth is not None:
            depth = temporal_alpha * depth + (1 - temporal_alpha) * prev_depth
        prev_depth = depth
        depths.append(depth)
        dmax = float(np.nanmax(depth))
        depth_vis = (depth / dmax * 255).astype(np.uint8) if dmax > 0 else depth.astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, f"depth_{i:06d}.png"), cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO))
        np.save(os.path.join(out_dir, f"depth_{i:06d}.npy"), depth)

    if chunk_size:
        from pipeline.chunked_processor import process_in_chunks

        overlap = getattr(args, "overlap", 0)
        # Drive the per-frame emit over chunks.  Depth-Anything is stateless,
        # so the warmup prefix is a no-op (nothing to rebuild); the EMA
        # ``prev_depth`` is carried across chunks by the sequential emit below,
        # making the output bit-exact vs the whole-sequence run.  Peak RAM is
        # ∝ chunk_size rather than clip length.
        gi = 0

        def _depth_process_fn(chunk_frames, warm_offset, emit_offset):
            nonlocal gi
            # Warmup: no temporal state to rebuild for a stateless estimator.
            outs = []
            for f in chunk_frames[warm_offset:]:
                _emit_depth(gi, f)
                outs.append(depths[-1])
                gi += 1
            return iter(outs)

        list(
            tqdm(
                process_in_chunks(frames, _depth_process_fn, chunk_size=chunk_size, overlap=overlap),
                desc="Estimating depth (chunked)",
                total=len(frames) if hasattr(frames, "__len__") else None,
            )
        )
    else:
        for i, frame in enumerate(tqdm(frames, desc="Estimating depth")):
            _emit_depth(i, frame)

    save_depth_meta(args, len(depths), out_dir)
    log.info(f"Depth maps saved to {out_dir}/")
    return depths


def run_stereo_stage(args, frames, depths):
    """Stage 2: Generate stereo left/right views."""
    log.info("=== Stage 2: Stereo Disparity Rendering ===")

    # StereoCrafter mode — process whole video via external inference
    if args.stereo_model == "stereocrafter":
        log.info("Using StereoCrafter for depth-aware stereo with disocclusion inpainting")
        return _run_stereocrafter_stage(args, frames, depths)

    # Default: per-frame depth-shift renderer
    renderer = StereoRenderer(
        ipd=args.ipd,
        max_disparity=args.max_disparity,
        convergence=getattr(args, "convergence", 0.3),
        temporal_smooth=getattr(args, "temporal_smooth", not args.no_temporal),
    )

    left_dir = get_temp_dir(args, "left")
    right_dir = get_temp_dir(args, "right")

    pairs: list[tuple] = list(zip(frames, depths, strict=False))
    chunk_size = getattr(args, "chunk_size", None)
    overlap = getattr(args, "overlap", 0)

    left_frames, right_frames = [], []
    if chunk_size:
        # V-4 (#37): memory-bounded chunked render.  One StereoRenderer
        # instance drives every chunk so ``_prev_disparity`` is continuous →
        # bit-exact vs whole-sequence with overlap=0.  Peak RAM ∝ chunk_size.
        chunked = renderer.render_sequence_chunked(frames, depths, chunk_size=chunk_size, overlap=overlap)
        iterator = tqdm(chunked, desc="Rendering stereo (chunked)", total=len(pairs))
        for i, (left, right) in enumerate(iterator):
            left_frames.append(left)
            right_frames.append(right)
            cv2.imwrite(os.path.join(left_dir, f"left_{i:06d}.png"), cv2.cvtColor(left, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(right_dir, f"right_{i:06d}.png"), cv2.cvtColor(right, cv2.COLOR_RGB2BGR))
    else:
        for i, (frame, depth) in enumerate(tqdm(pairs, desc="Rendering stereo", total=len(pairs))):
            left, right = renderer.render(frame, depth)
            left_frames.append(left)
            right_frames.append(right)

            # Save intermediate files
            cv2.imwrite(os.path.join(left_dir, f"left_{i:06d}.png"), cv2.cvtColor(left, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(right_dir, f"right_{i:06d}.png"), cv2.cvtColor(right, cv2.COLOR_RGB2BGR))

    log.info(f"Stereo views: {len(left_frames)} frames each")
    return left_frames, right_frames


def _run_stereocrafter_stage(args, frames, depths):
    """Run StereoCrafter inference: frames + depth -> L/R video files -> load frames."""

    # Save frames as a temp video so StereoCrafter can read them
    temp_dir = get_temp_dir(args)
    depth_dir = get_depth_dir(args)

    # Write frames to a temp video file for StereoCrafter input
    H, W = frames[0].shape[:2]
    temp_video = os.path.join(temp_dir, "_stereocrafter_input.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(temp_video, fourcc, args.fps or 30, (W, H))
    for frame in tqdm(frames, desc="Preparing frames for StereoCrafter"):
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    log.info("StereoCrafter: wrote %d frames to %s", len(frames), temp_video)

    # Save depth maps to depth_dir if not already there
    # (they should already exist from run_depth_stage, but ensure npy files are present)
    for i, depth in enumerate(depths):
        npy_path = os.path.join(depth_dir, f"depth_{i:06d}.npy")
        if not os.path.exists(npy_path):
            np.save(npy_path, depth)

    # Output paths
    left_video = os.path.join(temp_dir, "_stereocrafter_left.mp4")
    right_video = os.path.join(temp_dir, "_stereocrafter_right.mp4")

    # Run StereoCrafter.  I-5 (#120): construction is shared with the streaming
    # path via build_stereo_backend (no duplication); fallback=False keeps the
    # batch stereo stage's pre-I-5 hard-fail contract.
    renderer, _ = build_stereo_backend(args, fallback=False)
    result_left, result_right = renderer.render_video(
        input_path=temp_video,
        depth_dir=depth_dir,
        output_left=left_video,
        output_right=right_video,
    )

    # Load output videos back as frame arrays
    left_frames = _load_video_frames(result_left)
    right_frames = _load_video_frames(result_right)

    # Save individual frames for checkpoint restore
    left_dir = get_temp_dir(args, "left")
    right_dir = get_temp_dir(args, "right")
    for i, (left, right) in enumerate(zip(left_frames, right_frames, strict=False)):
        cv2.imwrite(os.path.join(left_dir, f"left_{i:06d}.png"), cv2.cvtColor(left, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(right_dir, f"right_{i:06d}.png"), cv2.cvtColor(right, cv2.COLOR_RGB2BGR))

    log.info("StereoCrafter: %d L/R frame pairs loaded", len(left_frames))
    return left_frames, right_frames


def _load_video_frames(video_path: str) -> list[np.ndarray]:
    """Load all frames from a video file as RGB ndarrays."""
    cap = cv2.VideoCapture(video_path)
    frames: list[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def run_equirect_stage(args, left_frames, right_frames):
    """Stage 3: Map stereo views to equirectangular.

    Uses batched ``map_sequence()`` by default (~10× faster than per-frame).
    Falls back to per-frame ``map_stereo_pair()`` when ``--no-equirect-batched``
    is set (e.g., for testing or OpenCV fallback).
    """
    log.info("=== Stage 3: Equirectangular Projection ===")

    mapper = EquirectangularMapper(
        output_width=args.output_width,
        output_height=args.output_height,
        src_hfov=args.src_hfov,
        use_ffmpeg=not args.no_ffmpeg_v360,
    )

    out_dir = get_temp_dir(args, "equirect")
    temp_dir = get_temp_dir(args)

    if args.no_equirect_batched:
        # Per-frame path (fallback)
        sbs_frames = []
        for i, (left, right) in enumerate(
            tqdm(
                zip(left_frames, right_frames, strict=False),
                desc="Mapping to equirect (per-frame)",
                total=len(left_frames),
            )
        ):
            sbs = mapper.map_stereo_pair(left, right)
            sbs_frames.append(sbs)
            cv2.imwrite(os.path.join(out_dir, f"equirect_{i:06d}.png"), cv2.cvtColor(sbs, cv2.COLOR_RGB2BGR))
    else:
        # Batched path — single ffmpeg v360 call per eye on the whole sequence
        sbs_frames = mapper.map_sequence(left_frames, right_frames, temp_dir)
        # Write frames to disk for checkpoint restore
        for i, sbs in enumerate(sbs_frames):
            cv2.imwrite(os.path.join(out_dir, f"equirect_{i:06d}.png"), cv2.cvtColor(sbs, cv2.COLOR_RGB2BGR))

    log.info(
        f"Generated {len(sbs_frames)} equirectangular SBS frames ({sbs_frames[0].shape[1]}×{sbs_frames[0].shape[0]})"
    )
    return sbs_frames


def run_outpaint_stage(args, sbs_frames):
    """Stage 3.5: Outpaint black boundary regions in equirect frames."""
    log.info("=== Stage 3.5: 180° Outpaint Fill (%s) ===", args.outpaint)

    if args.outpaint == "none":
        log.info("Outpainting disabled (--outpaint none) — skipping")
        return sbs_frames

    outpainter = Outpainter(
        mode=args.outpaint,
        mask_threshold=args.outpaint_mask_threshold,
        mask_top_ratio=args.outpaint_mask_top_ratio,
        mask_bottom_ratio=args.outpaint_mask_bottom_ratio,
    )

    result = outpainter.process(sbs_frames)

    # Overwrite equirect checkpoint files with outpainted versions
    out_dir = get_temp_dir(args, "equirect")
    for i, frame in enumerate(result):
        cv2.imwrite(os.path.join(out_dir, f"equirect_{i:06d}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    log.info("Outpainted %d frames (mode=%s)", len(result), args.outpaint)
    return result


def run_metadata_stage(args, sbs_frames):
    """Stage 4: Encode video with VR metadata."""
    log.info("=== Stage 4: VR Metadata Embedding ===")

    embedder = VRMetadataEmbedder(
        codec=args.codec,
        crf=args.crf,
        fps=args.fps,
        bitrate=args.bitrate,
        gop=getattr(args, "gop", None),
        force_idr=getattr(args, "_preset_force_idr", False),
        faststart=getattr(args, "_preset_faststart", None),
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


def run_chunked_fused_stage(args, frames, depths):
    """V-4.1b (#89): fused stereo→equirect→encode under ``--chunk-size``.

    Instead of ``run_stereo_stage`` returning full ``left_frames`` /
    ``right_frames`` lists, ``run_equirect_stage`` accumulating all SBS frames,
    and ``run_metadata_stage``'s ``b"".join``-ing the whole raw byte stream, this
    fuses all three into a single persistent ffmpeg raw-pipe process and writes
    each chunk's per-frame SBS bytes incrementally.  Peak RAM is proportional to
    ``chunk_size`` (one chunk's L/R + SBS frames), **not** to clip length — the
    V-4.1b memory contract.

    Equivalence: the StereoRenderer is a single persistent instance across chunks
    (so ``_prev_disparity`` is continuous → bit-exact vs whole-sequence with
    overlap=0); equirect uses the per-frame ``map_stereo_pair`` path (the batched
    ``map_sequence`` requires the full sequence, so the two are mutually
    exclusive — chunk mode forces per-frame equirect).

    Args:
        args: Parsed CLI args (``--chunk-size``, ``--overlap``, ``--codec``,
            ``--crf``, ``--bitrate``, ``--hw-encoder``, ``--output-width`` …).
        frames: Materialised list of source RGB frames (same length as depths).
        depths: Matching depth maps (same length).

    Returns:
        Output video path (with sv3d/st3d injected, matching the legacy
        metadata stage's contract).
    """
    from pipeline.streaming_pipeline import RawFrameFFmpegWriter, _resolve_hw_encoder

    log.info("=== Stage 2+3+4 (fused, chunked incremental encode) ===")

    renderer = StereoRenderer(
        ipd=args.ipd,
        max_disparity=args.max_disparity,
        convergence=getattr(args, "convergence", 0.3),
        temporal_smooth=getattr(args, "temporal_smooth", not args.no_temporal),
    )
    mapper = EquirectangularMapper(
        output_width=args.output_width,
        output_height=args.output_height,
        src_hfov=args.src_hfov,
        use_ffmpeg=not args.no_ffmpeg_v360,
    )

    output_path = get_output_path(args)
    out_w = args.output_width * 2
    out_h = args.output_height
    log.info("Target output: %dx%d (SBS, %d² per eye, chunked incremental)", out_w, out_h, args.output_width)

    # Resolve NVENC once (issue #49): CUDA availability does not imply a working
    # NVENC encoder.  Reuse the streaming path's probe + select_encoder.
    device = resolve_device(args.device)
    hw = _resolve_hw_encoder(device, args.codec, out_w, getattr(args, "hw_encoder", "auto"))

    # Per-chunk L/R checkpoint dirs (kept for checkpoint restore compatibility
    # with the separate stereo/equirect stages — the manifest validates them).
    left_dir = get_temp_dir(args, "left")
    right_dir = get_temp_dir(args, "right")
    eq_dir = get_temp_dir(args, "equirect")

    chunk_size = getattr(args, "chunk_size", None) or len(frames)
    overlap = getattr(args, "overlap", 0)

    # render_sequence_chunked drives one persistent StereoRenderer across chunks
    # → _prev_disparity continuous → bit-exact vs whole-sequence (overlap=0).
    chunked = renderer.render_sequence_chunked(frames, depths, chunk_size=chunk_size, overlap=overlap)
    iterator = tqdm(chunked, desc="Stereo→project→encode (chunked)", total=len(frames))

    with RawFrameFFmpegWriter(
        output_path,
        out_w,
        out_h,
        codec=args.codec,
        crf=args.crf,
        fps=args.fps or 30,
        bitrate=args.bitrate,
        hw_encoder=hw,
        gop=getattr(args, "gop", None),
        force_idr=getattr(args, "_preset_force_idr", False),
        faststart=getattr(args, "_preset_faststart", None),
    ) as writer:
        for i, (left, right) in enumerate(iterator):
            # Per-frame equirect (chunk mode forces this — batched map_sequence
            # needs the full sequence and is mutually exclusive with chunking).
            sbs = mapper.map_stereo_pair(left, right)
            writer.write(sbs)

            # Checkpoint PNGs so resume/manifest hash validation still works.
            cv2.imwrite(
                os.path.join(left_dir, f"left_{i:06d}.png"),
                cv2.cvtColor(left, cv2.COLOR_RGB2BGR),
            )
            cv2.imwrite(
                os.path.join(right_dir, f"right_{i:06d}.png"),
                cv2.cvtColor(right, cv2.COLOR_RGB2BGR),
            )
            cv2.imwrite(
                os.path.join(eq_dir, f"equirect_{i:06d}.png"),
                cv2.cvtColor(sbs, cv2.COLOR_RGB2BGR),
            )
            # Release the chunk's intermediates so only the current frame's L/R/SBS
            # are resident — peak RAM ∝ chunk_size, not clip length.
            del left, right, sbs

    log.info("✅ Chunked incremental encode complete → %s", output_path)

    # Inject sv3d/st3d so HMDs recognise the stream as 180° 3D (matches the
    # streaming path + the legacy metadata stage's contract).
    from pipeline.spherical_injector import inject_spherical_metadata

    injected = output_path + ".vr.mp4"
    try:
        inject_spherical_metadata(
            output_path,
            injected,
            width=out_w,
            height=out_h,
            stereo_mode="sbs",
        )
    except Exception as e:
        log.error("❌ VR metadata injection failed for %s: %s", output_path, e)
        raise
    os.replace(injected, output_path)
    log.info("✅ sv3d/st3d injected → %s", output_path)
    return output_path


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


STAGE_ORDER = ["upscale", "depth", "stereo", "equirect", "outpaint", "metadata"]
STAGE_ORDER_SBS = ["upscale", "equirect", "outpaint", "metadata"]  # Skip depth & stereo for SBS input


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


def apply_quality_preset(args):
    """Resolve --quality into concrete resolution / streaming / bitrate defaults.

    Explicit --output-width/--output-height/--bitrate/--streaming always win;
    the preset only fills in values the user did not specify.
    """
    eye_size, streaming = resolve_quality(args.quality, args.output_width)
    args.output_width = eye_size
    if args.output_height is None:
        args.output_height = eye_size
    if streaming and not args.streaming and args.projection == "vr180":
        args.streaming = True
        log.info("🎚️  --quality %s → %d²/eye, streaming mode (O(1) memory)", args.quality, eye_size)
    else:
        log.info("🎚️  --quality %s → %d²/eye", args.quality, eye_size)
    if args.bitrate is None:
        mbps = scaled_bitrate_mbps(eye_size, max_mbps=args.max_bitrate)
        args.bitrate = f"{mbps:g}M"
        log.info("📶 Adaptive bitrate: %.1f Mbps (scaled from 1920² baseline)", mbps)


def _apply_comfort_preset(args) -> None:
    """Resolve ``--comfort`` into concrete stereo comfort values (I-3, #88).

    Mirrors :func:`apply_quality_preset`: the preset fills in values the user
    did not explicitly pass, and explicit ``--max-disparity`` /
    ``--convergence`` / ``--no-temporal`` always win.  After this call
    ``args.max_disparity`` and ``args.convergence`` are guaranteed to be
    concrete floats and ``args.temporal_smooth`` is a concrete bool, so both
    the batch and streaming stereo paths see fully-resolved values.
    """
    explicit: dict[str, object] = {}
    if args.max_disparity is not None:
        explicit["max_disparity"] = args.max_disparity
    if args.convergence is not None:
        explicit["convergence"] = args.convergence
    if args.no_temporal:
        explicit["temporal_smooth"] = False

    resolved = resolve_comfort(args.comfort, explicit or None)
    args.max_disparity = float(resolved["max_disparity"])
    args.convergence = float(resolved["convergence"])
    args.temporal_smooth = bool(resolved["temporal_smooth"])

    log.info(
        "🎧 --comfort %s → max_disparity=%.3f, convergence=%.2f, temporal_smooth=%s",
        args.comfort,
        args.max_disparity,
        args.convergence,
        args.temporal_smooth,
    )


def apply_playback_preset(args) -> None:
    """Resolve ``--preset`` into concrete encode knobs (D-2, #79).

    Mirrors :func:`apply_quality_preset` / :func:`_apply_comfort_preset`: the
    preset fills in values the user did not explicitly pass, and explicit
    ``--codec`` / ``--crf`` / ``--bitrate`` / ``--gop`` always win.  After
    this call ``args.codec`` and ``args.crf`` are guaranteed to be concrete
    (``h264`` / ``23`` when nothing was set, matching the pre-D-2 defaults),
    and ``args.gop`` is either a concrete frame count or ``None`` (let ffmpeg
    pick).

    ``source`` (default) fills in nothing → the legacy ``h264``/``23``
    defaults are applied as the back-fill, so behaviour is bit-exact with
    pre-D-2 unless the operator opts into ``pcvr``/``standalone``.
    """
    explicit: dict[str, object] = {}
    if args.codec is not None:
        explicit["codec"] = args.codec
    if args.crf is not None:
        explicit["crf"] = args.crf
    # --bitrate is set by apply_quality_preset (adaptive scaling); when the
    # quality path already set a bitrate it must win over the preset's None.
    if args.bitrate is not None:
        explicit["bitrate"] = args.bitrate
    if args.gop is not None:
        explicit["gop_seconds"] = None  # explicit frame count wins; skip preset GOP

    resolved = resolve_playback(args.preset, explicit or None)

    # codec / crf: back-fill the legacy defaults when the preset left them
    # None (source path, or the operator explicitly unset via --preset source).
    args.codec = resolved.get("codec") or "h264"
    args.crf = resolved.get("crf") if resolved.get("crf") is not None else 23

    # GOP: translate the preset's gop_seconds into a frame count using the
    # resolved fps (1s default per the playback-side seek-precision guidance).
    # An explicit --gop (frame count) always wins; it is already on args.gop.
    gop_seconds = resolved.get("gop_seconds")
    if args.gop is None and gop_seconds is not None:
        fps = int(args.fps) if args.fps else 30
        args.gop = round(gop_seconds * fps)
        # keyint_min follows the same count so the first keyframe lands on the
        # GOP boundary (IDR-seekable segments).
        args._gop_keyint_min = args.gop  # type: ignore[attr-defined]
    else:
        args._gop_keyint_min = args.gop  # type: ignore[attr-defined]  # may be None

    # faststart / force_idr are read by the ffmpeg command-builder; store
    # them on args so the streaming / chunked / metadata paths all see them.
    args._preset_faststart = resolved.get("faststart")  # type: ignore[attr-defined]
    args._preset_force_idr = resolved.get("force_idr")  # type: ignore[attr-defined]

    log.info(
        "📺 --preset %s → codec=%s, crf=%s, gop=%s, faststart=%s",
        args.preset,
        args.codec,
        args.crf,
        args.gop,
        args._preset_faststart,
    )


# ---------------------------------------------------------------------------
# V-3: job manifest / cross-machine relay helpers (issue #36)
# ---------------------------------------------------------------------------

# Canonical --stages names → internal stage names used by the stage loop.
# "project" covers the equirect projection + optional outpaint fill;
# "encode" is the metadata/encode stage.
_MANIFEST_TO_INTERNAL = {
    "upscale": ["upscale"],
    "depth": ["depth"],
    "stereo": ["stereo"],
    "project": ["equirect", "outpaint"],
    "encode": ["metadata"],
}
_INTERNAL_TO_MANIFEST = {
    "upscale": "upscale",
    "depth": "depth",
    "stereo": "stereo",
    "equirect": "project",
    "outpaint": "project",
    "metadata": "encode",
}
MANIFEST_STAGE_NAMES = list(_MANIFEST_TO_INTERNAL)


def parse_stages_arg(value):
    """Parse --stages 'a,b,c' into canonical manifest stage names (ordered)."""
    if value is None:
        return list(MANIFEST_STAGE_NAMES)
    names = [s.strip() for s in str(value).split(",") if s.strip()]
    unknown = [n for n in names if n not in _MANIFEST_TO_INTERNAL]
    if unknown:
        raise ValueError(
            f"Unknown stage(s) in --stages: {', '.join(unknown)} (valid: {', '.join(MANIFEST_STAGE_NAMES)})"
        )
    # De-duplicate, preserve canonical order
    return [n for n in MANIFEST_STAGE_NAMES if n in names]


def _machine_label(args):
    """Machine label for manifest entries: explicit --machine, else auto."""
    if getattr(args, "machine", None):
        return args.machine
    return f"{platform.system().lower()}-{args.device or 'cpu'}"


def _stage_artifacts(args, manifest_name):
    """(inputs, outputs, params) recorded for a completed manifest stage."""
    params = {}
    if manifest_name == "upscale":
        outputs = [args.input] if args.video_upscale == "seedvr2" else []
        params = {
            "video_upscale": args.video_upscale,
            "video_upscale_factor": args.video_upscale_factor,
            "upscale": args.upscale,
        }
    elif manifest_name == "depth":
        outputs = [get_depth_dir(args)]
        params = {"depth_model": args.depth_model, "model_size": args.model_size}
    elif manifest_name == "stereo":
        outputs = [get_temp_dir(args, "left"), get_temp_dir(args, "right")]
        params = {"stereo_model": args.stereo_model, "ipd": args.ipd}
    elif manifest_name == "project":
        outputs = [get_temp_dir(args, "equirect")]
        params = {
            "output_width": args.output_width,
            "output_height": args.output_height,
            "src_hfov": args.src_hfov,
            "outpaint": args.outpaint,
        }
    elif manifest_name == "encode":
        outputs = [get_output_path(args)] if args.stage == "all" else []
        params = {"codec": args.codec, "crf": args.crf, "bitrate": args.bitrate, "fps": args.fps}
    else:  # pragma: no cover - guarded by parse_stages_arg
        outputs, params = [], {}
    return [], outputs, params


# ---------------------------------------------------------------------------
# C-1b (#191): multi-segment concat pre-stage
# ---------------------------------------------------------------------------


def _concat_segments_preprocess(args) -> Path:
    """Concatenate ``args.inputs`` into one intermediate file under --temp-dir.

    Called from :func:`main` when ``--inputs`` is given.  Returns the absolute
    path to the intermediate clip.  The intermediate lives under ``--temp-dir``
    (or the default temp dir next to the *first* input when --temp-dir is
    omitted) so it never lands under ``video/``.

    Raises :class:`ConcatError` if the segments are incompatible (its message
    names the offending segment, which main() prints verbatim).
    """
    segments = [ConcatSegment(Path(p)) for p in args.inputs]

    # Resolve the concat output path under the temp dir.  The intermediate
    # filename is stable so a retry/resume does not leave stray artefacts.
    concat_dir = get_temp_dir(args, "concat")
    concat_output = Path(concat_dir) / "_concat_intermediate.mp4"

    log.info(
        "🧩 Concatenating %d segment(s) → %s (mode=%s, crossfade=%.3fs)",
        len(segments),
        concat_output,
        args.concat_mode,
        args.concat_crossfade,
    )

    result = concat_segments(
        segments,
        concat_output,
        mode=args.concat_mode,
        crossfade=args.concat_crossfade,
    )

    # Estimate total duration by probing the intermediate so the log line
    # can report a concrete "total X seconds".  ffprobe duration may be None
    # for some codecs; fall back to the segment-count phrasing in that case.
    from pipeline.segment_concat import probe_segment as _probe_segment

    try:
        meta = _probe_segment(result)
        dur = meta.get("duration")
        if dur is not None:
            log.info(
                "🧩 已将 %d 段拼接为 %s（总时长 %.2f 秒）",
                len(segments),
                result,
                dur,
            )
        else:
            log.info(
                "🧩 已将 %d 段拼接为 %s（总时长未知）",
                len(segments),
                result,
            )
    except ConcatError:
        # Probe failure (e.g. duration unreadable) must not abort the pipeline;
        # the concat itself succeeded.  Log a weaker confirmation line.
        log.info(
            "🧩 已将 %d 段拼接为 %s（时长探测失败，不阻塞管线）",
            len(segments),
            result,
        )

    return Path(result).resolve()


def _manifest_record_stage(manifest, args, manifest_name):
    """Record one completed stage into the manifest (hashes file outputs)."""
    from pipeline.job_manifest import mark_stage_done

    inputs, outputs, params = _stage_artifacts(args, manifest_name)
    mark_stage_done(
        manifest,
        manifest_name,
        machine=_machine_label(args),
        inputs=inputs,
        outputs=outputs,
        params=params,
    )


def _manifest_prepare(args):
    """Load/create the job manifest and compute which stages to skip.

    Returns (manifest, skip_internal, stages_to_run_internal) — all None
    when no manifest/stages flags are in play (zero behaviour change).
    """
    from pipeline.job_manifest import (
        STATUS_DONE,
        ManifestError,
        completed_stages,
        get_stage,
        load_manifest,
        new_manifest,
        validate_source,
        validate_stage_outputs,
    )

    stages_arg = args.stages if isinstance(args.stages, str) else None
    manifest_arg = args.manifest if isinstance(args.manifest, str) else None
    resume_arg = args.resume_from if isinstance(args.resume_from, str) else None

    if stages_arg is None and manifest_arg is None and resume_arg is None:
        return None, None, None

    try:
        wanted = parse_stages_arg(stages_arg)
    except ValueError as e:
        log.error("❌ %s", e)
        sys.exit(2)

    manifest = None
    done = []
    if resume_arg:
        try:
            manifest = load_manifest(resume_arg)
            validate_source(manifest, args.input)
            for name in completed_stages(manifest):
                validate_stage_outputs(manifest, name)
        except ManifestError as e:
            log.error("❌ Cannot resume from %s:\n%s", resume_arg, e)
            sys.exit(1)
        done = completed_stages(manifest)
        log.info("📂 Resuming from manifest %s — completed stages: %s", resume_arg, done or "[]")
    elif manifest_arg:
        if os.path.exists(manifest_arg):
            try:
                manifest = load_manifest(manifest_arg)
            except ManifestError as e:
                log.error("❌ %s", e)
                sys.exit(1)
        else:
            # job id: input stem + short source hash prefix
            from pipeline.job_manifest import sha256_file

            src_hash = sha256_file(args.input) if os.path.isfile(args.input) else "nohash"
            job_id = f"{Path(args.input).stem}-{src_hash[:8]}"
            manifest = new_manifest(job_id, args.input, machine=_machine_label(args))

    # Map to internal stage names, skipping manifest-done stages.
    skip_internal = []
    stages_internal = []
    for name in wanted:
        stage_entry = get_stage(manifest, name) if manifest else None
        if resume_arg and stage_entry and stage_entry.get("status") == STATUS_DONE:
            skip_internal.extend(_MANIFEST_TO_INTERNAL[name])
            continue
        stages_internal.extend(_MANIFEST_TO_INTERNAL[name])

    return manifest, skip_internal, stages_internal


# ---------------------------------------------------------------------------
# K-18 (#210): the stage-all body, extracted so main() can wrap it in
# try/finally for temp-dir lifecycle (auto-derived cleanup on success,
# preserve + print path on exception).
# ---------------------------------------------------------------------------


def _stage_all_body(args, temp_dir, is_sbs, manifest, manifest_skip, manifest_stages, manifest_touched):
    """Run the ``--stage all`` pipeline stages (the body of main()'s ``if args.stage == "all"``).

    Extracted into a standalone function solely so ``main()`` can wrap the
    entire stage loop in try/finally for K-18 temp-dir lifecycle management.
    The logic inside is bit-for-bit identical to the pre-K-18 inline block;
    see the original inline version for the per-stage rationale comments.
    """
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

    # V-3: apply --stages subset + manifest-resume skip list.
    if manifest_stages is not None:
        stages_to_run = [s for s in stages_to_run if s in manifest_stages]
        if manifest_skip:
            skipped = [s for s in base_order if s in manifest_skip and s in manifest_stages]
            if skipped:
                log.info("⏭️  Manifest-resume: skipping completed stage(s): %s", skipped)
        log.info("🧩 Stage subset (--stages): running %s", stages_to_run or "[]")

    def _record(internal_stage):
        """Record a completed internal stage into the job manifest."""
        if manifest is None:
            return
        mname = _INTERNAL_TO_MANIFEST.get(internal_stage)
        if mname and mname not in manifest_touched:
            _manifest_record_stage(manifest, args, mname)
            manifest_touched.add(mname)

    _chunked = bool(getattr(args, "chunk_size", None))
    frames = None
    if need_frames or "depth" in stages_to_run:
        frames, _total = _intake_frames(args.input, args.max_frames, lazy=_chunked)

    depths = None
    left_frames, right_frames = None, None
    sbs_frames = None
    output = None
    _fused_done: set[str] = set()

    _fused_pending = (
        _chunked
        and "stereo" in stages_to_run
        and "equirect" in stages_to_run
        and "metadata" in stages_to_run
        and getattr(args, "outpaint", "none") == "none"
        and not is_sbs
    )

    for stage in stages_to_run:
        if stage == "upscale":
            frames = run_upscale_stage(args, frames)
            save_checkpoint(temp_dir, "upscale")
            _record("upscale")

        elif stage == "depth":
            if frames is None:
                frames = list(read_frames(args.input, args.max_frames))
            depths = run_depth_stage(args, frames)
            save_checkpoint(temp_dir, "depth", {"num_frames": len(depths)})
            _record("depth")

        elif stage == "stereo" and _fused_pending:
            if depths is None:
                depths = load_depth_checkpoint(args)
            if frames is None or _chunked:
                frames, _ = _intake_frames(args.input, args.max_frames, lazy=False)
            if depths is None:
                log.warning("⚠️  Recomputing depth (stale checkpoint) before stereo.")
                depths = run_depth_stage(args, frames)
                save_checkpoint(temp_dir, "depth", {"num_frames": len(depths)})
            output = run_chunked_fused_stage(args, frames, depths)
            save_checkpoint(temp_dir, "stereo", {"num_frames": len(depths)})
            _record("stereo")
            _record("equirect")
            _record("metadata")
            _fused_done = {"equirect", "outpaint", "metadata"}

        elif stage == "stereo":
            if depths is None:
                depths = load_depth_checkpoint(args)
            if frames is None or _chunked:
                frames, _ = _intake_frames(args.input, args.max_frames, lazy=False)
            if depths is None:
                log.warning("⚠️  Recomputing depth (stale checkpoint) before stereo.")
                depths = run_depth_stage(args, frames)
                save_checkpoint(temp_dir, "depth", {"num_frames": len(depths)})
            left_frames, right_frames = run_stereo_stage(args, frames, depths)
            save_checkpoint(temp_dir, "stereo", {"num_frames": len(left_frames)})
            _record("stereo")

        elif stage == "equirect":
            if stage in _fused_done:
                continue
            if is_sbs and left_frames is None:
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
            _record("equirect")

        elif stage == "outpaint":
            if stage in _fused_done:
                continue
            if sbs_frames is None:
                import glob

                eq_dir = get_temp_dir(args, "equirect")
                files = sorted(glob.glob(os.path.join(eq_dir, "*.png")))
                sbs_frames = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in files]
                log.info(f"📂 Loaded {len(sbs_frames)} equirect frames from checkpoint for outpainting")
            sbs_frames = run_outpaint_stage(args, sbs_frames)
            save_checkpoint(temp_dir, "outpaint", {"num_frames": len(sbs_frames)})
            _record("outpaint")

        elif stage == "metadata":
            if stage in _fused_done:
                continue
            if sbs_frames is None:
                import glob

                eq_dir = get_temp_dir(args, "equirect")
                files = sorted(glob.glob(os.path.join(eq_dir, "*.png")))
                sbs_frames = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in files]
                log.info(f"📂 Loaded {len(sbs_frames)} equirect frames from checkpoint")
            output = run_metadata_stage(args, sbs_frames)
            _record("metadata")

    if output:
        log.info(f"✅ Pipeline complete → {output}")

    if output and getattr(args, "copy_audio_from", None):
        _copy_audio_to_output(output, args.copy_audio_from, re_inject=True)
    elif output:
        _maybe_copy_audio_from_input(output, args.input, re_inject=True)

    if output:
        _write_sidecar_from_args(output, "vr180", args)

    manifest_out = args.manifest if isinstance(args.manifest, str) else None
    if manifest is not None and manifest_out:
        from pipeline.job_manifest import save_manifest

        save_manifest(manifest, manifest_out)


def main():
    args = parse_args()

    # C-1b (#191): enforce "exactly one" of --input / --inputs.  The argparse
    # mutually-exclusive group above already errors when both are given; this
    # covers the other case (neither given) with a message that names both
    # flags, matching the operator-facing tone of the other entry checks.
    # args.inputs is ``None`` or a real ``list[str]`` when parsed by argparse.
    # Some callers (existing streaming-backend tests) construct ``args`` as a
    # bare MagicMock where every unset attribute is a truthy Mock; treat any
    # non-list value as "no --inputs" so those tests keep working unchanged.
    # Normalise args.inputs in place so the concat pre-stage (which reads it)
    # and every downstream consumer see the same canonical value.
    args.inputs = args.inputs if isinstance(getattr(args, "inputs", None), list) else None
    if args.input is None and args.inputs is None:
        err = "❌ either --input/-i or --inputs is required"
        log.error(err)
        print(err, file=sys.stderr)
        sys.exit(2)

    # C-1b (#191): when --inputs is given, concatenate the segments into a
    # single intermediate under --temp-dir and feed that intermediate into the
    # existing pipeline by replacing args.input.  Everything downstream (fps
    # inheritance, SBS detection, stage loop, audio remux, sidecar) sees a
    # single "input" and is otherwise unchanged.
    if args.inputs is not None:
        try:
            args.input = str(_concat_segments_preprocess(args))
        except ConcatError as exc:
            # ConcatError messages already name the offending segment with its
            # actual values — surface them verbatim so the operator can spot
            # which clip is the odd one out (acceptance criterion).
            msg = f"❌ Segment compatibility check failed:\n{exc}"
            log.error(msg)
            print(msg, file=sys.stderr)
            sys.exit(2)

    apply_quality_preset(args)
    _apply_comfort_preset(args)
    # D-2 (#79): the playback preset is resolved after fps inheritance below
    # (the 1-second GOP needs a frame rate to translate to a frame count).

    # V-3: job manifest — load/create, hash-validate completed stages,
    # compute which stages to skip.  None when no manifest flags are given
    # (behaviour then identical to before issue #36).
    manifest, manifest_skip, manifest_stages = _manifest_prepare(args)
    manifest_touched: set[str] = set()

    # Auto-detect device if not specified
    if args.device is None:
        args.device = detect_best_device()
    else:
        args.device = resolve_device(args.device)

    # R-1: SeedVR2 pre-stage — upscale the input video file before any frame loading
    if args.video_upscale == "seedvr2":
        if manifest_skip and "upscale" in manifest_skip:
            log.info("⏭️  Skipping SeedVR2 pre-stage (upscale already done in manifest)")
        else:
            original_input = args.input
            args.input = run_seedvr2_prestage(args)
            log.info("SeedVR2 pre-stage: input replaced %s → %s", original_input, args.input)
            if manifest is not None:
                _manifest_record_stage(manifest, args, "upscale")
                manifest_touched.add("upscale")

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

    # D-2 (#79): resolve the playback preset now that fps is known (the 1s
    # GOP must be translated to a frame count).  codec/crf back-fill, GOP,
    # +faststart, and IDR flags land on args here so every encode path below
    # (streaming / chunked / metadata) sees them.  ``source`` = no-op.
    apply_playback_preset(args)

    # Streaming pipeline mode (PRD §7.2).  V-3: the streaming path is a
    # single fused run — it cannot be split across machines, so --stages /
    # --resume-from force the batch path.
    if args.streaming and args.stage == "all" and manifest_stages is None:
        log.info("🚀 Streaming pipeline mode (O(1) memory)")
        # I-5 (#120): the streaming path previously hard-coded Depth-Anything +
        # StereoRenderer and *silently ignored* --depth-model/--stereo-model.
        # Build the requested backends via the shared factory (same construction
        # as the batch stages) and inject them.  fallback=True → an unavailable
        # DepthCrafter/StereoCrafter logs a loud WARNING and degrades to the
        # default instead of running silently with the wrong model.
        depth_backend, depth_backend_name = build_depth_backend(args, fallback=True)
        stereo_backend, stereo_backend_name = build_stereo_backend(args, fallback=True)
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
            bitrate=args.bitrate,
            hw_encoder=args.hw_encoder,
            gop=getattr(args, "gop", None),
            force_idr=getattr(args, "_preset_force_idr", False),
            faststart=getattr(args, "_preset_faststart", None),
            depth_estimator=depth_backend,
            stereo_renderer=stereo_backend,
            depth_backend_name=depth_backend_name,
            stereo_backend_name=stereo_backend_name,
        )
        output = get_output_path(args)
        result = pipeline.process_stream(args.input, output, max_frames=args.max_frames)
        # Inject sv3d/st3d VR metadata so HMDs recognise the stream as 180° 3D
        # (issue #45 defect 2: streaming path previously skipped this entirely).
        # SBS frame is (2×output_width) × output_height (horizontal concat).
        from pipeline.spherical_injector import inject_spherical_metadata

        injected = result + ".vr.mp4"
        try:
            inject_spherical_metadata(
                result,
                injected,
                width=args.output_width * 2,
                height=args.output_height,
                stereo_mode="sbs",
            )
        except Exception as e:
            log.error(f"❌ VR metadata injection failed for {result}: {e}")
            raise
        os.replace(injected, result)
        log.info(f"✅ Streaming pipeline complete (sv3d/st3d injected) → {result}")
        # H-1.2 (#132): audio passthrough for the streaming branch.  The
        # batch path below remuxes --copy-audio-from (or the input's own
        # audio) after metadata injection; the streaming branch previously
        # returned before reaching it, silently producing a silent video.
        # Reuse the same helpers — re_inject=True re-embeds sv3d/st3d after
        # the remux (issue #91: ffmpeg -c copy drops the sample-entry boxes).
        if getattr(args, "copy_audio_from", None):
            _copy_audio_to_output(result, args.copy_audio_from, re_inject=True)
        else:
            _maybe_copy_audio_from_input(result, args.input, re_inject=True)
        # D-1: sidecar — one JSON per output artefact (see pipeline.sidecar).
        _write_sidecar_from_args(result, "vr180", args)
        return

    # R-5: Fulldome projection mode — skip all depth/stereo/equirect/metadata.
    # Fulldome is a single fused conversion with no VR180 stage loop, so the
    # V-3 manifest/stage-subset flags do not apply to it.
    if args.projection == "fulldome":
        if manifest_stages is not None:
            log.error("❌ --stages/--manifest/--resume-from do not apply to --projection fulldome")
            sys.exit(2)
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
        _write_sidecar_from_args(result, "fulldome", args, fov=args.dome_fov, eye_size=(args.dome_size, args.dome_size))
        return

    if args.stage == "all":
        # Smart SBS detection: if input is already SBS, skip depth/stereo
        is_sbs = detect_sbs_input(args.input, force_sbs=args.force_sbs)

        # K-18 (#210): resolve the temp-dir lifecycle.  The lifecycle dict
        # records whether the dir is auto-derived, whether this run created
        # it, and whether --keep-temp was given.  The stage loop below is
        # wrapped in try/finally: on success the auto-derived dir is cleaned
        # up if this run owns it; on exception the dir is preserved and its
        # path is printed for debugging.  ``get_temp_dir`` (subdirectory
        # children like depth/ / left/) still works because the lifecycle
        # helper already created the parent.
        _lifecycle = _resolve_temp_dir_lifecycle(args)
        temp_dir = str(_lifecycle["temp_dir"])

        _success = False
        try:
            _stage_all_body(args, temp_dir, is_sbs, manifest, manifest_skip, manifest_stages, manifest_touched)
            _success = True
        except BaseException:
            # K-18: exception path — preserve the temp dir for debugging
            # and print where it lives so the operator can inspect artefacts.
            if _lifecycle.get("auto"):
                print(f"❌ 管线异常退出，中间目录已保留用于排障: {_lifecycle['temp_dir']}")
            raise
        finally:
            # K-18: success path only — clean up if this run owns the
            # auto-derived dir and --keep-temp was not given.  On exception
            # the dir is preserved (do not delete); --keep-temp prints the
            # path on both success and exception (so the operator always
            # knows where to look).
            if _success:
                _cleanup_temp_dir_if_owned(_lifecycle)
            if _lifecycle.get("auto") and _lifecycle.get("keep"):
                _print_temp_dir_path(_lifecycle)

    elif args.stage == "depth":
        # Depth is the sole chunked consumer here — intake lazily when
        # --chunk-size is set (V-4.1a); legacy list otherwise.
        _lazy_depth = bool(getattr(args, "chunk_size", None))
        frames, _ = _intake_frames(args.input, args.max_frames, lazy=_lazy_depth)
        run_depth_stage(args, frames)

    elif args.stage == "stereo":
        # Stereo needs frames as a materialised list; keep legacy intake.
        frames = list(read_frames(args.input, args.max_frames))
        depth_dir = get_depth_dir(args)
        # I-6 (#121): warn if the cached depth came from a different model/
        # params — this single-stage entry just loads the checkpoint, so it
        # cannot recompute; the warning surfaces the mismatch to the operator.
        meta = load_depth_meta(depth_dir)
        if not depth_meta_matches(args, meta):
            log.warning(
                "⚠️  Depth checkpoint at %s does not match current model/params "
                "(meta: %s) — loading anyway for --stage stereo (re-run --stage "
                "depth to regenerate).",
                depth_dir,
                meta,
            )
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

    elif args.stage == "outpaint":
        eq_dir = get_temp_dir(args, "equirect")
        import glob

        files = sorted(glob.glob(os.path.join(eq_dir, "*.png")))
        frames = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in files]
        run_outpaint_stage(args, frames)

    elif args.stage == "metadata":
        eq_dir = get_temp_dir(args, "equirect")
        import glob

        files = sorted(glob.glob(os.path.join(eq_dir, "*.png")))
        frames = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in files]
        run_metadata_stage(args, frames)


# ---------------------------------------------------------------------------
# H-1: audio passthrough helpers (issue #73)
# ---------------------------------------------------------------------------


def _copy_audio_to_output(output: str, source: str, *, re_inject: bool = False) -> None:
    """Attach an audio track from *source* to the finished VR180 *output*.

    When *re_inject* is True, re-embeds sv3d/st3d after the remux — required
    because ffmpeg ``-c copy`` with ``-map 0:v -map 1:a`` drops the sample-
    entry boxes (issue #91). The re-injection is self-verifying and raises
    on failure.
    """
    from pipeline.audio_mux import copy_audio_to
    from pipeline.spherical_injector import inject_spherical_metadata

    log.info("🔊 Copying audio track %s → %s", source, output)
    copy_audio_to(output, source)
    if re_inject:
        log.info("🔊 Re-injecting sv3d/st3d (ffmpeg -c copy drops sample-entry boxes)")
        inject_spherical_metadata(output, output + ".vr.mp4", stereo_mode="sbs")
        os.replace(output + ".vr.mp4", output)
    log.info("✅ Audio remux complete → %s", output)


def _maybe_copy_audio_from_input(output: str, input_path: str, *, re_inject: bool = False) -> None:
    """If the input video has an audio stream, copy it into the VR180 output.

    No-op (silent) when the input has no audio — matches the orchestrator's
    behaviour. Does not raise on missing audio; only on a real remux failure.
    """
    from pipeline.audio_mux import has_audio_stream

    if not has_audio_stream(input_path):
        return
    log.info("🔊 Input %s has an audio stream — remuxing into %s", input_path, output)
    _copy_audio_to_output(output, input_path, re_inject=re_inject)


# ---------------------------------------------------------------------------
# D-1: sidecar JSON metadata (issue #78)
# ---------------------------------------------------------------------------


def _write_sidecar(output_path: str, route: str) -> None:
    """Best-effort sidecar write for *output_path*.

    Route is ``vr180`` (SBS equirect) or ``fulldome`` (mono fisheye). Never
    fails the pipeline for a sidecar write error — the artefact itself is
    already delivered; the JSON is supplementary metadata.

    ``eye_resolution`` is omitted here: the run_pipeline CLI resolves it
    lazily into ``args.output_width`` / ``args.output_height`` *after* quality
    preset application, and the streaming/fulldome paths already have those
    values populated on ``args`` when they reach this helper. We prefer to
    read them off ``args`` directly (see ``_write_sidecar_from_args`` below).
    """
    _write_sidecar_from_args(output_path, route, None)


def _write_sidecar_from_args(
    output_path: str,
    route: str,
    args: argparse.Namespace | None,
    *,
    fov: float | None = None,
    eye_size: tuple[int, int] | None = None,
) -> None:
    """``_write_sidecar`` with optional args so the batch stage (which holds
    ``args.output_width`` / ``args.output_height``) can include eye resolution.

    ``fov`` and ``eye_size`` are optional overrides so the fulldome path can
    pass its (potentially non-180°) dome_fov and its square output_size.  When
    omitted the VR180 defaults (180°, args.resolved eye) apply.

    D-3: the ``immersive`` block is normalised through
    :func:`pipeline.sidecar.normalize_immersive` so the D-3 projection
    contract (required projection/fov_deg/stereo_layout/eye_resolution) is
    enforced and the projection/stereo tags come from the canonical enums.
    """
    from pipeline.sidecar import (
        PROJECTION_EQUIRECT180,
        PROJECTION_FISHEYE_DOME,
        STEREO_MONO,
        STEREO_SIDE_BY_SIDE,
        normalize_immersive,
        write_sidecar,
    )

    vr180 = route == "vr180"
    immersive = {
        "projection": PROJECTION_EQUIRECT180 if vr180 else PROJECTION_FISHEYE_DOME,
        "fov_deg": (fov if fov is not None else 180),
        "stereo_layout": STEREO_SIDE_BY_SIDE if vr180 else STEREO_MONO,
        "spatial_metadata": ["sv3d", "st3d"] if vr180 else [],
    }

    # eye_resolution: required by the D-3 contract.  Prefer the explicit
    # eye_size override (fulldome), else resolve from args (VR180 SBS).
    if eye_size is not None:
        w, h = eye_size
        immersive["eye_resolution"] = [int(w), int(h)]
    elif args is not None:
        w = int(getattr(args, "output_width", 0) or 0)
        h = int(getattr(args, "output_height", 0) or 0)
        if w and h:
            immersive["eye_resolution"] = [w, h]

    # D-2 (#79): record the playback preset in the generation block so the
    # downstream player knows which encode tier the artefact was tuned for.
    generation: dict[str, Any] = {"route": route}
    if args is not None:
        preset = getattr(args, "preset", None)
        if preset:
            generation["preset"] = preset

    try:
        write_sidecar(output_path, immersive=normalize_immersive(immersive), generation=generation)
    except Exception as exc:
        log.warning("⚠️  Sidecar write failed for %s: %s", output_path, exc)


if __name__ == "__main__":
    main()
