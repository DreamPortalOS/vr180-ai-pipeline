"""
Streaming Pipeline (PRD §7.2)
O(1) memory video processing — reads frames one-at-a-time via cv2.VideoCapture,
processes depth → stereo → equirect, and writes directly to an ffmpeg output pipe.
No frame buffers accumulate in RAM.
"""

import contextlib
import logging
import subprocess

import cv2

from pipeline.depth_estimator import DepthEstimator
from pipeline.device_utils import resolve_device
from pipeline.equirectangular_mapper import EquirectangularMapper
from pipeline.stereo_renderer import StereoRenderer

log = logging.getLogger("vr180-streaming")

# Reference eye resolution for bitrate scaling (the legacy default).
REFERENCE_EYE_SIZE = 1920
# Baseline bitrate (Mbps) that produced acceptable quality at 1920²/eye.
BASELINE_BITRATE_MBPS = 20.0

# --quality presets: name -> per-eye square size (px).
# preview  : 1920²/eye — fast iteration, legacy resolution.
# standard : 2880²/eye — streaming, default quality path.
# high     : 3840²/eye — streaming, max sharpness for Quest-class HMDs.
QUALITY_PRESETS: dict[str, int] = {
    "preview": 1920,
    "standard": 2880,
    "high": 3840,
}
DEFAULT_QUALITY = "standard"

# H.264 NVENC hardware width cap: frames wider than this need HEVC (issue #45).
NVENC_MAX_WIDTH = 4096


def select_encoder(codec: str, sbs_width: int, hw: bool = False) -> list[str]:
    """Pick the ffmpeg encoder args for an SBS frame of the given width.

    Large-frame safety (issue #45 defect 3): libx264 on an 8K-class SBS frame
    (7680×3840) exhausts process RAM during encoder init, and H.264 NVENC
    hard-caps at 4096 px wide — so wide frames must use HEVC.

    Args:
        codec: Requested codec ('h264' or 'h265').
        sbs_width: Full side-by-side frame width in pixels (2× per-eye width).
        hw: Whether NVENC hardware encoding is available.

    Returns:
        ffmpeg encoder args, e.g. ``["-c:v", "hevc_nvenc"]`` or
        ``["-c:v", "libx265", "-preset", "fast"]``.

    Raises:
        ValueError: If libx264 would be used on a frame wider than
            NVENC_MAX_WIDTH (the known OOM configuration).
    """
    if hw:
        if sbs_width > NVENC_MAX_WIDTH:
            return ["-c:v", "hevc_nvenc"]
        return ["-c:v", "h264_nvenc" if codec == "h264" else "hevc_nvenc"]

    # Software encoding.
    if codec == "h265" or sbs_width > NVENC_MAX_WIDTH:
        if codec == "h264" and sbs_width > NVENC_MAX_WIDTH:
            log.warning(
                f"⚠️  SBS width {sbs_width} > {NVENC_MAX_WIDTH}: libx264 would OOM on "
                "frames this large — forcing libx265 (-preset fast). Expect high RAM "
                "usage; use a CUDA machine for hardware encoding if possible."
            )
        return ["-c:v", "libx265", "-preset", "fast"]
    return ["-c:v", "libx264"]


def resolve_quality(
    quality: str,
    explicit_eye_size: int | None = None,
) -> tuple[int, bool]:
    """Map a --quality preset to (eye_size, streaming).

    Args:
        quality: One of QUALITY_PRESETS keys.
        explicit_eye_size: Explicit per-eye size override (e.g. from
            --output-width). When given, it wins over the preset.

    Returns:
        (eye_size, streaming) — per-eye square resolution and whether the
        streaming (O(1) memory) pipeline should be used. All presets except
        ``preview`` imply streaming.

    Raises:
        ValueError: On unknown quality preset.
    """
    if quality not in QUALITY_PRESETS:
        raise ValueError(f"Unknown quality preset: {quality!r} (choose from {sorted(QUALITY_PRESETS)})")
    eye_size = explicit_eye_size if explicit_eye_size is not None else QUALITY_PRESETS[quality]
    streaming = quality != "preview"
    return eye_size, streaming


def scaled_bitrate_mbps(
    eye_size: int,
    base_mbps: float = BASELINE_BITRATE_MBPS,
    reference_eye_size: int = REFERENCE_EYE_SIZE,
    max_mbps: float | None = None,
) -> float:
    """Scale output bitrate linearly with total pixel area.

    Bitrate scales with (eye_size / reference)² so that bits-per-pixel stay
    constant relative to the 1920²/eye reference tier:
      - 1920² → base (1×)
      - 2880² → 2.25× base
      - 3840² → 4× base

    Args:
        eye_size: Per-eye square resolution (px).
        base_mbps: Bitrate at the reference resolution (Mbps).
        reference_eye_size: Resolution the base bitrate was tuned for.
        max_mbps: Optional upper clamp (Mbps). ``None`` = no cap.

    Returns:
        Scaled bitrate in Mbps, clamped to ``max_mbps`` if given.
    """
    scale = (eye_size / reference_eye_size) ** 2
    bitrate = base_mbps * scale
    if max_mbps is not None:
        bitrate = min(bitrate, max_mbps)
    return bitrate


class StreamingPipeline:
    """Stream-based VR180 conversion with O(1) memory footprint.

    Instead of loading all frames into RAM, this pipeline:
      1. Opens the input video with cv2.VideoCapture
      2. Reads one frame at a time
      3. Runs depth estimation → stereo rendering → equirectangular mapping
      4. Pipes the processed frame directly into ffmpeg for encoding
      5. Releases intermediate tensors after each frame

    This prevents the ~98.4 GB memory overflow that occurs when caching
    all frames in a list for a long 8K video.
    """

    def __init__(
        self,
        model_size: str = "small",
        device: str | None = None,
        ipd: float = 0.064,
        max_disparity: float = 0.05,
        output_width: int = 3840,
        output_height: int = 1920,
        src_hfov: float = 120.0,
        codec: str = "h264",
        crf: int = 23,
        fps: int = 30,
        bitrate: str | None = None,
        hw_encoder: bool | None = None,
    ):
        self.model_size = model_size
        self.device = resolve_device(device)
        self.ipd = ipd
        self.max_disparity = max_disparity
        self.output_width = output_width
        self.output_height = output_height
        self.src_hfov = src_hfov
        self.codec = codec
        self.crf = crf
        self.fps = fps
        self.bitrate = bitrate
        # Hardware (NVENC) encoding: auto-enabled on CUDA machines unless
        # explicitly overridden (injectable for tests — issue #45 defect 3).
        self.hw_encoder = self.device.startswith("cuda") if hw_encoder is None else hw_encoder

        # Initialise pipeline stages
        self.depth_estimator = DepthEstimator(
            model_size=model_size,
            device=self.device,
            calibrate=True,
        )
        self.stereo_renderer = StereoRenderer(
            ipd=ipd,
            max_disparity=max_disparity,
        )
        self.eq_mapper = EquirectangularMapper(
            output_width=output_width,
            output_height=output_height,
            src_hfov=src_hfov,
            use_ffmpeg=True,
        )

    def _build_ffmpeg_cmd(self, output_path: str, width: int, height: int) -> list[str]:
        """Build the ffmpeg command list for raw-frame piping.

        Returns:
            List of command-line arguments for subprocess.
        """
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(self.fps),
            "-i",
            "pipe:0",
        ]
        # Encoder chosen by frame size + hardware availability (issue #45
        # defect 3): libx264 OOMs on >4096-wide SBS frames; NVENC H.264 caps
        # at 4096 wide so 8K-class output must go through HEVC.
        cmd += select_encoder(self.codec, width, hw=self.hw_encoder)
        if self.bitrate:
            # Explicit target bitrate (e.g. "80M") overrides CRF.
            cmd += ["-b:v", self.bitrate]
        else:
            cmd += ["-crf", str(self.crf)]
        cmd += [
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            output_path,
        ]
        return cmd

    def _open_ffmpeg_writer(self, output_path: str, width: int, height: int) -> subprocess.Popen:
        """Open an ffmpeg subprocess that accepts raw RGB frames on stdin.

        stderr is discarded (DEVNULL): an undrained PIPE fills its 64 KB
        buffer and deadlocks the whole pipeline (same class of bug as #21).
        """
        cmd = self._build_ffmpeg_cmd(output_path, width, height)
        log.info(f"ffmpeg cmd: {' '.join(cmd)}")
        return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def process_stream(
        self,
        input_path: str,
        output_path: str,
        max_frames: int | None = None,
    ) -> str:
        """Process video frame-by-frame, writing directly to ffmpeg pipe.

        Args:
            input_path: Path to input 2D video.
            output_path: Path for output VR180 video.
            max_frames: Optional cap on number of frames (for testing).

        Returns:
            Path to the written output video.

        Raises:
            RuntimeError: If input video cannot be opened.
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {input_path}")

        in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if max_frames:
            total = min(total, max_frames)

        log.info(f"Input: {in_w}×{in_h}, {fps:.2f} fps, {total} frames")

        # SBS output = side-by-side stereo: map_stereo_pair concatenates
        # left|right horizontally (axis=1), so the frame is (H, 2W, 3) —
        # the declared ffmpeg size must be 2W×H, not W×2H (issue #45 defect 1:
        # the old W×2H declaration produced row-interleaved garbage video).
        out_w = self.output_width * 2
        out_h = self.output_height
        log.info(f"Target output: {out_w}×{out_h} (SBS, {self.output_width}² per eye)")

        proc = self._open_ffmpeg_writer(output_path, out_w, out_h)

        frame_idx = 0
        try:
            while frame_idx < total:
                ret, bgr = cap.read()
                if not ret:
                    break

                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

                # --- Stage 1: Depth estimation ---
                depth = self.depth_estimator.estimate(rgb)

                # --- Stage 2: Stereo rendering ---
                left, right = self.stereo_renderer.render(rgb, depth)

                # --- Stage 3: Equirectangular mapping ---
                sbs = self.eq_mapper.map_stereo_pair(left, right)

                # Write raw RGB to ffmpeg pipe
                try:
                    proc.stdin.write(sbs.tobytes())
                except BrokenPipeError:
                    raise RuntimeError(
                        f"ffmpeg encoder died after {frame_idx} frames — "
                        "check encoder availability/limits for this resolution "
                        f"({out_w}×{out_h})"
                    ) from None

                # Release intermediates to keep memory O(1)
                del depth, left, right, sbs, rgb, bgr

                frame_idx += 1
                if frame_idx % 10 == 0:
                    log.info(f"  [{frame_idx}/{total}] frames processed")

        finally:
            cap.release()
            # encoder may already be dead; returncode check below reports it
            with contextlib.suppress(BrokenPipeError, OSError):
                proc.stdin.close()
            proc.wait()

        # Reached only when the loop finished without an exception — a non-zero
        # exit here means ffmpeg failed silently (would previously return
        # "success" with a corrupt/empty file).
        if proc.returncode != 0:
            raise RuntimeError(
                f"ffmpeg encoding failed (exit code {proc.returncode}) after "
                f"{frame_idx} frames — rerun with stderr visible to see the ffmpeg error"
            )

        log.info(f"✅ Streaming complete: {frame_idx} frames → {output_path}")
        return output_path


def run_streaming_pipeline(
    input_path: str,
    output_path: str,
    model_size: str = "small",
    device: str | None = None,
    ipd: float = 0.064,
    max_disparity: float = 0.05,
    output_width: int = 3840,
    output_height: int = 1920,
    src_hfov: float = 70.0,
    codec: str = "h264",
    crf: int = 23,
    fps: int = 30,
    flip_vertical: bool = True,
    max_frames: int | None = None,
    bitrate: str | None = None,
    hw_encoder: bool | None = None,
) -> str:
    """Convenience function to run the streaming pipeline in one call.

    Args:
        input_path: Source 2D video path.
        output_path: Destination VR180 video path.
        model_size: Depth model size.
        device: Compute device (auto-detected if None).
        ipd: Inter-pupillary distance in metres.
        max_disparity: Max stereo disparity fraction.
        output_width: Equirectangular width per eye.
        output_height: Equirectangular height per eye.
        src_hfov: Source camera horizontal FOV.
        codec: Output codec ('h264' or 'h265').
        crf: Constant rate factor.
        fps: Output frame rate.
        flip_vertical: Flip for VR headset compatibility.
        max_frames: Optional frame cap (for testing).

    Returns:
        Path to the output video.
    """
    pipeline = StreamingPipeline(
        model_size=model_size,
        device=device,
        ipd=ipd,
        max_disparity=max_disparity,
        output_width=output_width,
        output_height=output_height,
        src_hfov=src_hfov,
        codec=codec,
        crf=crf,
        fps=fps,
        flip_vertical=flip_vertical,
        bitrate=bitrate,
        hw_encoder=hw_encoder,
    )
    return pipeline.process_stream(input_path, output_path, max_frames=max_frames)
