"""
Streaming Pipeline (PRD §7.2)
O(1) memory video processing — reads frames one-at-a-time via cv2.VideoCapture,
processes depth → stereo → equirect, and writes directly to an ffmpeg output pipe.
No frame buffers accumulate in RAM.
"""

import contextlib
import logging
import subprocess
import tempfile

import cv2
import numpy as np

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

# Process-local NVENC probe cache: encoder name -> (available, stderr_summary).
# A probe runs ffmpeg once per encoder per process (issue #49).
_NVENC_PROBE_CACHE: dict[str, tuple[bool, str]] = {}

# How many bytes of stderr tail to keep for diagnostics.
STDERR_TAIL_BYTES = 4096


def _probe_nvenc_detail(encoder: str, ffmpeg: str = "ffmpeg") -> tuple[bool, str]:
    """Run a minimal ffmpeg job with *encoder* and report availability.

    Returns:
        (available, stderr_summary) — ``available`` is True only when ffmpeg
        exits 0; ``stderr_summary`` is the tail of ffmpeg's stderr for logging.
    """
    cmd = [
        ffmpeg,
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=256x256:d=0.1",
        "-frames:v",
        "1",
        "-c:v",
        encoder,
        "-f",
        "null",
        "-",
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as e:
        # ffmpeg binary missing / failed to launch / timed out.
        return False, str(e)
    stderr_tail = (result.stderr or "")[-STDERR_TAIL_BYTES:].strip()
    return result.returncode == 0, stderr_tail


def probe_nvenc(encoder: str, ffmpeg: str = "ffmpeg") -> bool:
    """Probe whether an NVENC encoder actually works on this machine.

    CUDA being available does NOT imply NVENC works — e.g. driver 580.88 with
    a newer ffmpeg nightly reports "Driver does not support the required nvenc
    API version" and the encoder fails to open (issue #49). This runs a tiny
    synthetic one-frame encode with the target encoder and checks the exit
    code. The result is cached per encoder for the lifetime of the process.

    Args:
        encoder: ffmpeg encoder name, e.g. ``hevc_nvenc`` / ``h264_nvenc``.
        ffmpeg: ffmpeg executable (default ``ffmpeg`` from PATH).

    Returns:
        True if the encoder successfully encoded one synthetic frame.
    """
    if encoder not in _NVENC_PROBE_CACHE:
        _NVENC_PROBE_CACHE[encoder] = _probe_nvenc_detail(encoder, ffmpeg)
    return _NVENC_PROBE_CACHE[encoder][0]


def _resolve_hw_encoder(
    device: str,
    codec: str,
    sbs_width: int,
    hw_encoder: bool | str | None,
) -> bool:
    """Resolve the effective hardware-encoding flag (issue #49).

    Args:
        device: Resolved compute device string ("cuda", "mps", "cpu").
        codec: Requested codec ('h264' or 'h265').
        sbs_width: Full SBS frame width — determines which NVENC encoder the
            pipeline would actually use (H.264 NVENC caps at NVENC_MAX_WIDTH).
        hw_encoder: ``None``/``"auto"`` = probe; ``True``/``"on"`` = force
            NVENC (skip probe, user takes the risk); ``False``/``"off"`` =
            software encoding.

    Returns:
        True to use NVENC, False for software encoding.
    """
    if hw_encoder in (True, "on"):
        return True
    if hw_encoder in (False, "off"):
        return False
    # auto: only CUDA machines can have NVENC at all.
    if not device.startswith("cuda"):
        return False
    # Probe the encoder the pipeline would actually select for this frame size.
    encoder = select_encoder(codec, sbs_width, hw=True)[1]
    if probe_nvenc(encoder):
        return True
    stderr_summary = _NVENC_PROBE_CACHE.get(encoder, (False, ""))[1]
    log.warning(
        "⚠️  NVENC encoder %s unavailable — falling back to software encoding. "
        "ffmpeg says: %s — 升级 NVIDIA 驱动 ≥610 可启用硬编 (upgrade NVIDIA driver to ≥610 to enable NVENC)",
        encoder,
        stderr_summary or "(no stderr)",
    )
    return False


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


def preset_encode_args(
    gop: int | None = None,
    force_idr: bool = False,
    faststart: bool | None = None,
) -> list[str]:
    """Build the ffmpeg args for the D-2 playback-preset encode knobs.

    Encapsulates the three playback-side constraints (issue #79):

      - **Fixed 1s GOP** (``-g`` / ``-keyint_min``) so seek precision is
        bounded to one second.  ``-sc_threshold 0`` disables scene-cut
        keyframes so the GOP stays fixed.
      - **Segment-head IDR** (``-force_key_frames``) — a closed, seekable
        GOP head as the playback end's dual-MediaPlayer crossfade needs.
      - **+faststart** (``-movflags +faststart``) — moov 前置 for fast起播.
        ``None`` = leave the caller's movflags untouched (passthrough).

    Args:
        gop: GOP length in frames (already translated from seconds by the
            caller).  ``None``/``0`` = let ffmpeg pick (no GOP args).
        force_idr: force an IDR at every GOP boundary.
        faststart: ``True`` adds ``+faststart``; ``False`` drops it; ``None``
            leaves the caller's movflags as-is.

    Returns:
        A flat list of ffmpeg arguments, possibly empty.
    """
    args: list[str] = []
    if gop:  # 0 or None = unset
        args += ["-g", str(gop), "-keyint_min", str(gop), "-sc_threshold", "0"]
        if force_idr:
            # Force an IDR-style keyframe at the GOP interval.  Using the
            # interval form avoids pinning every single frame position
            # (which would defeat the fixed-GOP intent on variable fps).
            args += ["-force_key_frames", f"expr:gte(t,n_forced*{gop})"]
    if faststart is True:
        args += ["-movflags", "+faststart"]
    elif faststart is False:
        args += ["-movflags", "0"]
    # None = no movflags args (caller keeps its own)
    return args


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
        hw_encoder: bool | str | None = None,
        gop: int | None = None,
        force_idr: bool = False,
        faststart: bool | None = None,
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
        # D-2 (#79): playback-preset encode knobs — fixed 1s GOP, segment-head
        # IDR, +faststart.  ``None``/False = off (pre-D-2 behaviour).
        self.gop = gop
        self.force_idr = force_idr
        self.faststart = faststart
        # Hardware (NVENC) encoding — issue #49: CUDA availability does NOT
        # imply NVENC works (driver/ffmpeg ABI mismatch). "auto" (None) probes
        # the actual encoder with a tiny synthetic encode and falls back to
        # software on failure; "on" forces NVENC without probing; "off" forces
        # software. Booleans kept for backward compatibility (True=on, False=off).
        self.hw_encoder = _resolve_hw_encoder(
            self.device,
            codec,
            output_width * 2,
            hw_encoder,
        )

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
        # D-2 (#79): playback-preset encode knobs (fixed GOP / IDR / faststart).
        # When no preset set a movflags (faststart is None), preserve the
        # pre-D-2 default of +faststart so behaviour is unchanged without a
        # preset; otherwise preset_encode_args owns the movflags arg.
        cmd += preset_encode_args(self.gop, self.force_idr, self.faststart)
        if self.faststart is None:
            cmd += ["-movflags", "+faststart"]
        cmd += [
            "-pix_fmt",
            "yuv420p",
            output_path,
        ]
        return cmd

    def _open_ffmpeg_writer(self, output_path: str, width: int, height: int) -> subprocess.Popen:
        """Open an ffmpeg subprocess that accepts raw RGB frames on stdin.

        stderr goes to a temp file (issue #49): DEVNULL hid fatal encoder
        errors (e.g. NVENC driver mismatch), while an undrained PIPE fills its
        64 KB buffer and deadlocks the pipeline (#21/#45). A file is drained
        by the OS at no cost, and on failure we read back only the tail for
        the error message. The file is deleted once the process finishes.
        """
        cmd = self._build_ffmpeg_cmd(output_path, width, height)
        log.info(f"ffmpeg cmd: {' '.join(cmd)}")
        # Closed later in process_stream's finally (after ffmpeg exits and the
        # tail has been read), not at this scope's exit — hence no `with`.
        stderr_file = tempfile.TemporaryFile(prefix="vr180-ffmpeg-stderr-")  # noqa: SIM115
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=stderr_file)
        # Stash the file object on the process so process_stream can read the
        # tail on failure and close it afterwards.
        proc._stderr_file = stderr_file  # type: ignore[attr-defined]
        return proc

    @staticmethod
    def _ffmpeg_stderr_summary(proc: subprocess.Popen) -> str:
        """Read the tail of ffmpeg's captured stderr for error messages."""
        # vars() not getattr(): mock objects auto-create attributes on access.
        stderr_file = vars(proc).get("_stderr_file")
        if stderr_file is None:
            return "(stderr not captured)"
        try:
            stderr_file.flush()
            stderr_file.seek(0, 2)
            size = stderr_file.tell()
            stderr_file.seek(max(0, size - STDERR_TAIL_BYTES))
            tail = stderr_file.read()
            text = tail.decode("utf-8", errors="replace").strip()
            return text or "(ffmpeg produced no stderr output)"
        except (OSError, ValueError):
            return "(stderr unreadable)"

    @staticmethod
    def _close_ffmpeg_stderr(proc: subprocess.Popen) -> None:
        stderr_file = vars(proc).get("_stderr_file")
        if stderr_file is not None:
            with contextlib.suppress(OSError):
                stderr_file.close()

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

        try:
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

                    # Write raw RGB to ffmpeg pipe. Issue #49: on Windows a dead
                    # ffmpeg (e.g. NVENC failed to open) surfaces as
                    # OSError(errno=22) from stdin.write, not BrokenPipeError —
                    # catch both, and report ffmpeg's stderr tail + exit code.
                    try:
                        proc.stdin.write(sbs.tobytes())
                    except BrokenPipeError:
                        raise RuntimeError(
                            f"ffmpeg encoder died after {frame_idx} frames — "
                            "check encoder availability/limits for this resolution "
                            f"({out_w}×{out_h}). ffmpeg stderr: {self._ffmpeg_stderr_summary(proc)}"
                        ) from None
                    except OSError as e:
                        if e.errno in (22, 32):  # EINVAL (Windows broken pipe) / EPIPE
                            raise RuntimeError(
                                f"ffmpeg encoder died after {frame_idx} frames "
                                f"(pipe write failed, errno={e.errno}; exit code "
                                f"{proc.poll()}). ffmpeg stderr: {self._ffmpeg_stderr_summary(proc)}"
                            ) from None
                        raise

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

            # Reached only when the loop finished without an exception — a
            # non-zero exit here means ffmpeg failed (issue #49: include the
            # captured stderr tail so the actual ffmpeg error is visible).
            if proc.returncode != 0:
                raise RuntimeError(
                    f"ffmpeg encoding failed (exit code {proc.returncode}) after "
                    f"{frame_idx} frames. ffmpeg stderr: {self._ffmpeg_stderr_summary(proc)}"
                )
        finally:
            self._close_ffmpeg_stderr(proc)

        log.info(f"✅ Streaming complete: {frame_idx} frames → {output_path}")
        return output_path


class RawFrameFFmpegWriter:
    """Persistent raw-pipe ffmpeg writer for incremental, chunked encoding (V-4.1b, #89).

    The streaming :class:`StreamingPipeline` fuses depth→stereo→project→encode
    per *frame* (O(1)).  The batch ``--chunk-size`` path cannot fuse per-frame
    (it needs chunk-sized depth/stereo batches for temporal state), but it can
    still avoid buffering the **whole clip's** SBS bytes: it writes each chunk's
    per-frame SBS raw bytes straight into a *single* persistent ffmpeg process
    opened here, instead of accumulating a full-length ``left_frames`` /
    ``right_frames`` / ``sbs_frames`` list and ``b"".join``-ing the raw bytes at
    the end (the V-4.1b memory contract — peak RSS ∝ ``chunk_size``, not clip
    length).

    Encoder selection reuses :func:`select_encoder` + :func:`probe_nvenc`
    (the same machinery the streaming path uses) — *imported, not duplicated*.
    stderr is captured to a temp file (issue #49) so a dead encoder surfaces a
    real error with its stderr tail instead of silently producing a bad file.

    Use as a context manager::

        with RawFrameFFmpegWriter(out, W, H, codec="h264", ...) as w:
            for chunk ...:
                w.write(sbs_frame)   # one ndarray per frame, in order
    """

    def __init__(
        self,
        output_path: str,
        width: int,
        height: int,
        *,
        codec: str = "h264",
        crf: int = 23,
        fps: int = 30,
        bitrate: str | None = None,
        hw_encoder: bool = False,
        ffmpeg: str = "ffmpeg",
        gop: int | None = None,
        force_idr: bool = False,
        faststart: bool | None = None,
    ):
        self.output_path = output_path
        self.width = width
        self.height = height
        self.codec = codec
        self.crf = crf
        self.fps = fps
        self.bitrate = bitrate
        self.hw_encoder = hw_encoder
        self.ffmpeg = ffmpeg
        # D-2 (#79): playback-preset encode knobs.
        self.gop = gop
        self.force_idr = force_idr
        self.faststart = faststart
        self._proc: subprocess.Popen | None = None
        self._stderr_file = None
        self._frames_written = 0

    def _build_cmd(self) -> list[str]:
        cmd = [
            self.ffmpeg,
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            str(self.fps),
            "-i",
            "pipe:0",
        ]
        cmd += select_encoder(self.codec, self.width, hw=self.hw_encoder)
        if self.bitrate:
            cmd += ["-b:v", self.bitrate]
        else:
            cmd += ["-crf", str(self.crf)]
        # D-2 (#79): preset encode knobs; preserve +faststart default when no
        # preset is in play (faststart None).
        cmd += preset_encode_args(self.gop, self.force_idr, self.faststart)
        if self.faststart is None:
            cmd += ["-movflags", "+faststart"]
        cmd += ["-pix_fmt", "yuv420p", self.output_path]
        return cmd

    def open(self) -> "RawFrameFFmpegWriter":
        """Spawn the persistent ffmpeg process. Called by ``__enter__``."""
        cmd = self._build_cmd()
        log.info("ffmpeg cmd: %s", " ".join(cmd))
        # stderr → temp file (issue #49): a PIPE would deadlock once its 64 KB
        # buffer fills; DEVNULL hides fatal errors. The file is OS-drained at
        # no cost; we read only the tail on failure.
        self._stderr_file = tempfile.TemporaryFile(prefix="vr180-ffmpeg-stderr-")  # noqa: SIM115
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=self._stderr_file,
        )
        return self

    def __enter__(self) -> "RawFrameFFmpegWriter":
        return self.open()

    def write(self, frame: np.ndarray) -> None:
        """Write one SBS frame's raw RGB bytes to the ffmpeg pipe.

        Args:
            frame: ``(H, W, 3)`` uint8 ndarray — must match the ``width`` /
                ``height`` this writer was opened with.
        """
        assert self._proc is not None and self._proc.stdin is not None, "writer not open"
        try:
            self._proc.stdin.write(frame.astype(np.uint8).tobytes())
        except BrokenPipeError:
            raise RuntimeError(
                f"ffmpeg encoder died after {self._frames_written} frames — "
                "check encoder availability/limits for this resolution "
                f"({self.width}×{self.height}). ffmpeg stderr: {self._stderr_summary()}"
            ) from None
        except OSError as e:
            if e.errno in (22, 32):  # EINVAL (Windows broken pipe) / EPIPE
                raise RuntimeError(
                    f"ffmpeg encoder died after {self._frames_written} frames "
                    f"(pipe write failed, errno={e.errno}; exit code "
                    f"{self._proc.poll()}). ffmpeg stderr: {self._stderr_summary()}"
                ) from None
            raise
        self._frames_written += 1

    def close(self) -> None:
        """Finalise the ffmpeg process and raise on failure (no silent bad files)."""
        if self._proc is None:
            return
        try:
            if self._proc.stdin is not None:
                with contextlib.suppress(BrokenPipeError, OSError):
                    self._proc.stdin.close()
            self._proc.wait()
            if self._proc.returncode != 0:
                raise RuntimeError(
                    f"ffmpeg encoding failed (exit code {self._proc.returncode}) "
                    f"after {self._frames_written} frames. "
                    f"ffmpeg stderr: {self._stderr_summary()}"
                )
        finally:
            if self._stderr_file is not None:
                with contextlib.suppress(OSError):
                    self._stderr_file.close()
            self._proc = None
            self._stderr_file = None

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _stderr_summary(self) -> str:
        f = self._stderr_file
        if f is None:
            return "(stderr not captured)"
        try:
            f.flush()
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - STDERR_TAIL_BYTES))
            tail = f.read()
            text = tail.decode("utf-8", errors="replace").strip()
            return text or "(ffmpeg produced no stderr output)"
        except (OSError, ValueError):
            return "(stderr unreadable)"


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
    hw_encoder: bool | str | None = None,
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
