"""
Streaming Pipeline (PRD §7.2)
O(1) memory video processing — reads frames one-at-a-time via cv2.VideoCapture,
processes depth → stereo → equirect, and writes directly to an ffmpeg output pipe.
No frame buffers accumulate in RAM.
"""

import contextlib
import logging
import os
import subprocess
import tempfile
import time

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


# ---------------------------------------------------------------------------
# P-1 (#216): lightweight per-stage wall-clock timer.
#
# Records cumulative wall-clock seconds per logical pipeline stage
# (depth / stereo / equirect / encode / metadata) using
# ``time.perf_counter``.  Always on — the cost is a handful of
# perf_counter() calls per frame, negligible next to inference.
#
# The timer is deliberately fault-tolerant: if ``time.perf_counter``
# itself raises (broken clock, hostile test double), a single stage
# record is simply skipped.  It never re-raises, never swallows a
# stage's own exception, and never lets its own failure take down the
# pipeline.  ``start`` returns a sentinel that ``stop`` uses to guard
# against double-stops (e.g. when a stage raises between start and stop).
# ---------------------------------------------------------------------------

# Ordered list of logical stages the timer knows about.  ``encode`` is the
# residual stage (total wall-clock minus the measured depth/stereo/equirect/
# metadata), which is why it has no ``total`` bucket of its own and why the
# logged percentages always sum to 100%.
_STAGE_ORDER = ("depth", "stereo", "equirect", "metadata")


class _StageTimer:
    """Accumulate wall-clock seconds per pipeline stage via ``time.perf_counter``.

    Usage::

        timer = _StageTimer()
        timer.start("depth")
        ...do depth work, possibly raising...
        timer.stop("depth")      # guards against double-stop

        # The caller's outermost block measures total wall-clock so encode can
        # be derived as the residual:  total - sum(measured stages).
        timer.start_total()
        try:
            ...whole pipeline...
        finally:
            timer.stop_total()

        timings = timer.report()  # dict {stage: seconds, "_total": seconds}
        timer.log_table()         # logging.info aligned table
    """

    def __init__(self):
        self._total: dict[str, float] = {}
        self._active: dict[str, float] = {}
        self._started: set[str] = set()

    def start_total(self) -> None:
        try:
            self._total["ts"] = time.perf_counter()
        except Exception:
            # Timer failure must never crash the pipeline.
            self._total.clear()

    def stop_total(self) -> None:
        if "ts" not in self._total:
            return
        try:
            start = self._total.pop("ts")
            self._total["_total"] = time.perf_counter() - start
        except Exception:
            self._total.pop("ts", None)

    def start(self, stage: str) -> bool:
        """Mark the start of *stage*.

        Returns True on success so ``stop`` can guard against a
        no-op/redundant stop.  On ``perf_counter`` failure the record is
        silently skipped (returns False) and the stage continues.
        """
        try:
            if stage in self._started:
                return False
            self._active[stage] = time.perf_counter()
            self._started.add(stage)
            return True
        except Exception:
            return False

    def stop(self, stage: str) -> None:
        """Accumulate the elapsed time for *stage* since its ``start``.

        Idempotent: a stage that raised before ``stop`` (no active
        timestamp) is a no-op rather than a crash.  A second ``stop``
        for the same stage is also a no-op (double-stop guard for the
        per-frame fuse loop where a stage may skip frames).
        """
        if stage not in self._active or stage not in self._started:
            return
        ts = self._active.pop(stage)
        try:
            elapsed = time.perf_counter() - ts
            self._total[stage] = self._total.get(stage, 0.0) + elapsed
        except Exception:
            # perf_counter failed mid-stop — drop this frame's record
            # rather than crash the pipeline.
            pass
        finally:
            self._started.discard(stage)

    def report(self) -> dict[str, float]:
        """Return ``{stage: seconds}`` plus ``_total``, with ``encode``
        filled in as the residual so percentages sum to exactly 100%.

        Every known stage appears as a key (0.0 when the stage had no
        work to do — e.g. ``metadata`` in the streaming path, which does
        sv3d/st3d injection downstream rather than here).  This keeps the
        sidecar payload schema stable regardless of which branch ran.
        """
        out = dict(self._total)
        # Ensure every known stage is present (0.0 if it never ran).
        for stage in _STAGE_ORDER:
            out.setdefault(stage, 0.0)
        measured_sum = sum(out.get(s, 0.0) for s in _STAGE_ORDER)
        total = out.get("_total", 0.0)
        out["encode"] = max(0.0, total - measured_sum)
        # Encode's wall-clock includes the pipe writes PLUS the ffmpeg
        # process-lifetime overhead (open/wait/cleanup) that isn't
        # attributed to depth/stereo/equirect.  That is the desired
        # "where did the seconds go" attribution: encode owns the
        # pipeline's non-model-cost baseline.
        return out

    def log_table(self, logger: logging.Logger | None = None) -> None:
        """Emit an aligned stage-timing table via logging.info.

        The table is the card's core deliverable: lead reads it from a
        real render's log to decide which stage to optimise next.
        """
        out = self.report()
        total = out.get("_total", 0.0)
        lines = ["📊 stage_timings (wall-clock):"]
        header = f"{'stage':<10} {'seconds':>10} {'pct':>7}"
        lines.append(header)
        lines.append("-" * len(header))
        for stage in (*_STAGE_ORDER, "encode"):
            secs = out.get(stage, 0.0)
            pct = (100.0 * secs / total) if total > 0 else 0.0
            lines.append(f"{stage:<10} {secs:>10.3f} {pct:>6.1f}%")
        lines.append(f"{'total':<10} {total:>10.3f}")
        log_fn = logger.info if logger is not None else log.info
        for line in lines:
            log_fn(line)


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


def _load_video_frames(video_path: str) -> list[np.ndarray]:
    """Load all frames from a video file as RGB ndarrays (I-5, #120).

    Used by the streaming whole-clip-stereo path to read back the L/R output
    videos a StereoCrafter backend writes, so each frame can be fed into the
    per-frame equirect→encode fuse loop.  Mirrors the same-named helper in
    ``scripts/run_pipeline.py`` (kept here so the streaming module is
    self-contained for injected whole-clip stereo backends).
    """
    cap = cv2.VideoCapture(video_path)
    frames: list[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


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
        # K-21 (#224): when a caller owns an intermediate directory (e.g.
        # run_pipeline's --temp-dir / make_comparison's per-recipe work dir),
        # pass it here so the streaming path writes depth products into
        # ``<temp_dir>/depth/`` and stereo intermediates into
        # ``<temp_dir>/stereo/`` — the layout the comparison
        # default_depth_dir_resolver expects.  When None the streaming path
        # falls back to the historical tempfile.mkdtemp (self-cleaning)
        # behaviour, so existing callers / pre-K-21 behaviour is bit-exact.
        temp_dir: str | None = None,
        # K-22 / #243 (P0-2): --no-ffmpeg-v360 was previously swallowed by the
        # streaming path because EquirectangularMapper was constructed with
        # use_ffmpeg=True hard-coded.  Passing use_ffmpeg (default True) here
        # lets the OpenCV fallback path be selected when the operator disables
        # ffmpeg v360 — the default True keeps pre-#243 behaviour bit-exact.
        use_ffmpeg: bool = True,
        # K-22 / #243 (P0-2): --outpaint and its sub-params were previously
        # silently dropped on the streaming path (same anti-pattern as #120).
        # They are stored as attributes so downstream streaming outpaint logic
        # can honour them; defaults match the CLI argparse defaults so existing
        # callers that omit them are unaffected and behaviour is unchanged.
        outpaint: str = "none",
        outpaint_mask_threshold: int = 10,
        outpaint_mask_top_ratio: float = 0.25,
        outpaint_mask_bottom_ratio: float = 0.25,
        # I-5 (#120): injectable depth/stereo backends.  When None the defaults
        # (Depth-Anything V2 per-frame + StereoRenderer depth-shift) are used,
        # so pre-I-5 behaviour is bit-exact.  The CLI streaming branch injects
        # the --depth-model / --stereo-model backend here so the streaming path
        # no longer silently ignores those flags.  ``depth_backend_name`` /
        # ``stereo_backend_name`` are logged at stream startup for one-glance
        # acceptance verification.
        depth_estimator=None,
        stereo_renderer=None,
        depth_backend_name: str | None = None,
        stereo_backend_name: str | None = None,
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
        # K-21 (#224): caller-owned work directory. None ⇒ streaming path uses
        # (and owns cleanup of) a fresh tempfile.mkdtemp.
        self.temp_dir = temp_dir
        # K-22 / #243 (P0-2): ffmpeg v360 toggle — honoured when constructing
        # the equirect mapper below instead of being hard-coded to True.
        self.use_ffmpeg = use_ffmpeg
        # K-22 / #243 (P0-2): outpaint fill settings forwarded from the CLI
        # (--outpaint / --outpaint-mask-*).  ``outpaint == "none"`` is the
        # default and is a no-op (matching run_outpaint_stage's contract), so
        # storing them without further work changes nothing about the existing
        # default streaming run.
        self.outpaint = outpaint
        self.outpaint_mask_threshold = outpaint_mask_threshold
        self.outpaint_mask_top_ratio = outpaint_mask_top_ratio
        self.outpaint_mask_bottom_ratio = outpaint_mask_bottom_ratio
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

        # Initialise pipeline stages.  I-5 (#120): the depth estimator and
        # stereo renderer are injectable so the streaming path can honour
        # ``--depth-model depthcrafter`` / ``--stereo-model stereocrafter``
        # instead of always using the per-frame Depth-Anything + StereoRenderer
        # defaults.  Injected whole-clip backends (DepthCrafter's
        # ``estimate_video``, StereoCrafter's ``render_video``) are detected in
        # ``process_stream`` and run once for the whole clip; their per-frame
        # outputs then feed the same O(1) equirect→encode fuse loop.
        #
        # The whole-clip detection keys off *explicit injection*: only a backend
        # the caller passed in is ever treated as whole-clip.  The built-in
        # DepthEstimator / StereoRenderer defaults are always per-frame.  This
        # keeps detection robust against attribute-happy test doubles (a bare
        # MagicMock auto-creates ``estimate_video``/``render_video`` and would
        # otherwise be misdetected as whole-clip).
        self._depth_injected = depth_estimator is not None
        self._stereo_injected = stereo_renderer is not None
        self.depth_estimator = depth_estimator or DepthEstimator(
            model_size=model_size,
            device=self.device,
            calibrate=True,
        )
        self.stereo_renderer = stereo_renderer or StereoRenderer(
            ipd=ipd,
            max_disparity=max_disparity,
        )
        self.depth_backend_name = depth_backend_name or "depth-anything"
        self.stereo_backend_name = stereo_backend_name or "default"
        self.eq_mapper = EquirectangularMapper(
            output_width=output_width,
            output_height=output_height,
            src_hfov=src_hfov,
            # K-22 / #243 (P0-2): was hard-coded to True, which silently
            # ignored --no-ffmpeg-v360 on the streaming path.
            use_ffmpeg=use_ffmpeg,
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

    def _is_wholeclip_depth(self, backend) -> bool:
        """True if *backend* is a whole-clip depth estimator (I-5, #120).

        Whole-clip backends (e.g. :class:`DepthCrafterEstimator`) expose
        ``estimate_video(input_path, output_dir)`` instead of the per-frame
        ``estimate(rgb)`` contract — DepthCrafter is a temporal model that needs
        the whole clip for flicker-free depth.  Such a backend cannot be called
        inside the per-frame fuse loop; ``process_stream`` runs it once for the
        whole clip up front and then feeds the precomputed depths per frame.

        Detection is gated on explicit injection (``self._depth_injected``):
        the built-in DepthEstimator default is always per-frame, and a bare
        test double that merely *auto-creates* ``estimate_video`` is not
        mistaken for a real whole-clip backend.
        """
        return self._depth_injected and callable(getattr(backend, "estimate_video", None))

    def _is_wholeclip_stereo(self, backend) -> bool:
        """True if *backend* is a whole-clip stereo renderer (I-5, #120).

        Whole-clip stereo backends (e.g. :class:`StereoCrafterRenderer`) expose
        ``render_video(input_path, depth_dir, output_left, output_right)`` —
        they run their own diffusion-based inference over the whole clip and
        emit L/R videos.  ``process_stream`` runs them once up front and then
        reads the L/R frames back per-frame into the equirect→encode fuse.

        Detection is gated on explicit injection (``self._stereo_injected``),
        matching :meth:`_is_wholeclip_depth`.
        """
        return self._stereo_injected and callable(getattr(backend, "render_video", None))

    def _precompute_depths(
        self,
        input_path: str,
        total: int,
        out_dir: str,
    ) -> list[np.ndarray]:
        """Run the injected whole-clip depth backend once (I-5, #120).

        Returns the per-frame depth maps (length ``total`` or fewer — the
        backend may produce fewer frames than ``total`` if the source is
        truncated).  The fuse loop bounds itself to whatever this returns.
        """
        log.info(
            "🎬 [Depth] whole-clip backend (%s): running estimate_video on %s",
            self.depth_backend_name,
            input_path,
        )
        depths = self.depth_estimator.estimate_video(
            input_path=input_path,
            output_dir=out_dir,
        )
        log.info(
            "🎬 [Depth] %s produced %d depth map(s)",
            self.depth_backend_name,
            len(depths),
        )
        return list(depths)

    def _precompute_stereo(
        self,
        input_path: str,
        depth_dir: str,
        out_left: str,
        out_right: str,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Run the injected whole-clip stereo backend once (I-5, #120; I-7, #137).

        Returns ``(left_frames, right_frames)`` read back from the L/R output
        videos the backend wrote (the paths ``render_video`` *returns*, matching
        the batch ``_run_stereocrafter_stage``).  Since issue #140, StereoCrafter
        consumes the pipeline's own per-frame depth maps via *depth_dir* (the
        in-repo forward-splat replaces the removed upstream Stage 1); the
        caller is responsible for pointing *depth_dir* at the depth stage's
        real output directory (I-7.2, #143) — it must be non-empty before the
        call.
        """
        log.info(
            "🎬 [Stereo] whole-clip backend (%s): running render_video on %s",
            self.stereo_backend_name,
            input_path,
        )
        # Use the paths the backend actually wrote (its return value), not the
        # assumed ones — the batch ``_run_stereocrafter_stage`` does the same.
        # A backend that resolves outputs elsewhere would otherwise have its
        # frames read back from the wrong location.
        result_left, result_right = self.stereo_renderer.render_video(
            input_path=input_path,
            depth_dir=depth_dir,
            output_left=out_left,
            output_right=out_right,
        )
        left_frames = _load_video_frames(result_left)
        right_frames = _load_video_frames(result_right)
        log.info(
            "🎬 [Stereo] %s produced %d L/R frame pair(s)",
            self.stereo_backend_name,
            len(left_frames),
        )
        return left_frames, right_frames

    def _emit_perframe_depths(
        self,
        cap: cv2.VideoCapture,
        total: int,
        depth_dir: str,
    ) -> None:
        """Auto-emit per-frame depth maps into *depth_dir* (I-7.2, #143).

        Used when a whole-clip stereo backend (StereoCrafter) is in play but no
        whole-clip depth backend (DepthCrafter) was injected: StereoCrafter
        consumes the pipeline's own per-frame depth maps (issue #140), so the
        per-frame estimator is run over the clip once here and each map is
        checkpointed as ``depth_<idx:06d>.npy`` — the exact layout the stereo
        backend's depth loader globs for.  This replaces the pre-fix behaviour
        of handing StereoCrafter a guaranteed-empty dir and crashing deep
        inside the splat assembly.

        The frames are read from the already-open *cap*; the whole-clip stereo
        branch never reads from ``cap`` again afterwards (its fuse loop
        consumes the precomputed L/R frames instead), so consuming it here is
        safe.

        For temporally-stable depth the recommendation remains ``--depth-model
        depthcrafter`` (whole-clip); this fallback is per-frame Depth-Anything
        quality — functional, but not flicker-free.
        """
        log.warning(
            "⚠️  --stereo-model stereocrafter without a whole-clip depth backend: "
            "auto-emitting per-frame depth (%s) into %s for the stereo stage. "
            "For flicker-free depth, pass --depth-model depthcrafter.",
            self.depth_backend_name,
            depth_dir,
        )
        emitted = 0
        for idx in range(total):
            ret, bgr = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            depth = self.depth_estimator.estimate(rgb)
            np.save(os.path.join(depth_dir, f"depth_{idx:06d}.npy"), np.asarray(depth))
            del depth, rgb, bgr
            emitted = idx + 1
        log.info(
            "🎬 [Depth] %s emitted %d per-frame depth map(s) → %s",
            self.depth_backend_name,
            emitted,
            depth_dir,
        )

    def _write_sbs_frame(
        self,
        proc: subprocess.Popen,
        sbs: np.ndarray,
        frame_idx: int,
        out_w: int,
        out_h: int,
    ) -> None:
        """Write one SBS frame's raw RGB to the ffmpeg pipe (I-5, #120).

        Factored out of :meth:`process_stream` so both the per-frame fuse path
        and the whole-clip-precompute path share the identical write + pipe-
        death diagnostics (issue #49: BrokenPipeError / Windows EINVAL/EPIPE).
        """
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

    # K-21 (#224): work-directory helpers.  When ``self.temp_dir`` is set the
    # streaming path writes products into caller-owned subdirs (the depth dir
    # is the flat ``<temp_dir>/depth/`` that
    # make_comparison.default_depth_dir_resolver globs) instead of an
    # ephemeral ``tempfile.mkdtemp``.  The lifetime of a caller-owned dir is
    # the caller's (run_pipeline / make_comparison); the ephemeral mkdtemp
    # dir is left in system temp (the pre-K-21 behaviour the rest of the
    # suite relies on for "depth dir survives the run"), cleaned by the OS.
    def _depth_work_dir(self) -> str:
        """Return the depth-product directory (created if absent)."""
        if self.temp_dir:
            depth_dir = os.path.join(self.temp_dir, "depth")
        else:
            depth_dir = os.path.join(tempfile.mkdtemp(prefix="vr180-streaming-depth_"), "depth")
        os.makedirs(depth_dir, exist_ok=True)
        return depth_dir

    def _stereo_work_dir(self) -> str:
        """Return the stereo-intermediate directory (created if absent)."""
        if self.temp_dir:
            stereo_dir = os.path.join(self.temp_dir, "stereo")
            os.makedirs(stereo_dir, exist_ok=True)
            return stereo_dir
        return tempfile.mkdtemp(prefix="vr180-streaming-stereo_")

    def process_stream(
        self,
        input_path: str,
        output_path: str,
        max_frames: int | None = None,
    ) -> str:
        """Process video frame-by-frame, writing directly to ffmpeg pipe.

        I-5 (#120): honours the injected depth/stereo backends.  When the
        default per-frame Depth-Anything + StereoRenderer are in use (no
        ``--depth-model`` / ``--stereo-model`` override) the per-frame fuse
        loop is bit-exact with pre-I-5.  When a whole-clip backend
        (DepthCrafter ``estimate_video`` / StereoCrafter ``render_video``) is
        injected, it is run **once for the whole clip** up front (it cannot be
        called per-frame — it needs the whole clip for temporal consistency),
        and the precomputed depth / L-R maps then feed the same O(1)
        equirect→encode fuse loop.

        Args:
            input_path: Path to input 2D video.
            output_path: Path for output VR180 video.
            max_frames: Optional cap on number of frames (for testing).

        Returns:
            Path to the written output video.

        Raises:
            RuntimeError: If input video cannot be opened.
        """
        # I-5 (#120): log the effective depth/stereo backend at stream startup
        # so acceptance can confirm DepthCrafter/StereoCrafter is actually in
        # play (not silently swapped for Depth-Anything).
        log.info(
            "🎚️  Streaming backends: depth=%s, stereo=%s",
            self.depth_backend_name,
            self.stereo_backend_name,
        )

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

        # I-5 (#120): detect whole-clip backends.  They cannot run in the per-
        # frame fuse loop, so precompute their outputs up front.
        #
        # I-7.2 (#143): StereoCrafter (#140) consumes the pipeline's own
        # per-frame depth maps via ``depth_dir`` — the in-repo forward-splat
        # replaced the removed upstream Stage 1.  The streaming path therefore
        # has to hand the stereo backend the *depth stage's real output dir*
        # (populated, alive through the stereo call), never a fresh empty one.
        stereo_wholeclip = self._is_wholeclip_stereo(self.stereo_renderer)
        depth_wholeclip = self._is_wholeclip_depth(self.depth_estimator)
        precomp_left: list[np.ndarray] | None = None
        precomp_right: list[np.ndarray] | None = None
        precomp_depths: list[np.ndarray] | None = None

        if stereo_wholeclip:
            work_dir = self._stereo_work_dir()
            left_path = os.path.join(work_dir, "_stereo_left.mp4")
            right_path = os.path.join(work_dir, "_stereo_right.mp4")

            if depth_wholeclip:
                # Whole-clip depth backend (DepthCrafter) injected: run it once
                # into its own work dir and hand THAT dir to the stereo backend.
                # The dir must outlive the stereo call, so it lives for the
                # whole process_stream scope (never deleted here).
                depth_dir = self._depth_work_dir()
                self._precompute_depths(input_path, total, depth_dir)
            else:
                # StereoCrafter needs per-frame depth maps to splat with, but no
                # whole-clip depth backend is in play.  Emit them with the
                # per-frame estimator so the stereo stage has something real to
                # consume (rather than dying on an empty dir).
                depth_dir = self._depth_work_dir()
                self._emit_perframe_depths(cap, total, depth_dir)

            precomp_left, precomp_right = self._precompute_stereo(input_path, depth_dir, left_path, right_path)
            # StereoCrafter may produce fewer frames than the source count;
            # bound the fuse loop to what was actually produced.
            if precomp_left:
                total = min(total, len(precomp_left))
        elif depth_wholeclip:
            depth_dir = self._depth_work_dir()
            precomp_depths = self._precompute_depths(input_path, total, depth_dir)
            if precomp_depths:
                total = min(total, len(precomp_depths))

        # P-1 (#216): lightweight per-stage wall-clock timer.  Always on; cost
        # = a handful of time.perf_counter() calls.  Total wall-clock wraps the
        # whole ffmpeg-driven run so ``encode`` can be derived as the residual
        # (total − sum(measured stages)), guaranteeing percentages sum to 100%.
        timer = _StageTimer()
        timer.start_total()

        proc = self._open_ffmpeg_writer(output_path, out_w, out_h)

        try:
            frame_idx = 0
            try:
                # I-5 (#120): branch on whole-clip precompute.  Each branch
                # produces the SBS frame for ``_write_sbs_frame`` identically;
                # only the source of (depth, L/R) differs.
                if precomp_left is not None:
                    # Whole-clip stereo (StereoCrafter): L/R precomputed, skip
                    # depth + stereo-render; just map → encode per frame.
                    for left, right in zip(precomp_left, precomp_right, strict=False):
                        timer.start("equirect")
                        sbs = self.eq_mapper.map_stereo_pair(left, right)
                        timer.stop("equirect")
                        self._write_sbs_frame(proc, sbs, frame_idx, out_w, out_h)
                        del sbs
                        frame_idx += 1
                        if frame_idx % 10 == 0:
                            log.info(f"  [{frame_idx}/{total}] frames processed")
                        if max_frames and frame_idx >= max_frames:
                            break
                else:
                    while frame_idx < total:
                        ret, bgr = cap.read()
                        if not ret:
                            break

                        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

                        # --- Stage 1: Depth estimation ---
                        # I-5 (#216): the per-frame estimate call is the depth
                        # stage.  Whole-clip precomputed depths are indexed
                        # inside the timer so the (near-zero) lookup cost is
                        # attributed to depth, not silently lost.
                        #
                        # I-5 (#120): whole-clip depth (DepthCrafter) was
                        # precomputed; index into it.  Otherwise call the
                        # per-frame estimator (Depth-Anything) inline.
                        timer.start("depth")
                        if precomp_depths is not None:
                            depth = precomp_depths[frame_idx]
                        else:
                            depth = self.depth_estimator.estimate(rgb)
                        timer.stop("depth")

                        # --- Stage 2: Stereo rendering ---
                        timer.start("stereo")
                        left, right = self.stereo_renderer.render(rgb, depth)
                        timer.stop("stereo")

                        # --- Stage 3: Equirectangular mapping ---
                        timer.start("equirect")
                        sbs = self.eq_mapper.map_stereo_pair(left, right)
                        timer.stop("equirect")

                        self._write_sbs_frame(proc, sbs, frame_idx, out_w, out_h)

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
            # Stop the total wrapper in the finally so the timer always sees a
            # _total even when a stage raises.  Report/log in the finally so the
            # aligned table is always emitted (the card's core deliverable) and
            # stage_timings is always attached — even on failure — before the
            # original exception re-raises.  Timer/report failures are isolated
            # so they never replace a real stage error.
            timer.stop_total()
            try:
                self.stage_timings = timer.report()
                timer.log_table()
            except Exception:
                pass

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
    # I-5 (#120): injectable backends — pass through to StreamingPipeline so the
    # --depth-model / --stereo-model streaming path can be exercised with fakes.
    depth_estimator=None,
    stereo_renderer=None,
    depth_backend_name: str | None = None,
    stereo_backend_name: str | None = None,
    # K-22 / #243 (P0-2): outpaint / ffmpeg-v360 knobs passed through to
    # StreamingPipeline (default True/"none"/... so pre-#243 callers unchanged).
    use_ffmpeg: bool = True,
    outpaint: str = "none",
    outpaint_mask_threshold: int = 10,
    outpaint_mask_top_ratio: float = 0.25,
    outpaint_mask_bottom_ratio: float = 0.25,
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
        depth_estimator: I-5 (#120) injected depth backend (None = Depth-Anything default).
        stereo_renderer: I-5 (#120) injected stereo backend (None = StereoRenderer default).
        depth_backend_name: I-5 (#120) label logged at startup for acceptance.
        stereo_backend_name: I-5 (#120) label logged at startup for acceptance.
        use_ffmpeg: K-22 / #243 (P0-2) ffmpeg v360 toggle (default True).
        outpaint: K-22 / #243 (P0-2) outpaint mode (default "none").
        outpaint_mask_threshold: K-22 / #243 (P0-2) outpaint mask brightness threshold.
        outpaint_mask_top_ratio: K-22 / #243 (P0-2) top black-boundary scan ratio.
        outpaint_mask_bottom_ratio: K-22 / #243 (P0-2) bottom black-boundary scan ratio.

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
        depth_estimator=depth_estimator,
        stereo_renderer=stereo_renderer,
        depth_backend_name=depth_backend_name,
        stereo_backend_name=stereo_backend_name,
        use_ffmpeg=use_ffmpeg,
        outpaint=outpaint,
        outpaint_mask_threshold=outpaint_mask_threshold,
        outpaint_mask_top_ratio=outpaint_mask_top_ratio,
        outpaint_mask_bottom_ratio=outpaint_mask_bottom_ratio,
    )
    return pipeline.process_stream(input_path, output_path, max_frames=max_frames)
