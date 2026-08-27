"""Local Stable Video Diffusion (SVD) image-to-video provider.

PRD §3.1 路 B: the no-API-key local generation path.  Runs Stability AI's
Stable Video Diffusion (``diffusers.StableVideoDiffusionPipeline``) directly
on the host GPU — no network call, no API key.

Device / VRAM trade-offs baked in per the owner's hardware reality:

- **RTX 4070 SUPER 12 GB (CUDA)**: SVD only fits at low resolution with CPU
  offload.  The default params here target that envelope.
- **Apple M2 Max 96 GB (MPS)**: comfortable VRAM → full resolution.

``diffusers`` is an **optional** dependency (not in ``requirements.txt``);
the real backend imports it lazily so the module is importable on CI, which
has neither ``diffusers`` nor the model weights.  When the dependency or
model is missing the provider raises an actionable ``RuntimeError`` naming
the install command and the VRAM requirement.

The backend is **pluggable**: callers (and tests) can inject a fake backend
implementing the :class:`SVDBackend` protocol to exercise the full path
without any model.
"""

from __future__ import annotations

import logging
import os
import re
import uuid
from typing import Any, Protocol, runtime_checkable

from integrations.base import GenerationResult, VideoGenProvider
from pipeline.device_utils import detect_best_device
from pipeline.image_prep import validate_image_for_i2v

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Device / VRAM-aware default parameters
# ---------------------------------------------------------------------------

# 12 GB CUDA cards: SVD fits only at low resolution with CPU offload.
_CUDA_LOW_VRAM_WIDTH = 576
_CUDA_LOW_VRAM_HEIGHT = 320
_CUDA_LOW_VRAM_VRAM_GB = 12.0
# MPS / high-VRAM hosts: native full resolution.
_MPS_FULL_WIDTH = 1024
_MPS_FULL_HEIGHT = 576

# Default motion / decode knobs.  SVD produces ~25 frames at most; we let the
# backend decide the exact count and just report duration.
_DEFAULT_FPS = 7  # SVD native rate
_DEFAULT_MOTION_AMPLITUDE = 6.0  # decoder motion bucket (std-dev of noise)

# Hugging Face model id for SVD-XT (25 frames).  Override via kwargs/ctor.
_DEFAULT_MODEL_ID = "stabilityai/stable-video-diffusion-img2vid-xt"


# ---------------------------------------------------------------------------
# Backend protocol + implementations
# ---------------------------------------------------------------------------


@runtime_checkable
class SVDBackend(Protocol):
    """Pluggable SVD inference backend.

    Implementations wrap a real ``StableVideoDiffusionPipeline`` or a test
    mock.  The contract is intentionally narrow: load a model (or fake one)
    onto a device, then run image-to-video returning a list of frame paths
    that encode into the final ``.mp4``.

    Implementations need not subclass — duck typing via this Protocol is the
    intended injection seam (see :meth:`LocalSVDProvider.__init__`).
    """

    def load(self, model_id: str, device: str, enable_cpu_offload: bool) -> None:
        """Load (or pretend to load) the model onto *device*.

        Raises ``RuntimeError`` with an actionable message if the backing
        library or model weights are unavailable.
        """
        ...

    def generate(
        self,
        image_path: str,
        width: int,
        height: int,
        fps: int,
        motion_amplitude: float,
        num_frames: int,
        output_dir: str,
    ) -> list[str]:
        """Run image-to-video, returning ordered frame image paths.

        The frames are written into *output_dir*; the caller encodes them
        into an ``.mp4``.
        """
        ...


class _DiffusersSVDBackend:
    """Real backend wrapping ``diffusers.StableVideoDiffusionPipeline``.

    ``diffusers`` (and ``torch``) are imported lazily inside :meth:`load` so
    that simply importing this module does not require them — CI has neither.
    """

    def __init__(self) -> None:
        self._pipe: Any = None
        self._device: str = "cpu"

    def load(self, model_id: str, device: str, enable_cpu_offload: bool) -> None:
        try:
            import torch  # noqa: F401  (paired with diffusers; surfaced below)
            from diffusers import StableVideoDiffusionPipeline  # type: ignore[import-not-found]
        except ImportError as exc:
            raise self._missing_dependency_error(exc) from None

        try:
            self._pipe = StableVideoDiffusionPipeline.from_pretrained(model_id, torch_dtype=self._dtype(device))
        except Exception as exc:
            # Missing weights / no network / gated repo → actionable message.
            raise self._missing_model_error(model_id, exc) from None

        self._device = device
        if enable_cpu_offload:
            # `enable_model_cpu_offload` moves submodules to GPU on demand,
            # the only way SVD fits on a 12 GB card.
            self._pipe.enable_model_cpu_offload()
        else:
            self._pipe.to(device)
        log.info("SVD backend loaded (model=%s, device=%s, offload=%s)", model_id, device, enable_cpu_offload)

    def generate(
        self,
        image_path: str,
        width: int,
        height: int,
        fps: int,
        motion_amplitude: float,
        num_frames: int,
        output_dir: str,
    ) -> list[str]:
        if self._pipe is None:
            raise RuntimeError("SVD backend used before load(); call load() first.")
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        # diffusers expects height/width on the image; the pipeline resizes.
        image = image.resize((width, height))
        generator = None  # deterministic seeding left to a future task
        result = self._pipe(
            image=image,
            motion_bucket_id=motion_amplitude,
            num_frames=num_frames,
            fps=fps,
            generator=generator,
        )
        frames = result.frames[0] if hasattr(result, "frames") else result[0]
        return _write_frames(frames, output_dir)

    @staticmethod
    def _dtype(device: str) -> Any:
        import torch

        if device == "cuda":
            return torch.float16
        # MPS / CPU — fp16 path on MPS is uneven; keep fp32 for safety.
        return torch.float32

    @staticmethod
    def _missing_dependency_error(exc: ImportError) -> RuntimeError:
        return RuntimeError(
            "Local SVD backend requires the optional 'diffusers' and 'torch' "
            "packages. Install them with:\n"
            "    pip install diffusers torch\n"
            f"(original import error: {exc})"
        )

    @staticmethod
    def _missing_model_error(model_id: str, exc: Exception) -> RuntimeError:
        return RuntimeError(
            f"Failed to load SVD model '{model_id}'. The weights must be "
            "downloaded from Hugging Face (first run downloads ~5 GB; "
            "requires ~12 GB VRAM for low-res on CUDA, more for full-res).\n"
            "    pip install diffusers torch\n"
            f"(original error: {exc})"
        )


class MockSVDBackend:
    """In-process fake backend for tests / dry runs.

    Generates ``num_frames`` tiny solid-color PNG frames (no model, no
    network) so the full image→frames→encode path is exercisable on CI.
    """

    def __init__(self) -> None:
        self._loaded = False

    def load(self, model_id: str, device: str, enable_cpu_offload: bool) -> None:
        self._loaded = True
        log.info("MockSVD backend: pretend-load (model=%s, device=%s)", model_id, device)

    def generate(
        self,
        image_path: str,
        width: int,
        height: int,
        fps: int,
        motion_amplitude: float,
        num_frames: int,
        output_dir: str,
    ) -> list[str]:
        if not self._loaded:
            raise RuntimeError("MockSVD backend used before load(); call load() first.")
        try:
            import cv2
            import numpy as np
        except ImportError as exc:  # pragma: no cover - cv2 is in requirements
            raise RuntimeError(f"MockSVD backend needs opencv-python/numpy: {exc}") from None

        os.makedirs(output_dir, exist_ok=True)
        # Seed the frame color from the source image so the mock is at least
        # visually derived from the input (a single average color).
        src = cv2.imread(image_path)
        color = (0, 0, 0) if src is None else tuple(int(c) for c in src.mean(axis=(0, 1)).tolist())
        frames: list[str] = []
        for i in range(num_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:] = color
            path = os.path.join(output_dir, f"frame_{i:05d}.png")
            cv2.imwrite(path, frame)
            frames.append(path)
        log.info("MockSVD backend: wrote %d frames (%dx%d) to %s", num_frames, width, height, output_dir)
        return frames


# ---------------------------------------------------------------------------
# Backend helpers (module-level so tests can monkeypatch)
# ---------------------------------------------------------------------------


def _write_frames(frames: Any, output_dir: str) -> list[str]:
    """Persist an iterable of PIL Images (or arrays) as PNGs in *output_dir*."""
    os.makedirs(output_dir, exist_ok=True)
    paths: list[str] = []
    for i, frame in enumerate(frames):
        path = os.path.join(output_dir, f"frame_{i:05d}.png")
        # PIL Image has .save(); numpy arrays go through cv2.
        save = getattr(frame, "save", None)
        if callable(save):
            save(path)
        else:
            import cv2

            cv2.imwrite(path, frame)
        paths.append(path)
    return paths


def _encode_frames_to_mp4(frames: list[str], out_path: str, fps: int) -> str:
    """Encode ordered frame image paths into an ``.mp4`` via ffmpeg.

    Uses the ``image2`` demuxer with a ``%0Nd`` sequence pattern and an
    explicit input ``-framerate``.  The concat demuxer was tried first but
    does not assign per-frame durations to a bare PNG sequence, so ffmpeg
    emits zero output; ``image2`` with ``-framerate`` is the reliable path
    on CI (list-form subprocess, no ``shell=True``).
    """
    if not frames:
        raise RuntimeError("No frames to encode.")
    ffmpeg = os.environ.get("FFMPEG_BINARY", "ffmpeg")
    pattern = _frame_sequence_pattern(frames)
    cmd = [
        ffmpeg,
        "-y",
        "-nostdin",
        "-framerate",
        str(fps),
        "-i",
        pattern,
        "-vf",
        "format=yuv420p",
        "-movflags",
        "+faststart",
        out_path,
    ]
    import subprocess

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180, check=False)
    except FileNotFoundError as exc:
        raise RuntimeError(f"ffmpeg not found on PATH (set FFMPEG_BINARY): {exc}") from exc
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg frame encode failed (exit {result.returncode}):\nstderr:\n{result.stderr[-2000:]}")
    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        raise RuntimeError(f"ffmpeg produced no output at {out_path}")
    return out_path


def _frame_sequence_pattern(frames: list[str]) -> str:
    """Build an ffmpeg ``image2`` ``%0Nd`` pattern from ordered frame paths.

    The backends write frames as ``frame_00000.png``, ``frame_00001.png``,
    ... so the numeric index has a fixed width.  We infer that width from the
    first path and emit a glob-style pattern the ``image2`` demuxer accepts.

    Raises
    ------
    ValueError
        If the frame names do not share a single contiguous numeric index.
    """
    # Parse the numeric run in the basename of the first frame.
    name = os.path.basename(frames[0])
    match = re.search(r"\d+", name)
    if match is None:
        raise ValueError(f"Frame path {frames[0]!r} has no numeric index for an image2 sequence pattern.")
    width = len(match.group(0))
    # Replace the numeric run with the ffmpeg %0Nd placeholder.  Both backends
    # write frames starting at index 0, so the image2 demuxer finds them
    # without an explicit -start_number.
    head = name[: match.start()]
    tail = name[match.end() :]
    pattern_name = f"{head}%0{width}d{tail}"
    pattern = os.path.join(os.path.dirname(frames[0]), pattern_name)
    # ffmpeg image2 wants forward slashes even on Windows.
    return pattern.replace(os.sep, "/")


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class LocalSVDProvider(VideoGenProvider):
    """Local Stable Video Diffusion image-to-video provider.

    No API key is required (the mock key ``"local"`` is returned).  The real
    backend is ``diffusers.StableVideoDiffusionPipeline`` imported lazily;
    tests inject a fake backend via the ``backend`` constructor arg.

    Parameters
    ----------
    backend:
        Optional :class:`SVDBackend` instance.  ``None`` → real
        ``_DiffusersSVDBackend`` (imported lazily on first use).
    device:
        Compute device override (``"cuda"`` / ``"mps"`` / ``"cpu"``).
        ``None`` → :func:`pipeline.device_utils.detect_best_device`.
    model_id:
        Hugging Face model id.  Defaults to SVD-XT.
    """

    def __init__(
        self,
        api_key: str | None = None,
        backend: SVDBackend | None = None,
        device: str | None = None,
        model_id: str | None = None,
    ) -> None:
        # Skip the env-var key machinery — local generation needs no key.
        self._api_key = api_key or self._load_api_key()
        self._backend = backend
        self._device = device or detect_best_device()
        self._model_id = model_id or os.environ.get("SVD_MODEL_ID", _DEFAULT_MODEL_ID)

    # ------------------------------------------------------------------
    # VideoGenProvider contract
    # ------------------------------------------------------------------

    def _load_api_key(self) -> str:
        return "local"

    def generate(
        self,
        prompt: str,
        duration: int = 5,
        aspect_ratio: str = "16:9",
        fps: int = 24,
        **kwargs: Any,
    ) -> GenerationResult:
        """Text-to-video is not supported by SVD (image-to-video only).

        Raises ``NotImplementedError`` with a pointer to
        :meth:`generate_from_image`.
        """
        del prompt, duration, aspect_ratio, fps, kwargs  # unused
        raise NotImplementedError("local-svd only supports image-to-video; use generate_from_image().")

    def generate_from_image(
        self,
        image_path: str,
        prompt: str = "",
        duration: int = 5,
        aspect_ratio: str = "16:9",
        **kwargs: Any,
    ) -> GenerationResult:
        """Generate a video from a starting image via local SVD.

        The image is validated for I2V constraints before the backend is
        touched (fail fast, clear message).  Resolution / offload params are
        chosen from the detected device (12 GB CUDA → low-res + offload; MPS
        → full-res), overridable via kwargs.
        """
        del prompt  # SVD img2vid does not consume a text prompt.
        validate_image_for_i2v(image_path)

        params = self._select_params(**kwargs)
        num_frames = int(kwargs.get("num_frames", _frames_for_duration(duration, params["fps"])))

        backend = self._get_backend()
        backend.load(
            model_id=self._model_id,
            device=self._device,
            enable_cpu_offload=params["enable_cpu_offload"],
        )
        out_dir = self._out_dir()
        frames = backend.generate(
            image_path=image_path,
            width=params["width"],
            height=params["height"],
            fps=params["fps"],
            motion_amplitude=params["motion_amplitude"],
            num_frames=num_frames,
            output_dir=out_dir,
        )
        out_path = os.path.join(out_dir, f"svd_{uuid.uuid4().hex[:8]}.mp4")
        _encode_frames_to_mp4(frames, out_path, params["fps"])

        return GenerationResult(
            video_url=out_path,
            provider=self.provider_name,
            job_id=f"local-svd-{uuid.uuid4().hex[:8]}",
            metadata={
                "duration": duration,
                "aspect_ratio": aspect_ratio,
                "fps": params["fps"],
                "width": params["width"],
                "height": params["height"],
                "num_frames": num_frames,
                "device": self._device,
                "model_id": self._model_id,
                "cpu_offload": params["enable_cpu_offload"],
                "image_path": image_path,
                "backend": type(backend).__name__,
            },
        )

    # ------------------------------------------------------------------
    # Parameter selection (device/VRAM-aware, testable)
    # ------------------------------------------------------------------

    def _select_params(self, **kwargs: Any) -> dict[str, Any]:
        """Pick resolution / offload / motion params for the current device.

        Honours explicit kwargs above all else, then falls back to the
        device-aware defaults.  Pure function of ``(device, vram, kwargs)``
        so it is unit-testable with an injected fake device.
        """
        device = self._device
        vram_gb = kwargs.get("vram_gb")

        if device == "cuda":
            if vram_gb is None:
                vram_gb = _CUDA_LOW_VRAM_VRAM_GB
            low_vram = float(vram_gb) <= _CUDA_LOW_VRAM_VRAM_GB
            width = kwargs.get("width", _CUDA_LOW_VRAM_WIDTH if low_vram else _MPS_FULL_WIDTH)
            height = kwargs.get("height", _CUDA_LOW_VRAM_HEIGHT if low_vram else _MPS_FULL_HEIGHT)
            enable_cpu_offload = kwargs.get("enable_cpu_offload", low_vram)
        elif device == "mps":
            # MPS hosts (M2 Max 96 GB) have ample unified memory → full res.
            width = kwargs.get("width", _MPS_FULL_WIDTH)
            height = kwargs.get("height", _MPS_FULL_HEIGHT)
            enable_cpu_offload = kwargs.get("enable_cpu_offload", False)
        else:
            # CPU fallback — tiny res, offload moot but kept false.
            width = kwargs.get("width", _CUDA_LOW_VRAM_WIDTH)
            height = kwargs.get("height", _CUDA_LOW_VRAM_HEIGHT)
            enable_cpu_offload = kwargs.get("enable_cpu_offload", False)

        return {
            "width": int(width),
            "height": int(height),
            "fps": int(kwargs.get("fps", _DEFAULT_FPS)),
            "motion_amplitude": float(kwargs.get("motion_amplitude", _DEFAULT_MOTION_AMPLITUDE)),
            "enable_cpu_offload": bool(enable_cpu_offload),
        }

    # ------------------------------------------------------------------
    # Backend resolution
    # ------------------------------------------------------------------

    def _get_backend(self) -> SVDBackend:
        """Return the injected backend, or lazily build the real one."""
        if self._backend is not None:
            return self._backend
        # Lazy construction keeps diffusers out of the import graph entirely
        # until a real generation is attempted.
        self._backend = _DiffusersSVDBackend()
        return self._backend

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _out_dir() -> str:
        """Unique per-run output dir under the configured video root.

        ``SVD_PROVIDER_OUTPUT_DIR`` overrides the root (used by tests to keep
        artifacts inside a ``tmp_path``).
        """
        root = os.environ.get("SVD_PROVIDER_OUTPUT_DIR") or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "video"
        )
        out_dir = os.path.join(root, f"svd_{uuid.uuid4().hex[:8]}")
        os.makedirs(out_dir, exist_ok=True)
        return out_dir


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _frames_for_duration(duration: int, fps: int) -> int:
    """Number of frames for *duration* seconds at *fps*.  SVD caps at ~25."""
    n = max(1, round(duration * fps))
    return min(n, 25)
