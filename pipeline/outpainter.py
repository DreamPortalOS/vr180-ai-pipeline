"""
Stage 3.5 — 180° Outpaint Fill
================================

Optional stage that fills the black boundary regions in equirectangular
frames that result from limited source vertical FOV.  When a planar 2D
source is projected onto a VR180 hemisphere, the top (zenith) and bottom
(nadir) regions are pure black because no source content covers those
angles.

Three modes:
1. **none** (default) — passthrough, no change.
2. **gradient** — Gaussian-based vertical extension + smooth blending.
   OpenCV-only, no model required, works out of the box.
3. **ai** — Pluggable AI backend (SDXL inpaint / Seedance / etc.).
   Requires external deployment; clear actionable error if unavailable.

Usage:
    from pipeline.outpainter import Outpainter
    outpainter = Outpainter(mode="gradient")
    frames = outpainter.process(frames)
"""

import abc
import logging

import cv2
import numpy as np

log = logging.getLogger("outpainter")


# ---------------------------------------------------------------------------
#  ABC for pluggable AI backends
# ---------------------------------------------------------------------------


class AIOutpaintBackend(abc.ABC):
    """Abstract interface for AI-based outpainting backends.

    Subclasses must implement ``outpaint(frames, mask)`` where *frames*
    is a list of RGB ndarrays and *mask* is a binary ndarray (255 = fill).
    Returns outpainted frames.
    """

    @abc.abstractmethod
    def outpaint(self, frames: list[np.ndarray], mask: np.ndarray) -> list[np.ndarray]: ...


class MockAIOutpaintBackend(AIOutpaintBackend):
    """Mock backend for testing — fills masked regions with green."""

    def outpaint(self, frames: list[np.ndarray], mask: np.ndarray) -> list[np.ndarray]:
        mask_2d = mask > 0  # (H, W) boolean — broadcasts to each channel
        result = []
        for f in frames:
            out = f.copy()
            out[mask_2d] = [0, 255, 0]  # green fill
            result.append(out)
        return result


class SDInpaintBackend(AIOutpaintBackend):
    """Stable Diffusion inpaint backend (placeholder).

    Requires ``diffusers`` + ``torch`` and a deployed SDXL/Flux inpaint
    model on disk.  See ``docs/OUTPAINT_SETUP.md`` for deployment guide.
    """

    def __init__(self, model_path: str | None = None, device: str = "cuda"):
        self._model_path = model_path
        self._device = device
        self._pipe = None

    def _lazy_init(self):
        if self._pipe is not None:
            return
        try:
            import torch
            from diffusers import StableDiffusionInpaintPipeline as SDIP  # noqa: N817

            model_path = self._model_path or "stabilityai/stable-diffusion-2-inpainting"
            self._pipe = SDIP.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if "cuda" in self._device else torch.float32,
            ).to(self._device)
        except ImportError as e:
            raise RuntimeError(
                f"AI outpainting requires 'diffusers' and 'torch': {e}\n"
                f"  pip install diffusers torch\n"
                f"  See docs/OUTPAINT_SETUP.md for details."
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to load SD inpaint model from '{model_path}': {e}\n"
                f"  Make sure the model path is correct.\n"
                f"  See docs/OUTPAINT_SETUP.md for deployment instructions."
            ) from e

    def outpaint(self, frames: list[np.ndarray], mask: np.ndarray) -> list[np.ndarray]:
        self._lazy_init()
        from PIL import Image

        result = []
        for f in frames:
            f_rgb = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
            pil_img = Image.fromarray(f_rgb)
            pil_mask = Image.fromarray(mask)

            out = self._pipe(
                prompt="seamless equirectangular sky environment, continuous panorama",
                image=pil_img,
                mask_image=pil_mask,
                num_inference_steps=20,
                guidance_scale=7.5,
            ).images[0]

            result.append(cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR))
        return result


# ---------------------------------------------------------------------------
#  Mask detection helpers
# ---------------------------------------------------------------------------


def detect_black_boundary_mask(
    frame: np.ndarray,
    threshold: int = 10,
    top_ratio: float = 0.2,
    bottom_ratio: float = 0.2,
) -> np.ndarray:
    """Detect black boundary regions in an equirectangular frame.

    Args:
        frame: RGB ndarray (H, W, 3).
        threshold: Mean pixel value below which a row is considered "black".
        top_ratio: Fraction of frame height to scan from top.
        bottom_ratio: Fraction of frame height to scan from bottom.

    Returns:
        Binary mask (H, W) uint8, 255 = needs outpainting.
    """
    h, _w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)

    mask = np.zeros((h, frame.shape[1]), dtype=np.uint8)

    # Top boundary
    top_h = int(h * top_ratio)
    for row in range(top_h):
        if gray[row, :].mean() < threshold:
            mask[row, :] = 255
        else:
            break  # stop at first non-black row

    # Bottom boundary
    bottom_start = int(h * (1.0 - bottom_ratio))
    for row in range(h - 1, bottom_start - 1, -1):
        if gray[row, :].mean() < threshold:
            mask[row, :] = 255
        else:
            break

    return mask


# ---------------------------------------------------------------------------
#  Gradient mode — OpenCV-based vertical extension
# ---------------------------------------------------------------------------


def _gradient_outpaint_single(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fill masked regions using Gaussian-blurred edge extension.

    Strategy (for each masked row):
    1. Find the nearest non-masked row (above or below).
    2. Copy that row's content outward with decreasing alpha.
    3. Apply 1D vertical Gaussian blur to blend seams.
    """
    out = frame.copy()
    h, _w = frame.shape[:2]
    mask_bool = mask > 0

    # Clamp: if mask covers everything (degenerate), return frame as-is
    if not np.any(mask_bool):
        return out

    # Find first and last non-masked rows
    row_mask = mask_bool[:, 0]  # mask is same across all columns
    valid_rows = np.where(~row_mask)[0]

    if len(valid_rows) == 0:
        log.warning("Mask covers entire frame — cannot outpaint")
        return out

    first_valid = valid_rows[0]
    last_valid = valid_rows[-1]

    # --- Top region (0 .. first_valid) ---
    if first_valid > 0:
        src_row = frame[first_valid, :, :].astype(np.float32)
        for row in range(first_valid - 1, -1, -1):
            alpha = 1.0 - (first_valid - row) / max(first_valid, 1)
            alpha = max(0.0, min(1.0, alpha * 1.5))  # aggressive fade
            out[row, :, :] = (src_row * (1.0 - alpha) + frame[row, :, :].astype(np.float32) * alpha).astype(np.uint8)

    # --- Bottom region (last_valid .. h-1) ---
    if last_valid < h - 1:
        src_row = frame[last_valid, :, :].astype(np.float32)
        for row in range(last_valid + 1, h):
            alpha = 1.0 - (row - last_valid) / max(h - 1 - last_valid, 1)
            alpha = max(0.0, min(1.0, alpha * 1.5))
            out[row, :, :] = (src_row * (1.0 - alpha) + frame[row, :, :].astype(np.float32) * alpha).astype(np.uint8)

    # Apply vertical Gaussian blur to smooth transition seam
    blur_ksize = (1, max(3, h // 32 * 2 + 1))  # odd height
    out = cv2.GaussianBlur(out, blur_ksize, sigmaX=0, sigmaY=h / 16.0)

    # Restore original non-masked pixels
    out[~mask_bool] = frame[~mask_bool]

    return out


# ---------------------------------------------------------------------------
#  Main Outpainter class
# ---------------------------------------------------------------------------


class Outpainter:
    """Outpaint black boundary regions in equirectangular VR180 frames.

    Args:
        mode: One of ``"none"`` (passthrough), ``"gradient"`` (OpenCV-based
            vertical extension), or ``"ai"`` (pluggable AI backend).
        ai_backend: An instance of :class:`AIOutpaintBackend`.  Required when
            *mode* is ``"ai"``.  Ignored otherwise.
        mask_threshold: Pixel brightness threshold for black detection.
        mask_top_ratio: Fraction of height scanned from the top.
        mask_bottom_ratio: Fraction of height scanned from the bottom.
    """

    def __init__(
        self,
        mode: str = "none",
        ai_backend: AIOutpaintBackend | None = None,
        mask_threshold: int = 10,
        mask_top_ratio: float = 0.25,
        mask_bottom_ratio: float = 0.25,
    ):
        if mode not in ("none", "gradient", "ai"):
            raise ValueError(f"Unknown outpaint mode: {mode!r}.  Choose 'none', 'gradient', or 'ai'.")
        if mode == "ai" and ai_backend is None:
            raise ValueError("AI outpainting requires an 'ai_backend' argument.")

        self._mode = mode
        self._ai_backend = ai_backend
        self._mask_threshold = mask_threshold
        self._mask_top_ratio = mask_top_ratio
        self._mask_bottom_ratio = mask_bottom_ratio

    @property
    def mode(self) -> str:
        return self._mode

    def process(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """Outpaint a sequence of equirectangular frames.

        Args:
            frames: List of RGB ndarrays (H, W, 3).

        Returns:
            Outpainted frames, same shape and count.
        """
        if not frames:
            return []

        if self._mode == "none":
            return frames

        if self._mode == "gradient":
            return self._process_gradient(frames)

        # AI mode
        mask = detect_black_boundary_mask(
            frames[0],
            threshold=self._mask_threshold,
            top_ratio=self._mask_top_ratio,
            bottom_ratio=self._mask_bottom_ratio,
        )
        if not np.any(mask > 0):
            log.info("No black boundaries detected — skipping AI outpainting")
            return frames

        assert self._ai_backend is not None  # guaranteed by __init__
        return self._ai_backend.outpaint(frames, mask)

    def _process_gradient(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """Gradient-based outpainting for all frames."""
        mask = detect_black_boundary_mask(
            frames[0],
            threshold=self._mask_threshold,
            top_ratio=self._mask_top_ratio,
            bottom_ratio=self._mask_bottom_ratio,
        )

        if not np.any(mask > 0):
            log.info("No black boundaries detected — no outpainting needed")
            return frames

        pct = float(np.sum(mask > 0)) / mask.size * 100.0
        log.info("Gradient outpainting %d frames (mask covers %.1f%%)", len(frames), pct)

        result = []
        for f in frames:
            result.append(_gradient_outpaint_single(f, mask))
        return result
