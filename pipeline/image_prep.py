"""Image preprocessing for image-to-video (I2V) input normalization.

I2V models are sensitive to input resolution and aspect ratio. This module
normalizes a source image into a predictable, provider-friendly form:

- validates the input (rejects corrupt / non-image files with a path-aware
  ``ValueError``);
- applies EXIF orientation correction (Pillow ``ImageOps.exif_transpose``);
- resizes to ``target_width`` at the correct scale (Lanczos when upscaling,
  AREA when downscaling);
- normalizes the aspect ratio to ``target_aspect`` via ``letterbox`` (black
  padding) or ``crop`` (center crop).

Does NOT introduce any new model dependency. Optional AI-based upscaling is
deliberately left to a later task.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from PIL.ImageOps import exif_transpose

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

_MIN_WIDTH_WARN = 1024
_VALID_MODES = ("letterbox", "crop")
_ASPECT_RE = re.compile(r"^\d+(?:\.\d+)?[:/]\d+(?:\.\d+)?$")

# Image-to-video input constraints (shared by providers that accept a starting
# image). Validated before upload so providers fail fast with a clear message.
_VALID_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
_MIN_ASPECT = 0.4
_MAX_ASPECT = 2.5
_MIN_EDGE_PX = 300
_MAX_EDGE_PX = 6000
_MAX_IMAGE_BYTES = 30 * 1024 * 1024  # 30 MB


def validate_image_for_i2v(image_path: str, file_bytes: bytes | None = None) -> None:
    """Validate a local image for image-to-video upload.

    Checks (all required by the image-to-video backends):

    * file exists and is readable;
    * extension is one of jpeg / png / webp;
    * byte size is under 30 MB;
    * the decoded frame is an actual image (not corrupt / not a text file);
    * pixel width and height are each within ``(300, 6000)``;
    * aspect ratio (width / height) is within ``(0.4, 2.5)``.

    Parameters
    ----------
    image_path:
        Local filesystem path to the image.
    file_bytes:
        Optional pre-read bytes (used by callers that have already read the
        file, e.g. to base64-encode). When provided this is used instead of
        re-reading the file.

    Raises
    ------
    ValueError
        If any constraint is violated, with a message naming the offending path.
    """
    if image_path.lower().startswith("http://") or image_path.lower().startswith("https://"):
        return  # URL pass-through — the backend validates it.

    path = Path(image_path)
    if not path.exists():
        raise ValueError(f"Image file not found: {image_path}")

    suffix = path.suffix.lower()
    if suffix not in _VALID_IMAGE_EXTENSIONS:
        raise ValueError(
            f"Image {image_path!r}: unsupported format {suffix!r}. Allowed: {list(_VALID_IMAGE_EXTENSIONS)}"
        )

    size = path.stat().st_size if file_bytes is None else len(file_bytes)
    if size >= _MAX_IMAGE_BYTES:
        raise ValueError(
            f"Image {image_path!r}: size {size} bytes exceeds the {_MAX_IMAGE_BYTES // (1024 * 1024)} MB limit"
        )

    # Decode to confirm it's a valid image and to read the dimensions.
    try:
        img = cv2.imread(image_path)
    except Exception as exc:
        raise ValueError(f"Failed to decode image at {image_path!r}: {exc}") from None
    if img is None or img.size == 0:
        raise ValueError(f"Could not decode image at {image_path!r} (not a valid image).")

    height, width = img.shape[:2]

    if not (_MIN_EDGE_PX <= width <= _MAX_EDGE_PX):
        raise ValueError(
            f"Image {image_path!r}: width {width} is outside the allowed range ({_MIN_EDGE_PX}, {_MAX_EDGE_PX}) px"
        )
    if not (_MIN_EDGE_PX <= height <= _MAX_EDGE_PX):
        raise ValueError(
            f"Image {image_path!r}: height {height} is outside the allowed range ({_MIN_EDGE_PX}, {_MAX_EDGE_PX}) px"
        )

    aspect = width / height
    if not (_MIN_ASPECT <= aspect <= _MAX_ASPECT):
        raise ValueError(
            f"Image {image_path!r}: aspect ratio {aspect:.3f} is outside the "
            f"allowed range ({_MIN_ASPECT}, {_MAX_ASPECT})"
        )


@dataclass
class PreparedImage:
    """Result of :func:`prepare_image`."""

    path: str
    width: int
    height: int
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Aspect ratio parsing
# ---------------------------------------------------------------------------


def parse_aspect(aspect: str) -> float:
    """Parse an aspect ratio string like ``"16:9"`` (or ``"16/9"``) into a
    width-to-height float.

    Parameters
    ----------
    aspect:
        A ``"<numerator>:<denominator>"`` or ``"<numerator>/<denominator>"``
        string. Integer or decimal numbers are accepted (e.g. ``"2.35:1"``).

    Returns
    -------
    float
        width / height.

    Raises
    ------
    ValueError
        If the string is not a valid ``<number>:<number>`` ratio, or if the
        denominator is zero.
    """
    if not isinstance(aspect, str) or not _ASPECT_RE.match(aspect):
        raise ValueError(f"Invalid aspect ratio format: {aspect!r}. Expected '<num>:<num>' (e.g. '16:9').")

    match = re.search(r"([\d.]+)[:/]([\d.]+)", aspect)
    if match is None:  # pragma: no cover - regex already validated
        raise ValueError(f"Invalid aspect ratio format: {aspect!r}.")

    width = float(match.group(1))
    height = float(match.group(2))
    if height <= 0:
        raise ValueError(f"Aspect ratio denominator must be > 0, got {aspect!r}.")
    return width / height


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------


def prepare_image(
    input_path: str,
    out_path: str | None = None,
    target_aspect: str = "16:9",
    target_width: int = 1920,
    mode: str = "letterbox",
) -> PreparedImage:
    """Normalize an image for image-to-video ingestion.

    Parameters
    ----------
    input_path:
        Path to the source image.
    out_path:
        Where to write the prepared image. If ``None``, writes to
        ``<input_dir>/<name>_prep.png`` next to the source.
    target_aspect:
        Target aspect ratio string, e.g. ``"16:9"``. Parsed by
        :func:`parse_aspect`.
    target_width:
        Target width in pixels. The height is derived from the target aspect
        ratio, then the image is letterboxed/cropped to fit.
    mode:
        Either ``"letterbox"`` (proportional scale + black padding) or
        ``"crop"`` (center crop to target aspect before scaling).

    Returns
    -------
    PreparedImage
        Path, width, height and any non-fatal warnings.

    Raises
    ------
    ValueError
        If the file is not a valid/readable image, if ``mode`` is invalid, or
        if ``target_aspect`` cannot be parsed.
    """
    input_path = str(input_path)
    if mode not in _VALID_MODES:
        raise ValueError(f"Invalid mode {mode!r}. Must be one of {list(_VALID_MODES)}.")

    target_ratio = parse_aspect(target_aspect)
    target_height = round(target_width / target_ratio)

    src = Path(input_path)
    if out_path is None:
        out_path = str(src.with_name(src.stem + "_prep").with_suffix(".png"))

    # Read & validate. cv2.imread returns None on failure; a corrupt file or
    # non-image must surface as a ValueError with the path.
    try:
        img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    except Exception as exc:
        raise ValueError(f"Failed to read image at {input_path!r}: {exc}") from None

    if img is None or img.size == 0:
        raise ValueError(f"Could not decode image at {input_path!r} (not a valid image).")

    warnings: list[str] = []
    orig_width = img.shape[1]
    if orig_width < _MIN_WIDTH_WARN:
        warnings.append(f"Input image is narrow (width={orig_width} < {_MIN_WIDTH_WARN}); upscaling recommended.")

    # EXIF orientation correction. cv2 does not honour EXIF; apply via Pillow
    # which also catches format quirks Pillow handles but cv2 does not.
    img = _apply_exif_transpose(input_path, img)

    img = _prepare(img, target_ratio, target_width, target_height, mode)

    _write(img, out_path)

    return PreparedImage(path=out_path, width=img.shape[1], height=img.shape[0], warnings=warnings)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _apply_exif_transpose(input_path: str, img: np.ndarray) -> np.ndarray:
    """Re-orient ``img`` to honour EXIF Orientation. Returns the (possibly
    rotated/flipped) array.

    Uses Pillow for EXIF parsing, then converts back to a cv2-compatible
    BGR ndarray so the rest of this module keeps a single cv2/numpy pipeline.
    """
    try:
        pil = Image.open(input_path)
        transposed = exif_transpose(pil)
        # Re-encode to the same dtype/layout as cv2.imread (BGR for 3ch,
        # grayscale otherwise). Converting back through PIL guarantees EXIF
        # is baked into the pixel data.
        if transposed.mode == "L":
            result = np.asarray(transposed)
            # cv2 grayscale is single-channel; ensure that shape.
            if result.ndim == 2:
                return result
        # Color image: PIL is RGB, cv2 expects BGR.
        return cv2.cvtColor(np.asarray(transposed.convert("RGB")), cv2.COLOR_RGB2BGR)
    except Exception:  # EXIF is best-effort; never break a valid image
        return img


def _compute_letterbox(src_w: int, src_h: int, dst_w: int, dst_h: int) -> tuple[int, int]:
    """Compute the (x, y) offset of the source inside a ``dst_w x dst_h``
    canvas with the source scaled proportionally and letterboxed."""
    scale = min(dst_w / src_w, dst_h / src_h)
    sw, sh = round(src_w * scale), round(src_h * scale)
    x = round((dst_w - sw) / 2)
    y = round((dst_h - sh) / 2)
    return x, y


def _resize_for_scale(img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Resize using Lanczos when upscaling and AREA when downscaling."""
    src_w = img.shape[1]
    src_h = img.shape[0]
    upscaling = target_w >= src_w or target_h >= src_h
    method = cv2.INTER_LANCZOS4 if upscaling else cv2.INTER_AREA
    return cv2.resize(img, (target_w, target_h), interpolation=method)


def _prepare(img: np.ndarray, target_ratio: float, target_w: int, target_h: int, mode: str) -> np.ndarray:
    """Apply the aspect mode (letterbox or crop) and scale to target size.

    The crop/letterbox decision is applied *before* the final scale so the
    destination is always exactly ``(target_w, target_h)``.
    """
    src_w = img.shape[1]
    src_h = img.shape[0]
    src_ratio = src_w / src_h

    if mode == "letterbox":
        # Proportionally scale so the image fits inside the target box.
        scale = min(target_w / src_w, target_h / src_h)
        scaled_w = round(src_w * scale)
        scaled_h = round(src_h * scale)
        scaled = _resize_for_scale(img, scaled_w, scaled_h)

        canvas = np.zeros((target_h, target_w, img.shape[2]), dtype=img.dtype)
        x, y = _compute_letterbox(scaled_w, scaled_h, target_w, target_h)
        if len(img.shape) == 3 and img.shape[2] == 1:
            canvas[y : y + scaled_h, x : x + scaled_w, 0] = scaled[:, :, 0]
        else:
            canvas[y : y + scaled_h, x : x + scaled_w] = scaled
        return canvas

    # mode == "crop"
    if src_ratio > target_ratio:
        new_w = round(src_h * target_ratio)
        y, y2 = 0, src_h
        x = round((src_w - new_w) / 2)
        x2 = x + new_w
    else:
        new_h = round(src_w / target_ratio)
        x, x2 = 0, src_w
        y = round((src_h - new_h) / 2)
        y2 = y + new_h
    cropped = img[y:y2, x:x2]
    return _resize_for_scale(cropped, target_w, target_h)


def _write(img: np.ndarray, out_path: str) -> None:
    """Write ``img`` to ``out_path`` via cv2.imwrite. Fails with ValueError
    on write failure so callers know something went wrong."""
    try:
        ok = cv2.imwrite(out_path, img)
    except Exception as exc:
        raise ValueError(f"Failed to write prepared image to {out_path!r}: {exc}") from None
    if not ok:
        raise ValueError(f"cv2.imwrite returned failure for {out_path!r}.")
