"""
Stage 3.5 — 180° Outpaint Fill + Edge Feather
=============================================

Optional stage that treats the boundary of equirectangular VR180 frames.
When a planar 2D source is projected onto a 180° hemisphere only the source
FOV is covered; the rest is an ``alpha == 0`` hole rendered pure black by
:class:`pipeline.equirectangular_mapper.EquirectangularMapper` (issues #240 /
#255).  The hole itself is a **hard edge** the viewer can see in the headset.

Fill modes (``mode=``):
1. **none** (default) — no fill.
2. **gradient** — OpenCV edge smear into the hole + smooth blending.
   No model required, works out of the box.
3. **ai** — Pluggable AI backend (SDXL inpaint / Seedance / etc.).
   Requires external deployment; clear actionable error if unavailable.

The region to fill is taken from the mapper's **alpha plane** when the caller
passes one (``process(frames, alpha=...)``); the legacy black-row scan is only
a fallback for frames without alpha.

Edge feather (issue #244, independent of the fill mode, **off by default**):
an *angle-weighted* fade of the content to pure black between
``edge_feather_start`` and ``edge_feather_end`` degrees (0–180 FOV scale),
anchored at the alpha boundary so the fade always meets the hole.

Usage:
    from pipeline.outpainter import Outpainter
    outpainter = Outpainter(mode="gradient", edge_feather_start=165, edge_feather_end=180)
    frames = outpainter.process(frames, alpha=alpha_sbs)
"""

import abc
import logging
import math

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
#  Mask helpers
# ---------------------------------------------------------------------------


def alpha_to_fill_mask(alpha: np.ndarray) -> np.ndarray:
    """Turn an alpha plane (0 outside the source FOV) into a fill mask.

    Args:
        alpha: (H, W) alpha plane as produced by
            ``EquirectangularMapper.map_single(..., with_alpha=True)[..., 3]``
            (uint8 0/255 or bool).  An (H, W, 4) RGBA frame is accepted too.

    Returns:
        Binary mask (H, W) uint8, 255 where alpha == 0 (needs filling).
    """
    if alpha.ndim == 3:
        alpha = alpha[:, :, -1]
    return np.where(alpha > 0, 0, 255).astype(np.uint8)


def detect_black_boundary_mask(
    frame: np.ndarray,
    threshold: int = 10,
    top_ratio: float = 0.2,
    bottom_ratio: float = 0.2,
) -> np.ndarray:
    """Detect the black boundary bands of an equirectangular frame (no alpha).

    Fallback for frames that arrive **without** an alpha plane.  Rows are
    scanned inward from the top and from the bottom and the scan follows the
    contiguous black band until the first row that carries content:

    * inside the first ``top_ratio`` / ``bottom_ratio`` of the height a row is
      black when its *mean* brightness is below *threshold* (pre-#244 rule);
    * beyond that window the scan only continues through rows whose *every*
      pixel is below *threshold* — a real hole is exact (0,0,0), so this keeps
      following the band while refusing to eat merely dark content.

    Before issue #244 the scan simply **stopped** at ``ratio * H``.  With the
    default 0.25 and any source whose vertical FOV is under ~97° (a 16:9 source
    at 70–126° hfov) the black band is taller than the window, so the mask
    ended *inside* the band and the gradient filler sourced its fill from a
    black row — ``--outpaint gradient`` changed zero pixels.

    Args:
        frame: RGB ndarray (H, W, 3).
        threshold: Brightness below which a pixel/row counts as black.
        top_ratio: Fraction of the height judged by the row-mean rule from the top.
        bottom_ratio: Same, from the bottom.

    Returns:
        Binary mask (H, W) uint8, 255 = needs outpainting.
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)
    row_mean_black = gray.mean(axis=1) < threshold
    row_all_black = gray.max(axis=1) < threshold

    def _band(order: np.ndarray, window: int) -> int:
        """Length of the black band along *order* (row indices, edge first)."""
        n = 0
        for i, row in enumerate(order):
            black = row_mean_black[row] if i < window else row_all_black[row]
            if not black:
                break
            n += 1
        return n

    mask = np.zeros((h, w), dtype=np.uint8)
    top_n = _band(np.arange(h), int(h * top_ratio))
    mask[:top_n, :] = 255
    bottom_n = _band(np.arange(h - 1, -1, -1), int(h * bottom_ratio))
    if bottom_n:
        mask[h - bottom_n :, :] = 255
    return mask


# ---------------------------------------------------------------------------
#  Gradient mode — OpenCV-based edge extension
# ---------------------------------------------------------------------------


def _smear_weights(distance: np.ndarray, band: np.ndarray | float) -> np.ndarray:
    """Fade for a smeared pixel *distance* px into a hole that is *band* px deep.

    Full copy for the inner third of the band, then a linear fade that reaches
    black at the frame edge (``distance == band``).
    """
    return np.clip(1.5 * (1.0 - distance / np.maximum(band, 1.0)), 0.0, 1.0)


def _smear_columns(out: np.ndarray, src_col: int, cols: range) -> None:
    """Smear column *src_col* of *out* (in place) across *cols*, fading with distance.

    The fade reaches black at the column of *cols* farthest from *src_col*
    (the frame border, or the midpoint of an interior gap).
    """
    band = float(max(abs(c - src_col) for c in cols))
    for c in cols:
        out[:, c] = out[:, src_col] * _smear_weights(float(abs(c - src_col)), band)


def _gradient_outpaint_single(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fill masked regions by smearing the nearest content pixel outward.

    Strategy:
    1. **Vertical pass** — for every column that has content, copy its first /
       last content pixel into the masked rows above / below, fading to black
       towards the frame edge (:func:`_smear_weights`).
    2. **Horizontal pass** — columns with no content at all (the side holes of
       a partial-hfov source) are filled from the nearest filled column with
       the same fade.
    3. Vertical Gaussian blur to soften the seam, then every non-masked pixel
       is restored byte-exact.

    Works for any mask shape.  The alpha hole of a pinhole source is a rounded
    rectangle, not two horizontal bands — the pre-#244 version read column 0
    as "the" row mask, so a fully-masked column 0 was mistaken for "mask
    covers the whole frame" and nothing was filled.
    """
    mask_bool = mask > 0
    if not np.any(mask_bool):
        return frame.copy()
    content = ~mask_bool
    if not np.any(content):
        log.warning("Mask covers entire frame — cannot outpaint")
        return frame.copy()

    h, w = mask_bool.shape
    cols = np.arange(w)
    out = frame.astype(np.float32)

    # --- Vertical pass (per column, row loop keeps memory O(W)) ---
    col_has = content.any(axis=0)
    first = np.where(col_has, content.argmax(axis=0), 0)
    last = np.where(col_has, h - 1 - content[::-1, :].argmax(axis=0), h - 1)
    src_first = out[first, cols]  # (W, 3) first content pixel per column
    src_last = out[last, cols]
    band_top = first.astype(np.float32)
    band_bot = (h - 1 - last).astype(np.float32)

    for r in range(int(first[col_has].max())):
        d = first - r
        sel = col_has & (d > 0)
        if sel.any():
            out[r, sel] = src_first[sel] * _smear_weights(d[sel], band_top[sel])[:, None]
    for r in range(int(last[col_has].min()) + 1, h):
        d = r - last
        sel = col_has & (d > 0)
        if sel.any():
            out[r, sel] = src_last[sel] * _smear_weights(d[sel], band_bot[sel])[:, None]

    # --- Horizontal pass (runs of columns without any content) ---
    # Edge runs fade to black at the frame border; an *interior* run (e.g. the
    # two side holes meeting between the eyes of an SBS frame) is split at its
    # midpoint and each half is smeared from its own side.
    if not col_has.all():
        padded = np.concatenate([[True], col_has, [True]])
        run_starts = np.flatnonzero(padded[:-1] & ~padded[1:])
        run_ends = np.flatnonzero(~padded[:-1] & padded[1:]) - 1
        for a, b in zip(run_starts, run_ends, strict=True):
            if a == 0 and b == w - 1:
                continue  # no content column at all — nothing to smear from
            if a == 0:
                _smear_columns(out, src_col=b + 1, cols=range(a, b + 1))
            elif b == w - 1:
                _smear_columns(out, src_col=a - 1, cols=range(a, b + 1))
            else:
                mid = (a + b) // 2
                _smear_columns(out, src_col=a - 1, cols=range(a, mid + 1))
                _smear_columns(out, src_col=b + 1, cols=range(mid + 1, b + 1))

    out_u8 = np.clip(np.rint(out), 0, 255).astype(np.uint8)

    # Vertical Gaussian blur to smooth the transition seam
    blur_ksize = (1, max(3, h // 32 * 2 + 1))  # odd height
    out_u8 = cv2.GaussianBlur(out_u8, blur_ksize, sigmaX=0, sigmaY=h / 16.0)

    # Restore original non-masked pixels
    out_u8[content] = frame[content]
    return out_u8


# ---------------------------------------------------------------------------
#  Edge feather — angle-weighted fade to black at the content edge (issue #244)
# ---------------------------------------------------------------------------

#: Defaults used when the caller enables the feather with only one bound.
DEFAULT_EDGE_FEATHER_START = 165.0
DEFAULT_EDGE_FEATHER_END = 180.0

#: Azimuth samples used to trace the content edge around the forward axis.
_EDGE_AZIMUTH_SAMPLES = 1440


def resolve_edge_feather(start: float | None, end: float | None) -> tuple[float, float] | None:
    """Validate the edge-feather angles.

    Returns ``None`` when both are ``None`` — the feather is **off** and the
    outpainter's output is byte-identical to pre-#244.  A single bound enables
    the feather and the other takes its documented default (start 165°, end
    180°).

    Angles are on the 0–180° FOV scale (0 = forward axis, 180 = hemisphere
    rim) and must satisfy ``0 <= start <= end <= 180``; anything else raises
    :class:`ValueError` rather than being silently clamped.  ``start == end``
    is a hard cut at that angle.
    """
    if start is None and end is None:
        return None
    s = DEFAULT_EDGE_FEATHER_START if start is None else float(start)
    e = DEFAULT_EDGE_FEATHER_END if end is None else float(end)
    if not (math.isfinite(s) and math.isfinite(e)):
        raise ValueError(f"edge feather angles must be finite; got start={s}, end={e}")
    if not (0.0 <= s <= 180.0 and 0.0 <= e <= 180.0):
        raise ValueError(f"edge feather angles must lie in [0, 180]; got start={s}, end={e}")
    if s > e:
        raise ValueError(f"edge feather start ({s}) must not exceed end ({e})")
    return s, e


def _hemisphere_pixel_angles(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel ``(theta_deg, psi_rad)`` for one half-equirectangular eye.

    Same spherical convention as ``EquirectangularMapper._build_mesh``
    (longitude −90..+90° across the width, colatitude 0..180° down the
    height), evaluated at pixel centres.  *theta* is the angular distance from
    the forward axis (0 at the centre, 90 at the rim); *psi* is the azimuth
    around that axis in ``[0, 2π)``.
    """
    lon = ((np.arange(w, dtype=np.float64) + 0.5) / w - 0.5) * np.pi
    colat = (np.arange(h, dtype=np.float64) + 0.5) / h * np.pi
    sin_colat = np.sin(colat)[:, None]
    x = np.sin(lon)[None, :] * sin_colat
    y = np.broadcast_to(np.cos(colat)[:, None], (h, w))
    z = np.cos(lon)[None, :] * sin_colat
    theta = np.degrees(np.arccos(np.clip(z, -1.0, 1.0)))
    psi = np.mod(np.arctan2(y, x), 2.0 * np.pi)
    return theta.astype(np.float32), psi.astype(np.float32)


def content_edge_angles(alpha_eye: np.ndarray, n_psi: int = _EDGE_AZIMUTH_SAMPLES) -> np.ndarray:
    """Trace the content edge of one eye's alpha plane.

    For each of *n_psi* azimuths around the forward axis, march a ray from the
    centre outward (half-pixel steps) and return the angle (degrees) at which
    the first ``alpha == 0`` pixel is met.  90° where the content reaches the
    hemisphere rim (frame border).
    """
    h, w = alpha_eye.shape
    covered = alpha_eye > 0
    n_theta = max(h, w) + 1
    theta = np.linspace(0.0, np.pi / 2.0, n_theta)[:, None]
    psi = np.linspace(0.0, 2.0 * np.pi, n_psi, endpoint=False)[None, :]
    sin_t = np.sin(theta)
    x = sin_t * np.cos(psi)
    y = sin_t * np.sin(psi)
    z = np.broadcast_to(np.cos(theta), x.shape)
    lon = np.arctan2(x, z)
    colat = np.arccos(np.clip(y, -1.0, 1.0))
    u = np.clip(np.floor((lon / np.pi + 0.5) * w).astype(np.int64), 0, w - 1)
    v = np.clip(np.floor(colat / np.pi * h).astype(np.int64), 0, h - 1)
    hole = ~covered[v, u]
    hole[-1, :] = True  # the rim terminates every ray
    first = hole.argmax(axis=0)
    return np.degrees(theta[first, 0]).astype(np.float32)


def compute_edge_feather_weights(alpha_eye: np.ndarray, start_deg: float, end_deg: float) -> np.ndarray:
    """Per-pixel brightness multiplier in ``[0, 1]`` for **one eye**.

    Angle-weighted: every pixel is placed by its angular distance ``θ`` from
    the forward axis and the fade is a function of angle, not of pixel
    position.  With half-angles ``s = start/2`` and ``e = end/2``:

    * the black end is ``e_eff = min(e, θ_edge)`` where ``θ_edge`` is the
      content edge in that pixel's azimuth (from the alpha plane, see
      :func:`content_edge_angles`) — so the fade always meets the actual
      alpha boundary, whether the source fills the hemisphere or only 126° of
      it;
    * the ramp is linear from ``e_eff − width`` (weight 1) to ``e_eff``
      (weight 0) with ``width = e − s`` clamped to ``e_eff`` — a feather wider
      than the content half-angle starts at the centre instead of overshooting
      it;
    * pixels beyond ``e_eff``, ``alpha == 0`` pixels and the one-pixel ring
      touching the hole/frame border are exactly 0, so the ramp reaches black
      with no residual step.

    Args:
        alpha_eye: (H, W) alpha plane of one hemisphere (or (H, W, 4) RGBA).
        start_deg: Angle (0–180 FOV scale) where darkening starts.
        end_deg: Angle where the image is fully black.
    """
    if alpha_eye.ndim == 3:
        alpha_eye = alpha_eye[:, :, -1]
    h, w = alpha_eye.shape
    covered = alpha_eye > 0
    if not covered.any():
        return np.zeros((h, w), dtype=np.float32)

    theta_p, psi_p = _hemisphere_pixel_angles(h, w)
    edge = content_edge_angles(alpha_eye)
    n = edge.shape[0]
    pos = psi_p.astype(np.float64) / (2.0 * np.pi) * n
    i0 = np.floor(pos).astype(np.int64) % n
    frac = (pos - np.floor(pos)).astype(np.float32)
    theta_edge = edge[i0] * (1.0 - frac) + edge[(i0 + 1) % n] * frac

    px_deg = 180.0 / h  # angular size of one pixel at the centre
    s, e = start_deg / 2.0, end_deg / 2.0
    e_eff = np.minimum(e, theta_edge) - px_deg
    width = np.maximum(np.minimum(e - s, e_eff), 1e-6)
    weights = np.clip((e_eff - theta_p) / width, 0.0, 1.0).astype(np.float32)

    # Guard: anything touching the hole or the frame border goes fully black.
    inner = cv2.erode(
        covered.astype(np.uint8), np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0
    )
    weights[inner == 0] = 0.0
    return weights


def sbs_edge_feather_weights(alpha_sbs: np.ndarray, start_deg: float, end_deg: float) -> np.ndarray:
    """Weights for a VR180 side-by-side frame — each half is one hemisphere."""
    if alpha_sbs.ndim == 3:
        alpha_sbs = alpha_sbs[:, :, -1]
    w2 = alpha_sbs.shape[1]
    if w2 % 2:
        raise ValueError(f"SBS alpha width must be even, got {w2}")
    w = w2 // 2
    left, right = alpha_sbs[:, :w], alpha_sbs[:, w:]
    wl = compute_edge_feather_weights(left, start_deg, end_deg)
    wr = wl if np.array_equal(left, right) else compute_edge_feather_weights(right, start_deg, end_deg)
    return np.concatenate([wl, wr], axis=1)


def apply_edge_feather(frame: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Multiply an RGB frame by per-pixel *weights* (H, W); returns uint8."""
    out = frame.astype(np.float32) * weights[:, :, None]
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
#  Main Outpainter class
# ---------------------------------------------------------------------------


class Outpainter:
    """Outpaint / feather the boundary of equirectangular VR180 SBS frames.

    Args:
        mode: One of ``"none"`` (no fill), ``"gradient"`` (OpenCV-based edge
            smear), or ``"ai"`` (pluggable AI backend).
        ai_backend: An instance of :class:`AIOutpaintBackend`.  Required when
            *mode* is ``"ai"``.  Ignored otherwise.
        mask_threshold: Pixel brightness threshold for the black-row fallback
            (only used when :meth:`process` gets no *alpha*).
        mask_top_ratio: Fraction of height judged by the row-mean rule from
            the top (fallback only, see :func:`detect_black_boundary_mask`).
        mask_bottom_ratio: Same, from the bottom.
        edge_feather_start: Angle (0–180 FOV scale) where the edge fade to
            black starts.  ``None`` for both feather args = feather **off**
            (default; output unchanged).  Passing only one enables the
            feather with the other at its default (165 / 180).
        edge_feather_end: Angle where the image is fully black.

    The feather assumes the VR180 SBS layout (left hemisphere | right
    hemisphere), which is what Stage 3 produces.
    """

    def __init__(
        self,
        mode: str = "none",
        ai_backend: AIOutpaintBackend | None = None,
        mask_threshold: int = 10,
        mask_top_ratio: float = 0.25,
        mask_bottom_ratio: float = 0.25,
        edge_feather_start: float | None = None,
        edge_feather_end: float | None = None,
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
        self._edge_feather = resolve_edge_feather(edge_feather_start, edge_feather_end)

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def edge_feather(self) -> tuple[float, float] | None:
        """``(start, end)`` in degrees, or ``None`` when the feather is off."""
        return self._edge_feather

    def process(self, frames: list[np.ndarray], alpha: np.ndarray | None = None) -> list[np.ndarray]:
        """Outpaint (and/or edge-feather) a sequence of equirectangular frames.

        Args:
            frames: List of RGB ndarrays (H, W, 3) — VR180 SBS.
            alpha: Optional (H, W) alpha plane matching the frames (0 outside
                the source FOV), e.g. from
                ``EquirectangularMapper.map_single(..., with_alpha=True)``
                tiled for both eyes.  When given, the fill mask **is** the
                alpha hole; otherwise the black-row fallback is used.  The
                same geometry is assumed for every frame.

        Returns:
            Frames, same shape and count.  With ``mode="none"`` and the
            feather off this is the input list itself (passthrough).
        """
        if not frames:
            return []

        if self._mode == "none" and self._edge_feather is None:
            return frames

        alpha_2d = self._alpha_plane(alpha, frames[0].shape)

        if self._mode == "none":
            result = frames
        elif self._mode == "gradient":
            result = self._process_gradient(frames, alpha_2d)
        else:
            result = self._process_ai(frames, alpha_2d)

        if self._edge_feather is not None:
            # After a fill the hemisphere is (by construction) fully covered, so
            # the feather anchors at the physical rim instead of the alpha edge —
            # otherwise it would simply black out what the filler just painted.
            feather_alpha = alpha_2d if self._mode == "none" else None
            result = self._apply_edge_feather(result, feather_alpha)
        return result

    # ------------------------------------------------------------------ #
    #  helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _alpha_plane(alpha: np.ndarray | None, frame_shape: tuple[int, ...]) -> np.ndarray | None:
        if alpha is None:
            return None
        a = np.asarray(alpha)
        if a.ndim == 3:
            a = a[:, :, -1]
        if a.ndim != 2 or a.shape != tuple(frame_shape[:2]):
            raise ValueError(f"alpha shape {a.shape} does not match frame shape {tuple(frame_shape[:2])}")
        return a

    def _fill_mask(self, frame: np.ndarray, alpha: np.ndarray | None) -> np.ndarray:
        if alpha is not None:
            return alpha_to_fill_mask(alpha)
        return detect_black_boundary_mask(
            frame,
            threshold=self._mask_threshold,
            top_ratio=self._mask_top_ratio,
            bottom_ratio=self._mask_bottom_ratio,
        )

    def _process_gradient(self, frames: list[np.ndarray], alpha: np.ndarray | None) -> list[np.ndarray]:
        """Gradient-based outpainting for all frames."""
        mask = self._fill_mask(frames[0], alpha)

        if not np.any(mask > 0):
            log.info("No black boundaries detected — no outpainting needed")
            return frames

        pct = float(np.sum(mask > 0)) / mask.size * 100.0
        result = [_gradient_outpaint_single(f, mask) for f in frames]
        changed = int(np.count_nonzero(np.any(result[0] != frames[0], axis=2)))
        log.info(
            "Gradient outpainting %d frames (mask %s, covers %.1f%%, %d px changed in frame 0)",
            len(frames),
            "from alpha" if alpha is not None else "from black rows",
            pct,
            changed,
        )
        return result

    def _process_ai(self, frames: list[np.ndarray], alpha: np.ndarray | None) -> list[np.ndarray]:
        mask = self._fill_mask(frames[0], alpha)
        if not np.any(mask > 0):
            log.info("No black boundaries detected — skipping AI outpainting")
            return frames

        assert self._ai_backend is not None  # guaranteed by __init__
        return self._ai_backend.outpaint(frames, mask)

    def _apply_edge_feather(self, frames: list[np.ndarray], alpha: np.ndarray | None) -> list[np.ndarray]:
        assert self._edge_feather is not None
        start, end = self._edge_feather
        h, w = frames[0].shape[:2]
        if alpha is None:
            log.info("Edge feather: no alpha plane — treating the hemisphere as fully covered (fade at the rim)")
            alpha = np.full((h, w), 255, dtype=np.uint8)
        weights = sbs_edge_feather_weights(alpha, start, end)
        faded = float(np.mean(weights < 1.0)) * 100.0
        log.info("Edge feather %.1f°→%.1f° on %d frames (%.1f%% of pixels darkened)", start, end, len(frames), faded)
        return [apply_edge_feather(f, weights) for f in frames]
