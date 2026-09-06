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
import threading
from collections import OrderedDict
from typing import NamedTuple

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


#: Column strip for the vectorised filler (#271).  One 5760-wide SBS row is
#: 17 KB, so the 181-row window of the vertical Gaussian (2880²/eye) thrashes
#: L2 when the whole frame is filtered in one call; 512-column strips keep it
#: cache-resident (~4× cheaper per pixel, identical output — the kernel is
#: purely vertical, so columns are independent).
_GRADIENT_STRIP_COLS = 512
#: Row chunk for the smears — bounds the per-chunk temporaries (float64
#: weights 64 × 512 × 8 B ≈ 260 KB, float32 product ≈ 390 KB) so every numpy
#: pass stays in cache; larger chunks measured slower, smaller ones pay more
#: per-call overhead.
_GRADIENT_SMEAR_ROWS = 64
#: Row block for the per-column content-bounds scan.
_GRADIENT_BOUNDS_ROWS = 32


def _true_runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive ``(start, end)`` pairs of the maximal ``True`` runs of a 1-D bool array."""
    padded = np.concatenate(([False], flags, [False]))
    starts = np.flatnonzero(padded[1:] & ~padded[:-1])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:]) - 1
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def _column_content_bounds(mask_bool: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-column ``(has_content, first_content_row, last_content_row)``.

    Same values as ``content.argmax(axis=0)`` / the reversed argmax (``0`` and
    ``h - 1`` for columns without content), without the strided axis-0 scans
    that cost ~50 ms each at 5760×2880: a row-blocked ``all`` locates the block
    holding the edge, a small gather refines the row inside it.
    """
    h, w = mask_bool.shape
    B = _GRADIENT_BOUNDS_ROWS
    n_full, rem = divmod(h, B)
    n_blk = n_full + (1 if rem else 0)
    blk_has = np.empty((n_blk, w), dtype=bool)  # True where the block holds content
    if n_full:
        np.logical_not(mask_bool[: n_full * B].reshape(n_full, B, w).all(axis=1), out=blk_has[:n_full])
    if rem:
        np.logical_not(mask_bool[n_full * B :].all(axis=0), out=blk_has[n_full])
    col_has = blk_has.any(axis=0)
    cols = np.arange(w)[None, :]
    k = np.arange(B)[:, None]

    first_blk = blk_has.argmax(axis=0)
    rows = np.minimum(first_blk[None, :] * B + k, h - 1)
    first = first_blk * B + (~mask_bool[rows, cols]).argmax(axis=0)

    last_blk = n_blk - 1 - blk_has[::-1].argmax(axis=0)
    rows = last_blk[None, :] * B + k
    content_rows = ~mask_bool[np.minimum(rows, h - 1), cols] & (rows < h)
    last = last_blk * B + (B - 1 - content_rows[::-1].argmax(axis=0))
    return col_has, np.where(col_has, first, 0), np.where(col_has, last, h - 1)


def _smear_band(
    out: np.ndarray,
    frame: np.ndarray,
    anchor: np.ndarray,
    band: np.ndarray,
    col_has: np.ndarray,
    *,
    top: bool,
) -> None:
    """Vertical pass for one side, written straight into the uint8 *out*.

    For every content column the pixel at row ``anchor[c]`` is copied into the
    rows before it (*top*) or after it, with :func:`_smear_weights` over the
    distance and the ``band[c]``-deep hole.  Arithmetic is the reference row
    loop's: float64 weights × the float64-promoted source pixel, rounded to
    float32 (the old float32 frame), then ``rint`` — same bytes.
    """
    h, w = out.shape[:2]
    n_ch = out.shape[2]
    src_u8 = frame[anchor, np.arange(w)]  # (W, C) anchor pixel per column
    src64 = np.ascontiguousarray(src_u8.T, dtype=np.float64)  # (C, W)
    anchor64 = anchor.astype(np.float64)
    band64 = band.astype(np.float64)
    strip, chunk = _GRADIENT_STRIP_COLS, _GRADIENT_SMEAR_ROWS
    vals = np.empty((n_ch, chunk, strip), dtype=np.float32)
    for c0 in range(0, w, strip):
        c1 = min(w, c0 + strip)
        has = col_has[c0:c1]
        if not has.any():
            continue
        edge = anchor[c0:c1][has]
        r_start, r_stop = (0, int(edge.max())) if top else (int(edge.min()) + 1, h)
        # Rows before (top) / after (bottom) the nearest edge of the strip are selected in
        # every content column; the mask is only needed past that point.
        all_rows_end = int(edge.min()) if top else h
        all_rows_start = 0 if top else int(edge.max()) + 1
        anc, bnd = anchor64[None, c0:c1], band64[None, c0:c1]
        src, src_row = src64[:, None, c0:c1], src_u8[None, c0:c1]
        strip_all_content = bool(has.all())
        for r0 in range(r_start, r_stop, chunk):
            r1 = min(r_stop, r0 + chunk)
            rows = np.arange(r0, r1, dtype=np.float64)[:, None]
            d = anc - rows if top else rows - anc
            full = strip_all_content and all_rows_start <= r0 and r1 <= all_rows_end
            sel = None if full else (d > 0) & has[None, :]
            if sel is not None and not sel.any():
                continue
            weights = _smear_weights(d, bnd)  # (rows, cols) float64, unselected cells clip to 1
            dst = out[r0:r1, c0:c1]
            if weights.min() == 1.0:
                # Inner third of every band: weight exactly 1 → the anchor pixel byte-exact.
                if full or sel.all():
                    dst[...] = src_row
                else:
                    cv2.copyTo(np.ascontiguousarray(np.broadcast_to(src_row, dst.shape)), sel.view(np.uint8), dst)
                continue
            v = vals[:, : r1 - r0, : c1 - c0]
            # float64 product, rounded once to float32 on output — as the old float32 frame took it
            np.multiply(src, weights[None, :, :], out=v)
            np.rint(v, out=v)
            block = cv2.merge(list(v.astype(np.uint8)))  # channels-last (rows, cols, C)
            if full or sel.all():
                dst[...] = block
            else:
                cv2.copyTo(block, sel.view(np.uint8), dst)


def _smeared_column_f32(frame: np.ndarray, c: int, first_c: int, last_c: int) -> np.ndarray:
    """Column *c* as the float32 frame held it after the vertical pass — ``(C, H)``.

    Source column of the horizontal pass: content rows are the frame bytes,
    the rows beyond ``first_c`` / ``last_c`` carry the (unrounded) vertical
    smear exactly as :func:`_smear_band` computes it.
    """
    h = frame.shape[0]
    col = frame[:, c].astype(np.float32)  # (H, C)
    if first_c > 0:
        d = np.arange(first_c, 0, -1, dtype=np.float64)  # first_c - r for r = 0 .. first_c-1
        col[:first_c] = col[first_c].astype(np.float64)[None, :] * _smear_weights(d, float(first_c))[:, None]
    depth = h - 1 - last_c
    if depth > 0:
        d = np.arange(1, depth + 1, dtype=np.float64)  # r - last_c for r = last_c+1 .. h-1
        col[last_c + 1 :] = col[last_c].astype(np.float64)[None, :] * _smear_weights(d, float(depth))[:, None]
    return np.ascontiguousarray(col.T)


def _smear_run(out: np.ndarray, col_t: np.ndarray, src_col: int, c_start: int, c_stop: int) -> None:
    """Horizontal pass: fill columns ``[c_start, c_stop)`` from *col_t* (``(C, H)`` float32,
    the vertically smeared column *src_col*), fading with distance.

    The fade reaches black at the column farthest from *src_col* (the frame
    border, or the midpoint of an interior gap).  The reference multiplied the
    float32 column by a float64 *scalar* weight, which NumPy 1.x performs in
    float32 — hence the explicit float32 weights here.
    """
    h = out.shape[0]
    dist = np.abs(np.arange(c_start, c_stop, dtype=np.float64) - src_col)
    w32 = _smear_weights(dist, float(dist.max())).astype(np.float32)
    chunk = _GRADIENT_SMEAR_ROWS
    for r0 in range(0, h, chunk):
        r1 = min(h, r0 + chunk)
        vals = col_t[:, r0:r1, None] * w32[None, None, :]  # (C, rows, cols) float32
        np.rint(vals, out=vals)
        out[r0:r1, c_start:c_stop] = cv2.merge(list(vals.astype(np.uint8)))


def _blur_hole(out: np.ndarray, mask_bool: np.ndarray) -> None:
    """Vertical Gaussian over the hole, written back to the masked pixels only.

    The kernel is ``(1, K)`` — purely vertical — so each column strip is
    filtered independently, and only the row bands within ``K // 2`` of a
    masked row are filtered at all.  Reflection at the true frame border is
    preserved because a band is only ever clamped *at* that border.  Bands
    closer than ``h // 2`` are merged: OpenCV's per-call overhead with a tall
    kernel outweighs the rows saved.
    """
    h, w = mask_bool.shape
    ksize = (1, max(3, h // 32 * 2 + 1))  # odd height, as before
    sigma_y = h / 16.0
    radius = ksize[1] // 2
    merge_gap = h // 2
    mask_u8 = mask_bool.view(np.uint8)
    strip = _GRADIENT_STRIP_COLS
    for c0 in range(0, w, strip):
        c1 = min(w, c0 + strip)
        runs = _true_runs(mask_bool[:, c0:c1].any(axis=1))
        if not runs:
            continue
        groups: list[list[tuple[int, int]]] = [[runs[0]]]
        for run in runs[1:]:
            if run[0] - groups[-1][-1][1] - 1 <= merge_gap:
                groups[-1].append(run)
            else:
                groups.append([run])
        blurred = []
        for grp in groups:
            r0, r1 = max(0, grp[0][0] - radius), min(h, grp[-1][1] + 1 + radius)
            if r1 - r0 < ksize[1]:  # clamped at a border: keep one full kernel of real rows
                r1 = min(h, r0 + ksize[1])
                r0 = max(0, r1 - ksize[1])
            blurred.append((r0, cv2.GaussianBlur(out[r0:r1, c0:c1], ksize, sigmaX=0, sigmaY=sigma_y)))
        # Write back only after every band of the strip has been read (in place).
        for grp, (r0, blk) in zip(groups, blurred, strict=True):
            for ra, rb in grp:
                cv2.copyTo(blk[ra - r0 : rb + 1 - r0], mask_u8[ra : rb + 1, c0:c1], out[ra : rb + 1, c0:c1])


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

    Vectorised in #271 (1.25 s → well under 0.25 s per 5760×2880 frame) with
    **byte-identical output**: the smears are computed per column strip / row
    chunk in the same float64-then-float32 arithmetic the old row and column
    loops used, the frame is never converted to float as a whole, the blur is
    limited to the rows around the hole (per 512-column strip), and only the
    masked pixels are written back.  ``frame`` is an RGB uint8 array; *mask*
    is ``> 0`` where the frame needs filling.
    """
    mask_bool = np.ascontiguousarray(mask > 0)
    if not mask_bool.any():
        return frame.copy()
    if mask_bool.all():
        log.warning("Mask covers entire frame — cannot outpaint")
        return frame.copy()
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)

    h, w = mask_bool.shape
    col_has, first, last = _column_content_bounds(mask_bool)
    out = frame.copy()

    # --- Vertical pass ---
    _smear_band(out, frame, first, first, col_has, top=True)
    _smear_band(out, frame, last, h - 1 - last, col_has, top=False)

    # --- Horizontal pass (runs of columns without any content) ---
    # Edge runs fade to black at the frame border; an *interior* run (e.g. the
    # two side holes meeting between the eyes of an SBS frame) is split at its
    # midpoint and each half is smeared from its own side.
    if not col_has.all():
        for a, b in _true_runs(~col_has):
            if a == 0 and b == w - 1:
                continue  # no content column at all — nothing to smear from
            if a == 0:
                _smear_run(out, _smeared_column_f32(frame, b + 1, first[b + 1], last[b + 1]), b + 1, a, b + 1)
            elif b == w - 1:
                _smear_run(out, _smeared_column_f32(frame, a - 1, first[a - 1], last[a - 1]), a - 1, a, b + 1)
            else:
                mid = (a + b) // 2
                _smear_run(out, _smeared_column_f32(frame, a - 1, first[a - 1], last[a - 1]), a - 1, a, mid + 1)
                _smear_run(out, _smeared_column_f32(frame, b + 1, first[b + 1], last[b + 1]), b + 1, mid + 1, b + 1)

    # --- Vertical Gaussian blur over the hole; non-masked pixels never change ---
    _blur_hole(out, mask_bool)
    return out


# ---------------------------------------------------------------------------
#  Edge feather — angle-weighted fade to black at the content edge (issue #244)
# ---------------------------------------------------------------------------

#: Defaults used when the caller enables the feather with only one bound.
DEFAULT_EDGE_FEATHER_START = 165.0
DEFAULT_EDGE_FEATHER_END = 180.0

#: Azimuth samples used to trace the content edge around the forward axis.
_EDGE_AZIMUTH_SAMPLES = 1440

# ---------------------------------------------------------------------------
#  Issue #264 — feather caches.
#
#  ``compute_edge_feather_weights`` used to rebuild the whole per-pixel angle
#  table and re-trace 1440 azimuth rays on every call (~0.7 s at 2880²/eye,
#  every frame on the streaming / SphereAccumulator paths).  Two bounded,
#  process-local LRUs now remove that work:
#
#  * ``_GEOMETRY_CACHE`` — keyed by canvas size only.  Holds the per-pixel
#    angular-distance table, the azimuth ray-index / interpolation tables and
#    the ray sampling table.  Hits on every call at a known size, whether or
#    not the mask changes (the accumulator advances its mask every frame).
#  * ``_WEIGHTS_CACHE`` — keyed by the mask.  A cheap sub-sampled digest is
#    the dict key (fast rejection); a hit is confirmed with ``np.array_equal``
#    against a stored copy of the alpha, so a stale result can never be
#    returned.  Hits when the mask is static (the streaming path).
#
#  Both are in-memory only (never persisted, never written to disk), capped at
#  ``_FEATHER_CACHE_MAXSIZE`` entries, lock-guarded, and byte-transparent: a
#  cached call returns exactly the bytes an uncached call would.
# ---------------------------------------------------------------------------

#: Upper bound on entries per feather cache (distinct canvas sizes / masks).
_FEATHER_CACHE_MAXSIZE = 4


class _LRUCache:
    """Tiny bounded LRU.  Every operation takes the lock, so a cache may be
    shared between threads; values are computed *outside* the lock, so two
    threads racing on one key may both compute it (same bytes, last writer
    wins).  ``hits``/``misses`` are key-level counters for tests."""

    def __init__(self, maxsize: int) -> None:
        self.maxsize = int(maxsize)
        self._data: OrderedDict = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key):
        with self._lock:
            try:
                value = self._data[key]
            except KeyError:
                self.misses += 1
                return None
            self._data.move_to_end(key)
            self.hits += 1
            return value

    def touch(self, key) -> None:
        """Mark *key* most recently used if present (no hit/miss accounting)."""
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)

    def put(self, key, value) -> None:
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)

    def note_false_hit(self) -> None:
        """A key matched but the exact check rejected it: count it as a miss."""
        with self._lock:
            self.hits -= 1
            self.misses += 1

    def ordered_keys(self) -> list:
        """Snapshot of the keys, least recently used first."""
        with self._lock:
            return list(self._data)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self.hits = self.misses = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


#: ``(H, W, n_psi)`` -> :class:`_FeatherTables` (depend on the canvas size only).
_GEOMETRY_CACHE = _LRUCache(_FEATHER_CACHE_MAXSIZE)
#: ``(H, W, dtype, start, end, mask digest)`` -> ``(alpha copy, weights)``.
_WEIGHTS_CACHE = _LRUCache(_FEATHER_CACHE_MAXSIZE)


def _clear_feather_caches() -> None:
    """Drop every cached feather table (tests / benchmarks)."""
    _GEOMETRY_CACHE.clear()
    _WEIGHTS_CACHE.clear()


class _FeatherTables(NamedTuple):
    """Size-only tables for one ``(H, W, n_psi)``; every array is read-only."""

    theta_p: np.ndarray  # (H, W) float32 — angular distance of each pixel from the forward axis
    i0: np.ndarray  # (H, W) int64 — azimuth ray index below each pixel
    i1: np.ndarray  # (H, W) int64 — the next ray (wraps)
    frac: np.ndarray  # (H, W) float32 — interpolation weight towards ``i1``
    omf: np.ndarray  # (H, W) float32 — ``1 - frac``
    ray_flat: np.ndarray  # (n_theta, n_psi) int64 — flat pixel index sampled by each ray step
    ray_theta_deg: np.ndarray  # (n_theta,) float32 — angle of each ray step


def _geometry_tables(h: int, w: int, n_psi: int) -> _FeatherTables:
    """Tables that depend only on ``(H, W, n_psi)``, built once per size.

    The arithmetic is the PR #259 code verbatim (same dtypes, same operation
    order) so the tables — and everything derived from them — are bit-identical
    to the previous per-call evaluation.
    """
    key = (int(h), int(w), int(n_psi))
    tables = _GEOMETRY_CACHE.get(key)
    if tables is not None:
        return tables

    theta_p, psi_p = _hemisphere_pixel_angles(h, w)
    pos = psi_p.astype(np.float64) / (2.0 * np.pi) * n_psi
    i0 = np.floor(pos).astype(np.int64) % n_psi
    frac = (pos - np.floor(pos)).astype(np.float32)
    i1 = (i0 + 1) % n_psi
    omf = 1.0 - frac

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
    ray_flat = v * w + u
    ray_theta_deg = np.degrees(theta[:, 0]).astype(np.float32)

    tables = _FeatherTables(theta_p, i0, i1, frac, omf, ray_flat, ray_theta_deg)
    for arr in tables:
        arr.setflags(write=False)
    _GEOMETRY_CACHE.put(key, tables)
    return tables


def _edge_from_covered(covered: np.ndarray, tables: _FeatherTables) -> np.ndarray:
    """Content-edge angle per azimuth ray from a boolean coverage mask."""
    hole = ~np.take(np.ascontiguousarray(covered).reshape(-1), tables.ray_flat)
    hole[-1, :] = True  # the rim terminates every ray
    first = hole.argmax(axis=0)
    return tables.ray_theta_deg[first]


def _weights_cache_key(alpha_eye: np.ndarray, start_deg: float, end_deg: float) -> tuple:
    """Cheap mask digest: size, dtype, angles and a ≤65×65 sub-sample of the alpha.

    Deliberately *not* a hash of the whole plane (that alone costs a sizeable
    fraction of the work being saved).  The digest only has to reject
    different masks quickly; a match is always confirmed against the stored
    alpha with ``np.array_equal`` in :func:`_cached_weights`.
    """
    h, w = alpha_eye.shape
    step = max(1, max(h, w) // 64)
    digest = np.ascontiguousarray(alpha_eye[::step, ::step]).tobytes()
    return (h, w, alpha_eye.dtype.str, float(start_deg), float(end_deg), digest)


def _cached_weights(key: tuple, alpha_eye: np.ndarray) -> np.ndarray | None:
    """Weights previously computed for exactly this alpha plane, or ``None``."""
    entry = _WEIGHTS_CACHE.get(key)
    if entry is None:
        return None
    if np.array_equal(entry[0], alpha_eye):
        return entry[1]
    _WEIGHTS_CACHE.note_false_hit()  # digest collision: recompute, never trust it
    return None


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
    return _edge_from_covered(covered, _geometry_tables(h, w, n_psi))


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

    Issue #264: the size-only tables come from ``_GEOMETRY_CACHE`` and the
    final weights for an unchanged mask from ``_WEIGHTS_CACHE`` (see the
    section comment above ``_LRUCache``).  The result is bit-identical with or
    without a cache hit, and the returned array is always a fresh, writable
    copy that shares no memory with the caches.

    Args:
        alpha_eye: (H, W) alpha plane of one hemisphere (or (H, W, 4) RGBA).
        start_deg: Angle (0–180 FOV scale) where darkening starts.
        end_deg: Angle where the image is fully black.
    """
    if alpha_eye.ndim == 3:
        alpha_eye = alpha_eye[:, :, -1]
    h, w = alpha_eye.shape
    key = _weights_cache_key(alpha_eye, start_deg, end_deg)
    cached = _cached_weights(key, alpha_eye)
    if cached is not None:
        _GEOMETRY_CACHE.touch((h, w, _EDGE_AZIMUTH_SAMPLES))  # keep this size's tables warm too
        return cached.copy()

    covered = alpha_eye > 0
    if not covered.any():
        return np.zeros((h, w), dtype=np.float32)

    t = _geometry_tables(h, w, _EDGE_AZIMUTH_SAMPLES)
    edge = _edge_from_covered(covered, t)
    theta_edge = edge[t.i0] * t.omf + edge[t.i1] * t.frac

    px_deg = 180.0 / h  # angular size of one pixel at the centre
    s, e = start_deg / 2.0, end_deg / 2.0
    e_eff = np.minimum(e, theta_edge) - px_deg
    width = np.maximum(np.minimum(e - s, e_eff), 1e-6)
    weights = np.clip((e_eff - t.theta_p) / width, 0.0, 1.0).astype(np.float32)

    # Guard: anything touching the hole or the frame border goes fully black.
    inner = cv2.erode(
        covered.astype(np.uint8), np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0
    )
    weights[inner == 0] = 0.0

    stored = weights.copy()
    stored.setflags(write=False)
    _WEIGHTS_CACHE.put(key, (np.array(alpha_eye, copy=True), stored))
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
