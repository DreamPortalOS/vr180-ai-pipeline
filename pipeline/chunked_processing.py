"""Chunked sequence processing — long-video memory management (issue #37, V-4).

The non-streaming pipeline historically kept every frame of the video in RAM
(``list(read_frames(...))`` + per-stage result lists), so peak memory grew
linearly with video length (~98 GB for a long 8K clip). The streaming pipeline
(``pipeline/streaming_pipeline.py``) is O(1) but only covers the
depth→stereo→equirect "all" path; the staged / checkpoint / outpaint paths
still process whole sequences.

This module provides the shared chunking primitive used by those paths:

- ``iter_chunks`` — split a sequence into ``chunk_size`` windows with a
  leading/trailing ``overlap`` of context frames on each side.
- ``process_in_chunks`` — apply a per-frame function chunk-by-chunk and
  reassemble the results in original order.

For **per-frame functions** (depth estimation, stereo render, equirect map,
outpaint, upscale) chunked results are *bit-identical* to whole-sequence
processing — chunking only changes how many frames are alive in RAM at once,
never the values produced. For functions that need temporal context
(e.g. an EMA/temporal filter), the ``overlap`` frames provide the same
neighbourhood the whole-sequence run would have seen, so results stay
consistent across chunk boundaries.

Peak memory becomes proportional to ``chunk_size + 2 * overlap`` frames
instead of the total frame count.

Stages that **cannot** be chunked (they need the whole clip at once) are
documented in ``docs/DEV_GUIDE.md`` § 内存管理 with explicit upper-bound
estimates — see ``UNCHUNKABLE_STAGES`` below for the machine-readable list.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import TypeVar

log = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")

#: Default chunk size (frames) for the non-streaming pipeline stages.
#: 64 frames × 1920×1920×3 bytes ≈ 700 MB worst case — safe for 12 GB cards.
DEFAULT_CHUNK_SIZE = 64

#: Stages that need the entire clip in memory and cannot be chunked.
#: Each entry: stage name → why it is global + memory upper bound.
#: (Mirrored in docs/DEV_GUIDE.md — keep both in sync.)
UNCHUNKABLE_STAGES: dict[str, str] = {
    "depthcrafter": (
        "Temporal diffusion over the whole clip (DepthCrafter backend reads "
        "the full video file itself); RAM bound by the external process, "
        "not this pipeline. Upper bound ≈ backend's own VRAM/RAM usage "
        "(~10–16 GB at 1024px short side); wrapper holds only the returned "
        "depth list: N × H × W × 4 bytes (float32)."
    ),
    "stereocrafter": (
        "Video-diffusion inpainting needs global temporal context; runs as "
        "an external CLI on the whole video. Wrapper-side bound: input "
        "frame list N × H × W × 3 bytes plus the re-loaded L/R outputs "
        "2 × N × H × W × 3 bytes. Use --max-frames or the streaming path "
        "for long clips."
    ),
}


def iter_chunks(
    total: int,
    chunk_size: int,
    overlap: int = 0,
) -> Iterator[tuple[int, int, int, int]]:
    """Yield ``(core_start, core_end, ext_start, ext_end)`` windows.

    Each window covers the core range ``[core_start, core_end)`` — the frames
    whose results are kept — extended by up to ``overlap`` context frames on
    each side: ``[ext_start, ext_end)``. Context frames are clipped at the
    sequence boundaries, so the first chunk has no left context and the last
    chunk has no right context (same as whole-sequence processing).

    Args:
        total: Total number of frames (``>= 0``).
        chunk_size: Frames per core chunk (``>= 1``). Values ``>= total``
            yield a single chunk covering the whole sequence.
        overlap: Context frames on each side (``>= 0``).

    Yields:
        Tuples of ``(core_start, core_end, ext_start, ext_end)`` with
        ``ext_start = max(0, core_start - overlap)`` and
        ``ext_end = min(total, core_end + overlap)``.

    Raises:
        ValueError: If ``chunk_size < 1`` or ``overlap < 0``.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    if overlap < 0:
        raise ValueError(f"overlap must be >= 0, got {overlap}")

    start = 0
    while start < total:
        end = min(start + chunk_size, total)
        yield start, end, max(0, start - overlap), min(total, end + overlap)
        start = end


def process_in_chunks(
    items: Sequence[T],
    fn: Callable[[T], R],
    chunk_size: int | None = DEFAULT_CHUNK_SIZE,
    overlap: int = 0,
    desc: str | None = None,
) -> list[R]:
    """Apply *fn* to every item, processing ``chunk_size`` items at a time.

    Memory contract: at most one chunk's worth of *input* slices and *output*
    results are newly allocated per iteration; the returned list still holds
    all results (callers that need true O(chunk) output should consume the
    results incrementally — see ``process_in_chunks_iter``).

    For per-frame functions the result is identical to ``[fn(x) for x in
    items]`` for every ``chunk_size``/``overlap`` combination — this is the
    property the V-4 acceptance tests assert.

    Args:
        items: Full input sequence.
        fn: Per-item function.
        chunk_size: Chunk size in items; ``None`` disables chunking
            (single pass over the whole sequence).
        overlap: Context items made available around each chunk. Only
            relevant for stateful *fn* wrappers that read neighbours via
            ``iter_chunks`` directly; ``process_in_chunks`` itself is
            per-item, so overlap does not change its output.
        desc: Optional label for progress logging.

    Returns:
        List of results, same length and order as *items*.
    """
    return list(process_in_chunks_iter(items, fn, chunk_size=chunk_size, overlap=overlap, desc=desc))


def process_in_chunks_iter(
    items: Sequence[T],
    fn: Callable[[T], R],
    chunk_size: int | None = DEFAULT_CHUNK_SIZE,
    overlap: int = 0,
    desc: str | None = None,
) -> Iterator[R]:
    """Generator form of :func:`process_in_chunks` — yields results in order.

    Peak *extra* memory is one chunk of results at a time; the consumer
    decides what to keep (e.g. write each frame to disk / ffmpeg pipe and
    drop it). This is the primitive the pipeline stages use to keep RAM
    proportional to ``chunk_size`` instead of the video length.
    """
    total = len(items)
    if chunk_size is None:
        chunk_size = max(total, 1)

    n_chunks = (total + chunk_size - 1) // chunk_size if total else 0
    for idx, (core_start, core_end, _ext_start, _ext_end) in enumerate(iter_chunks(total, chunk_size, overlap)):
        if desc:
            log.info("%s: chunk %d/%d (frames %d–%d of %d)", desc, idx + 1, n_chunks, core_start, core_end - 1, total)
        for i in range(core_start, core_end):
            yield fn(items[i])


def chunked_windows(
    items: Sequence[T],
    chunk_size: int | None = DEFAULT_CHUNK_SIZE,
    overlap: int = 0,
) -> Iterator[tuple[Sequence[T], int, int]]:
    """Yield ``(window, core_start, core_end)`` slices including overlap context.

    Unlike :func:`process_in_chunks_iter` (per-item), this hands the caller
    the whole *window* (core + context) so stateful stages — temporal
    filters, EMA smoothers — can prime their state on the overlap frames and
    only emit results for ``items[core_start:core_end]``.

    Args:
        items: Full input sequence.
        chunk_size: Core window size; ``None`` = one window over everything.
        overlap: Context frames on each side of the core window.

    Yields:
        ``(window, core_start, core_end)`` where ``window`` is
        ``items[ext_start:ext_end]`` and the caller-owned results correspond
        to ``window[core_start - ext_start : core_end - ext_start]``.
    """
    total = len(items)
    if chunk_size is None:
        chunk_size = max(total, 1)
    for core_start, core_end, ext_start, ext_end in iter_chunks(total, chunk_size, overlap):
        yield items[ext_start:ext_end], core_start, core_end


def pairwise_in_chunks(
    left: Sequence[T],
    right: Sequence[T],
    fn: Callable[[T, T], R],
    chunk_size: int | None = DEFAULT_CHUNK_SIZE,
    desc: str | None = None,
) -> Iterator[R]:
    """Chunked variant of ``fn(l, r) for l, r in zip(left, right)``.

    Used by stages that consume two aligned sequences (stereo L/R frames,
    frame+depth pairs). Sequences must have equal length.
    """
    if len(left) != len(right):
        raise ValueError(f"Sequence length mismatch: {len(left)} vs {len(right)}")
    total = len(left)
    if chunk_size is None:
        chunk_size = max(total, 1)
    n_chunks = (total + chunk_size - 1) // chunk_size if total else 0
    for idx, (core_start, core_end, _es, _ee) in enumerate(iter_chunks(total, chunk_size)):
        if desc:
            log.info("%s: chunk %d/%d (frames %d–%d of %d)", desc, idx + 1, n_chunks, core_start, core_end - 1, total)
        for i in range(core_start, core_end):
            yield fn(left[i], right[i])


def write_png_sequence(
    frames: Iterable,
    out_dir: str,
    prefix: str,
    start_index: int = 0,
    to_bgr: bool = True,
) -> int:
    """Write frames as ``{prefix}_{i:06d}.png`` and return the next free index.

    Shared helper for chunked stages that checkpoint every frame to disk
    (the pipeline's existing resume mechanism) without keeping them in RAM.
    """
    import os

    import cv2
    import numpy as np

    idx = start_index
    for frame in frames:
        img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if to_bgr else np.asarray(frame)
        cv2.imwrite(os.path.join(out_dir, f"{prefix}_{idx:06d}.png"), img)
        idx += 1
    return idx
