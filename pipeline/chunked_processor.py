"""
Chunked frame-sequence processor (V-4 — issue #37)
===================================================

The non-streaming (batch) pipeline historically holds *all* frames of a long
clip in RAM simultaneously — ``frames``, ``depths``, ``left_frames``,
``right_frames`` and ``sbs_frames`` all accumulate as Python lists.  For a long
8K source that is tens of GB and the process OOMs.

The streaming path (:mod:`pipeline.streaming_pipeline`) is already O(1) memory
but only covers the fused depth→stereo→project→encode run; the checkpointed
batch stages (``--stage depth|stereo|equirect|…``) still buffer everything.

This module provides the **memory-bounded chunking primitive** shared by the
batch stages:

* :func:`chunk_ranges` — split ``[0, n)`` into overlapping windows.  Peak RAM
  is proportional to ``chunk_size`` (+``overlap`` warmup), **independent of
  clip length**.
* :func:`process_in_chunks` — drive a per-frame callable over the chunks,
  replaying *temporal state* across chunk boundaries so the chunked result is
  **bit-for-bit identical to processing the whole sequence at once**
  (the V-4 acceptance bar).

Stages with true global semantics (single-pass ffmpeg ``v360`` over the whole
sequence, DepthCrafter/StereoCrafter whole-video inference, the final
metadata encode) cannot be chunked without changing their output — those are
documented in their call sites with an explicit RAM upper-bound estimate.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import TypeVar

T = TypeVar("T")

__all__ = ["chunk_ranges", "default_chunk_size", "process_in_chunks"]


def default_chunk_size() -> int:
    """Default chunk size (frames) used when a caller does not override it.

    Chosen so that a 4K-per-eye SBS chunk (≈7680×3840×3 uint8 ≈ 84 MB/frame,
    two eyes + intermediates) stays well under 1 GB per chunk on a 12 GB GPU
    machine while keeping per-chunk ffmpeg/overhead amortisable.  16 frames →
    ≈1.3 GB worst case for the SBS-tier; safe for the 12 GB RTX 4070 SUPER.
    """
    return 16


def chunk_ranges(
    n: int,
    chunk_size: int,
    overlap: int = 0,
) -> Iterator[tuple[int, int, int]]:
    """Yield ``(emit_start, emit_end, warm_start)`` index windows over ``[0, n)``.

        Each window covers the half-open emit range ``[emit_start, emit_end)`` —
        the frames whose *output* is kept — preceded by a **warmup** prefix
        ``[warm_start, emit_start)`` of length ``≤ overlap`` that is processed
        *only to rebuild temporal state* and whose output is discarded.

    Guarantees (the V-4 correctness contract):
          * every index in ``[0, n)`` is emitted by exactly one chunk;
          * for a chunk whose ``emit_start > 0``, ``warm_start = max(0,
            emit_start - overlap)`` — so the last ``overlap`` frames before the
            emit boundary are replayed, rebuilding a bounded temporal context
            (a window-W filter needs ``overlap ≥ W-1``: warmup frames + the first
            emitted frame form a full window) to the state the whole-sequence run
            would have at that boundary;
          * with ``overlap == 0`` the warmup is empty and chunking is a plain
            partition — output still matches whole-sequence for *stateless*
            per-frame callables (depth without EMA, equirect per-frame, outpaint),
            and for *stateful* callables whose state survives across chunks;
          * ``chunk_size == 1`` ⇒ one emitted frame per chunk (warmup still
            applies, so temporal state is replayed frame-by-frame — output matches
            the whole-sequence run exactly, even when the per-chunk callable
            re-instantiates the stage);
          * ``chunk_size >= n`` ⇒ a single chunk covering everything (identical to
            no chunking at all).

        Args:
            n: Total frame count (must be ≥ 0).
            chunk_size: Max emitted (kept) frames per chunk (≥ 1).
            overlap: Warmup frames before each non-first chunk (≥ 0).  No upper
                bound is enforced: a large overlap simply replays more history
                (the emit window still advances by ``chunk_size ≥ 1`` each step,
                so the loop always terminates).

        Yields:
            ``(emit_start, emit_end, warm_start)`` tuples.  ``warm_start ==
            emit_start`` for the first chunk (no warmup before frame 0).

        Raises:
            ValueError: If ``n < 0``, ``chunk_size < 1`` or ``overlap < 0``.
    """
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    if overlap < 0:
        raise ValueError(f"overlap must be >= 0, got {overlap}")

    emit_start = 0
    first = True
    while emit_start < n:
        emit_end = min(emit_start + chunk_size, n)
        # The first chunk has nothing before it to warm up from.  Every
        # subsequent chunk replays the last `overlap` emitted frames to
        # rebuild temporal state to exactly the value the whole-sequence run
        # would hold at `emit_start`.
        warm_start = emit_start if first else max(0, emit_start - overlap)
        yield (emit_start, emit_end, warm_start)
        emit_start = emit_end
        first = False


def process_in_chunks(
    frames: Sequence | Iterable,
    process_fn: Callable[[list, int, int], Iterator[T]],
    *,
    chunk_size: int | None = None,
    overlap: int = 0,
) -> Iterator[T]:
    """Apply a per-chunk ``process_fn`` over ``frames`` in memory-bounded chunks.

    The result stream is **identical to processing the whole sequence at once**
    — chunking is purely a memory optimisation and must not change output.
    This is the V-4 acceptance criterion.  Two scenarios satisfy it:

    * **Stateless stages** (equirect per-frame, outpaint, depth-without-EMA):
      ``overlap == 0`` suffices — each frame is independent.
    * **Stateful stages re-instantiated per chunk** (the realistic batch case:
      ``run_stereo_stage`` builds a fresh ``StereoRenderer`` per call, so
      ``_prev_disparity`` resets at every chunk boundary): set ``overlap ≥ 1``
      (≥ the IIR filter's effective memory) so the warmup prefix rebuilds the
      temporal state to exactly what the whole-sequence run would hold at the
      emit boundary.  A stage whose state *survives across chunks* (a single
      instance reused for every chunk) needs no overlap — its state is already
      continuous — but the per-chunk-reinstantiation pattern is the one the
      batch stages actually use, so ``overlap`` is the documented safety knob.

    ``process_fn`` is called once per chunk as
    ``process_fn(chunk_frames, warm_offset, emit_offset)`` where:

      * ``chunk_frames`` — the slice covering ``[warm_start, emit_end)`` (warmup
        prefix + emit window);
      * ``warm_offset`` — number of leading warmup frames in ``chunk_frames``
        whose output must be **discarded** (state rebuilt only);
      * ``emit_offset`` — number of frames in ``chunk_frames`` whose output
        must be **kept** (== ``emit_end - emit_start``).

    So ``len(chunk_frames) == warm_offset + emit_offset``.  ``process_fn``
    must yield exactly ``emit_offset`` outputs (one per emitted frame, in
    order) — it processes the warmup frames for their side effect on temporal
    state (e.g. ``StereoRenderer._prev_disparity``) but yields nothing for
    them.

    ``frames`` may be a random-access ``Sequence`` (list) **or** a lazy
    ``Iterable`` (generator).  In the lazy case the total frame count is
    discovered by streaming; a circular buffer keeps at most
    ``chunk_size + overlap`` frames resident at once, so peak memory is
    bounded by the chunk, not by clip length (V-4.1a).

    Args:
        frames: Input frames — a ``Sequence`` or a lazy ``Iterable``/generator.
            Read-only here; never fully materialised when an ``Iterable``.
        process_fn: Per-chunk callable described above.
        chunk_size: Emitted frames per chunk (default
            :func:`default_chunk_size`).  ``None`` ⇒ default.
        overlap: Warmup frame count for temporal-state replay (default 0 —
            use ``≥ 1`` for stages with IIR temporal filters).

    Yields:
        ``process_fn`` outputs for every emitted frame, in order.
    """
    if chunk_size is None:
        chunk_size = default_chunk_size()

    if _is_sequence(frames):
        return _process_in_chunks_seq(frames, process_fn, chunk_size=chunk_size, overlap=overlap)
    return _process_in_chunks_lazy(frames, process_fn, chunk_size=chunk_size, overlap=overlap)


def _is_sequence(obj: Sequence | Iterable) -> bool:
    """True when *obj* supports random access (``__len__`` + ``__getitem__``).

    ``str``/``bytes`` are excluded (never valid frame streams). Generators and
    other iterables lacking ``__len__`` return False.
    """
    if isinstance(obj, (str, bytes)):
        return False
    return hasattr(obj, "__len__") and hasattr(obj, "__getitem__")


def _process_in_chunks_seq(
    frames: Sequence,
    process_fn: Callable[[list, int, int], Iterator[T]],
    *,
    chunk_size: int,
    overlap: int,
) -> Iterator[T]:
    """Random-access (list) path — original behaviour, unchanged."""
    n = len(frames)
    for emit_start, emit_end, warm_start in chunk_ranges(n, chunk_size, overlap):
        chunk = list(frames[warm_start:emit_end])
        warm_offset = emit_start - warm_start
        emit_offset = emit_end - emit_start
        yield from process_fn(chunk, warm_offset, emit_offset)


def _process_in_chunks_lazy(
    frames: Iterable,
    process_fn: Callable[[list, int, int], Iterator[T]],
    *,
    chunk_size: int,
    overlap: int,
) -> Iterator[T]:
    """Lazy (generator) path — bounded circular buffer, single pass, no full materialisation.

    A single streaming pass over the source both counts frames (to drive the
    chunk window boundaries) and maintains a circular ``deque`` that holds at
    most ``chunk_size + overlap`` frames.  As each emitted frame arrives, the
    buffer is kept to exactly the current window ``[warm_start .. emit_end)``
    (warmup + emit); the oldest frames are popped from the head, so at every
    moment at most ``chunk_size + overlap`` frames are resident.

    Because a generator is single-use, we cannot count first and replay — the
    windows are emitted in a single forward pass.  The first ``overlap`` frames
    before ``chunk_size`` cannot form a full emit window, so they are held as
    the initial warmup buffer and only ``process_fn`` is called once the first
    full emit window ``[0, chunk_size)`` is complete; subsequent windows emit
    as soon as ``emit_end`` is reached.
    """
    it = iter(frames)

    buf: deque = deque(maxlen=chunk_size + overlap)
    idx = 0  # next frame index to read from the source (0-based)
    emit_start = 0  # start of the current emit window [emit_start, emit_end)
    first = True

    while True:
        warm_start = emit_start if first else max(0, emit_start - overlap)

        # Pull frames until we have filled the current emit window.
        pulled = False
        try:
            while True:
                frame = next(it)
                buf.append(frame)
                idx += 1
                pulled = True

                # Once the buffer holds frames through this new ``idx-1``, check
                # whether the current emit window [emit_start, emit_end) is full.
                # emit_end = min(emit_start + chunk_size, idx).  The window is
                # full when idx reaches emit_start + chunk_size OR the source
                # is exhausted (handled by the StopIteration below).
                target = emit_start + chunk_size
                if idx >= target:
                    break
        except StopIteration:
            pass  # source exhausted — emit whatever remains, then stop.

        if not pulled and idx <= emit_start:
            # Source ended before we even reached the current window — done.
            break

        emit_end = min(emit_start + chunk_size, idx)
        if emit_end <= emit_start:
            # Nothing new to emit this cycle.
            first = False
            continue

        # Trim head so buffer covers exactly [warm_start, emit_end).
        while buf and (idx - len(buf)) < warm_start:
            buf.popleft()

        chunk = list(buf)
        warm_offset = emit_start - warm_start
        emit_offset = emit_end - emit_start
        yield from process_fn(chunk, warm_offset, emit_offset)

        emit_start = emit_end
        first = False
        if idx < emit_start:
            # Source exhausted before the next window starts.
            break
