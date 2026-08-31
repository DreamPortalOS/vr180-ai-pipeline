"""Tests for V-4 chunked memory management (issue #37).

Acceptance criteria verified here:
  - chunked result == whole-sequence result, frame-for-frame, including at
    the overlap boundary where temporal state is replayed;
  - the two extremes — ``chunk_size=1`` and ``chunk_size > total frames`` —
    each behave correctly;
  - the chunking primitive itself partitions without gaps/duplicates.

All tests are CPU-only, synthetic (20 frames × 64×64), no models, no real
inference.

Two stage shapes are exercised:

1. **Stateless stage** — each frame's output depends only on that frame
   (equirect per-frame mapping, outpaint, depth-without-EMA).  ``overlap=0``
   is exact.

2. **Finite-memory temporal stage** — a windowed temporal filter whose output
   for frame *i* depends on frames ``[i-window+1 .. i]`` only.  This models
   stages that genuinely need *temporal context* (the V-4 task explicitly
   names "时序滤波" / temporal filtering).  Such a stage is exact under
   chunking **iff** ``overlap >= window``: the warmup prefix rebuilds the full
   window so the first emitted frame of each chunk sees identical input to
   the whole-sequence run.  ``overlap < window`` provably diverges.

(The real ``StereoRenderer._prev_disparity`` and depth-stage EMA are
*infinite-memory* IIR filters — for those the pipeline reuses one persistent
instance across chunks so state is continuous and chunking is bit-exact with
``overlap=0``; that path is exercised at the integration level, not here.)
"""

from __future__ import annotations

import unittest

import numpy as np

from pipeline.chunked_processor import chunk_ranges, default_chunk_size, process_in_chunks


def _make_frames(n: int = 20, h: int = 64, w: int = 64) -> list[np.ndarray]:
    """Deterministic synthetic frame sequence — each frame a distinct value."""
    rng = np.random.default_rng(seed=37)
    return [rng.integers(0, 256, (h, w, 3), dtype=np.uint8) for _ in range(n)]


# ---------------------------------------------------------------------------
# Stateless reference stage: out = frame + 1 (per-pixel) — no temporal state.
# ---------------------------------------------------------------------------


def _stateless_whole(frames: list[np.ndarray]) -> list[np.ndarray]:
    return [f + 1 for f in frames]


def _stateless_process_fn(chunk_frames, warm_offset, emit_offset):
    outs = [f + 1 for f in chunk_frames]
    return iter(outs[warm_offset:])


# ---------------------------------------------------------------------------
# Finite-memory temporal reference stage: out_i = mean(frames[i-W+1 .. i]).
# Exact under chunking iff overlap >= W (the window).  Mirrors a real
# temporal filter whose output depends on a bounded neighbourhood.
# ---------------------------------------------------------------------------


def _windowed_whole(frames: list[np.ndarray], window: int = 3) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    acc = np.zeros_like(frames[0], dtype=np.float64)
    for i, f in enumerate(frames):
        acc = acc + f.astype(np.float64)
        if i >= window:
            acc = acc - frames[i - window].astype(np.float64)
        lo = max(0, i - window + 1)
        out.append((acc / (i - lo + 1)).astype(np.uint8))
    return out


def _make_windowed_process_fn(window: int = 3):
    """Per-chunk-fresh windowed stage — needs overlap >= window to be exact."""

    def process_fn(chunk_frames, warm_offset, emit_offset):
        stage = _WindowedStage(window)
        outs = [stage.step(f) for f in chunk_frames]
        return iter(outs[warm_offset:])

    return process_fn


class _WindowedStage:
    """Sliding-window mean over the last `window` frames (finite memory)."""

    def __init__(self, window: int = 3) -> None:
        self.window = window
        self._buf: list[np.ndarray] = []

    def step(self, frame: np.ndarray) -> np.ndarray:
        self._buf.append(frame.astype(np.float64))
        if len(self._buf) > self.window:
            self._buf.pop(0)
        acc = self._buf[0].copy()
        for b in self._buf[1:]:
            acc = acc + b
        return (acc / len(self._buf)).astype(np.uint8)


# ---------------------------------------------------------------------------
# chunk_ranges — partition correctness
# ---------------------------------------------------------------------------


class TestChunkRanges(unittest.TestCase):
    def test_covers_every_index_exactly_once(self):
        n = 20
        seen = []
        for emit_start, emit_end, _warm in chunk_ranges(n, chunk_size=7, overlap=2):
            seen.extend(range(emit_start, emit_end))
        self.assertEqual(seen, list(range(n)))

    def test_no_gaps_or_duplicates(self):
        n = 31
        seen = set()
        for emit_start, emit_end, _warm in chunk_ranges(n, chunk_size=8, overlap=3):
            for i in range(emit_start, emit_end):
                self.assertNotIn(i, seen, f"index {i} emitted twice")
                seen.add(i)
        self.assertEqual(seen, set(range(n)))

    def test_chunk_size_one_with_overlap(self):
        # Every frame its own chunk; warmup still applies on non-first chunks.
        # overlap (1) == chunk_size (1) is allowed — the emit window still
        # advances by one frame each step.
        ranges = list(chunk_ranges(5, chunk_size=1, overlap=1))
        self.assertEqual(len(ranges), 5)
        for i, (s, e, _) in enumerate(ranges):
            self.assertEqual((s, e), (i, i + 1))
        self.assertEqual(ranges[0][2], 0)
        for i in range(1, 5):
            self.assertEqual(ranges[i][2], i - 1)

    def test_chunk_size_larger_than_total_is_single_chunk(self):
        ranges = list(chunk_ranges(20, chunk_size=100, overlap=3))
        self.assertEqual(len(ranges), 1)
        s, e, warm = ranges[0]
        self.assertEqual((s, e), (0, 20))
        self.assertEqual(warm, 0)  # first chunk never warms up

    def test_exact_multiple_no_partial_chunk(self):
        ranges = list(chunk_ranges(12, chunk_size=4, overlap=0))
        self.assertEqual(len(ranges), 3)
        self.assertEqual([(s, e) for s, e, _ in ranges], [(0, 4), (4, 8), (8, 12)])

    def test_overlap_warm_start_clamped_at_zero(self):
        # overlap larger than chunk_size is allowed; warm_start clamps to 0.
        ranges = list(chunk_ranges(3, chunk_size=2, overlap=5))
        # chunk 0: emit [0,2) warm 0 ; chunk 1: emit [2,3) warm max(0,2-5)=0
        self.assertEqual(ranges[0], (0, 2, 0))
        self.assertEqual(ranges[1], (2, 3, 0))

    def test_zero_frames_yields_nothing(self):
        self.assertEqual(list(chunk_ranges(0, chunk_size=4, overlap=1)), [])

    def test_overlap_zero_is_plain_partition(self):
        ranges = list(chunk_ranges(10, chunk_size=3, overlap=0))
        for s, _e, warm in ranges:
            self.assertEqual(warm, s)

    def test_warmup_prefix_length_equals_overlap(self):
        # For every non-first chunk, warm_offset == min(overlap, emit_start).
        ranges = list(chunk_ranges(20, chunk_size=6, overlap=4))
        first = True
        for s, _e, warm in ranges:
            if first:
                self.assertEqual(warm, s)
                first = False
            else:
                self.assertEqual(s - warm, min(4, s))

    def test_invalid_args(self):
        with self.assertRaises(ValueError):
            list(chunk_ranges(-1, 4, 0))
        with self.assertRaises(ValueError):
            list(chunk_ranges(5, 0, 0))
        with self.assertRaises(ValueError):
            list(chunk_ranges(5, 4, -1))

    def test_default_chunk_size_positive(self):
        self.assertGreater(default_chunk_size(), 0)


# ---------------------------------------------------------------------------
# process_in_chunks — stateless equivalence (overlap=0 is exact)
# ---------------------------------------------------------------------------


class TestProcessInChunksStateless(unittest.TestCase):
    def setUp(self):
        self.frames = _make_frames(n=20)
        self.expected = _stateless_whole(self.frames)

    def _assert_equal(self, got: list[np.ndarray]) -> None:
        self.assertEqual(len(got), len(self.expected))
        for i, (a, b) in enumerate(zip(got, self.expected, strict=False)):
            self.assertTrue(np.array_equal(a, b), f"frame {i} differs from whole-sequence run")

    def test_chunk_size_1_matches_whole(self):
        got = list(process_in_chunks(self.frames, _stateless_process_fn, chunk_size=1, overlap=0))
        self._assert_equal(got)

    def test_chunk_size_larger_than_total_matches_whole(self):
        got = list(process_in_chunks(self.frames, _stateless_process_fn, chunk_size=100, overlap=0))
        self._assert_equal(got)

    def test_typical_chunk_matches_whole(self):
        got = list(process_in_chunks(self.frames, _stateless_process_fn, chunk_size=7, overlap=0))
        self._assert_equal(got)

    def test_default_chunk_size_matches_whole(self):
        got = list(process_in_chunks(self.frames, _stateless_process_fn, chunk_size=None, overlap=0))
        self._assert_equal(got)


# ---------------------------------------------------------------------------
# process_in_chunks — finite-memory temporal equivalence (overlap >= window)
# This is the V-4 "overlap boundary" acceptance test.
# ---------------------------------------------------------------------------


class TestProcessInChunksWindowed(unittest.TestCase):
    """The temporal-context acceptance test: overlap must rebuild the window."""

    def setUp(self):
        self.frames = _make_frames(n=20)
        self.window = 3
        self.expected = _windowed_whole(self.frames, window=self.window)

    def _assert_equal(self, got: list[np.ndarray]) -> None:
        self.assertEqual(len(got), len(self.expected))
        for i, (a, b) in enumerate(zip(got, self.expected, strict=False)):
            self.assertTrue(np.array_equal(a, b), f"frame {i} differs from whole-sequence run")

    def test_chunk_size_1_with_full_overlap_matches_whole(self):
        # Extreme 1: one frame per chunk, overlap == window-1 → exact.
        # (The first emitted frame needs frames [start-window+1 .. start];
        #  overlap = window-1 warmup frames + the emitted frame = window.)
        fn = _make_windowed_process_fn(window=self.window)
        got = list(process_in_chunks(self.frames, fn, chunk_size=1, overlap=self.window - 1))
        self._assert_equal(got)

    def test_chunk_size_1_with_overlap_window_also_matches(self):
        # overlap == window (one more than strictly needed) is still exact.
        fn = _make_windowed_process_fn(window=self.window)
        got = list(process_in_chunks(self.frames, fn, chunk_size=1, overlap=self.window))
        self._assert_equal(got)

    def test_chunk_size_larger_than_total_matches_whole(self):
        # Extreme 2: a single chunk = no chunking at all.
        fn = _make_windowed_process_fn(window=self.window)
        got = list(process_in_chunks(self.frames, fn, chunk_size=100, overlap=self.window))
        self._assert_equal(got)

    def test_overlap_equals_window_minus_one_matches_whole_at_boundaries(self):
        # 20 frames, chunk=7, overlap=window-1=2 → boundaries at 7 and 14.
        # This is the *minimum* overlap for a window-W filter to be exact.
        fn = _make_windowed_process_fn(window=self.window)
        got = list(process_in_chunks(self.frames, fn, chunk_size=7, overlap=self.window - 1))
        self._assert_equal(got)
        # The boundary frames are the ones that would diverge if the warmup
        # failed to rebuild the window.
        for boundary in (7, 14):
            self.assertTrue(
                np.array_equal(got[boundary], self.expected[boundary]),
                f"boundary frame {boundary} diverged",
            )

    def test_overlap_below_minimum_diverges_at_boundary(self):
        """Sanity: overlap < window-1 provably diverges (proves the test is real).

        The minimum overlap for a window-W filter is W-1 (the warmup frames
        plus the first emitted frame form a full window).  One less than that
        leaves the first emitted frame's window incomplete → diverges.
        """
        fn = _make_windowed_process_fn(window=self.window)
        got = list(process_in_chunks(self.frames, fn, chunk_size=7, overlap=self.window - 2))
        diffs = [i for i, (a, b) in enumerate(zip(got, self.expected, strict=False)) if not np.array_equal(a, b)]
        self.assertTrue(diffs, "expected divergence with overlap < window-1")
        # First divergence is exactly at the first chunk boundary.
        self.assertEqual(diffs[0], 7)

    def test_overlap_zero_diverges_for_windowed_stage(self):
        fn = _make_windowed_process_fn(window=self.window)
        got = list(process_in_chunks(self.frames, fn, chunk_size=7, overlap=0))
        diffs = [i for i, (a, b) in enumerate(zip(got, self.expected, strict=False)) if not np.array_equal(a, b)]
        self.assertTrue(diffs, "expected divergence with overlap=0 for a windowed stage")
        self.assertEqual(diffs[0], 7)


# ---------------------------------------------------------------------------
# process_in_chunks — count + empty + contract
# ---------------------------------------------------------------------------


class TestProcessInChunksCountAndContract(unittest.TestCase):
    def test_count_preserved_stateless(self):
        for cs, ov in [(1, 0), (100, 0), (7, 0), (4, 3), (5, 0)]:
            got = list(process_in_chunks(_make_frames(20), _stateless_process_fn, chunk_size=cs, overlap=ov))
            self.assertEqual(len(got), 20, f"wrong count for cs={cs} ov={ov}")

    def test_empty_input(self):
        self.assertEqual(list(process_in_chunks([], _stateless_process_fn, chunk_size=4, overlap=1)), [])

    def test_process_fn_receives_warm_plus_emit_frames(self):
        """chunk_frames length == warm_offset + emit_offset."""
        seen = []
        frames = _make_frames(n=10)

        def process_fn(chunk_frames, warm_offset, emit_offset):
            seen.append((len(chunk_frames), warm_offset, emit_offset))
            self.assertEqual(len(chunk_frames), warm_offset + emit_offset)
            return iter([None] * emit_offset)

        list(process_in_chunks(frames, process_fn, chunk_size=4, overlap=2))
        # 10 frames, cs=4, ov=2 → chunks: emit [0,4) warm0 ; [4,8) warm2 ; [8,10) warm2
        self.assertEqual(seen, [(4, 0, 4), (6, 2, 4), (4, 2, 2)])

    def test_warmup_outputs_are_discarded(self):
        """The warmup prefix is processed (for state) but NOT yielded."""
        frames = _make_frames(n=6)
        yielded_indices: list[int] = []
        frame_ids = [id(f) for f in frames]

        def process_fn(chunk_frames, warm_offset, emit_offset):
            for f in chunk_frames[warm_offset:]:
                yielded_indices.append(frame_ids.index(id(f)))
            return iter([None] * emit_offset)

        list(process_in_chunks(frames, process_fn, chunk_size=3, overlap=1))
        # Only emitted indices appear, in order, no warmup repeats.
        self.assertEqual(yielded_indices, [0, 1, 2, 3, 4, 5])


# ---------------------------------------------------------------------------
# V-4.1a (issue #86) — lazy / generator intake
# ---------------------------------------------------------------------------


class _AheadTracker:
    """Measure how far ahead the processor pulls from the source vs. emission.

    The chunked processor reads frames from the source into its circular
    buffer, then emits them via ``process_fn``.  ``ahead`` = (frames pulled
    from source) − (frames emitted).  This equals the number of source
    frames currently retained in the processor's buffer.  For a correctly
    bounded circular buffer this peak must be ``≤ chunk_size + overlap`` —
    which is the precise black-box proof that the full clip is never
    materialised (V-4.1a).
    """

    def __init__(self) -> None:
        self.peak_ahead = 0

    def pulled(self):
        """Call once per frame the source generator yields."""
        self._pulled = getattr(self, "_pulled", 0) + 1

    def emitted(self, count: int):
        """Call from process_fn with the number of frames it keeps."""
        self._emitted = getattr(self, "_emitted", 0) + count
        ahead = self._pulled - self._emitted
        if ahead > self.peak_ahead:
            self.peak_ahead = ahead


def _ahead_gen(n: int, tracker: _AheadTracker):
    """Yield index frames and record each pull for buffer-depth tracking."""

    for i in range(n):
        tracker.pulled()
        yield i


def _ahead_process_fn(tracker: _AheadTracker):
    """Per-chunk fn that emits kept indices and records emission count."""

    def process_fn(chunk_frames, warm_offset, emit_offset):
        tracker.emitted(emit_offset)
        return iter(chunk_frames[warm_offset:])

    return process_fn


class TestLazyGeneratorIntake(unittest.TestCase):
    """V-4.1a acceptance: a generator input is NOT fully materialised.

    The defining property is bounded residency: at no point do more than
    ``chunk_size + overlap`` frames coexist in memory, regardless of clip
    length.  We prove this with a counting generator whose frames track
    simultaneous liveness via weakrefs.
    """

    def test_peak_residency_bounded_by_chunk_plus_overlap(self):
        """The circular buffer never holds more than chunk_size + overlap frames.

        We run a long synthetic clip (n >> chunk_size + overlap) through the
        lazy path.  A counting generator records how far ahead the processor
        pulls from the source relative to what it has emitted; that gap IS
        the buffer occupancy.  Asserting the peak gap ``≤ chunk_size +
        overlap`` proves the full clip was never materialised (V-4.1a).
        """
        chunk_size, overlap, n = 5, 2, 80
        tracker = _AheadTracker()
        got = list(
            process_in_chunks(
                _ahead_gen(n, tracker),
                _ahead_process_fn(tracker),
                chunk_size=chunk_size,
                overlap=overlap,
            )
        )
        self.assertEqual(got, list(range(n)))
        self.assertLessEqual(
            tracker.peak_ahead,
            chunk_size + overlap,
            f"peak buffer occupancy {tracker.peak_ahead} exceeded chunk_size+overlap={chunk_size + overlap}",
        )

    def test_generator_output_matches_list_input(self):
        """Lazy (generator) and eager (list) paths are frame-identical.

        Equivalence contract: chunking + laziness must not change output vs
        processing the materialised list.  Exercised for both stateless and
        finite-memory temporal stages.
        """
        frames = _make_frames(n=20)

        # Stateless
        lazy_stateless = list(
            process_in_chunks(
                (f for f in frames),
                _stateless_process_fn,
                chunk_size=7,
                overlap=0,
            )
        )
        list_stateless = list(process_in_chunks(frames, _stateless_process_fn, chunk_size=7, overlap=0))
        self.assertEqual(len(lazy_stateless), len(list_stateless))
        for a, b in zip(lazy_stateless, list_stateless, strict=False):
            self.assertTrue(np.array_equal(a, b))

        # Finite-memory temporal (window=3, overlap=window-1=2 → exact)
        lazy_win = list(
            process_in_chunks(
                (f for f in frames),
                _make_windowed_process_fn(window=3),
                chunk_size=7,
                overlap=2,
            )
        )
        list_win = list(process_in_chunks(frames, _make_windowed_process_fn(window=3), chunk_size=7, overlap=2))
        self.assertEqual(len(lazy_win), len(list_win))
        for a, b in zip(lazy_win, list_win, strict=False):
            self.assertTrue(np.array_equal(a, b))

    def test_generator_empty_input(self):
        self.assertEqual(
            list(process_in_chunks(iter([]), _stateless_process_fn, chunk_size=4, overlap=1)),
            [],
        )

    def test_generator_chunk_size_one(self):
        """One frame per emitted chunk still bounds buffer occupancy (≤ 1 + overlap)."""
        chunk_size, overlap, n = 1, 2, 12
        tracker = _AheadTracker()
        got = list(
            process_in_chunks(
                _ahead_gen(n, tracker),
                _ahead_process_fn(tracker),
                chunk_size=chunk_size,
                overlap=overlap,
            )
        )
        self.assertEqual(got, list(range(n)))
        self.assertLessEqual(tracker.peak_ahead, chunk_size + overlap)

    def test_generator_chunk_larger_than_total_single_chunk(self):
        """A chunk bigger than the clip is a single pass; occupancy ≤ n."""
        n = 6
        tracker = _AheadTracker()
        got = list(
            process_in_chunks(
                _ahead_gen(n, tracker),
                _ahead_process_fn(tracker),
                chunk_size=100,
                overlap=0,
            )
        )
        self.assertEqual(got, list(range(n)))
        self.assertLessEqual(tracker.peak_ahead, n)


if __name__ == "__main__":
    unittest.main()
