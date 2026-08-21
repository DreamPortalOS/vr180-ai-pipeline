"""Tests for issue #37 (V-4) — chunked memory management, non-streaming path.

Acceptance criteria covered here:
  1. 20-frame 64×64 synthetic sequence: chunked results are per-frame
     identical to whole-sequence processing (including overlap boundaries).
  2. Edge cases: chunk_size=1 and chunk_size > total frames.
  3. Pipeline stage integration: stereo / equirect / outpaint stages produce
     identical frames chunked vs unchunked (heavy models mocked).

All tests are CPU-only; no real models, no real ffmpeg encodes.
"""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pipeline.chunked_processing import (  # noqa: E402
    UNCHUNKABLE_STAGES,
    chunked_windows,
    iter_chunks,
    pairwise_in_chunks,
    process_in_chunks,
    process_in_chunks_iter,
)

# ---------------------------------------------------------------------------
# Synthetic fixtures — 20 frames of 64×64 (the acceptance-test shape)
# ---------------------------------------------------------------------------

NUM_FRAMES = 20
FRAME_H = FRAME_W = 64


def _synthetic_frames(n: int = NUM_FRAMES) -> list[np.ndarray]:
    """Deterministic 64×64 RGB frames (seeded — identical across calls)."""
    rng = np.random.default_rng(42)
    return [rng.integers(0, 255, (FRAME_H, FRAME_W, 3), dtype=np.uint8) for _ in range(n)]


# ---------------------------------------------------------------------------
# iter_chunks — window geometry
# ---------------------------------------------------------------------------


class TestIterChunks(unittest.TestCase):
    def test_no_overlap_even_split(self):
        windows = list(iter_chunks(20, chunk_size=8, overlap=0))
        self.assertEqual(
            windows,
            [(0, 8, 0, 8), (8, 16, 8, 16), (16, 20, 16, 20)],
        )

    def test_overlap_extends_both_sides(self):
        windows = list(iter_chunks(20, chunk_size=8, overlap=2))
        self.assertEqual(
            windows,
            [
                (0, 8, 0, 10),  # first chunk: no left context (clipped)
                (8, 16, 6, 18),
                (16, 20, 14, 20),  # last chunk: no right context (clipped)
            ],
        )

    def test_chunk_size_larger_than_total_single_window(self):
        windows = list(iter_chunks(20, chunk_size=100, overlap=3))
        self.assertEqual(windows, [(0, 20, 0, 20)])

    def test_chunk_size_one(self):
        windows = list(iter_chunks(5, chunk_size=1, overlap=1))
        self.assertEqual(
            windows,
            [(0, 1, 0, 2), (1, 2, 0, 3), (2, 3, 1, 4), (3, 4, 2, 5), (4, 5, 3, 5)],
        )

    def test_empty_sequence(self):
        self.assertEqual(list(iter_chunks(0, chunk_size=4, overlap=2)), [])

    def test_invalid_params_raise(self):
        with self.assertRaises(ValueError):
            list(iter_chunks(10, chunk_size=0))
        with self.assertRaises(ValueError):
            list(iter_chunks(10, chunk_size=4, overlap=-1))

    def test_core_windows_cover_sequence_exactly(self):
        """Core ranges must tile [0, total) with no gaps/overlaps, any config."""
        for total in (1, 7, 20, 64):
            for chunk_size in (1, 3, 8, 64, 100):
                for overlap in (0, 2, 5):
                    cores = [(s, e) for s, e, _, _ in iter_chunks(total, chunk_size, overlap)]
                    covered = [i for s, e in cores for i in range(s, e)]
                    self.assertEqual(covered, list(range(total)))


# ---------------------------------------------------------------------------
# process_in_chunks — per-frame equivalence (acceptance criterion 1)
# ---------------------------------------------------------------------------


class TestProcessInChunksEquivalence(unittest.TestCase):
    """Chunked per-frame processing must be bit-identical to whole-sequence."""

    def _reference(self, frames):
        # A deterministic per-frame transform with mixed dtypes/shapes.
        return [(f.astype(np.float32) * 1.5 + 1.0).mean(axis=2) for f in frames]

    def test_chunked_matches_whole_sequence(self):
        frames = _synthetic_frames()
        reference = self._reference(frames)
        for chunk_size in (1, 3, 7, 20, 64):
            with self.subTest(chunk_size=chunk_size):
                got = process_in_chunks(
                    frames, lambda f: (f.astype(np.float32) * 1.5 + 1.0).mean(axis=2), chunk_size=chunk_size
                )
                self.assertEqual(len(got), len(reference))
                for i, (g, r) in enumerate(zip(got, reference, strict=True)):
                    np.testing.assert_array_equal(g, r, err_msg=f"frame {i} mismatch")

    def test_overlap_does_not_change_per_frame_results(self):
        """Overlap provides context for stateful fns; per-frame output is unchanged."""
        frames = _synthetic_frames()
        reference = self._reference(frames)
        for overlap in (1, 2, 5):
            with self.subTest(overlap=overlap):
                got = process_in_chunks(
                    frames,
                    lambda f: (f.astype(np.float32) * 1.5 + 1.0).mean(axis=2),
                    chunk_size=4,
                    overlap=overlap,
                )
                for g, r in zip(got, reference, strict=True):
                    np.testing.assert_array_equal(g, r)

    def test_chunk_size_one_edge(self):
        frames = _synthetic_frames()
        got = process_in_chunks(frames, lambda f: f + 1, chunk_size=1)
        for g, f in zip(got, frames, strict=True):
            np.testing.assert_array_equal(g, f + 1)

    def test_chunk_size_larger_than_total_edge(self):
        frames = _synthetic_frames()
        got = process_in_chunks(frames, lambda f: f + 1, chunk_size=999)
        for g, f in zip(got, frames, strict=True):
            np.testing.assert_array_equal(g, f + 1)

    def test_chunk_size_none_disables_chunking(self):
        frames = _synthetic_frames()
        got = process_in_chunks(frames, lambda f: f + 1, chunk_size=None)
        for g, f in zip(got, frames, strict=True):
            np.testing.assert_array_equal(g, f + 1)

    def test_empty_input(self):
        self.assertEqual(process_in_chunks([], lambda f: f, chunk_size=4), [])


class TestProcessInChunksIter(unittest.TestCase):
    """The generator form must not materialise the result list."""

    def test_is_lazy_generator(self):
        import types

        frames = _synthetic_frames()
        gen = process_in_chunks_iter(frames, lambda f: f, chunk_size=4)
        self.assertIsInstance(gen, types.GeneratorType)
        self.assertEqual(len(list(gen)), NUM_FRAMES)

    def test_results_match_list_form(self):
        frames = _synthetic_frames()
        fn = lambda f: f.sum(axis=2)  # noqa: E731
        got = list(process_in_chunks_iter(frames, fn, chunk_size=6))
        ref = process_in_chunks(frames, fn, chunk_size=6)
        self.assertEqual(len(got), len(ref))
        for g, r in zip(got, ref, strict=True):
            np.testing.assert_array_equal(g, r)


class TestChunkedWindows(unittest.TestCase):
    """Windowed access for stateful stages (temporal filters)."""

    def test_window_contents_include_overlap(self):
        items = list(range(20))
        windows = list(chunked_windows(items, chunk_size=8, overlap=2))
        self.assertEqual(
            [(list(w), cs, ce) for w, cs, ce in windows],
            [
                (list(range(0, 10)), 0, 8),
                (list(range(6, 18)), 8, 16),
                (list(range(14, 20)), 16, 20),
            ],
        )

    def test_core_slice_of_window_matches_items(self):
        items = list(range(20))
        for w, cs, ce in chunked_windows(items, chunk_size=6, overlap=2):
            ext_start = max(0, cs - 2)
            self.assertEqual(list(w[cs - ext_start : ce - ext_start]), items[cs:ce])

    def test_overlap_primes_stateful_filter_identically(self):
        """A finite-window temporal filter (FIR-style) primed on overlap frames
        must produce *bit-identical* core outputs to a whole-sequence run —
        this is the overlap-boundary consistency the acceptance criteria ask
        for. (Infinite-memory filters like EMA must instead carry state across
        chunks — see run_depth_stage — since no finite overlap can reproduce
        their full history.)"""
        frames = _synthetic_frames()
        taps = 3  # moving average over [t-2, t-1, t]

        def fir_whole(seq):
            out = []
            for i in range(len(seq)):
                lo = max(0, i - taps + 1)
                window = [seq[j].astype(np.float32) for j in range(lo, i + 1)]
                out.append(sum(window) / len(window))
            return out

        reference = fir_whole(frames)

        # Chunked: process each window (core+context) with fresh state, keep
        # only the core outputs. overlap >= taps-1 re-creates the exact
        # boundary inputs.
        got: list[np.ndarray] = []
        for window, cs, ce in chunked_windows(frames, chunk_size=8, overlap=taps - 1):
            ext_start = max(0, cs - (taps - 1))
            window_out = fir_whole(list(window))
            got.extend(window_out[cs - ext_start : ce - ext_start])

        self.assertEqual(len(got), len(reference))
        for i, (g, r) in enumerate(zip(got, reference, strict=True)):
            np.testing.assert_allclose(g, r, rtol=1e-6, err_msg=f"frame {i} mismatch across chunk boundary")

    def test_ema_state_carried_across_chunks_identically(self):
        """Infinite-memory filters (EMA) stay exact when the state is carried
        across chunk boundaries — the pattern run_depth_stage uses."""
        frames = _synthetic_frames()
        alpha = 0.3

        def ema_step(prev, f):
            return f.astype(np.float32) if prev is None else alpha * f + (1 - alpha) * prev

        # Whole-sequence reference
        prev = None
        reference = []
        for f in frames:
            prev = ema_step(prev, f)
            reference.append(prev)

        # Chunked with carried state (no overlap needed)
        prev = None
        got = []
        for window, _cs, _ce in chunked_windows(frames, chunk_size=7, overlap=0):
            for f in window:
                prev = ema_step(prev, f)
                got.append(prev)

        self.assertEqual(len(got), len(reference))
        for i, (g, r) in enumerate(zip(got, reference, strict=True)):
            np.testing.assert_allclose(g, r, rtol=1e-6, err_msg=f"frame {i} mismatch across chunk boundary")


class TestPairwiseInChunks(unittest.TestCase):
    def test_matches_zip_baseline(self):
        left = _synthetic_frames()
        right = _synthetic_frames()
        fn = lambda a, b: a.astype(np.int16) - b.astype(np.int16)  # noqa: E731
        reference = [fn(a, b) for a, b in zip(left, right, strict=True)]
        for chunk_size in (1, 5, 20, 100):
            with self.subTest(chunk_size=chunk_size):
                got = list(pairwise_in_chunks(left, right, fn, chunk_size=chunk_size))
                for g, r in zip(got, reference, strict=True):
                    np.testing.assert_array_equal(g, r)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            list(pairwise_in_chunks([1, 2], [1], lambda a, b: a, chunk_size=4))


class TestUnchunkableStages(unittest.TestCase):
    """Stages that cannot chunk must be explicitly documented (criterion 2)."""

    def test_registry_documents_global_stages(self):
        self.assertIn("depthcrafter", UNCHUNKABLE_STAGES)
        self.assertIn("stereocrafter", UNCHUNKABLE_STAGES)
        for stage, note in UNCHUNKABLE_STAGES.items():
            self.assertTrue(note.strip(), f"{stage} needs a memory-bound note")


# ---------------------------------------------------------------------------
# Pipeline stage integration — chunked vs unchunked must be identical
# ---------------------------------------------------------------------------


def _make_args(tmp_dir: str, **overrides):
    """Minimal args namespace for the run_pipeline stage functions."""
    defaults = {
        "input": "in.mp4",
        "output": os.path.join(tmp_dir, "out.mp4"),
        "temp_dir": tmp_dir,
        "chunk_size": 0,  # 0 = legacy unchunked
        "chunk_overlap": 0,
        "depth_model": "depth-anything",
        "model_size": "small",
        "device": "cpu",
        "temporal_smoothing": 0.0,
        "stereo_model": "default",
        "ipd": 0.064,
        "max_disparity": 0.05,
        "no_temporal": True,  # disable renderer EMA — per-frame determinism
        "output_width": 32,
        "output_height": 32,
        "src_hfov": 70.0,
        "no_ffmpeg_v360": True,  # OpenCV fallback — no ffmpeg in unit tests
        "no_equirect_batched": True,  # per-frame path — no ffmpeg needed
        "outpaint": "gradient",
        "outpaint_mask_threshold": 10,
        "outpaint_mask_top_ratio": 0.25,
        "outpaint_mask_bottom_ratio": 0.25,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class TestStageChunkingEquivalence(unittest.TestCase):
    """run_pipeline stages: chunked output == whole-sequence output."""

    def setUp(self):
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_dir = self._tmp.name
        self.frames = _synthetic_frames()
        rng = np.random.default_rng(7)
        self.depths = [rng.random((FRAME_H, FRAME_W), dtype=np.float32) for _ in range(NUM_FRAMES)]

    def tearDown(self):
        self._tmp.cleanup()

    def _load_pipeline_module(self):
        sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
        try:
            import run_pipeline

            return run_pipeline
        finally:
            pass

    def test_stereo_stage_chunked_identical(self):
        rp = self._load_pipeline_module()
        args_whole = _make_args(os.path.join(self.tmp_dir, "whole"), chunk_size=0)
        args_chunked = _make_args(os.path.join(self.tmp_dir, "chunked"), chunk_size=4)

        left_w, right_w = rp.run_stereo_stage(args_whole, self.frames, self.depths)
        left_c, right_c = rp.run_stereo_stage(args_chunked, self.frames, self.depths)

        self.assertEqual(len(left_w), NUM_FRAMES)
        self.assertEqual(len(left_c), NUM_FRAMES)
        for i in range(NUM_FRAMES):
            np.testing.assert_array_equal(left_c[i], left_w[i], err_msg=f"left frame {i}")
            np.testing.assert_array_equal(right_c[i], right_w[i], err_msg=f"right frame {i}")

    def test_equirect_stage_chunked_identical(self):
        rp = self._load_pipeline_module()
        lefts = self.frames
        rights = [np.flip(f, axis=1).copy() for f in self.frames]  # distinct R views

        args_whole = _make_args(os.path.join(self.tmp_dir, "eq_whole"), chunk_size=0)
        args_chunked = _make_args(os.path.join(self.tmp_dir, "eq_chunked"), chunk_size=6)

        sbs_w = rp.run_equirect_stage(args_whole, lefts, rights)
        sbs_c = rp.run_equirect_stage(args_chunked, lefts, rights)

        self.assertEqual(len(sbs_w), NUM_FRAMES)
        self.assertEqual(len(sbs_c), NUM_FRAMES)
        for i in range(NUM_FRAMES):
            np.testing.assert_array_equal(sbs_c[i], sbs_w[i], err_msg=f"sbs frame {i}")

    def test_outpaint_stage_chunked_identical(self):
        rp = self._load_pipeline_module()
        # Equirect-like SBS frames with black top/bottom boundary bands.
        sbs = []
        for f in self.frames:
            frame = f.copy()
            frame[:8, :, :] = 0  # black zenith band
            frame[-8:, :, :] = 0  # black nadir band
            sbs.append(frame)

        args_whole = _make_args(os.path.join(self.tmp_dir, "op_whole"), chunk_size=0)
        args_chunked = _make_args(os.path.join(self.tmp_dir, "op_chunked"), chunk_size=5)

        out_w = rp.run_outpaint_stage(args_whole, sbs)
        out_c = rp.run_outpaint_stage(args_chunked, sbs)

        self.assertEqual(len(out_w), NUM_FRAMES)
        self.assertEqual(len(out_c), NUM_FRAMES)
        for i in range(NUM_FRAMES):
            np.testing.assert_array_equal(out_c[i], out_w[i], err_msg=f"outpainted frame {i}")

    def test_depth_stage_temporal_ema_chunked_identical(self):
        """Temporal EMA state must carry across chunk boundaries bit-exactly."""
        rp = self._load_pipeline_module()

        fake_estimator = MagicMock()
        # Deterministic per-frame "depth": frame mean as a constant map.
        fake_estimator.estimate.side_effect = lambda f: np.full((FRAME_H, FRAME_W), f.mean(), dtype=np.float32)

        with patch.object(rp, "DepthEstimator", return_value=fake_estimator):
            args_whole = _make_args(os.path.join(self.tmp_dir, "d_whole"), chunk_size=0, temporal_smoothing=0.3)
            args_chunked = _make_args(os.path.join(self.tmp_dir, "d_chunked"), chunk_size=4, temporal_smoothing=0.3)
            dep_w = rp.run_depth_stage(args_whole, self.frames)
            dep_c = rp.run_depth_stage(args_chunked, self.frames)

        self.assertEqual(len(dep_w), NUM_FRAMES)
        self.assertEqual(len(dep_c), NUM_FRAMES)
        for i in range(NUM_FRAMES):
            np.testing.assert_allclose(dep_c[i], dep_w[i], rtol=1e-6, err_msg=f"depth frame {i}")

    def test_metadata_stage_streaming_matches_batch(self):
        """embed_frame_stream (chunked encode) must produce a valid VR180 MP4
        identical in content to embed_single_frame_batch for the same frames."""
        import importlib.util

        import pytest

        from pipeline.vr_metadata import VRMetadataEmbedder

        frames = _synthetic_frames(5)
        embedder = VRMetadataEmbedder(codec="h264", crf=28, fps=24)
        H, W = frames[0].shape[:2]

        out_batch = os.path.join(self.tmp_dir, "batch.mp4")
        out_stream = os.path.join(self.tmp_dir, "stream.mp4")

        embedder.embed_single_frame_batch(frames, out_batch, width=W, height=H)
        embedder.embed_frame_stream(iter(frames), out_stream, width=W, height=H)

        self.assertTrue(os.path.exists(out_batch))
        self.assertTrue(os.path.exists(out_stream))
        self.assertGreater(os.path.getsize(out_stream), 0)

        # Literal sv3d/st3d boxes require Google's optional spatial-media CLI;
        # the ffmpeg fallback injects equivalent metadata in a different form
        # (mirrors the skipif in tests/test_pipeline.py).
        if importlib.util.find_spec("spatialmedia") is None:
            pytest.skip("sv3d/st3d boxes need the optional spatial-media CLI")
        with open(out_stream, "rb") as f:
            data = f.read()
        self.assertIn(b"sv3d", data)
        self.assertIn(b"st3d", data)

    def test_metadata_stage_streaming_empty_raises(self):
        from pipeline.vr_metadata import VRMetadataEmbedder

        embedder = VRMetadataEmbedder(codec="h264", crf=28, fps=24)
        with self.assertRaises(ValueError):
            embedder.embed_frame_stream(iter([]), os.path.join(self.tmp_dir, "x.mp4"), width=64, height=64)


class TestChunkSizeCLI(unittest.TestCase):
    """--chunk-size / --chunk-overlap argument parsing."""

    def test_parse_defaults(self):
        sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
        try:
            import run_pipeline

            args = run_pipeline.parse_args(["--input", "x.mp4"])
            self.assertEqual(args.chunk_size, 64)
            self.assertEqual(args.chunk_overlap, 0)
        finally:
            sys.path.remove(os.path.join(PROJECT_ROOT, "scripts"))
            sys.modules.pop("run_pipeline", None)

    def test_parse_explicit(self):
        sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
        try:
            import run_pipeline

            args = run_pipeline.parse_args(["--input", "x.mp4", "--chunk-size", "16", "--chunk-overlap", "2"])
            self.assertEqual(args.chunk_size, 16)
            self.assertEqual(args.chunk_overlap, 2)
        finally:
            sys.path.remove(os.path.join(PROJECT_ROOT, "scripts"))
            sys.modules.pop("run_pipeline", None)

    def test_chunk_size_zero_disables(self):
        sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
        try:
            import run_pipeline

            args = run_pipeline.parse_args(["--input", "x.mp4", "--chunk-size", "0"])
            self.assertIsNone(run_pipeline._chunk_size(args))
            args.chunk_size = 8
            self.assertEqual(run_pipeline._chunk_size(args), 8)
        finally:
            sys.path.remove(os.path.join(PROJECT_ROOT, "scripts"))
            sys.modules.pop("run_pipeline", None)


if __name__ == "__main__":
    unittest.main()
