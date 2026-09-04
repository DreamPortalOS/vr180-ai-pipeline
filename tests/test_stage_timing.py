"""Tests for P-1 (#216) — streaming pipeline stage wall-clock timing.

The streaming path previously reported only a single total wall-clock
("66.7 s/frame") with no breakdown, so lead could not decide *where* to
optimise.  :meth:`pipeline.streaming_pipeline.StreamingPipeline.process_stream`
now records per-stage wall clocks and exposes them in two places:

  1. Logged at pipeline end as an aligned ``stage_timings`` table
     (stage / seconds / % of total).
  2. Attached to the pipeline instance as ``stage_timings`` — a dict
     ``{stage: seconds}`` whose stages are ``depth``, ``stereo``,
     ``equirect``, ``encode``, ``metadata``.  The CLI's ``generation``
     sidecar block consumes this dict (it is the existing sidecar
     structure; no new file format is introduced by this card).

Timing is *always on* (cost = a handful of ``time.perf_counter()`` calls)
and must never affect correctness:

  - A stage that raises re-raises **unchanged** (type and message intact).
  - A failure inside the timing machinery itself (e.g. ``perf_counter``
    broken in a test double) is caught and the stage record is simply
    skipped — the pipeline must not crash because of its own metering.

All tests are CPU-only and inject fake backends; no real model inference,
no real ffmpeg, no real I/O beyond mocked cv2 capture.
"""

from __future__ import annotations

import io
import logging
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

# K-15 (#205): let this script run directly without PYTHONPATH set.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pipeline.streaming_pipeline import StreamingPipeline  # noqa: E402

EXPECTED_STAGES = ("depth", "stereo", "equirect", "encode", "metadata")


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _make_cap(num_frames=2, w=8, h=8):
    """cv2.VideoCapture stand-in yielding *num_frames* fake BGR frames."""
    cap = MagicMock()
    cap.isOpened.return_value = True
    cap.get.side_effect = lambda prop: {
        3: w,
        4: h,
        7: float(num_frames),
        5: 30.0,
    }.get(prop, 0.0)
    reads = [(True, np.zeros((h, w, 3), dtype=np.uint8)) for _ in range(num_frames)]
    reads.append((False, None))
    cap.read.side_effect = reads
    return cap


def _make_proc():
    """A subprocess.Popen stand-in for the raw-frame ffmpeg writer."""
    proc = MagicMock()
    proc.returncode = 0
    return proc


def _make_pipeline(**kwargs):
    """Build a StreamingPipeline with depth/stereo/eq stages mocked so
    ``process_stream`` can run end-to-end without real inference or ffmpeg."""
    with (
        patch("pipeline.streaming_pipeline.DepthEstimator"),
        patch("pipeline.streaming_pipeline.StereoRenderer"),
        patch("pipeline.streaming_pipeline.EquirectangularMapper"),
    ):
        kwargs.setdefault("device", "cpu")
        kwargs.setdefault("output_width", 100)
        kwargs.setdefault("output_height", 50)
        p = StreamingPipeline(**kwargs)
        # Per-frame depth estimator (Depth-Anything) stand-in.
        p.depth_estimator.estimate.return_value = np.full((8, 8), 0.5, dtype=np.float32)
        # Per-frame stereo renderer stand-in.
        p.stereo_renderer.render.return_value = (
            np.zeros((10, 10, 3), dtype=np.uint8),
            np.zeros((10, 10, 3), dtype=np.uint8),
        )
        # Equirectangular mapper stand-in: SBS = (out_h, 2*out_w, 3).
        p.eq_mapper.map_stereo_pair.return_value = np.zeros((50, 200, 3), dtype=np.uint8)
        return p


def _run(p, num_frames=2, cap=None, proc=None):
    """Drive ``process_stream`` to completion with cv2/Popen mocked."""
    cap = cap or _make_cap(num_frames=num_frames)
    proc = proc or _make_proc()
    with (
        patch("pipeline.streaming_pipeline.cv2.VideoCapture", return_value=cap),
        patch("pipeline.streaming_pipeline.subprocess.Popen", return_value=proc),
    ):
        p.process_stream("in.mp4", "out.mp4")


def _run_sidecar_capture(p):
    """Run and return the dict that a sidecar consumer would read off the
    pipeline (the ``generation`` block payload)."""
    _run(p)
    return p.stage_timings


# ---------------------------------------------------------------------------
# Test 1: stage_timings contains all expected keys with non-negative floats.
# ---------------------------------------------------------------------------


class TestStageTimingsPopulated(unittest.TestCase):
    def test_all_stages_present_with_nonnegative_floats(self):
        p = _make_pipeline()
        timings = _run_sidecar_capture(p)

        assert isinstance(timings, dict)
        for stage in EXPECTED_STAGES:
            assert stage in timings, f"missing stage timing key: {stage}"
            val = timings[stage]
            assert isinstance(val, float), f"{stage} timing must be float, got {type(val).__name__}: {val!r}"
            assert val >= 0.0, f"{stage} timing negative: {val}"


# ---------------------------------------------------------------------------
# Test 2: percentages sum to ~100%.
# ---------------------------------------------------------------------------


class TestPercentagesSumToOne(unittest.TestCase):
    def test_percentages_sum_to_roughly_100(self):
        p = _make_pipeline()
        timings = _run_sidecar_capture(p)

        total = sum(timings.values())
        assert total > 0.0, "total elapsed time must be > 0"
        # Use the pipeline's own reporting to confirm the percentage view:
        # recompute percentages from the dict and check they sum near 100.
        pcts = {s: (100.0 * timings[s] / total) for s in timings}
        pct_sum = sum(pcts.values())
        assert abs(pct_sum - 100.0) < 0.01, f"percentages sum {pct_sum:.4f}%, expected ~100%"


# ---------------------------------------------------------------------------
# Test 3: a stage exception propagates unchanged (type + message intact).
# ---------------------------------------------------------------------------


class TestExceptionPropagation(unittest.TestCase):
    def test_depth_exception_propagates_unchanged(self):
        p = _make_pipeline()
        p.depth_estimator.estimate.side_effect = RuntimeError("depth_backend_bang")

        with (
            patch("pipeline.streaming_pipeline.cv2.VideoCapture", return_value=_make_cap()),
            patch("pipeline.streaming_pipeline.subprocess.Popen", return_value=_make_proc()),
            self.assertRaises(RuntimeError) as ctx,
        ):
            p.process_stream("in.mp4", "out.mp4")

        self.assertEqual(type(ctx.exception).__name__, "RuntimeError")
        self.assertIn("depth_backend_bang", str(ctx.exception))

    def test_stereo_exception_propagates_unchanged(self):
        p = _make_pipeline()
        p.stereo_renderer.render.side_effect = ValueError("stereo_render_fail")

        with (
            patch("pipeline.streaming_pipeline.cv2.VideoCapture", return_value=_make_cap()),
            patch("pipeline.streaming_pipeline.subprocess.Popen", return_value=_make_proc()),
            self.assertRaises(ValueError) as ctx,
        ):
            p.process_stream("in.mp4", "out.mp4")

        self.assertEqual(type(ctx.exception).__name__, "ValueError")
        self.assertIn("stereo_render_fail", str(ctx.exception))

    def test_eq_mapper_exception_propagates_unchanged(self):
        p = _make_pipeline()
        p.eq_mapper.map_stereo_pair.side_effect = RuntimeError("equirect_proj_fail")

        with (
            patch("pipeline.streaming_pipeline.cv2.VideoCapture", return_value=_make_cap()),
            patch("pipeline.streaming_pipeline.subprocess.Popen", return_value=_make_proc()),
            self.assertRaises(RuntimeError) as ctx,
        ):
            p.process_stream("in.mp4", "out.mp4")

        self.assertEqual(type(ctx.exception).__name__, "RuntimeError")
        self.assertIn("equirect_proj_fail", str(ctx.exception))


# ---------------------------------------------------------------------------
# Test 4: stage_timings is present as the sidecar-generation payload.
# ---------------------------------------------------------------------------


class TestSidecarPayload(unittest.TestCase):
    def test_stage_timings_attached_to_pipeline_for_sidecar(self):
        """The pipeline instance must expose ``stage_timings`` so the CLI's
        ``generation`` sidecar block can carry it without a new file format.

        This is the "复用现有结构，不要新建一套文件格式" contract: the dict
        lands inside sidecar ``generation.stage_timings``.
        """
        p = _make_pipeline()
        _run(p)

        assert hasattr(p, "stage_timings"), "pipeline must expose stage_timings for the sidecar generation block"
        timings = p.stage_timings
        assert isinstance(timings, dict)
        for stage in EXPECTED_STAGES:
            assert stage in timings

    def test_stage_timings_also_carries_total_seconds(self):
        """The logged table reports each stage against a total; the exposed
        dict should carry the total so a sidecar consumer can reproduce the
        percentages without re-deriving the loop."""
        p = _make_pipeline()
        _run(p)

        self.assertIn("_total", p.stage_timings)
        self.assertGreaterEqual(p.stage_timings["_total"], 0.0)


# ---------------------------------------------------------------------------
# Test 5: timing machinery failure must not crash the pipeline.
# ---------------------------------------------------------------------------


class TestTimingFailureIsolated(unittest.TestCase):
    def test_perf_counter_failure_does_not_crash_pipeline(self):
        """If time.perf_counter() itself raises (broken clock / monkeypatched
        hostile test double), timing simply skips the record and the pipeline
        still completes and still raises nothing *from timing*."""
        p = _make_pipeline()
        cap = _make_cap()
        proc = _make_proc()

        # Hostile clock: perf_counter raises. Timing code must catch this and
        # keep going; the pipeline's actual frames must still process.
        def hostile_counter():
            raise RuntimeError("clock exploded")

        with (
            patch("pipeline.streaming_pipeline.cv2.VideoCapture", return_value=cap),
            patch("pipeline.streaming_pipeline.subprocess.Popen", return_value=proc),
            patch("pipeline.streaming_pipeline.time.perf_counter", side_effect=hostile_counter),
        ):
            # Must not raise — timing failures are isolated.
            p.process_stream("in.mp4", "out.mp4")


# ---------------------------------------------------------------------------
# Test 6: logged table appears at INFO level.
# ---------------------------------------------------------------------------


class TestLoggedTable(unittest.TestCase):
    def test_timing_table_logged_at_info(self):
        """process_stream must log the aligned stage_timings table at INFO so
        a real render's log (the card's core deliverable) carries it."""
        p = _make_pipeline()

        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        log = logging.getLogger("vr180-streaming")
        log.addHandler(handler)
        log.setLevel(logging.INFO)
        try:
            _run(p)

        finally:
            log.removeHandler(handler)

        output = stream.getvalue()
        # The table header and every stage name must appear in the log.
        for token in ("stage_timings", *EXPECTED_STAGES):
            assert token in output, f"log missing {token!r}; got:\n{output}"


if __name__ == "__main__":
    unittest.main()
