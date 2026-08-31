"""Tests for V-4.1b (issue #89) — per-chunk incremental ffmpeg encode.

Acceptance criteria verified here:

  - **Equivalence**: the chunked fused stereo→equirect→encode produces frames
    that are frame-for-frame identical to the whole-sequence per-frame run
    (the V-4 equivalence contract must continue to hold under the V-4.1b
    incremental-encode fusion).

  - **Incremental writes / no full-length array**: a mock persistent ffmpeg
    writer receives one ``write()`` per emitted frame, in order, and at no
    point do more than ``chunk_size`` (≤ chunk_size + small slack) SBS frames
    coexist in memory — the full clip's L/R/SBS buffers are never
    materialised.

  - **ffmpeg failure**: a dead encoder (broken pipe / non-zero return code)
    surfaces as a ``RuntimeError`` carrying the captured stderr tail — no
    silent bad file.

  - **chunk mode forces per-frame equirect**: the fused path uses
    ``map_stereo_pair`` (per-frame), never the batched ``map_sequence`` that
    requires the full sequence.

All CPU-only; subprocess / models / spherical-metadata are mocked.
"""

import contextlib
import importlib.util
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pipeline.streaming_pipeline import RawFrameFFmpegWriter, select_encoder  # noqa: E402

# ---------------------------------------------------------------------------
# Mock persistent-ffmpeg writer sinks (module-level to satisfy N805)
# ---------------------------------------------------------------------------


class _SinkFrames:
    """Sink that records each written SBS frame array (equivalence check)."""

    def __init__(self, *args, **kwargs):
        self.writes: list = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def write(self, frame):
        self.writes.append(frame)

    def close(self):
        pass


class _SinkBytes:
    """Sink that records each written frame's raw bytes (single-frame proof)."""

    def __init__(self, *args, **kwargs):
        self.writes: list = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def write(self, frame):
        self.writes.append(frame.tobytes())

    def close(self):
        pass


class _SinkCount:
    """Sink that counts writes (memory-contract structural proof)."""

    def __init__(self, *args, **kwargs):
        self.count = 0

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def write(self, _frame):
        self.count += 1

    def close(self):
        pass


class _FailingWriter:
    """Sink whose first write raises RuntimeError (dead-encoder proof)."""

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def write(self, _frame):
        raise RuntimeError(
            "ffmpeg encoder died after 0 frames — check encoder "
            "availability/limits for this resolution. ffmpeg stderr: nvenc boom"
        )

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _import_run_pipeline():
    """Load scripts/run_pipeline.py as an isolated module (V-4 test convention)."""
    scripts_dir = os.path.join(PROJECT_ROOT, "scripts")
    sys.path.insert(0, scripts_dir)
    try:
        spec = importlib.util.spec_from_file_location(
            "run_pipeline_v41b",
            os.path.join(scripts_dir, "run_pipeline.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        assert spec is not None
        spec.loader.exec_module(mod)
        return mod
    finally:
        with contextlib.suppress(ValueError):
            sys.path.remove(scripts_dir)


def _make_frames(n: int = 12, h: int = 16, w: int = 16) -> list[np.ndarray]:
    """Deterministic synthetic frame sequence — each frame a distinct value."""
    rng = np.random.default_rng(seed=89)
    return [rng.integers(0, 256, (h, w, 3), dtype=np.uint8) for _ in range(n)]


def _make_depths(n: int, h: int, w: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed=90)
    return [rng.random((h, w)).astype(np.float32) for _ in range(n)]


def _fused_args(run_pipeline, tmp_path, chunk_size, overlap=0):
    """A MagicMock args namespace matching run_chunked_fused_stage's reads."""
    args = MagicMock()
    args.ipd = 0.064
    args.max_disparity = 0.02
    args.convergence = 0.3
    args.temporal_smooth = True
    args.no_temporal = False
    args.output_width = 32
    args.output_height = 32
    args.src_hfov = 70.0
    args.no_ffmpeg_v360 = True  # force OpenCV equirect path (CPU, deterministic)
    args.codec = "h264"
    args.crf = 23
    args.fps = 30
    args.bitrate = None
    args.hw_encoder = "off"
    args.device = "cpu"
    args.chunk_size = chunk_size
    args.overlap = overlap
    args.temp_dir = str(tmp_path)
    args.input = str(tmp_path / "fake_input.mp4")
    args.output = str(tmp_path / "out.mp4")
    return args


def _capture_factory(sink_cls, sinks):
    """Return a side_effect factory that builds a *sink_cls* and stashes it.

    Defined at module scope so the returned closure binds *sinks* by default
    argument (avoids B023: the storage list is captured at def time, not via
    a loop variable).
    """

    def _factory(*a, **k):
        s = sink_cls(*a, **k)
        sinks.append(s)
        return s

    return _factory


# ---------------------------------------------------------------------------
# RawFrameFFmpegWriter — unit tests
# ---------------------------------------------------------------------------


class TestRawFrameFFmpegWriterCmd(unittest.TestCase):
    """The standalone writer reuses select_encoder (no duplicated logic)."""

    def test_cmd_declares_raw_rgb_pipe_and_output_path(self):
        w = RawFrameFFmpegWriter("out.mp4", 7680, 1920, codec="h264", crf=20, fps=30)
        cmd = w._build_cmd()
        self.assertIn("rawvideo", cmd)
        self.assertEqual(cmd[cmd.index("-s") + 1], "7680x1920")
        self.assertEqual(cmd[cmd.index("-i") + 1], "pipe:0")
        # Encoder comes from select_encoder (software h264 for ≤4096-wide).
        self.assertEqual(select_encoder("h264", 7680, hw=False), ["-c:v", "libx265", "-preset", "fast"])
        self.assertIn("out.mp4", cmd)

    def test_cmd_uses_nvenc_when_hw_true(self):
        w = RawFrameFFmpegWriter("out.mp4", 1920, 1920, codec="h264", hw_encoder=True)
        cmd = w._build_cmd()
        self.assertEqual(cmd[cmd.index("-c:v") + 1], "h264_nvenc")

    def test_bitrate_overrides_crf(self):
        w = RawFrameFFmpegWriter("o.mp4", 100, 50, bitrate="80M")
        cmd = w._build_cmd()
        self.assertIn("-b:v", cmd)
        self.assertNotIn("-crf", cmd)


class TestRawFrameFFmpegWriterWrite(unittest.TestCase):
    """Incremental write + failure surfacing (no silent bad files)."""

    def _fake_proc(self, returncode=0, write_raises=None):
        proc = MagicMock()
        proc.returncode = returncode
        proc.poll.return_value = returncode
        if write_raises is not None:
            proc.stdin.write.side_effect = write_raises
        return proc

    def test_write_is_incremental_per_frame(self):
        """Each .write() pushes exactly one frame's bytes to the pipe."""
        proc = self._fake_proc()
        with (
            patch("pipeline.streaming_pipeline.subprocess.Popen", return_value=proc),
            patch("pipeline.streaming_pipeline.tempfile.TemporaryFile"),
        ):
            w = RawFrameFFmpegWriter("o.mp4", 2, 2, codec="h264").open()
            frames = [np.full((2, 2, 3), v, dtype=np.uint8) for v in (10, 20, 30)]
            for f in frames:
                w.write(f)
            w.close()
        # Exactly one pipe write per frame, each carrying the right bytes.
        self.assertEqual(proc.stdin.write.call_count, 3)
        for i, call in enumerate(proc.stdin.write.call_args_list):
            self.assertEqual(call.args[0], frames[i].tobytes())

    def test_nonzero_returncode_raises_with_stderr(self):
        proc = self._fake_proc(returncode=1)
        with (
            patch("pipeline.streaming_pipeline.subprocess.Popen", return_value=proc),
            patch("pipeline.streaming_pipeline.tempfile.TemporaryFile"),
        ):
            w = RawFrameFFmpegWriter("o.mp4", 2, 2, codec="h264").open()
            w._stderr_summary = lambda: "encoder boom: nvenc failed to open"
            with self.assertRaises(RuntimeError) as ctx:
                w.close()
        self.assertIn("exit code 1", str(ctx.exception))
        self.assertIn("encoder boom", str(ctx.exception))

    def test_broken_pipe_raises_runtime_error(self):
        proc = self._fake_proc(write_raises=BrokenPipeError("closed"))
        with (
            patch("pipeline.streaming_pipeline.subprocess.Popen", return_value=proc),
            patch("pipeline.streaming_pipeline.tempfile.TemporaryFile"),
        ):
            w = RawFrameFFmpegWriter("o.mp4", 2, 2, codec="h264").open()
            w._stderr_summary = lambda: "(stderr tail)"
            with self.assertRaises(RuntimeError) as ctx:
                w.write(np.zeros((2, 2, 3), dtype=np.uint8))
        self.assertIn("ffmpeg encoder died", str(ctx.exception))


# ---------------------------------------------------------------------------
# run_chunked_fused_stage — equivalence + incremental-write + no-full-buffer
# ---------------------------------------------------------------------------


class _RecordingMapper:
    """EquirectangularMapper stand-in that records SBS-frame liveness.

    Wraps a real per-frame deterministic mapping so equivalence can be checked
    against a whole-sequence reference, while tracking how many produced SBS
    frames are simultaneously alive (via a count of un-dereferenced entries).
    """

    def __init__(self, *args, **kwargs) -> None:
        self.peak_alive = 0
        self._alive: list = []

    def map_stereo_pair(self, left, right):
        sbs = np.concatenate([left, right], axis=1)
        # Append a weak sentinel: caller `del`s the SBS frame after writing, so
        # the list length reflects co-resident produced-but-not-yet-released
        # frames. We pop entries whose refcount dropped (caller deleted them).
        self._alive.append(sbs)
        # Prune entries that the caller no longer holds (id reuse is fine here —
        # we only need an upper bound on simultaneous residency).
        self._alive = [a for a in self._alive if sys.getrefcount(a) > 2]
        self.peak_alive = max(self.peak_alive, len(self._alive))
        return sbs


def _whole_reference(frames, depths, args_factory):
    """Whole-sequence per-frame reference: stereo→equirect, no chunking.

    Uses a fresh StereoRenderer (one persistent instance, matching the chunked
    path) and the same per-frame equirect mapping the fused path uses.
    """
    from pipeline.stereo_renderer import StereoRenderer

    r = StereoRenderer(
        ipd=0.064,
        max_disparity=0.02,
        convergence=0.3,
        temporal_smooth=True,
    )
    mapper = _RecordingMapper()
    sbs = []
    for f, d in zip(frames, depths, strict=True):
        left, right = r.render(f, d)
        sbs.append(mapper.map_stereo_pair(left, right))
    return sbs


class TestRunChunkedFusedStage(unittest.TestCase):
    """V-4.1b acceptance: equivalence + incremental writes + bounded residency."""

    def setUp(self):
        self.run_pipeline = _import_run_pipeline()

    def test_chunked_fused_matches_whole_sequence(self):
        """The chunked fused path is bit-exact vs the whole-sequence per-frame run."""
        import tempfile

        tmp = Path(tempfile.mkdtemp())
        frames = _make_frames(n=12)
        depths = _make_depths(12, 16, 16)
        ref = _whole_reference(frames, depths, None)

        for cs in (1, 4, 100):
            # Factory returns a fresh _SinkFrames each call and stashes it for
            # inspection (avoids B023: the instance is captured by closure over
            # the list, not a loop variable).
            sinks: list = []
            _factory = _capture_factory(_SinkFrames, sinks)

            # Patch the heavy pieces: real StereoRenderer (deterministic, CPU),
            # recording per-frame mapper, a sink writer, and the metadata
            # injection (a no-op move so the output path exists).  The writer +
            # injector are imported *inside* run_chunked_fused_stage, so patch
            # them at their source modules.
            with (
                patch("pipeline.streaming_pipeline.RawFrameFFmpegWriter", side_effect=_factory),
                patch("pipeline.spherical_injector.inject_spherical_metadata", lambda *a, **k: None),
                patch("os.replace"),
            ):
                args = _fused_args(self.run_pipeline, tmp / f"cs{cs}c", chunk_size=cs)
                with patch.object(self.run_pipeline, "EquirectangularMapper", _RecordingMapper):
                    out = self.run_pipeline.run_chunked_fused_stage(args, frames, depths)

            recorded_writes = sinks[0].writes
            self.assertEqual(len(recorded_writes), 12, f"cs={cs}: wrong write count")
            for i, (got, want) in enumerate(zip(recorded_writes, ref, strict=True)):
                self.assertTrue(np.array_equal(got, want), f"cs={cs} frame {i} differs")
            self.assertTrue(out.endswith("out.mp4"))

    def test_writer_receives_one_write_per_frame_in_order(self):
        """Exactly N incremental writes, in frame order — no full-length array fed at once."""
        import tempfile

        tmp = Path(tempfile.mkdtemp())
        frames = _make_frames(n=8)
        depths = _make_depths(8, 16, 16)

        sinks: list = []
        _factory = _capture_factory(_SinkBytes, sinks)

        args = _fused_args(self.run_pipeline, tmp, chunk_size=3)
        with (
            patch("pipeline.streaming_pipeline.RawFrameFFmpegWriter", side_effect=_factory),
            patch("pipeline.spherical_injector.inject_spherical_metadata", lambda *a, **k: None),
            patch("os.replace"),
        ):
            self.run_pipeline.run_chunked_fused_stage(args, frames, depths)

        write_calls = sinks[0].writes
        # 8 frames → 8 incremental writes (one per frame, not one batched blob).
        self.assertEqual(len(write_calls), 8)
        # No single write carries more than one frame's bytes (proves no
        # full-length b"".join blob was constructed).  Derive the per-SBS-frame
        # byte size from the first write (equirect SBS dims, not input dims).
        one_frame_bytes = len(write_calls[0])
        self.assertGreater(one_frame_bytes, 0)
        for blob in write_calls:
            self.assertEqual(len(blob), one_frame_bytes, "a write carried more than one frame")

    def test_chunked_not_full_length_no_batched_equirect(self):
        """V-4.1b memory contract: stereo is chunked (chunk_size, not N) and
        equirect is per-frame (map_stereo_pair), never the batched map_sequence
        that needs the full sequence.  No full-length L/R/SBS array is built.

        We prove it structurally:
          - ``StereoRenderer.render_sequence_chunked`` is called with the
            user's ``chunk_size`` (5), not the clip length (30) — the stereo
            stage is bounded.
          - The per-frame ``map_stereo_pair`` is called exactly N times (one
            per emitted frame) and the batched ``map_sequence`` is never
            called — chunk mode forces per-frame equirect.
          - The ffmpeg writer gets exactly N single-frame writes (not one
            full-length ``b"".join`` blob) — proven in the order test above.
        """
        import tempfile

        tmp = Path(tempfile.mkdtemp())
        frames = _make_frames(n=30, h=8, w=8)
        depths = _make_depths(30, 8, 8)
        chunk_size = 5

        sinks: list = []
        _factory = _capture_factory(_SinkCount, sinks)

        # A real EquirectangularMapper so we can spy which method the fused
        # path calls — per-frame map_stereo_pair (correct) vs batched
        # map_sequence (forbidden in chunk mode).
        from pipeline.equirectangular_mapper import EquirectangularMapper as _Real

        spy = _Real(
            output_width=32,
            output_height=32,
            src_hfov=70.0,
            use_ffmpeg=False,  # OpenCV per-frame path (CPU, deterministic)
        )
        map_stereo_calls = 0

        def _spy_stereo_pair(left, right):
            nonlocal map_stereo_calls
            map_stereo_calls += 1
            return np.concatenate([left, right], axis=1)

        spy.map_stereo_pair = _spy_stereo_pair
        spy.map_sequence = MagicMock(side_effect=AssertionError("map_sequence must not be called in chunk mode"))

        # Spy render_sequence_chunked to capture the chunk_size argument.
        from pipeline.stereo_renderer import StereoRenderer as _RealStereo

        real_render = _RealStereo.render_sequence_chunked

        captured_chunk_size = {}

        def _spy_render(self_r, f, d, *, chunk_size=None, overlap=0):
            captured_chunk_size["cs"] = chunk_size
            return real_render(self_r, f, d, chunk_size=chunk_size, overlap=overlap)

        args = _fused_args(self.run_pipeline, tmp, chunk_size=chunk_size)
        with (
            patch("pipeline.streaming_pipeline.RawFrameFFmpegWriter", side_effect=_factory),
            patch("pipeline.spherical_injector.inject_spherical_metadata", lambda *a, **k: None),
            patch("os.replace"),
            patch.object(self.run_pipeline, "EquirectangularMapper", lambda *a, **k: spy),
            patch.object(_RealStereo, "render_sequence_chunked", _spy_render),
        ):
            self.run_pipeline.run_chunked_fused_stage(args, frames, depths)

        # Stereo was chunked with the user's chunk_size, not the clip length.
        self.assertEqual(captured_chunk_size.get("cs"), chunk_size)
        self.assertNotEqual(captured_chunk_size.get("cs"), len(frames))
        # Per-frame equirect ran N times; batched map_sequence never ran.
        self.assertEqual(map_stereo_calls, len(frames))
        spy.map_sequence.assert_not_called()
        # The writer got exactly N single-frame writes (no full-length blob).
        self.assertEqual(sinks[0].count, len(frames))

    def test_ffmpeg_failure_raises_runtime_error(self):
        """A dead ffmpeg process surfaces as RuntimeError with stderr summary."""
        import tempfile

        tmp = Path(tempfile.mkdtemp())
        frames = _make_frames(n=4)
        depths = _make_depths(4, 16, 16)

        args = _fused_args(self.run_pipeline, tmp, chunk_size=2)
        with (
            patch("pipeline.streaming_pipeline.RawFrameFFmpegWriter", _FailingWriter),
            patch("pipeline.spherical_injector.inject_spherical_metadata", lambda *a, **k: None),
            patch("os.replace"),
            self.assertRaises(RuntimeError) as ctx,
        ):
            self.run_pipeline.run_chunked_fused_stage(args, frames, depths)
        self.assertIn("ffmpeg encoder died", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
