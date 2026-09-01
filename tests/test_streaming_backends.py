"""Tests for issue #120 (I-5) — streaming path honours --depth-model/--stereo-model.

The streaming pipeline (``pipeline/streaming_pipeline.py``) previously
hard-coded Depth-Anything + StereoRenderer and silently ignored the
``--depth-model depthcrafter`` / ``--stereo-model stereocrafter`` CLI flags —
the anti-motion-sickness backends never ran under the default/recommended
``--quality standard|high`` path.

These tests cover the I-5 fix:

  - StreamingPipeline accepts injectable depth/stereo backends; defaults are
    unchanged (Depth-Anything + StereoRenderer) — regression assertion.
  - Injected whole-clip backends (``estimate_video`` / ``render_video``) are
    detected, run **once** for the whole clip, and their outputs feed the
    per-frame equirect→encode fuse loop.
  - The effective backend names are logged at stream startup.
  - The run_pipeline factory falls back with a loud WARNING when a requested
    backend is unavailable (CUDA missing / not deployed) — never silent.

Issue #137 (I-7) closes the verification gap for the StereoCrafter side
specifically:

  - The streaming whole-clip stereo path logs BOTH the startup
    ``stereo=stereocrafter`` line AND the ``🎬 [Stereo]`` real-call line in the
    same run (the pre-#137 logging test mocked the stereo renderer as
    per-frame, so it never exercised the whole-clip stereo path's logging).
  - The streaming CLI branch calls ``build_stereo_backend(args, fallback=True)``
    — the unavailable→WARNING→default policy the card requires.
  - The whole-clip stereo precompute reads frames back from the paths the
    backend *returns* (not the assumed ones), matching the batch stage.

All tests are CPU-only; cv2 capture / ffmpeg / model stages are mocked or
faked.  No real model download, no real inference, no real API calls.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pipeline.streaming_pipeline import StreamingPipeline  # noqa: E402

# ---------------------------------------------------------------------------
# Fake whole-clip backends (no CUDA, no real inference)
# ---------------------------------------------------------------------------


class FakeWholeClipDepth:
    """Whole-clip depth estimator: exposes estimate_video (like DepthCrafter).

    Records that it was called and returns a fixed number of fake depth maps.
    """

    def __init__(self, num_frames=3, h=8, w=8):
        self.num_frames = num_frames
        self.h = h
        self.w = w
        self.calls = []

    def estimate_video(self, input_path, output_dir):
        self.calls.append((input_path, output_dir))
        rng = np.random.default_rng(7)
        return [rng.random((self.h, self.w)).astype(np.float32) for _ in range(self.num_frames)]


class FakeWholeClipStereo:
    """Whole-clip stereo renderer: exposes render_video (like StereoCrafter).

    Writes tiny L/R videos so the pipeline can read the frames back, records
    the call, and returns the output paths.
    """

    def __init__(self, num_frames=3, h=8, w=8):
        self.num_frames = num_frames
        self.h = h
        self.w = w
        self.calls = []

    def render_video(self, input_path, depth_dir, output_left, output_right):
        self.calls.append((input_path, depth_dir, output_left, output_right))
        import cv2

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        for path, val in ((output_left, 40), (output_right, 200)):
            wr = cv2.VideoWriter(path, fourcc, 30, (self.w, self.h))
            for _ in range(self.num_frames):
                wr.write(np.full((self.h, self.w, 3), val, dtype=np.uint8))
            wr.release()
        return output_left, output_right


def _make_pipeline(**kwargs):
    """Build a StreamingPipeline with the eq_mapper mocked (heavy) but the
    injected depth/stereo backends left as-is."""
    with patch("pipeline.streaming_pipeline.EquirectangularMapper"):
        kwargs.setdefault("device", "cpu")
        return StreamingPipeline(**kwargs)


def _fake_cap(num_frames=3, w=8, h=8):
    """A cv2.VideoCapture stand-in that yields *num_frames* fake frames."""
    cap = MagicMock()
    cap.isOpened.return_value = True
    cap.get.side_effect = lambda prop: {
        3: w,  # CAP_PROP_FRAME_WIDTH
        4: h,  # CAP_PROP_FRAME_HEIGHT
        7: float(num_frames),  # CAP_PROP_FRAME_COUNT
        5: 30.0,  # CAP_PROP_FPS
    }.get(prop, 0.0)
    reads = [(True, np.zeros((h, w, 3), dtype=np.uint8)) for _ in range(num_frames)]
    reads.append((False, None))
    cap.read.side_effect = reads
    return cap


# ---------------------------------------------------------------------------
# Regression: defaults unchanged
# ---------------------------------------------------------------------------


class TestDefaultsUnchanged(unittest.TestCase):
    """Regression assertion (I-5): without injection, behaviour is bit-exact."""

    def test_default_backends_are_depth_anything_and_stereo_renderer(self):
        p = _make_pipeline(output_width=100, output_height=50)
        from pipeline.depth_estimator import DepthEstimator
        from pipeline.stereo_renderer import StereoRenderer

        self.assertIsInstance(p.depth_estimator, DepthEstimator)
        self.assertIsInstance(p.stereo_renderer, StereoRenderer)
        self.assertEqual(p.depth_backend_name, "depth-anything")
        self.assertEqual(p.stereo_backend_name, "default")

    def test_default_backends_are_not_wholeclip(self):
        """The default per-frame backends must NOT take the whole-clip path."""
        p = _make_pipeline(output_width=100, output_height=50)
        self.assertFalse(p._is_wholeclip_depth(p.depth_estimator))
        self.assertFalse(p._is_wholeclip_stereo(p.stereo_renderer))


# ---------------------------------------------------------------------------
# Whole-clip detection helpers
# ---------------------------------------------------------------------------


class TestWholeClipDetection(unittest.TestCase):
    """Detection is gated on explicit injection: only a caller-injected backend
    with the whole-clip interface is treated as whole-clip.  A bare MagicMock
    that auto-creates estimate_video/render_video but was NOT injected must not
    be misdetected (this is what protects the pre-I-5 default path)."""

    def test_detects_injected_wholeclip_depth(self):
        p = _make_pipeline(
            output_width=100,
            output_height=50,
            depth_estimator=FakeWholeClipDepth(),
        )
        self.assertTrue(p._is_wholeclip_depth(p.depth_estimator))

    def test_non_injected_depth_is_never_wholeclip(self):
        # Not injected (default DepthEstimator) → per-frame even though a bare
        # MagicMock would expose estimate_video.
        p = _make_pipeline(output_width=100, output_height=50)
        self.assertFalse(p._is_wholeclip_depth(p.depth_estimator))

    def test_injected_per_frame_depth_not_wholeclip(self):
        # Injected but only exposes the per-frame contract (no estimate_video).
        p = _make_pipeline(
            output_width=100,
            output_height=50,
            depth_estimator=MagicMock(spec=["estimate"]),
        )
        self.assertFalse(p._is_wholeclip_depth(p.depth_estimator))

    def test_detects_injected_wholeclip_stereo(self):
        p = _make_pipeline(
            output_width=100,
            output_height=50,
            stereo_renderer=FakeWholeClipStereo(),
        )
        self.assertTrue(p._is_wholeclip_stereo(p.stereo_renderer))

    def test_injected_per_frame_stereo_not_wholeclip(self):
        p = _make_pipeline(
            output_width=100,
            output_height=50,
            stereo_renderer=MagicMock(spec=["render"]),
        )
        self.assertFalse(p._is_wholeclip_stereo(p.stereo_renderer))


# ---------------------------------------------------------------------------
# Whole-clip depth precompute path
# ---------------------------------------------------------------------------


class TestWholeClipDepthPrecompute(unittest.TestCase):
    """--depth-model depthcrafter: estimate_video runs once, per-frame fuse
    loop consumes the precomputed depths (stereo stays the default renderer)."""

    def test_estimate_video_called_once_and_feeds_loop(self):
        depth_backend = FakeWholeClipDepth(num_frames=3)
        p = _make_pipeline(
            output_width=100,
            output_height=50,
            depth_estimator=depth_backend,
            depth_backend_name="depthcrafter",
        )
        # Default stereo renderer (per-frame) — mock render, but spec it so it
        # does NOT auto-create a render_video attribute (which would make the
        # pipeline mistake it for a whole-clip stereo backend).
        p.stereo_renderer = MagicMock(spec=["render"])
        p.stereo_renderer.render.return_value = (
            np.zeros((8, 8, 3), dtype=np.uint8),
            np.zeros((8, 8, 3), dtype=np.uint8),
        )
        p.eq_mapper.map_stereo_pair.return_value = np.zeros((50, 200, 3), dtype=np.uint8)

        cap = _fake_cap(num_frames=3)
        proc = MagicMock()
        proc.returncode = 0

        with (
            patch("pipeline.streaming_pipeline.cv2.VideoCapture", return_value=cap),
            patch("pipeline.streaming_pipeline.subprocess.Popen", return_value=proc),
        ):
            p.process_stream("in.mp4", "out.mp4")

        # The whole-clip depth backend ran exactly once for the whole clip.
        self.assertEqual(len(depth_backend.calls), 1)
        # The default per-frame DepthEstimator.estimate was NOT used.
        # (it was replaced by the injected backend — no .estimate attribute used)
        # The stereo renderer was driven once per frame with precomputed depth.
        self.assertEqual(p.stereo_renderer.render.call_count, 3)
        # The injected depth backend's output (8x8) is what stereo saw, not a
        # per-frame Depth-Anything estimate.
        depth_arg = p.stereo_renderer.render.call_args_list[0][0][1]
        self.assertEqual(depth_arg.shape, (8, 8))


# ---------------------------------------------------------------------------
# Whole-clip stereo precompute path
# ---------------------------------------------------------------------------


class TestWholeClipStereoPrecompute(unittest.TestCase):
    """--stereo-model stereocrafter: render_video runs once, per-frame loop
    reads the L/R frames back and feeds equirect→encode (depth stage skipped)."""

    def test_render_video_called_once_and_feeds_loop(self):
        stereo_backend = FakeWholeClipStereo(num_frames=3)
        p = _make_pipeline(
            output_width=100,
            output_height=50,
            stereo_renderer=stereo_backend,
            stereo_backend_name="stereocrafter",
        )
        # Per-frame depth estimator (mocked).  I-7.2 (#143): since issue #140
        # StereoCrafter consumes the pipeline's OWN depth maps (the in-repo
        # forward-splat replaced the removed upstream Stage 1), so with no
        # whole-clip depth backend injected the streaming path auto-emits
        # per-frame depth maps via this estimator before the stereo call.
        p.depth_estimator = MagicMock(spec=["estimate"])
        p.depth_estimator.estimate.return_value = np.zeros((8, 8), dtype=np.float32)
        p.eq_mapper.map_stereo_pair.return_value = np.zeros((50, 200, 3), dtype=np.uint8)

        cap = _fake_cap(num_frames=3)
        proc = MagicMock()
        proc.returncode = 0

        # Patch _load_video_frames so the test doesn't depend on cv2 reading the
        # fake backend's mp4 back (keeps the test hermetic and CPU-only).  The
        # fake returns 3 L/R frame pairs, matching num_frames.
        left = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(3)]
        right = [np.full((8, 8, 3), 255, dtype=np.uint8) for _ in range(3)]

        with (
            patch("pipeline.streaming_pipeline.cv2.VideoCapture", return_value=cap),
            patch("pipeline.streaming_pipeline.subprocess.Popen", return_value=proc),
            patch("pipeline.streaming_pipeline._load_video_frames", side_effect=[left, right]),
        ):
            p.process_stream("in.mp4", "out.mp4")

        # The whole-clip stereo backend ran exactly once.
        self.assertEqual(len(stereo_backend.calls), 1)
        # eq_mapper was driven once per precomputed L/R pair.
        self.assertEqual(p.eq_mapper.map_stereo_pair.call_count, 3)
        # I-7.2 (#143): the per-frame depth estimator WAS invoked (once per
        # source frame) to auto-emit the depth maps StereoCrafter's in-repo
        # forward-splat consumes — it is NOT invoked inside the fuse loop
        # itself (that loop reads the precomputed L/R frames).
        self.assertEqual(p.depth_estimator.estimate.call_count, 3)

    def test_frames_read_back_from_returned_paths(self):
        """I-7 (#137): the precompute reads L/R frames back from the paths
        ``render_video`` *returns*, not the assumed ``out_left``/``out_right``.

        A backend (or a StereoCrafterRenderer with default-resolved outputs)
        may write to different paths than the caller proposed; the streaming
        path must follow the return value, exactly like the batch
        ``_run_stereocrafter_stage``.
        """
        stereo_backend = FakeWholeClipStereo(num_frames=2)
        p = _make_pipeline(
            output_width=100,
            output_height=50,
            stereo_renderer=stereo_backend,
            stereo_backend_name="stereocrafter",
        )
        p.depth_estimator = MagicMock(spec=["estimate"])
        p.depth_estimator.estimate.return_value = np.zeros((8, 8), dtype=np.float32)
        p.eq_mapper.map_stereo_pair.return_value = np.zeros((50, 200, 3), dtype=np.uint8)

        cap = _fake_cap(num_frames=2)
        proc = MagicMock()
        proc.returncode = 0

        # Return DIFFERENT paths than the caller passed; if the pipeline reads
        # from the returned ones, _load_video_frames sees the returned paths.
        returned_left, returned_right = "actual_left.mp4", "actual_right.mp4"
        stereo_backend.render_video = MagicMock(return_value=(returned_left, returned_right))
        left = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(2)]
        right = [np.full((8, 8, 3), 255, dtype=np.uint8) for _ in range(2)]

        with (
            patch("pipeline.streaming_pipeline.cv2.VideoCapture", return_value=cap),
            patch("pipeline.streaming_pipeline.subprocess.Popen", return_value=proc),
            patch("pipeline.streaming_pipeline._load_video_frames", side_effect=[left, right]) as mock_load,
        ):
            p.process_stream("in.mp4", "out.mp4")

        # _load_video_frames was called with the RETURNED paths, in order.
        loaded_paths = [c.args[0] for c in mock_load.call_args_list]
        self.assertEqual(loaded_paths, [returned_left, returned_right])


class TestWholeClipStereoLogging(unittest.TestCase):
    """I-7 (#137) acceptance: the streaming whole-clip stereo path logs BOTH
    the startup ``stereo=stereocrafter`` line AND the ``🎬 [Stereo]`` real-call
    line in the SAME run — the self-proof the lead greps for.

    The pre-#137 ``TestBackendNameLogging`` mocked the stereo renderer as
    *per-frame* (spec=["render"]), so it never drove the whole-clip stereo
    path and never saw the ``🎬 [Stereo]`` line.  This test injects a real
    whole-clip stereo backend so the actual StereoCrafter branch runs.
    """

    def test_startup_and_real_call_lines_logged_for_wholeclip_stereo(self):
        stereo_backend = FakeWholeClipStereo(num_frames=2)
        p = _make_pipeline(
            output_width=100,
            output_height=50,
            stereo_renderer=stereo_backend,
            stereo_backend_name="stereocrafter",
        )
        p.depth_estimator = MagicMock(spec=["estimate"])
        p.depth_estimator.estimate.return_value = np.zeros((8, 8), dtype=np.float32)
        p.eq_mapper.map_stereo_pair.return_value = np.zeros((50, 200, 3), dtype=np.uint8)

        cap = _fake_cap(num_frames=2)
        proc = MagicMock()
        proc.returncode = 0
        left = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(2)]
        right = [np.full((8, 8, 3), 255, dtype=np.uint8) for _ in range(2)]

        with (
            patch("pipeline.streaming_pipeline.cv2.VideoCapture", return_value=cap),
            patch("pipeline.streaming_pipeline.subprocess.Popen", return_value=proc),
            patch("pipeline.streaming_pipeline._load_video_frames", side_effect=[left, right]),
            self.assertLogs("vr180-streaming", level="INFO") as cm,
        ):
            p.process_stream("in.mp4", "out.mp4")

        joined = "\n".join(cm.output)
        # Startup line names the stereo backend.
        self.assertIn("stereo=stereocrafter", joined)
        # Real-call trace (mirrors the depth side's "🎬 [Depth] ...").
        self.assertIn("🎬 [Stereo]", joined)
        self.assertIn("render_video", joined)
        # Frame-count confirmation line.
        self.assertIn("L/R frame pair(s)", joined)
        # The backend really ran once.
        self.assertEqual(len(stereo_backend.calls), 1)


# ---------------------------------------------------------------------------
# I-7.2 (#143): stereo backend receives the depth stage's REAL output dir
# ---------------------------------------------------------------------------


class FakeWholeClipDepthThatWrites(FakeWholeClipDepth):
    """Whole-clip depth backend that actually checkpoints depth_*.npy files
    into output_dir (like the real DepthCrafterEstimator), so the dir the
    stereo stage receives can be verified to be populated."""

    def estimate_video(self, input_path, output_dir):
        super().estimate_video(input_path, output_dir)
        os.makedirs(output_dir, exist_ok=True)
        rng = np.random.default_rng(7)
        for i in range(self.num_frames):
            np.save(
                os.path.join(output_dir, f"depth_{i:06d}.npy"),
                rng.random((self.h, self.w)).astype(np.float32),
            )
        return [np.zeros((self.h, self.w), dtype=np.float32) for _ in range(self.num_frames)]


class TestStereoReceivesRealDepthDir(unittest.TestCase):
    """I-7.2 (#143): the pre-#143 streaming path created TWO independent temp
    dirs — the depth stage wrote its maps into ``vr180-streaming-depth_XXXX``
    while the stereo backend was handed a fresh, guaranteed-empty
    ``vr180-streaming-stereo_YYYY/depth`` and crashed inside the splat assembly
    ("No depth maps found in ...").  These tests assert ``render_video`` gets
    the depth stage's REAL output dir, populated and still on disk."""

    def _run(self, p, cap_frames=3):
        cap = _fake_cap(num_frames=cap_frames)
        proc = MagicMock()
        proc.returncode = 0
        left = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(cap_frames)]
        right = [np.full((8, 8, 3), 255, dtype=np.uint8) for _ in range(cap_frames)]
        with (
            patch("pipeline.streaming_pipeline.cv2.VideoCapture", return_value=cap),
            patch("pipeline.streaming_pipeline.subprocess.Popen", return_value=proc),
            patch("pipeline.streaming_pipeline._load_video_frames", side_effect=[left, right]),
        ):
            p.process_stream("in.mp4", "out.mp4")

    def test_depthcrafter_dir_is_handed_to_stereo(self):
        """--depth-model depthcrafter + --stereo-model stereocrafter: the dir
        passed to render_video is EXACTLY the one estimate_video wrote into
        (this is the assertion that would have caught the defect)."""
        depth_backend = FakeWholeClipDepthThatWrites(num_frames=3)
        stereo_backend = FakeWholeClipStereo(num_frames=3)
        p = _make_pipeline(
            output_width=100,
            output_height=50,
            depth_estimator=depth_backend,
            stereo_renderer=stereo_backend,
            depth_backend_name="depthcrafter",
            stereo_backend_name="stereocrafter",
        )
        p.eq_mapper.map_stereo_pair.return_value = np.zeros((50, 200, 3), dtype=np.uint8)

        self._run(p)

        # Both whole-clip backends ran exactly once.
        self.assertEqual(len(depth_backend.calls), 1)
        self.assertEqual(len(stereo_backend.calls), 1)
        depth_out_dir = depth_backend.calls[0][1]
        stereo_depth_dir = stereo_backend.calls[0][1]
        # The stereo backend received the depth stage's real output dir.
        self.assertEqual(stereo_depth_dir, depth_out_dir)
        # …and it was populated + alive through the stereo call.
        self.assertTrue(os.path.isdir(stereo_depth_dir))
        self.assertEqual(len([f for f in os.listdir(stereo_depth_dir) if f.endswith(".npy")]), 3)

    def test_depth_dir_alive_after_stereo_call(self):
        """Lifetime: the depth dir must not be cleaned up before/during the
        stereo stage — it is verified to still exist after process_stream."""
        depth_backend = FakeWholeClipDepthThatWrites(num_frames=2)
        stereo_backend = FakeWholeClipStereo(num_frames=2)

        # Assert from inside the stereo call: the dir is populated AT CALL TIME.
        seen = {}

        original_render_video = stereo_backend.render_video

        def spy_render_video(input_path, depth_dir, output_left, output_right):
            seen["depth_dir"] = depth_dir
            seen["npy_count_at_call"] = len([f for f in os.listdir(depth_dir) if f.endswith(".npy")])
            return original_render_video(input_path, depth_dir, output_left, output_right)

        stereo_backend.render_video = spy_render_video

        p = _make_pipeline(
            output_width=100,
            output_height=50,
            depth_estimator=depth_backend,
            stereo_renderer=stereo_backend,
            depth_backend_name="depthcrafter",
            stereo_backend_name="stereocrafter",
        )
        p.eq_mapper.map_stereo_pair.return_value = np.zeros((50, 200, 3), dtype=np.uint8)

        self._run(p, cap_frames=2)

        self.assertIn("depth_dir", seen)
        self.assertEqual(seen["npy_count_at_call"], 2)
        # Still on disk after the whole run (not deleted at depth-stage exit).
        self.assertTrue(os.path.isdir(seen["depth_dir"]))

    def test_stereo_alone_auto_emits_perframe_depths(self):
        """--stereo-model stereocrafter WITHOUT a whole-clip depth backend:
        the streaming path must not hand StereoCrafter an empty dir (the
        pre-#143 crash).  Instead it auto-emits per-frame depth maps with the
        per-frame estimator and hands over THAT populated dir."""
        stereo_backend = FakeWholeClipStereo(num_frames=3)
        p = _make_pipeline(
            output_width=100,
            output_height=50,
            stereo_renderer=stereo_backend,
            stereo_backend_name="stereocrafter",
        )
        # Injected per-frame estimator (Depth-Anything stand-in).
        p.depth_estimator = MagicMock(spec=["estimate"])
        p.depth_estimator.estimate.return_value = np.full((8, 8), 0.5, dtype=np.float32)
        p.eq_mapper.map_stereo_pair.return_value = np.zeros((50, 200, 3), dtype=np.uint8)

        self._run(p)

        self.assertEqual(len(stereo_backend.calls), 1)
        stereo_depth_dir = stereo_backend.calls[0][1]
        # The dir the stereo backend got is populated with per-frame maps.
        self.assertTrue(os.path.isdir(stereo_depth_dir))
        self.assertEqual(len([f for f in os.listdir(stereo_depth_dir) if f.endswith(".npy")]), 3)
        # The per-frame estimator ran once per frame (the auto-emit path).
        self.assertEqual(p.depth_estimator.estimate.call_count, 3)


# ---------------------------------------------------------------------------
# Startup backend-name logging
# ---------------------------------------------------------------------------


class TestBackendNameLogging(unittest.TestCase):
    def test_startup_logs_effective_backend_names(self):
        # Inject at construction time so the injected flags are set and the
        # whole-clip depth path is exercised.
        stereo = MagicMock(spec=["render"])
        stereo.render.return_value = (
            np.zeros((8, 8, 3), dtype=np.uint8),
            np.zeros((8, 8, 3), dtype=np.uint8),
        )
        p = _make_pipeline(
            output_width=100,
            output_height=50,
            depth_estimator=FakeWholeClipDepth(num_frames=1),
            stereo_renderer=stereo,
            depth_backend_name="depthcrafter",
            stereo_backend_name="stereocrafter",
        )
        p.eq_mapper.map_stereo_pair.return_value = np.zeros((50, 200, 3), dtype=np.uint8)

        cap = _fake_cap(num_frames=1)
        proc = MagicMock()
        proc.returncode = 0

        with (
            patch("pipeline.streaming_pipeline.cv2.VideoCapture", return_value=cap),
            patch("pipeline.streaming_pipeline.subprocess.Popen", return_value=proc),
            self.assertLogs("vr180-streaming", level="INFO") as cm,
        ):
            p.process_stream("in.mp4", "out.mp4")

        joined = "\n".join(cm.output)
        self.assertIn("depth=depthcrafter", joined)
        self.assertIn("stereo=stereocrafter", joined)

    def test_startup_logs_default_names_when_not_injected(self):
        p = _make_pipeline(output_width=100, output_height=50)
        p.depth_estimator = MagicMock(spec=["estimate"])
        p.depth_estimator.estimate.return_value = np.zeros((8, 8), dtype=np.float32)
        # spec=["render"] so the mock is NOT mistaken for a whole-clip backend.
        p.stereo_renderer = MagicMock(spec=["render"])
        p.stereo_renderer.render.return_value = (
            np.zeros((8, 8, 3), dtype=np.uint8),
            np.zeros((8, 8, 3), dtype=np.uint8),
        )
        p.eq_mapper.map_stereo_pair.return_value = np.zeros((50, 200, 3), dtype=np.uint8)

        cap = _fake_cap(num_frames=1)
        proc = MagicMock()
        proc.returncode = 0

        with (
            patch("pipeline.streaming_pipeline.cv2.VideoCapture", return_value=cap),
            patch("pipeline.streaming_pipeline.subprocess.Popen", return_value=proc),
            self.assertLogs("vr180-streaming", level="INFO") as cm,
        ):
            p.process_stream("in.mp4", "out.mp4")

        joined = "\n".join(cm.output)
        self.assertIn("depth=depth-anything", joined)
        self.assertIn("stereo=default", joined)


# ---------------------------------------------------------------------------
# run_pipeline factory: fallback policy (no silent degrade)
# ---------------------------------------------------------------------------


class TestRunPipelineFactory(unittest.TestCase):
    """The I-5 factory: requested-but-unavailable backend → loud WARNING +
    fallback (streaming), or hard-fail (batch).  Never silent."""

    def _import_run_pipeline(self):
        sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
        try:
            import run_pipeline

            return run_pipeline
        finally:
            pass

    def setUp(self):
        self.rp = self._import_run_pipeline()

    def tearDown(self):
        sys.modules.pop("run_pipeline", None)
        with __import__("contextlib").suppress(ValueError):
            sys.path.remove(os.path.join(PROJECT_ROOT, "scripts"))

    def test_default_depth_model_returns_none_backend(self):
        args = MagicMock()
        args.depth_model = "depth-anything"
        est, name = self.rp.build_depth_backend(args)
        self.assertIsNone(est)
        self.assertEqual(name, "depth-anything")

    def test_default_stereo_model_returns_none_backend(self):
        args = MagicMock()
        args.stereo_model = "default"
        r, name = self.rp.build_stereo_backend(args)
        self.assertIsNone(r)
        self.assertEqual(name, "default")

    def test_depthcrafter_unavailable_falls_back_with_warning(self):
        """CUDA missing → DepthCrafter constructor raises → WARNING + fallback."""
        args = MagicMock()
        args.depth_model = "depthcrafter"
        args.depthcrafter_repo_dir = None
        args.depthcrafter_python = None
        args.depthcrafter_checkpoint_dir = None
        args.depthcrafter_max_res = None

        with (
            patch("pipeline.depth_crafter._assert_cuda", side_effect=RuntimeError("CUDA is not available")),
            self.assertLogs("vr180-pipeline", level="WARNING") as cm,
        ):
            est, name = self.rp.build_depth_backend(args, fallback=True)

        self.assertIsNone(est)
        self.assertEqual(name, "depth-anything")
        joined = "\n".join(cm.output)
        self.assertIn("FALLING BACK", joined)
        self.assertIn("depthcrafter", joined)

    def test_stereocrafter_unavailable_falls_back_with_warning(self):
        args = MagicMock()
        args.stereo_model = "stereocrafter"
        args.stereocrafter_repo_dir = None
        args.stereocrafter_python = None
        args.stereocrafter_checkpoint_dir = None
        args.stereocrafter_max_res = None

        with (
            patch("pipeline.stereo_crafter._assert_cuda", side_effect=RuntimeError("CUDA is not available")),
            self.assertLogs("vr180-pipeline", level="WARNING") as cm,
        ):
            r, name = self.rp.build_stereo_backend(args, fallback=True)

        self.assertIsNone(r)
        self.assertEqual(name, "default")
        self.assertIn("FALLING BACK", "\n".join(cm.output))

    def test_batch_path_hard_fails_when_unavailable(self):
        """fallback=False (batch stages) preserves the pre-I-5 hard-fail contract."""
        args = MagicMock()
        args.depth_model = "depthcrafter"
        args.depthcrafter_repo_dir = None
        args.depthcrafter_python = None
        args.depthcrafter_checkpoint_dir = None
        args.depthcrafter_max_res = None

        with (
            patch("pipeline.depth_crafter._assert_cuda", side_effect=RuntimeError("CUDA is not available")),
            self.assertRaises(RuntimeError),
        ):
            self.rp.build_depth_backend(args, fallback=False)

    def test_factory_constructs_depthcrafter_when_available(self):
        """When the backend constructs cleanly, it is returned (not None)."""
        args = MagicMock()
        args.depth_model = "depthcrafter"
        args.depthcrafter_repo_dir = None
        args.depthcrafter_python = None
        args.depthcrafter_checkpoint_dir = None
        args.depthcrafter_max_res = None

        sentinel = MagicMock()
        with patch.object(self.rp, "DepthCrafterEstimator", return_value=sentinel):
            est, name = self.rp.build_depth_backend(args, fallback=True)
        self.assertIs(est, sentinel)
        self.assertEqual(name, "depthcrafter")


# ---------------------------------------------------------------------------
# run_pipeline streaming branch wires the factory into StreamingPipeline
# ---------------------------------------------------------------------------


class TestStreamingBranchInjection(unittest.TestCase):
    """The streaming CLI branch must inject the factory-built backends into
    StreamingPipeline (this is the wiring that was missing pre-I-5)."""

    def _run_main(self):
        sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
        try:
            import run_pipeline

            args = MagicMock()
            args.video_upscale = "none"
            args.device = "cpu"
            args.validate_input = False
            args.fps = 30
            args.streaming = True
            args.stage = "all"
            args.projection = "vr180"
            args.model_size = "small"
            args.ipd = 0.064
            args.max_disparity = 0.05
            args.output_width = 2880
            args.output_height = 2880
            args.src_hfov = 70.0
            args.codec = "h264"
            args.crf = 23
            args.bitrate = "45M"
            args.input = "in.mp4"
            args.output = "out.mp4"
            args.max_frames = None
            args.comfort = "balanced"
            args.convergence = None
            args.no_temporal = False
            args.preset = "source"
            args.gop = None
            # I-5: request the advanced backends.
            args.depth_model = "depthcrafter"
            args.stereo_model = "stereocrafter"
            # H-1.2 (#132): bare MagicMock attributes are truthy — pin the
            # audio-passthrough flag so the branch under test is explicit.
            args.copy_audio_from = None

            captured = {}

            def fake_pipeline_ctor(**kwargs):
                captured.update(kwargs)
                inst = MagicMock()
                inst.process_stream.return_value = "out.mp4"
                return inst

            fake_depth = MagicMock(name="FakeDepthCrafter")
            fake_stereo = MagicMock(name="FakeStereoCrafter")

            with (
                patch.object(run_pipeline, "parse_args", return_value=args),
                patch.object(run_pipeline, "apply_quality_preset"),
                patch.object(run_pipeline, "build_depth_backend", return_value=(fake_depth, "depthcrafter")),
                patch.object(
                    run_pipeline, "build_stereo_backend", return_value=(fake_stereo, "stereocrafter")
                ) as mock_stereo_factory,
                patch.object(run_pipeline, "StreamingPipeline", side_effect=fake_pipeline_ctor),
                # H-1.2 (#132): the streaming branch now remuxes audio after
                # sv3d/st3d injection.  Stub the passthrough helpers so no
                # real ffmpeg/ffprobe runs on CI; the new TestStreamingAudio
                # class below asserts this wiring directly.
                patch.object(run_pipeline, "_copy_audio_to_output"),
                patch.object(run_pipeline, "_maybe_copy_audio_from_input"),
                patch.object(run_pipeline, "_write_sidecar_from_args"),
                patch("pipeline.spherical_injector.inject_spherical_metadata"),
                patch("os.replace"),
            ):
                run_pipeline.main()
            return captured, mock_stereo_factory
        finally:
            import contextlib

            with contextlib.suppress(ValueError):
                sys.path.remove(os.path.join(PROJECT_ROOT, "scripts"))
            sys.modules.pop("run_pipeline", None)

    def test_streaming_branch_injects_factory_backends(self):
        captured, _ = self._run_main()
        # StreamingPipeline must have been constructed with the injected
        # backends + their names — this is the exact wiring that was missing.
        self.assertIn("depth_estimator", captured)
        self.assertIn("stereo_renderer", captured)
        self.assertEqual(captured.get("depth_backend_name"), "depthcrafter")
        self.assertEqual(captured.get("stereo_backend_name"), "stereocrafter")

    def test_streaming_branch_uses_fallback_policy_for_stereo(self):
        """I-7 (#137) acceptance: the streaming branch builds the stereo backend
        with ``fallback=True`` — an unavailable StereoCrafter degrades to the
        default renderer with a loud WARNING, never a silent wrong-model run and
        never a hard crash of the whole streaming job."""
        _, mock_stereo_factory = self._run_main()
        mock_stereo_factory.assert_called_once()
        _, kwargs = mock_stereo_factory.call_args
        self.assertEqual(kwargs.get("fallback"), True)


# ---------------------------------------------------------------------------
# H-1.2 (#132): streaming branch remuxes audio after sv3d/st3d injection
# ---------------------------------------------------------------------------


class TestStreamingAudioPassthrough(unittest.TestCase):
    """The streaming CLI branch (--quality standard|high) must run the same
    H-1 audio passthrough as the batch path — pre-H-1.2 it returned before
    reaching it, silently producing a silent video.

    The real module-level helpers (``_copy_audio_to_output`` /
    ``_maybe_copy_audio_from_input``) run here; only ffmpeg/ffprobe
    (``pipeline.audio_mux``) and the metadata injector are mocked, so the
    wiring — including the ``re_inject=True`` re-embedding of sv3d/st3d
    (issue #91) — is genuinely exercised.
    """

    def _run_streaming_main(self, *, copy_audio_from, has_audio):
        sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
        try:
            import run_pipeline

            args = MagicMock()
            args.video_upscale = "none"
            args.device = "cpu"
            args.validate_input = False
            args.fps = 30
            args.streaming = True
            args.stage = "all"
            args.projection = "vr180"
            args.model_size = "small"
            args.ipd = 0.064
            args.max_disparity = 0.05
            args.output_width = 2880
            args.output_height = 2880
            args.src_hfov = 70.0
            args.codec = "h264"
            args.crf = 23
            args.bitrate = "45M"
            args.input = "in.mp4"
            args.output = "out.mp4"
            args.max_frames = None
            args.comfort = "balanced"
            args.convergence = None
            args.no_temporal = False
            args.preset = "source"
            args.gop = None
            args.depth_model = "depthcrafter"
            args.stereo_model = "stereocrafter"
            args.copy_audio_from = copy_audio_from

            pipeline_inst = MagicMock()
            pipeline_inst.process_stream.return_value = "out.mp4"

            with (
                patch.object(run_pipeline, "parse_args", return_value=args),
                patch.object(run_pipeline, "apply_quality_preset"),
                patch.object(run_pipeline, "build_depth_backend", return_value=(MagicMock(), "depthcrafter")),
                patch.object(run_pipeline, "build_stereo_backend", return_value=(MagicMock(), "stereocrafter")),
                patch.object(run_pipeline, "StreamingPipeline", return_value=pipeline_inst),
                patch.object(run_pipeline, "_write_sidecar_from_args"),
                patch("pipeline.audio_mux.copy_audio_to") as mock_remux,
                patch("pipeline.audio_mux.has_audio_stream", return_value=has_audio),
                patch("pipeline.spherical_injector.inject_spherical_metadata") as mock_inject,
                patch("os.replace"),
            ):
                run_pipeline.main()
            return mock_remux, mock_inject
        finally:
            import contextlib

            with contextlib.suppress(ValueError):
                sys.path.remove(os.path.join(PROJECT_ROOT, "scripts"))
            sys.modules.pop("run_pipeline", None)

    def test_copy_audio_from_flag_remuxes_and_reinjects(self):
        """--copy-audio-from <src>: streaming output gets the src's audio AND
        sv3d/st3d is re-injected after the remux (issue #91)."""
        mock_remux, mock_inject = self._run_streaming_main(copy_audio_from="src.mp4", has_audio=False)
        mock_remux.assert_called_once_with("out.mp4", "src.mp4")
        # Two injections: post-process_stream sv3d/st3d + post-remux re-inject.
        self.assertEqual(mock_inject.call_count, 2)

    def test_input_audio_remuxed_when_no_flag(self):
        """No flag + input has an audio stream → implicit passthrough from
        the input video, also with sv3d/st3d re-injection."""
        mock_remux, mock_inject = self._run_streaming_main(copy_audio_from=None, has_audio=True)
        mock_remux.assert_called_once_with("out.mp4", "in.mp4")
        self.assertEqual(mock_inject.call_count, 2)

    def test_no_audio_source_is_silent_noop(self):
        """No flag + input has NO audio → no remux, no error (log-only)."""
        mock_remux, mock_inject = self._run_streaming_main(copy_audio_from=None, has_audio=False)
        mock_remux.assert_not_called()
        # Only the post-process_stream injection runs.
        self.assertEqual(mock_inject.call_count, 1)


if __name__ == "__main__":
    unittest.main()
