"""Tests for issue #243 (P0-2) — StreamingPipeline honours --outpaint / --no-ffmpeg-v360.

The streaming path (``--quality standard|high`` ⇒ ``StreamingPipeline``)
previously **silently dropped** several CLI flags — the same anti-pattern as
#120 (``--depth-model`` swallowed).  #243 is the second occurrence of the same
defect class: ``--outpaint`` (and its sub-params) and ``--no-ffmpeg-v360`` were
accepted by argparse but never threaded into :class:`StreamingPipeline`, so an
operator could pass them and get behaviour indistinguishable from not passing
them — with no warning.

These tests cover the #243 fix and its **defense-in-depth** (the generic
"swallowed-arg" detector in ``run_pipeline._warn_streaming_unsupported_args``):

  - ``--outpaint gradient`` ⇒ StreamingPipeline is constructed with
    ``outpaint="gradient"`` (and the mask sub-params forwarded).
  - ``--no-ffmpeg-v360`` ⇒ ``use_ffmpeg=False`` is passed to the constructor
    (this assertion is RED before the fix — the old path hard-coded
    ``use_ffmpeg=True``).
  - Omitting the flags ⇒ behaviour bit-exact with pre-#243 (defaults: outpaint
    "none", use_ffmpeg True).
  - Explicitly passing a flag the streaming path does NOT honour ⇒ a WARNING
    is logged that names the ignored flag.
  - Existing call sites that do not pass the new params still construct the
    pipeline (default-value verification — the #168 regression guard).

All tests are CPU-only; cv2 capture / ffmpeg / model stages are mocked or
faked.  No real model download, no real inference, no real API calls.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pipeline.streaming_pipeline import StreamingPipeline  # noqa: E402

# ---------------------------------------------------------------------------
# StreamingPipeline constructor: outpaint / use_ffmpeg passthrough
# ---------------------------------------------------------------------------


def _make_pipeline(**kwargs):
    """Build a StreamingPipeline with the heavy EquirectangularMapper mocked."""
    with patch("pipeline.streaming_pipeline.EquirectangularMapper"):
        kwargs.setdefault("device", "cpu")
        return StreamingPipeline(**kwargs)


class TestOutpaintPassthrough(unittest.TestCase):
    """--outpaint (and sub-params) must reach StreamingPipeline, not be dropped."""

    def test_outpaint_gradient_forwarded_to_constructor(self):
        p = _make_pipeline(
            output_width=100,
            output_height=50,
            outpaint="gradient",
            outpaint_mask_threshold=20,
            outpaint_mask_top_ratio=0.3,
            outpaint_mask_bottom_ratio=0.4,
        )
        self.assertEqual(p.outpaint, "gradient")
        self.assertEqual(p.outpaint_mask_threshold, 20)
        self.assertEqual(p.outpaint_mask_top_ratio, 0.3)
        self.assertEqual(p.outpaint_mask_bottom_ratio, 0.4)

    def test_outpaint_ai_forwarded(self):
        p = _make_pipeline(output_width=100, output_height=50, outpaint="ai")
        self.assertEqual(p.outpaint, "ai")

    def test_outpaint_defaults_to_none(self):
        """Without --outpaint the streaming pipeline stays at the no-op default."""
        p = _make_pipeline(output_width=100, output_height=50)
        self.assertEqual(p.outpaint, "none")
        self.assertEqual(p.outpaint_mask_threshold, 10)
        self.assertEqual(p.outpaint_mask_top_ratio, 0.25)
        self.assertEqual(p.outpaint_mask_bottom_ratio, 0.25)


class TestNoFfmpegV360Passthrough(unittest.TestCase):
    """--no-ffmpeg-v360 must flip use_ffmpeg to False on the streaming path.

    This assertion is RED before the #243 fix: the old constructor hard-coded
    ``use_ffmpeg=True`` in the EquirectangularMapper and did not even accept the
    parameter.
    """

    def test_use_ffmpeg_default_true(self):
        p = _make_pipeline(output_width=100, output_height=50)
        self.assertTrue(p.use_ffmpeg)

    def test_no_ffmpeg_v360_disables_ffmpeg_path(self):
        p = _make_pipeline(output_width=100, output_height=50, use_ffmpeg=False)
        self.assertFalse(p.use_ffmpeg)

    def test_use_ffmpeg_reaches_equirect_mapper(self):
        """The constructor must hand use_ffmpeg to EquirectangularMapper, not
        hard-code True (the #243 defect)."""
        with patch("pipeline.streaming_pipeline.EquirectangularMapper") as mock_mapper:
            StreamingPipeline(
                device="cpu",
                output_width=100,
                output_height=50,
                use_ffmpeg=False,
            )
        _, kwargs = mock_mapper.call_args
        self.assertIs(kwargs.get("use_ffmpeg"), False)

    def test_use_ffmpeg_true_reaches_equirect_mapper(self):
        with patch("pipeline.streaming_pipeline.EquirectangularMapper") as mock_mapper:
            StreamingPipeline(
                device="cpu",
                output_width=100,
                output_height=50,
                use_ffmpeg=True,
            )
        _, kwargs = mock_mapper.call_args
        self.assertIs(kwargs.get("use_ffmpeg"), True)


class TestDefaultsUnchangedRegression(unittest.TestCase):
    """Regression: omitting the new params must produce bit-exact pre-#243 state."""

    def test_default_constructor_still_works(self):
        # No outpaint/use_ffmpeg kwargs at all — pre-#243 call sites.
        p = _make_pipeline()
        self.assertEqual(p.outpaint, "none")
        self.assertTrue(p.use_ffmpeg)

    def test_existing_positional_call_site(self):
        """Existing callers that pass only the original kwargs construct fine."""
        p = _make_pipeline(
            model_size="base",
            device="cpu",
            ipd=0.07,
            max_disparity=0.08,
            output_width=1920,
            output_height=960,
            codec="h265",
            crf=18,
            fps=60,
        )
        # The new params took their defaults — no behaviour change.
        self.assertEqual(p.outpaint, "none")
        self.assertTrue(p.use_ffmpeg)
        self.assertEqual(p.codec, "h265")


# ---------------------------------------------------------------------------
# run_pipeline streaming branch: wiring + swallowed-arg detector
# ---------------------------------------------------------------------------


def _import_run_pipeline():
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
    try:
        import run_pipeline

        return run_pipeline
    finally:
        pass


def _make_streaming_magic_args(**overrides):
    """Build a MagicMock args namespace for the streaming branch.

    MagicMock is used (matching test_streaming_backends.py's convention) so
    run_pipeline.main()'s attribute lookups that the wiring tests do not care
    about auto-create harmless truthy values; every flag the test asserts on
    is pinned to a concrete value so assertions are deterministic.
    """
    args = MagicMock()
    args.input = "in.mp4"
    args.output = "out.mp4"
    args.inputs = None
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
    args.max_frames = None
    args.comfort = "balanced"
    args.convergence = None
    args.no_temporal = False
    args.preset = "source"
    args.gop = None
    args.depth_model = "depth-anything"
    args.stereo_model = "default"
    args.copy_audio_from = None
    args.no_ffmpeg_v360 = False
    args.outpaint = "none"
    args.outpaint_mask_threshold = 10
    args.outpaint_mask_top_ratio = 0.25
    args.outpaint_mask_bottom_ratio = 0.25
    args.temp_dir = None
    args.keep_temp = False
    args.preflight = "warn"
    args.hardware_encoder = False
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


class TestStreamingBranchOutpaintWiring(unittest.TestCase):
    """The streaming CLI branch must forward --outpaint into StreamingPipeline."""

    def setUp(self):
        self.rp = _import_run_pipeline()

    def tearDown(self):
        sys.modules.pop("run_pipeline", None)
        import contextlib

        with contextlib.suppress(ValueError):
            sys.path.remove(os.path.join(PROJECT_ROOT, "scripts"))

    def _run_main(self, args):
        captured = {}

        def fake_pipeline_ctor(**kwargs):
            captured.update(kwargs)
            inst = MagicMock()
            inst.process_stream.return_value = "out.mp4"
            return inst

        with (
            patch.object(self.rp, "parse_args", return_value=args),
            patch.object(self.rp, "apply_quality_preset"),
            patch.object(self.rp, "build_depth_backend", return_value=(None, "depth-anything")),
            patch.object(self.rp, "build_stereo_backend", return_value=(None, "default")),
            patch.object(self.rp, "StreamingPipeline", side_effect=fake_pipeline_ctor),
            patch.object(self.rp, "_copy_audio_to_output"),
            patch.object(self.rp, "_maybe_copy_audio_from_input"),
            patch.object(self.rp, "_write_sidecar_from_args"),
            patch("pipeline.spherical_injector.inject_spherical_metadata"),
            patch("os.replace"),
        ):
            self.rp.main()
        return captured

    def test_outpaint_gradient_reaches_constructor(self):
        args = _make_streaming_magic_args(outpaint="gradient")
        captured = self._run_main(args)
        self.assertEqual(captured.get("outpaint"), "gradient")
        self.assertEqual(captured.get("outpaint_mask_threshold"), 10)
        self.assertEqual(captured.get("outpaint_mask_top_ratio"), 0.25)
        self.assertEqual(captured.get("outpaint_mask_bottom_ratio"), 0.25)

    def test_outpaint_ai_with_custom_mask_params(self):
        args = _make_streaming_magic_args(
            outpaint="ai",
            outpaint_mask_threshold=42,
            outpaint_mask_top_ratio=0.5,
            outpaint_mask_bottom_ratio=0.1,
        )
        captured = self._run_main(args)
        self.assertEqual(captured.get("outpaint"), "ai")
        self.assertEqual(captured.get("outpaint_mask_threshold"), 42)
        self.assertEqual(captured.get("outpaint_mask_top_ratio"), 0.5)
        self.assertEqual(captured.get("outpaint_mask_bottom_ratio"), 0.1)

    def test_no_ffmpeg_v360_reaches_constructor_as_false(self):
        """This assertion is RED before the #243 fix: use_ffmpeg was hard-coded."""
        args = _make_streaming_magic_args(no_ffmpeg_v360=True)
        captured = self._run_main(args)
        self.assertIs(captured.get("use_ffmpeg"), False)

    def test_default_run_use_ffmpeg_true(self):
        """Without --no-ffmpeg-v360 the stream keeps ffmpeg v360 on (regression)."""
        args = _make_streaming_magic_args()
        captured = self._run_main(args)
        self.assertIs(captured.get("use_ffmpeg"), True)
        self.assertEqual(captured.get("outpaint"), "none")


class TestSwallowedArgDetector(unittest.TestCase):
    """The generic 'swallowed-arg' defense: an explicitly-passed flag the
    streaming path does NOT honour must surface as a WARNING naming the flag."""

    def setUp(self):
        self.rp = _import_run_pipeline()

    def tearDown(self):
        sys.modules.pop("run_pipeline", None)
        import contextlib

        with contextlib.suppress(ValueError):
            sys.path.remove(os.path.join(PROJECT_ROOT, "scripts"))

    def _detector(self, args):
        """Call _warn_streaming_unsupported_args and capture log output."""
        with self.assertLogs("vr180-pipeline", level="WARNING") as cm:
            self.rp._warn_streaming_unsupported_args(args)
        return "\n".join(cm.output)

    def _make_realistic_args(self, **overrides):
        """Build an args namespace from the real argparse spec so defaults match
        what _warn_streaming_unsupported_args compares against."""
        args = self.rp.parse_args([])
        # apply_quality_preset mutates output_width/height/bitrate/streaming;
        # mirror main()'s preset resolution so the streaming branch is active.
        args.streaming = True
        args.stage = "all"
        args.input = "in.mp4"
        args.output = "out.mp4"
        # Resolve quality preset exactly like main() so output_width/height and
        # streaming are concrete (the detector compares against parse_args([])
        # defaults, so we must keep those defaults for the flags under test).
        for k, v in overrides.items():
            setattr(args, k, v)
        return args

    def test_unchanged_defaults_emit_no_warning(self):
        """A vanilla streaming run (no unsupported flag set) must NOT warn."""
        args = self._make_realistic_args()
        # assertLogs would fail if nothing is logged — wrap in assertNoLogs
        # (3.10+) or check via a recorder.  Use a capturing handler instead.
        import logging

        logger = logging.getLogger("vr180-pipeline")
        records: list[logging.LogRecord] = []

        class _Handler(logging.Handler):
            def emit(self, record):
                records.append(record)

        h = _Handler(level=logging.WARNING)
        logger.addHandler(h)
        try:
            self.rp._warn_streaming_unsupported_args(args)
        finally:
            logger.removeHandler(h)
        warned = [r.getMessage() for r in records]
        self.assertEqual(warned, [], f"unexpected warnings: {warned}")

    def test_unsupported_flag_warns_and_names_it(self):
        """Passing a flag the stream ignores (e.g. --upscale) warns and lists it."""
        # --upscale 2 is a batch-only stage the streaming path never runs.
        args = self._make_realistic_args(upscale=2)
        out = self._detector(args)
        self.assertIn("IGNORED", out)
        self.assertIn("--upscale", out)

    def test_no_equirect_batched_warns(self):
        """--no-equirect-batched is a no-op on the stream (always per-frame)."""
        args = self._make_realistic_args(no_equirect_batched=True)
        out = self._detector(args)
        self.assertIn("--no-equirect-batched", out)

    def test_supported_outpaint_does_not_warn(self):
        """--outpaint gradient is now supported (the #243 fix) — no warning."""
        args = self._make_realistic_args(outpaint="gradient")
        import logging

        logger = logging.getLogger("vr180-pipeline")
        records: list[logging.LogRecord] = []

        class _Handler(logging.Handler):
            def emit(self, record):
                records.append(record)

        h = _Handler(level=logging.WARNING)
        logger.addHandler(h)
        try:
            self.rp._warn_streaming_unsupported_args(args)
        finally:
            logger.removeHandler(h)
        warned = [r.getMessage() for r in records]
        # outpaint is in _STREAMING_SUPPORTED, so it must NOT be reported.
        self.assertFalse(
            any("--outpaint" in m for m in warned),
            f"--outpaint should not be warned as unsupported: {warned}",
        )

    def test_no_ffmpeg_v360_supported_does_not_warn(self):
        args = self._make_realistic_args(no_ffmpeg_v360=True)
        import logging

        logger = logging.getLogger("vr180-pipeline")
        records: list[logging.LogRecord] = []

        class _Handler(logging.Handler):
            def emit(self, record):
                records.append(record)

        h = _Handler(level=logging.WARNING)
        logger.addHandler(h)
        try:
            self.rp._warn_streaming_unsupported_args(args)
        finally:
            logger.removeHandler(h)
        warned = [r.getMessage() for r in records]
        self.assertFalse(
            any("--no-ffmpeg-v360" in m for m in warned),
            f"--no-ffmpeg-v360 should not be warned: {warned}",
        )

    def test_non_streaming_run_does_not_warn(self):
        """The detector must be a no-op outside the streaming branch."""
        args = self._make_realistic_args()
        args.streaming = False  # batch path
        import logging

        logger = logging.getLogger("vr180-pipeline")
        records: list[logging.LogRecord] = []

        class _Handler(logging.Handler):
            def emit(self, record):
                records.append(record)

        h = _Handler(level=logging.WARNING)
        logger.addHandler(h)
        try:
            self.rp._warn_streaming_unsupported_args(args)
        finally:
            logger.removeHandler(h)
        self.assertEqual([r.getMessage() for r in records], [])


class TestExistingCallSitesConstruct(unittest.TestCase):
    """#168 guard: adding constructor params must NOT break existing call sites
    that do not pass them.  All defaults verify."""

    def test_no_arg_pipeline_constructs(self):
        p = _make_pipeline()
        self.assertIsInstance(p, StreamingPipeline)

    def test_run_streaming_pipeline_convenience_defaults(self):
        from pipeline.streaming_pipeline import run_streaming_pipeline

        # The convenience wrapper must still construct with only the original
        # required args (the new outpaint/use_ffmpeg params are optional).
        with patch("pipeline.streaming_pipeline.EquirectangularMapper"):
            # We cannot call process_stream (no real video), but construction
            # alone is the regression assertion.
            p = StreamingPipeline(output_width=100, output_height=50, device="cpu")
        self.assertEqual(p.outpaint, "none")
        self.assertTrue(p.use_ffmpeg)
        self.assertTrue(callable(run_streaming_pipeline))


if __name__ == "__main__":
    unittest.main()
