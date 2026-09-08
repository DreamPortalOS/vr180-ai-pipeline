"""W-4 (#314): the ``--source-check`` gate fires *before* the SeedVR2 pre-stage.

W-2 (#311) wired ``scripts/check_source_quality.py`` into ``run_pipeline.main()``
but hung the call next to the P-4b preflight, which sits well below the R-1
SeedVR2 pre-stage.  ``run_seedvr2_prestage`` rewrites ``args.input`` to the
upscaled intermediate, so that placement had two defects:

1. **It saved nothing on the runs it fired.**  A ``strict`` rejection came back
   only after the upscale had already run — the single most expensive step on
   a 12GB card.  A gate whose whole justification is "don't pay for a source
   that can't work" must not itself be charged the biggest bill on the page.
2. **It graded the wrong file.**  The four checks ran against the upscaled
   intermediate rather than the footage the operator handed us.  Upscaling
   does not change the camera move, so ``forward_motion`` — the check the
   entire VR180 route rests on — is only meaningful on the original, and a
   letterbox bar is likewise a property of the source, not of a derivative.

This file pins the ordering and the graded path.  It is deliberately separate
from ``tests/test_run_pipeline_source_check.py`` (W-2's behaviour suite, which
this card does not touch): everything here is about *where* the call happens.

Mutation check the card asks for: move ``_run_source_check(args)`` back below
``_run_preflight(args)`` and every ordering assertion in this file goes red.

Everything is mocked — no ffmpeg, no models, no real video files — and every
path a test hands to ``main()`` lives under ``tmp_path``, so the suite never
writes into the repo.
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from scripts.check_source_quality import (
    STATUS_FAIL,
    STATUS_PASS,
    CheckResult,
    SourceReport,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _import_run_pipeline():
    """Load scripts/run_pipeline.py as an isolated module (V-4 test convention)."""
    scripts_dir = os.path.join(PROJECT_ROOT, "scripts")
    sys.path.insert(0, scripts_dir)
    try:
        name = f"run_pipeline_sco{os.getpid()}_{id(__file__)}"
        spec = importlib.util.spec_from_file_location(
            name,
            os.path.join(scripts_dir, "run_pipeline.py"),
        )
        assert spec is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        with contextlib.suppress(ValueError):
            sys.path.remove(scripts_dir)


@pytest.fixture(scope="module")
def run_pipeline():
    return _import_run_pipeline()


def _passing_report(source: str = "in.mp4") -> SourceReport:
    return SourceReport(
        source=source,
        checks=[
            CheckResult("decodable", STATUS_PASS, "h264 2880x2880 @30fps"),
            CheckResult("square", STATUS_PASS, "2880x2880 (1:1)"),
            CheckResult("forward_motion", STATUS_PASS, "径向外流 +3.61px"),
            CheckResult("edges", STATUS_PASS, "四边干净"),
        ],
    )


def _failing_report(source: str = "in.mp4") -> SourceReport:
    """A locked-off shot: no parallax, so the conversion cannot work at all."""
    return SourceReport(
        source=source,
        checks=[
            CheckResult("decodable", STATUS_PASS, "h264 2880x2880 @30fps"),
            CheckResult("square", STATUS_PASS, "2880x2880 (1:1)"),
            CheckResult(
                "forward_motion",
                STATUS_FAIL,
                "径向外流 -0.0002px（静态机位）",
                {"radial_rate": -0.0002},
                "重新生成素材，提示词强调镜头持续向前飞行。",
            ),
            CheckResult("edges", STATUS_PASS, "四边干净"),
        ],
    )


#: Module-level names in run_pipeline.py whose construction costs real GPU or
#: model-load time.  A strict rejection must leave every one of them untouched
#: — SeedVR2Upscaler above all, since it is the one this card moved past.
_HEAVY_NAMES = (
    "run_seedvr2_prestage",
    "SeedVR2Upscaler",
    "build_depth_backend",
    "build_stereo_backend",
    "DepthEstimator",
    "DepthCrafterEstimator",
    "StereoRenderer",
    "StereoCrafterRenderer",
    "StreamingPipeline",
)


def _main_args(tmp_path: Path, **overrides):
    """Args for a minimal streaming run through ``main()``.

    Every filesystem-facing attribute points into ``tmp_path``: a MagicMock
    left on ``temp_dir`` would make ``get_temp_dir`` mkdir a literal
    ``MagicMock/`` tree in the repo root.
    """
    args = MagicMock()
    args.inputs = None
    args.input = str(tmp_path / "in.mp4")
    args.output = str(tmp_path / "out.mp4")
    args.temp_dir = str(tmp_path / "temp")
    args.video_upscale = "none"
    args.video_upscale_factor = 2
    args.device = "cpu"
    args.validate_input = False
    args.fps = 30
    args.streaming = True
    args.stage = "all"
    args.projection = "vr180"
    args.input_projection = "rectilinear"
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
    args.manifest = None
    args.stages = None
    args.resume_from = None
    # The P-4b host-memory preflight is a different gate with a different job;
    # 'off' keeps these tests about the source gate only.
    args.preflight = "off"
    args.source_check = "warn"
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


@contextlib.contextmanager
def _mocked_main(run_pipeline, args, report, *, concat_output: Path | None = None):
    """Run ``main()`` with every expensive site mocked; yield the mock dict.

    ``report`` is the :class:`SourceReport` ``run_checks`` returns.  The yielded
    dict carries ``run_checks``, ``_run_preflight`` and every name in
    :data:`_HEAVY_NAMES` so a caller can assert both call counts and the paths
    those calls received.
    """
    mocks: dict[str, MagicMock] = {}
    with contextlib.ExitStack() as stack:
        mocks["run_checks"] = stack.enter_context(patch.object(run_pipeline, "run_checks", return_value=report))
        for name in _HEAVY_NAMES:
            mocks[name] = stack.enter_context(patch.object(run_pipeline, name))
        mocks["_run_preflight"] = stack.enter_context(patch.object(run_pipeline, "_run_preflight"))
        # run_seedvr2_prestage returns the path it rewrote args.input to; that
        # rewrite is exactly what the gate must not be graded on.
        mocks["run_seedvr2_prestage"].return_value = "upscaled_2x.mp4"
        mocks["build_depth_backend"].return_value = (MagicMock(), "depth-anything")
        mocks["build_stereo_backend"].return_value = (MagicMock(), "default")
        mocks["StreamingPipeline"].return_value.process_stream.return_value = args.output
        if concat_output is not None:
            mocks["_concat_segments_preprocess"] = stack.enter_context(
                patch.object(run_pipeline, "_concat_segments_preprocess", return_value=concat_output)
            )
        stack.enter_context(patch.object(run_pipeline, "parse_args", return_value=args))
        stack.enter_context(patch.object(run_pipeline, "apply_quality_preset"))
        stack.enter_context(patch.object(run_pipeline, "validate_input_projection"))
        stack.enter_context(patch.object(run_pipeline, "_copy_audio_to_output"))
        stack.enter_context(patch.object(run_pipeline, "_maybe_copy_audio_from_input"))
        stack.enter_context(patch.object(run_pipeline, "_write_sidecar_from_args"))
        stack.enter_context(patch("pipeline.spherical_injector.inject_spherical_metadata"))
        stack.enter_context(patch("os.replace"))
        yield mocks


class TestGateRunsBeforeTheUpscalePreStage:
    """``--video-upscale seedvr2``: the gate must fire first, on the original."""

    # Acceptance: strict + FAIL + --video-upscale seedvr2 → SeedVR2 is never
    # constructed nor called, and the exit code is non-zero.
    def test_strict_fail_never_reaches_seedvr2(self, run_pipeline, tmp_path):
        args = _main_args(tmp_path, source_check="strict", video_upscale="seedvr2")
        with (
            _mocked_main(run_pipeline, args, _failing_report()) as mocks,
            pytest.raises(SystemExit) as exc_info,
        ):
            run_pipeline.main()
        assert exc_info.value.code == 1
        assert mocks["run_seedvr2_prestage"].call_count == 0
        assert mocks["SeedVR2Upscaler"].call_count == 0
        for name in _HEAVY_NAMES:
            assert mocks[name].call_count == 0, f"{name} ran despite a strict source-check rejection"

    def test_strict_fail_leaves_args_input_at_the_original(self, run_pipeline, tmp_path):
        """The rejection message must name the operator's file, so args.input
        cannot have been rewritten to the upscaled intermediate first."""
        args = _main_args(tmp_path, source_check="strict", video_upscale="seedvr2")
        original = args.input
        with _mocked_main(run_pipeline, args, _failing_report()), pytest.raises(SystemExit):
            run_pipeline.main()
        assert args.input == original

    # Acceptance: the path handed to the check is the ORIGINAL args.input.
    def test_check_grades_the_original_not_the_upscaled_intermediate(self, run_pipeline, tmp_path):
        args = _main_args(tmp_path, video_upscale="seedvr2")
        original = args.input
        with _mocked_main(run_pipeline, args, _passing_report()) as mocks:
            run_pipeline.main()
        assert mocks["run_checks"].call_args.args[0] == original
        # Sanity: the pre-stage really did run and really did rewrite the input,
        # so the assertion above is not passing by accident.
        assert mocks["run_seedvr2_prestage"].call_count == 1
        assert args.input == "upscaled_2x.mp4"

    def test_call_order_is_check_then_upscale_then_preflight(self, run_pipeline, tmp_path):
        """Pins both halves of the card: the gate moved above the pre-stage,
        and --preflight (a host-resource check, unrelated to the footage)
        stayed exactly where P-4b put it — below the pre-stage."""
        args = _main_args(tmp_path, video_upscale="seedvr2")
        order: list[str] = []
        with _mocked_main(run_pipeline, args, _passing_report()) as mocks:
            mocks["run_checks"].side_effect = lambda *a, **kw: (
                order.append("source_check"),
                _passing_report(),
            )[1]
            mocks["run_seedvr2_prestage"].side_effect = lambda *a, **kw: (
                order.append("seedvr2"),
                "upscaled_2x.mp4",
            )[1]
            mocks["_run_preflight"].side_effect = lambda *a, **kw: order.append("preflight")
            run_pipeline.main()
        assert order == ["source_check", "seedvr2", "preflight"]

    def test_off_pays_nothing_and_still_upscales(self, run_pipeline, tmp_path):
        """'off' must not have become "check anyway and discard" in the move."""
        args = _main_args(tmp_path, source_check="off", video_upscale="seedvr2")
        with _mocked_main(run_pipeline, args, _passing_report()) as mocks:
            run_pipeline.main()
        assert mocks["run_checks"].call_count == 0
        assert mocks["run_seedvr2_prestage"].call_count == 1

    def test_warn_fail_still_upscales_and_completes(self, run_pipeline, tmp_path, caplog):
        """warn is the default and must stay non-blocking: a FAILing source
        with seedvr2 still gets upscaled and still runs to completion."""
        caplog.set_level(run_pipeline.logging.INFO)
        args = _main_args(tmp_path, source_check="warn", video_upscale="seedvr2")
        with _mocked_main(run_pipeline, args, _failing_report()) as mocks:
            run_pipeline.main()
        assert "Source check FAILED (warn mode)" in caplog.text
        assert mocks["run_seedvr2_prestage"].call_count == 1
        assert mocks["StreamingPipeline"].call_count == 1


class TestInputsConcatPath:
    """``--inputs``: the graded object is the concat intermediate, and it is
    still graded before any upscaling.

    The concat has to happen first — the run has one timeline and grading
    segment 1 alone would be arbitrary — but that intermediate is a *join* of
    the originals (``demux`` mode is a lossless ``-c copy``), unchanged in
    camera motion, frame size and edge bands.  What it is never is the
    upscaled file.  This is the behaviour the ``--help`` text promises.
    """

    def test_graded_path_is_the_concat_intermediate_not_the_upscale(self, run_pipeline, tmp_path):
        concat_output = tmp_path / "concat.mp4"
        args = _main_args(
            tmp_path,
            input=None,
            inputs=["a.mp4", "b.mp4"],
            video_upscale="seedvr2",
        )
        with _mocked_main(run_pipeline, args, _passing_report(), concat_output=concat_output) as mocks:
            run_pipeline.main()
        assert mocks["run_checks"].call_args.args[0] == str(concat_output)
        assert mocks["run_seedvr2_prestage"].call_count == 1
        assert args.input == "upscaled_2x.mp4"

    def test_strict_fail_on_the_concat_path_never_reaches_seedvr2(self, run_pipeline, tmp_path):
        concat_output = tmp_path / "concat.mp4"
        args = _main_args(
            tmp_path,
            input=None,
            inputs=["a.mp4", "b.mp4"],
            source_check="strict",
            video_upscale="seedvr2",
        )
        with (
            _mocked_main(run_pipeline, args, _failing_report(), concat_output=concat_output) as mocks,
            pytest.raises(SystemExit) as exc_info,
        ):
            run_pipeline.main()
        assert exc_info.value.code == 1
        assert mocks["_concat_segments_preprocess"].call_count == 1
        assert mocks["run_seedvr2_prestage"].call_count == 0
        assert mocks["SeedVR2Upscaler"].call_count == 0

    def test_concat_runs_before_the_gate(self, run_pipeline, tmp_path):
        """Ordering the other way round would mean grading nothing at all on
        this path, since args.input is None until the concat rewrites it."""
        concat_output = tmp_path / "concat.mp4"
        args = _main_args(tmp_path, input=None, inputs=["a.mp4", "b.mp4"])
        order: list[str] = []
        with _mocked_main(run_pipeline, args, _passing_report(), concat_output=concat_output) as mocks:
            mocks["_concat_segments_preprocess"].side_effect = lambda *a, **kw: (
                order.append("concat"),
                concat_output,
            )[1]
            mocks["run_checks"].side_effect = lambda *a, **kw: (
                order.append("source_check"),
                _passing_report(),
            )[1]
            run_pipeline.main()
        assert order == ["concat", "source_check"]


class TestHelpDocumentsThePlacement:
    def test_help_states_the_check_precedes_the_upscale(self, run_pipeline):
        help_text = run_pipeline._build_parser().format_help()
        assert "--source-check {warn,strict,off}" in help_text
        # Squash argparse's own wrapping before matching prose.
        flat = " ".join(help_text.split())
        assert "BEFORE the --video-upscale seedvr2 pre-stage" in flat
        assert "grades the ORIGINAL source" in flat

    def test_help_defines_the_inputs_behaviour(self, run_pipeline):
        """Acceptance: the --inputs case must be spelled out, not left to the
        reader to infer from the call order in main()."""
        flat = " ".join(run_pipeline._build_parser().format_help().split())
        assert "With --inputs it grades the concatenated intermediate" in flat
