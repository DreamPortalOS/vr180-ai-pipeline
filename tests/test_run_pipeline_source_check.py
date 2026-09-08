"""W-2 (#311): ``run_pipeline.py --source-check`` wiring tests.

W-1 (#305) delivered ``scripts/check_source_quality.py`` — four cheap checks
(decodable / square / forward_motion / edges) with an exit code.  As a
standalone script nobody remembers to run it, and the most expensive failure
mode in this repo is discovering *after* a forty-minute GPU run that the source
was a locked-off shot with no parallax to convert.

W-2 wires that check into the pipeline entry point.  The acceptance criteria
this file pins (paraphrased from the issue card):

* ``--help`` carries ``--source-check {warn,strict,off}`` and the default is
  ``warn`` — **not** ``strict``, because a non-square source is legitimate on
  the fisheye and 16:9 routes and W-1 grades those WARN.
* FAILing source + ``strict`` → non-zero exit **and no heavy backend was ever
  instantiated**.  That last clause is the entire point of the card: the gate
  has to fire before ``build_depth_backend`` / ``build_stereo_backend``,
  otherwise it saves nothing.
* FAILing source + ``warn`` → the run continues and the log carries a WARNING.
* ``off`` → :func:`run_checks` is never called (call count 0) — the optical
  flow frame sampling costs decode time, so "compute then discard" is not
  acceptable.
* A crash inside the check itself → the pipeline continues with a warning; the
  health check must never take the pipeline down.
* The batch path and the ``--inputs`` concat path are gated too, not just the
  streaming branch.

Everything is mocked: no ffmpeg, no model loads, no real video files.
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
    DEFAULT_PAIRS,
    FORWARD_FAIL_ADVICE,
    NON_SQUARE_ADVICE,
    STATUS_FAIL,
    STATUS_PASS,
    STATUS_WARN,
    CheckResult,
    SourceReport,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _import_run_pipeline():
    """Load scripts/run_pipeline.py as an isolated module (V-4 test convention).

    Same helper as tests/test_run_pipeline_preflight.py: a fresh, uniquely
    named module per call so the cv2/torch/pipeline.* module-level imports do
    not leak state between test classes.
    """
    scripts_dir = os.path.join(PROJECT_ROOT, "scripts")
    sys.path.insert(0, scripts_dir)
    try:
        name = f"run_pipeline_sc{os.getpid()}_{id(__file__)}"
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


@pytest.fixture(scope="class")
def run_pipeline():
    return _import_run_pipeline()


# ---------------------------------------------------------------------------
# Report fixtures — real SourceReport/CheckResult objects, not MagicMocks, so
# the formatting path under test is exercised for real.
# ---------------------------------------------------------------------------


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


def _warning_report(source: str = "in.mp4") -> SourceReport:
    """Non-square 16:9 source — WARN only, which must never gate a run."""
    return SourceReport(
        source=source,
        checks=[
            CheckResult("decodable", STATUS_PASS, "h264 1920x1080 @30fps"),
            CheckResult("square", STATUS_WARN, "1920x1080 (16:9，非 1:1)", {}, NON_SQUARE_ADVICE),
            CheckResult("forward_motion", STATUS_PASS, "径向外流 +3.61px"),
            CheckResult("edges", STATUS_PASS, "四边干净"),
        ],
    )


def _failing_report(source: str = "in.mp4") -> SourceReport:
    """The expensive case: a static shot with no parallax to convert."""
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
                FORWARD_FAIL_ADVICE,
            ),
            CheckResult("edges", STATUS_PASS, "四边干净"),
        ],
    )


def _args(**overrides):
    """Bare attr-holder args stub (no MagicMock truthiness traps)."""

    class _Args:
        pass

    defaults = {"source_check": "warn", "input": "in.mp4"}
    defaults.update(overrides)
    args = _Args()
    for k, v in defaults.items():
        setattr(args, k, v)
    return args


# ---------------------------------------------------------------------------
# parse_args / --help: the flag exists, defaults to warn, validates choices.
# ---------------------------------------------------------------------------


class TestParseArgs:
    def test_default_is_warn(self, run_pipeline):
        """Acceptance: default is warn, NOT strict — strict would block the
        fisheye route and every existing 16:9 source on the square check."""
        args = run_pipeline.parse_args(["--input", "x.mp4"])
        assert args.source_check == "warn"

    @pytest.mark.parametrize("mode", ["warn", "strict", "off"])
    def test_explicit_modes_accepted(self, run_pipeline, mode):
        args = run_pipeline.parse_args(["--input", "x.mp4", "--source-check", mode])
        assert args.source_check == mode

    def test_invalid_choice_rejected(self, run_pipeline):
        with pytest.raises(SystemExit):
            run_pipeline.parse_args(["--input", "x.mp4", "--source-check", "explode"])

    def test_help_advertises_the_three_modes(self, run_pipeline):
        """Acceptance: ``--help`` shows ``--source-check {warn,strict,off}``."""
        help_text = run_pipeline._build_parser().format_help()
        assert "--source-check {warn,strict,off}" in help_text

    def test_streaming_supported_table_lists_it(self, run_pipeline):
        """The gate is consumed in main() before the streaming branch, so the
        K-22 swallowed-arg detector must not report it as ignored."""
        assert "source_check" in run_pipeline._STREAMING_SUPPORTED


# ---------------------------------------------------------------------------
# _format_source_check: one-line summary shaped like the preflight line.
# ---------------------------------------------------------------------------


class TestFormatSourceCheck:
    def test_all_pass_reads_ok(self, run_pipeline):
        line = run_pipeline._format_source_check(_passing_report())
        assert line.startswith("source-check [OK]")
        assert "4 pass / 0 warn / 0 fail / 0 skipped" in line
        assert "reasons:" not in line

    def test_warn_only_reads_warn_and_lists_no_reasons(self, run_pipeline):
        """WARNs alone never gate, so they never appear as failure reasons."""
        line = run_pipeline._format_source_check(_warning_report())
        assert line.startswith("source-check [WARN]")
        assert "reasons:" not in line

    def test_fail_reads_fail_and_names_the_check(self, run_pipeline):
        line = run_pipeline._format_source_check(_failing_report())
        assert line.startswith("source-check [FAIL]")
        assert "reasons: forward_motion:" in line
        assert "静态机位" in line

    def test_is_a_single_line(self, run_pipeline):
        """Format parity with format_preflight — one scannable log line."""
        assert "\n" not in run_pipeline._format_source_check(_failing_report())


# ---------------------------------------------------------------------------
# _run_source_check: the three-state behaviour.
# ---------------------------------------------------------------------------


class TestRunSourceCheck:
    # Acceptance: warn + FAIL → continues, log carries a WARNING.
    def test_warn_mode_logs_warning_and_continues(self, run_pipeline, caplog):
        caplog.set_level(run_pipeline.logging.INFO)
        with patch.object(run_pipeline, "run_checks", return_value=_failing_report()) as checks:
            run_pipeline._run_source_check(_args())  # must not raise / exit
        checks.assert_called_once()
        assert "Source check FAILED (warn mode)" in caplog.text
        assert "forward_motion" in caplog.text
        assert "source-check [FAIL]" in caplog.text

    def test_warn_mode_pass_logs_summary_without_warning(self, run_pipeline, caplog):
        caplog.set_level(run_pipeline.logging.INFO)
        with patch.object(run_pipeline, "run_checks", return_value=_passing_report()):
            run_pipeline._run_source_check(_args())
        assert "source-check [OK]" in caplog.text
        assert "FAILED" not in caplog.text

    def test_warn_only_report_never_gates_even_in_strict(self, run_pipeline, caplog):
        """A 16:9 / fisheye source is WARN, and WARN must never exit — this is
        why the default can safely stay 'warn' AND why strict is still usable."""
        caplog.set_level(run_pipeline.logging.INFO)
        with patch.object(run_pipeline, "run_checks", return_value=_warning_report()):
            run_pipeline._run_source_check(_args(source_check="strict"))  # no SystemExit
        assert "source-check [WARN]" in caplog.text

    # Acceptance: strict + FAIL → non-zero exit, reasons + actionable advice.
    def test_strict_mode_exits_with_reasons_and_advice(self, run_pipeline, capsys):
        with (
            patch.object(run_pipeline, "run_checks", return_value=_failing_report("clip.mp4")),
            pytest.raises(SystemExit) as ei,
        ):
            run_pipeline._run_source_check(_args(source_check="strict", input="clip.mp4"))
        assert ei.value.code == 1
        err = capsys.readouterr().err
        assert "--source-check strict" in err
        assert "forward_motion" in err
        # W-1's advice text is surfaced verbatim — the operator must not have
        # to go re-run the standalone script to learn what to do.
        assert FORWARD_FAIL_ADVICE in err
        # ...and the exact command to get the full report is printed.
        assert "python scripts/check_source_quality.py clip.mp4" in err

    def test_strict_mode_pass_continues(self, run_pipeline):
        with patch.object(run_pipeline, "run_checks", return_value=_passing_report()):
            run_pipeline._run_source_check(_args(source_check="strict"))  # must not raise

    # Acceptance: off → run_checks is never called (frame sampling not paid for).
    def test_off_mode_never_calls_run_checks(self, run_pipeline):
        with patch.object(run_pipeline, "run_checks") as checks:
            run_pipeline._run_source_check(_args(source_check="off"))
        assert checks.call_count == 0

    # Acceptance: a crash inside the check must not take the pipeline down.
    def test_run_checks_exception_is_caught(self, run_pipeline, caplog):
        with patch.object(run_pipeline, "run_checks", side_effect=OSError("ffmpeg not found")):
            run_pipeline._run_source_check(_args())  # must not raise
        assert "Source check failed to run" in caplog.text
        assert "OSError" in caplog.text

    def test_run_checks_exception_is_caught_in_strict_too(self, run_pipeline, caplog):
        """Strict gates on a FAILing *verdict*, not on the checker misbehaving —
        a broken check must not become an un-diagnosable strict rejection."""
        with patch.object(run_pipeline, "run_checks", side_effect=RuntimeError("boom")):
            run_pipeline._run_source_check(_args(source_check="strict"))  # must not raise
        assert "Source check failed to run" in caplog.text
        assert "RuntimeError" in caplog.text

    # Acceptance: the flow sampling stays cheap — W-1's own small default.
    def test_uses_w1_default_pair_count(self, run_pipeline):
        with patch.object(run_pipeline, "run_checks", return_value=_passing_report()) as checks:
            run_pipeline._run_source_check(_args(input="clip.mp4"))
        assert checks.call_args.args[0] == "clip.mp4"
        assert checks.call_args.kwargs["pairs"] == DEFAULT_PAIRS
        assert run_pipeline.SOURCE_CHECK_PAIRS == DEFAULT_PAIRS

    def test_missing_input_is_a_no_op(self, run_pipeline):
        with patch.object(run_pipeline, "run_checks") as checks:
            run_pipeline._run_source_check(_args(input=None))
        assert checks.call_count == 0

    def test_unset_mode_defaults_to_warn(self, run_pipeline, caplog):
        """Callers that build args by hand (older tests, batch_runner shims)
        get warn semantics, never a surprise strict exit."""

        class _Bare:
            input = "in.mp4"

        with patch.object(run_pipeline, "run_checks", return_value=_failing_report()):
            run_pipeline._run_source_check(_Bare())  # must not raise
        assert "Source check FAILED (warn mode)" in caplog.text


# ---------------------------------------------------------------------------
# main(): the gate fires before ANY heavy backend is constructed.
# This is the core value of the card — everything above is scaffolding.
# ---------------------------------------------------------------------------

#: Every module-level name in run_pipeline.py whose construction costs real
#: model-load time.  A strict rejection must leave all of them untouched.
_HEAVY_NAMES = (
    "build_depth_backend",
    "build_stereo_backend",
    "DepthEstimator",
    "DepthCrafterEstimator",
    "StereoRenderer",
    "StereoCrafterRenderer",
    "StreamingPipeline",
    "SeedVR2Upscaler",
)


def _main_args(**overrides):
    """A MagicMock args wired for a minimal streaming run through main()."""
    args = MagicMock()
    args.inputs = None
    args.input = "in.mp4"
    args.output = "out.mp4"
    args.video_upscale = "none"
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
    # Keep the P-4b host-memory preflight out of the picture: this file is
    # about the source gate, and 'off' makes the test hermetic.
    args.preflight = "off"
    args.source_check = "warn"
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def _run_main(run_pipeline, args, report):
    """Run ``main()`` with every heavy construction site mocked.

    ``report`` is either a :class:`SourceReport` (returned by ``run_checks``)
    or an exception instance (raised by it).  Returns the dict of heavy-name
    mocks plus the ``run_checks`` mock so the caller can assert call counts.
    """
    checks_kwargs = {"side_effect": report} if isinstance(report, BaseException) else {"return_value": report}
    heavy = {}
    with contextlib.ExitStack() as stack:
        checks = stack.enter_context(patch.object(run_pipeline, "run_checks", **checks_kwargs))
        for name in _HEAVY_NAMES:
            heavy[name] = stack.enter_context(patch.object(run_pipeline, name))
        # build_* must still return the (backend, name) 2-tuple main() unpacks.
        heavy["build_depth_backend"].return_value = (MagicMock(), "depth-anything")
        heavy["build_stereo_backend"].return_value = (MagicMock(), "default")
        heavy["StreamingPipeline"].return_value.process_stream.return_value = "out.mp4"

        stack.enter_context(patch.object(run_pipeline, "parse_args", return_value=args))
        stack.enter_context(patch.object(run_pipeline, "apply_quality_preset"))
        stack.enter_context(patch.object(run_pipeline, "validate_input_projection"))
        stack.enter_context(patch.object(run_pipeline, "_copy_audio_to_output"))
        stack.enter_context(patch.object(run_pipeline, "_maybe_copy_audio_from_input"))
        stack.enter_context(patch.object(run_pipeline, "_write_sidecar_from_args"))
        stack.enter_context(patch("pipeline.spherical_injector.inject_spherical_metadata"))
        stack.enter_context(patch("os.replace"))
        run_pipeline.main()
    return heavy, checks


def _assert_no_heavy_backend(heavy):
    for name, mock in heavy.items():
        assert mock.call_count == 0, f"{name} was constructed despite a strict source-check rejection"


class TestMainGate:
    """The gate has to fire before the expensive part, or it saves nothing."""

    # Acceptance (core): FAIL + strict → non-zero exit AND no heavy backend.
    def test_strict_fail_exits_before_any_heavy_backend_streaming(self, run_pipeline):
        args = _main_args(source_check="strict")
        heavy = {}
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(run_pipeline, "run_checks", return_value=_failing_report()))
            for name in _HEAVY_NAMES:
                heavy[name] = stack.enter_context(patch.object(run_pipeline, name))
            stack.enter_context(patch.object(run_pipeline, "parse_args", return_value=args))
            stack.enter_context(patch.object(run_pipeline, "apply_quality_preset"))
            stack.enter_context(patch.object(run_pipeline, "validate_input_projection"))
            with pytest.raises(SystemExit) as ei:
                run_pipeline.main()
        assert ei.value.code == 1
        _assert_no_heavy_backend(heavy)

    def test_strict_fail_exits_before_any_heavy_backend_batch(self, run_pipeline):
        """Same gate on the batch path (--streaming off ⇒ _stage_all_body,
        which builds its backends via the very same two factories)."""
        args = _main_args(source_check="strict", streaming=False)
        heavy = {}
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(run_pipeline, "run_checks", return_value=_failing_report()))
            for name in _HEAVY_NAMES:
                heavy[name] = stack.enter_context(patch.object(run_pipeline, name))
            stack.enter_context(patch.object(run_pipeline, "parse_args", return_value=args))
            stack.enter_context(patch.object(run_pipeline, "apply_quality_preset"))
            stack.enter_context(patch.object(run_pipeline, "validate_input_projection"))
            with pytest.raises(SystemExit) as ei:
                run_pipeline.main()
        assert ei.value.code == 1
        _assert_no_heavy_backend(heavy)

    def test_strict_fail_exits_before_any_heavy_backend_single_stage(self, run_pipeline):
        """--stage depth is a third entry into build_depth_backend; the gate
        runs before the stage dispatch, so it covers that one too."""
        args = _main_args(source_check="strict", streaming=False, stage="depth")
        heavy = {}
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(run_pipeline, "run_checks", return_value=_failing_report()))
            for name in _HEAVY_NAMES:
                heavy[name] = stack.enter_context(patch.object(run_pipeline, name))
            stack.enter_context(patch.object(run_pipeline, "parse_args", return_value=args))
            stack.enter_context(patch.object(run_pipeline, "apply_quality_preset"))
            stack.enter_context(patch.object(run_pipeline, "validate_input_projection"))
            with pytest.raises(SystemExit) as ei:
                run_pipeline.main()
        assert ei.value.code == 1
        _assert_no_heavy_backend(heavy)

    # Acceptance: FAIL + warn → the run continues to completion.
    def test_warn_fail_runs_to_completion_with_warning(self, run_pipeline, caplog):
        caplog.set_level(run_pipeline.logging.INFO)
        args = _main_args(source_check="warn")
        heavy, checks = _run_main(run_pipeline, args, _failing_report())
        checks.assert_called_once()
        assert "Source check FAILED (warn mode)" in caplog.text
        # The pipeline really did continue past the gate.
        assert heavy["build_depth_backend"].call_count == 1
        assert heavy["build_stereo_backend"].call_count == 1
        assert heavy["StreamingPipeline"].call_count == 1

    # Acceptance: off → run_checks never called from the pipeline entry point.
    def test_off_never_calls_run_checks_from_main(self, run_pipeline):
        args = _main_args(source_check="off")
        _, checks = _run_main(run_pipeline, args, _passing_report())
        assert checks.call_count == 0

    # Acceptance: a crashing check must not take a real run down.
    def test_check_crash_does_not_stop_main(self, run_pipeline, caplog):
        args = _main_args(source_check="strict")
        heavy, _ = _run_main(run_pipeline, args, OSError("ffprobe missing"))
        assert "Source check failed to run" in caplog.text
        assert "OSError" in caplog.text
        # Even in strict mode: a broken checker is not a failing verdict.
        assert heavy["StreamingPipeline"].call_count == 1

    def test_gate_checks_the_actual_input_path(self, run_pipeline):
        args = _main_args(input="video/clip.mp4")
        _, checks = _run_main(run_pipeline, args, _passing_report())
        assert checks.call_args.args[0] == "video/clip.mp4"


class TestMainGateConcatPath:
    """C-1b ``--inputs``: main() concatenates the segments and rewrites
    ``args.input`` to the intermediate *before* the gate, so the footage that
    actually enters the pipeline is the footage that gets checked."""

    def _run(self, run_pipeline, args, report, expect_exit):
        heavy = {}
        with contextlib.ExitStack() as stack:
            checks = stack.enter_context(patch.object(run_pipeline, "run_checks", return_value=report))
            for name in _HEAVY_NAMES:
                heavy[name] = stack.enter_context(patch.object(run_pipeline, name))
            heavy["build_depth_backend"].return_value = (MagicMock(), "depth-anything")
            heavy["build_stereo_backend"].return_value = (MagicMock(), "default")
            heavy["StreamingPipeline"].return_value.process_stream.return_value = "out.mp4"
            stack.enter_context(patch.object(run_pipeline, "parse_args", return_value=args))
            stack.enter_context(patch.object(run_pipeline, "apply_quality_preset"))
            stack.enter_context(patch.object(run_pipeline, "validate_input_projection"))
            stack.enter_context(
                patch.object(run_pipeline, "_concat_segments_preprocess", return_value=Path("tmp/concat.mp4"))
            )
            stack.enter_context(patch.object(run_pipeline, "_copy_audio_to_output"))
            stack.enter_context(patch.object(run_pipeline, "_maybe_copy_audio_from_input"))
            stack.enter_context(patch.object(run_pipeline, "_write_sidecar_from_args"))
            stack.enter_context(patch("pipeline.spherical_injector.inject_spherical_metadata"))
            stack.enter_context(patch("os.replace"))
            if expect_exit:
                with pytest.raises(SystemExit) as ei:
                    run_pipeline.main()
                return heavy, checks, ei.value.code
            run_pipeline.main()
            return heavy, checks, None

    def test_concat_intermediate_is_the_thing_checked(self, run_pipeline):
        args = _main_args(inputs=["a.mp4", "b.mp4"], input=None)
        _, checks, _ = self._run(run_pipeline, args, _passing_report(), expect_exit=False)
        checks.assert_called_once()
        assert checks.call_args.args[0] == str(Path("tmp/concat.mp4"))

    def test_strict_fail_gates_the_concat_path_too(self, run_pipeline):
        args = _main_args(inputs=["a.mp4", "b.mp4"], input=None, source_check="strict")
        heavy, _, code = self._run(run_pipeline, args, _failing_report(), expect_exit=True)
        assert code == 1
        _assert_no_heavy_backend(heavy)

    def test_off_skips_the_check_on_the_concat_path(self, run_pipeline):
        args = _main_args(inputs=["a.mp4", "b.mp4"], input=None, source_check="off")
        _, checks, _ = self._run(run_pipeline, args, _passing_report(), expect_exit=False)
        assert checks.call_count == 0
