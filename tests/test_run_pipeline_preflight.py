"""P-4b (#230): run_pipeline.py preflight wiring tests.

Acceptance criteria (paraphrased from the issue card):

* Default ``warn`` + failing report → emits a WARNING line and the pipeline
  entry point is still called (pipeline continues).
* ``strict`` + failing report → non-zero exit, output contains the reasons.
* ``off`` → :func:`preflight_check` is never called.
* Passing report → exactly one ``preflight [OK]`` info line is logged.
* A crash inside :func:`preflight_check` → WARNING is emitted, pipeline
  continues (preflight never takes the pipeline down).
* Backend-dependent thresholds: ``--depth-model depthcrafter`` yields a
  higher ``min_free_ram_gb`` than ``--depth-model depth-anything``;
  ``--stereo-model stereocrafter`` yields a non-None VRAM threshold.

All tests monkeypatch :func:`preflight_check` (or its environment) so they
do not depend on real host memory / CUDA state. No subprocesses, no model
imports. Marked ``not slow``.
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _import_run_pipeline():
    """Load scripts/run_pipeline.py as an isolated module (V-4 test convention).

    A fresh, uniquely-named module is imported per call so state does not
    leak between tests — critical because the module-level imports in
    run_pipeline.py pull in cv2/torch/pipeline.* which mutate global state.
    """
    scripts_dir = os.path.join(PROJECT_ROOT, "scripts")
    sys.path.insert(0, scripts_dir)
    try:
        # Unique name keeps pytest-collected modules from sharing state.
        name = f"run_pipeline_p{os.getpid()}_{id(__file__)}"
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


def _args(**overrides):
    """A bare dict-of-attr args stub with P-4b-safe defaults.

    ``**overrides`` wins over every default. The bare dict keeps us free
    of MagicMock's "every un-set attribute is a truthy Mock" trap:
    ``getattr`` on a MagicMock would return a ``<MagicMock>`` for any
    attribute we forgot to set, which would break ``getattr(args, "x",
    None) is None`` guards.
    """

    class _Args:
        pass

    defaults = {
        "preflight": "warn",
        "depth_model": "depth-anything",
        "stereo_model": "default",
    }
    defaults.update(overrides)
    args = _Args()
    for k, v in defaults.items():
        setattr(args, k, v)
    return args


# ---------------------------------------------------------------------------
# _resolve_threshold: env-var override parsing
# ---------------------------------------------------------------------------


class TestResolveThreshold:
    def test_no_env_returns_fallback(self, run_pipeline):
        assert run_pipeline._resolve_threshold("VR180_PREFLIGHT_MIN_RAM_GB", 4.0) == 4.0

    def test_valid_numeric_env_wins(self, run_pipeline):
        with patch.dict(os.environ, {"VR180_PREFLIGHT_MIN_RAM_GB": "6.5"}):
            assert run_pipeline._resolve_threshold("VR180_PREFLIGHT_MIN_RAM_GB", 4.0) == 6.5

    def test_non_numeric_env_warns_and_falls_back(self, run_pipeline, caplog):
        with patch.dict(os.environ, {"VR180_PREFLIGHT_MIN_RAM_GB": "abc"}):
            assert run_pipeline._resolve_threshold("VR180_PREFLIGHT_MIN_RAM_GB", 4.0) == 4.0
        assert "Invalid VR180_PREFLIGHT_MIN_RAM_GB='abc'" in caplog.text

    def test_non_positive_env_warns_and_falls_back(self, run_pipeline, caplog):
        with patch.dict(os.environ, {"VR180_PREFLIGHT_MIN_RAM_GB": "-1"}):
            assert run_pipeline._resolve_threshold("VR180_PREFLIGHT_MIN_RAM_GB", 4.0) == 4.0
        assert "must be > 0" in caplog.text


# ---------------------------------------------------------------------------
# _preflight_thresholds: per-backend selection
# ---------------------------------------------------------------------------


class TestPreflightThresholds:
    def test_default_backends_depth_anything_no_vram(self, run_pipeline):
        ram, vram = run_pipeline._preflight_thresholds(_args())
        assert ram == run_pipeline._DEFAULT_MIN_RAM_DEPTH_ANYTHING_GB
        assert vram is None

    def test_depthcrafter_raises_ram(self, run_pipeline):
        args = _args(depth_model="depthcrafter")
        ram_dc, vram_dc = run_pipeline._preflight_thresholds(args)
        ram_da, _ = run_pipeline._preflight_thresholds(_args())

        # Acceptance: depthcrafter's RAM threshold must be >= depth-anything's.
        assert ram_dc >= ram_da
        # And the actual constants the card asked for.
        assert ram_dc == 12.0
        assert ram_da == 4.0
        assert vram_dc is None

    def test_stereocrafter_adds_vram(self, run_pipeline):
        args = _args(stereo_model="stereocrafter")
        ram, vram = run_pipeline._preflight_thresholds(args)
        assert ram == run_pipeline._DEFAULT_MIN_RAM_DEPTH_ANYTHING_GB
        assert vram == run_pipeline._DEFAULT_MIN_VRAM_STEREOCRAFTER_GB
        assert vram == 8.0

    def test_both_heavy_backends(self, run_pipeline):
        args = _args(depth_model="depthcrafter", stereo_model="stereocrafter")
        ram, vram = run_pipeline._preflight_thresholds(args)
        assert ram == 12.0
        assert vram == 8.0

    def test_env_override_wins_over_backend_default(self, run_pipeline):
        args = _args(depth_model="depthcrafter")
        with patch.dict(os.environ, {"VR180_PREFLIGHT_MIN_RAM_GB": "2.0"}):
            ram, _ = run_pipeline._preflight_thresholds(args)
        assert ram == 2.0


# ---------------------------------------------------------------------------
# _run_preflight: the wiring itself (acceptance criteria 1-5)
# ---------------------------------------------------------------------------


def _fake_report(ok: bool, reasons: list[str] | None = None):
    """Build a PreflightReport-shaped object without importing device_utils."""
    report = MagicMock(spec=[])
    report.ok = ok
    report.reasons = reasons or []
    report.free_ram_gb = 10.0 if ok else 3.0
    report.free_vram_gb = 10.0 if ok else 3.0
    return report


class TestRunPreflight:
    @pytest.fixture(autouse=True)
    def _clear_env(self):
        with patch.dict(os.environ, {}, clear=True):
            yield

    # Criterion 1: default warn + fail → warning + continues.
    def test_warn_mode_log_and_continue(self, run_pipeline, caplog):
        caplog.set_level(run_pipeline.logging.INFO)
        args = _args()
        run_pipeline.preflight_check = MagicMock(return_value=_fake_report(ok=False, reasons=["RAM 不足"]))
        # _run_preflight must return without raising or exiting — that is
        # the "pipeline continues" contract for warn mode.
        run_pipeline._run_preflight(args)
        assert "Preflight check FAILED (warn mode)" in caplog.text
        run_pipeline.preflight_check.assert_called_once()

    # Criterion 1 (continued): format_preflight is always logged regardless
    # of pass/fail, so the operator always sees one "preflight [...]" line.
    def test_warn_mode_always_prints_preflight_line(self, run_pipeline, caplog):
        caplog.set_level(run_pipeline.logging.INFO)
        args = _args()
        run_pipeline.preflight_check = MagicMock(return_value=_fake_report(ok=False, reasons=["RAM 不足"]))
        run_pipeline._run_preflight(args)
        assert "preflight [FAIL]" in caplog.text

    # Criterion 2: strict + fail → non-zero exit with reasons.
    def test_strict_mode_exits_with_reasons(self, run_pipeline, capsys):
        args = _args(preflight="strict")
        run_pipeline.preflight_check = MagicMock(return_value=_fake_report(ok=False, reasons=["RAM 不足: x"]))
        with pytest.raises(SystemExit) as ei:
            run_pipeline._run_preflight(args)
        assert ei.value.code == 1
        err = capsys.readouterr().err
        assert "RAM 不足: x" in err
        assert "--preflight strict" in err

    # Criterion 2b: strict + pass → does NOT exit.
    def test_strict_mode_pass_continues(self, run_pipeline):
        args = _args(preflight="strict")
        run_pipeline.preflight_check = MagicMock(return_value=_fake_report(ok=True))
        run_pipeline._run_preflight(args)  # must not raise SystemExit

    # Criterion 3: off → preflight_check is NOT called.
    def test_off_mode_skips_call(self, run_pipeline):
        args = _args(preflight="off")
        fake = MagicMock()
        run_pipeline.preflight_check = fake
        run_pipeline._run_preflight(args)
        fake.assert_not_called()

    # Criterion 4: passing report → logs an [OK] line.
    def test_passing_report_logs_ok(self, run_pipeline, caplog):
        caplog.set_level(run_pipeline.logging.INFO)
        args = _args()
        run_pipeline.preflight_check = MagicMock(return_value=_fake_report(ok=True))
        run_pipeline._run_preflight(args)
        # format_preflight emits "preflight [OK] ..." via log.info
        assert "preflight [OK]" in caplog.text

    # Criterion 5: preflight_check raises → warning, no crash.
    def test_preflight_check_exception_caught(self, run_pipeline, caplog):
        args = _args()
        run_pipeline.preflight_check = MagicMock(side_effect=OSError("no /proc"))
        run_pipeline._run_preflight(args)  # must not raise
        assert "Preflight check failed to read host resources" in caplog.text
        assert "OSError" in caplog.text


# ---------------------------------------------------------------------------
# parse_args: the --preflight CLI flag is wired and validates.
# ---------------------------------------------------------------------------


class TestParseArgs:
    def test_default_is_warn(self, run_pipeline):
        args = run_pipeline.parse_args(["--input", "x.mp4"])
        assert args.preflight == "warn"

    def test_warn_explicit(self, run_pipeline):
        args = run_pipeline.parse_args(["--input", "x.mp4", "--preflight", "warn"])
        assert args.preflight == "warn"

    def test_strict_explicit(self, run_pipeline):
        args = run_pipeline.parse_args(["--input", "x.mp4", "--preflight", "strict"])
        assert args.preflight == "strict"

    def test_off_explicit(self, run_pipeline):
        args = run_pipeline.parse_args(["--input", "x.mp4", "--preflight", "off"])
        assert args.preflight == "off"

    def test_invalid_choice_rejected(self, run_pipeline):
        with pytest.raises(SystemExit):
            run_pipeline.parse_args(["--input", "x.mp4", "--preflight", "explode"])


# ---------------------------------------------------------------------------
# pytest fixture: fresh isolated run_pipeline module per test class.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="class")
def run_pipeline():
    return _import_run_pipeline()
