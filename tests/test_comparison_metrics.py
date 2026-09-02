"""Tests for the depth-stability metrics wired into comparison.md (K-16, #206).

These tests pin the *contract* the issue lays out:

- A recipe that produced a depth-product dir → the three metric columns
  (temporal_jitter / flicker_ratio / edge_consistency) appear in
  ``comparison.md`` and carry an OK/WARN/FAIL mark.
- A recipe with **no** depth product → the three cells read ``—`` and the
  comparison is still generated (exit code unchanged).
- ``depth_stability`` raising mid-computation → the cell degrades to ``—`` with
  a warning, and the comparison still completes (this is the headline rule:
  an after-the-fact statistic must never tank a render).
- ``--no-metrics`` → depth_stability is never called.
- The OK/WARN/FAIL marks come from ``scripts.depth_stability``'s own threshold
  constants — not a parallel set hardcoded in make_comparison.

Everything is mocked: no real render, no real ffprobe, no real model.  The
only "real" call is ``depth_stability.compute_report`` over tiny synthetic
numpy frames (CPU-only, no GPU/cv2) — used to prove the threshold linkage.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from scripts import make_comparison as mc
from scripts.depth_stability import (
    EDGE_OK,
    EDGE_WARN,
    FLICKER_OK,
    FLICKER_WARN,
    JITTER_OK,
    JITTER_WARN,
    compute_report,
)
from scripts.make_comparison import (
    DEPTH_BASELINE_EDGE,
    DEPTH_BASELINE_FLICKER,
    DEPTH_METRIC_COLUMNS,
    DEPTH_METRIC_NA,
    Recipe,
    RecipeResult,
    apply_depth_metrics,
    main,
    render_comparison_md,
    run_comparison,
)

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _fake_runner_ok():
    """A render runner that 'renders' by touching the output file."""
    calls: list[list[str]] = []

    def runner(cmd) -> str:
        calls.append(list(cmd))
        out = str(cmd[list(cmd).index("--output") + 1])
        Path(out).touch()
        return out

    return runner, calls


def _fake_qa_ok():
    def qa(output_path: str, row: RecipeResult) -> RecipeResult:
        row.resolution = "5760×2880"
        row.audio = "aac"
        row.verdict = "VR180 (180° 3D SBS)"
        row.qa_failed = False
        return row

    return qa


@dataclass
class _FakeMetric:
    """Stand-in for depth_stability.MetricResult — name/value/verdict."""

    name: str
    value: float
    ok: str  # "OK" | "WARN" | "FAIL"


@dataclass
class _FakeReport:
    """Stand-in for depth_stability.StabilityReport."""

    temporal_jitter: _FakeMetric
    flicker_ratio: _FakeMetric
    edge_consistency: _FakeMetric


def _fake_report_from_metric(value, ok):
    """Build a one-metric-value report where every column uses the same mark.

    Used when the test only cares that *a* mark lands in the markdown, not which
    tier — the threshold-linkage test below uses the real compute_report.
    """
    return _FakeReport(
        temporal_jitter=_FakeMetric("temporal_jitter", value, ok),
        flicker_ratio=_FakeMetric("flicker_ratio", value, ok),
        edge_consistency=_FakeMetric("edge_consistency", value, ok),
    )


def _resolver(return_dir: str | None):
    """A depth-dir resolver that returns a fixed value and records calls."""
    calls: list[tuple] = []

    def resolver(input_path, recipe, result):
        calls.append((input_path, recipe.name))
        return return_dir

    return resolver, calls


def _metrics_runner_factory(calls: list[str], report=None, exc: Exception | None = None):
    """A metrics runner that records its arg and returns *report* or raises *exc*."""

    def runner(depth_dir):
        calls.append(str(depth_dir))
        if exc is not None:
            raise exc
        return report

    return runner


# ---------------------------------------------------------------------------
# apply_depth_metrics — unit-level contract
# ---------------------------------------------------------------------------


class TestApplyDepthMetricsUnit:
    def test_missing_depth_dir_leaves_na(self) -> None:
        resolver, _ = _resolver(None)
        runner = _metrics_runner_factory([], report=None)
        # must_not_be_called guard: a None dir must never reach the runner.
        runner_calls: list[str] = []
        runner = _metrics_runner_factory(runner_calls, report=None)
        row = apply_depth_metrics(
            "src.mp4",
            Recipe(name="baseline", args=[]),
            RecipeResult(recipe="baseline"),
            depth_dir_resolver=resolver,
            metrics_runner=runner,
        )
        assert row.temporal_jitter == DEPTH_METRIC_NA
        assert row.flicker_ratio == DEPTH_METRIC_NA
        assert row.edge_consistency == DEPTH_METRIC_NA
        assert runner_calls == []  # no depth dir → no depth_stability call

    def test_present_depth_dir_fills_cells_with_verdict(self) -> None:
        resolver, _ = _resolver("/some/depth/depth-anything")
        runner_calls: list[str] = []
        report = _fake_report_from_metric(0.1234, "WARN")
        runner = _metrics_runner_factory(runner_calls, report=report)
        row = apply_depth_metrics(
            "src.mp4",
            Recipe(name="baseline", args=[]),
            RecipeResult(recipe="baseline"),
            depth_dir_resolver=resolver,
            metrics_runner=runner,
        )
        assert row.temporal_jitter == "0.1234 WARN"
        assert row.flicker_ratio == "0.1234 WARN"
        assert row.edge_consistency == "0.1234 WARN"
        # the runner received the resolved dir verbatim
        assert runner_calls == ["/some/depth/depth-anything"]

    def test_depth_stability_exception_degrades_to_na(self, caplog: pytest.LogCaptureFixture) -> None:
        resolver, _ = _resolver("/some/depth/depth-anything")
        runner_calls: list[str] = []
        runner = _metrics_runner_factory(runner_calls, exc=RuntimeError("model blew up"))
        row = apply_depth_metrics(
            "src.mp4",
            Recipe(name="baseline", args=[]),
            RecipeResult(recipe="baseline"),
            depth_dir_resolver=resolver,
            metrics_runner=runner,
        )
        # headline rule: never raise, leave cells as —
        assert row.temporal_jitter == DEPTH_METRIC_NA
        assert row.flicker_ratio == DEPTH_METRIC_NA
        assert row.edge_consistency == DEPTH_METRIC_NA
        # and surface a warning so the miss isn't silent
        assert any("depth_stability failed" in r.message for r in caplog.records)

    def test_resolver_exception_degrades_to_na(self, caplog: pytest.LogCaptureFixture) -> None:
        def bad_resolver(input_path, recipe, result):
            raise OSError("disk on fire")

        runner_calls: list[str] = []
        runner = _metrics_runner_factory(runner_calls, report=None)
        row = apply_depth_metrics(
            "src.mp4",
            Recipe(name="baseline", args=[]),
            RecipeResult(recipe="baseline"),
            depth_dir_resolver=bad_resolver,
            metrics_runner=runner,
        )
        assert row.temporal_jitter == DEPTH_METRIC_NA
        assert row.flicker_ratio == DEPTH_METRIC_NA
        assert row.edge_consistency == DEPTH_METRIC_NA
        assert runner_calls == []  # resolver blew up before the runner ran


# ---------------------------------------------------------------------------
# end-to-end (still fully mocked) through run_comparison
# ---------------------------------------------------------------------------


class TestRunComparisonMetrics:
    def test_metrics_columns_appear_with_marks_when_depth_present(self, tmp_path: Path) -> None:
        runner, _ = _fake_runner_ok()
        resolver, _ = _resolver(str(tmp_path / "depths"))
        report = _fake_report_from_metric(0.42, "FAIL")
        results = run_comparison(
            input_path="src.mp4",
            outdir=tmp_path,
            render_runner=runner,
            qa_runner=_fake_qa_ok(),
            depth_dir_resolver=resolver,
            metrics_runner=_metrics_runner_factory([], report=report),
        )
        assert all(r.ok for r in results)
        for r in results:
            assert r.temporal_jitter == "0.4200 FAIL"
            assert r.flicker_ratio == "0.4200 FAIL"
            assert r.edge_consistency == "0.4200 FAIL"

        md = (tmp_path / "comparison.md").read_text(encoding="utf-8")
        header = md.splitlines()[md.splitlines().index("## 汇总") + 2]
        for col in DEPTH_METRIC_COLUMNS:
            assert col in header
        # marks land in the body
        assert "FAIL" in md

    def test_no_depth_products_yields_na_cells_and_still_writes_md(self, tmp_path: Path) -> None:
        runner, _ = _fake_runner_ok()
        resolver, _ = _resolver(None)  # no depth dir for any recipe
        metrics_calls: list[str] = []
        results = run_comparison(
            input_path="src.mp4",
            outdir=tmp_path,
            render_runner=runner,
            qa_runner=_fake_qa_ok(),
            depth_dir_resolver=resolver,
            metrics_runner=_metrics_runner_factory(metrics_calls, report=None),
        )
        assert all(r.ok for r in results)
        for r in results:
            assert r.temporal_jitter == DEPTH_METRIC_NA
            assert r.flicker_ratio == DEPTH_METRIC_NA
            assert r.edge_consistency == DEPTH_METRIC_NA
        # depth_stability was never invoked (no dir resolved)
        assert metrics_calls == []
        # comparison still generated and lists all three columns as —
        md = (tmp_path / "comparison.md").read_text(encoding="utf-8")
        assert DEPTH_METRIC_NA in md
        assert "汇总" in md

    def test_depth_stability_exception_does_not_fail_comparison(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        runner, _ = _fake_runner_ok()
        resolver, _ = _resolver(str(tmp_path / "depths"))
        results = run_comparison(
            input_path="src.mp4",
            outdir=tmp_path,
            render_runner=runner,
            qa_runner=_fake_qa_ok(),
            depth_dir_resolver=resolver,
            metrics_runner=_metrics_runner_factory([], exc=RuntimeError("kaboom")),
        )
        # every recipe still 'ok' (render + QA succeeded); metrics just blank
        assert all(r.ok for r in results)
        for r in results:
            assert r.temporal_jitter == DEPTH_METRIC_NA
            assert r.flicker_ratio == DEPTH_METRIC_NA
            assert r.edge_consistency == DEPTH_METRIC_NA
        # a warning is logged so the miss is not silent
        assert any("depth_stability failed" in r.message for r in caplog.records)
        # comparison.md written despite the metric failures
        md = (tmp_path / "comparison.md").read_text(encoding="utf-8")
        assert "汇总" in md

    def test_no_metrics_flag_skips_depth_stability(self, tmp_path: Path) -> None:
        runner, _ = _fake_runner_ok()
        resolver, resolver_calls = _resolver("/should/not/matter")
        metrics_calls: list[str] = []
        results = run_comparison(
            input_path="src.mp4",
            outdir=tmp_path,
            render_runner=runner,
            qa_runner=_fake_qa_ok(),
            metrics=False,  # --no-metrics
            depth_dir_resolver=resolver,
            metrics_runner=_metrics_runner_factory(metrics_calls, report=None),
        )
        assert all(r.ok for r in results)
        # neither the resolver nor depth_stability is consulted under --no-metrics
        assert resolver_calls == []
        assert metrics_calls == []
        for r in results:
            assert r.temporal_jitter == DEPTH_METRIC_NA
            assert r.flicker_ratio == DEPTH_METRIC_NA
            assert r.edge_consistency == DEPTH_METRIC_NA
        # columns still exist in the table (as —) so the layout is stable
        md = (tmp_path / "comparison.md").read_text(encoding="utf-8")
        for col in DEPTH_METRIC_COLUMNS:
            assert col in md


# ---------------------------------------------------------------------------
# CLI --no-metrics
# ---------------------------------------------------------------------------


class TestNoMetricsCLI:
    def test_no_metrics_flag_in_help(self, capsys: pytest.CaptureFixture) -> None:
        with pytest.raises(SystemExit) as exc_info:
            mc.parse_args(["--help"])
        assert exc_info.value.code == 0
        assert "--no-metrics" in capsys.readouterr().out

    def test_no_metrics_flag_parsed_true(self) -> None:
        args = mc.parse_args(["--input", "a.mp4", "--outdir", "o", "--no-metrics"])
        assert args.no_metrics is True

    def test_no_metrics_defaults_false(self) -> None:
        args = mc.parse_args(["--input", "a.mp4", "--outdir", "o"])
        assert args.no_metrics is False

    def test_main_no_metrics_passes_through(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """main(--no-metrics) must call run_comparison with metrics=False."""
        captured: dict = {}

        def _capture_run_comparison(**kwargs):
            captured.update(kwargs)
            return []

        monkeypatch.setattr(mc, "run_comparison", _capture_run_comparison)
        rc = main(["--input", "src.mp4", "--outdir", str(tmp_path), "--no-metrics", "--dry-run"])
        assert rc == 0
        assert captured.get("metrics") is False


# ---------------------------------------------------------------------------
# Threshold linkage — marks come from depth_stability's constants, not literals
# ---------------------------------------------------------------------------


class TestThresholdLinkage:
    """The OK/WARN/FAIL marks must be produced by depth_stability's own
    threshold constants, not a parallel set hardcoded in make_comparison.

    We prove this by (a) constructing synthetic frames whose metric values land
    in each tier, (b) running the *real* ``compute_report`` (the function
    make_comparison calls through run_depth_metrics), and (c) asserting the mark
    that flows into comparison.md matches the tier those constants define.
    """

    def test_constants_define_three_tiers(self) -> None:
        # sanity: the constants are ordered and non-trivial
        assert JITTER_OK < JITTER_WARN
        assert FLICKER_OK < FLICKER_WARN
        assert EDGE_OK > EDGE_WARN  # higher-is-better → OK threshold is larger

    def test_mark_flows_from_compute_report_not_hardcoded(self) -> None:
        # static frames → every metric OK; compute_report grades via the
        # JITTER_*/FLICKER_*/EDGE_* constants, and _metric_cell just echoes
        # report.<metric>.ok. So if the markdown shows "OK", that mark
        # originated from those constants — not from a literal in mc.
        base = np.linspace(0.1, 0.9, 8 * 8, dtype=np.float32).reshape(8, 8)
        depths = [base.copy() for _ in range(6)]
        report = compute_report(depths)
        assert report.temporal_jitter.ok == "OK"
        assert report.flicker_ratio.ok == "OK"
        assert report.edge_consistency.ok == "OK"

        row = RecipeResult(recipe="baseline")
        row.temporal_jitter = mc._metric_cell(report.temporal_jitter)
        row.flicker_ratio = mc._metric_cell(report.flicker_ratio)
        row.edge_consistency = mc._metric_cell(report.edge_consistency)
        assert row.temporal_jitter.endswith(" OK")
        assert row.flicker_ratio.endswith(" OK")
        assert row.edge_consistency.endswith(" OK")

    def test_fail_mark_matches_threshold_boundary(self) -> None:
        # A flicker ratio at/above FLICKER_WARN must grade FAIL under the real
        # thresholds — and _metric_cell must surface exactly that mark.
        # Build a report object whose mark we derive from the constants, then
        # confirm _metric_cell echoes it verbatim (no parallel grading logic).
        # value deliberately above the WARN threshold → FAIL.
        value = FLICKER_WARN + 0.01
        # compute the expected mark with depth_stability's own _grade path by
        # constructing a MetricResult the same way compute_report does.
        from scripts.depth_stability import MetricResult, _grade

        expected_ok = _grade(value, {"OK": FLICKER_OK, "WARN": FLICKER_WARN}, higher_is_better=False)
        assert expected_ok == "FAIL"
        metric = MetricResult(
            name="flicker_ratio",
            value=value,
            ok=expected_ok,
            thresholds={"OK": FLICKER_OK, "WARN": FLICKER_WARN},
            higher_is_better=False,
        )
        cell = mc._metric_cell(metric)
        assert cell == f"{value:.4f} FAIL"
        assert "FAIL" in cell


# ---------------------------------------------------------------------------
# comparison.md rendering — baseline reference + columns
# ---------------------------------------------------------------------------


class TestComparisonMdMetricsRendering:
    def test_columns_and_baseline_note_present(self) -> None:
        rows = [
            RecipeResult(
                recipe="baseline",
                description="基线",
                output="/o/a_baseline.mp4",
                ok=True,
                resolution="5760×2880",
                audio="aac",
                verdict="VR180 (180° 3D SBS)",
                temporal_jitter="0.5221 FAIL",
                flicker_ratio="0.5221 FAIL",
                edge_consistency="0.1726 FAIL",
            ),
            RecipeResult(
                recipe="temporal",
                description="时序",
                output="/o/a_temporal.mp4",
                ok=True,
                resolution="5760×2880",
                audio="aac",
                verdict="VR180 (180° 3D SBS)",
                temporal_jitter=DEPTH_METRIC_NA,
                flicker_ratio=DEPTH_METRIC_NA,
                edge_consistency=DEPTH_METRIC_NA,
            ),
        ]
        md = render_comparison_md(rows, source="src.mp4", outdir="/o")
        # three columns in the header
        header = next(line for line in md.splitlines() if line.startswith("| recipe"))
        for col in DEPTH_METRIC_COLUMNS:
            assert col in header
        # filled cells carry the FAIL mark
        assert "0.5221 FAIL" in md
        assert "0.1726 FAIL" in md
        # missing cells show —
        # (count occurrences: one row all-—  → three — on that row)
        assert DEPTH_METRIC_NA in md
        # baseline reference note under the table
        assert f"{DEPTH_BASELINE_FLICKER:.4f}" in md
        assert f"{DEPTH_BASELINE_EDGE:.4f}" in md
        assert "flicker_ratio 越低越好" in md
        assert "edge_consistency 越高越好" in md

    def test_baseline_note_absent_when_metrics_off_is_not_a_thing(self) -> None:
        """The baseline note is part of the summary block and always present —
        it documents the reference, independent of whether any row had a value.
        This just guards against accidentally gating the note on metrics=True."""
        rows = [RecipeResult(recipe="x", ok=True)]
        md = render_comparison_md(rows, source="s.mp4", outdir="/o")
        assert f"{DEPTH_BASELINE_FLICKER:.4f}" in md


# ---------------------------------------------------------------------------
# K-17 (#208): per-recipe depth dirs, no cross-model fallback, freshness gate
# ---------------------------------------------------------------------------


def _write_npy_at(tmp_path: Path, name: str, mtime: float) -> Path:
    """Write a tiny depth_*.npy and stamp an arbitrary mtime onto it."""
    p = tmp_path / name
    np.save(p, np.zeros((4, 4), dtype=np.float32))
    os.utime(p, (mtime, mtime))
    return p


class TestPerRecipeDepthDir:
    """Each recipe must render into, and be graded from, its own depth dir."""

    def test_build_render_command_appends_distinct_temp_dir_per_recipe(self, tmp_path: Path) -> None:
        recipes = [
            mc.Recipe(name="baseline", args=["--comfort", "strong"]),
            mc.Recipe(name="temporal", args=["--depth-model", "depthcrafter", "--comfort", "safe"]),
        ]
        cmd_a = mc.build_render_command("in.mp4", "out/a.mp4", recipes[0], outdir=tmp_path)
        cmd_b = mc.build_render_command("in.mp4", "out/b.mp4", recipes[1], outdir=tmp_path)

        assert "--temp-dir" in cmd_a and "--temp-dir" in cmd_b
        temp_a = cmd_a[cmd_a.index("--temp-dir") + 1]
        temp_b = cmd_b[cmd_b.index("--temp-dir") + 1]
        # the two recipes resolve to different per-recipe work dirs
        assert temp_a != temp_b
        assert temp_a.endswith("baseline")
        assert temp_b.endswith("temporal")

    def test_no_outdir_keeps_argv_shape_unchanged(self) -> None:
        """Unit tests that only check argv order still get the pre-#208 shape."""
        recipe = mc.Recipe(name="x", args=[])
        cmd = mc.build_render_command("in.mp4", "out/x.mp4", recipe)
        assert "--temp-dir" not in cmd


class TestDefaultDepthDirResolver:
    """The default resolver must only grade this recipe's own, fresh depth dir."""

    def test_fresh_npy_in_recipe_dir_is_returned(self, tmp_path: Path) -> None:
        work = tmp_path / "_work" / "baseline"
        depth_dir = work / "depth"
        depth_dir.mkdir(parents=True)
        now = time.time()
        _write_npy_at(depth_dir, "depth_001.npy", now + 5)
        _write_npy_at(depth_dir, "depth_002.npy", now + 6)

        result = RecipeResult(
            recipe="baseline",
            temp_dir=str(work),
            render_started=now - 60,
        )
        got = mc.default_depth_dir_resolver("src.mp4", mc.Recipe(name="baseline", args=[]), result)
        assert got == str(depth_dir)

    def test_missing_recipe_dir_returns_none(self, tmp_path: Path) -> None:
        """No depth dir at all → None, cells render as —."""
        work = tmp_path / "_work" / "baseline"  # deliberately absent
        result = RecipeResult(recipe="baseline", temp_dir=str(work), render_started=0.0)
        assert mc.default_depth_dir_resolver("src.mp4", mc.Recipe(name="baseline", args=[]), result) is None

    def test_no_temp_dir_returns_none(self, tmp_path: Path) -> None:
        """Row without a per-recipe temp dir (e.g. a legacy/injected unit row) → None."""
        result = RecipeResult(recipe="baseline", temp_dir=None)
        assert mc.default_depth_dir_resolver("src.mp4", mc.Recipe(name="baseline", args=[]), result) is None

    def test_empty_recipe_dir_returns_none(self, tmp_path: Path) -> None:
        work = tmp_path / "_work" / "baseline"
        (work / "depth").mkdir(parents=True)  # dir exists but has no npy
        result = RecipeResult(recipe="baseline", temp_dir=str(work), render_started=0.0)
        assert mc.default_depth_dir_resolver("src.mp4", mc.Recipe(name="baseline", args=[]), result) is None

    def test_other_recipe_dir_with_npy_is_never_read(self, tmp_path: Path) -> None:
        """A depth dir populated for a *different* recipe must not be graded as
        this recipe's result — the exact cross-recipe contamination this bug
        introduced.  Even when the other dir is full of fresh-looking npy, the
        answer is —.
        """
        baseline_work = tmp_path / "_work" / "baseline"
        temporal_work = tmp_path / "_work" / "temporal"
        (temporal_work / "depth").mkdir(parents=True)
        now = time.time()
        _write_npy_at(temporal_work / "depth", "depth_001.npy", now + 5)
        _write_npy_at(temporal_work / "depth", "depth_002.npy", now + 6)

        # Ask about the baseline recipe — whose own dir does NOT exist.
        result = RecipeResult(
            recipe="baseline",
            temp_dir=str(baseline_work),
            render_started=now - 60,
        )
        assert mc.default_depth_dir_resolver("src.mp4", mc.Recipe(name="baseline", args=[]), result) is None

    def test_stale_npy_older_than_render_returns_none_and_warns(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """npy present but mtime predates this render → — with a warning.

        This is the headline K-17 bug: an 18-hour-old depth dir being graded as
        the current recipe's result produced identical flicker numbers for both
        backends.  After the fix the resolver refuses stale files.
        """
        work = tmp_path / "_work" / "baseline"
        depth_dir = work / "depth"
        depth_dir.mkdir(parents=True)
        now = time.time()
        # write npy as if it were 2 hours old — way before this render started
        _write_npy_at(depth_dir, "depth_001.npy", now - 7200)

        result = RecipeResult(
            recipe="baseline",
            temp_dir=str(work),
            render_started=now,
        )
        assert mc.default_depth_dir_resolver("src.mp4", mc.Recipe(name="baseline", args=[]), result) is None
        assert any("older than this render" in r.message for r in caplog.records)

    def test_recently_rendered_npy_at_start_of_render_is_not_stale(self, tmp_path: Path) -> None:
        """A file written essentially at render start (within the 1s grace
        window) must NOT be rejected — genuine near-instant renders are valid."""
        work = tmp_path / "_work" / "baseline"
        depth_dir = work / "depth"
        depth_dir.mkdir(parents=True)
        now = time.time()
        # written just a tick before render_started — still within the window
        _write_npy_at(depth_dir, "depth_001.npy", now - 0.5)

        result = RecipeResult(recipe="baseline", temp_dir=str(work), render_started=now)
        assert mc.default_depth_dir_resolver("src.mp4", mc.Recipe(name="baseline", args=[]), result) == str(depth_dir)

    def test_different_backends_resolve_to_different_dirs(self, tmp_path: Path) -> None:
        """baseline and temporal must grade from *different* depth dirs, so
        their flicker numbers can actually diverge."""
        now = time.time()

        baseline_work = tmp_path / "_work" / "baseline"
        temporal_work = tmp_path / "_work" / "temporal"
        (baseline_work / "depth").mkdir(parents=True)
        (temporal_work / "depth").mkdir(parents=True)
        _write_npy_at(baseline_work / "depth", "depth_001.npy", now + 5)
        _write_npy_at(temporal_work / "depth", "depth_001.npy", now + 5)

        baseline_result = RecipeResult(recipe="baseline", temp_dir=str(baseline_work), render_started=now - 60)
        temporal_result = RecipeResult(recipe="temporal", temp_dir=str(temporal_work), render_started=now - 60)
        got_baseline = mc.default_depth_dir_resolver("src.mp4", mc.Recipe(name="baseline", args=[]), baseline_result)
        got_temporal = mc.default_depth_dir_resolver(
            "src.mp4",
            mc.Recipe(name="temporal", args=["--depth-model", "depthcrafter"]),
            temporal_result,
        )
        assert got_baseline is not None and got_temporal is not None
        assert got_baseline != got_temporal
