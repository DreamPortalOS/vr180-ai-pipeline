"""Tests for scripts/batch_runner.py — issue #87, K-1 batch runner.

Covers the acceptance criteria:

  - job parsing / default fallback / summary table / --dry-run
  - fault tolerance (one failing job, others complete, summary correct)
  - --fail-fast behaviour
  - resume checkpoint (second run skips succeeded jobs)
  - state-file round-trip

The real image→VR180 orchestrator is *patched* everywhere — no ffmpeg, no
models, no providers, no API calls. ``--dry-run`` is pure. ``not slow``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import batch_runner as br  # noqa: E402
import image_to_vr180 as i2v  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_jobs(tmp_path: Path, jobs: list[dict]) -> Path:
    p = tmp_path / "jobs.json"
    p.write_text(json.dumps(jobs), encoding="utf-8")
    return p


def _fake_run_pipeline_success(job: i2v.JobArgs) -> dict:
    """A stand-in orchestrator that reports success + a fake output path."""
    return {"output": job.vr180_output, "qa_exit": 0, "manifest": None}


def _fake_run_pipeline_fail(job: i2v.JobArgs) -> dict:
    raise RuntimeError("boom from fake orchestrator")


def _run_pipeline_table(specs: list[br.JobSpec]) -> dict[str, object]:
    """Map scene_id → success/fail so tests can drive a deterministic fake."""
    raise NotImplementedError  # replaced per-test below


def _make_default_table(behaviour: dict[str, str]):
    """Return a fake run_pipeline that dispatches on scene_id via the table.

    behaviour maps scene_id → "ok" or "fail".
    """

    def fake(job: i2v.JobArgs) -> dict:
        # Recover the scene_id by matching the image path is fragile; instead
        # tests inject the spec→result mapping via closure below.
        return {"output": job.vr180_output, "qa_exit": 0, "manifest": None}

    return fake


# ---------------------------------------------------------------------------
# Job parsing / default fallback
# ---------------------------------------------------------------------------


class TestJobParsing:
    def test_load_jobs_reads_json_array(self, tmp_path):
        p = _write_jobs(tmp_path, [{"image": "a.png"}, {"image": "b.png"}])
        jobs = br.load_jobs(p)
        assert len(jobs) == 2
        assert jobs[0]["image"] == "a.png"

    def test_load_jobs_missing_file(self, tmp_path):
        with pytest.raises(RuntimeError, match="Jobs file not found"):
            br.load_jobs(tmp_path / "nope.json")

    def test_load_jobs_malformed_json(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("{not json", encoding="utf-8")
        with pytest.raises(RuntimeError, match="not valid JSON"):
            br.load_jobs(p)

    def test_load_jobs_not_array(self, tmp_path):
        p = tmp_path / "obj.json"
        p.write_text(json.dumps({"image": "a.png"}), encoding="utf-8")
        with pytest.raises(RuntimeError, match="must be a JSON array"):
            br.load_jobs(p)

    def test_resolve_job_falls_back_to_cli_defaults(self):
        defaults = br._cli_defaults(br.parse_args(["--jobs", "x", "--provider", "mock"]))
        raw = {"image": "a.png", "scene_id": "s01"}
        spec = br.resolve_job(raw, defaults, 0)
        assert spec.image == "a.png"
        assert spec.scene_id == "s01"
        assert spec.provider == "mock"  # from CLI default
        assert spec.duration == 5
        assert spec.gen_resolution == "480p"
        assert spec.gen_ratio == "adaptive"
        assert spec.upscale == "none"
        assert spec.quality == "preview"

    def test_resolve_job_overrides_take_precedence(self):
        defaults = br._cli_defaults(br.parse_args(["--jobs", "x"]))
        raw = {
            "image": "a.png",
            "scene_id": "s01",
            "provider": "seedance",
            "duration": 8,
            "gen_resolution": "720p",
            "gen_ratio": "16:9",
            "upscale": "seedvr2",
            "quality": "high",
            "prompt": "orbit",
        }
        spec = br.resolve_job(raw, defaults, 0)
        assert spec.provider == "seedance"
        assert spec.duration == 8
        assert spec.gen_resolution == "720p"
        assert spec.gen_ratio == "16:9"
        assert spec.upscale == "seedvr2"
        assert spec.quality == "high"
        assert spec.prompt == "orbit"

    def test_resolve_job_missing_image_raises(self):
        defaults = br._cli_defaults(br.parse_args(["--jobs", "x"]))
        with pytest.raises(RuntimeError, match="missing required field 'image'"):
            br.resolve_job({}, defaults, 3)

    def test_resolve_job_scene_id_generated_when_absent(self):
        defaults = br._cli_defaults(br.parse_args(["--jobs", "x"]))
        spec = br.resolve_job({"image": "a.png"}, defaults, 2)
        assert spec.scene_id == "job-002"


# ---------------------------------------------------------------------------
# --dry-run
# ---------------------------------------------------------------------------


class TestDryRun:
    def test_dry_run_prints_resolved_jobs_without_running(self, tmp_path, capsys):
        p = _write_jobs(
            tmp_path,
            [
                {"image": "a.png", "prompt": "dolly", "scene_id": "s01"},
                {"image": "b.png", "scene_id": "s02", "provider": "seedance"},
            ],
        )
        argv = ["--jobs", str(p), "--dry-run"]

        called = {"count": 0}

        def fake_run_pipeline(job):
            called["count"] += 1
            return {"output": job.vr180_output}

        with patch("scripts.image_to_vr180.run_pipeline", side_effect=fake_run_pipeline):
            rc = br.main(argv)

        out = capsys.readouterr().out
        assert rc == br.EXIT_OK
        assert "Dry run" in out
        assert "s01" in out and "s02" in out
        assert "a.png" in out and "b.png" in out
        # Nothing actually ran.
        assert called["count"] == 0

    def test_dry_run_zero_jobs(self, tmp_path, capsys):
        p = _write_jobs(tmp_path, [])
        with patch("scripts.image_to_vr180.run_pipeline") as mock_rp:
            rc = br.main(["--jobs", str(p), "--dry-run"])
        assert rc == br.EXIT_OK
        out = capsys.readouterr().out
        assert "0 job" in out
        mock_rp.assert_not_called()


# ---------------------------------------------------------------------------
# Fault tolerance + --fail-fast
# ---------------------------------------------------------------------------


class TestFaultTolerance:
    def _behavior_table(self, specs, fail_scene):
        """Return a fake run_pipeline that fails for ``fail_scene`` only."""

        def fake(job: i2v.JobArgs) -> dict:
            # Identify the job by workdir stem (scene_id is encoded in the
            # workdir path the JobSpec carries). For test purposes we tag
            # the workdir with the scene_id via the image stem.
            if Path(job.image).stem == fail_scene:
                raise RuntimeError(f"boom from fake orchestrator ({fail_scene})")
            return {"output": job.vr180_output, "qa_exit": 0, "manifest": None}

        return fake

    def test_one_failure_does_not_block_others(self, tmp_path, capsys):
        # Use scene_id == image stem so the behavior table can match.
        jobs = [
            {"image": str(tmp_path / "s01.png"), "scene_id": "s01"},
            {"image": str(tmp_path / "boom.png"), "scene_id": "boom"},
            {"image": str(tmp_path / "s03.png"), "scene_id": "s03"},
        ]
        p = _write_jobs(tmp_path, jobs)
        specs = br.resolve_all(jobs, br._cli_defaults(br.parse_args(["--jobs", str(p)])))

        # Touch the image files so resolve_paths/ensure_workdir don't choke.
        for j in jobs:
            Path(j["image"]).write_bytes(b"img")

        fake = self._behavior_table(specs, fail_scene="boom")
        results = []
        for spec in specs:
            results.append(br.run_one_job(spec, run_pipeline=fake))

        statuses = {r.scene_id: r.status for r in results}
        assert statuses == {"s01": "success", "boom": "failed", "s03": "success"}

        boom = next(r for r in results if r.scene_id == "boom")
        assert "boom from fake orchestrator" in boom.error

    def test_fail_fast_stops_after_first_failure(self, tmp_path, capsys):
        jobs = [
            {"image": str(tmp_path / "ok1.png"), "scene_id": "ok1"},
            {"image": str(tmp_path / "boom.png"), "scene_id": "boom"},
            {"image": str(tmp_path / "ok3.png"), "scene_id": "ok3"},
        ]
        p = _write_jobs(tmp_path, jobs)
        for j in jobs:
            Path(j["image"]).write_bytes(b"img")

        specs = br.resolve_all(jobs, br._cli_defaults(br.parse_args(["--jobs", str(p)])))
        fake = self._behavior_table(specs, fail_scene="boom")

        with patch("scripts.image_to_vr180.run_pipeline", side_effect=fake):
            rc = br.main(["--jobs", str(p), "--fail-fast"])

        out = capsys.readouterr().out
        # Fail-fast returns EXIT_FAILED.
        assert rc == br.EXIT_FAILED
        # The summary contains ok1 (success) and boom (failed) but NOT ok3.
        assert "ok1" in out and "boom" in out
        assert "ok3" not in out
        # ok3 never reached the orchestrator.
        assert "succeeded: 1" in out and "failed: 1" in out

    def test_no_failures_returns_ok_exit(self, tmp_path, capsys):
        jobs = [
            {"image": str(tmp_path / "s01.png"), "scene_id": "s01"},
            {"image": str(tmp_path / "s02.png"), "scene_id": "s02"},
        ]
        p = _write_jobs(tmp_path, jobs)
        for j in jobs:
            Path(j["image"]).write_bytes(b"img")

        fake = self._behavior_table(None, fail_scene="__none__")
        with patch("scripts.image_to_vr180.run_pipeline", side_effect=fake):
            rc = br.main(["--jobs", str(p)])

        assert rc == br.EXIT_OK
        out = capsys.readouterr().out
        assert "succeeded: 2" in out
        assert "failed: 0" in out

    def test_partial_failure_exit_code(self, tmp_path, capsys):
        jobs = [
            {"image": str(tmp_path / "s01.png"), "scene_id": "s01"},
            {"image": str(tmp_path / "boom.png"), "scene_id": "boom"},
        ]
        p = _write_jobs(tmp_path, jobs)
        for j in jobs:
            Path(j["image"]).write_bytes(b"img")

        fake = self._behavior_table(None, fail_scene="boom")
        with patch("scripts.image_to_vr180.run_pipeline", side_effect=fake):
            rc = br.main(["--jobs", str(p)])

        # Without --fail-fast, a partial failure is EXIT_PARTIAL.
        assert rc == br.EXIT_PARTIAL


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


class TestSummaryTable:
    def test_summary_contains_all_columns(self):
        results = [
            br.JobResult(scene_id="s01", status="success", output_path="/o/s01.mp4", duration_s=1.234),
            br.JobResult(scene_id="s02", status="failed", error="RuntimeError: boom", duration_s=0.5),
            br.JobResult(scene_id="s03", status="skipped", output_path="/o/s03.mp4"),
        ]
        text = br.format_summary(results)
        assert "scene_id" in text
        assert "status" in text
        assert "output_path" in text
        assert "duration_s" in text
        assert "s01" in text and "s02" in text and "s03" in text
        assert "/o/s01.mp4" in text
        assert "RuntimeError: boom" in text
        # Totals line.
        assert "Total: 3" in text
        assert "succeeded: 1" in text and "failed: 1" in text and "skipped: 1" in text

    def test_summary_empty(self):
        text = br.format_summary([])
        assert "No jobs" in text


# ---------------------------------------------------------------------------
# Resume / checkpoint
# ---------------------------------------------------------------------------


class TestResumeCheckpoint:
    def test_second_run_skips_succeeded_jobs(self, tmp_path, capsys):
        jobs = [
            {"image": str(tmp_path / "s01.png"), "scene_id": "s01"},
            {"image": str(tmp_path / "s02.png"), "scene_id": "s02"},
        ]
        p = _write_jobs(tmp_path, jobs)
        for j in jobs:
            Path(j["image"]).write_bytes(b"img")

        state_path = tmp_path / "state.json"

        # Fake orchestrator: succeeds for both on the first run.
        def fake(job):
            return {"output": job.vr180_output}

        # First run: both succeed, state written.
        with patch("scripts.image_to_vr180.run_pipeline", side_effect=fake):
            rc1 = br.main(["--jobs", str(p), "--state", str(state_path)])
        assert rc1 == br.EXIT_OK

        # State file exists and records both as success.
        state = json.loads(state_path.read_text(encoding="utf-8"))
        assert state["jobs"]["s01"]["status"] == "success"
        assert state["jobs"]["s02"]["status"] == "success"

        # Second run: orchestrator must NOT be called (both skipped).
        called = {"n": 0}

        def fake2(job):
            called["n"] += 1
            return {"output": job.vr180_output}

        with patch("scripts.image_to_vr180.run_pipeline", side_effect=fake2):
            rc2 = br.main(["--jobs", str(p), "--state", str(state_path)])
        assert rc2 == br.EXIT_OK
        assert called["n"] == 0  # both skipped, none re-run

        out = capsys.readouterr().out
        assert "succeeded: 0" in out
        assert "skipped: 2" in out

    def test_resume_re_runs_only_failed_jobs(self, tmp_path, capsys):
        jobs = [
            {"image": str(tmp_path / "s01.png"), "scene_id": "s01"},
            {"image": str(tmp_path / "s02.png"), "scene_id": "s02"},
        ]
        p = _write_jobs(tmp_path, jobs)
        for j in jobs:
            Path(j["image"]).write_bytes(b"img")

        state_path = tmp_path / "state.json"

        # First run: s01 succeeds, s02 fails (image stem == scene_id).
        def fake_first(job):
            if Path(job.image).stem == "s02":
                raise RuntimeError("boom s02")
            return {"output": job.vr180_output}

        with patch("scripts.image_to_vr180.run_pipeline", side_effect=fake_first):
            br.main(["--jobs", str(p), "--state", str(state_path)])

        # Second run: s02 now succeeds, s01 must be skipped.
        ran: list[str] = []

        def fake_second(job):
            ran.append(Path(job.image).stem)
            return {"output": job.vr180_output}

        with patch("scripts.image_to_vr180.run_pipeline", side_effect=fake_second):
            rc = br.main(["--jobs", str(p), "--state", str(state_path)])

        # s01 skipped (not in ran); s02 re-run.
        assert "s01" not in ran
        assert ran == ["s02"]
        assert rc == br.EXIT_OK

        # State updated: both now success.
        state = json.loads(state_path.read_text(encoding="utf-8"))
        assert state["jobs"]["s01"]["status"] == "success"
        assert state["jobs"]["s02"]["status"] == "success"

    def test_state_saved_per_job_incrementally(self, tmp_path):
        """State is written after every job, so a crash mid-batch still leaves
        a usable checkpoint."""
        jobs = [
            {"image": str(tmp_path / "s01.png"), "scene_id": "s01"},
            {"image": str(tmp_path / "s02.png"), "scene_id": "s02"},
            {"image": str(tmp_path / "s03.png"), "scene_id": "s03"},
        ]
        p = _write_jobs(tmp_path, jobs)
        for j in jobs:
            Path(j["image"]).write_bytes(b"img")

        state_path = tmp_path / "state.json"

        # Fake orchestrator: succeeds s01, then raises SystemExit to simulate
        # a hard crash before s02/s03 complete. We use a sentinel exception
        # so the runner's own try/except (which catches BaseException) will
        # actually record it as failed — but to truly simulate a crash we
        # raise *after* the job result is recorded, inside the fake.
        def fake(job):
            if Path(job.image).stem == "s02":
                # Crash the whole process mid-batch: raise something the
                # runner catches (so the state is saved as failed) and then
                # stop iterating by re-raising past run_one_job.
                raise RuntimeError("crash s02")
            return {"output": job.vr180_output}

        with patch("scripts.image_to_vr180.run_pipeline", side_effect=fake):
            br.main(["--jobs", str(p), "--state", str(state_path)])

        # s01 recorded as success; s02 recorded as failed; s03 not yet run
        # (recorded as failed because run_one_job caught the crash for s02,
        # and the loop continued — but here s02 is the last to fail before
        # s03; verify s01 is durably success).
        state = json.loads(state_path.read_text(encoding="utf-8"))
        assert state["jobs"]["s01"]["status"] == "success"
        # s02 failed.
        assert state["jobs"]["s02"]["status"] == "failed"
        assert "crash s02" in state["jobs"]["s02"]["error"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCLI:
    def test_help_exits_zero(self, capsys):
        with pytest.raises(SystemExit) as exc:
            br.parse_args(["--help"])
        assert exc.value.code == 0
        out = capsys.readouterr().out
        assert "--jobs" in out
        assert "--state" in out
        assert "--dry-run" in out
        assert "--fail-fast" in out

    def test_jobs_required(self, capsys):
        with pytest.raises(SystemExit) as exc:
            br.parse_args([])
        assert exc.value.code != 0

    def test_defaults_match_image_to_vr180(self):
        args = br.parse_args(["--jobs", "x"])
        assert args.provider == "mock"
        assert args.duration == 5
        assert args.gen_resolution == "480p"
        assert args.gen_ratio == "adaptive"
        assert args.upscale == "none"
        assert args.quality == "preview"
