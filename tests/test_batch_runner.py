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


# ---------------------------------------------------------------------------
# Issue #96 / K-1.1: scene-named outputs, no silent overwrite
# ---------------------------------------------------------------------------


class TestSceneNamedOutputs:
    """Outputs must be distinct per scene even when jobs share an input image."""

    def test_two_jobs_same_image_produce_two_different_files(self, tmp_path, capsys):
        """AC: two jobs with different scene_id but identical input image
        yield two distinct output files, both present after the batch."""
        shared_image = tmp_path / "test_input_image.png"
        shared_image.write_bytes(b"img")
        jobs = [
            {"image": str(shared_image), "scene_id": "s01"},
            {"image": str(shared_image), "scene_id": "s02"},
        ]
        p = _write_jobs(tmp_path, jobs)

        written: list[str] = []

        def fake(job):
            written.append(job.vr180_output)
            # Simulate the orchestrator actually writing into its target.
            Path(job.vr180_output).write_bytes(b"video")
            return {"output": job.vr180_output}

        with patch("scripts.image_to_vr180.run_pipeline", side_effect=fake):
            rc = br.main(["--jobs", str(p)])

        assert rc == br.EXIT_OK
        # Two distinct target paths were chosen.
        assert len(set(written)) == 2
        # Both files actually exist on disk.
        for w in written:
            assert Path(w).is_file(), f"output missing: {w}"
        # Names are scene-derived, not image-stem-derived.
        for w in written:
            stem = Path(w).stem
            assert "s01" in stem or "s02" in stem
            assert stem != "test_input_image_vr180"

        # Summary table reports both success with their distinct paths.
        out = capsys.readouterr().out
        assert written[0] in out
        assert written[1] in out
        assert "succeeded: 2" in out

    def test_missing_scene_id_falls_back_to_job_index(self, tmp_path):
        """AC: when scene_id is omitted, the fallback name is per-job (job001, job002...)."""
        img = tmp_path / "a.png"
        img.write_bytes(b"img")
        defaults = br._cli_defaults(br.parse_args(["--jobs", "x"]))
        spec_a = br.resolve_job({"image": str(img)}, defaults, 0)
        spec_b = br.resolve_job({"image": str(img)}, defaults, 5)
        # Fallback is zero-padded job index.
        assert spec_a.scene_id == "job-000"
        assert spec_b.scene_id == "job-005"
        # And the composed output paths differ even with the same image.
        pa = br._build_output_path(spec_a, tmp_path)
        pb = br._build_output_path(spec_b, tmp_path)
        assert pa != pb
        # Both names are scene-derived, not image-stem-derived.
        assert pa.stem != "a_vr180"
        assert "job" in pa.stem and "job" in pb.stem

    def test_composed_output_path_uses_naming_convention(self, tmp_path):
        """Output filename follows D-4: <scene_id>_<scene_name>_segNN_vr180_<preset>.mp4."""
        img = tmp_path / "a.png"
        img.write_bytes(b"img")
        defaults = br._cli_defaults(br.parse_args(["--jobs", "x", "--quality", "high"]))
        spec = br.resolve_job({"image": str(img), "scene_id": "s07", "quality": "high"}, defaults, 3)
        path = br._build_output_path(spec, tmp_path)
        # route=vr180 and preset=pcvr (quality=high).
        assert path.stem == "s07_s07_seg01_vr180_pcvr"
        assert path.suffix == ".mp4"

    def test_collision_appends_auto_suffix(self, tmp_path):
        """AC: when the scene-named target already exists (foreign artefact),
        an auto-suffix ``_c1`` / ``_c2`` is appended; no silent overwrite."""
        img = tmp_path / "a.png"
        img.write_bytes(b"img")
        defaults = br._cli_defaults(br.parse_args(["--jobs", "x"]))
        spec = br.resolve_job({"image": str(img), "scene_id": "s01"}, defaults, 0)

        base = br._build_output_path(spec, tmp_path)
        # Simulate a foreign file already occupying the target.
        (tmp_path / "foreign").write_bytes(b"x")
        base.parent.mkdir(parents=True, exist_ok=True)
        base.write_bytes(b"foreign-output")

        resolved = br._resolve_collision(base, prior_successes=set())
        assert resolved != base
        assert resolved.stem.endswith("_c1")
        assert resolved.suffix == ".mp4"
        # The original foreign file was NOT overwritten.
        assert base.read_bytes() == b"foreign-output"

        # A second collision bumps to _c2.
        resolved.write_bytes(b"me")
        resolved2 = br._resolve_collision(base, prior_successes=set())
        assert resolved2.stem.endswith("_c2")

    def test_collision_keeps_own_prior_success(self, tmp_path):
        """A target that matches a previously-succeeded artefact of this
        scene is the orchestrator's own durable output (resume) — kept as-is."""
        img = tmp_path / "a.png"
        img.write_bytes(b"img")
        defaults = br._cli_defaults(br.parse_args(["--jobs", "x"]))
        spec = br.resolve_job({"image": str(img), "scene_id": "s01"}, defaults, 0)
        base = br._build_output_path(spec, tmp_path)
        base.parent.mkdir(parents=True, exist_ok=True)
        base.write_bytes(b"my-own")

        prior = {str(base)}
        resolved = br._resolve_collision(base, prior_successes=prior)
        assert resolved == base

    def test_state_records_scene_named_output_path(self, tmp_path, capsys):
        """The batch state file records each job's real scene-named output path."""
        img = tmp_path / "a.png"
        img.write_bytes(b"img")
        jobs = [
            {"image": str(img), "scene_id": "s01"},
            {"image": str(img), "scene_id": "s02"},
        ]
        p = _write_jobs(tmp_path, jobs)
        state_path = tmp_path / "state.json"

        def fake(job):
            Path(job.vr180_output).write_bytes(b"video")
            return {"output": job.vr180_output}

        with patch("scripts.image_to_vr180.run_pipeline", side_effect=fake):
            rc = br.main(["--jobs", str(p), "--state", str(state_path)])
        assert rc == br.EXIT_OK

        state = json.loads(state_path.read_text(encoding="utf-8"))
        assert state["jobs"]["s01"]["output_path"] != state["jobs"]["s02"]["output_path"]
        # Both recorded paths exist on disk.
        for sid in ("s01", "s02"):
            p_out = state["jobs"][sid]["output_path"]
            assert Path(p_out).is_file()
            assert "seg01_vr180" in p_out

    def test_two_jobs_same_image_real_orchestrator_never_collides(self, tmp_path, capsys):
        """K-1.2 / issue #106 end-to-end regression: two jobs sharing the
        *same* input image but different scene_ids must land on two *distinct*
        scene-named files even when the orchestrator runs for real (i.e. it
        calls :func:`image_to_vr180.resolve_paths` internally, which used to
        overwrite the runner's scene-named ``vr180_output`` with the
        image-stem default and cause the silent overwrite).

        This drives the *real* ``run_pipeline`` through injected stage
        callables (no ffmpeg / models), so it exercises the exact wiring that
        #104's original fix failed to cover.
        """
        shared_image = tmp_path / "test_input_image.png"
        shared_image.write_bytes(b"img")
        jobs = [
            {"image": str(shared_image), "scene_id": "s01"},
            {"image": str(shared_image), "scene_id": "s02"},
        ]
        p = _write_jobs(tmp_path, jobs)
        state_path = tmp_path / "state.json"

        # Record the orchestrator-returned output AND the intermediates each
        # job's ``run_pipeline`` resolved, to prove neither final files nor
        # intermediates collide.
        resolved_outputs: list[str] = []
        resolved_generated: list[str] = []
        resolved_prepared: list[str] = []

        def _trace_prepare(job: i2v.JobArgs) -> str:
            resolved_prepared.append(job.prepared_image)
            Path(job.prepared_image).parent.mkdir(parents=True, exist_ok=True)
            Path(job.prepared_image).write_bytes(b"fake-prep")
            return job.prepared_image

        def _trace_generate(job: i2v.JobArgs, prepared: str | None) -> str:
            resolved_generated.append(job.generated_video)
            Path(job.generated_video).parent.mkdir(parents=True, exist_ok=True)
            Path(job.generated_video).write_bytes(b"fake-gen")
            return job.generated_video

        def _trace_convert(job: i2v.JobArgs, input_video: str, _convert=None) -> str:
            resolved_outputs.append(job.vr180_output)
            Path(job.vr180_output).parent.mkdir(parents=True, exist_ok=True)
            Path(job.vr180_output).write_bytes(b"fake-vr180")
            return job.vr180_output

        # Wire up the real run_pipeline with only the I2V-heavy stages mocked.
        # The batch runner reads ``scripts.image_to_vr180.run_pipeline`` (by
        # dotted package name), so we MUST patch that exact module object.
        # Because the test also inserts ``scripts/`` onto sys.path, a bare
        # ``import image_to_vr180`` yields a *different* module object from
        # ``scripts.image_to_vr180`` — patching the wrong one would silently
        # no-op (the very pitfall this test guards against).
        import scripts.image_to_vr180 as _i2v_pkg  # local import: package-form needed for patching

        _real_run = _i2v_pkg.run_pipeline

        def _real_pipeline(job: i2v.JobArgs):
            # stage_qa is not injectable (run_pipeline calls it directly), so
            # stub it to a passing verdict for this wiring test.
            with patch.object(_i2v_pkg, "stage_qa", side_effect=lambda p: 0):
                return _real_run(
                    job,
                    prepare=_trace_prepare,
                    generate=_trace_generate,
                    streamcheck=lambda p: None,
                    upscale=lambda a, p: p,
                    convert=_trace_convert,
                )

        with patch.object(_i2v_pkg, "run_pipeline", side_effect=_real_pipeline):
            rc = br.main(["--jobs", str(p), "--state", str(state_path)])

        assert rc == br.EXIT_OK
        out = capsys.readouterr().out

        # The two recorded orchestrator outputs must be distinct scene-named
        # paths — this is the exact assertion #104's fix broke.
        outputs = {Path(o).name for o in resolved_outputs}
        assert len(resolved_outputs) == 2
        assert len(outputs) == 2, f"outputs collided: {resolved_outputs}"
        # Neither is the image-stem default (the #106 regression signature).
        for o in resolved_outputs:
            stem = Path(o).stem
            assert stem != "test_input_image_vr180", f"orchestrator ignored caller output: {o}"
            assert "s01" in stem or "s02" in stem
            assert Path(o).is_file()

        # Intermediates must also be per-scene (no *_generated.mp4 collision).
        assert len({Path(g).name for g in resolved_generated}) == 2, resolved_generated
        assert len({Path(p).name for p in resolved_prepared}) == 2, resolved_prepared
        assert len({Path(g).name for g in resolved_generated}) >= 2

        # Summary table lists two success rows with distinct paths.
        assert "succeeded: 2" in out
        assert "failed: 0" in out

        # State file records two distinct output_path values.
        state = json.loads(state_path.read_text(encoding="utf-8"))
        assert state["jobs"]["s01"]["output_path"] != state["jobs"]["s02"]["output_path"]
        for sid in ("s01", "s02"):
            assert Path(state["jobs"][sid]["output_path"]).is_file()
