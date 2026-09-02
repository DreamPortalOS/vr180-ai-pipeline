"""Tests for the Stage-2 performance bench (scripts/stage2_bench.py).

P-3 (#223): verifies the grid-expansion / runner-injection / report surface
of the bench tool.  Every runner call is a **fake** — no ffmpeg, CUDA,
GPU, or nvidia-smi is ever touched.  The fake runner mimics real behaviour
(exit codes, OOM stderr) so the bench's failure-tolerance and OOM marking
are exercised without any inference.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts.stage2_bench import (
    _MISSING,
    ENV_FRAMES_CHUNK,
    ENV_OVERLAP,
    ENV_TILE_NUM,
    build_grid,
    build_run_pipeline_args,
    combo_env,
    combo_name,
    main,
    render_bench_json,
    render_bench_md,
    run_bench,
    run_combo,
)


class FakeCompletedProcess:
    """CompletedProcess-like object the fake runner returns."""

    def __init__(
        self,
        returncode: int = 0,
        stdout: str = "",
        stderr: str = "",
    ) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class TestBuildGrid:
    """Grid expansion = Cartesian product of the three knobs."""

    def test_grid_size_cartesian_product(self) -> None:
        """3 chunks × 2 tiles × 1 overlap = 6 combos (the acceptance example)."""
        grid = build_grid(
            frames_chunks=[8, 16, 23],
            tile_nums=[1, 2],
            overlaps=[3],
        )
        assert len(grid) == 6

    def test_grid_is_full_cartesian_product(self) -> None:
        combos = build_grid(frames_chunks=[8, 16], tile_nums=[1, 2], overlaps=[3])
        got = {(c["frames_chunk"], c["tile_num"], c["overlap"]) for c in combos}
        assert got == {
            (8, 1, 3),
            (8, 2, 3),
            (16, 1, 3),
            (16, 2, 3),
        }

    def test_order_outer_to_inner(self) -> None:
        """Order is frames_chunk (outer) → tile_num → overlap (inner)."""
        grid = build_grid(frames_chunks=[8, 16], tile_nums=[1, 2], overlaps=[3])
        assert [c["frames_chunk"] for c in grid] == [8, 8, 16, 16]

    def test_empty_axis_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            build_grid(frames_chunks=[], tile_nums=[1], overlaps=[3])
        with pytest.raises(ValueError, match="at least one"):
            build_grid(frames_chunks=[8], tile_nums=[])
        with pytest.raises(ValueError, match="at least one"):
            build_grid(frames_chunks=[8], tile_nums=[1], overlaps=[])


class TestComboNamingAndEnv:
    def test_combo_name_encodes_all_params(self) -> None:
        name = combo_name({"frames_chunk": 16, "tile_num": 2, "overlap": 3})
        assert name == "fc16_tn2_ov3"

    def test_combo_env_contains_all_three_vars(self) -> None:
        env = combo_env({"frames_chunk": 23, "tile_num": 1, "overlap": 3})
        assert env == {
            ENV_FRAMES_CHUNK: "23",
            ENV_OVERLAP: "3",
            ENV_TILE_NUM: "1",
        }


class TestBuildRunPipelineArgs:
    def test_argv_is_list_not_shell_string(self) -> None:
        argv = build_run_pipeline_args(
            input_path="clip.mp4",
            output_path="/out/x.mp4",
            temp_dir=Path("/out/_work/c1"),
            max_frames=24,
        )
        assert isinstance(argv, list)
        assert all(isinstance(a, str) for a in argv)

    def test_argv_selects_stereo_and_depth_models(self) -> None:
        argv = build_run_pipeline_args("clip.mp4", "/out/x.mp4", Path("/t"), max_frames=24)
        assert "--stereo-model" in argv and "stereocrafter" in argv
        assert "--depth-model" in argv and "depthcrafter" in argv

    def test_argv_contains_per_combo_temp_dir(self) -> None:
        temp = Path("/out/_work/fc8_tn1_ov3")
        argv = build_run_pipeline_args("clip.mp4", "/out/x.mp4", temp)
        assert "--temp-dir" in argv
        assert argv[argv.index("--temp-dir") + 1] == str(temp)

    def test_max_frames_in_argv(self) -> None:
        argv = build_run_pipeline_args("clip.mp4", "/out/x.mp4", Path("/t"), max_frames=24)
        assert "--max-frames" in argv
        assert argv[argv.index("--max-frames") + 1] == "24"

    def test_no_max_frames_omits_flag(self) -> None:
        argv = build_run_pipeline_args("clip.mp4", "/out/x.mp4", Path("/t"), max_frames=None)
        assert "--max-frames" not in argv


class TestRunComboWithFakeRunner:
    """Single-combo runner behaviour with an injected fake runner."""

    def _run(self, **kwargs) -> dict:
        calls: list[tuple] = []

        def fake_runner(argv: list[str], env: dict[str, str]) -> FakeCompletedProcess:
            calls.append((argv, env))
            return FakeCompletedProcess(**kwargs)

        result = run_combo(
            input_path="clip.mp4",
            output_path="/out/_work/c1/bench_out.mp4",
            combo={"frames_chunk": 8, "tile_num": 1, "overlap": 3},
            max_frames=24,
            runner=fake_runner,
        )
        return {"result": result, "calls": calls}

    def test_runner_receives_correct_env_vars(self) -> None:
        out = self._run()
        _, env = out["calls"][0]
        assert env[ENV_FRAMES_CHUNK] == "8"
        assert env[ENV_OVERLAP] == "3"
        assert env[ENV_TILE_NUM] == "1"

    def test_runner_receives_list_form_argv(self) -> None:
        out = self._run()
        argv, _ = out["calls"][0]
        assert isinstance(argv, list)
        assert "--stereo-model" in argv

    def test_wall_clock_and_spf_recorded(self) -> None:
        result = self._run()["result"]
        assert result.wall_seconds >= 0.0
        assert result.frames == 24
        assert result.seconds_per_frame is not None
        assert result.seconds_per_frame > 0

    def test_nonzero_exit_recorded_and_not_raised(self) -> None:
        result = self._run(returncode=2)["result"]
        assert result.returncode == 2
        assert result.ok is False
        # test still reaches here → the non-zero exit did not raise

    def test_oom_detected_from_stderr_cuda_marker(self) -> None:
        result = self._run(
            returncode=1,
            stderr="ERROR: CUDA out of memory. Tried to allocate 2.00 GiB",
        )["result"]
        assert result.oom is True

    def test_oom_detected_from_stderr_outofmemory_marker(self) -> None:
        result = self._run(
            returncode=1,
            stderr="RuntimeError: torch.cuda.OutOfMemoryError: allocation failed",
        )["result"]
        assert result.oom is True

    def test_no_oom_when_stderr_clean(self) -> None:
        result = self._run(stdout="done", stderr="")["result"]
        assert result.oom is False

    def test_runner_exception_recorded_not_raised(self) -> None:
        def boom_runner(argv: list[str], env: dict[str, str]) -> FakeCompletedProcess:
            raise RuntimeError("boom")

        result = run_combo(
            input_path="clip.mp4",
            output_path="/out/_work/c1/bench_out.mp4",
            combo={"frames_chunk": 8, "tile_num": 1, "overlap": 3},
            max_frames=24,
            runner=boom_runner,
        )
        assert result.returncode == 127
        assert "boom" in result.stderr


class TestNvidiaSmiFallback:
    """CPU-safe: when nvidia-smi is unavailable memory is ``—`` and the run succeeds."""

    def test_unavailable_nvidia_smi_yields_dash_memory(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from scripts import stage2_bench as bench_mod

        def fake_probe(self: object) -> None:
            self.available = False

        def fake_sample() -> int:
            raise RuntimeError("no gpu")

        monkeypatch.setattr(bench_mod.SampleState, "probe", fake_probe)
        monkeypatch.setattr(bench_mod, "sample_peak_memory", fake_sample)

        calls: list[tuple] = []

        def fake_runner(argv: list[str], env: dict[str, str]) -> FakeCompletedProcess:
            calls.append((argv, env))
            return FakeCompletedProcess(returncode=0, stdout="ok", stderr="")

        result = run_combo(
            input_path="clip.mp4",
            output_path="/out/_work/c1/bench_out.mp4",
            combo={"frames_chunk": 8, "tile_num": 1, "overlap": 3},
            max_frames=24,
            runner=fake_runner,
        )
        assert result.peak_memory_mib == _MISSING
        assert result.returncode == 0
        assert result.ok is True
        assert len(calls) == 1


class TestRunBenchEndToEnd:
    """Full grid orchestration with an injected fake runner."""

    def test_grid_expands_to_six_combos_each_with_correct_env(self, tmp_path: Path) -> None:
        ok = FakeCompletedProcess(returncode=0, stdout="ok", stderr="")
        calls: list[tuple] = []

        def accepting_runner(argv: list[str], env: dict[str, str]) -> FakeCompletedProcess:
            calls.append((argv, env))
            return ok

        results = run_bench(
            input_path="clip.mp4",
            outdir=tmp_path,
            frames_chunks=[8, 16, 23],
            tile_nums=[1, 2],
            overlaps=[3],
            max_frames=24,
            runner=accepting_runner,
        )
        assert len(results) == 6
        assert len(calls) == 6
        # Every call carries exactly the three knobs for its combo.
        combos = build_grid([8, 16, 23], [1, 2], [3])
        got_envs = {(c[ENV_FRAMES_CHUNK], c[ENV_TILE_NUM], c[ENV_OVERLAP]) for _, c in calls}
        expected = {(str(c["frames_chunk"]), str(c["tile_num"]), str(c["overlap"])) for c in combos}
        assert got_envs == expected

    def test_each_combo_gets_distinct_temp_dir(self, tmp_path: Path) -> None:
        ok = FakeCompletedProcess(returncode=0, stdout="ok", stderr="")

        def runner(argv: list[str], env: dict[str, str]) -> FakeCompletedProcess:
            return ok

        results = run_bench(
            input_path="clip.mp4",
            outdir=tmp_path,
            frames_chunks=[8, 16],
            tile_nums=[1],
            overlaps=[3],
            max_frames=24,
            runner=runner,
        )
        temp_dirs = {r.cmd[r.cmd.index("--temp-dir") + 1] for r in results}
        assert len(temp_dirs) == len(results)
        for td in temp_dirs:
            assert "_work" in td

    def test_one_combo_fails_does_not_abort_rest(self, tmp_path: Path) -> None:
        fc8 = FakeCompletedProcess(returncode=1, stdout="", stderr="CUDA out of memory")
        ok = FakeCompletedProcess(returncode=0, stdout="ok", stderr="")

        def runner(argv: list[str], env: dict[str, str]) -> FakeCompletedProcess:
            fc = env[ENV_FRAMES_CHUNK]
            return fc8 if fc == "8" else ok

        results = run_bench(
            input_path="clip.mp4",
            outdir=tmp_path,
            frames_chunks=[8, 16],
            tile_nums=[1],
            overlaps=[3],
            max_frames=24,
            runner=runner,
        )
        assert len(results) == 2
        by_fc = {r.combo["frames_chunk"]: r for r in results}
        assert by_fc[8].returncode == 1
        assert by_fc[8].oom is True
        assert by_fc[16].returncode == 0
        assert by_fc[16].ok is True

    def test_md_sorted_by_sframe_ascending(self, tmp_path: Path) -> None:
        # Slow combo (frames_chunk=8) and fast combo (frames_chunk=16).
        # We can't easily control wall time in the fake runner, but the
        # report writer's sort is deterministic on the recorded values.
        slow = FakeCompletedProcess(returncode=0, stdout="ok", stderr="")
        fast = FakeCompletedProcess(returncode=0, stdout="ok", stderr="")

        def runner(argv: list[str], env: dict[str, str]) -> FakeCompletedProcess:
            fc = env[ENV_FRAMES_CHUNK]
            if fc == "8":
                return slow
            return fast

        results = run_bench(
            input_path="clip.mp4",
            outdir=tmp_path,
            frames_chunks=[8, 16],
            tile_nums=[1],
            overlaps=[3],
            max_frames=24,
            runner=runner,
        )
        assert len(results) == 2
        md = (tmp_path / "bench.md").read_text(encoding="utf-8")
        # Both combos appear in the table.
        assert "fc8_tn1_ov3" in md
        assert "fc16_tn1_ov3" in md

    def test_json_has_all_combos(self, tmp_path: Path) -> None:
        ok = FakeCompletedProcess(returncode=0, stdout="ok", stderr="")

        def runner(argv: list[str], env: dict[str, str]) -> FakeCompletedProcess:
            return ok

        run_bench(
            input_path="clip.mp4",
            outdir=tmp_path,
            frames_chunks=[8, 16],
            tile_nums=[1],
            overlaps=[3],
            max_frames=24,
            runner=runner,
        )
        payload = json.loads((tmp_path / "bench.json").read_text(encoding="utf-8"))
        assert len(payload["combos"]) == 2
        for combo in payload["combos"]:
            assert "frames_chunk" in combo
            assert "tile_num" in combo
            assert "overlap" in combo
            assert "seconds_per_frame" in combo
            assert "peak_memory_mib" in combo

    def test_dry_run_does_not_call_runner(self, tmp_path: Path) -> None:
        def must_not_run(argv: list[str], env: dict[str, str]) -> FakeCompletedProcess:
            raise AssertionError("dry-run must not spawn the runner")

        results = run_bench(
            input_path="clip.mp4",
            outdir=tmp_path,
            frames_chunks=[8, 16, 23],
            tile_nums=[1, 2],
            overlaps=[3],
            max_frames=24,
            dry_run=True,
            runner=must_not_run,
        )
        # Still returns one result stub per combo so callers have the plan.
        assert len(results) == 6
        # And no output files were written.
        assert not (tmp_path / "bench.md").exists()
        assert not (tmp_path / "bench.json").exists()


class TestRenderReports:
    """Report writers — pure functions, sorted output, OOM marking."""

    def _results(self, combos_spf: list[tuple[dict[str, int], float | None, bool]]) -> list:
        from scripts.stage2_bench import ComboResult

        out = []
        for combo, spf, oom in combos_spf:
            name = combo_name(combo)
            out.append(
                ComboResult(
                    combo=combo,
                    name=name,
                    cmd=["run_pipeline"],
                    env=combo_env(combo),
                    wall_seconds=(spf if spf is not None else 1.0),
                    frames=24,
                    seconds_per_frame=spf,
                    returncode=(1 if oom else 0),
                    oom=oom,
                    peak_memory_mib="8200",
                )
            )
        return out

    def test_md_sorted_by_sframe_ascending(self) -> None:
        # Deliberately feed rows out of order.
        results = self._results(
            [
                ({"frames_chunk": 8, "tile_num": 1, "overlap": 3}, 5.0, False),
                ({"frames_chunk": 16, "tile_num": 1, "overlap": 3}, 2.0, False),
            ]
        )
        md = render_bench_md(input_path="clip.mp4", max_frames=24, results=results)
        pos_fast = md.index("fc16_tn1_ov3")
        pos_slow = md.index("fc8_tn1_ov3")
        assert pos_fast < pos_slow

    def test_md_marks_oom_row_with_red_cross(self) -> None:
        results = self._results(
            [
                ({"frames_chunk": 23, "tile_num": 2, "overlap": 3}, None, True),
                ({"frames_chunk": 16, "tile_num": 1, "overlap": 3}, 2.0, False),
            ]
        )
        md = render_bench_md(input_path="clip.mp4", max_frames=24, results=results)
        assert "❌" in md
        assert "fc23_tn2_ov3" in md

    def test_oom_row_sorts_to_bottom(self) -> None:
        results = self._results(
            [
                ({"frames_chunk": 23, "tile_num": 2, "overlap": 3}, None, True),
                ({"frames_chunk": 16, "tile_num": 1, "overlap": 3}, 2.0, False),
                ({"frames_chunk": 8, "tile_num": 1, "overlap": 3}, 4.0, False),
            ]
        )
        md = render_bench_md(input_path="clip.mp4", max_frames=24, results=results)
        assert md.index("fc16_tn1_ov3") < md.index("fc8_tn1_ov3")
        assert md.index("fc8_tn1_ov3") < md.index("fc23_tn2_ov3")

    def test_json_contains_all_combo_fields(self) -> None:
        results = self._results(
            [
                ({"frames_chunk": 8, "tile_num": 1, "overlap": 3}, 3.0, False),
            ]
        )
        payload = render_bench_json(input_path="clip.mp4", max_frames=24, results=results)
        assert payload["input"] == "clip.mp4"
        assert payload["max_frames"] == 24
        assert len(payload["combos"]) == 1
        combo = payload["combos"][0]
        assert combo["frames_chunk"] == 8
        assert combo["tile_num"] == 1
        assert combo["overlap"] == 3
        assert combo["ok"] is True


class TestCLI:
    def test_help_exits_zero(self, capsys: pytest.CaptureFixture) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "--frames-chunk" in out
        assert "--tile-num" in out
        assert "--overlap" in out
        assert "--dry-run" in out

    def test_main_rejects_bad_int_list(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        rc = main(
            [
                "--input",
                "a.mp4",
                "--outdir",
                str(tmp_path),
                "--frames-chunk",
                "8,abc,23",
                "--tile-num",
                "1",
                "--overlap",
                "3",
            ]
        )
        assert rc == 2
        assert "--frames-chunk" in capsys.readouterr().err

    def test_dry_run_exits_zero_without_writing_reports(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        rc = main(
            [
                "--input",
                "clip.mp4",
                "--outdir",
                str(tmp_path),
                "--frames-chunk",
                "8,16",
                "--tile-num",
                "1,2",
                "--overlap",
                "3",
                "--max-frames",
                "24",
                "--dry-run",
            ]
        )
        assert rc == 0
        printed = capsys.readouterr().out
        # dry-run prints the list-form command (not via shell).
        assert "run_pipeline" in printed
        assert ENV_FRAMES_CHUNK in printed
        assert not (tmp_path / "bench.md").exists()
        assert not (tmp_path / "bench.json").exists()
