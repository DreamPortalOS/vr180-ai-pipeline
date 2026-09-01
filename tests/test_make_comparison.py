"""Tests for the comparison-sample builder (scripts/make_comparison.py, K-4 #138).

Verifies:
- Recipe table expansion + validation (defaults, duplicates, bad args)
- Output naming via pipeline.naming (recipe short name round-trips in filename)
- Render command assembly (input/output/base args/recipe args, list form)
- Summary table + comparison.md generation (empty owner scoring columns)
- --dry-run output (prints recipes + commands, never renders)
- --push-to-quest: no serial → graceful skip; with serial → push runner called

A fake render runner / QA runner / push runner is injected everywhere —
no real conversion, no ffprobe, no adb.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pytest
from scripts.make_comparison import (
    DEFAULT_RECIPES,
    QUEST_SERIAL_ENV,
    SCORE_COLUMNS,
    RecipeResult,
    build_render_command,
    load_recipes,
    main,
    parse_args,
    qa_recipe_output,
    recipe_output_name,
    render_comparison_md,
    render_summary_table,
    resolve_quest_serial,
    run_comparison,
)

from pipeline.naming import parse_scene_name


def _fake_runner_ok(touch: bool = True):
    calls: list[list[str]] = []

    def runner(cmd) -> str:
        calls.append(list(cmd))
        out = str(cmd[list(cmd).index("--output") + 1])
        if touch:
            Path(out).touch()
        return out

    return runner, calls


def _fake_qa(result_verdict: str = "VR180 (180° 3D SBS)"):
    def qa(output_path: str, row: RecipeResult) -> RecipeResult:
        row.resolution = "5760×2880"
        row.audio = "aac"
        row.verdict = result_verdict
        row.qa_failed = False
        return row

    return qa


class TestLoadRecipes:
    """Recipe table expansion + validation."""

    def test_default_recipes_present_in_order(self) -> None:
        names = [r.name for r in load_recipes()]
        assert names == ["baseline", "temporal", "temporal_occl"]
        assert names == [r["name"] for r in DEFAULT_RECIPES]

    def test_default_recipe_args(self) -> None:
        recipes = {r.name: r.args for r in load_recipes()}
        assert recipes["baseline"] == ["--comfort", "strong"]
        assert "--depth-model" in recipes["temporal"]
        assert "depthcrafter" in recipes["temporal"]
        assert "--stereo-model" in recipes["temporal_occl"]
        assert "stereocrafter" in recipes["temporal_occl"]

    def test_custom_table_expands(self) -> None:
        recipes = load_recipes([{"name": "x", "args": ["--foo"], "description": "d"}])
        assert len(recipes) == 1
        assert recipes[0].name == "x" and recipes[0].args == ["--foo"]
        assert recipes[0].description == "d"

    def test_duplicate_names_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicate recipe name"):
            load_recipes([{"name": "a", "args": []}, {"name": "a", "args": []}])

    def test_missing_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="name"):
            load_recipes([{"args": []}])

    def test_non_string_args_rejected(self) -> None:
        with pytest.raises(ValueError, match="args"):
            load_recipes([{"name": "a", "args": "--comfort strong"}])

    def test_empty_table_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            load_recipes([])


class TestRecipeOutputName:
    """Filename naming via D-4 pipeline.naming."""

    def test_recipe_short_name_in_filename(self) -> None:
        for recipe in ("baseline", "temporal", "temporal_occl"):
            name = recipe_output_name("My Clip 01.mp4", recipe)
            assert f"_{recipe}_seg01_" in name
            assert name.endswith(".mp4")

    def test_filename_roundtrips_through_parse_scene_name(self) -> None:
        name = recipe_output_name("beach.mp4", "temporal_occl")
        spec = parse_scene_name(name)
        assert spec.scene_name == "beach_temporal_occl"
        assert spec.route == "vr180"
        assert spec.projection_mark == "sbs"

    def test_unique_per_recipe(self) -> None:
        names = {recipe_output_name("a.mp4", r.name) for r in load_recipes()}
        assert len(names) == len(DEFAULT_RECIPES)

    def test_non_slug_source_stem_is_sanitised(self) -> None:
        """Arbitrary stems (spaces, uppercase) still compose without error."""
        name = recipe_output_name("My Cool Video!.mp4", "baseline")
        assert "baseline" in name
        parse_scene_name(name)  # must parse back


class TestBuildRenderCommand:
    """run_pipeline command-line assembly."""

    def test_command_shape(self) -> None:
        recipe = load_recipes()[1]  # temporal
        cmd = build_render_command("in.mp4", "out/x.mp4", recipe, ["--quality", "standard"])
        assert "--input" in cmd and "in.mp4" in cmd
        assert "--output" in cmd and "out/x.mp4" in cmd
        assert cmd[cmd.index("--output") + 1] == "out/x.mp4"
        # base args come before recipe args so recipes can override
        assert cmd.index("--quality") < cmd.index("--depth-model")
        assert "run_pipeline.py" in cmd[1]

    def test_command_is_list_form(self) -> None:
        recipe = load_recipes()[0]
        cmd = build_render_command("in.mp4", "o.mp4", recipe, [])
        assert isinstance(cmd, list)
        assert all(isinstance(tok, str) for tok in cmd)


class TestRunComparisonWithFakeRunner:
    """Orchestration with injected fakes (no real render/QA/adb)."""

    def test_renders_once_per_recipe_and_qa_each(self, tmp_path: Path) -> None:
        runner, calls = _fake_runner_ok()
        results = run_comparison(
            input_path="src.mp4",
            outdir=tmp_path,
            render_runner=runner,
            qa_runner=_fake_qa(),
        )
        assert len(calls) == len(DEFAULT_RECIPES)
        assert all(r.ok for r in results)
        assert all(r.resolution == "5760×2880" for r in results)
        assert all(r.audio == "aac" for r in results)
        # every render target lands in outdir with the recipe in its name
        for r in results:
            assert Path(r.output).parent == tmp_path
            assert Path(r.output).exists()

    def test_comparison_md_written(self, tmp_path: Path) -> None:
        runner, _ = _fake_runner_ok()
        run_comparison(
            input_path="src.mp4",
            outdir=tmp_path,
            render_runner=runner,
            qa_runner=_fake_qa(),
        )
        md = (tmp_path / "comparison.md").read_text(encoding="utf-8")
        for col in SCORE_COLUMNS:
            assert col in md
        for recipe in ("baseline", "temporal", "temporal_occl"):
            assert recipe in md
        assert "汇总" in md

    def test_failed_recipe_does_not_abort_others(self, tmp_path: Path) -> None:
        def flaky(cmd) -> str:
            out = str(cmd[list(cmd).index("--output") + 1])
            if "_temporal_seg01_" in out:  # the plain 'temporal' recipe fails
                raise RuntimeError("boom")
            Path(out).touch()
            return out

        results = run_comparison(
            input_path="src.mp4",
            outdir=tmp_path,
            render_runner=flaky,
            qa_runner=_fake_qa(),
        )
        by_name = {r.recipe: r for r in results}
        assert by_name["temporal"].ok is False
        assert "boom" in by_name["temporal"].error
        assert by_name["baseline"].ok and by_name["temporal_occl"].ok
        # comparison.md still written and lists the failure
        md = (tmp_path / "comparison.md").read_text(encoding="utf-8")
        assert "boom" in md


class TestSummaryAndComparisonMd:
    """Summary table + comparison.md rendering."""

    def _rows(self) -> list[RecipeResult]:
        return [
            RecipeResult(
                recipe="baseline",
                description="基线",
                output="/o/a_baseline.mp4",
                ok=True,
                elapsed_s=12.5,
                resolution="5760×2880",
                audio="aac",
                verdict="VR180 (180° 3D SBS)",
            ),
            RecipeResult(
                recipe="temporal",
                description="时序",
                output="/o/a_temporal.mp4",
                ok=False,
                elapsed_s=3.0,
                error="CUDA missing",
            ),
        ]

    def test_summary_table_columns(self) -> None:
        md = render_summary_table(self._rows())
        header = md.splitlines()[0]
        for col in ("recipe", "分辨率", "音频", "QA 判定", "耗时"):
            assert col in header
        assert "5760×2880" in md
        assert "aac" in md
        assert "12.5s" in md
        assert "CUDA missing" in md

    def test_comparison_md_has_empty_score_columns(self) -> None:
        md = render_comparison_md(self._rows(), source="src.mp4", outdir="/o")
        header = next(line for line in md.splitlines() if line.startswith("| 文件"))
        for col in SCORE_COLUMNS:
            assert col in header
        rows = [line for line in md.splitlines() if "_baseline.mp4" in line or "_temporal.mp4" in line]
        assert len(rows) == 2
        for row in rows:
            cells = [c.strip() for c in row.strip("|").split("|")]
            # 文件, recipe, 4 empty score cols, 备注
            assert len(cells) == 2 + len(SCORE_COLUMNS) + 1
            assert all(cell == "" for cell in cells[2:])

    def test_comparison_md_records_push_state(self) -> None:
        md_pushed = render_comparison_md(self._rows(), "s.mp4", "/o", pushed=True)
        md_not = render_comparison_md(self._rows(), "s.mp4", "/o", pushed=False)
        assert "是" in md_pushed
        assert "否" in md_not


class TestQAIntegration:
    """qa_recipe_output fills the row from a vr180_qa report (QA itself stubbed)."""

    def test_row_filled_from_report(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from scripts import make_comparison as mc

        class _FakeReport:
            width, height = 5760, 2880
            audio_codec = "aac"
            verdict = "VR180 (180° 3D SBS)"
            failed = False
            checks: ClassVar[list] = []

        monkeypatch.setattr(mc, "run_qa", lambda path: _FakeReport())
        row = RecipeResult(recipe="baseline", output=str(tmp_path / "x.mp4"))
        qa_recipe_output(str(tmp_path / "x.mp4"), row)
        assert row.resolution == "5760×2880"
        assert row.audio == "aac"
        assert row.verdict == "VR180 (180° 3D SBS)"
        assert row.qa_failed is False

    def test_missing_audio_shows_placeholder(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from scripts import make_comparison as mc

        class _FakeReport:
            width, height = 5760, 2880
            audio_codec = ""
            verdict = "VR180 (180° 3D SBS)"
            failed = False
            checks: ClassVar[list] = []

        monkeypatch.setattr(mc, "run_qa", lambda path: _FakeReport())
        row = qa_recipe_output("x.mp4", RecipeResult(recipe="r"))
        assert row.audio == "无"


class TestPushToQuest:
    """--push-to-quest behaviour."""

    def test_serial_resolution_priority(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(QUEST_SERIAL_ENV, raising=False)
        assert resolve_quest_serial(None) is None
        monkeypatch.setenv(QUEST_SERIAL_ENV, "envserial")
        assert resolve_quest_serial(None) == "envserial"
        assert resolve_quest_serial("flagserial") == "flagserial"
        monkeypatch.setenv(QUEST_SERIAL_ENV, "  ")
        assert resolve_quest_serial(None) is None

    def test_push_without_serial_skips_gracefully(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(QUEST_SERIAL_ENV, raising=False)
        runner, _ = _fake_runner_ok()

        def must_not_push(*a, **k):  # pragma: no cover - must not be called
            raise AssertionError("push runner must not be called without a serial")

        results = run_comparison(
            input_path="src.mp4",
            outdir=tmp_path,
            render_runner=runner,
            qa_runner=_fake_qa(),
            push=True,
            push_runner=must_not_push,
        )
        assert all(r.ok for r in results)
        md = (tmp_path / "comparison.md").read_text(encoding="utf-8")
        assert "已推送 Quest: 否" in md

    def test_push_with_serial_pushes_ok_artefacts_only(self, tmp_path: Path) -> None:
        pushed: list[tuple[list, str]] = []

        def fake_push(paths, serial):
            pushed.append((list(paths), serial))
            return []

        def flaky(cmd) -> str:
            out = str(cmd[list(cmd).index("--output") + 1])
            if "_baseline_seg01_" in out:
                raise RuntimeError("boom")
            Path(out).touch()
            return out

        run_comparison(
            input_path="src.mp4",
            outdir=tmp_path,
            render_runner=flaky,
            qa_runner=_fake_qa(),
            push=True,
            quest_serial="ABC123",
            push_runner=fake_push,
        )
        assert len(pushed) == 1
        paths, serial = pushed[0]
        assert serial == "ABC123"
        # failed recipe's artefact is not pushed
        assert len(paths) == 2
        assert all("_baseline_seg01_" not in p for p in paths)
        md = (tmp_path / "comparison.md").read_text(encoding="utf-8")
        assert "已推送 Quest: 是" in md

    def test_push_skipped_when_nothing_rendered(self, tmp_path: Path) -> None:
        def always_fail(cmd) -> str:
            raise RuntimeError("no GPU")

        def must_not_push(*a, **k):  # pragma: no cover
            raise AssertionError("push runner must not be called with zero artefacts")

        results = run_comparison(
            input_path="src.mp4",
            outdir=tmp_path,
            render_runner=always_fail,
            qa_runner=_fake_qa(),
            push=True,
            quest_serial="ABC123",
            push_runner=must_not_push,
        )
        assert not any(r.ok for r in results)


class TestDryRun:
    """--dry-run prints recipes + commands and never renders."""

    def test_dry_run_prints_and_skips_render(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        def must_not_render(*a, **k):  # pragma: no cover
            raise AssertionError("render runner must not be called in dry-run")

        results = run_comparison(
            input_path="clip.mp4",
            outdir=tmp_path,
            render_runner=must_not_render,
            qa_runner=_fake_qa(),
            dry_run=True,
        )
        out = capsys.readouterr().out
        assert "DRY RUN" in out
        for recipe in ("baseline", "temporal", "temporal_occl"):
            assert recipe in out
        assert "run_pipeline.py" in out
        assert "--depth-model depthcrafter" in out
        # no render happened → nothing rendered, no comparison.md
        assert not (tmp_path / "comparison.md").exists()
        assert len(results) == len(DEFAULT_RECIPES)
        assert all(not r.ok for r in results)

    def test_main_dry_run_exit_zero(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        rc = main(["--input", "clip.mp4", "--outdir", str(tmp_path), "--dry-run"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "DRY RUN" in out
        assert "命令:" in out


class TestCLI:
    """CLI parsing / entry-point behaviour."""

    def test_help_exits_zero(self, capsys: pytest.CaptureFixture) -> None:
        with pytest.raises(SystemExit) as exc_info:
            parse_args(["--help"])
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        for flag in ("--input", "--outdir", "--push-to-quest", "--quest-serial", "--dry-run", "--recipe"):
            assert flag in out

    def test_recipe_filter_unknown_rejected(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        rc = main(["--input", "a.mp4", "--outdir", str(tmp_path), "--recipe", "bogus"])
        assert rc == 1
        assert "unknown recipe" in capsys.readouterr().err

    def test_recipe_filter_dry_run_subset(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        rc = main(
            ["--input", "a.mp4", "--outdir", str(tmp_path), "--dry-run", "--recipe", "baseline", "--recipe", "temporal"]
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "[baseline]" in out and "[temporal]" in out
        assert "[temporal_occl]" not in out

    def test_extra_arg_passed_through(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        rc = main(
            [
                "--input",
                "a.mp4",
                "--outdir",
                str(tmp_path),
                "--dry-run",
                "--extra-arg=--max-frames",
                "--extra-arg=30",
            ]
        )
        assert rc == 0
        assert "--max-frames 30" in capsys.readouterr().out
