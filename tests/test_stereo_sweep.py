"""Tests for the stereo parameter sweep tool (scripts/stereo_sweep.py).

Verifies:
- Grid is the Cartesian product of disparities × convergences
- Filenames encode the parameter combination
- variants.md manifest table completeness (one row per variant, empty score columns)
- Render calls are injectable — a fake runner means no real conversion/ffmpeg
- --help works and invalid input is rejected cleanly

No GPU, models, cv2, or ffmpeg are touched here.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from scripts.stereo_sweep import (
    CONVERGENCE_PRESETS,
    DEFAULT_CONVERGENCES,
    DEFAULT_DISPARITIES,
    SCORE_COLUMNS,
    build_grid,
    main,
    parse_args,
    render_variants_md,
    run_sweep,
    variant_filename,
)


class TestBuildGrid:
    """Grid Cartesian product tests."""

    def test_default_grid_size(self) -> None:
        """Default grid = 4 disparities × 3 convergences = 12 variants."""
        grid = build_grid()
        assert len(grid) == len(DEFAULT_DISPARITIES) * len(DEFAULT_CONVERGENCES) == 12

    def test_grid_is_cartesian_product(self) -> None:
        """Every (disparity, convergence) pair appears exactly once."""
        disps = [0.02, 0.05]
        convs = ["near", "far"]
        grid = build_grid(disparities=disps, convergences=convs)
        combos = [(v["max_disparity"], v["convergence_name"]) for v in grid]
        assert combos == [(0.02, "near"), (0.02, "far"), (0.05, "near"), (0.05, "far")]

    def test_convergence_preset_values(self) -> None:
        """Named presets map to their StereoRenderer convergence floats."""
        grid = build_grid(disparities=[0.04], convergences=["near", "mid", "far"])
        by_name = {v["convergence_name"]: v["convergence"] for v in grid}
        assert by_name == CONVERGENCE_PRESETS
        assert by_name["near"] < by_name["mid"] < by_name["far"]

    def test_single_axis_grid(self) -> None:
        """A single convergence degrades the grid to disparity-only."""
        grid = build_grid(disparities=[0.02, 0.04, 0.06], convergences=["mid"])
        assert len(grid) == 3
        assert all(v["convergence_name"] == "mid" for v in grid)

    def test_unknown_convergence_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown convergence"):
            build_grid(convergences=["near", "bogus"])

    def test_non_positive_disparity_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            build_grid(disparities=[0.02, 0.0])
        with pytest.raises(ValueError, match="positive"):
            build_grid(disparities=[-0.1])

    def test_empty_axes_raise(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            build_grid(disparities=[])
        with pytest.raises(ValueError, match="at least one"):
            build_grid(convergences=[])


class TestVariantFilename:
    """Filename parameter-encoding tests."""

    def test_filename_encodes_params(self) -> None:
        v = {"max_disparity": 0.04, "convergence_name": "mid"}
        assert variant_filename(v) == "sweep_d0.04_cmid.mp4"

    def test_filename_roundtrip_unique_per_variant(self) -> None:
        """Every grid variant gets a unique filename (no collisions)."""
        grid = build_grid()
        names = [variant_filename(v) for v in grid]
        assert len(names) == len(set(names))

    def test_filename_contains_both_params(self) -> None:
        grid = build_grid(disparities=[0.08], convergences=["far"])
        name = variant_filename(grid[0])
        assert "d0.08" in name
        assert "cfar" in name
        assert name.endswith(".mp4")


class TestVariantsMd:
    """Manifest table completeness tests."""

    def _variants_with_filenames(self) -> list[dict]:
        grid = build_grid(disparities=[0.02, 0.08], convergences=["near", "far"])
        for v in grid:
            v["filename"] = variant_filename(v)
        return grid

    def test_manifest_has_one_row_per_variant(self) -> None:
        variants = self._variants_with_filenames()
        md = render_variants_md(variants, source="in.mp4", limit_seconds=5.0)
        for v in variants:
            assert v["filename"] in md
        # Header separator + one row per variant
        rows = [line for line in md.splitlines() if line.startswith("| sweep_")]
        assert len(rows) == len(variants)

    def test_manifest_score_columns_present_and_empty(self) -> None:
        variants = self._variants_with_filenames()
        md = render_variants_md(variants, source="in.mp4", limit_seconds=5.0)
        header = next(line for line in md.splitlines() if line.startswith("| 文件"))
        for col in SCORE_COLUMNS:
            assert col in header
        # Score cells are empty: each data row ends with the empty score
        # columns followed by the empty 备注 column.
        rows = [line for line in md.splitlines() if line.startswith("| sweep_")]
        for row in rows:
            cells = [c.strip() for c in row.strip("|").split("|")]
            # cells: 文件, max_disparity, convergence, 4 score cols, 备注
            assert len(cells) == 3 + len(SCORE_COLUMNS) + 1
            assert all(cell == "" for cell in cells[3:])

    def test_manifest_records_source_and_limit(self) -> None:
        variants = self._variants_with_filenames()
        md = render_variants_md(variants, source="clip.mp4", limit_seconds=7.5)
        assert "clip.mp4" in md
        assert "7.5" in md

    def test_manifest_param_values_in_rows(self) -> None:
        variants = self._variants_with_filenames()
        md = render_variants_md(variants, source="in.mp4", limit_seconds=5.0)
        assert "0.02" in md and "0.08" in md
        assert "near" in md and "far" in md


class TestRunSweepWithFakeRunner:
    """Sweep orchestration with an injected fake runner (no real conversion)."""

    def test_fake_runner_called_once_per_variant(self, tmp_path: Path) -> None:
        calls: list[tuple[str, str, dict, float]] = []

        def fake_runner(input_path: str, output_path: str, variant: dict, limit_seconds: float) -> str:
            calls.append((input_path, output_path, variant, limit_seconds))
            Path(output_path).touch()  # pretend we rendered it
            return output_path

        variants = run_sweep(
            input_path="fake_input.mp4",
            outdir=tmp_path,
            limit_seconds=3.0,
            disparities=[0.02, 0.04],
            convergences=["near", "mid", "far"],
            render_variant=fake_runner,
        )
        assert len(calls) == 6
        assert len(variants) == 6
        # limit_seconds propagated to every render call
        assert all(call[3] == 3.0 for call in calls)
        # output paths land inside outdir with encoded filenames
        for _, output_path, variant, _ in calls:
            assert Path(output_path).parent == tmp_path
            assert Path(output_path).name == variant_filename(variant)

    def test_sweep_writes_variants_md(self, tmp_path: Path) -> None:
        def fake_runner(input_path: str, output_path: str, variant: dict, limit_seconds: float) -> str:
            Path(output_path).touch()
            return output_path

        run_sweep(
            input_path="fake_input.mp4",
            outdir=tmp_path,
            limit_seconds=5.0,
            disparities=[0.02, 0.06],
            convergences=["mid"],
            render_variant=fake_runner,
        )
        manifest = tmp_path / "variants.md"
        assert manifest.exists()
        text = manifest.read_text(encoding="utf-8")
        assert "sweep_d0.02_cmid.mp4" in text
        assert "sweep_d0.06_cmid.mp4" in text
        for col in SCORE_COLUMNS:
            assert col in text

    def test_sweep_invalid_grid_raises_before_rendering(self, tmp_path: Path) -> None:
        def fail_runner(*args, **kwargs):  # pragma: no cover - must not be called
            raise AssertionError("runner must not be called for invalid grids")

        with pytest.raises(ValueError, match="unknown convergence"):
            run_sweep(
                input_path="fake.mp4",
                outdir=tmp_path,
                convergences=["bogus"],
                render_variant=fail_runner,
            )


class TestCLI:
    """CLI parsing and entry-point behavior."""

    def test_help_exits_zero(self, capsys: pytest.CaptureFixture) -> None:
        with pytest.raises(SystemExit) as exc_info:
            parse_args(["--help"])
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "--limit-seconds" in out
        assert "--disparities" in out
        assert "--convergences" in out

    def test_parse_defaults(self) -> None:
        args = parse_args(["--input", "a.mp4", "--outdir", "out"])
        assert args.limit_seconds == 5.0
        assert args.input == "a.mp4"
        assert args.outdir == "out"

    def test_main_rejects_bad_disparities(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        rc = main(["--input", "a.mp4", "--outdir", str(tmp_path), "--disparities", "abc"])
        assert rc == 1
        assert "--disparities" in capsys.readouterr().err

    def test_main_rejects_unknown_convergence(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        rc = main(["--input", "a.mp4", "--outdir", str(tmp_path), "--convergences", "bogus"])
        assert rc == 1
        assert "unknown convergence" in capsys.readouterr().err
