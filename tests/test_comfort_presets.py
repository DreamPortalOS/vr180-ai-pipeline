"""Tests for I-3 comfort presets (issue #88) and their CLI wiring.

Covers the three acceptance criteria in the card:

  1. ``resolve_comfort`` is a pure function — three preset values, explicit
     override precedence, invalid name raises ``ValueError``.
  2. Both CLIs accept ``--comfort {safe,balanced,strong}``, default is
     ``balanced``, and explicit ``--max-disparity`` / ``--convergence`` win
     over the preset (the "override always wins" invariant).
  3. The stereo sweep defaults to the three comfort presets while still
     allowing a custom grid via ``--disparities`` / ``--convergences``.

No GPU, models, cv2, or ffmpeg are touched here.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

from pipeline.comfort_presets import DEFAULT_COMFORT, resolve_comfort

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
RUN_PIPELINE = SCRIPTS / "run_pipeline.py"
IMAGE_TO_VR180 = SCRIPTS / "image_to_vr180.py"
STEREO_SWEEP = SCRIPTS / "stereo_sweep.py"


def _load_run_pipeline() -> importlib.util.module_spec:
    spec = importlib.util.spec_from_file_location("run_pipeline_comfort", RUN_PIPELINE)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_stereo_sweep() -> importlib.util.module_spec:
    spec = importlib.util.spec_from_file_location("stereo_sweep_comfort", STEREO_SWEEP)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_image_to_vr180():
    """Load image_to_vr180 via sys.path (NOT spec_from_file_location): its
    ``JobArgs`` dataclass contains forward-reference annotations (``Path``)
    that dataclasses tries to resolve against the module's ``__module__``
    namespace — which is None for a spec_from_file_location module, raising
    ``AttributeError``.  The sys.path route mirrors tests/test_image_to_vr180.py.
    """
    scripts_dir = str(SCRIPTS)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    try:
        import image_to_vr180 as _i2v

        return _i2v
    finally:
        with contextlib.suppress(ValueError):
            sys.path.remove(scripts_dir)


# ---------------------------------------------------------------------------
# 1. resolve_comfort pure-function tests
# ---------------------------------------------------------------------------


class TestResolveComfort:
    """Preset values, explicit override precedence, invalid-name rejection."""

    def test_three_presets_have_documented_values(self) -> None:
        """Each tier resolves to its card-specified numbers (no override)."""
        assert resolve_comfort("safe") == {
            "max_disparity": 0.02,
            "convergence": 0.5,
            "temporal_smooth": True,
        }
        assert resolve_comfort("balanced") == {
            "max_disparity": 0.035,
            "convergence": 0.35,
            "temporal_smooth": True,
        }
        assert resolve_comfort("strong") == {
            "max_disparity": 0.06,
            "convergence": 0.2,
            "temporal_smooth": True,
        }

    def test_none_uses_default_preset(self) -> None:
        """``None`` name collapses to :data:`DEFAULT_COMFORT` (balanced)."""
        assert resolve_comfort(None) == resolve_comfort(DEFAULT_COMFORT)

    def test_explicit_max_disparity_overrides(self) -> None:
        """An explicit max_disparity always wins, even from the low tier."""
        assert resolve_comfort("safe", {"max_disparity": 0.06})["max_disparity"] == 0.06
        # The non-overridden keys keep their preset values.
        assert resolve_comfort("safe", {"max_disparity": 0.06})["convergence"] == 0.5

    def test_explicit_convergence_overrides(self) -> None:
        assert resolve_comfort("strong", {"convergence": 0.4})["convergence"] == 0.4
        assert resolve_comfort("strong", {"convergence": 0.4})["max_disparity"] == 0.06

    def test_explicit_temporal_smooth_overrides(self) -> None:
        """``--no-temporal`` maps to temporal_smooth=False and must override."""
        assert resolve_comfort("balanced", {"temporal_smooth": False})["temporal_smooth"] is False
        assert resolve_comfort("balanced", {"temporal_smooth": False})["max_disparity"] == 0.035

    def test_none_explicit_is_a_noop(self) -> None:
        """A ``None`` explicit dict behaves like no overrides."""
        assert resolve_comfort("balanced", None) == resolve_comfort("balanced")
        assert resolve_comfort("balanced", {}) == resolve_comfort("balanced")

    def test_returns_fresh_dict(self) -> None:
        """Mutating the result must not mutate the preset (defensive copy)."""
        a = resolve_comfort("safe")
        b = resolve_comfort("safe")
        a["max_disparity"] = -1.0
        assert resolve_comfort("safe")["max_disparity"] == 0.02
        assert b["max_disparity"] == 0.02

    def test_unknown_preset_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown comfort preset"):
            resolve_comfort("ultrasafe")
        with pytest.raises(ValueError, match="choose from"):
            resolve_comfort("balanced ")

    def test_unknown_explicit_keys_pass_through(self) -> None:
        """Unknown override keys pass through so the result stays drop-in
        for StereoRenderer(**resolved) style construction."""
        assert resolve_comfort("balanced", {"ipd": 0.07})["ipd"] == 0.07


# ---------------------------------------------------------------------------
# 2. CLI wiring — run_pipeline.py
# ---------------------------------------------------------------------------


class TestRunPipelineComfortCLI:
    """--comfort defaults to balanced; explicit flags override the preset."""

    def _args(self, argv: list[str]) -> argparse.Namespace:
        mod = _load_run_pipeline()
        return mod.parse_args(argv)

    def test_help_lists_comfort(self, capsys: pytest.CaptureFixture) -> None:
        mod = _load_run_pipeline()
        with pytest.raises(SystemExit) as exc:
            mod.parse_args(["--help"])
        assert exc.value.code == 0
        out = capsys.readouterr().out
        assert "--comfort" in out
        assert "--convergence" in out
        assert "--max-disparity" in out

    def test_default_comfort_is_balanced(self) -> None:
        """Default regression assertion: no --comfort → balanced."""
        args = self._args(["--input", "x.mp4"])
        assert args.comfort == "balanced"

    def test_max_disparity_default_is_none_sentinel(self) -> None:
        """Before preset resolution the explicit flag is None (unset)."""
        args = self._args(["--input", "x.mp4"])
        assert args.max_disparity is None
        assert args.convergence is None

    def test_comfort_choices_enforced(self) -> None:
        with pytest.raises(SystemExit):
            self._args(["--input", "x.mp4", "--comfort", "ultrasafe"])

    def test_preset_resolves_to_effective_values(self, capsys: pytest.CaptureFixture) -> None:
        """apply_quality_preset + _apply_comfort_preset produce concrete values.

        Regression: max_disparity/convergence must be floats (never None) and
        temporal_smooth a bool after resolution, matching the chosen tier.
        """
        mod = _load_run_pipeline()

        class FakeArgs:
            def __init__(self):
                self.quality = "preview"
                self.max_bitrate = 200.0
                self.output_width = None
                self.output_height = None
                self.streaming = False
                self.projection = "vr180"
                self.bitrate = None
                self.comfort = "safe"
                self.max_disparity = None
                self.convergence = None
                self.no_temporal = False

        args = FakeArgs()
        mod.apply_quality_preset(args)
        mod._apply_comfort_preset(args)

        assert args.max_disparity == pytest.approx(0.02)
        assert args.convergence == pytest.approx(0.5)
        assert args.temporal_smooth is True

    def test_explicit_max_disparity_wins_over_preset(self) -> None:
        mod = _load_run_pipeline()

        class FakeArgs:
            def __init__(self):
                self.quality = "preview"
                self.max_bitrate = 200.0
                self.output_width = None
                self.output_height = None
                self.streaming = False
                self.projection = "vr180"
                self.bitrate = None
                self.comfort = "safe"
                self.max_disparity = 0.06
                self.convergence = None
                self.no_temporal = False

        args = FakeArgs()
        mod.apply_quality_preset(args)
        mod._apply_comfort_preset(args)
        assert args.max_disparity == pytest.approx(0.06)  # explicit won
        assert args.convergence == pytest.approx(0.5)  # preset kept


# ---------------------------------------------------------------------------
# 2b. CLI wiring — image_to_vr180.py
# ---------------------------------------------------------------------------


class TestImageToVR180ComfortPassthrough:
    """--comfort flows from the I2V CLI into the converter's synthetic argv."""

    def _mod(self):
        return _load_image_to_vr180()

    def _parse(self, argv: list[str]):
        return self._mod().parse_args(argv)

    def test_parse_defaults_balanced(self) -> None:
        args = self._parse(["--image", "x.png"])
        assert args.comfort == "balanced"
        assert args.max_disparity is None
        assert args.convergence is None

    def test_explicit_comfort_accepted(self) -> None:
        for tier in ("safe", "balanced", "strong"):
            args = self._parse(["--image", "x.png", "--comfort", tier])
            assert args.comfort == tier

    def test_construction_populates_job_args(self) -> None:
        """main()-style JobArgs construction carries comfort through."""
        i2v = self._mod()

        job = i2v.JobArgs(image="x.png", comfort="safe", max_disparity=0.07, convergence=0.4)
        assert job.comfort == "safe"
        assert job.max_disparity == 0.07
        assert job.convergence == 0.4

    def test_convert_argv_carries_comfort_and_overrides(self, tmp_path: Path) -> None:
        """The synthetic argv run_convert_default builds must forward comfort
        and any explicit overrides, so the converter resolves the same values.

        run_convert_default delegates via ``import scripts.run_pipeline as rp``
        then ``rp.main()``. We install a fake ``scripts.run_pipeline`` module
        into sys.modules (and the ``scripts`` namespace parent) so the delegate
        records the synthetic argv instead of firing the real pipeline.
        """
        i2v = self._mod()

        calls: list[list[str]] = []

        def fake_rp_main() -> None:
            calls.append(sys.argv[:])

        fake_rp = types.ModuleType("scripts.run_pipeline")
        fake_rp.main = fake_rp_main  # type: ignore[attr-defined]
        fake_scripts = types.ModuleType("scripts")
        fake_scripts.__path__ = []  # mark as namespace package

        job = i2v.JobArgs(
            image="x.png",
            vr180_output=str(tmp_path / "out.mp4"),
            quality="preview",
            bitrate="50M",
            comfort="safe",
            max_disparity=0.06,  # explicit override
        )
        Path(job.vr180_output).parent.mkdir(parents=True, exist_ok=True)

        with patch.dict(sys.modules, {"scripts.run_pipeline": fake_rp, "scripts": fake_scripts}):
            i2v.run_convert_default(job, str(tmp_path / "in.mp4"))

        assert len(calls) == 1
        argv = calls[0]
        assert "--comfort" in argv
        idx = argv.index("--comfort")
        assert argv[idx + 1] == "safe"
        # Explicit override forwarded; convergence (None) must NOT appear.
        assert "--max-disparity" in argv
        assert argv[argv.index("--max-disparity") + 1] == "0.06"
        assert "--convergence" not in argv


# ---------------------------------------------------------------------------
# 3. Stereo sweep default grid
# ---------------------------------------------------------------------------


class TestStereoSweepComfortDefault:
    """Default sweep = three comfort presets; custom grid still works."""

    def test_build_comfort_grid_three_variants(self) -> None:
        mod = _load_stereo_sweep()
        grid = mod.build_comfort_grid()
        assert [v["convergence_name"] for v in grid] == ["safe", "balanced", "strong"]
        assert [v["max_disparity"] for v in grid] == pytest.approx([0.02, 0.035, 0.06])
        assert [v["convergence"] for v in grid] == pytest.approx([0.5, 0.35, 0.2])
        assert all(v["temporal_smooth"] is True for v in grid)

    def test_run_sweep_defaults_to_comfort_grid(self, tmp_path: Path) -> None:
        calls: list[dict] = []

        def fake(input_path: str, output_path: str, variant: dict, limit_seconds: float) -> str:
            calls.append(variant)
            Path(output_path).touch()
            return output_path

        mod = _load_stereo_sweep()
        mod.run_sweep(
            input_path="in.mp4",
            outdir=tmp_path,
            render_variant=fake,
        )
        assert len(calls) == 3
        assert {v["convergence_name"] for v in calls} == {"safe", "balanced", "strong"}

    def test_custom_grid_still_works(self, tmp_path: Path) -> None:
        calls: list[dict] = []

        def fake(input_path: str, output_path: str, variant: dict, limit_seconds: float) -> str:
            calls.append(variant)
            Path(output_path).touch()
            return output_path

        mod = _load_stereo_sweep()
        mod.run_sweep(
            input_path="in.mp4",
            outdir=tmp_path,
            disparities=[0.02, 0.04],
            convergences=["near", "far"],
            render_variant=fake,
        )
        assert len(calls) == 4  # 2 × 2
        assert {v["convergence_name"] for v in calls} == {"near", "far"}

    def test_parse_defaults_no_custom_axes(self) -> None:
        mod = _load_stereo_sweep()
        args = mod.parse_args(["--input", "a.mp4", "--outdir", "out"])
        assert args.disparities is None
        assert args.convergences is None

    def test_main_defaults_to_comfort_grid(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        """main() with no custom axes renders the three comfort variants."""
        calls: list[dict] = []

        def fake(input_path: str, output_path: str, variant: dict, limit_seconds: float) -> str:
            calls.append(variant)
            Path(output_path).touch()
            return output_path

        mod = _load_stereo_sweep()
        with patch.object(mod, "run_pipeline_variant", fake):
            rc = mod.main(["--input", "a.mp4", "--outdir", str(tmp_path)])
        assert rc == 0
        assert len(calls) == 3
        assert {v["convergence_name"] for v in calls} == {"safe", "balanced", "strong"}
        assert (tmp_path / "variants.md").exists()

    def test_main_custom_disparities_runs_legacy_grid(self, tmp_path: Path) -> None:
        calls: list[dict] = []

        def fake(input_path: str, output_path: str, variant: dict, limit_seconds: float) -> str:
            calls.append(variant)
            Path(output_path).touch()
            return output_path

        mod = _load_stereo_sweep()
        with patch.object(mod, "run_pipeline_variant", fake):
            rc = mod.main(
                [
                    "--input",
                    "a.mp4",
                    "--outdir",
                    str(tmp_path),
                    "--disparities",
                    "0.02,0.04",
                    "--convergences",
                    "near,far",
                ]
            )
        assert rc == 0
        assert len(calls) == 4

    def test_main_bad_disparities_still_rejected(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        mod = _load_stereo_sweep()
        rc = mod.main(["--input", "a.mp4", "--outdir", str(tmp_path), "--disparities", "abc"])
        assert rc == 1
        assert "--disparities" in capsys.readouterr().err
