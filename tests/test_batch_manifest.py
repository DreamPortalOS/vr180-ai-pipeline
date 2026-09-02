"""Tests for the manifest-driven batch path of scripts/batch_runner.py.

C-4 / issue #202: ``--manifest PATH`` turns a scene manifest JSON into a batch
of run_pipeline.py runs. These tests cover the acceptance criteria:

  - ``defaults`` vs scene-field override priority (scene wins)
  - output filename composed by :func:`pipeline.naming.compose_scene_name`
  - a scene failing mid-batch does NOT abort the rest; summary marks it;
    exit code is non-zero
  - all-success → exit 0
  - ``--dry-run`` invokes no runner at all
  - a missing / mistyped field raises an error naming the scene + field

The real run_pipeline.py / ffmpeg / models are never touched — the per-scene
runner is injected (a fake callable), exactly as ``run_one_job``'s
``run_pipeline=`` is patched in test_batch_runner.py. ``not slow``.
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
from pipeline.naming import SceneAssetSpec, compose_scene_name  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_manifest(tmp_path: Path, manifest: dict) -> Path:
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(manifest), encoding="utf-8")
    return p


def _two_scene_manifest(output_dir: str = ".") -> dict:
    return {
        "defaults": {"comfort": "safe", "max_frames": 60},
        "scenes": [
            {
                "scene_id": "s03",
                "name": "santorini",
                "inputs": ["video/seg01.mp4", "video/seg02.mp4"],
                "concat_crossfade": 0.3,
                "comfort": "balanced",
            },
            {
                "scene_id": "s07",
                "name": "kyoto",
                "inputs": ["video/seg10.mp4"],
            },
        ],
    }


def _ok_runner(captured: list[list[str]]):
    """A fake scene runner that records argv and reports success."""

    def fake(argv: list[str]) -> str:
        captured.append(argv)
        # Mirror _default_scene_runner's --output extraction.
        for i, tok in enumerate(argv):
            if tok == "--output" and i + 1 < len(argv):
                return argv[i + 1]
        return ""

    return fake


# ---------------------------------------------------------------------------
# load_manifest
# ---------------------------------------------------------------------------


class TestLoadManifest:
    def test_load_manifest_reads_object(self, tmp_path):
        p = _write_manifest(tmp_path, _two_scene_manifest())
        m = br.load_manifest(p)
        assert "scenes" in m and len(m["scenes"]) == 2
        assert m["defaults"]["comfort"] == "safe"

    def test_load_manifest_missing_file(self, tmp_path):
        with pytest.raises(RuntimeError, match="Manifest file not found"):
            br.load_manifest(tmp_path / "nope.json")

    def test_load_manifest_bad_json(self, tmp_path):
        p = tmp_path / "m.json"
        p.write_text("{not json", encoding="utf-8")
        with pytest.raises(RuntimeError, match="not valid JSON"):
            br.load_manifest(p)

    def test_load_manifest_not_object(self, tmp_path):
        p = _write_manifest(tmp_path, [1, 2, 3])
        with pytest.raises(RuntimeError, match="must be a JSON object"):
            br.load_manifest(p)

    def test_load_manifest_no_scenes_key(self, tmp_path):
        p = _write_manifest(tmp_path, {"defaults": {}})
        with pytest.raises(RuntimeError, match="'scenes'"):
            br.load_manifest(p)

    def test_load_manifest_empty_scenes(self, tmp_path):
        p = _write_manifest(tmp_path, {"scenes": []})
        with pytest.raises(RuntimeError, match="empty"):
            br.load_manifest(p)

    def test_load_manifest_defaults_optional(self, tmp_path):
        p = _write_manifest(tmp_path, {"scenes": [{"scene_id": "s01", "inputs": ["a.mp4"]}]})
        m = br.load_manifest(p)
        assert m["defaults"] == {}


# ---------------------------------------------------------------------------
# resolve_scene: defaults vs override priority
# ---------------------------------------------------------------------------


class TestResolveScenePriority:
    def test_scene_overrides_defaults(self):
        manifest = _two_scene_manifest()
        specs = br.resolve_all_scenes(manifest)
        # s03 overrides comfort=balanced (default safe).
        s03 = next(s for s in specs if s.scene_id == "s03")
        assert s03.comfort == "balanced"
        assert s03.concat_crossfade == 0.3

    def test_defaults_fill_missing_fields(self):
        manifest = _two_scene_manifest()
        specs = br.resolve_all_scenes(manifest)
        # s07 omits comfort/max_frames/concat_crossfade → inherits defaults.
        s07 = next(s for s in specs if s.scene_id == "s07")
        assert s07.comfort == "safe"
        assert s07.max_frames == 60
        assert s07.concat_crossfade == 0.0

    def test_scene_id_not_inherited(self):
        """scene_id is per-scene-only; inheriting it would make every scene
        identical (a manifest typo), so it is NOT inheritable."""
        manifest = {
            "defaults": {"scene_id": "should-not-leak"},
            "scenes": [{"inputs": ["a.mp4"]}],
        }
        # Missing scene_id on the scene itself is an error even though defaults
        # has one — because scene_id is not in _INHERITABLE_FIELDS.
        with pytest.raises(RuntimeError, match="missing required field 'scene_id'"):
            br.resolve_all_scenes(manifest)

    def test_inputs_not_inherited(self):
        manifest = {
            "defaults": {"inputs": ["shared.mp4"]},
            "scenes": [{"scene_id": "s01"}],
        }
        with pytest.raises(RuntimeError, match="missing required field 'inputs'"):
            br.resolve_all_scenes(manifest)

    def test_name_falls_back_to_scene_id(self):
        manifest = {
            "defaults": {},
            "scenes": [{"scene_id": "s01", "inputs": ["a.mp4"]}],
        }
        specs = br.resolve_all_scenes(manifest)
        assert specs[0].name == "s01"

    def test_cli_output_dir_overrides_everything(self):
        manifest = _two_scene_manifest()
        specs = br.resolve_all_scenes(manifest, cli_output_dir="/tmp/out")
        for s in specs:
            assert s.output_dir == "/tmp/out"


# ---------------------------------------------------------------------------
# Output naming via compose_scene_name
# ---------------------------------------------------------------------------


class TestSceneOutputNaming:
    def test_output_path_uses_compose_scene_name(self):
        spec = br.SceneSpec(
            scene_id="s03",
            name="santorini",
            inputs=["video/seg01.mp4", "video/seg02.mp4"],
            output_dir="/out",
        )
        path = br._scene_output_path(spec)
        # The filename must equal what compose_scene_name produces for the
        # same identity — proving the runner routes through that function
        # rather than hand-building a string.
        expected_name = compose_scene_name(
            SceneAssetSpec(
                scene_id="s03",
                scene_name="santorini",
                segment_index=1,
                route="vr180",
                preset="standalone",
            )
        )
        assert path.name == expected_name
        # Path-OS-agnostic: the output dir is a prefix of the resolved path
        # (use os.sep so the assertion holds on both POSIX and Windows).
        assert str(path).endswith(os.path.join("out", expected_name)) or "out" in path.parts

    def test_scene_argv_forwards_scene_named_output(self):
        spec = br.SceneSpec(
            scene_id="s03",
            name="santorini",
            inputs=["video/seg01.mp4", "video/seg02.mp4"],
            concat_crossfade=0.3,
            comfort="balanced",
            output_dir=".",
        )
        argv = br.scene_argv(spec, script="scripts/run_pipeline.py")
        # --output is the scene-named path, not an input-stem path.
        assert "--output" in argv
        out_idx = argv.index("--output")
        out_val = argv[out_idx + 1]
        assert "s03" in out_val
        assert "santorini" in out_val
        assert "seg01" in out_val
        assert "vr180" in out_val
        assert out_val.endswith(".mp4")
        # It is NOT the input stem.
        assert "seg01.mp4" not in out_val  # output is a composed name, not an input

    def test_scene_argv_inputs_and_flags(self):
        spec = br.SceneSpec(
            scene_id="s03",
            inputs=["a.mp4", "b.mp4"],
            concat_crossfade=0.5,
            comfort="safe",
            max_frames=30,
        )
        argv = br.scene_argv(spec)
        assert "--inputs" in argv
        i = argv.index("--inputs")
        # inputs follow --inputs until the next --flag
        assert argv[i + 1] == "a.mp4"
        assert argv[i + 2] == "b.mp4"
        assert argv[i + 3] == "--concat-crossfade"
        assert argv[i + 4] == "0.5"
        assert "--comfort" in argv and "safe" in argv
        assert "--max-frames" in argv and "30" in argv

    def test_scene_argv_omits_default_crossfade(self):
        """A 0.0 crossfade is the run_pipeline.py default — don't emit noise."""
        spec = br.SceneSpec(scene_id="s01", inputs=["a.mp4"])
        argv = br.scene_argv(spec)
        assert "--concat-crossfade" not in argv

    def test_two_scenes_distinct_outputs(self):
        specs = br.resolve_all_scenes(_two_scene_manifest())
        outs = {str(br._scene_output_path(s)) for s in specs}
        assert len(outs) == 2  # distinct per scene_id


# ---------------------------------------------------------------------------
# run_one_scene: fault tolerance + result shape
# ---------------------------------------------------------------------------


class TestRunOneScene:
    def test_success_records_output(self):
        spec = br.SceneSpec(scene_id="s01", name="n", inputs=["a.mp4"])
        captured: list[list[str]] = []
        r = br.run_one_scene(spec, runner=_ok_runner(captured))
        assert r.status == "success"
        assert r.scene_id == "s01"
        assert r.output_path  # non-empty
        assert r.error == ""
        assert captured  # runner was actually called

    def test_failure_does_not_raise(self):
        spec = br.SceneSpec(scene_id="s01", inputs=["a.mp4"])

        def boom(argv):
            raise br.SceneRunError("exit 1\nstderr: boom")

        r = br.run_one_scene(spec, runner=boom)
        assert r.status == "failed"
        assert "boom" in r.error
        assert r.scene_id == "s01"


# ---------------------------------------------------------------------------
# Full batch via run_manifest + main()
# ---------------------------------------------------------------------------


class TestBatchFaultTolerance:
    def test_mid_failure_continues_rest(self, tmp_path, capsys):
        manifest = {
            "defaults": {},
            "scenes": [
                {"scene_id": "s01", "inputs": ["a.mp4"]},
                {"scene_id": "boom", "inputs": ["b.mp4"]},
                {"scene_id": "s03", "inputs": ["c.mp4"]},
            ],
        }
        specs = br.resolve_all_scenes(manifest)

        called: list[str] = []

        def fake(argv: list[str]) -> str:
            # Identify the scene by the --output value (encodes scene_id).
            out = argv[argv.index("--output") + 1]
            called.append(out)
            if "boom" in out:
                raise br.SceneRunError("boom from fake runner")
            return out

        results = br.run_manifest(specs, runner=fake)
        statuses = {r.scene_id: r.status for r in results}
        assert statuses == {"s01": "success", "boom": "failed", "s03": "success"}
        # All three were attempted (boom did not abort s03).
        assert len(called) == 3
        boom = next(r for r in results if r.scene_id == "boom")
        assert "boom from fake runner" in boom.error

    def test_main_partial_failure_nonzero_exit(self, tmp_path, capsys):
        manifest = {
            "defaults": {},
            "scenes": [
                {"scene_id": "s01", "inputs": ["a.mp4"]},
                {"scene_id": "boom", "inputs": ["b.mp4"]},
            ],
        }
        p = _write_manifest(tmp_path, manifest)

        def fake(argv: list[str]) -> str:
            out = argv[argv.index("--output") + 1]
            if "boom" in out:
                raise br.SceneRunError("boom")
            return out

        with patch.object(br, "_default_scene_runner", side_effect=fake):
            rc = br.main(["--manifest", str(p)])
        assert rc == br.EXIT_PARTIAL  # non-zero
        out = capsys.readouterr().out
        assert "boom" in out and "failed" in out
        assert "succeeded: 1" in out and "failed: 1" in out

    def test_main_all_success_exit_zero(self, tmp_path, capsys):
        manifest = {
            "defaults": {"comfort": "safe"},
            "scenes": [
                {"scene_id": "s01", "inputs": ["a.mp4"]},
                {"scene_id": "s02", "inputs": ["b.mp4"], "comfort": "balanced"},
            ],
        }
        p = _write_manifest(tmp_path, manifest)

        def fake(argv: list[str]) -> str:
            return argv[argv.index("--output") + 1]

        with patch.object(br, "_default_scene_runner", side_effect=fake):
            rc = br.main(["--manifest", str(p)])
        assert rc == br.EXIT_OK
        out = capsys.readouterr().out
        assert "succeeded: 2" in out and "failed: 0" in out


# ---------------------------------------------------------------------------
# --dry-run: no runner invocation
# ---------------------------------------------------------------------------


class TestDryRun:
    def test_dry_run_prints_scenes_without_running(self, tmp_path, capsys):
        manifest = _two_scene_manifest()
        p = _write_manifest(tmp_path, manifest)

        called = {"n": 0}

        def fake(argv):
            called["n"] += 1
            return argv[argv.index("--output") + 1]

        with patch.object(br, "_default_scene_runner", side_effect=fake):
            rc = br.main(["--manifest", str(p), "--dry-run"])

        assert rc == br.EXIT_OK
        out = capsys.readouterr().out
        assert "Dry run" in out
        assert "s03" in out and "s07" in out
        assert "santorini" in out and "kyoto" in out
        # The effective argv is shown (inputs + flags).
        assert "--inputs" in out
        assert "--concat-crossfade" in out  # s03 has 0.3
        # Nothing actually ran.
        assert called["n"] == 0

    def test_dry_run_does_not_invoke_default_runner(self, tmp_path, capsys):
        """AC: --dry-run must not trigger any real call. Patch the real
        subprocess-backed runner and assert it is never called."""
        p = _write_manifest(tmp_path, _two_scene_manifest())
        with patch.object(br, "_default_scene_runner") as mock_runner:
            rc = br.main(["--manifest", str(p), "--dry-run"])
        assert rc == br.EXIT_OK
        mock_runner.assert_not_called()


# ---------------------------------------------------------------------------
# Field validation: errors name the scene + field
# ---------------------------------------------------------------------------


class TestFieldValidation:
    def test_missing_inputs_names_scene_and_field(self, tmp_path):
        manifest = {
            "defaults": {},
            "scenes": [{"scene_id": "s05", "name": "bad"}],
        }
        p = _write_manifest(tmp_path, manifest)
        with patch.object(br, "_default_scene_runner"):
            rc = br.main(["--manifest", str(p)])
        assert rc == br.EXIT_FAILED  # bad manifest = fatal

    def test_missing_inputs_message_names_scene(self):
        with pytest.raises(RuntimeError) as exc:
            br.resolve_all_scenes({"defaults": {}, "scenes": [{"scene_id": "s05"}]})
        msg = str(exc.value)
        assert "s05" in msg  # scene identified
        assert "inputs" in msg  # field identified

    def test_missing_scene_id_names_scene_by_index(self):
        with pytest.raises(RuntimeError, match=r"#1.*scene_id"):
            br.resolve_all_scenes({"defaults": {}, "scenes": [{"inputs": ["a.mp4"]}]})

    def test_wrong_type_inputs_names_scene_and_field(self):
        with pytest.raises(RuntimeError) as exc:
            br.resolve_all_scenes({"defaults": {}, "scenes": [{"scene_id": "s05", "inputs": "a.mp4"}]})
        msg = str(exc.value)
        assert "s05" in msg
        assert "inputs" in msg
        assert "list" in msg

    def test_wrong_type_crossfade_names_field(self):
        with pytest.raises(RuntimeError) as exc:
            br.resolve_all_scenes(
                {
                    "defaults": {},
                    "scenes": [{"scene_id": "s05", "inputs": ["a.mp4"], "concat_crossfade": "fast"}],
                }
            )
        msg = str(exc.value)
        assert "s05" in msg
        assert "concat_crossfade" in msg

    def test_negative_crossfade_rejected(self):
        with pytest.raises(RuntimeError, match=r"concat_crossfade.*>= 0"):
            br.resolve_all_scenes(
                {
                    "defaults": {},
                    "scenes": [{"scene_id": "s05", "inputs": ["a.mp4"], "concat_crossfade": -1.0}],
                }
            )

    def test_empty_inputs_rejected(self):
        with pytest.raises(RuntimeError, match="non-empty"):
            br.resolve_all_scenes({"defaults": {}, "scenes": [{"scene_id": "s05", "inputs": []}]})

    def test_bool_for_int_field_rejected(self):
        """JSON ``true`` must not silently coerce to max_frames=1."""
        with pytest.raises(RuntimeError, match=r"max_frames.*bool"):
            br.resolve_all_scenes(
                {
                    "defaults": {},
                    "scenes": [{"scene_id": "s05", "inputs": ["a.mp4"], "max_frames": True}],
                }
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCLI:
    def test_jobs_and_manifest_mutually_exclusive(self, capsys):
        with pytest.raises(SystemExit) as exc:
            br.parse_args(["--jobs", "a.json", "--manifest", "b.json"])
        assert exc.value.code != 0

    def test_one_of_jobs_manifest_required(self, capsys):
        with pytest.raises(SystemExit) as exc:
            br.parse_args(["--dry-run"])
        assert exc.value.code != 0

    def test_manifest_flag_present_in_help(self, capsys):
        with pytest.raises(SystemExit) as exc:
            br.parse_args(["--help"])
        assert exc.value.code == 0
        out = capsys.readouterr().out
        assert "--manifest" in out
        assert "--output-dir" in out

    def test_manifest_dry_run_help_lists_both_modes(self, capsys):
        with pytest.raises(SystemExit):
            br.parse_args(["--help"])
        out = capsys.readouterr().out
        assert "--jobs" in out and "--manifest" in out


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


class TestManifestSummary:
    def test_summary_columns_and_totals(self):
        results = [
            br.SceneResult(scene_id="s01", status="success", output_path="/o/s01.mp4", duration_s=1.2),
            br.SceneResult(scene_id="boom", status="failed", error="SceneRunError: x", duration_s=0.4),
        ]
        text = br.format_manifest_summary(results)
        assert "scene_id" in text and "status" in text and "output_path" in text
        assert "s01" in text and "boom" in text
        assert "/o/s01.mp4" in text
        assert "succeeded: 1" in text and "failed: 1" in text

    def test_summary_empty(self):
        assert "No scenes" in br.format_manifest_summary([])
