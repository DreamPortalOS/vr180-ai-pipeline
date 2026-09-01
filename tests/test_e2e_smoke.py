"""Tests for the e2e smoke script (scripts/e2e_smoke.py, K-5 #149).

Covers each assertion's pass/fail logic and the exit-code contract, with the
pipeline subprocess, ffprobe, byte-scan and sidecar all mocked/injected — the
test-suite never runs a real conversion or shells out to ffprobe.

Assertion families (one per smoke check):
- pipeline exit code
- output exists + ffprobe + resolution vs quality tier
- sv3d/st3d BYTE-SCAN (raw bytes, not the QA report)
- audio stream present when --copy-audio-from was requested
- backend log assertion (requested vs effective depth/stereo backend names)
- sidecar JSON presence + D-3 required immersive fields
- run_smoke orchestration + exit-code aggregation (runner injected)
"""

from __future__ import annotations

import json
import struct
import subprocess
from pathlib import Path

import pytest
from scripts.e2e_smoke import (
    PROFILES,
    _scan_boxes,
    build_pipeline_command,
    check_audio_stream,
    check_backend_log,
    check_depth_meta,
    check_exit_code,
    check_metadata_bytes,
    check_output_probe,
    check_sidecar,
    main,
    parse_args,
    run_smoke,
)

from pipeline.spherical_injector import _box4, _build_st3d, _build_sv3d

# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------


def _synthetic_vr180_bytes(stereo_mode: str = "sbs") -> bytes:
    """A minimal mp4-like byte blob carrying real sv3d + st3d boxes."""
    hvc1 = _box4(b"hvc1", b"\x00" * 78 + _build_sv3d(3840, 1920, stereo_mode) + _build_st3d(stereo_mode))
    stsd = _box4(b"stsd", b"\x00\x00\x00\x00" + struct.pack(">I", 1) + hvc1)
    stbl = _box4(b"stbl", stsd)
    minf = _box4(b"minf", stbl)
    mdia = _box4(b"mdia", minf)
    trak = _box4(b"trak", mdia)
    moov = _box4(b"moov", trak)
    ftyp = _box4(b"ftyp", b"isom\x00\x00\x02\x00isomiso2")
    return ftyp + moov


def _probe(width: int, height: int, audio: bool = False) -> dict:
    """A fake ffprobe JSON payload."""
    streams = [{"codec_type": "video", "width": width, "height": height, "codec_name": "h264"}]
    if audio:
        streams.append({"codec_type": "audio", "codec_name": "aac", "bit_rate": "128000"})
    return {"streams": streams, "format": {"duration": "1.0"}}


def _fake_proc(returncode: int = 0, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=["x"], returncode=returncode, stdout=stdout, stderr=stderr)


# ---------------------------------------------------------------------------
# Command assembly
# ---------------------------------------------------------------------------


class TestBuildCommand:
    def test_fast_profile_args(self) -> None:
        cmd = build_pipeline_command("in.mp4", "out.mp4", "fast")
        assert "--quality" in cmd and "preview" in cmd
        assert "--max-frames" in cmd and "8" in cmd
        # fast = pure Depth-Anything: no heavy-model flags
        assert "--depth-model" not in cmd and "--stereo-model" not in cmd

    def test_full_profile_args(self) -> None:
        cmd = build_pipeline_command("in.mp4", "out.mp4", "full")
        joined = " ".join(cmd)
        assert "--depth-model depthcrafter" in joined
        assert "--stereo-model stereocrafter" in joined
        assert "--comfort safe" in joined
        assert "--quality high" in joined
        # K-7 (#160): heavy-model acceptance args — lead's whole ritual in one cmd.
        assert "--src-hfov 150" in joined
        assert "--max-frames 60" in joined

    def test_copy_audio_from_appended(self) -> None:
        cmd = build_pipeline_command("in.mp4", "out.mp4", "full", copy_audio_from="src.mp4")
        assert "--copy-audio-from" in cmd and "src.mp4" in cmd

    def test_ci_profile_args(self) -> None:
        cmd = build_pipeline_command("in.mp4", "out.mp4", "ci")
        joined = " ".join(cmd)
        assert "--quality preview" in joined
        assert "--max-frames 4" in joined
        # Tiny 256²/eye override keeps CPU projection + encode at seconds.
        assert "--output-width 256" in joined and "--output-height 256" in joined
        # --force-sbs skips depth/stereo entirely → no model is ever downloaded.
        assert "--force-sbs" in cmd
        # Deterministic OpenCV remap — no dependence on ffmpeg shipping v360.
        assert "--no-ffmpeg-v360" in cmd
        # ci = no model flags at all (nothing to download).
        assert "--depth-model" not in cmd and "--stereo-model" not in cmd

    def test_unknown_profile_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown profile"):
            build_pipeline_command("in.mp4", "out.mp4", "turbo")

    def test_list_form_no_shell(self) -> None:
        cmd = build_pipeline_command("in.mp4", "out.mp4", "fast")
        assert isinstance(cmd, list) and all(isinstance(t, str) for t in cmd)


# ---------------------------------------------------------------------------
# 1. exit code
# ---------------------------------------------------------------------------


class TestExitCode:
    def test_zero_passes(self) -> None:
        c = check_exit_code(0)
        assert c.ok and "exit=0" in c.measured

    def test_nonzero_fails_with_measured(self) -> None:
        c = check_exit_code(3)
        assert not c.ok and "exit=3" in c.measured and c.hint


# ---------------------------------------------------------------------------
# 2. output probe + resolution
# ---------------------------------------------------------------------------


class TestOutputProbe:
    def test_missing_file_fails(self, tmp_path: Path) -> None:
        c = check_output_probe(str(tmp_path / "nope.mp4"), "high")
        assert not c.ok and "missing" in c.measured and c.hint

    def test_resolution_matches_quality(self, tmp_path: Path) -> None:
        f = tmp_path / "o.mp4"
        f.touch()
        # high = 3840²/eye → 7680×3840 SBS
        c = check_output_probe(str(f), "high", probe=_probe(7680, 3840))
        assert c.ok and "7680×3840" in c.measured

    def test_resolution_mismatch_reports_measured(self, tmp_path: Path) -> None:
        f = tmp_path / "o.mp4"
        f.touch()
        c = check_output_probe(str(f), "high", probe=_probe(3840, 1920))
        assert not c.ok
        assert "3840×1920" in c.measured and "7680×3840" in c.measured and c.hint

    def test_no_video_stream_fails(self, tmp_path: Path) -> None:
        f = tmp_path / "o.mp4"
        f.touch()
        c = check_output_probe(str(f), "high", probe={"streams": [], "format": {}})
        assert not c.ok and "0×0" in c.measured

    def test_eye_override_used(self, tmp_path: Path) -> None:
        f = tmp_path / "o.mp4"
        f.touch()
        # ci profile: preview tier + explicit 256²/eye → 512×256 SBS.
        c = check_output_probe(str(f), "preview", probe=_probe(512, 256), eye=256)
        assert c.ok and "512×256" in c.measured

    def test_eye_override_mismatch_reports_override(self, tmp_path: Path) -> None:
        f = tmp_path / "o.mp4"
        f.touch()
        c = check_output_probe(str(f), "preview", probe=_probe(3840, 1920), eye=256)
        assert not c.ok and "512×256" in c.measured and c.hint


# ---------------------------------------------------------------------------
# 3. sv3d/st3d byte-scan
# ---------------------------------------------------------------------------


class TestMetadataBytes:
    def test_real_boxes_pass(self, tmp_path: Path) -> None:
        f = tmp_path / "o.mp4"
        f.write_bytes(_synthetic_vr180_bytes("sbs"))
        c = check_metadata_bytes(str(f))
        assert c.ok and "sv3d" in c.measured and "st3d" in c.measured

    def test_missing_boxes_fail_with_measured(self, tmp_path: Path) -> None:
        f = tmp_path / "o.mp4"
        f.write_bytes(_box4(b"ftyp", b"isom\x00\x00\x02\x00isomiso2") + _box4(b"moov", b""))
        c = check_metadata_bytes(str(f))
        assert not c.ok and "none" in c.measured and c.hint

    def test_wrong_stereo_mode_fails(self, tmp_path: Path) -> None:
        f = tmp_path / "o.mp4"
        f.write_bytes(_synthetic_vr180_bytes("mono"))
        c = check_metadata_bytes(str(f))
        assert not c.ok and "st3d_mode=" in c.measured

    def test_injected_boxes_used(self) -> None:
        boxes = {"sv3d": {"offset": 0, "stereo_mode": None}, "st3d": {"offset": 1, "stereo_mode": 2}}
        c = check_metadata_bytes("unused.mp4", boxes=boxes)
        assert c.ok

    def test_scan_boxes_finds_both(self, tmp_path: Path) -> None:
        f = tmp_path / "o.mp4"
        f.write_bytes(_synthetic_vr180_bytes("sbs"))
        found = _scan_boxes(str(f))
        assert "sv3d" in found and "st3d" in found
        assert found["st3d"]["stereo_mode"] == 2


# ---------------------------------------------------------------------------
# 4. audio stream
# ---------------------------------------------------------------------------


class TestAudioStream:
    def test_not_requested_is_na_pass(self) -> None:
        c = check_audio_stream("o.mp4", None)
        assert c.ok and "N/A" in c.measured

    def test_audio_present_passes(self, tmp_path: Path) -> None:
        f = tmp_path / "o.mp4"
        f.touch()
        c = check_audio_stream(str(f), "src.mp4", probe=_probe(7680, 3840, audio=True))
        assert c.ok and "aac" in c.measured

    def test_audio_missing_fails_with_measured(self, tmp_path: Path) -> None:
        f = tmp_path / "o.mp4"
        f.touch()
        c = check_audio_stream(str(f), "src.mp4", probe=_probe(7680, 3840, audio=False))
        assert not c.ok and "audio=none" in c.measured and "src.mp4" in c.measured and c.hint


# ---------------------------------------------------------------------------
# 5. backend log assertion
# ---------------------------------------------------------------------------


class TestBackendLog:
    def test_matching_backends_pass(self) -> None:
        log_text = "🎚️  Streaming backends: depth=depthcrafter, stereo=stereocrafter"
        c = check_backend_log(log_text, "depthcrafter", "stereocrafter")
        assert c.ok and "depthcrafter" in c.measured

    def test_mismatch_fails_with_measured(self) -> None:
        log_text = "🎚️  Streaming backends: depth=depth-anything, stereo=default"
        c = check_backend_log(log_text, "depthcrafter", "stereocrafter")
        assert not c.ok
        assert "depth-anything" in c.measured and "depthcrafter" in c.measured and c.hint

    def test_missing_line_fails(self) -> None:
        c = check_backend_log("some unrelated log", "depthcrafter", "stereocrafter")
        assert not c.ok and "no 'Streaming backends' line" in c.measured and c.hint


# ---------------------------------------------------------------------------
# 6. sidecar
# ---------------------------------------------------------------------------


class TestSidecar:
    def _good_sidecar(self) -> dict:
        return {
            "immersive": {
                "projection": "equirect180",
                "fov_deg": 180,
                "stereo_layout": "side_by_side",
                "eye_resolution": [3840, 3840],
            }
        }

    def test_missing_sidecar_fails(self, tmp_path: Path) -> None:
        f = tmp_path / "o.mp4"
        f.touch()
        c = check_sidecar(str(f))
        assert not c.ok and "missing" in c.measured and c.hint

    def test_valid_sidecar_passes(self, tmp_path: Path) -> None:
        f = tmp_path / "o.mp4"
        f.touch()
        (f.parent / (f.stem + ".json")).write_text(json.dumps(self._good_sidecar()), encoding="utf-8")
        c = check_sidecar(str(f))
        assert c.ok

    def test_missing_required_field_fails_with_names(self, tmp_path: Path) -> None:
        f = tmp_path / "o.mp4"
        f.touch()
        bad = self._good_sidecar()
        del bad["immersive"]["eye_resolution"]
        c = check_sidecar(str(f), sidecar=bad)
        assert not c.ok and "eye_resolution" in c.measured and c.hint

    def test_injected_sidecar_used(self) -> None:
        c = check_sidecar("unused.mp4", sidecar=self._good_sidecar())
        assert c.ok


# ---------------------------------------------------------------------------
# Orchestration + exit-code aggregation
# ---------------------------------------------------------------------------


class TestRunSmoke:
    def _runner_ok(self, tmp_path: Path, output_path: str, log_line: str):
        """Fake subprocess.run that writes a valid artefact + sidecar."""

        def runner(cmd, capture_output=True, text=True):
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            # Full profile asserts the streaming-backends log line; fast skips it.
            out.write_bytes(_synthetic_vr180_bytes("sbs"))
            sidecar = {
                "immersive": {
                    "projection": "equirect180",
                    "fov_deg": 180,
                    "stereo_layout": "side_by_side",
                    "eye_resolution": [3840, 3840],
                }
            }
            (out.parent / (out.stem + ".json")).write_text(json.dumps(sidecar), encoding="utf-8")
            return _fake_proc(0, stdout=log_line, stderr="")

        return runner

    def test_fast_profile_passes(self, tmp_path: Path, monkeypatch) -> None:
        out = str(tmp_path / "o.mp4")
        # Probe + byte-scan run for real against the fake artefact; monkeypatch
        # the ffprobe helper so no real ffprobe is needed.
        monkeypatch.setattr(
            "scripts.e2e_smoke._ffprobe_streams",
            lambda path, ffprobe="ffprobe": _probe(3840, 1920),  # preview = 1920²/eye → 3840×1920
        )
        runner = self._runner_ok(tmp_path, out, log_line="")
        report = run_smoke("in.mp4", out, "fast", runner=runner)
        assert report.ok, [(c.name, c.measured) for c in report.failed]

    def test_full_profile_log_assertion_fires(self, tmp_path: Path, monkeypatch) -> None:
        out = str(tmp_path / "o.mp4")
        monkeypatch.setattr(
            "scripts.e2e_smoke._ffprobe_streams",
            lambda path, ffprobe="ffprobe": _probe(7680, 3840),  # high = 3840²/eye
        )
        # Log claims default backends but full requested depthcrafter/stereocrafter.
        runner = self._runner_ok(tmp_path, out, log_line="🎚️  Streaming backends: depth=depth-anything, stereo=default")
        report = run_smoke("in.mp4", out, "full", runner=runner)
        assert not report.ok
        backend_check = next(c for c in report.checks if c.name == "backend log assertion")
        assert not backend_check.ok and "depth-anything" in backend_check.measured

    def test_pipeline_failure_propagates(self, tmp_path: Path, monkeypatch) -> None:
        out = str(tmp_path / "o.mp4")
        monkeypatch.setattr(
            "scripts.e2e_smoke._ffprobe_streams",
            lambda path, ffprobe="ffprobe": _probe(3840, 1920),
        )
        runner = lambda cmd, capture_output=True, text=True: _fake_proc(1, stderr="boom")  # noqa: E731
        report = run_smoke("in.mp4", out, "fast", runner=runner)
        assert not report.ok
        exit_check = next(c for c in report.checks if c.name == "pipeline exit code")
        assert not exit_check.ok and "exit=1" in exit_check.measured

    def test_ci_profile_passes(self, tmp_path: Path, monkeypatch) -> None:
        out = str(tmp_path / "o.mp4")
        # ci = preview tier + 256²/eye override → 512×256 SBS.
        monkeypatch.setattr(
            "scripts.e2e_smoke._ffprobe_streams",
            lambda path, ffprobe="ffprobe": _probe(512, 256),
        )
        runner = self._runner_ok(tmp_path, out, log_line="")
        report = run_smoke("in.mp4", out, "ci", runner=runner)
        assert report.ok, [(c.name, c.measured) for c in report.failed]

    def test_ci_profile_skips_backend_log_check(self, tmp_path: Path, monkeypatch) -> None:
        out = str(tmp_path / "o.mp4")
        monkeypatch.setattr(
            "scripts.e2e_smoke._ffprobe_streams",
            lambda path, ffprobe="ffprobe": _probe(512, 256),
        )
        runner = self._runner_ok(tmp_path, out, log_line="")
        report = run_smoke("in.mp4", out, "ci", runner=runner)
        # The SBS path never constructs a depth/stereo backend, so the
        # streaming-backends assertion is intentionally absent (not an N/A
        # pass — it simply does not apply to this profile).
        assert "backend log assertion" not in [c.name for c in report.checks]


# ---------------------------------------------------------------------------
# CLI wiring (argparse + main exit code, pipeline mocked)
# ---------------------------------------------------------------------------


class TestCli:
    def test_parse_defaults(self) -> None:
        args = parse_args([])
        assert args.profile == "fast" and args.json is False

    def test_parse_full_profile(self) -> None:
        args = parse_args(["--profile", "full", "--copy-audio-from", "src.mp4", "--json"])
        assert args.profile == "full" and args.copy_audio_from == "src.mp4" and args.json

    def test_missing_input_exits_1(self, tmp_path: Path, capsys) -> None:
        rc = main(["--input", str(tmp_path / "nope.mp4"), "--profile", "fast"])
        assert rc == 1
        assert "input not found" in capsys.readouterr().err

    def test_profiles_table_has_fast_ci_and_full(self) -> None:
        assert set(PROFILES) == {"fast", "ci", "full"}
        assert PROFILES["fast"]["expected_depth"] == "depth-anything"
        assert PROFILES["full"]["expected_depth"] == "depthcrafter"
        assert PROFILES["full"]["expected_stereo"] == "stereocrafter"
        # ci: 256²/eye override + no backend-log check (SBS path, no backends).
        assert PROFILES["ci"]["eye"] == 256
        assert PROFILES["fast"]["eye"] is None and PROFILES["full"]["eye"] is None
        assert "backend_log" not in PROFILES["ci"]["checks"]
        assert "backend_log_na" not in PROFILES["ci"]["checks"]
        # K-7 (#160): full auto-wires --copy-audio-from=<self> and adds the
        # fresh-depth meta.json assertion.
        assert PROFILES["full"].get("copy_audio_self") is True
        assert "depth_meta" in PROFILES["full"]["checks"]


# ---------------------------------------------------------------------------
# K-7 (#160): full profile --copy-audio-from self + depth meta.json assertion
# ---------------------------------------------------------------------------


class TestFullCopyAudioSelf:
    """full profile auto-appends --copy-audio-from=<input itself>."""

    def test_full_profile_auto_appends_copy_audio_self(self) -> None:
        cmd = build_pipeline_command("in.mp4", "out.mp4", "full")
        # build_pipeline_command sees copy_audio_from=None; the self-wiring
        # happens in run_smoke.  Here we verify the profile flag is set and
        # that run_smoke surfaces it into the command.
        assert PROFILES["full"].get("copy_audio_self") is True
        joined = " ".join(cmd)
        # The base args must NOT already contain copy-audio-from (run_smoke
        # adds it from input_path).
        assert "--copy-audio-from" not in joined

    def test_run_smoke_full_wires_copy_audio_from_self(self) -> None:
        # Capture the command the runner was called with.
        captured = []

        def runner(cmd, capture_output=True, text=True):
            captured.append(cmd)
            return _fake_proc(0, stdout="")

        # run_smoke with profile=full and no explicit copy_audio_from must
        # build a command that carries --copy-audio-from <input>.
        report = run_smoke("in.mp4", "out.mp4", "full", runner=runner)
        assert report is not None  # depth_meta check will fail (no meta); we
        # only care about the built command here.
        cmd = captured[0]
        joined = " ".join(cmd)
        assert "--copy-audio-from in.mp4" in joined

    def test_explicit_copy_audio_from_overrides_self(self) -> None:
        captured = []

        def runner(cmd, capture_output=True, text=True):
            captured.append(cmd)
            return _fake_proc(0, stdout="")

        run_smoke("in.mp4", "out.mp4", "full", copy_audio_from="src.mp4", runner=runner)
        joined = " ".join(captured[0])
        assert "--copy-audio-from src.mp4" in joined
        assert "--copy-audio-from in.mp4" not in joined


class TestDepthMeta:
    """7. (full only) depth meta.json fresh + matches this run (I-6 / #121)."""

    def test_full_fresh_meta_passes(self, tmp_path: Path) -> None:
        meta = {
            "depth_model": "depthcrafter",
            "num_frames": 60,
            "model_size": "small",
            "max_res": 512,
            "temporal_smoothing": 0.0,
            "timestamp": "2026-09-01T03:00:00",
        }
        c = check_depth_meta("o.mp4", "full", meta=meta)
        assert c.ok
        assert "depth_model=depthcrafter" in c.measured
        assert "timestamp=2026-09-01T03:00:00" in c.measured

    def test_full_wrong_model_fails_with_measured(self) -> None:
        meta = {"depth_model": "depth-anything", "timestamp": "2026-09-01T03:00:00"}
        c = check_depth_meta("o.mp4", "full", meta=meta)
        assert not c.ok
        assert "depth_model=depth-anything" in c.measured and c.hint

    def test_full_missing_timestamp_fails(self) -> None:
        meta = {"depth_model": "depthcrafter", "timestamp": None}
        c = check_depth_meta("o.mp4", "full", meta=meta)
        assert not c.ok and "timestamp=None" in c.measured and c.hint

    def test_full_empty_timestamp_fails(self) -> None:
        meta = {"depth_model": "depthcrafter", "timestamp": "  "}
        c = check_depth_meta("o.mp4", "full", meta=meta)
        assert not c.ok and "timestamp=  " in c.measured and c.hint

    def test_full_no_meta_supplied_fails(self) -> None:
        c = check_depth_meta("o.mp4", "full", meta=None)
        assert not c.ok and "no depth meta supplied" in c.measured and c.hint

    def test_non_full_profile_is_na_pass(self) -> None:
        # depth_meta assertion is full-only: fast/ci must never fail on it.
        c = check_depth_meta("o.mp4", "fast", meta=None)
        assert c.ok and "N/A" in c.measured
        c_ci = check_depth_meta("o.mp4", "ci", meta=None)
        assert c_ci.ok and "N/A" in c_ci.measured


class TestFullOrchestration:
    def _runner_ok(self, tmp_path: Path, output_path: str, log_line: str):
        def runner(cmd, capture_output=True, text=True):
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(_synthetic_vr180_bytes("sbs"))
            sidecar = {
                "immersive": {
                    "projection": "equirect180",
                    "fov_deg": 180,
                    "stereo_layout": "side_by_side",
                    "eye_resolution": [3840, 3840],
                }
            }
            (out.parent / (out.stem + ".json")).write_text(json.dumps(sidecar), encoding="utf-8")
            return _fake_proc(0, stdout=log_line, stderr="")

        return runner

    def test_full_profile_passes_with_fresh_meta(self, tmp_path: Path, monkeypatch) -> None:
        out = str(tmp_path / "o.mp4")
        # full auto-wires --copy-audio-from=<self>, so the audio check runs
        # and needs an audio stream in the probe.
        monkeypatch.setattr(
            "scripts.e2e_smoke._ffprobe_streams",
            lambda path, ffprobe="ffprobe": _probe(7680, 3840, audio=True),  # high = 3840²/eye
        )
        runner = self._runner_ok(
            tmp_path,
            out,
            log_line="🎚️  Streaming backends: depth=depthcrafter, stereo=stereocrafter",
        )
        meta = {
            "depth_model": "depthcrafter",
            "num_frames": 60,
            "max_res": 512,
            "timestamp": "2026-09-01T03:00:00",
        }
        report = run_smoke("in.mp4", out, "full", depth_meta=meta, runner=runner)
        assert report.ok, [(c.name, c.measured) for c in report.failed]
        # Confirm the full-specific checks actually ran (not silently skipped).
        names = [c.name for c in report.checks]
        assert "backend log assertion" in names
        assert "depth meta.json" in names

    def test_full_profile_fails_when_meta_stale(self, tmp_path: Path, monkeypatch) -> None:
        out = str(tmp_path / "o.mp4")
        monkeypatch.setattr(
            "scripts.e2e_smoke._ffprobe_streams",
            lambda path, ffprobe="ffprobe": _probe(7680, 3840, audio=True),
        )
        runner = self._runner_ok(
            tmp_path,
            out,
            log_line="🎚️  Streaming backends: depth=depthcrafter, stereo=stereocrafter",
        )
        meta = {"depth_model": "depth-anything", "timestamp": "2026-09-01T02:00:00"}
        report = run_smoke("in.mp4", out, "full", depth_meta=meta, runner=runner)
        assert not report.ok
        depth_check = next(c for c in report.checks if c.name == "depth meta.json")
        assert not depth_check.ok and "depth_model=depth-anything" in depth_check.measured


# ---------------------------------------------------------------------------
# Regression: ci/fast behaviour must be unchanged (K-7 did not touch them)
# ---------------------------------------------------------------------------


class TestRegressionFastCiUnchanged:
    def test_fast_args_unchanged(self) -> None:
        cmd = build_pipeline_command("in.mp4", "out.mp4", "fast")
        joined = " ".join(cmd)
        assert "--quality preview" in joined and "--max-frames 8" in joined
        assert "--depth-model" not in cmd and "--stereo-model" not in cmd
        assert "--copy-audio-from" not in joined

    def test_ci_args_unchanged(self) -> None:
        cmd = build_pipeline_command("in.mp4", "out.mp4", "ci")
        joined = " ".join(cmd)
        assert "--quality preview" in joined and "--max-frames 4" in joined
        assert "--output-width 256" in joined and "--output-height 256" in joined
        assert "--force-sbs" in cmd and "--no-ffmpeg-v360" in cmd
        assert "--depth-model" not in cmd and "--stereo-model" not in cmd
        assert "--copy-audio-from" not in joined

    def test_fast_profile_run_skips_depth_meta(self, tmp_path: Path, monkeypatch) -> None:
        out = str(tmp_path / "o.mp4")
        monkeypatch.setattr(
            "scripts.e2e_smoke._ffprobe_streams",
            lambda path, ffprobe="ffprobe": _probe(3840, 1920),
        )

        # _runner_ok from TestRunSmoke is not imported; build a minimal one.
        def runner(cmd, capture_output=True, text=True):
            p = Path(out)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(_synthetic_vr180_bytes("sbs"))
            (p.parent / (p.stem + ".json")).write_text(
                json.dumps(
                    {
                        "immersive": {
                            "projection": "equirect180",
                            "fov_deg": 180,
                            "stereo_layout": "side_by_side",
                            "eye_resolution": [3840, 3840],
                        }
                    }
                ),
                encoding="utf-8",
            )
            return _fake_proc(0, stdout="", stderr="")

        report = run_smoke("in.mp4", out, "fast", runner=runner)
        assert report.ok, [(c.name, c.measured) for c in report.failed]
        # fast has backend_log_na (N/A), NOT depth_meta.
        assert "backend log assertion" in [c.name for c in report.checks]
        assert "depth meta.json" not in [c.name for c in report.checks]
