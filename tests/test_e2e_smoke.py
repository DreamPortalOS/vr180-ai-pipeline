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

    def test_copy_audio_from_appended(self) -> None:
        cmd = build_pipeline_command("in.mp4", "out.mp4", "full", copy_audio_from="src.mp4")
        assert "--copy-audio-from" in cmd and "src.mp4" in cmd

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

    def test_profiles_table_has_fast_and_full(self) -> None:
        assert set(PROFILES) == {"fast", "full"}
        assert PROFILES["fast"]["expected_depth"] == "depth-anything"
        assert PROFILES["full"]["expected_depth"] == "depthcrafter"
        assert PROFILES["full"]["expected_stereo"] == "stereocrafter"
