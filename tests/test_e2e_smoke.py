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
- C-3 (#292): the ci profile's 10-bit HEVC (yuv420p10le) sample — encoder
  fallback, source-really-10-bit, decodable + not-all-black, vr180_qa
  verdict, sidecar pix_fmt, report/JSON surfacing — all faked from one
  argv-dispatching runner; plus ONE real-chain test (ffmpeg + run_pipeline
  --force-sbs, no model) whose only permitted skip is "no 10-bit encoder".
"""

from __future__ import annotations

import json
import struct
import subprocess
from dataclasses import asdict
from pathlib import Path

import pytest
import scripts.vr180_qa as vr180_qa
from scripts.e2e_smoke import (
    MIN_MEAN_LUMA,
    PROFILES,
    TENBIT_ENCODERS,
    TENBIT_PIX_FMT,
    Check,
    SmokeReport,
    TenbitSampleError,
    _decode_gray_frames,
    _print_report,
    _scan_boxes,
    build_pipeline_command,
    build_tenbit_sample_command,
    check_audio_stream,
    check_backend_log,
    check_decodable,
    check_depth_meta,
    check_exit_code,
    check_metadata_bytes,
    check_output_probe,
    check_qa_verdict,
    check_sidecar,
    check_sidecar_pix_fmt,
    check_source_pix_fmt,
    main,
    parse_args,
    probe_pix_fmt,
    run_smoke,
    run_tenbit_sample,
    synthesize_tenbit_sample,
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


def _probe(
    width: int,
    height: int,
    audio: bool = False,
    pix_fmt: str = "yuv420p",
    codec: str = "h264",
) -> dict:
    """A fake ffprobe JSON payload."""
    streams = [{"codec_type": "video", "width": width, "height": height, "codec_name": codec, "pix_fmt": pix_fmt}]
    if audio:
        streams.append({"codec_type": "audio", "codec_name": "aac", "bit_rate": "128000"})
    return {"streams": streams, "format": {"duration": "1.0"}}


def _fake_proc(returncode: int = 0, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=["x"], returncode=returncode, stdout=stdout, stderr=stderr)


# --- C-3 (#292): hermetic fakes for everything run_smoke(ci) shells out to ---

_GRAY_FRAME = 512 * 256  # ci product = 512×256 → bytes per raw gray frame


def _write_fake_artefact(out: Path, pix_fmt: str = "yuv420p") -> None:
    """A valid sv3d/st3d artefact + D-3 sidecar (with a video.pix_fmt) at *out*."""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(_synthetic_vr180_bytes("sbs"))
    sidecar = {
        "video": {"codec": "h264", "pix_fmt": pix_fmt},
        "immersive": {
            "projection": "equirect180",
            "fov_deg": 180,
            "stereo_layout": "side_by_side",
            "eye_resolution": [3840, 3840],
        },
    }
    (out.parent / (out.stem + ".json")).write_text(json.dumps(sidecar), encoding="utf-8")


def _ci_probe(path: str, ffprobe: str = "ffprobe") -> dict:
    """Path-aware fake ffprobe: the synthesised 10-bit sample vs the 512×256 product."""
    if "_10bit_src" in str(path):
        return _probe(256, 256, pix_fmt="yuv420p10le", codec="hevc")
    return _probe(512, 256)


def _qa_report(path: str, verdict: str = vr180_qa.VERDICT_VR180, checks: tuple = ()) -> vr180_qa.QAReport:
    return vr180_qa.QAReport(path=path, verdict=verdict, checks=list(checks))


def _ci_runner(log_line: str = "", pipeline_rc: int = 0, synth_rc: int = 0, gray_value: int = 0x80):
    """Fake subprocess.run dispatching on argv shape (records every argv in ``.calls``).

    ffmpeg raw-gray decode → 4 frames of *gray_value*; ffmpeg 10-bit synth →
    writes its dest (or fails with *synth_rc*); run_pipeline → writes artefact
    + sidecar at its ``--output`` (or fails with *pipeline_rc*).
    """
    calls: list[list[str]] = []

    def runner(cmd, capture_output=True, text=True, **kwargs):
        calls.append(list(cmd))
        if cmd[0] == "ffmpeg" and "rawvideo" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout=bytes([gray_value]) * (_GRAY_FRAME * 4), stderr=b"")
        if cmd[0] == "ffmpeg":
            if synth_rc == 0:
                Path(cmd[-1]).write_bytes(b"\x00" * 64)
            return _fake_proc(synth_rc, stderr="" if synth_rc == 0 else "Unknown encoder 'libx265'")
        out = Path(cmd[cmd.index("--output") + 1])
        if pipeline_rc == 0:
            _write_fake_artefact(out)
        return _fake_proc(pipeline_rc, stdout=log_line, stderr="boom" if pipeline_rc else "")

    runner.calls = calls
    return runner


@pytest.fixture
def ci_chain(monkeypatch):
    """Hermetic ci chain: path-aware ffprobe + a VR180 vr180_qa verdict (no shell-outs)."""
    monkeypatch.setattr("scripts.e2e_smoke._ffprobe_streams", _ci_probe)
    monkeypatch.setattr("scripts.e2e_smoke.run_qa", lambda path, ffprobe="ffprobe": _qa_report(path))


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
    def _runner_ok(self, log_line: str):
        """Fake subprocess.run that writes a valid artefact + sidecar at --output.

        C-3 (#292): the ci profile also shells out for the 10-bit sample
        (ffmpeg synth + raw-gray decode) and a second pipeline run — the
        argv-dispatching ``_ci_runner`` fakes all of them.
        """
        return _ci_runner(log_line=log_line)

    def test_fast_profile_passes(self, tmp_path: Path, monkeypatch) -> None:
        out = str(tmp_path / "o.mp4")
        # Probe + byte-scan run for real against the fake artefact; monkeypatch
        # the ffprobe helper so no real ffprobe is needed.
        monkeypatch.setattr(
            "scripts.e2e_smoke._ffprobe_streams",
            lambda path, ffprobe="ffprobe": _probe(3840, 1920),  # preview = 1920²/eye → 3840×1920
        )
        runner = self._runner_ok(log_line="")
        report = run_smoke("in.mp4", out, "fast", runner=runner)
        assert report.ok, [(c.name, c.measured) for c in report.failed]

    def test_full_profile_log_assertion_fires(self, tmp_path: Path, monkeypatch) -> None:
        out = str(tmp_path / "o.mp4")
        monkeypatch.setattr(
            "scripts.e2e_smoke._ffprobe_streams",
            lambda path, ffprobe="ffprobe": _probe(7680, 3840),  # high = 3840²/eye
        )
        # Log claims default backends but full requested depthcrafter/stereocrafter.
        runner = self._runner_ok(log_line="🎚️  Streaming backends: depth=depth-anything, stereo=default")
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

    def test_ci_profile_passes(self, tmp_path: Path, ci_chain) -> None:
        out = str(tmp_path / "o.mp4")
        # ci = preview tier + 256²/eye override → 512×256 SBS (path-aware
        # probe: the 10-bit sample reads back as 256² yuv420p10le hevc).
        runner = self._runner_ok(log_line="")
        report = run_smoke("in.mp4", out, "ci", runner=runner)
        assert report.ok, [(c.name, c.measured) for c in report.failed]
        # C-3 (#292): the 10-bit sub-run's checks are part of the SAME report.
        names = [c.name for c in report.checks]
        assert "10bit sample synthesis" in names and "10bit output decodable" in names
        assert "10bit vr180_qa verdict" in names and "10bit sidecar pix_fmt" in names
        assert report.tenbit_encoder == "libx265"
        assert report.tenbit_source_pix_fmt == TENBIT_PIX_FMT
        assert report.source_pix_fmt == "yuv420p"
        assert report.tenbit_sample.endswith("o_10bit_src.mp4") and report.tenbit_output.endswith("o_10bit.mp4")

    def test_ci_profile_skips_backend_log_check(self, tmp_path: Path, ci_chain) -> None:
        out = str(tmp_path / "o.mp4")
        runner = self._runner_ok(log_line="")
        report = run_smoke("in.mp4", out, "ci", runner=runner)
        # The SBS path never constructs a depth/stereo backend, so the
        # streaming-backends assertion is intentionally absent (not an N/A
        # pass — it simply does not apply to this profile).
        assert "backend log assertion" not in [c.name for c in report.checks]

    def test_ci_report_fails_when_only_tenbit_fails(self, tmp_path: Path, ci_chain) -> None:
        # C-3 (#292): the 8-bit main run is green, the 10-bit sample cannot be
        # synthesised → the WHOLE report is red (no silent skip, ever).
        out = str(tmp_path / "o.mp4")
        report = run_smoke("in.mp4", out, "ci", runner=_ci_runner(synth_rc=1))
        assert not report.ok
        assert [c.name for c in report.failed] == ["10bit sample synthesis"]
        assert all(c.ok for c in report.checks if not c.name.startswith("10bit"))
        assert report.tenbit_encoder == "" and report.tenbit_sample.endswith("o_10bit_src.mp4")

    def test_fast_profile_has_no_tenbit_run(self, tmp_path: Path, monkeypatch) -> None:
        out = str(tmp_path / "o.mp4")
        monkeypatch.setattr("scripts.e2e_smoke._ffprobe_streams", lambda path, ffprobe="ffprobe": _probe(3840, 1920))
        runner = _ci_runner()
        report = run_smoke("in.mp4", out, "fast", runner=runner)
        assert report.ok
        assert not [c for c in report.checks if c.name.startswith("10bit")]
        assert report.tenbit_sample == "" and report.tenbit_encoder == ""
        # fast issues exactly one subprocess: the pipeline itself.
        assert len(runner.calls) == 1 and "run_pipeline.py" in runner.calls[0][1]


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
        # C-3 (#292): only ci carries the 10-bit HEVC sample sub-run.
        assert PROFILES["ci"].get("tenbit_sample") is True
        assert not PROFILES["fast"].get("tenbit_sample") and not PROFILES["full"].get("tenbit_sample")

    def test_report_header_prints_source_pix_fmt(self, capsys) -> None:
        # C-3 (#292): "e2e_smoke.py 输出里打印源 pix_fmt" — for the main input
        # AND the 10-bit sample, plus the encoder that produced the sample.
        report = SmokeReport(
            input="in.mp4",
            output="out.mp4",
            profile="ci",
            checks=[Check("pipeline exit code", True, measured="exit=0")],
            source_pix_fmt="yuv420p",
            tenbit_sample="out_10bit_src.mp4",
            tenbit_output="out_10bit.mp4",
            tenbit_encoder="libx265",
            tenbit_source_pix_fmt="yuv420p10le",
        )
        _print_report(report)
        text = capsys.readouterr().out
        assert "input : in.mp4  (source pix_fmt=yuv420p)" in text
        assert "10bit : out_10bit_src.mp4  (encoder=libx265, source pix_fmt=yuv420p10le) → out_10bit.mp4" in text
        assert "ALL 1 CHECKS PASSED" in text

    def test_report_header_without_tenbit_run(self, capsys) -> None:
        report = SmokeReport(input="in.mp4", output="out.mp4", profile="fast", checks=[Check("x", True)])
        _print_report(report)
        text = capsys.readouterr().out
        assert "(source pix_fmt=unknown)" in text and "10bit :" not in text

    def test_json_report_carries_pix_fmt_fields(self, tmp_path: Path, monkeypatch, capsys) -> None:
        src = tmp_path / "in.mp4"
        src.write_bytes(b"\x00")
        fake = SmokeReport(
            input=str(src),
            output="out.mp4",
            profile="ci",
            checks=[Check("10bit sidecar pix_fmt", True, measured="sidecar video.pix_fmt=yuv420p")],
            source_pix_fmt="yuv420p",
            tenbit_sample="out_10bit_src.mp4",
            tenbit_output="out_10bit.mp4",
            tenbit_encoder="libx264",
            tenbit_source_pix_fmt="yuv420p10le",
        )
        monkeypatch.setattr("scripts.e2e_smoke.run_smoke", lambda **kw: fake)
        rc = main(["--input", str(src), "--profile", "ci", "--json"])
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["source_pix_fmt"] == "yuv420p"
        assert payload["tenbit_encoder"] == "libx264" and payload["tenbit_source_pix_fmt"] == "yuv420p10le"
        assert payload["tenbit_sample"] == "out_10bit_src.mp4" and payload["tenbit_output"] == "out_10bit.mp4"
        assert payload["checks"][0]["name"] == "10bit sidecar pix_fmt"
        assert set(asdict(fake)) == set(payload)


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


# ---------------------------------------------------------------------------
# C-3 (#292): 10-bit HEVC (yuv420p10le) sample — Seedance 4k decode smoke
# ---------------------------------------------------------------------------


class TestTenbitSampleCommand:
    def test_command_is_the_cards_recipe(self) -> None:
        cmd = build_tenbit_sample_command("s.mp4", "libx265")
        assert isinstance(cmd, list) and all(isinstance(t, str) for t in cmd)
        assert cmd[0] == "ffmpeg" and cmd[-1] == "s.mp4"
        joined = " ".join(cmd)
        assert "-f lavfi -i testsrc2=size=256x256:rate=24" in joined
        assert "-t 1" in joined
        assert "-c:v libx265" in joined
        assert "-pix_fmt yuv420p10le" in joined

    def test_encoder_and_ffmpeg_are_substituted(self) -> None:
        cmd = build_tenbit_sample_command("s.mp4", "libx264", ffmpeg="/opt/ffmpeg")
        assert cmd[0] == "/opt/ffmpeg" and "libx264" in cmd and "libx265" not in cmd

    def test_encoder_preference_is_x265_then_x264(self) -> None:
        assert TENBIT_ENCODERS == ("libx265", "libx264")
        assert TENBIT_PIX_FMT == "yuv420p10le"


class TestSynthesizeTenbitSample:
    def _runner(self, outcomes: dict[str, int], write: bool = True):
        """Fake ffmpeg: per-encoder exit code; writes dest on success when *write*."""
        calls: list[str] = []

        def runner(cmd, capture_output=True, text=True, **kwargs):
            encoder = cmd[cmd.index("-c:v") + 1]
            calls.append(encoder)
            rc = outcomes[encoder]
            if rc == 0 and write:
                Path(cmd[-1]).write_bytes(b"\x00" * 16)
            return _fake_proc(rc, stderr="" if rc == 0 else f"Unknown encoder '{encoder}'")

        runner.calls = calls
        return runner

    def test_libx265_first(self, tmp_path: Path) -> None:
        runner = self._runner({"libx265": 0, "libx264": 0})
        assert synthesize_tenbit_sample(str(tmp_path / "s.mp4"), runner=runner) == "libx265"
        assert runner.calls == ["libx265"]

    def test_falls_back_to_libx264(self, tmp_path: Path) -> None:
        runner = self._runner({"libx265": 1, "libx264": 0})
        assert synthesize_tenbit_sample(str(tmp_path / "s.mp4"), runner=runner) == "libx264"
        assert runner.calls == ["libx265", "libx264"]

    def test_both_fail_raises_with_both_errors(self, tmp_path: Path) -> None:
        runner = self._runner({"libx265": 1, "libx264": 1})
        with pytest.raises(TenbitSampleError) as exc_info:
            synthesize_tenbit_sample(str(tmp_path / "s.mp4"), runner=runner)
        msg = str(exc_info.value)
        assert "libx265: exit=1" in msg and "libx264: exit=1" in msg and "Unknown encoder" in msg

    def test_missing_ffmpeg_binary_raises(self, tmp_path: Path) -> None:
        def runner(cmd, **kwargs):
            raise FileNotFoundError("ffmpeg")

        with pytest.raises(TenbitSampleError, match="FileNotFoundError"):
            synthesize_tenbit_sample(str(tmp_path / "s.mp4"), runner=runner)

    def test_exit_zero_without_file_counts_as_failure(self, tmp_path: Path) -> None:
        # An encoder that "succeeds" but leaves nothing behind must not win.
        runner = self._runner({"libx265": 0, "libx264": 0}, write=False)
        with pytest.raises(TenbitSampleError, match="libx264"):
            synthesize_tenbit_sample(str(tmp_path / "s.mp4"), runner=runner)
        assert runner.calls == ["libx265", "libx264"]


class TestSourcePixFmt:
    def test_10bit_source_passes(self) -> None:
        c = check_source_pix_fmt("s.mp4", probe=_probe(256, 256, pix_fmt="yuv420p10le", codec="hevc"))
        assert c.ok and "pix_fmt=yuv420p10le" in c.measured and "codec=hevc" in c.measured

    def test_8bit_source_fails_with_measured(self) -> None:
        # The encoder silently fell back to 8-bit → the whole sub-run would be vacuous.
        c = check_source_pix_fmt("s.mp4", probe=_probe(256, 256, pix_fmt="yuv420p"))
        assert not c.ok and "pix_fmt=yuv420p," in c.measured and "want yuv420p10le" in c.measured and c.hint

    def test_unreadable_source_fails(self, monkeypatch) -> None:
        def boom(path, ffprobe="ffprobe"):
            raise RuntimeError("ffprobe exit 1: moov atom not found")

        monkeypatch.setattr("scripts.e2e_smoke._ffprobe_streams", boom)
        c = check_source_pix_fmt("s.mp4")
        assert not c.ok and "moov atom" in c.measured and c.hint

    def test_probe_pix_fmt_helper_never_raises(self, monkeypatch) -> None:
        assert probe_pix_fmt("x", probe=_probe(1, 1, pix_fmt="yuv420p10le")) == "yuv420p10le"
        assert probe_pix_fmt("x", probe={"streams": [{"codec_type": "audio"}]}) == "unknown"
        assert probe_pix_fmt("x", probe={"streams": [{"codec_type": "video"}]}) == "unknown"

        def boom(path, ffprobe="ffprobe"):
            raise FileNotFoundError("ffprobe")

        monkeypatch.setattr("scripts.e2e_smoke._ffprobe_streams", boom)
        assert probe_pix_fmt("missing.mp4") == "unknown"


class TestDecodable:
    def test_frames_and_luma_pass(self) -> None:
        c = check_decodable("o.mp4", decoded=(4, 28.1), expect_frames=4)
        assert c.ok and "frames=4, want 4" in c.measured and "mean_luma=28.1" in c.measured

    def test_all_black_fails(self) -> None:
        c = check_decodable("o.mp4", decoded=(4, 0.0), expect_frames=4)
        assert not c.ok and "mean_luma=0.0" in c.measured and f"min {MIN_MEAN_LUMA}" in c.measured and c.hint

    def test_frame_count_mismatch_fails(self) -> None:
        c = check_decodable("o.mp4", decoded=(3, 28.1), expect_frames=4)
        assert not c.ok and "frames=3, want 4" in c.measured and c.hint

    def test_zero_frames_fails_without_expectation(self) -> None:
        c = check_decodable("o.mp4", decoded=(0, 0.0))
        assert not c.ok and "frames=0," in c.measured and "want" not in c.measured

    def test_any_frames_pass_without_expectation(self) -> None:
        assert check_decodable("o.mp4", decoded=(7, 50.0)).ok

    def test_decode_error_fails_with_measured(self, monkeypatch) -> None:
        def boom(path, ffmpeg="ffmpeg", runner=None):
            raise RuntimeError("ffmpeg exit 1: Invalid data found when processing input")

        monkeypatch.setattr("scripts.e2e_smoke._decode_gray_frames", boom)
        c = check_decodable("o.mp4", expect_frames=4)
        assert not c.ok and "decode failed" in c.measured and "Invalid data" in c.measured and c.hint

    def test_decode_gray_frames_counts_frames_and_means(self, monkeypatch) -> None:
        monkeypatch.setattr("scripts.e2e_smoke._ffprobe_streams", lambda path, ffprobe="ffprobe": _probe(512, 256))
        seen: list[list[str]] = []

        def runner(cmd, capture_output=True, **kwargs):
            seen.append(list(cmd))
            return subprocess.CompletedProcess(cmd, 0, stdout=bytes([100]) * (_GRAY_FRAME * 4), stderr=b"")

        assert _decode_gray_frames("o.mp4", runner=runner) == (4, 100.0)
        cmd = seen[0]
        assert cmd[0] == "ffmpeg" and cmd[-1] == "-" and "o.mp4" in cmd
        assert "-f rawvideo -pix_fmt gray" in " ".join(cmd)

    def test_decode_gray_frames_treats_stderr_as_failure(self, monkeypatch) -> None:
        # ``-v error`` → any stderr is a real decode error even when exit == 0.
        monkeypatch.setattr("scripts.e2e_smoke._ffprobe_streams", lambda path, ffprobe="ffprobe": _probe(512, 256))

        def runner(cmd, **kw):
            return subprocess.CompletedProcess(
                cmd, 0, stdout=b"\x80" * _GRAY_FRAME, stderr=b"[hevc] Could not find ref"
            )

        with pytest.raises(RuntimeError, match="Could not find ref"):
            _decode_gray_frames("o.mp4", runner=runner)

    def test_decode_gray_frames_nonzero_exit_raises(self, monkeypatch) -> None:
        monkeypatch.setattr("scripts.e2e_smoke._ffprobe_streams", lambda path, ffprobe="ffprobe": _probe(512, 256))

        def runner(cmd, **kw):
            return subprocess.CompletedProcess(cmd, 1, stdout=b"", stderr=b"boom")

        with pytest.raises(RuntimeError, match="ffmpeg exit 1"):
            _decode_gray_frames("o.mp4", runner=runner)

    def test_decode_gray_frames_needs_a_resolution(self, monkeypatch) -> None:
        monkeypatch.setattr("scripts.e2e_smoke._ffprobe_streams", lambda path, ffprobe="ffprobe": {"streams": []})
        with pytest.raises(RuntimeError, match="no video stream"):
            _decode_gray_frames("o.mp4", runner=lambda cmd, **kw: pytest.fail("must not decode without a size"))


class TestQaVerdict:
    def test_vr180_verdict_with_warns_passes(self) -> None:
        qa = _qa_report(
            "o.mp4",
            checks=(
                vr180_qa.Check("sv3d/st3d boxes", "pass", "sv3d + st3d present"),
                vr180_qa.Check("audio stream", "warn", "no audio stream"),
                vr180_qa.Check("per-eye resolution", "warn", "256px < 2880px"),
            ),
        )
        c = check_qa_verdict("o.mp4", qa=qa)
        assert c.ok and "pass=1 warn=2 fail=0" in c.measured and "VR180" in c.measured and "fails=" not in c.measured

    def test_fail_check_fails_with_names(self) -> None:
        qa = _qa_report(
            "o.mp4", checks=(vr180_qa.Check("stereo mode", "fail", "st3d mode=1, expected left-right (2)"),)
        )
        c = check_qa_verdict("o.mp4", qa=qa)
        assert not c.ok and "fail=1" in c.measured and "stereo mode: st3d mode=1" in c.measured and c.hint

    def test_non_vr180_verdict_fails(self) -> None:
        c = check_qa_verdict("o.mp4", qa=_qa_report("o.mp4", verdict=vr180_qa.VERDICT_PLAIN_2D))
        assert not c.ok and "plain 2D" in c.measured and c.hint

    def test_run_qa_is_called_on_the_product(self, monkeypatch) -> None:
        seen: list[str] = []

        def fake_run_qa(path, ffprobe="ffprobe"):
            seen.append(path)
            return _qa_report(path)

        monkeypatch.setattr("scripts.e2e_smoke.run_qa", fake_run_qa)
        assert check_qa_verdict("product.mp4").ok and seen == ["product.mp4"]

    def test_run_qa_crash_is_a_failed_check(self, monkeypatch) -> None:
        def boom(path, ffprobe="ffprobe"):
            raise FileNotFoundError("ffprobe")

        monkeypatch.setattr("scripts.e2e_smoke.run_qa", boom)
        c = check_qa_verdict("o.mp4")
        assert not c.ok and "run_qa crashed: FileNotFoundError" in c.measured and c.hint


class TestSidecarPixFmt:
    def test_recorded_pix_fmt_passes_and_shows_source(self) -> None:
        c = check_sidecar_pix_fmt("o.mp4", sidecar={"video": {"pix_fmt": "yuv420p"}}, source_pix_fmt="yuv420p10le")
        assert c.ok
        # The 10-bit → 8-bit down-conversion is visible at a glance.
        assert c.measured == "sidecar video.pix_fmt=yuv420p (source pix_fmt=yuv420p10le)"

    def test_missing_video_block_fails(self) -> None:
        c = check_sidecar_pix_fmt("o.mp4", sidecar={"immersive": {}})
        assert not c.ok and "video.pix_fmt=None" in c.measured and c.hint

    def test_empty_pix_fmt_fails(self) -> None:
        assert not check_sidecar_pix_fmt("o.mp4", sidecar={"video": {"pix_fmt": "  "}}).ok

    def test_missing_sidecar_file_fails(self, tmp_path: Path) -> None:
        c = check_sidecar_pix_fmt(str(tmp_path / "o.mp4"))
        assert not c.ok and "missing" in c.measured and c.hint

    def test_unreadable_sidecar_fails(self, tmp_path: Path) -> None:
        (tmp_path / "o.json").write_text("{not json", encoding="utf-8")
        c = check_sidecar_pix_fmt(str(tmp_path / "o.mp4"))
        assert not c.ok and "unreadable" in c.measured

    def test_reads_sidecar_from_disk(self, tmp_path: Path) -> None:
        _write_fake_artefact(tmp_path / "o.mp4", pix_fmt="yuv420p10le")
        c = check_sidecar_pix_fmt(str(tmp_path / "o.mp4"), source_pix_fmt="yuv420p10le")
        assert c.ok and "video.pix_fmt=yuv420p10le" in c.measured


#: The 10-bit sub-run's check names, in report order (C-3, #292).
_TENBIT_CHECK_NAMES = (
    "10bit sample synthesis",
    "10bit source pix_fmt",
    "10bit pipeline exit code",
    "10bit output exists + ffprobe",
    "10bit sv3d/st3d byte-scan",
    "10bit output decodable",
    "10bit vr180_qa verdict",
    "10bit sidecar JSON",
    "10bit sidecar pix_fmt",
)


class TestRunTenbitSample:
    """Orchestration of the 10-bit sub-run with every shell-out faked."""

    def test_all_checks_pass_with_fake_chain(self, tmp_path: Path, ci_chain) -> None:
        runner = _ci_runner()
        res = run_tenbit_sample(str(tmp_path / "o.mp4"), runner=runner)
        assert tuple(c.name for c in res.checks) == _TENBIT_CHECK_NAMES
        assert all(c.ok for c in res.checks), [(c.name, c.measured) for c in res.checks if not c.ok]
        assert res.encoder == "libx265" and res.source_pix_fmt == "yuv420p10le"
        assert res.sample == str(tmp_path / "o_10bit_src.mp4") and res.output == str(tmp_path / "o_10bit.mp4")
        # argv trail: x265 synth → run_pipeline (ci args, on the sample) → raw-gray decode.
        synth, pipeline, decode = runner.calls
        assert synth[0] == "ffmpeg" and "libx265" in synth and synth[-1] == res.sample
        assert "run_pipeline.py" in pipeline[1] and "--force-sbs" in pipeline and "--no-ffmpeg-v360" in pipeline
        assert pipeline[pipeline.index("--input") + 1] == res.sample
        assert pipeline[pipeline.index("--output") + 1] == res.output
        assert "--copy-audio-from" not in pipeline
        assert decode[0] == "ffmpeg" and "rawvideo" in decode and res.output in decode
        decodable = next(c for c in res.checks if c.name == "10bit output decodable")
        assert "frames=4, want 4" in decodable.measured  # ci --max-frames 4

    def test_synth_failure_is_a_failed_check_not_a_skip(self, tmp_path: Path, ci_chain) -> None:
        runner = _ci_runner(synth_rc=1)
        res = run_tenbit_sample(str(tmp_path / "o.mp4"), runner=runner)
        assert len(res.checks) == 1
        synth = res.checks[0]
        assert synth.name == "10bit sample synthesis" and not synth.ok and synth.hint
        assert "libx265: exit=1" in synth.measured and "libx264: exit=1" in synth.measured
        assert res.encoder == "" and res.source_pix_fmt == ""
        # Nothing else ran: two ffmpeg attempts, no pipeline, no decode.
        assert [c[0] for c in runner.calls] == ["ffmpeg", "ffmpeg"]

    def test_pipeline_failure_propagates(self, tmp_path: Path, ci_chain) -> None:
        res = run_tenbit_sample(str(tmp_path / "o.mp4"), runner=_ci_runner(pipeline_rc=1))
        exit_check = next(c for c in res.checks if c.name == "10bit pipeline exit code")
        assert not exit_check.ok and "exit=1" in exit_check.measured
        # Synthesis + source check still green: the failure is localised to the chain.
        assert res.checks[0].ok and res.checks[1].ok

    def test_black_product_fails_decodable(self, tmp_path: Path, ci_chain) -> None:
        res = run_tenbit_sample(str(tmp_path / "o.mp4"), runner=_ci_runner(gray_value=0))
        decodable = next(c for c in res.checks if c.name == "10bit output decodable")
        assert not decodable.ok and "mean_luma=0.0" in decodable.measured
        assert [c.name for c in res.checks if not c.ok] == ["10bit output decodable"]

    def test_8bit_sample_is_caught(self, tmp_path: Path, monkeypatch) -> None:
        # ffprobe says the "10-bit" sample is plain yuv420p → the source check fires.
        monkeypatch.setattr("scripts.e2e_smoke._ffprobe_streams", lambda path, ffprobe="ffprobe": _probe(512, 256))
        monkeypatch.setattr("scripts.e2e_smoke.run_qa", lambda path, ffprobe="ffprobe": _qa_report(path))
        res = run_tenbit_sample(str(tmp_path / "o.mp4"), runner=_ci_runner())
        assert res.source_pix_fmt == "yuv420p"
        assert [c.name for c in res.checks if not c.ok] == ["10bit source pix_fmt"]

    def test_qa_failure_propagates(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr("scripts.e2e_smoke._ffprobe_streams", _ci_probe)
        monkeypatch.setattr(
            "scripts.e2e_smoke.run_qa",
            lambda path, ffprobe="ffprobe": _qa_report(
                path, verdict=vr180_qa.VERDICT_PLAIN_2D, checks=(vr180_qa.Check("sv3d/st3d boxes", "fail", "none"),)
            ),
        )
        res = run_tenbit_sample(str(tmp_path / "o.mp4"), runner=_ci_runner())
        assert [c.name for c in res.checks if not c.ok] == ["10bit vr180_qa verdict"]


class TestTenbitRealChain:
    """C-3 (#292): the 10-bit HEVC sample REALLY walks the ci no-model chain.

    ffmpeg synth + ``run_pipeline.py --force-sbs`` + ffprobe + vr180_qa run for
    real (CPU-only, no model download, seconds).  The ONLY permitted skip is
    "no 10-bit encoder on this machine" (neither libx265 nor libx264 High-10,
    ffmpeg missing included) — any failure past synthesis is a real finding
    and must fail (or be ``xfail(strict=True)``-ed with the issue reference).
    """

    def test_ci_tenbit_sample_walks_real_chain(self, tmp_path: Path) -> None:
        res = run_tenbit_sample(str(tmp_path / "e2e_out.mp4"))
        synth = res.checks[0]
        assert synth.name == "10bit sample synthesis"
        if not synth.ok:
            pytest.skip(f"no 10-bit encoder (libx265 / libx264 High 10) on this machine: {synth.measured}")
        failed = [(c.name, c.measured, c.hint) for c in res.checks if not c.ok]
        assert not failed, f"10-bit HEVC sample broke the ci chain — module/error per check: {failed}"
        assert res.source_pix_fmt == TENBIT_PIX_FMT
        assert res.encoder in TENBIT_ENCODERS
        assert tuple(c.name for c in res.checks) == _TENBIT_CHECK_NAMES
