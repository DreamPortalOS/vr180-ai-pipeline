"""Tests for scripts/vr180_qa.py — VR180 output QA validator.

ffprobe is mocked; mp4 box layer uses real spherical_injector primitives to
build tiny synthetic ISOBMFF files (ftyp + moov/trak/mdia/minf/stbl) so the
box scanning runs against real bytes, not mocks.
"""

from __future__ import annotations

import json

import pytest
from scripts import vr180_qa
from scripts.vr180_qa import (
    VERDICT_FULLDOME,
    VERDICT_PLAIN_2D,
    VERDICT_VR180,
)

from pipeline.spherical_injector import _box4, _build_st3d, _build_sv3d


def _make_mp4(tmp_path, name: str, boxes: bytes = b"") -> str:
    """Build a minimal synthetic mp4 with the given boxes inside stbl."""
    stbl = _box4(b"stbl", boxes)
    minf = _box4(b"minf", stbl)
    mdia = _box4(b"mdia", minf)
    trak = _box4(b"trak", mdia)
    moov = _box4(b"moov", trak)
    ftyp = _box4(b"ftyp", b"isom\x00\x00\x02\x00isomiso2")
    path = tmp_path / name
    path.write_bytes(ftyp + moov)
    return str(path)


def _probe_json(width: int, height: int, fps: str = "30/1", bitrate: str = "45000000") -> dict:
    return {
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": width,
                "height": height,
                "avg_frame_rate": fps,
                "bit_rate": bitrate,
            }
        ],
        "format": {"bit_rate": bitrate, "duration": "12.5"},
    }


def _mock_probe(monkeypatch, width: int, height: int, fps: str = "30/1"):
    monkeypatch.setattr(vr180_qa, "_probe", lambda path, ffprobe="ffprobe": _probe_json(width, height, fps))


def _status(report, name: str) -> str:
    for check in report.checks:
        if check.name == name:
            return check.status
    raise AssertionError(f"check {name!r} not found in report")


class TestVerdicts:
    """One case per verdict class, per the issue's acceptance criteria."""

    def test_vr180_verdict(self, tmp_path, monkeypatch):
        path = _make_mp4(tmp_path, "vr180.mp4", _build_sv3d(5760, 2880, "sbs") + _build_st3d("sbs"))
        _mock_probe(monkeypatch, 5760, 2880)
        report = vr180_qa.run_qa(path)
        assert report.verdict == VERDICT_VR180
        assert not report.failed
        assert _status(report, "sv3d/st3d boxes") == "pass"
        assert _status(report, "stereo mode") == "pass"
        assert _status(report, "SBS square-eye layout") == "pass"
        assert _status(report, "per-eye resolution") == "pass"

    def test_fulldome_verdict(self, tmp_path, monkeypatch):
        path = _make_mp4(tmp_path, "dome.mp4")
        _mock_probe(monkeypatch, 2048, 2048)
        report = vr180_qa.run_qa(path)
        assert report.verdict == VERDICT_FULLDOME

    def test_plain_2d_verdict(self, tmp_path, monkeypatch):
        path = _make_mp4(tmp_path, "flat.mp4")
        _mock_probe(monkeypatch, 1920, 1080)
        report = vr180_qa.run_qa(path)
        assert report.verdict == VERDICT_PLAIN_2D
        assert report.failed


class TestFailures:
    def test_missing_st3d_fails(self, tmp_path, monkeypatch):
        path = _make_mp4(tmp_path, "half.mp4", _build_sv3d(5760, 2880, "sbs"))
        _mock_probe(monkeypatch, 5760, 2880)
        report = vr180_qa.run_qa(path)
        assert _status(report, "sv3d/st3d boxes") == "fail"
        assert report.failed

    def test_wrong_stereo_mode_fails(self, tmp_path, monkeypatch):
        path = _make_mp4(tmp_path, "tb.mp4", _build_sv3d(5760, 2880, "tb") + _build_st3d("tb"))
        _mock_probe(monkeypatch, 5760, 2880)
        report = vr180_qa.run_qa(path)
        assert _status(report, "stereo mode") == "fail"
        assert report.failed

    def test_non_square_eye_layout_fails(self, tmp_path, monkeypatch):
        # 3840×1080: SBS-ish but each eye is 1920×1080, not square
        path = _make_mp4(tmp_path, "wide.mp4", _build_sv3d(3840, 1080, "sbs") + _build_st3d("sbs"))
        _mock_probe(monkeypatch, 3840, 1080)
        report = vr180_qa.run_qa(path)
        assert _status(report, "SBS square-eye layout") == "fail"
        assert report.verdict != VERDICT_VR180

    def test_file_not_found_fails(self):
        report = vr180_qa.run_qa("does/not/exist.mp4")
        assert report.failed
        assert _status(report, "input file") == "fail"

    def test_ffprobe_failure_fails(self, tmp_path, monkeypatch):
        path = _make_mp4(tmp_path, "ok.mp4")

        def _boom(path, ffprobe="ffprobe"):
            raise RuntimeError("ffprobe failed (exit 1)")

        monkeypatch.setattr(vr180_qa, "_probe", _boom)
        report = vr180_qa.run_qa(path)
        assert report.failed


class TestWarnings:
    def test_low_per_eye_resolution_warns(self, tmp_path, monkeypatch):
        # 3840×1920 → per-eye 1920px < 2880px standard tier
        path = _make_mp4(tmp_path, "lowres.mp4", _build_sv3d(3840, 1920, "sbs") + _build_st3d("sbs"))
        _mock_probe(monkeypatch, 3840, 1920)
        report = vr180_qa.run_qa(path)
        assert _status(report, "per-eye resolution") == "warn"
        # WARN alone must not fail the run — this is still valid VR180
        assert report.verdict == VERDICT_VR180
        assert not report.failed


class TestExitCodes:
    def test_exit_zero_on_pass(self, tmp_path, monkeypatch, capsys):
        path = _make_mp4(tmp_path, "vr180.mp4", _build_sv3d(5760, 2880, "sbs") + _build_st3d("sbs"))
        _mock_probe(monkeypatch, 5760, 2880)
        assert vr180_qa.main([path]) == 0
        out = capsys.readouterr().out
        assert "✅" in out
        assert VERDICT_VR180 in out

    def test_exit_nonzero_on_fail(self, tmp_path, monkeypatch, capsys):
        path = _make_mp4(tmp_path, "flat.mp4")
        _mock_probe(monkeypatch, 1920, 1080)
        assert vr180_qa.main([path]) != 0
        out = capsys.readouterr().out
        assert "❌" in out

    def test_json_output(self, tmp_path, monkeypatch, capsys):
        path = _make_mp4(tmp_path, "vr180.mp4", _build_sv3d(5760, 2880, "sbs") + _build_st3d("sbs"))
        _mock_probe(monkeypatch, 5760, 2880)
        assert vr180_qa.main([path, "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["verdict"] == VERDICT_VR180
        assert payload["width"] == 5760
        assert payload["height"] == 2880
        assert payload["fps"] == pytest.approx(30.0)
        assert isinstance(payload["checks"], list) and payload["checks"]
        assert all({"name", "status", "detail"} <= set(c) for c in payload["checks"])

    def test_human_report_renders_all_checks(self, tmp_path, monkeypatch):
        path = _make_mp4(tmp_path, "vr180.mp4", _build_sv3d(5760, 2880, "sbs") + _build_st3d("sbs"))
        _mock_probe(monkeypatch, 5760, 2880)
        report = vr180_qa.run_qa(path)
        text = vr180_qa.format_human(report)
        for check in report.checks:
            assert check.name in text
        assert "Verdict:" in text
