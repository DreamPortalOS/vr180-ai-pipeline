"""Tests for the K-20 ``--strict`` exit-code semantics of scripts/vr180_qa.py
(issue #213).

Background: by default a clean pass and a pass-with-WARN both exit 0, so
scripted gating (batch / CI / stage gates) cannot tell them apart. This
suite asserts:

  * default behavior is UNCHANGED (WARN-only run exits 0)
  * ``--strict`` promotes a WARN-only run to exit code 2
  * FAIL still exits non-zero whether or not ``--strict`` is set
  * a clean pass exits 0 with or without ``--strict``
  * ``--json`` carries a top-level ``summary`` with pass/warn/fail counts and
    an overall verdict

ffprobe is fully mocked; the mp4 box layer uses the same real
spherical_injector primitives as ``test_vr180_qa.py`` to build tiny synthetic
ISOBMFF bytes, so no real video or real ffprobe is invoked.
"""

from __future__ import annotations

import json
import struct

from scripts import vr180_qa

from pipeline.spherical_injector import _box4, _build_st3d, _build_sv3d

# ---------------------------------------------------------------------------
# Synthetic-mp4 + mock-ffprobe helpers (self-contained; mirrors
# test_vr180_qa.py so this file does not depend on test-module internals).
# ---------------------------------------------------------------------------


def _stsd_with_hvc1(children: bytes) -> bytes:
    hvc1 = _box4(b"hvc1", b"\x00" * 78 + children)
    return _box4(b"stsd", b"\x00\x00\x00\x00" + struct.pack(">I", 1) + hvc1)


def _make_mp4(tmp_path, name: str, boxes: bytes = b"") -> str:
    stbl = _box4(b"stbl", _stsd_with_hvc1(boxes))
    minf = _box4(b"minf", stbl)
    mdia = _box4(b"mdia", minf)
    trak = _box4(b"trak", mdia)
    moov = _box4(b"moov", trak)
    ftyp = _box4(b"ftyp", b"isom\x00\x00\x02\x00isomiso2")
    path = tmp_path / name
    path.write_bytes(ftyp + moov)
    return str(path)


def _probe_json(width: int, height: int, fps: str = "30/1", bitrate: str = "45000000", audio: bool = False) -> dict:
    streams = [
        {
            "codec_type": "video",
            "codec_name": "h264",
            "width": width,
            "height": height,
            "avg_frame_rate": fps,
            "bit_rate": bitrate,
        }
    ]
    if audio:
        streams.append(
            {
                "codec_type": "audio",
                "codec_name": "aac",
                "bit_rate": "160000",
                "sample_rate": "48000",
            }
        )
    return {"streams": streams, "format": {"bit_rate": bitrate, "duration": "12.5"}}


def _mock_probe(monkeypatch, width: int, height: int, fps: str = "30/1", audio: bool = False) -> None:
    monkeypatch.setattr(
        vr180_qa, "_probe", lambda path, ffprobe="ffprobe": _probe_json(width, height, fps, audio=audio)
    )


def _status(report, name: str) -> str:
    for check in report.checks:
        if check.name == name:
            return check.status
    raise AssertionError(f"check {name!r} not found in report")


# ---------------------------------------------------------------------------
# Fixtures: build the three canonical scenarios once.
# ---------------------------------------------------------------------------


def _make_clean_pass(tmp_path, monkeypatch) -> str:
    """5760×2880 SBS with sv3d+st3d left-right + audio → every check passes."""
    path = _make_mp4(
        tmp_path,
        "clean.mp4",
        _build_sv3d(5760, 2880, "sbs") + _build_st3d("sbs"),
    )
    _mock_probe(monkeypatch, 5760, 2880, audio=True)
    return path


def _make_warn_only(tmp_path, monkeypatch) -> str:
    """3840×1920 SBS → valid VR180 but per-eye 1920px < 2880px → WARN, no FAIL.

    (Uses sv3d+st3d left-right so the only non-pass check is per-eye resolution
    plus the audio-absent WARN — both WARNs, no FAIL.)
    """
    path = _make_mp4(
        tmp_path,
        "warn.mp4",
        _build_sv3d(3840, 1920, "sbs") + _build_st3d("sbs"),
    )
    _mock_probe(monkeypatch, 3840, 1920)
    return path


def _make_fail(tmp_path, monkeypatch) -> str:
    """1920×1080 flat, no sv3d/st3d → FAIL (no spherical boxes + bad layout)."""
    path = _make_mp4(tmp_path, "flat.mp4")
    _mock_probe(monkeypatch, 1920, 1080)
    return path


# ---------------------------------------------------------------------------
# Exit-code matrix (the heart of K-20).
# ---------------------------------------------------------------------------


class TestStrictExitCodes:
    def test_clean_pass_exits_0_without_strict(self, tmp_path, monkeypatch):
        path = _make_clean_pass(tmp_path, monkeypatch)
        assert vr180_qa.main([path]) == 0

    def test_clean_pass_exits_0_with_strict(self, tmp_path, monkeypatch):
        path = _make_clean_pass(tmp_path, monkeypatch)
        assert vr180_qa.main([path, "--strict"]) == 0

    def test_warn_only_exits_0_without_strict(self, tmp_path, monkeypatch):
        """Regression: default behavior is UNCHANGED — WARN does not fail."""
        path = _make_warn_only(tmp_path, monkeypatch)
        assert vr180_qa.main([path]) == 0

    def test_warn_only_exits_2_with_strict(self, tmp_path, monkeypatch):
        """--strict promotes a WARN-only run to exit code 2 (not 1)."""
        path = _make_warn_only(tmp_path, monkeypatch)
        assert vr180_qa.main([path, "--strict"]) == 2

    def test_fail_exits_nonzero_without_strict(self, tmp_path, monkeypatch):
        """FAIL exits non-zero regardless of --strict (default behavior)."""
        path = _make_fail(tmp_path, monkeypatch)
        assert vr180_qa.main([path]) != 0
        assert vr180_qa.main([path]) == 1  # FAIL specifically maps to 1

    def test_fail_exits_nonzero_with_strict(self, tmp_path, monkeypatch):
        """--strict does not downgrade a FAIL; exit is still 1, not 2."""
        path = _make_fail(tmp_path, monkeypatch)
        assert vr180_qa.main([path, "--strict"]) != 0
        assert vr180_qa.main([path, "--strict"]) == 1


# ---------------------------------------------------------------------------
# Scenario sanity: confirm the fixtures actually produce WARN (not FAIL).
# ---------------------------------------------------------------------------


class TestScenarioSanity:
    def test_warn_only_fixture_is_warn_not_fail(self, tmp_path, monkeypatch):
        path = _make_warn_only(tmp_path, monkeypatch)
        report = vr180_qa.run_qa(path)
        assert not report.failed  # no FAIL
        assert report.warned  # at least one WARN
        assert _status(report, "per-eye resolution") == "warn"

    def test_fail_fixture_actually_fails(self, tmp_path, monkeypatch):
        path = _make_fail(tmp_path, monkeypatch)
        report = vr180_qa.run_qa(path)
        assert report.failed


# ---------------------------------------------------------------------------
# --json summary field (K-20 #213 requirement).
# ---------------------------------------------------------------------------


class TestJsonSummary:
    def test_json_summary_clean_pass(self, tmp_path, monkeypatch, capsys):
        path = _make_clean_pass(tmp_path, monkeypatch)
        assert vr180_qa.main([path, "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        summary = payload["summary"]
        assert set(summary) == {"pass", "warn", "fail", "overall"}
        assert summary["fail"] == 0
        assert summary["warn"] == 0
        assert summary["pass"] > 0
        assert summary["overall"] == "pass"

    def test_json_summary_warn_only(self, tmp_path, monkeypatch, capsys):
        path = _make_warn_only(tmp_path, monkeypatch)
        assert vr180_qa.main([path, "--json"]) == 0  # WARN-only, no --strict → 0
        payload = json.loads(capsys.readouterr().out)
        summary = payload["summary"]
        assert summary["fail"] == 0
        assert summary["warn"] > 0
        assert summary["overall"] == "warn"

    def test_json_summary_fail(self, tmp_path, monkeypatch, capsys):
        path = _make_fail(tmp_path, monkeypatch)
        assert vr180_qa.main([path, "--json"]) != 0
        payload = json.loads(capsys.readouterr().out)
        summary = payload["summary"]
        assert summary["fail"] > 0
        assert summary["overall"] == "fail"

    def test_json_summary_counts_match_checks(self, tmp_path, monkeypatch, capsys):
        """summary counts must equal the actual per-status check counts."""
        path = _make_warn_only(tmp_path, monkeypatch)
        vr180_qa.main([path, "--json"])
        payload = json.loads(capsys.readouterr().out)
        statuses = [c["status"] for c in payload["checks"]]
        summary = payload["summary"]
        assert summary["pass"] == statuses.count("pass")
        assert summary["warn"] == statuses.count("warn")
        assert summary["fail"] == statuses.count("fail")


# ---------------------------------------------------------------------------
# Report-level helpers (warned / summary) used by main().
# ---------------------------------------------------------------------------


class TestReportSummaryHelpers:
    def test_report_summary_overall_pass(self, tmp_path, monkeypatch):
        path = _make_clean_pass(tmp_path, monkeypatch)
        report = vr180_qa.run_qa(path)
        assert report.summary["overall"] == "pass"

    def test_report_summary_overall_warn(self, tmp_path, monkeypatch):
        path = _make_warn_only(tmp_path, monkeypatch)
        report = vr180_qa.run_qa(path)
        assert report.summary["overall"] == "warn"
        assert report.warned is True
        assert report.failed is False

    def test_report_summary_overall_fail(self, tmp_path, monkeypatch):
        path = _make_fail(tmp_path, monkeypatch)
        report = vr180_qa.run_qa(path)
        assert report.summary["overall"] == "fail"
        assert report.failed is True
