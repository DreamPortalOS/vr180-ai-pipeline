"""Tests for scripts/dome_qa.py — fulldome domemaster acceptance gate.

ffprobe/ffmpeg are mocked via the ``frames`` / ``probe_data`` /
``stereo_boxes`` injection seams, so CI without ffmpeg still exercises the
full gate on synthetic numpy domemaster frames. A real-ffmpeg small-sample
test runs only when ffmpeg is on PATH (tmp_path artefacts only).
"""

from __future__ import annotations

import json
import shutil
import struct
import subprocess

import numpy as np
import pytest
from scripts import dome_qa
from scripts.dome_qa import analyze_frame, run_qa

from pipeline.spherical_injector import _box4, _build_st3d, _build_sv3d

SIZE = 256


def _domemaster(fill_r: float, size: int = SIZE, seed: int = 0) -> np.ndarray:
    """Synthetic domemaster: noise texture out to r=fill_r, black beyond."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(float)
    rr = np.sqrt((xx - (size - 1) / 2.0) ** 2 + (yy - (size - 1) / 2.0) ** 2) / (size / 2.0)
    noise = rng.integers(0, 256, size=(size, size)).astype(np.uint8)
    frame = np.zeros((size, size, 3), dtype=np.uint8)
    frame[rr <= fill_r] = np.stack([noise[rr <= fill_r]] * 3, axis=1)
    return frame


def _half_dome(size: int = SIZE, seed: int = 0) -> np.ndarray:
    """Asymmetric dome: bottom semicircle filled to r=0.95, top only to r=0.6.

    The VR180→dome defect this gate exists to catch — content packed into one
    side and stretched to the rim. One azimuth reaches r≈0.95, but only ~half
    the circumference does, so full-ring-fill coverage must collapse to ~0.6.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(float)
    rr = np.sqrt((xx - (size - 1) / 2.0) ** 2 + (yy - (size - 1) / 2.0) ** 2) / (size / 2.0)
    noise = rng.integers(0, 256, size=(size, size)).astype(np.uint8)
    bottom = yy > (size - 1) / 2.0
    fill = (bottom & (rr <= 0.95)) | (~bottom & (rr <= 0.6))
    frame = np.zeros((size, size, 3), dtype=np.uint8)
    frame[fill] = np.stack([noise[fill]] * 3, axis=1)
    return frame


def _probe_data(size: int = SIZE) -> dict:
    return {
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": size,
                "height": size,
                "avg_frame_rate": "30/1",
                "bit_rate": "45000000",
            }
        ],
        "format": {"bit_rate": "45000000", "duration": "12.5"},
    }


def _status(report, name: str) -> str:
    for check in report.checks:
        if check.name == name:
            return check.status
    raise AssertionError(f"check {name!r} not found")


class TestCoverageRadius:
    """The r≈0.6 regression must FAIL; a full r=0.95 fill must PASS."""

    def test_partial_fill_fails(self):
        got = analyze_frame(_domemaster(0.6))
        assert got["coverage_r"] == pytest.approx(0.64, abs=0.05)

    def test_full_fill_passes(self):
        got = analyze_frame(_domemaster(0.95))
        assert got["coverage_r"] >= 0.9

    def test_gate_distinguishes(self, tmp_path):
        narrow = str(tmp_path / "narrow.mp4")
        Path = tmp_path / "narrow.mp4"
        Path.write_bytes(b"fake")
        bad = run_qa(
            narrow,
            size=SIZE,
            frames=[_domemaster(0.6)],
            probe_data=_probe_data(),
            stereo_boxes=[],
        )
        assert _status(bad, "coverage radius") == "fail"
        assert bad.failed
        assert bad.coverage_r < 0.9
        good = run_qa(
            narrow,
            size=SIZE,
            frames=[_domemaster(0.95)],
            probe_data=_probe_data(),
            stereo_boxes=[],
        )
        assert _status(good, "coverage radius") == "pass"
        assert not good.failed
        assert good.coverage_r >= 0.9

    def test_worst_frame_wins(self, tmp_path):
        path = str(tmp_path / "mix.mp4")
        tmp_path.joinpath("mix.mp4").write_bytes(b"fake")
        report = run_qa(
            path,
            size=SIZE,
            frames=[_domemaster(0.95), _domemaster(0.6)],
            probe_data=_probe_data(),
            stereo_boxes=[],
        )
        assert _status(report, "coverage radius") == "fail"


class TestCircularMask:
    def test_clean_dome_passes(self, tmp_path):
        path = str(tmp_path / "d.mp4")
        tmp_path.joinpath("d.mp4").write_bytes(b"fake")
        report = run_qa(path, size=SIZE, frames=[_domemaster(0.95)], probe_data=_probe_data(), stereo_boxes=[])
        assert _status(report, "circular mask (outside black)") == "pass"
        assert _status(report, "circular mask (inside non-black)") == "pass"

    def test_bright_corners_fail(self, tmp_path):
        frame = _domemaster(0.95)
        yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(float)
        rr = np.sqrt((xx - (SIZE - 1) / 2.0) ** 2 + (yy - (SIZE - 1) / 2.0) ** 2) / (SIZE / 2.0)
        frame[rr > 1.0] = 50  # corners lit well above the black threshold
        path = str(tmp_path / "c.mp4")
        tmp_path.joinpath("c.mp4").write_bytes(b"fake")
        report = run_qa(path, size=SIZE, frames=[frame], probe_data=_probe_data(), stereo_boxes=[])
        assert _status(report, "circular mask (outside black)") == "fail"

    def test_all_black_fails_inside(self, tmp_path):
        path = str(tmp_path / "b.mp4")
        tmp_path.joinpath("b.mp4").write_bytes(b"fake")
        report = run_qa(
            path,
            size=SIZE,
            frames=[np.zeros((SIZE, SIZE, 3), dtype=np.uint8)],
            probe_data=_probe_data(),
            stereo_boxes=[],
        )
        assert _status(report, "circular mask (inside non-black)") == "fail"


class TestResolution:
    def test_square_expected_passes(self, tmp_path):
        path = str(tmp_path / "d.mp4")
        tmp_path.joinpath("d.mp4").write_bytes(b"fake")
        report = run_qa(path, size=SIZE, frames=[_domemaster(0.95)], probe_data=_probe_data(SIZE), stereo_boxes=[])
        assert _status(report, "resolution") == "pass"

    def test_non_square_fails(self, tmp_path):
        path = str(tmp_path / "w.mp4")
        tmp_path.joinpath("w.mp4").write_bytes(b"fake")
        probe = _probe_data()
        probe["streams"][0]["width"] = 512
        probe["streams"][0]["height"] = 256
        report = run_qa(path, size=SIZE, frames=[_domemaster(0.95)], probe_data=probe, stereo_boxes=[])
        assert _status(report, "resolution") == "fail"

    def test_size_override(self, tmp_path):
        path = str(tmp_path / "d.mp4")
        tmp_path.joinpath("d.mp4").write_bytes(b"fake")
        report = run_qa(path, size=512, frames=[_domemaster(0.95)], probe_data=_probe_data(SIZE), stereo_boxes=[])
        assert _status(report, "resolution") == "fail"


class TestMonoMetadata:
    def _mp4(self, tmp_path, name, boxes: bytes = b"") -> str:
        inner = b"\x00\x00\x00\x00" + struct.pack(">I", 1) + _box4(b"hvc1", b"\x00" * 78 + boxes)
        stbl = _box4(b"stbl", _box4(b"stsd", inner))
        moov = _box4(b"moov", _box4(b"trak", _box4(b"mdia", _box4(b"minf", stbl))))
        ftyp = _box4(b"ftyp", b"isom\x00\x00\x02\x00isomiso2")
        path = tmp_path / name
        path.write_bytes(ftyp + moov)
        return str(path)

    def test_stereo_boxes_fail(self, tmp_path):
        path = self._mp4(tmp_path, "s.mp4", _build_sv3d(256, 256, "sbs") + _build_st3d("sbs"))
        found = dome_qa._scan_stereo_boxes(path)
        assert set(found) == {"sv3d", "st3d"}
        report = run_qa(path, size=SIZE, frames=[_domemaster(0.95)], probe_data=_probe_data(), stereo_boxes=found)
        assert _status(report, "mono metadata") == "fail"
        assert report.failed

    def test_no_boxes_pass(self, tmp_path):
        path = self._mp4(tmp_path, "m.mp4")
        boxes = dome_qa._scan_stereo_boxes(path)
        report = run_qa(path, size=SIZE, frames=[_domemaster(0.95)], probe_data=_probe_data(), stereo_boxes=boxes)
        assert _status(report, "mono metadata") == "pass"


class TestZenithReportOnly:
    def test_zenith_never_fails(self, tmp_path):
        path = str(tmp_path / "d.mp4")
        tmp_path.joinpath("d.mp4").write_bytes(b"fake")
        for fill in (0.6, 0.95):
            report = run_qa(path, size=SIZE, frames=[_domemaster(fill)], probe_data=_probe_data(), stereo_boxes=[])
            assert _status(report, "zenith orientation") == "pass"

    def test_zenith_values_reported(self):
        got = analyze_frame(_domemaster(0.95))
        assert got["zenith_center"] >= 0.0
        assert got["zenith_bottom"] >= 0.0


class TestExitCodesAndJson:
    def test_fail_summary(self, tmp_path):
        path = str(tmp_path / "d.mp4")
        tmp_path.joinpath("d.mp4").write_bytes(b"fake")
        report = run_qa(path, size=SIZE, frames=[_domemaster(0.6)], probe_data=_probe_data(), stereo_boxes=[])
        assert report.failed
        assert report.summary["overall"] == "fail"

    def test_json_main_pass(self, tmp_path, capsys, monkeypatch):
        path = str(tmp_path / "d.mp4")
        tmp_path.joinpath("d.mp4").write_bytes(b"fake")
        frame = _domemaster(0.95)
        monkeypatch.setattr(dome_qa, "_probe", lambda p, ffprobe="ffprobe": _probe_data())
        monkeypatch.setattr(dome_qa, "sample_frames", lambda p, n_frames=5, size=256, ffmpeg="ffmpeg": [frame])
        monkeypatch.setattr(dome_qa, "_scan_stereo_boxes", lambda p: [])
        assert dome_qa.main([path, "--json", "--size", str(SIZE)]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["summary"]["overall"] == "pass"
        assert payload["coverage_r"] >= 0.9
        assert isinstance(payload["checks"], list) and payload["checks"]

    def test_human_report(self, tmp_path):
        path = str(tmp_path / "d.mp4")
        tmp_path.joinpath("d.mp4").write_bytes(b"fake")
        report = run_qa(path, size=SIZE, frames=[_domemaster(0.95)], probe_data=_probe_data(), stereo_boxes=[])
        text = dome_qa.format_human(report)
        for check in report.checks:
            assert check.name in text
        assert "Verdict:" in text

    def test_missing_file(self):
        assert run_qa("does/not/exist.mp4").failed

    def test_ffprobe_failure(self, tmp_path, monkeypatch):
        path = str(tmp_path / "d.mp4")
        tmp_path.joinpath("d.mp4").write_bytes(b"fake")

        def _boom(p, ffprobe="ffprobe"):
            raise RuntimeError("ffprobe failed (exit 1)")

        monkeypatch.setattr(dome_qa, "_probe", _boom)
        assert run_qa(path, size=SIZE).failed


class TestHalfDomeRegression:
    """The VR180→dome defect: one side packed to the rim, the other short.

    Under the old any-azimuth rule every outer ring had *some* content (the
    filled bottom half) and the disc read as ~0.95 covered. Full-ring-fill
    coverage must instead collapse to the short side (~0.6) and FAIL.
    """

    def test_half_dome_coverage_collapses(self):
        got = analyze_frame(_half_dome())
        # bottom reaches r≈0.95 but only ~half the circumference does
        assert got["coverage_r"] < 0.7
        assert got["coverage_r"] < 0.9

    def test_half_dome_gate_rejects(self, tmp_path):
        path = str(tmp_path / "half.mp4")
        tmp_path.joinpath("half.mp4").write_bytes(b"fake")
        report = run_qa(path, size=SIZE, frames=[_half_dome()], probe_data=_probe_data(), stereo_boxes=[])
        assert _status(report, "coverage radius") == "fail"
        assert report.failed
        assert report.verdict == "FAIL"

    def test_half_dome_worst_frame_wins(self, tmp_path):
        # a clean full dome in the sample must not rescue the half-dome frame
        path = str(tmp_path / "mix.mp4")
        tmp_path.joinpath("mix.mp4").write_bytes(b"fake")
        report = run_qa(
            path,
            size=SIZE,
            frames=[_domemaster(0.95), _half_dome()],
            probe_data=_probe_data(),
            stereo_boxes=[],
        )
        assert _status(report, "coverage radius") == "fail"
        assert report.verdict == "FAIL"


class TestRingProfile:
    """JSON carries a per-ring {r, fill, mean} trace for human review."""

    def test_ring_profile_well_formed(self):
        got = analyze_frame(_domemaster(0.95))
        rp = got["ring_profile"]
        assert len(rp) == dome_qa.NBINS
        assert set(rp[0]) == {"r", "fill", "mean"}
        for entry in rp:
            assert 0.0 < entry["r"] <= 1.0
            assert 0.0 <= entry["fill"] <= 1.0

    def test_ring_profile_traces_fill_edge(self):
        # a 0.6 fill: rings inside are full, rings beyond are empty
        got = analyze_frame(_domemaster(0.6))
        rp = got["ring_profile"]
        full = [e["fill"] for e in rp if e["r"] <= 0.58]
        empty = [e["fill"] for e in rp if e["r"] >= 0.70]
        assert min(full) >= 0.9
        assert max(empty) < 0.1

    def test_ring_profile_in_payload(self, tmp_path):
        path = str(tmp_path / "d.mp4")
        tmp_path.joinpath("d.mp4").write_bytes(b"fake")
        report = run_qa(path, size=SIZE, frames=[_domemaster(0.95)], probe_data=_probe_data(), stereo_boxes=[])
        assert len(report.ring_profile) == dome_qa.NBINS
        assert report.ring_fill == dome_qa.DEFAULT_RING_FILL


class TestVerdictText:
    """Acceptance verdict is a plain PASS/FAIL, not 'domemaster'/'not domemaster'."""

    def test_pass_verdict(self, tmp_path):
        path = str(tmp_path / "d.mp4")
        tmp_path.joinpath("d.mp4").write_bytes(b"fake")
        report = run_qa(path, size=SIZE, frames=[_domemaster(0.95)], probe_data=_probe_data(), stereo_boxes=[])
        assert not report.failed
        assert report.verdict == "PASS"

    def test_fail_verdict(self, tmp_path):
        path = str(tmp_path / "d.mp4")
        tmp_path.joinpath("d.mp4").write_bytes(b"fake")
        report = run_qa(path, size=SIZE, frames=[_domemaster(0.6)], probe_data=_probe_data(), stereo_boxes=[])
        assert report.failed
        assert report.verdict == "FAIL"


class TestSubprocessDiscipline:
    def test_sample_frames_list_argv(self, tmp_path, monkeypatch):
        probe = _probe_data()

        class _R:
            returncode = 0
            stdout = bytes(SIZE * SIZE * 3)

        def _fake_run(cmd, **kwargs):
            assert isinstance(cmd, list)
            assert not kwargs.get("shell")
            return _R()

        monkeypatch.setattr(dome_qa, "_probe", lambda p, ffprobe="ffprobe": probe)
        monkeypatch.setattr(dome_qa.subprocess, "run", _fake_run)
        frames = dome_qa.sample_frames(str(tmp_path / "x.mp4"), n_frames=2, size=SIZE)
        assert len(frames) == 2
        assert all(f.shape == (SIZE, SIZE, 3) for f in frames)


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg/ffprobe unavailable")
def test_real_tiny_dome_sample(tmp_path):
    """End-to-end on a real ffmpeg-encoded domemaster (tmp_path only)."""
    import cv2 as _cv2

    src = tmp_path / "src.png"
    assert _cv2.imwrite(str(src), _domemaster(0.95, size=256))
    mp4 = tmp_path / "dome.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-loop",
            "1",
            "-i",
            str(src),
            "-frames:v",
            "10",
            "-pix_fmt",
            "yuv420p",
            str(mp4),
        ],
        check=True,
        timeout=60,
    )
    report = run_qa(str(mp4), size=256, n_frames=2)
    assert _status(report, "resolution") == "pass"
    assert _status(report, "mono metadata") == "pass"
    assert report.frames_sampled >= 1
