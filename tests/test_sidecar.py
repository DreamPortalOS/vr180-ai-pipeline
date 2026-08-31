"""Tests for pipeline.sidecar (D-1: sidecar JSON metadata).

ffprobe / ffmpeg are mocked so every test runs on CPU-only CI with no
real media files. The mp4 box layer uses real
:mod:`pipeline.spherical_injector` primitives so sv3d/st3d detection
runs against genuine ISOBMFF bytes rather than a stub.
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

import pipeline
from pipeline import sidecar
from pipeline.sidecar import (
    IMMERSIVE_REQUIRED_FIELDS,
    PROJECTION_EQUIRECT180,
    PROJECTION_EQUIRECT360,
    PROJECTION_FISHEYE_DOME,
    PROJECTIONS,
    ImmersiveError,
    SidecarError,
    build_sidecar,
    normalize_immersive,
    write_sidecar,
)
from pipeline.spherical_injector import (
    _box4,
    _build_st3d,
    _build_sv3d,
)


def _sv3d_st3d_for(w: int, h: int) -> bytes:
    return _build_sv3d(w, h, "sbs") + _build_st3d("sbs")


# ---------------------------------------------------------------------------
# Synthetic mp4 fixtures
# ---------------------------------------------------------------------------


def _stsd_with_hvc1(children: bytes) -> bytes:
    """stsd FullBox wrapping one hvc1 visual sample entry that carries *children*."""
    hvc1 = _box4(b"hvc1", b"\x00" * 78 + children)
    return _box4(b"stsd", b"\x00\x00\x00\x00" + struct.pack(">I", 1) + hvc1)


def _make_mp4(tmp_path: Path, name: str, boxes: bytes = b"") -> Path:
    """Build a minimal synthetic mp4 with optional sv3d/st3d boxes."""
    stbl = _box4(b"stbl", _stsd_with_hvc1(boxes))
    minf = _box4(b"minf", stbl)
    mdia = _box4(b"mdia", minf)
    trak = _box4(b"trak", mdia)
    moov = _box4(b"moov", trak)
    ftyp = _box4(b"ftyp", b"isom\x00\x00\x02\x00isomiso2")
    path = tmp_path / name
    path.write_bytes(ftyp + moov)
    return path


def _probe_json(
    width: int = 5760,
    height: int = 2880,
    fps: str = "30/1",
    codec: str = "hevc",
    bitrate: str = "45000000",
    pix_fmt: str = "yuv420p",
    duration: str = "12.5",
    audio: dict | None = None,
) -> dict:
    streams = [
        {
            "codec_type": "video",
            "codec_name": codec,
            "width": width,
            "height": height,
            "avg_frame_rate": fps,
            "bit_rate": bitrate,
            "pix_fmt": pix_fmt,
        }
    ]
    if audio is not None:
        streams.append(audio)
    return {
        "streams": streams,
        "format": {
            "bit_rate": bitrate,
            "duration": duration,
            "size": "1024",
        },
    }


def _mock_ffprobe(monkeypatch, probe_result: dict) -> None:
    monkeypatch.setattr(sidecar, "_probe_json", lambda path, ffprobe="ffprobe": probe_result)


def _mock_run_qa(monkeypatch, *, passed: bool = True) -> None:
    qa_mock = SimpleNamespace(
        failed=(not passed),
        verdict="VR180 (180° 3D SBS)" if passed else "plain 2D",
        checks=[],
    )
    if passed:
        from scripts.vr180_qa import Check

        qa_mock.checks.append(Check("stream info", "pass", "5760x2880 hevc 30 fps"))
        qa_mock.checks.append(Check("sv3d/st3d boxes", "pass", "sv3d + st3d present"))
        qa_mock.checks.append(Check("stereo mode", "pass", "left-right SBS"))
        qa_mock.checks.append(Check("SBS square-eye layout", "pass", "5760x2880"))
    monkeypatch.setattr(
        sidecar,
        "_run_qa",
        lambda path, ffprobe: {
            "passed": passed,
            "verdict": qa_mock.verdict,
            "checks": {c.name: {"status": c.status, "detail": c.detail} for c in qa_mock.checks},
        },
    )


# ---------------------------------------------------------------------------
# Sidecar builder — structural tests
# ---------------------------------------------------------------------------


class TestBuildSidecar:
    def test_full_vr180_record(self, tmp_path, monkeypatch):
        mp4 = _make_mp4(tmp_path, "scene01.mp4", _sv3d_st3d_for(5760, 2880))
        monkeypatch.setattr(sidecar, "_probe_json", lambda path, ffprobe="ffprobe": _probe_json())
        _mock_run_qa(monkeypatch, passed=True)

        record = build_sidecar(
            mp4,
            immersive={
                "projection": "equirect",
                "fov_deg": 180,
                "stereo_layout": "side_by_side",
                "eye_resolution": [2880, 2880],
                "spatial_metadata": ["sv3d", "st3d"],
            },
            generation={
                "route": "vr180",
                "i2v_backend": "seedance",
                "prompt": "dolly in",
                "seed": 42,
                "source_image": "/abs/cat.png",
            },
        )

        # Top-level identity
        assert record["name"] == "scene01.mp4"
        assert record["bytes"] == mp4.stat().st_size
        expected_sha = hashlib.sha256(mp4.read_bytes()).hexdigest()
        assert record["sha256"] == expected_sha
        assert record["duration_sec"] == 12.5

        # Video block
        v = record["video"]
        assert v == {
            "codec": "hevc",
            "width": 5760,
            "height": 2880,
            "fps": 30.0,
            "bitrate_bps": 45_000_000,
            "pix_fmt": "yuv420p",
        }

        # Caller-supplied immersive block is honored verbatim
        assert record["immersive"]["projection"] == "equirect"
        assert record["immersive"]["stereo_layout"] == "side_by_side"
        assert record["immersive"]["eye_resolution"] == [2880, 2880]

        # Generation: pipeline_version injected by default
        assert record["generation"]["route"] == "vr180"
        assert record["generation"]["i2v_backend"] == "seedance"
        assert record["generation"]["pipeline_version"] == pipeline.__version__

        # QA
        assert record["qa"]["passed"] is True
        assert "stream info" in record["qa"]["checks"]

    def test_silent_video_has_no_audio_block(self, tmp_path, monkeypatch):
        mp4 = _make_mp4(tmp_path, "silent.mp4", _sv3d_st3d_for(5760, 2880))
        _mock_ffprobe(monkeypatch, _probe_json())
        _mock_run_qa(monkeypatch, passed=True)

        record = build_sidecar(mp4)
        assert record["audio"] is None

    def test_audio_stream_fields(self, tmp_path, monkeypatch):
        mp4 = _make_mp4(tmp_path, "with_audio.mp4", _sv3d_st3d_for(5760, 2880))
        _mock_ffprobe(
            monkeypatch,
            _probe_json(
                audio={
                    "codec_type": "audio",
                    "codec_name": "aac",
                    "channels": 2,
                    "sample_rate": "48000",
                }
            ),
        )
        _mock_run_qa(monkeypatch, passed=True)

        record = build_sidecar(mp4)
        assert record["audio"] == {
            "codec": "aac",
            "channels": 2,
            "sample_rate": 48000,
        }

    def test_ffprobe_failure_raises_sidecar_error(self, tmp_path, monkeypatch):
        mp4 = _make_mp4(tmp_path, "broken.mp4")

        def _boom(path, ffprobe="ffprobe"):
            raise RuntimeError("probe failed")

        monkeypatch.setattr(sidecar, "_probe_json", _boom)
        with pytest.raises(SidecarError, match="ffprobe failed"):
            build_sidecar(mp4)

    def test_missing_file_raises_sidecar_error(self, monkeypatch):
        with pytest.raises(SidecarError, match="Video file not found"):
            build_sidecar("/no/such/file.mp4")

    def test_generation_defaults_to_version_only(self, tmp_path, monkeypatch):
        mp4 = _make_mp4(tmp_path, "gen_default.mp4", _sv3d_st3d_for(5760, 2880))
        _mock_ffprobe(monkeypatch, _probe_json())
        _mock_run_qa(monkeypatch, passed=True)

        record = build_sidecar(mp4)
        assert record["generation"] == {"pipeline_version": pipeline.__version__}

    def test_immersive_defaults_from_boxes_vr180(self, tmp_path, monkeypatch):
        mp4 = _make_mp4(tmp_path, "auto_vr180.mp4", _sv3d_st3d_for(5760, 2880))
        _mock_ffprobe(monkeypatch, _probe_json())
        _mock_run_qa(monkeypatch, passed=True)

        record = build_sidecar(mp4)
        assert record["immersive"]["projection"] == "equirect"
        assert record["immersive"]["stereo_layout"] == "side_by_side"
        assert "sv3d" in record["immersive"]["spatial_metadata"]
        assert "st3d" in record["immersive"]["spatial_metadata"]

    def test_immersive_defaults_to_fulldome_without_boxes(self, tmp_path, monkeypatch):
        mp4 = _make_mp4(tmp_path, "auto_dome.mp4")  # no sv3d/st3d
        _mock_ffprobe(monkeypatch, _probe_json(width=4096, height=4096))
        _mock_run_qa(monkeypatch, passed=True)

        record = build_sidecar(mp4)
        assert record["immersive"]["projection"] == "fisheye_domemaster"
        assert record["immersive"]["stereo_layout"] == "mono"
        assert record["immersive"]["spatial_metadata"] == []

    def test_fulldome_default_includes_eye_resolution(self, tmp_path, monkeypatch):
        """D-3: the default fulldome immersive block MUST carry eye_resolution
        (the dome frame size), not just VR180."""
        mp4 = _make_mp4(tmp_path, "auto_dome.mp4")  # no sv3d/st3d
        _mock_ffprobe(monkeypatch, _probe_json(width=4096, height=4096))
        _mock_run_qa(monkeypatch, passed=True)

        record = build_sidecar(mp4)
        assert record["immersive"]["eye_resolution"] == [4096, 4096]

    def test_vr180_default_includes_halved_eye_resolution(self, tmp_path, monkeypatch):
        """D-3: VR180 SBS default infers eye resolution by halving the SBS
        frame width (one eye = half the frame)."""
        mp4 = _make_mp4(tmp_path, "auto_vr180.mp4", _sv3d_st3d_for(5760, 2880))
        _mock_ffprobe(monkeypatch, _probe_json(width=5760, height=2880))
        _mock_run_qa(monkeypatch, passed=True)

        record = build_sidecar(mp4)
        assert record["immersive"]["eye_resolution"] == [2880, 2880]


# ---------------------------------------------------------------------------
# D-3: projection contract — normalize_immersive + required fields
# ---------------------------------------------------------------------------


class TestNormalizeImmersive:
    """Issue #80: the immersive block is the DreamPortal EPanoProjection
    contract anchor.  ``projection`` / ``fov_deg`` / ``stereo_layout`` /
    ``eye_resolution`` are all required and validated."""

    def test_canonical_vr180_block_passes(self):
        block = normalize_immersive(
            {
                "projection": "equirect",
                "fov_deg": 180,
                "stereo_layout": "side_by_side",
                "eye_resolution": [2880, 2880],
                "spatial_metadata": ["sv3d", "st3d"],
            }
        )
        assert block["projection"] == PROJECTION_EQUIRECT180
        assert block["fov_deg"] == 180
        assert block["eye_resolution"] == [2880, 2880]

    def test_canonical_fulldome_block_passes(self):
        block = normalize_immersive(
            {
                "projection": "fisheye_domemaster",
                "fov_deg": 180,
                "stereo_layout": "mono",
                "eye_resolution": [4096, 4096],
            }
        )
        assert block["projection"] == PROJECTION_FISHEYE_DOME

    def test_360_fov_passes(self):
        """Future 360 support uses the same field with fov_deg=360."""
        block = normalize_immersive(
            {
                "projection": "equirect360",
                "fov_deg": 360,
                "stereo_layout": "mono",
                "eye_resolution": [4096, 2048],
            }
        )
        assert block["fov_deg"] == 360
        assert block["projection"] == PROJECTION_EQUIRECT360

    def test_fov_kept_int_when_integer_valued(self):
        block = normalize_immersive(
            {"projection": "equirect", "fov_deg": 180, "stereo_layout": "side_by_side", "eye_resolution": [1, 1]}
        )
        assert isinstance(block["fov_deg"], int)
        assert block["fov_deg"] == 180

    def test_fov_kept_float_when_fractional(self):
        block = normalize_immersive(
            {"projection": "equirect", "fov_deg": 220.5, "stereo_layout": "mono", "eye_resolution": [1, 1]}
        )
        assert block["fov_deg"] == 220.5

    def test_missing_projection_raises(self):
        with pytest.raises(ImmersiveError, match="projection"):
            normalize_immersive({"fov_deg": 180, "stereo_layout": "mono", "eye_resolution": [1, 1]})

    def test_missing_fov_raises(self):
        with pytest.raises(ImmersiveError, match="fov_deg"):
            normalize_immersive({"projection": "equirect", "stereo_layout": "mono", "eye_resolution": [1, 1]})

    def test_missing_stereo_layout_raises(self):
        with pytest.raises(ImmersiveError, match="stereo_layout"):
            normalize_immersive({"projection": "equirect", "fov_deg": 180, "eye_resolution": [1, 1]})

    def test_missing_eye_resolution_raises(self):
        """The headline D-3 requirement: eye_resolution is now required."""
        with pytest.raises(ImmersiveError, match="eye_resolution"):
            normalize_immersive({"projection": "equirect", "fov_deg": 180, "stereo_layout": "side_by_side"})

    @pytest.mark.parametrize("bad_fov", [0, -1, 361, 400])
    def test_fov_out_of_range_raises(self, bad_fov):
        with pytest.raises(ImmersiveError, match="fov_deg"):
            normalize_immersive(
                {"projection": "equirect", "fov_deg": bad_fov, "stereo_layout": "mono", "eye_resolution": [1, 1]}
            )

    def test_unknown_stereo_layout_raises(self):
        with pytest.raises(ImmersiveError, match="stereo_layout"):
            normalize_immersive(
                {"projection": "equirect", "fov_deg": 180, "stereo_layout": "over_under", "eye_resolution": [1, 1]}
            )

    def test_unknown_projection_passes_with_warning(self, caplog):
        """Unknown projections pass through (extensibility for future
        projection modes the contract hasn't enumerated yet) but log a
        warning so typos surface."""
        with caplog.at_level("WARNING", logger="sidecar"):
            block = normalize_immersive(
                {
                    "projection": "equirect180_over_under",
                    "fov_deg": 180,
                    "stereo_layout": "mono",
                    "eye_resolution": [1, 1],
                }
            )
        assert block["projection"] == "equirect180_over_under"
        assert "not in" in caplog.text

    def test_eye_resolution_coerced_to_int_list(self):
        block = normalize_immersive(
            {
                "projection": "equirect",
                "fov_deg": 180,
                "stereo_layout": "side_by_side",
                "eye_resolution": ("2880", "2880"),
            }
        )
        assert block["eye_resolution"] == [2880, 2880]

    def test_eye_resolution_malformed_raises(self):
        with pytest.raises(ImmersiveError, match="eye_resolution"):
            normalize_immersive(
                {"projection": "equirect", "fov_deg": 180, "stereo_layout": "mono", "eye_resolution": [1]}
            )

    def test_input_not_mutated(self):
        src = {"projection": "equirect", "fov_deg": 180, "stereo_layout": "side_by_side", "eye_resolution": [1, 1]}
        normalize_immersive(src)
        # Caller's dict is untouched.
        assert src == {
            "projection": "equirect",
            "fov_deg": 180,
            "stereo_layout": "side_by_side",
            "eye_resolution": [1, 1],
        }

    def test_required_fields_constant_matches_contract(self):
        """The required-field tuple is the public contract surface; lock it."""
        assert IMMERSIVE_REQUIRED_FIELDS == ("projection", "fov_deg", "stereo_layout", "eye_resolution")

    def test_projections_constant_has_all_three(self):
        """The three EPanoProjection enum mirrors are present and ordered."""
        assert PROJECTIONS == ("equirect360", "equirect", "fisheye_domemaster")


class TestBuildSidecarEnforcesContract:
    """build_sidecar normalises caller-immersive through the D-3 contract,
    so a caller block missing eye_resolution raises (rather than silently
    writing a non-compliant sidecar)."""

    def test_caller_immersive_missing_eye_resolution_raises(self, tmp_path, monkeypatch):
        mp4 = _make_mp4(tmp_path, "bad.mp4", _sv3d_st3d_for(5760, 2880))
        _mock_ffprobe(monkeypatch, _probe_json())
        _mock_run_qa(monkeypatch, passed=True)

        with pytest.raises(ImmersiveError, match="eye_resolution"):
            build_sidecar(
                mp4,
                immersive={"projection": "equirect", "fov_deg": 180, "stereo_layout": "side_by_side"},
            )


class TestWriteSidecar:
    def test_writes_sibling_json(self, tmp_path, monkeypatch):
        mp4 = _make_mp4(tmp_path, "out.mp4", _sv3d_st3d_for(5760, 2880))
        _mock_ffprobe(monkeypatch, _probe_json())
        _mock_run_qa(monkeypatch, passed=True)

        out = write_sidecar(mp4)
        assert out == mp4.parent / "out.json"
        assert out.is_file()
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["name"] == "out.mp4"
        assert "sha256" in data

    def test_out_dir_redirects_json(self, tmp_path, monkeypatch):
        mp4_dir = tmp_path / "videos"
        mp4_dir.mkdir()
        mp4 = _make_mp4(mp4_dir, "movie.mp4")
        target_dir = tmp_path / "manifests"

        _mock_ffprobe(monkeypatch, _probe_json())
        _mock_run_qa(monkeypatch, passed=True)

        out = write_sidecar(mp4, out_dir=target_dir)
        assert out == target_dir / "movie.json"
        assert out.is_file()

    def test_creates_parent_dir(self, tmp_path, monkeypatch):
        mp4_dir = tmp_path / "videos"
        mp4_dir.mkdir()
        mp4 = _make_mp4(mp4_dir, "deep.mp4")
        target_dir = tmp_path / "new" / "dir" / "manifests"

        _mock_ffprobe(monkeypatch, _probe_json())
        _mock_run_qa(monkeypatch, passed=True)

        out = write_sidecar(mp4, out_dir=target_dir)
        assert target_dir.exists()
        assert out == target_dir / "deep.json"

    def test_json_is_pretty_printed_utf8(self, tmp_path, monkeypatch):
        mp4 = _make_mp4(tmp_path, "non_ascii.mp4", _sv3d_st3d_for(5760, 2880))
        _mock_ffprobe(monkeypatch, _probe_json())
        _mock_run_qa(monkeypatch, passed=True)

        out = write_sidecar(mp4, generation={"note": "包含中文字段值"})
        text = out.read_text(encoding="utf-8")
        # Round-trips and preserves non-ASCII values
        json.loads(text)
        assert "\n  " in text  # pretty-printed with indentation
        assert "包含中文字段值" in text


# ---------------------------------------------------------------------------
# Deterministic round-trip against a real mp4 fixture + real sha256
# ---------------------------------------------------------------------------


def test_sha256_matches_file(tmp_path, monkeypatch):
    mp4 = _make_mp4(tmp_path, "hash_test.mp4", _sv3d_st3d_for(5760, 2880))
    _mock_ffprobe(monkeypatch, _probe_json())
    _mock_run_qa(monkeypatch, passed=True)

    out = write_sidecar(mp4)
    data = json.loads(out.read_text(encoding="utf-8"))
    actual = hashlib.sha256(mp4.read_bytes()).hexdigest()
    assert data["sha256"] == actual


def test_qa_failure_survives_and_marks_unpassed(tmp_path, monkeypatch):
    """Even when QA fails, the sidecar is written with passed=false."""
    mp4 = _make_mp4(tmp_path, "qa_fail.mp4", _sv3d_st3d_for(5760, 2880))
    _mock_ffprobe(monkeypatch, _probe_json())
    _mock_run_qa(monkeypatch, passed=False)

    record = build_sidecar(mp4)
    assert record["qa"]["passed"] is False
    assert record["qa"]["verdict"] == "plain 2D"


def test_immersive_and_generation_override_default(tmp_path, monkeypatch):
    mp4 = _make_mp4(tmp_path, "override.mp4")  # no boxes → default would be fisheye
    _mock_ffprobe(monkeypatch, _probe_json(width=4096, height=4096))
    _mock_run_qa(monkeypatch, passed=True)

    record = build_sidecar(
        mp4,
        immersive={
            "projection": "equirect",
            "fov_deg": 180,
            "stereo_layout": "side_by_side",
            "eye_resolution": [2048, 2048],
        },
        generation={"route": "vr180"},
    )
    assert record["immersive"]["projection"] == "equirect"
    assert record["generation"]["route"] == "vr180"
    assert record["generation"]["pipeline_version"] == pipeline.__version__


# ---------------------------------------------------------------------------
# Integration: orchestrator + run_pipeline wiring
# ---------------------------------------------------------------------------


class TestOrchestratorWiring:
    """End-to-end smoke tests that the I2V orchestrator actually writes a
    sidecar alongside the output artefact. Uses fully-faked stages so we
    exercise only the orchestration + sidecar wiring, not any real model
    inference or ffmpeg conversion.
    """

    def test_run_pipeline_writes_sidecar_next_to_output(self, tmp_path, monkeypatch):
        from scripts import image_to_vr180 as i2v

        out_dir = tmp_path / "out"
        out_dir.mkdir()
        mp4 = out_dir / "final_vr180.mp4"
        mp4.write_bytes(_make_mp4(out_dir, "final_vr180.mp4", _sv3d_st3d_for(5760, 2880)).read_bytes())

        # Stub every stage to be a no-op returning the pre-made mp4.
        monkeypatch.setattr(i2v, "stage_prepare", lambda args: str(tmp_path / "prep.png"))
        monkeypatch.setattr(i2v, "stage_generate", lambda args, img=None: str(mp4))
        monkeypatch.setattr(i2v, "stage_streamcheck", lambda path: None)
        monkeypatch.setattr(i2v, "stage_upscale", lambda args, inp: inp)
        monkeypatch.setattr(
            i2v,
            "stage_convert",
            lambda args, inp, convert: str(mp4),
        )
        monkeypatch.setattr(i2v, "stage_audio", lambda args, vr180, gen: {"copied": False, "codec": None, "source": ""})
        monkeypatch.setattr(i2v, "stage_qa", lambda path: 0)

        # Feed ffprobe the same synthetic probe the sidecar tests use.
        monkeypatch.setattr(sidecar, "_probe_json", lambda path, ffprobe="ffprobe": _probe_json())
        _mock_run_qa(monkeypatch, passed=True)

        args = i2v.JobArgs(image=str(tmp_path / "img.png"), workdir=str(tmp_path), provider="mock")
        i2v.resolve_paths(args)
        # Point vr180_output at the pre-made mp4.
        args.vr180_output = str(mp4)
        args.output_width = 2880
        args.output_height = 2880

        result = i2v.run_pipeline(args)
        assert result["output"] == str(mp4)

        sidecar_path = mp4.parent / f"{mp4.stem}.json"
        assert sidecar_path.is_file(), "orchestrator must write a sidecar JSON next to the output"
        data = json.loads(sidecar_path.read_text(encoding="utf-8"))
        assert data["name"] == mp4.name
        assert data["immersive"]["projection"] == "equirect"
        assert data["immersive"]["stereo_layout"] == "side_by_side"
        assert data["immersive"]["eye_resolution"] == [2880, 2880]
        assert data["generation"]["route"] == "vr180"
        assert data["generation"]["i2v_backend"] == "mock"
        assert data["qa"]["passed"] is True
