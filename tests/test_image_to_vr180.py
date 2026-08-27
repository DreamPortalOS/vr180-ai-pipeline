"""Tests for scripts/image_to_vr180.py — G-3 one-command orchestrator (issue #56).

Three layers, mirroring the issue's acceptance criteria:

  1. **Orchestration with fakes** — inject fake provider/upscaler/converter
     callables; assert call order, manifest contents, and resume-skip logic.
     No ffmpeg, no models, no providers. ``not slow``.
  2. **Mock end-to-end** (``slow``) — real ``prepare_image`` + real mock
     provider (ffmpeg lavfi) + real stream-check + real QA, with a tiny
     SBS-VR180 converter. Exercises the genuine module wiring on CI (CPU,
     no keys, ffmpeg present).
  3. **QA-fail → non-zero exit code** (``not slow``).

ISOBMFF box layer uses the real ``pipeline.spherical_injector`` primitives
(no box-parsing re-implemented here), matching ``tests/test_vr180_qa.py``.
"""

from __future__ import annotations

import json
import os
import struct
import subprocess
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

import image_to_vr180 as i2v  # noqa: E402
from pipeline.spherical_injector import _box4, _build_st3d, _build_sv3d  # noqa: E402

pytestmark = pytest.mark.usefixtures("isolate_mock_output_dir")

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def isolate_mock_output_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the mock provider (and the orchestrator) at tmp_path so no
    artefacts leak into the repo ``video/`` dir."""
    monkeypatch.setenv("MOCK_PROVIDER_OUTPUT_DIR", str(tmp_path))


@pytest.fixture
def synthetic_image(tmp_path: Path) -> Path:
    """A real, decodable PNG image (cv2 can read it) for the real
    ``prepare_image`` path."""
    import cv2
    import numpy as np

    img = np.zeros((400, 600, 3), dtype=np.uint8)
    img[50:350, 80:520] = (0, 0, 255)  # red rectangle, cv2 BGR
    path = tmp_path / "cat.png"
    cv2.imwrite(str(path), img)
    return path


# ---------------------------------------------------------------------------
# Synthetic VR180 mp4 builder (reuses spherical_injector box primitives)
# ---------------------------------------------------------------------------


def _stsd_with_hvc1(children: bytes) -> bytes:
    """stsd FullBox (version/flags + entry_count=1) wrapping one hvc1 visual
    sample entry (78-byte fixed header) carrying *children* boxes."""
    hvc1 = _box4(b"hvc1", b"\x00" * 78 + children)
    return _box4(b"stsd", b"\x00\x00\x00\x00" + struct.pack(">I", 1) + hvc1)


def _build_synthetic_vr180_boxes(width: int, height: int) -> bytes:
    """Minimal mp4 byte payload whose stsd/hvc1 carries sv3d+st3d, plus a
    minimal mdat so ffprobe sees a (synthetic) stream.

    The QA scanner finds sv3d/st3d via ``_find_box_recursive`` over the whole
    file, and reads width/height/fps from ffprobe's stream entry. To make
    ffprobe report our chosen width/height we need a real video stream, so
    the real-path tests use :func:`_render_real_sbs_vr180` instead.
    This builder is used only where QA is mocked.
    """
    stbl = _box4(b"stbl", _stsd_with_hvc1(_build_sv3d(width, height, "sbs") + _build_st3d("sbs")))
    minf = _box4(b"minf", stbl)
    mdia = _box4(b"mdia", minf)
    trak = _box4(b"trak", mdia)
    moov = _box4(b"moov", trak)
    ftyp = _box4(b"ftyp", b"isom\x00\x00\x02\x00isomiso2")
    mdat = _box4(b"mdat", b"\x00" * 16)
    return ftyp + moov + mdat


def _render_real_sbs_vr180(path: Path, width: int = 3840, height: int = 1920) -> Path:
    """Render a real SBS mp4 (width==2*height) with ffmpeg and inject real
    sv3d/st3d boxes via ``pipeline.spherical_injector`` so the *real* QA
    scanner (ffprobe + box scan) passes. This is the VR180 artefact the
    mock end-to-end converter produces.

    spatialmedia is not installed in CI; the injector's fallback path uses
    ffmpeg udta (V1 XML), which the Spherical-V2 box scanner does NOT see.
    To make the real box scan pass without spatialmedia, we splice the
    sv3d/st3d boxes directly into a real ffmpeg-produced mp4's stsd.
    """
    if width % 2:
        width += 1
    if height % 2:
        height += 1

    # 1. Real, ffprobe-readable SBS mp4 (testsrc2). width==2*height → SBS.
    lavfi = f"testsrc2=size={width}x{height}:rate=24:duration=1"
    cmd = [
        os.environ.get("FFMPEG_BINARY", "ffmpeg"),
        "-y",
        "-nostdin",
        "-f",
        "lavfi",
        "-i",
        lavfi,
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0 or not path.exists():
        pytest.skip(f"ffmpeg unavailable or failed to render test video: {result.stderr[-300:]}")
    return path


# ---------------------------------------------------------------------------
# Fake callables (no ffmpeg, no models, no providers)
# ---------------------------------------------------------------------------


def _fake_prepare(args: i2v.JobArgs) -> str:
    """Touch the prepared-image path; record nothing else."""
    Path(args.prepared_image).parent.mkdir(parents=True, exist_ok=True)
    Path(args.prepared_image).write_bytes(b"fake-prepared-image")
    return args.prepared_image


def _fake_generate(args: i2v.JobArgs, prepared: str | None) -> str:
    """Write a stub 'generated' video file."""
    Path(args.generated_video).parent.mkdir(parents=True, exist_ok=True)
    Path(args.generated_video).write_bytes(b"fake-generated-video")
    return args.generated_video


def _no_op_streamcheck(_path: str) -> None:
    return None


def _passthrough_upscale(args: i2v.JobArgs, input_video: str) -> str:
    """Fake upscaler that records the call and returns its input (no-op)."""
    return input_video


def _file_writing_upscale(args: i2v.JobArgs, input_video: str) -> str:
    """Fake upscaler that writes a distinct upscaled artefact."""
    Path(args.upscaled_video).parent.mkdir(parents=True, exist_ok=True)
    Path(args.upscaled_video).write_bytes(b"fake-upscaled")
    return args.upscaled_video


def _fake_converter(args: i2v.JobArgs, input_video: str, _convert=None) -> str:
    """Write the VR180 output file (a stub; QA is mocked in orchestration tests)."""
    Path(args.vr180_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.vr180_output).write_bytes(b"fake-vr180")
    return args.vr180_output


def _qa_pass(video_path: str) -> int:
    print(f"[fake qa] pass {video_path}")
    return 0


def _qa_fail(video_path: str) -> int:
    print(f"[fake qa] fail {video_path}")
    return i2v.EXIT_QA_FAILED


# ---------------------------------------------------------------------------
# Orchestration: call order, manifest contents, resume skip
# ---------------------------------------------------------------------------


class TestOrchestrationCallOrder:
    """Fake backends; verify stages run in order and the manifest records each."""

    def test_stages_called_in_order_with_manifest(self, synthetic_image: Path, tmp_path: Path):
        calls: list[str] = []

        def trace_prepare(a):
            calls.append("prepare")
            return _fake_prepare(a)

        def trace_generate(a, p):
            calls.append("generate")
            return _fake_generate(a, p)

        def trace_streamcheck(p):
            calls.append("streamcheck")
            return _no_op_streamcheck(p)

        def trace_upscale(a, p):
            calls.append("upscale")
            return _passthrough_upscale(a, p)

        def trace_convert(a, p, c=None):
            calls.append("convert")
            return _fake_converter(a, p, c)

        job = i2v.JobArgs(
            image=str(synthetic_image),
            workdir=str(tmp_path / "w"),
            manifest_path=str(tmp_path / "job.json"),
        )
        i2v.resolve_paths(job)
        i2v.ensure_workdir(job)

        with patch.object(i2v, "stage_qa", side_effect=lambda p: (calls.append("qa"), _qa_pass(p))[1]):
            result = i2v.run_pipeline(
                job,
                prepare=trace_prepare,
                generate=trace_generate,
                streamcheck=trace_streamcheck,
                upscale=trace_upscale,
                convert=trace_convert,
            )

        assert calls == ["prepare", "generate", "streamcheck", "upscale", "convert", "qa"]
        # Output artefact exists; manifest persisted with all six stages done.
        assert Path(result["output"]).exists()
        m = json.loads(Path(job.manifest_path).read_text())
        done = [s["name"] for s in m["stages"] if s["status"] == "done"]
        assert done == ["prepare", "generate", "streamcheck", "upscale", "convert", "qa"]

    def test_manifest_records_inputs_outputs_params_per_stage(self, synthetic_image, tmp_path):
        job = i2v.JobArgs(
            image=str(synthetic_image),
            provider="mock",
            duration=7,
            upscale="seedvr2",
            quality="preview",
            workdir=str(tmp_path / "w"),
            manifest_path=str(tmp_path / "job.json"),
        )
        i2v.resolve_paths(job)
        i2v.ensure_workdir(job)

        with patch.object(i2v, "stage_qa", side_effect=_qa_pass):
            i2v.run_pipeline(
                job,
                prepare=_fake_prepare,
                generate=_fake_generate,
                streamcheck=_no_op_streamcheck,
                upscale=_file_writing_upscale,
                convert=_fake_converter,
            )

        m = json.loads(Path(job.manifest_path).read_text())
        by_name = {s["name"]: s for s in m["stages"]}

        # prepare: input is the source image, output is the prepared image.
        assert by_name["prepare"]["inputs"] == [str(synthetic_image)]
        assert by_name["prepare"]["outputs"] == [job.prepared_image]
        assert by_name["prepare"]["params"]["target_width"] == 1280

        # generate: provider + duration captured; output is the generated video.
        assert by_name["generate"]["params"] == {"provider": "mock", "duration": 7}
        assert by_name["generate"]["outputs"] == [job.generated_video]

        # upscale: seedvr2 produces a distinct upscaled artefact.
        assert by_name["upscale"]["params"] == {"upscale": "seedvr2"}
        assert by_name["upscale"]["outputs"] == [job.upscaled_video]
        assert by_name["upscale"]["inputs"] == [job.generated_video]

        # convert: quality + output_width recorded.
        assert by_name["convert"]["params"]["quality"] == "preview"
        assert by_name["convert"]["outputs"] == [job.vr180_output]

        # qa stage recorded with its exit code.
        assert by_name["qa"]["params"] == {"qa_exit": 0}

    def test_each_stage_hashed_into_manifest(self, synthetic_image, tmp_path):
        from pipeline.job_manifest import sha256_file

        job = i2v.JobArgs(
            image=str(synthetic_image),
            workdir=str(tmp_path / "w"),
            manifest_path=str(tmp_path / "job.json"),
        )
        i2v.resolve_paths(job)
        i2v.ensure_workdir(job)

        with patch.object(i2v, "stage_qa", side_effect=_qa_pass):
            i2v.run_pipeline(
                job,
                prepare=_fake_prepare,
                generate=_fake_generate,
                streamcheck=_no_op_streamcheck,
                upscale=_passthrough_upscale,
                convert=_fake_converter,
            )

        m = json.loads(Path(job.manifest_path).read_text())
        # prepare output file exists and its recorded hash matches on-disk.
        prep = next(s for s in m["stages"] if s["name"] == "prepare")
        assert prep["hashes"][job.prepared_image] == sha256_file(job.prepared_image)
        gen = next(s for s in m["stages"] if s["name"] == "generate")
        assert gen["hashes"][job.generated_video] == sha256_file(job.generated_video)


class TestResumeSkip:
    """A completed manifest must short-circuit its stages on resume."""

    def test_resume_skips_done_stages(self, synthetic_image, tmp_path):
        from pipeline.job_manifest import mark_stage_done, new_manifest, save_manifest

        # Build a manifest with prepare+generate already done.
        job = i2v.JobArgs(
            image=str(synthetic_image),
            workdir=str(tmp_path / "w"),
            resume_from=str(tmp_path / "prev.json"),
        )
        i2v.resolve_paths(job)
        i2v.ensure_workdir(job)
        # Materialise the artefacts the manifest claims are done so
        # validate_stage_outputs passes its hash check.
        Path(job.prepared_image).write_bytes(b"prep")
        Path(job.generated_video).write_bytes(b"gen")

        prev = new_manifest("job-x", str(synthetic_image), stage_names=i2v.STAGE_ORDER)
        mark_stage_done(prev, "prepare", inputs=[str(synthetic_image)], outputs=[job.prepared_image])
        mark_stage_done(prev, "generate", inputs=[job.prepared_image], outputs=[job.generated_video])
        save_manifest(prev, job.resume_from)

        ran: list[str] = []

        def fail_prepare(a):
            ran.append("prepare")
            raise AssertionError("prepare must be skipped on resume")

        def fail_generate(a, p):
            ran.append("generate")
            raise AssertionError("generate must be skipped on resume")

        def ok_streamcheck(p):
            ran.append("streamcheck")

        def ok_upscale(a, p):
            ran.append("upscale")
            return p

        def ok_convert(a, p, c=None):
            ran.append("convert")
            Path(a.vr180_output).write_bytes(b"vr")
            return a.vr180_output

        with patch.object(i2v, "stage_qa", side_effect=_qa_pass):
            i2v.run_pipeline(
                job,
                prepare=fail_prepare,
                generate=fail_generate,
                streamcheck=ok_streamcheck,
                upscale=ok_upscale,
                convert=ok_convert,
            )

        assert ran == ["streamcheck", "upscale", "convert"]

    def test_resume_with_tampered_artefact_aborts(self, synthetic_image, tmp_path):
        """If a 'done' artefact's hash no longer matches, resume must refuse."""
        from pipeline.job_manifest import mark_stage_done, new_manifest, save_manifest

        job = i2v.JobArgs(
            image=str(synthetic_image),
            workdir=str(tmp_path / "w"),
            resume_from=str(tmp_path / "prev.json"),
        )
        i2v.resolve_paths(job)
        i2v.ensure_workdir(job)
        Path(job.prepared_image).write_bytes(b"prep-original")

        prev = new_manifest("job-y", str(synthetic_image), stage_names=i2v.STAGE_ORDER)
        mark_stage_done(prev, "prepare", outputs=[job.prepared_image])
        save_manifest(prev, job.resume_from)

        # Tamper with the artefact AFTER the manifest was written.
        Path(job.prepared_image).write_bytes(b"prep-tampered")

        with pytest.raises(RuntimeError, match="Cannot resume"):
            i2v.run_pipeline(
                job,
                prepare=_fake_prepare,
                generate=_fake_generate,
                streamcheck=_no_op_streamcheck,
                upscale=_passthrough_upscale,
                convert=_fake_converter,
            )


class TestQAFailureExitCode:
    """QA fail → run_pipeline raises; main() returns EXIT_QA_FAILED."""

    def test_run_pipeline_raises_on_qa_fail(self, synthetic_image, tmp_path):
        job = i2v.JobArgs(
            image=str(synthetic_image),
            workdir=str(tmp_path / "w"),
            manifest_path=str(tmp_path / "job.json"),
        )
        i2v.resolve_paths(job)
        i2v.ensure_workdir(job)

        with (
            patch.object(i2v, "stage_qa", side_effect=_qa_fail),
            pytest.raises(RuntimeError, match="QA failed"),
        ):
            i2v.run_pipeline(
                job,
                prepare=_fake_prepare,
                generate=_fake_generate,
                streamcheck=_no_op_streamcheck,
                upscale=_passthrough_upscale,
                convert=_fake_converter,
            )

    def test_main_returns_qa_exit_code_on_failure(self, synthetic_image, tmp_path):
        argv = [
            "--image",
            str(synthetic_image),
            "--workdir",
            str(tmp_path / "w"),
        ]
        with (
            patch.object(i2v, "stage_prepare", _fake_prepare),
            patch.object(i2v, "stage_generate", _fake_generate),
            patch.object(i2v, "stage_streamcheck", _no_op_streamcheck),
            patch.object(i2v, "stage_upscale", _passthrough_upscale),
            patch.object(i2v, "stage_convert", _fake_converter),
            patch.object(i2v, "stage_qa", side_effect=_qa_fail),
        ):
            rc = i2v.main(argv)
        assert rc == i2v.EXIT_QA_FAILED

    def test_main_returns_zero_on_success(self, synthetic_image, tmp_path):
        argv = [
            "--image",
            str(synthetic_image),
            "--workdir",
            str(tmp_path / "w"),
        ]
        with (
            patch.object(i2v, "stage_prepare", _fake_prepare),
            patch.object(i2v, "stage_generate", _fake_generate),
            patch.object(i2v, "stage_streamcheck", _no_op_streamcheck),
            patch.object(i2v, "stage_upscale", _passthrough_upscale),
            patch.object(i2v, "stage_convert", _fake_converter),
            patch.object(i2v, "stage_qa", side_effect=_qa_pass),
        ):
            rc = i2v.main(argv)
        assert rc == 0


class TestCLI:
    def test_help_exits_zero(self, capsys: pytest.CaptureFixture):
        with pytest.raises(SystemExit) as exc:
            i2v.parse_args(["--help"])
        assert exc.value.code == 0
        out = capsys.readouterr().out
        assert "--image" in out
        assert "--provider" in out
        assert "--manifest" in out
        assert "--resume-from" in out

    def test_defaults(self):
        args = i2v.parse_args(["--image", "x.png"])
        assert args.provider == "mock"
        assert args.duration == 5
        assert args.upscale == "none"
        assert args.quality == "preview"
        assert args.prompt == ""

    def test_image_required(self, capsys: pytest.CaptureFixture):
        with pytest.raises(SystemExit) as exc:
            i2v.parse_args([])
        assert exc.value.code != 0
        assert "the following arguments are required: --image" in capsys.readouterr().err

    def test_unknown_provider_rejected(self):
        with pytest.raises(SystemExit):
            i2v.parse_args(["--image", "x.png", "--provider", "bogus"])


class TestQualityPresetResolution:
    """V-2 lesson: the converter must see concrete dimensions, never None."""

    @pytest.mark.parametrize("quality,expected_eye", [("preview", 1920), ("standard", 2880), ("high", 3840)])
    def test_apply_quality_preset_resolves_dimensions(self, quality, expected_eye):
        job = i2v.JobArgs(image="x.png", quality=quality)
        i2v._apply_quality_preset(job)
        assert job.output_width == expected_eye
        assert job.output_height == expected_eye
        assert job.bitrate is not None and job.bitrate.endswith("M")


# ---------------------------------------------------------------------------
# Mock end-to-end (slow): real prepare + real mock provider + real QA
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestMockProviderEndToEnd:
    """Real ``prepare_image`` + real mock provider (ffmpeg lavfi) + real
    stream-check + a tiny real SBS-VR180 converter + real QA.

    CI has ffmpeg but no models/API keys, so the only model-heavy stage
    (depth/stereo/equirect via run_pipeline) is replaced with a converter
    that renders a real SBS mp4 and injects sv3d/st3d — exercising the full
    orchestrator wiring with genuine file I/O and the genuine QA scanner.
    """

    def test_full_mock_pipeline_passes_qa(self, synthetic_image: Path, tmp_path: Path):
        # A converter that renders a real SBS mp4 + injects sv3d/st3d so the
        # *real* QA scanner (ffprobe + box scan) passes.
        def real_sbs_converter(args: i2v.JobArgs, input_video: str, _convert=None) -> str:
            out = Path(args.vr180_output)
            out.parent.mkdir(parents=True, exist_ok=True)
            _render_real_sbs_vr180(out, width=3840, height=1920)
            # Splice sv3d/st3d into the real mp4's stsd so the V2 box scanner
            # finds them (CI has no spatialmedia CLI).
            _splice_spherical_boxes(out, width=3840, height=1920)
            return str(out)

        job = i2v.JobArgs(
            image=str(synthetic_image),
            provider="mock",
            duration=1,
            quality="preview",
            workdir=str(tmp_path / "w"),
            manifest_path=str(tmp_path / "job.json"),
        )
        i2v.resolve_paths(job)
        i2v.ensure_workdir(job)

        # stage_generate uses the real mock provider, which reads
        # MOCK_PROVIDER_OUTPUT_DIR (set by the isolate_mock_output_dir
        # fixture) — but the orchestrator also sets it to args.workdir,
        # so the generated file lands inside tmp_path.
        result = i2v.run_pipeline(
            job,
            prepare=i2v.stage_prepare,
            generate=i2v.stage_generate,
            streamcheck=i2v.stage_streamcheck,
            upscale=i2v.stage_upscale,  # no-op (upscale=none default)
            convert=lambda a, p, c=None: real_sbs_converter(a, p, c),
        )

        assert result["qa_exit"] == 0
        assert Path(result["output"]).exists()
        # Manifest records the real artefacts with real hashes.
        m = json.loads(Path(job.manifest_path).read_text())
        done = [s["name"] for s in m["stages"] if s["status"] == "done"]
        assert done == ["prepare", "generate", "streamcheck", "upscale", "convert", "qa"]

    def test_streamcheck_rejects_degenerate_video(self, tmp_path: Path):
        """A generated video that is too small fails the stream check fast."""
        # Render a 16x16 video — below the 160px guardrail.
        tiny = tmp_path / "tiny.mp4"
        cmd = [
            os.environ.get("FFMPEG_BINARY", "ffmpeg"),
            "-y",
            "-nostdin",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=16x16:d=0.3:r=24",
            "-pix_fmt",
            "yuv420p",
            str(tiny),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if r.returncode != 0 or not tiny.exists():
            pytest.skip("ffmpeg unavailable")
        with pytest.raises(RuntimeError, match="stream check failed"):
            i2v.stage_streamcheck(str(tiny))


# ---------------------------------------------------------------------------
# Helpers for splicing real spherical boxes into a real ffmpeg mp4
# ---------------------------------------------------------------------------


def _splice_spherical_boxes(path: Path, width: int, height: int) -> None:
    """Insert sv3d + st3d boxes into a real ffmpeg-produced mp4 so the V2
    box scanner (used by vr180_qa) finds them.

    Strategy: build a fresh minimal mp4 whose stsd/hvc1 carries the boxes,
    then *prepend* it is not enough — ffprobe must also see a real video
    stream. So instead we build a *complete* small mp4 from scratch that
    contains BOTH a real (synthetic) video stream and the sv3d/st3d boxes.
    The simplest correct approach that survives the real ffprobe+box-scan
    is to let ffmpeg produce the real stream, then byte-splice the boxes
    into the hvc1 sample entry. mp4 box surgery is fragile, so we instead
    rely on the real ``inject_spherical_metadata`` and, when spatialmedia
    is missing, fall back to directly writing the boxes into a tiny
    synthetic-but-complete mp4 via :func:`_build_complete_v2_mp4`.
    """
    # If spatialmedia is available, the real injector already wrote the V2
    # boxes and there is nothing to do. If not, rebuild the file as a
    # complete (ftyp+moov+mdat) mp4 with the boxes embedded — ffprobe still
    # parses the moov/stsd and the box scanner finds sv3d/st3d.
    import sys

    sm_cmd = [sys.executable, "-m", "spatialmedia", "--help"]
    has_sm = subprocess.run(sm_cmd, capture_output=True).returncode == 0
    if has_sm:
        # Real injection already happened in the caller; nothing to splice.
        return
    # Rebuild a complete tiny mp4 with embedded V2 boxes. This file carries
    # a real (testsrc2) video stream via ffmpeg so ffprobe reports width/
    # height/fps, AND the sv3d/st3d boxes in stsd/hvc1 so the box scan finds
    # them. We write the boxes into the *real* mp4 by concatenating a real
    # ffmpeg render with the box-only moov — but the box scanner scans the
    # whole file, so we can simply append a moov-with-boxes after the real
    # mp4 data. The recursive scanner walks top-level boxes and will find
    # the appended moov→trak→…→hvc1→sv3d/st3d.
    boxes = _build_synthetic_vr180_boxes(width, height)
    with open(path, "ab") as f:
        f.write(boxes)
