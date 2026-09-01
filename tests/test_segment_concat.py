"""Tests for pipeline/segment_concat.py — lossless / xfade concat (issue #173, C-1).

All ffmpeg / ffprobe calls are mocked via an injected ``runner`` /
``patch.object(..., "subprocess")`` so these run on CI (CPU-only, no models,
no network) in the ``not slow`` suite. No real video files are produced.
"""

from __future__ import annotations

import json
from contextlib import suppress
from pathlib import Path
from unittest.mock import patch

import pytest

from pipeline import segment_concat as sc
from pipeline.segment_concat import (
    ConcatError,
    ConcatSegment,
    check_compatible,
    concat_segments,
    probe_segment,
)

# --------------------------------------------------------------------------- #
# Probe payload builders
# --------------------------------------------------------------------------- #


def _probe_streams(
    width=1280,
    height=720,
    fps="30/1",
    duration="5.0",
    codec="h264",
    has_audio=True,
) -> str:
    streams = [
        {
            "codec_type": "video",
            "codec_name": codec,
            "width": width,
            "height": height,
            "r_frame_rate": fps,
        }
    ]
    if has_audio:
        streams.append({"codec_type": "audio", "codec_name": "aac", "sample_rate": "48000"})
    return json.dumps({"streams": streams, "format": {"duration": duration}})


def _probe_result(width=1280, height=720, fps=30.0, duration=5.0, has_audio=True):
    return {
        "width": width,
        "height": height,
        "fps": fps,
        "duration": duration,
        "has_audio": has_audio,
    }


def _fake_probe_run(stdout: str, returncode: int = 0, stderr: str = ""):
    """Build a fake ``subprocess.run`` returning *stdout* as ffprobe JSON."""
    return lambda cmd, **kw: type("r", (), {"returncode": returncode, "stdout": stdout, "stderr": stderr})()


def _fake_runner(returncode: int = 0):
    """A fake runner that records the cmd it was given and writes the output."""
    calls: list[list[str]] = []

    def _run(cmd, **kwargs):
        calls.append(cmd)
        # Simulate ffmpeg writing the output file.
        Path(cmd[-1]).write_bytes(b"concatenated")
        return type("r", (), {"returncode": returncode, "stdout": "", "stderr": ""})()

    _run.calls = calls  # type: ignore[attr-defined]
    return _run


# --------------------------------------------------------------------------- #
# ConcatSegment dataclass
# --------------------------------------------------------------------------- #


class TestConcatSegment:
    def test_path_coerced_to_path(self, tmp_path: Path):
        seg = ConcatSegment(path=str(tmp_path / "a.mp4"))
        assert isinstance(seg.path, Path)

    def test_negative_start_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="start must be >= 0"):
            ConcatSegment(path=tmp_path / "a.mp4", start=-1)

    def test_end_before_start_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="end must be > start"):
            ConcatSegment(path=tmp_path / "a.mp4", start=2.0, end=1.0)

    def test_frozen(self, tmp_path: Path):
        import dataclasses

        seg = ConcatSegment(path=tmp_path / "a.mp4")
        with pytest.raises(dataclasses.FrozenInstanceError):
            seg.path = tmp_path / "b.mp4"  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# probe_segment
# --------------------------------------------------------------------------- #


class TestProbeSegment:
    def test_returns_width_height_fps_duration_has_audio(self):
        with patch.object(sc, "subprocess") as sp:
            sp.run = _fake_probe_run(_probe_streams())
            info = probe_segment(Path("a.mp4"))
            assert info == _probe_result()

    def test_video_only_has_audio_false(self):
        with patch.object(sc, "subprocess") as sp:
            sp.run = _fake_probe_run(_probe_streams(has_audio=False))
            info = probe_segment(Path("a.mp4"))
            assert info["has_audio"] is False

    def test_fps_fraction_parsed(self):
        # 30000/1001 ≈ 29.97
        with patch.object(sc, "subprocess") as sp:
            sp.run = _fake_probe_run(_probe_streams(fps="30000/1001", duration="10.0"))
            info = probe_segment(Path("a.mp4"))
            assert abs(info["fps"] - 29.97) < 0.01

    def test_no_video_stream_raises(self):
        payload = json.dumps({"streams": [{"codec_type": "audio"}], "format": {}})
        with patch.object(sc, "subprocess") as sp:
            sp.run = _fake_probe_run(payload)
            with pytest.raises(ConcatError, match="no video stream"):
                probe_segment(Path("a.mp4"))

    def test_ffprobe_failure_raises(self):
        with patch.object(sc, "subprocess") as sp:
            sp.run = _fake_probe_run("", returncode=1, stderr="boom")
            with pytest.raises(ConcatError, match="ffprobe failed"):
                probe_segment(Path("a.mp4"))

    def test_missing_width_raises(self):
        payload = json.dumps(
            {
                "streams": [{"codec_type": "video", "codec_name": "h264", "r_frame_rate": "30/1"}],
                "format": {"duration": "5.0"},
            }
        )
        with patch.object(sc, "subprocess") as sp:
            sp.run = _fake_probe_run(payload)
            with pytest.raises(ConcatError, match="could not determine width"):
                probe_segment(Path("a.mp4"))

    def test_cmd_is_list_not_shell(self):
        captured: list[list[str]] = []

        def _run(cmd, **kwargs):
            captured.append(cmd)
            assert kwargs.get("shell") is not True
            return type("r", (), {"returncode": 0, "stdout": _probe_streams(), "stderr": ""})()

        with patch.object(sc, "subprocess") as sp:
            sp.run = _run
            probe_segment(Path("a.mp4"))
        assert captured
        assert isinstance(captured[0], list)


# --------------------------------------------------------------------------- #
# check_compatible
# --------------------------------------------------------------------------- #


class TestCheckCompatible:
    def test_compatible_passes(self):
        segs = [ConcatSegment(Path(f"{i}.mp4")) for i in range(3)]
        with patch.object(sc, "subprocess") as sp:
            sp.run = _fake_probe_run(_probe_streams())
            # Should not raise.
            check_compatible(segs)

    def test_resolution_mismatch_raises_with_values(self):
        segs = [ConcatSegment(Path("a.mp4")), ConcatSegment(Path("b.mp4"))]

        def _run(cmd, **kw):
            # First probe → 1280x720, second → 1920x1080.
            idx = _run.calls  # type: ignore[attr-defined]
            _run.calls = idx + 1  # type: ignore[attr-defined]
            stdout = _probe_streams(width=1280, height=720) if idx == 0 else _probe_streams(width=1920, height=1080)
            return type("r", (), {"returncode": 0, "stdout": stdout, "stderr": ""})()

        _run.calls = 0  # type: ignore[attr-defined]
        with patch.object(sc, "subprocess") as sp:
            sp.run = _run
            with pytest.raises(ConcatError) as exc:
                check_compatible(segs)
            msg = str(exc.value)
            # Message must list actual values for both the reference and the
            # offending segment.
            assert "1920x1080" in msg
            assert "1280x720" in msg

    def test_fps_mismatch_raises_with_values(self):
        segs = [ConcatSegment(Path("a.mp4")), ConcatSegment(Path("b.mp4"))]

        def _run(cmd, **kw):
            idx = _run.calls  # type: ignore[attr-defined]
            _run.calls = idx + 1  # type: ignore[attr-defined]
            stdout = _probe_streams(fps="30/1") if idx == 0 else _probe_streams(fps="25/1")
            return type("r", (), {"returncode": 0, "stdout": stdout, "stderr": ""})()

        _run.calls = 0  # type: ignore[attr-defined]
        with patch.object(sc, "subprocess") as sp:
            sp.run = _run
            with pytest.raises(ConcatError) as exc:
                check_compatible(segs)
            msg = str(exc.value)
            assert "25" in msg and "30" in msg

    def test_empty_segments_raises(self):
        with pytest.raises(ConcatError, match="at least one segment"):
            check_compatible([])


# --------------------------------------------------------------------------- #
# concat_segments — demux mode
# --------------------------------------------------------------------------- #


class TestConcatDemux:
    def _segs(self, tmp_path: Path, n=2):
        segs = []
        for i in range(n):
            p = tmp_path / f"seg{i}.mp4"
            p.write_bytes(b"fake")
            segs.append(ConcatSegment(p))
        return segs

    def test_command_has_concat_and_c_copy(self, tmp_path: Path):
        segs = self._segs(tmp_path)
        out = tmp_path / "out.mp4"
        runner = _fake_runner()

        with patch.object(sc, "subprocess") as sp:
            sp.run = _fake_probe_run(_probe_streams())
            concat_segments(segs, out, mode="demux", runner=runner)

        cmd = runner.calls[0]
        assert "-f" in cmd and "concat" in cmd
        assert "-c" in cmd and "copy" in cmd

    def test_command_has_no_reencode_params(self, tmp_path: Path):
        """demux must NOT carry any libx264/libx265/CRF/encoder args."""
        segs = self._segs(tmp_path)
        out = tmp_path / "out.mp4"
        runner = _fake_runner()

        with patch.object(sc, "subprocess") as sp:
            sp.run = _fake_probe_run(_probe_streams())
            concat_segments(segs, out, mode="demux", runner=runner)

        cmd = runner.calls[0]
        reencode_markers = ["libx264", "libx265", "-crf", "-b:v", "hevc_nvenc", "h264_nvenc"]
        for marker in reencode_markers:
            assert marker not in cmd, f"demux cmd must not contain {marker}: {cmd}"

    def test_list_file_in_temp_not_repo(self, tmp_path: Path, monkeypatch):
        """The concat list file must live in the system temp dir, not the repo."""
        segs = self._segs(tmp_path)
        out = tmp_path / "out.mp4"
        runner = _fake_runner()

        # Capture the list file path by recording it from a patched writer.
        captured: list[Path] = []
        original_write = sc._write_concat_list

        def _spy(segments, list_path):
            captured.append(Path(list_path))
            return original_write(segments, list_path)

        monkeypatch.setattr(sc, "_write_concat_list", _spy)

        with patch.object(sc, "subprocess") as sp:
            sp.run = _fake_probe_run(_probe_streams())
            concat_segments(segs, out, mode="demux", runner=runner)

        assert captured, "concat list was never written"
        list_path = captured[0]
        repo_root = Path.cwd().resolve()
        list_resolved = list_path.resolve()
        # The list file must NOT be inside the repo working tree.
        assert not str(list_resolved).startswith(str(repo_root)), (
            f"concat list leaked into repo: {list_resolved} (repo root {repo_root})"
        )
        # And not under the local video/ asset dir either.
        assert "video" not in list_resolved.parts

    def test_list_file_references_segments(self, tmp_path: Path):
        segs = self._segs(tmp_path)
        out = tmp_path / "out.mp4"
        runner = _fake_runner()

        # The list file lives in a temp dir that is cleaned up once
        # concat_segments returns, so spy on the writer to capture its content
        # before the TemporaryDirectory context exits.
        captured: list[str] = []
        original_write = sc._write_concat_list

        def _spy(segments, list_path):
            original_write(segments, list_path)
            captured.append(list_path.read_text(encoding="utf-8"))

        with patch.object(sc, "_write_concat_list", _spy), patch.object(sc, "subprocess") as sp:
            sp.run = _fake_probe_run(_probe_streams())
            concat_segments(segs, out, mode="demux", runner=runner)

        content = captured[0]
        for seg in segs:
            assert str(seg.path) in content or seg.path.name in content

    def test_runner_failure_raises(self, tmp_path: Path):
        segs = self._segs(tmp_path)
        out = tmp_path / "out.mp4"

        def _bad(cmd, **kw):
            return type("r", (), {"returncode": 1, "stdout": "", "stderr": "bad"})()

        with patch.object(sc, "subprocess") as sp:
            sp.run = _fake_probe_run(_probe_streams())
            with pytest.raises(ConcatError, match="concat demux failed"):
                concat_segments(segs, out, mode="demux", runner=_bad)

    def test_list_file_uses_absolute_paths(self, tmp_path: Path):
        """Every ``file`` entry written into the concat list must be absolute.

        Regression for issue #180 (K-10): the concat list file lives in the
        system temp dir, so ffmpeg's concat demuxer resolves any relative
        entry inside it against the temp dir, not the caller's cwd. A
        caller-relative path like ``.tmp_concat/a.mp4`` then becomes
        ``TMPDIR/.tmp_concat/a.mp4`` and silently fails. The list writer
        MUST absolutize each segment path before writing it.
        """
        segs = self._segs(tmp_path)
        out = tmp_path / "out.mp4"
        runner = _fake_runner()

        captured: list[str] = []
        original_write = sc._write_concat_list

        def _spy(segments, list_path):
            original_write(segments, list_path)
            captured.append(list_path.read_text(encoding="utf-8"))

        with patch.object(sc, "_write_concat_list", _spy), patch.object(sc, "subprocess") as sp:
            sp.run = _fake_probe_run(_probe_streams())
            concat_segments(segs, out, mode="demux", runner=runner)

        content = captured[0]
        for line in content.splitlines():
            if not line.startswith("file"):
                continue
            # Strip the ``file '...'`` wrapper.
            path_tok = line.split(maxsplit=1)[1].strip().strip("'")
            assert Path(path_tok).is_absolute(), (
                f"concat list contains a relative path '{path_tok}' — "
                "demuxer will resolve it against TMPDIR: line={line!r}"
            )
            # Forward slashes throughout — concat demuxer accepts both, but
            # forward slash is the portable cross-platform form.
            assert "\\" not in path_tok, f"concat list path still uses backslashes: {path_tok!r}"

    def test_relative_path_segments_are_absolutized_in_list(self, tmp_path: Path):
        """Calling with caller-relative segment paths must not leak into the list.

        The entry point accepts relative paths (for usability), but they must
        be resolved before being written to the concat list. This test
        constructs a segment from a pure relative string and verifies the
        written list contains the absolute form, not the original relative one.
        """
        # Pure relative paths — do NOT pass them through tmp_path. These are
        # relative to the caller's cwd, exactly the shape that triggered
        # issue #180's real-run failure. Write them into a cwd subdirectory
        # so .resolve() produces a real, existing file (proving the resolved
        # path ffmpeg would open is valid).
        rel_dir = Path(".tmp_concat").resolve()
        rel_dir.mkdir(parents=True, exist_ok=True)
        (rel_dir / "a.mp4").write_bytes(b"fake")
        (rel_dir / "b.mp4").write_bytes(b"fake")

        rel_segs = [ConcatSegment(Path(".tmp_concat/a.mp4")), ConcatSegment(Path(".tmp_concat/b.mp4"))]
        out = tmp_path / "out.mp4"
        runner = _fake_runner()

        captured: list[str] = []
        original_write = sc._write_concat_list

        def _spy(segments, list_path):
            original_write(segments, list_path)
            captured.append(list_path.read_text(encoding="utf-8"))

        with patch.object(sc, "_write_concat_list", _spy), patch.object(sc, "subprocess") as sp:
            sp.run = _fake_probe_run(_probe_streams())
            concat_segments(rel_segs, out, mode="demux", runner=runner)

        content = captured[0]
        # The original relative tokens must NOT appear in the list.
        assert "file '.tmp_concat/a.mp4'" not in content
        assert "file '.tmp_concat/b.mp4'" not in content
        # Every ``file`` entry must be absolute AND point to an existing file
        # (the demuxer resolves entries against TMPDIR; a real relative path
        # would become TMPDIR/.tmp_concat/a.mp4, which does not exist).
        for line in content.splitlines():
            if line.startswith("file"):
                path_tok = line.split(maxsplit=1)[1].strip().strip("'")
                assert Path(path_tok).is_absolute()
                assert Path(path_tok).exists(), f"resolved concat-list path does not exist: {path_tok}"

        # Clean up the cwd-side helper dir so we don't pollute the working tree.
        (rel_dir / "a.mp4").unlink(missing_ok=True)
        (rel_dir / "b.mp4").unlink(missing_ok=True)
        with suppress(OSError):
            rel_dir.rmdir()

    def test_output_path_is_absolutized(self, tmp_path: Path):
        """The ffmpeg output argument must be absolute.

        Same root cause as PR #75 / issue #180: a relative output would be
        resolved against the cwd that the subprocess happens to run in. After
        the fix, output_path is .resolve()-d before use.
        """
        segs = self._segs(tmp_path)
        # Pass a caller-relative output path.
        out_rel = Path(".tmp_concat/out.mp4")
        out_rel.parent.mkdir(parents=True, exist_ok=True)
        runner = _fake_runner()

        with patch.object(sc, "subprocess") as sp:
            sp.run = _fake_probe_run(_probe_streams())
            concat_segments(segs, out_rel, mode="demux", runner=runner)

        cmd = runner.calls[0]
        output_arg = cmd[-1]
        assert Path(output_arg).is_absolute(), f"output path passed to ffmpeg is not absolute: {output_arg}"


# --------------------------------------------------------------------------- #
# concat_segments — filter / xfade mode
# --------------------------------------------------------------------------- #


class TestConcatFilter:
    def _segs(self, tmp_path: Path, n=2):
        segs = []
        for i in range(n):
            p = tmp_path / f"seg{i}.mp4"
            p.write_bytes(b"fake")
            segs.append(ConcatSegment(p))
        return segs

    def test_crossfade_forces_filter_and_has_xfade(self, tmp_path: Path):
        segs = self._segs(tmp_path)
        out = tmp_path / "out.mp4"
        runner = _fake_runner()

        with patch.object(sc, "subprocess") as sp:
            sp.run = _fake_probe_run(_probe_streams())
            concat_segments(segs, out, mode="demux", crossfade=0.5, runner=runner)

        cmd = runner.calls[0]
        joined = " ".join(cmd)
        assert "xfade" in joined
        # Real ffmpeg xfade params, not invented keys.
        assert "transition=fade" in joined
        assert "duration=0.5" in joined
        assert "offset=" in joined
        # Audio crossfade too, since both segments have audio.
        assert "acrossfade" in joined
        # demux flags must NOT appear — we switched to filter.
        assert not ("-c" in cmd and "copy" in cmd and "-f" in cmd and "concat" in cmd)

    def test_crossfade_chained_three_segments(self, tmp_path: Path):
        """n=3 daisy-chains xfade: [0][1]xfade→[xv1]; [xv1][2]xfade→[outv]."""
        segs = self._segs(tmp_path, n=3)
        out = tmp_path / "out.mp4"
        runner = _fake_runner()

        with patch.object(sc, "subprocess") as sp:
            sp.run = _fake_probe_run(_probe_streams(duration="5.0"))
            concat_segments(segs, out, mode="demux", crossfade=1.0, runner=runner)

        cmd = runner.calls[0]
        joined = " ".join(cmd)
        assert "xfade" in joined
        # Intermediate label must be referenced twice (produced then consumed).
        assert "[xv1]" in joined
        # Two xfade transitions for three segments.
        assert joined.count("xfade=") == 2

    def test_filter_mode_builds_filter_complex(self, tmp_path: Path):
        segs = self._segs(tmp_path, n=3)
        out = tmp_path / "out.mp4"
        runner = _fake_runner()

        with patch.object(sc, "subprocess") as sp:
            sp.run = _fake_probe_run(_probe_streams())
            concat_segments(segs, out, mode="filter", runner=runner)

        cmd = runner.calls[0]
        assert "-filter_complex" in cmd
        assert "concat" in " ".join(cmd)

    def test_filter_mode_audio_included_when_all_have_audio(self, tmp_path: Path):
        segs = self._segs(tmp_path, n=2)
        out = tmp_path / "out.mp4"
        runner = _fake_runner()

        with patch.object(sc, "subprocess") as sp:
            sp.run = _fake_probe_run(_probe_streams(has_audio=True))
            concat_segments(segs, out, mode="filter", runner=runner)

        cmd = runner.calls[0]
        assert "[outa]" in " ".join(cmd)
        assert "aac" in cmd

    def test_filter_mode_no_audio_when_any_segment_silent(self, tmp_path: Path):
        segs = self._segs(tmp_path, n=2)
        out = tmp_path / "out.mp4"
        runner = _fake_runner()

        # First segment has audio, second does not.
        def _run(cmd, **kw):
            idx = _run.calls  # type: ignore[attr-defined]
            _run.calls = idx + 1  # type: ignore[attr-defined]
            stdout = _probe_streams(has_audio=True) if idx % 2 == 0 else _probe_streams(has_audio=False)
            return type("r", (), {"returncode": 0, "stdout": stdout, "stderr": ""})()

        _run.calls = 0  # type: ignore[attr-defined]
        with patch.object(sc, "subprocess") as sp:
            sp.run = _run
            concat_segments(segs, out, mode="filter", runner=runner)

        cmd = runner.calls[0]
        joined = " ".join(cmd)
        # No audio output mapping, no aac encoder.
        assert "[outa]" not in joined
        assert "aac" not in joined
        # No audio stream mapping either.
        assert "-map" in cmd
        # The only -map present must be the video one.
        assert cmd.count("-map") == 1

    def test_filter_mode_custom_encoder_used(self, tmp_path: Path):
        segs = self._segs(tmp_path, n=2)
        out = tmp_path / "out.mp4"
        runner = _fake_runner()
        custom = ["-c:v", "libx265", "-preset", "fast"]

        with patch.object(sc, "subprocess") as sp:
            sp.run = _fake_probe_run(_probe_streams())
            concat_segments(segs, out, mode="filter", encoder=custom, runner=runner)

        cmd = runner.calls[0]
        for token in custom:
            assert token in cmd

    def test_filter_runner_failure_raises(self, tmp_path: Path):
        segs = self._segs(tmp_path, n=2)
        out = tmp_path / "out.mp4"

        def _bad(cmd, **kw):
            return type("r", (), {"returncode": 2, "stdout": "", "stderr": "bad"})()

        with patch.object(sc, "subprocess") as sp:
            sp.run = _fake_probe_run(_probe_streams())
            with pytest.raises(ConcatError, match="concat filter failed"):
                concat_segments(segs, out, mode="filter", runner=_bad)


# --------------------------------------------------------------------------- #
# Misc
# --------------------------------------------------------------------------- #


class TestConcatMisc:
    def test_unknown_mode_raises(self, tmp_path: Path):
        segs = [ConcatSegment(tmp_path / "a.mp4"), ConcatSegment(tmp_path / "b.mp4")]
        with patch.object(sc, "subprocess") as sp:
            sp.run = _fake_probe_run(_probe_streams())
            with pytest.raises(ConcatError, match="unknown concat mode"):
                concat_segments(segs, tmp_path / "o.mp4", mode="bogus")

    def test_negative_crossfade_raises(self, tmp_path: Path):
        segs = [ConcatSegment(tmp_path / "a.mp4"), ConcatSegment(tmp_path / "b.mp4")]
        with patch.object(sc, "subprocess") as sp:
            sp.run = _fake_probe_run(_probe_streams())
            with pytest.raises(ConcatError, match="crossfade must be >= 0"):
                concat_segments(segs, tmp_path / "o.mp4", crossfade=-0.1)

    def test_returns_output_path(self, tmp_path: Path):
        a = tmp_path / "a.mp4"
        b = tmp_path / "b.mp4"
        a.write_bytes(b"x")
        b.write_bytes(b"y")
        out = tmp_path / "out.mp4"
        runner = _fake_runner()
        with patch.object(sc, "subprocess") as sp:
            sp.run = _fake_probe_run(_probe_streams())
            result = concat_segments([ConcatSegment(a), ConcatSegment(b)], out, runner=runner)
        assert Path(result) == out

    def test_all_runners_use_list_not_shell(self, tmp_path: Path):
        """Every runner invocation must be list-form (CLAUDE.md red line)."""
        segs = [ConcatSegment(tmp_path / "a.mp4"), ConcatSegment(tmp_path / "b.mp4")]
        out = tmp_path / "out.mp4"
        seen_shells: list[object] = []

        def _run(cmd, **kwargs):
            seen_shells.append(kwargs.get("shell"))
            assert isinstance(cmd, list)
            Path(cmd[-1]).write_bytes(b"ok")
            return type("r", (), {"returncode": 0, "stdout": "", "stderr": ""})()

        with patch.object(sc, "subprocess") as sp:
            sp.run = _fake_probe_run(_probe_streams())
            concat_segments(segs, out, mode="demux", runner=_run)
        assert all(s is not True for s in seen_shells)
