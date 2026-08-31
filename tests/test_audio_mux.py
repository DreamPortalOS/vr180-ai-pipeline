"""Tests for pipeline/audio_mux.py — lossless audio passthrough (issue #73, H-1).

All heavy lifting (ffmpeg/ffprobe) is mocked via ``subprocess.run`` patches so
these run on CI (CPU-only, no models, no network) in the ``not slow`` suite.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from pipeline import audio_mux


def _probe_json_audio(codec="aac", bit_rate="158000", sample_rate="48000") -> str:
    return json.dumps(
        {
            "streams": [
                {"codec_type": "video", "codec_name": "h264", "width": 1280, "height": 720},
                {
                    "codec_type": "audio",
                    "codec_name": codec,
                    "bit_rate": str(bit_rate),
                    "sample_rate": str(sample_rate),
                },
            ]
        }
    )


def _probe_json_video_only() -> str:
    return json.dumps({"streams": [{"codec_type": "video", "codec_name": "h264", "width": 1280, "height": 720}]})


class TestHasAudioStream:
    def test_returns_true_when_audio_stream_present(self):
        with patch.object(audio_mux, "subprocess") as sp:
            sp.run.return_value = type("r", (), {"returncode": 0, "stdout": _probe_json_audio()})()
            assert audio_mux.has_audio_stream("video.mp4") is True

    def test_returns_false_when_video_only(self):
        with patch.object(audio_mux, "subprocess") as sp:
            sp.run.return_value = type("r", (), {"returncode": 0, "stdout": _probe_json_video_only()})()
            assert audio_mux.has_audio_stream("video.mp4") is False

    def test_returns_false_on_ffprobe_failure(self):
        with patch.object(audio_mux, "subprocess") as sp:
            sp.run.return_value = type("r", (), {"returncode": 1, "stderr": "not a file"})()
            assert audio_mux.has_audio_stream("bogus.mp4") is False

    def test_uses_list_cmd_not_shell(self):
        """subprocess.run must be called as a list (CLAUDE.md: 禁 shell=True)."""
        with patch.object(audio_mux, "subprocess") as sp:
            sp.run.return_value = type("r", (), {"returncode": 0, "stdout": _probe_json_audio()})()
            audio_mux.has_audio_stream("video.mp4")
            call = sp.run.call_args
            assert call.kwargs.get("shell") is not True, "must not use shell=True"
            cmd = call[0][0] if call[0] else call[1].get("args")
            assert isinstance(cmd, list)
            # Binary may be an absolute path (shutil.which) or bare name;
            # normalise so the check works on both POSIX and Windows.
            import os

            assert os.path.splitext(os.path.basename(cmd[0]))[0].lower() == "ffprobe"

    def test_ffprobe_cmd_contains_required_flags(self):
        with patch.object(audio_mux, "subprocess") as sp:
            sp.run.return_value = type("r", (), {"returncode": 0, "stdout": _probe_json_audio()})()
            audio_mux.has_audio_stream("video.mp4")
            cmd = sp.run.call_args[0][0]
            assert "-print_format" in cmd
            assert "json" in cmd
            assert "-show_streams" in cmd
            assert cmd[-1] == "video.mp4"

    def test_custom_ffprobe_binary(self):
        with patch.object(audio_mux, "subprocess") as sp:
            sp.run.return_value = type("r", (), {"returncode": 0, "stdout": _probe_json_audio()})()
            audio_mux.has_audio_stream("v.mp4", ffprobe="/custom/ffprobe")
            assert sp.run.call_args[0][0][0] == "/custom/ffprobe"


class TestAudioStreamInfo:
    def test_returns_codec_and_bitrate(self):
        with patch.object(audio_mux, "subprocess") as sp:
            sp.run.return_value = type("r", (), {"returncode": 0, "stdout": _probe_json_audio()})()
            info = audio_mux.audio_stream_info("video.mp4")
            assert info is not None
            assert info["codec_name"] == "aac"
            assert info["bit_rate"] == "158000"
            assert info["sample_rate"] == "48000"

    def test_returns_none_for_video_only(self):
        with patch.object(audio_mux, "subprocess") as sp:
            sp.run.return_value = type("r", (), {"returncode": 0, "stdout": _probe_json_video_only()})()
            assert audio_mux.audio_stream_info("video.mp4") is None

    def test_returns_none_on_failure(self):
        with patch.object(audio_mux, "subprocess") as sp:
            sp.run.return_value = type("r", (), {"returncode": 1, "stderr": "fail"})()
            assert audio_mux.audio_stream_info("v.mp4") is None

    def test_opus_codec_passes_through(self):
        with patch.object(audio_mux, "subprocess") as sp:
            sp.run.return_value = type("r", (), {"returncode": 0, "stdout": _probe_json_audio(codec="opus")})()
            info = audio_mux.audio_stream_info("v.mp4")
            assert info["codec_name"] == "opus"


class TestCopyAudioTo:
    """Remux command shape, atomic replace, and failure paths."""

    def _tmp_vr180(self, tmp_path: Path) -> Path:
        p = tmp_path / "vr180.mp4"
        p.write_bytes(b"fake-vr180-video")
        return p

    def _tmp_audio(self, tmp_path: Path) -> Path:
        p = tmp_path / "source.mp4"
        p.write_bytes(b"fake-audio-source")
        return p

    def _mock_run(self, tmp_path: Path):
        """Fake subprocess.run that writes a marker file at the output path,
        simulating ffmpeg's successful remux."""

        def _run(cmd, **kwargs):
            import os

            assert os.path.splitext(os.path.basename(cmd[0]))[0].lower() == "ffmpeg"
            # The last positional arg is the output file.
            out = cmd[-1]
            Path(out).write_bytes(b"remuxed-audio")
            return type("r", (), {"returncode": 0, "stderr": "", "stdout": ""})()

        mock_sp = type("sp", (), {"run": staticmethod(_run)})()
        return patch.object(audio_mux, "subprocess", new=mock_sp)

    def test_command_contains_map_and_c_copy(self, tmp_path):
        vr = self._tmp_vr180(tmp_path)
        aud = self._tmp_audio(tmp_path)
        recorded: list[list[str]] = []

        with patch.object(audio_mux, "subprocess") as sp:

            def _run(cmd, **kwargs):
                recorded.append(cmd)
                Path(cmd[-1]).write_bytes(b"remuxed")
                return type("r", (), {"returncode": 0, "stderr": "", "stdout": ""})()

            sp.run = _run
            audio_mux.copy_audio_to(str(vr), str(aud))

        cmd = recorded[0]
        # cmd[0] may be an absolute path (shutil.which) on Windows; compare the basename.
        import os

        assert os.path.splitext(os.path.basename(cmd[0]))[0].lower() == "ffmpeg"
        # Must contain: -i vr -i aud -map 0:v -map 1:a -c copy
        assert "-map" in cmd and "0:v" in cmd
        assert "-map" in cmd and "1:a" in cmd
        assert "-c" in cmd and "copy" in cmd

    def test_command_includes_shortest_by_default(self, tmp_path):
        vr = self._tmp_vr180(tmp_path)
        aud = self._tmp_audio(tmp_path)
        recorded: list[list[str]] = []

        with patch.object(audio_mux, "subprocess") as sp:

            def _run(cmd, **kwargs):
                recorded.append(cmd)
                Path(cmd[-1]).write_bytes(b"remuxed")
                return type("r", (), {"returncode": 0, "stderr": "", "stdout": ""})()

            sp.run = _run
            audio_mux.copy_audio_to(str(vr), str(aud))

        assert "-shortest" in recorded[0]

    def test_command_omits_shortest_when_disabled(self, tmp_path):
        vr = self._tmp_vr180(tmp_path)
        aud = self._tmp_audio(tmp_path)
        recorded: list[list[str]] = []

        with patch.object(audio_mux, "subprocess") as sp:

            def _run(cmd, **kwargs):
                recorded.append(cmd)
                Path(cmd[-1]).write_bytes(b"remuxed")
                return type("r", (), {"returncode": 0, "stderr": "", "stdout": ""})()

            sp.run = _run
            audio_mux.copy_audio_to(str(vr), str(aud), shortest=False)

        assert "-shortest" not in recorded[0]

    def test_atomic_replace_swaps_the_vr180_file(self, tmp_path):
        vr = self._tmp_vr180(tmp_path)
        aud = self._tmp_audio(tmp_path)

        with self._mock_run(tmp_path):
            out = audio_mux.copy_audio_to(str(vr), str(aud))

        assert out == str(vr)
        # Original vr180 file is now the remuxed content.
        assert vr.read_bytes() == b"remuxed-audio"
        # The .aout.mp4 temp should be gone after atomic replace.
        assert not Path(str(vr) + ".aout.mp4").exists()

    def test_vr180_not_found_raises(self, tmp_path):
        aud = self._tmp_audio(tmp_path)
        with pytest.raises(FileNotFoundError, match="VR180 video not found"):
            audio_mux.copy_audio_to("missing.mp4", str(aud))

    def test_audio_source_not_found_raises(self, tmp_path):
        vr = self._tmp_vr180(tmp_path)
        with pytest.raises(FileNotFoundError, match="Audio source not found"):
            audio_mux.copy_audio_to(str(vr), "missing_source.mp4")

    def test_ffmpeg_failure_raises(self, tmp_path):
        vr = self._tmp_vr180(tmp_path)
        aud = self._tmp_audio(tmp_path)

        with patch.object(audio_mux, "subprocess") as sp:

            def _run(cmd, **kwargs):
                return type("r", (), {"returncode": 2, "stderr": "no stream found"})()

            sp.run = _run
            with pytest.raises(RuntimeError, match="audio remux failed"):
                audio_mux.copy_audio_to(str(vr), str(aud))

    def test_uses_list_cmd_not_shell(self, tmp_path):
        vr = self._tmp_vr180(tmp_path)
        aud = self._tmp_audio(tmp_path)
        with patch.object(audio_mux, "subprocess") as sp:

            def _run(cmd, **kwargs):
                assert kwargs.get("shell") is not True
                assert isinstance(cmd, list)
                Path(cmd[-1]).write_bytes(b"remuxed")
                return type("r", (), {"returncode": 0, "stderr": "", "stdout": ""})()

            sp.run = _run
            audio_mux.copy_audio_to(str(vr), str(aud))

    def test_custom_ffmpeg_binary(self, tmp_path):
        vr = self._tmp_vr180(tmp_path)
        aud = self._tmp_audio(tmp_path)
        with patch.object(audio_mux, "subprocess") as sp:

            def _run(cmd, **kwargs):
                assert cmd[0] == "/opt/ffmpeg"
                Path(cmd[-1]).write_bytes(b"remuxed")
                return type("r", (), {"returncode": 0, "stderr": "", "stdout": ""})()

            sp.run = _run
            audio_mux.copy_audio_to(str(vr), str(aud), ffmpeg="/opt/ffmpeg")
