"""Tests for pipeline/audio_mix.py — external ambience mix-in (issue #396, S-4).

Command construction (``build_audio_mix_filter`` / ``build_audio_mix_command``)
is pure and asserted without any ffmpeg on PATH, so the core runs on CI
(CPU-only, no models, no network) inside the ``not slow`` suite.  The single
real-ffmpeg integration test synthesises a silent clip + 1 s sine tone via
lavfi (no downloads, writes into ``tmp_path``) and is skipped when ffmpeg /
ffprobe are absent.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from pipeline import audio_mix

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")


# ---------------------------------------------------------------------------
# Pure filter construction — the four required combos: gain / loop / fade / dual
# ---------------------------------------------------------------------------


class TestBuildAudioMixFilter:
    def test_gain_only(self):
        fc = audio_mix.build_audio_mix_filter(gain_db=6.0, loop=False, fade_s=0.0)
        assert fc == "[1:a]volume=6dB[aout]"

    def test_loop_with_duration_trims_in_graph(self):
        fc = audio_mix.build_audio_mix_filter(loop=True, fade_s=0.0, duration_s=10.0)
        assert fc == "[1:a]volume=0dB,aloop=loop=-1:size=2e9,atrim=0:10,asetpts=N/SR/TB[aout]"

    def test_loop_without_duration_falls_back_to_finite(self):
        fc = audio_mix.build_audio_mix_filter(loop=True, fade_s=0.0, duration_s=None)
        assert "aloop=loop=99" in fc
        assert "atrim" not in fc

    def test_fade_in_and_out(self):
        fc = audio_mix.build_audio_mix_filter(loop=False, fade_s=2.0)
        assert fc == ("[1:a]volume=0dB,afade=t=in:st=0:d=2,areverse,afade=t=in:st=0:d=2,areverse[aout]")

    def test_dual_track_passes_copy_through(self):
        fc = audio_mix.build_audio_mix_filter(dual=True, fade_s=0.0)
        assert fc == (
            "[1:a]anull[a1];[2:a]volume=0dB[a2];[a1][a2]amix=inputs=2:duration=longest:dropout_transition=0[aout]"
        )

    def test_zero_fade_disables_fades(self):
        fc = audio_mix.build_audio_mix_filter(loop=False, fade_s=0.0)
        assert "afade" not in fc


# ---------------------------------------------------------------------------
# Command shape (pure; ffprobe probe patched out where loop would trigger it)
# ---------------------------------------------------------------------------


class TestBuildAudioMixCommand:
    def test_argv_is_a_list(self):
        cmd = audio_mix.build_audio_mix_command("v.mp4", "amb.mp3", "out.mp4", fade_s=0.0)
        assert isinstance(cmd, list)

    def test_encode_contract_copy_video_aac_stereo_shortest(self):
        cmd = audio_mix.build_audio_mix_command("v.mp4", "amb.mp3", "out.mp4", fade_s=0.0)
        assert "-c:v" in cmd and cmd[cmd.index("-c:v") + 1] == "copy"
        assert "-c:a" in cmd and cmd[cmd.index("-c:a") + 1] == "aac"
        assert "-b:a" in cmd and cmd[cmd.index("-b:a") + 1] == "192k"
        assert "-ac" in cmd and cmd[cmd.index("-ac") + 1] == "2"
        assert "-shortest" in cmd
        # two -map entries: the copied video (0:v) and the mixed audio ([aout])
        assert cmd[cmd.index("-map") + 1] == "0:v"
        assert "[aout]" in cmd
        assert cmd[cmd.index("-map", cmd.index("0:v") + 1) + 1] == "[aout]"

    def test_single_track_inputs(self):
        cmd = audio_mix.build_audio_mix_command("v.mp4", "amb.mp3", "out.mp4", fade_s=0.0)
        # inputs: video (0), ambience (1)
        assert cmd.count("-i") == 2
        assert cmd[cmd.index("-i") + 1] == "v.mp4"
        assert cmd[cmd.index("-i", cmd.index("v.mp4") + 1) + 1] == "amb.mp3"

    def test_dual_track_inputs(self):
        cmd = audio_mix.build_audio_mix_command("v.mp4", "amb.mp3", "out.mp4", copy_audio_from="copy.mp4", fade_s=0.0)
        assert cmd.count("-i") == 3
        # input order: video, copy, ambience
        inputs = [cmd[i + 1] for i, a in enumerate(cmd) if a == "-i"]
        assert inputs == ["v.mp4", "copy.mp4", "amb.mp3"]

    def test_loop_probes_duration_when_unspecified(self):
        with patch.object(audio_mix, "_probe_duration_s", return_value=12.5) as probe:
            cmd = audio_mix.build_audio_mix_command("v.mp4", "amb.mp3", "out.mp4", loop=True, fade_s=0.0)
        probe.assert_called_once_with("v.mp4", ffprobe=audio_mix._FFPROBE_BIN)
        fc = cmd[cmd.index("-filter_complex") + 1]
        assert "atrim=0:12.5" in fc

    def test_explicit_duration_skips_probe(self):
        with patch.object(audio_mix, "_probe_duration_s") as probe:
            audio_mix.build_audio_mix_command("v.mp4", "amb.mp3", "out.mp4", loop=True, duration_s=5.0, fade_s=0.0)
        probe.assert_not_called()


# ---------------------------------------------------------------------------
# mix_external_audio — file checks, ffmpeg success / failure
# ---------------------------------------------------------------------------


class TestMixExternalAudio:
    def _file(self, tmp_path: Path, name: str) -> str:
        p = tmp_path / name
        p.write_bytes(b"x")
        return str(p)

    def test_missing_video_raises(self, tmp_path: Path):
        amb = self._file(tmp_path, "a.mp3")
        with pytest.raises(FileNotFoundError):
            audio_mix.mix_external_audio(str(tmp_path / "nope.mp4"), amb, str(tmp_path / "out.mp4"))

    def test_missing_ambience_raises(self, tmp_path: Path):
        vid = self._file(tmp_path, "v.mp4")
        with pytest.raises(FileNotFoundError):
            audio_mix.mix_external_audio(vid, str(tmp_path / "nope.mp3"), str(tmp_path / "out.mp4"))

    def test_missing_copy_audio_raises(self, tmp_path: Path):
        vid = self._file(tmp_path, "v.mp4")
        amb = self._file(tmp_path, "a.mp3")
        with pytest.raises(FileNotFoundError):
            audio_mix.mix_external_audio(
                vid, amb, str(tmp_path / "out.mp4"), copy_audio_from=str(tmp_path / "nope.mp4")
            )

    def test_ffmpeg_failure_raises_runtime_error(self, tmp_path: Path):
        vid = self._file(tmp_path, "v.mp4")
        amb = self._file(tmp_path, "a.mp3")
        with patch.object(audio_mix, "subprocess") as sp:
            sp.run.return_value = type("r", (), {"returncode": 1, "stderr": "boom"})()
            with pytest.raises(RuntimeError):
                audio_mix.mix_external_audio(vid, amb, str(tmp_path / "out.mp4"))

    def test_success_writes_out_path_and_returns_it(self, tmp_path: Path):
        vid = self._file(tmp_path, "v.mp4")
        amb = self._file(tmp_path, "a.mp3")
        out = tmp_path / "out.mp4"

        def _run(cmd, **kwargs):
            assert isinstance(cmd, list)
            assert os.path.splitext(os.path.basename(cmd[0]))[0].lower() == "ffmpeg"
            Path(cmd[-1]).write_bytes(b"mixed")
            return type("r", (), {"returncode": 0, "stderr": "", "stdout": ""})()

        with patch.object(audio_mix, "subprocess") as sp:
            sp.run.side_effect = _run
            result = audio_mix.mix_external_audio(vid, amb, str(out))
        assert result == str(out)
        assert out.read_bytes() == b"mixed"


# ---------------------------------------------------------------------------
# Real ffmpeg integration — silent 2 s clip + 1 s sine, looped, ffprobe-verified
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not FFMPEG or not FFPROBE, reason="ffmpeg/ffprobe not installed")
class TestRealFfmpegMix:
    def _codec_types(self, path: str) -> list[str]:
        probe = subprocess.run(
            [FFPROBE, "-v", "error", "-show_entries", "stream=codec_type", "-of", "json", path],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
        return [s["codec_type"] for s in json.loads(probe.stdout)["streams"]]

    def test_loop_mix_produces_an_audio_track(self, tmp_path: Path):
        # 2 s silent video (video-only) — no audio stream to start from.
        video = tmp_path / "silent.mp4"
        subprocess.run(
            [
                FFMPEG,
                "-y",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=320x180:r=24:d=2",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(video),
            ],
            capture_output=True,
            check=True,
            timeout=120,
        )
        # 1 s sine ambience — shorter than the video, so it must loop.
        sine = tmp_path / "amb.wav"
        subprocess.run(
            [FFMPEG, "-y", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=1", str(sine)],
            capture_output=True,
            check=True,
            timeout=120,
        )

        out = tmp_path / "mixed.mp4"
        result = audio_mix.mix_external_audio(str(video), str(sine), str(out), loop=True, fade_s=0.5)

        assert result == str(out)
        types = self._codec_types(str(out))
        assert "video" in types, "output must keep the copied video stream"
        assert "audio" in types, "output must carry the mixed ambience track"
