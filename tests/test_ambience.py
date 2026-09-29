"""Tests for scripts/ambience.py — deep-sea ambience placeholder (issue #429).

Pure command-construction tests run on CI (CPU-only, no models, no network)
without ffmpeg.  The single real-ffmpeg integration test renders a 3 s sample
into ``tmp_path`` (never the repo root) and is skipped when ffmpeg / ffprobe
are absent.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import ambience  # noqa: E402

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")


# ---------------------------------------------------------------------------
# Pure filter_complex construction
# ---------------------------------------------------------------------------


class TestFilterComplex:
    def test_water_bed_pink_noise_lowpass(self):
        fc = ambience.build_filter_complex(80.0, seed=7)
        assert "anoisesrc=color=pink" in fc
        assert "lowpass" in fc

    def test_hum_low_sine_with_slow_tremolo(self):
        fc = ambience.build_filter_complex(80.0, seed=7)
        m = re.search(r"sine=frequency=([\d.]+)", fc)
        assert m is not None
        assert 40.0 <= float(m.group(1)) <= 70.0
        assert "tremolo" in fc

    def test_bubbles_seeded_bandpass_pulses(self):
        fc = ambience.build_filter_complex(80.0, seed=7)
        assert "bandpass" in fc
        assert "adelay" in fc

    def test_headroom_ceiling(self):
        fc = ambience.build_filter_complex(80.0, seed=7)
        assert "alimiter" in fc
        assert "volume=-1dB" in fc

    def test_fades_in_and_out(self):
        fc = ambience.build_filter_complex(80.0, seed=7)
        assert "afade=t=in:st=0:d=2" in fc
        assert "afade=t=out:st=78:d=2" in fc

    def test_duration_propagates(self):
        fc = ambience.build_filter_complex(12.0, seed=7)
        assert "duration=12" in fc
        assert "afade=t=out:st=10:d=2" in fc

    def test_starts_from_silence_edge(self):
        fc = ambience.build_filter_complex(80.0, seed=7)
        assert "afade=t=in:st=0" in fc

    def test_output_contract_48k_stereo(self):
        cmd = ambience.build_ffmpeg_command(80.0, "ambience.wav")
        assert cmd[cmd.index("-ar") + 1] == "48000"
        assert cmd[cmd.index("-ac") + 1] == "2"
        assert cmd[cmd.index("-c:a") + 1] == "pcm_s16le"
        assert cmd[cmd.index("-t") + 1] == "80"
        assert cmd[-1].endswith(".wav")


class TestBubbleDeterminism:
    def test_same_seed_same_times(self):
        assert ambience.bubble_times(80.0, 7) == ambience.bubble_times(80.0, 7)

    def test_times_inside_programme(self):
        for t in ambience.bubble_times(80.0, 7):
            assert 1.0 <= t <= 79.0

    def test_times_sorted(self):
        times = ambience.bubble_times(80.0, 7)
        assert times == sorted(times) and len(times) >= 2

    def test_seed_changes_filter(self):
        a = ambience.build_filter_complex(80.0, seed=7)
        b = ambience.build_filter_complex(80.0, seed=8)
        assert a != b


class TestCommandShape:
    def test_argv_is_list_no_shell(self):
        cmd = ambience.build_ffmpeg_command(80.0, "ambience.wav")
        assert isinstance(cmd, list) and all(isinstance(x, str) for x in cmd)

    def test_uses_filter_complex_map(self):
        cmd = ambience.build_ffmpeg_command(80.0, "ambience.wav")
        assert "-filter_complex" in cmd
        assert "[aout]" in cmd


class TestCli:
    def test_defaults(self):
        args = ambience.parse_args(["--duration", "80", "--out", "ambience.wav"])
        assert args.seed == 7 and args.style == "underwater"

    def test_style_choices(self):
        with pytest.raises(SystemExit):
            ambience.parse_args(["--duration", "80", "--out", "a.wav", "--style", "forest"])

    def test_missing_ffmpeg_raises(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda _name: None)
        with pytest.raises(FileNotFoundError):
            ambience.render_ambience(1.0, str(tmp_path / "a.wav"))


# ---------------------------------------------------------------------------
# Real ffmpeg integration — tmp_path only, skipped without ffmpeg/ffprobe
# ---------------------------------------------------------------------------


def _peak_db(path: str) -> float:
    proc = subprocess.run(
        [FFMPEG, "-hide_banner", "-i", path, "-af", "volumedetect", "-f", "null", "-"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    m = re.search(r"max_volume:\s*(-?[\d.]+)\s*dB", proc.stderr)
    assert m is not None, f"volumedetect gave no max_volume:\n{proc.stderr}"
    return float(m.group(1))


@pytest.mark.skipif(not FFMPEG or not FFPROBE, reason="ffmpeg/ffprobe not installed")
def test_render_3s_sample_contract(tmp_path: Path):
    out = tmp_path / "ambience.wav"
    result = ambience.render_ambience(3.0, str(out), seed=7)
    assert result == str(out) and out.exists()

    probe = subprocess.run(
        [
            FFPROBE,
            "-v",
            "error",
            "-show_entries",
            "stream=sample_rate,channels",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    info = json.loads(probe.stdout)
    stream = info["streams"][0]
    assert stream["sample_rate"] == "48000"
    assert stream["channels"] == 2
    assert float(info["format"]["duration"]) == pytest.approx(3.0, abs=0.1)

    assert _peak_db(str(out)) <= -1.0
