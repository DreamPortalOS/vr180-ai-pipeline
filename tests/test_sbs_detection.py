"""Tests for Smart SBS Input Detection (Task 1.1).

The fixtures here encode real clips with ffmpeg.  Two things about that used to
be wrong (#357):

* The ``subprocess.run`` calls passed ``capture_output=True, timeout=30`` and no
  ``check=True``, so a failed or timed-out encode was swallowed whole and the
  test went on to assert against a file that had never been written.  The
  symptom reaching the reader was "file not found", never "ffmpeg said X".
  Everything now goes through :func:`_encode_solid_clip`, which turns a
  non-zero exit, a timeout or an empty output into a named failure carrying
  ffmpeg's own stderr.
* The SBS fixture really encoded 7680x1920 with libx264.  ``detect_sbs_input``
  reads only ``CAP_PROP_FRAME_WIDTH`` / ``CAP_PROP_FRAME_HEIGHT`` and compares
  ``w / h`` against 3.5 — the aspect ratio is the entire signal, and absolute
  pixel counts are not part of the logic under test.  The clips are therefore
  encoded at 1/100th the pixels with the ratios preserved exactly
  (768x192 = 4:1, 360x100 = 3.6:1, 192x108 = 16:9).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import NoReturn

import numpy as np
import pytest
from scripts.run_pipeline import detect_sbs_input

# Kept as a module constant so the #357 mutation check ("point this at a command
# that does not exist and confirm the test says so out loud") is a one-line edit.
FFMPEG = "ffmpeg"

# Each clip is 3 frames of a solid colour at <=0.15 MP, i.e. tens of
# milliseconds of work.  Anything approaching this ceiling means the encoder is
# wedged rather than merely sharing a busy box, and that must be reported.
ENCODE_TIMEOUT_S = 60.0

# Enough of ffmpeg's stderr to carry the actual diagnostic line, which it prints
# last, without dumping the whole banner into the report.
STDERR_TAIL_CHARS = 2000


def _stderr_tail(stderr: str | bytes | None) -> str:
    """Normalise captured stderr to a printable tail."""
    if not stderr:
        return "<no stderr captured>"
    if isinstance(stderr, bytes):
        stderr = stderr.decode("utf-8", "replace")
    return stderr[-STDERR_TAIL_CHARS:]


def _fail_encode(cmd: list[str], what: str, stderr: str | bytes | None) -> NoReturn:
    """Abort with a message that names the cause instead of its side effect."""
    pytest.fail(
        f"SBS fixture encode failed: {what}\n"
        f"  command: {subprocess.list2cmdline(cmd)}\n"
        f"  ffmpeg stderr (last {STDERR_TAIL_CHARS} chars):\n{_stderr_tail(stderr)}",
        pytrace=False,
    )


def _encode_solid_clip(dest: Path, size: str, color: str, timeout: float = ENCODE_TIMEOUT_S) -> str:
    """Encode a 3-frame solid-colour clip at ``size``, or say why it could not.

    Args:
        dest: Output path for the ``.mp4``.
        size: ffmpeg ``WxH`` geometry string; only its ratio matters to the code
            under test.
        color: lavfi colour name.
        timeout: Seconds to allow the encode; overridable so the guard tests can
            exercise the timeout branch cheaply.

    Returns:
        ``str(dest)`` once the file provably exists and is non-empty.

    Raises:
        Skipped: ffmpeg is not installed, so there is no encoding environment to
            test in.  A skip is visible in the report; a silent pass would not be.
        Failed: ffmpeg is installed but did not produce the clip.  The message
            carries the command line and ffmpeg's stderr.
    """
    if shutil.which(FFMPEG) is None:
        pytest.skip(f"{FFMPEG!r} is not on PATH: cannot build the SBS detection fixtures")

    cmd = [
        FFMPEG,
        # Keep the build banner out of stderr so the tail quoted on failure is
        # the diagnostic rather than 2 KB of ./configure flags.
        "-hide_banner",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"color=c={color}:s={size}:d=0.125:r=24",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(dest),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        _fail_encode(cmd, f"timed out after {timeout}s", exc.stderr)
    except OSError as exc:
        # which() found something we still cannot exec (permissions, broken
        # symlink, wrong architecture): no usable encoder, so skip rather than
        # let a FileNotFoundError traceback stand in for a diagnosis.
        pytest.skip(f"cannot execute {FFMPEG!r}: {exc}")

    if proc.returncode != 0:
        _fail_encode(cmd, f"exited with code {proc.returncode}", proc.stderr)
    if not dest.exists():
        _fail_encode(cmd, "exited 0 but wrote no output file", proc.stderr)
    if dest.stat().st_size == 0:
        _fail_encode(cmd, "exited 0 but wrote a 0-byte output file", proc.stderr)
    return str(dest)


# The clips are read-only inputs, so one encode per ratio is shared by the whole
# module instead of re-encoding per test.
@pytest.fixture(scope="module")
def standard_video(tmp_path_factory):
    """A 16:9 clip (standard 2D input): ratio 1.78, well under the 3.5 cut."""
    dest = tmp_path_factory.mktemp("sbs_detection") / "standard.mp4"
    return _encode_solid_clip(dest, "192x108", "blue")


@pytest.fixture(scope="module")
def sbs_video(tmp_path_factory):
    """A 4:1 SBS clip, standing in for the 7680x1920 original (see module docstring)."""
    dest = tmp_path_factory.mktemp("sbs_detection") / "sbs.mp4"
    return _encode_solid_clip(dest, "768x192", "red")


@pytest.fixture(scope="module")
def ultra_wide_video(tmp_path_factory):
    """A 3.6:1 clip: just above the 3.5 threshold."""
    dest = tmp_path_factory.mktemp("sbs_detection") / "ultrawide.mp4"
    return _encode_solid_clip(dest, "360x100", "green")


class TestFixtureEncodeGuard:
    """Guard the guard (#357).

    These pin the behaviour that the bug removed: if the fixture's encode does
    not happen, the run must say so in ffmpeg's own words at the point of
    failure — never hand a nonexistent path to the code under test and let
    "file not found" pose as the diagnosis.
    """

    def test_broken_encode_fails_loudly_with_ffmpeg_stderr(self, tmp_path):
        """A rejected geometry must surface as a named failure quoting stderr."""
        if shutil.which(FFMPEG) is None:
            pytest.skip(f"{FFMPEG!r} is not on PATH")

        dest = tmp_path / "broken.mp4"
        # lavfi rejects a 0x0 colour source, so ffmpeg exits non-zero and says why.
        with pytest.raises(pytest.fail.Exception) as excinfo:
            _encode_solid_clip(dest, "0x0", "red")

        message = str(excinfo.value)
        assert "SBS fixture encode failed" in message
        assert "exited with code" in message
        assert "ffmpeg stderr" in message
        # The real diagnostic, not an empty placeholder.
        assert "<no stderr captured>" not in message
        assert "Error opening input" in message
        # The command line is present so the failure is reproducible by hand.
        assert "color=c=red:s=0x0" in message

    def test_encode_timeout_fails_loudly(self, tmp_path):
        """A timed-out encode must fail, not fall through with no file."""
        if shutil.which(FFMPEG) is None:
            pytest.skip(f"{FFMPEG!r} is not on PATH")

        with pytest.raises(pytest.fail.Exception) as excinfo:
            _encode_solid_clip(tmp_path / "slow.mp4", "192x108", "blue", timeout=0.0)

        message = str(excinfo.value)
        assert "SBS fixture encode failed" in message
        assert "timed out after 0.0s" in message

    def test_missing_ffmpeg_skips_with_a_readable_reason(self, tmp_path, monkeypatch):
        """No encoder on the box is a visible skip, never a silent pass.

        This is the #357 mutation check made permanent: with the ffmpeg
        executable unreachable, the run reports a sentence a human can act on
        rather than a ``FileNotFoundError`` traceback.
        """
        monkeypatch.setattr(shutil, "which", lambda _name: None)

        with pytest.raises(pytest.skip.Exception, match="is not on PATH"):
            _encode_solid_clip(tmp_path / "never.mp4", "192x108", "blue")

    def test_zero_byte_output_fails_loudly(self, tmp_path, monkeypatch):
        """ffmpeg exiting 0 without real output is still a fixture failure."""
        dest = tmp_path / "empty.mp4"

        def _fake_run(cmd, **_kwargs):
            Path(cmd[-1]).touch()
            return subprocess.CompletedProcess(cmd, 0, "", "some ffmpeg noise")

        monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/ffmpeg")
        monkeypatch.setattr(subprocess, "run", _fake_run)

        with pytest.raises(pytest.fail.Exception) as excinfo:
            _encode_solid_clip(dest, "192x108", "blue")

        message = str(excinfo.value)
        assert "0-byte output file" in message
        assert "some ffmpeg noise" in message


class TestSBSDetection:
    """Test auto-detection of SBS stereo input."""

    def test_standard_16x9_not_sbs(self, standard_video):
        """Standard 16:9 video should NOT be detected as SBS."""
        result = detect_sbs_input(standard_video)
        assert result is False

    def test_sbs_4x1_detected(self, sbs_video):
        """4:1 SBS video should be detected as SBS."""
        result = detect_sbs_input(sbs_video)
        assert result is True

    def test_force_sbs_flag(self, standard_video):
        """--force-sbs should override detection for any input."""
        result = detect_sbs_input(standard_video, force_sbs=True)
        assert result is True

    def test_ultra_wide_detected(self, ultra_wide_video):
        """3.6:1 video (above 3.5 threshold) should be detected as SBS."""
        result = detect_sbs_input(ultra_wide_video)
        assert result is True

    def test_nonexistent_file_returns_false(self):
        """Non-existent file should return False gracefully."""
        result = detect_sbs_input("/nonexistent/video.mp4")
        assert result is False

    def test_stage_order_sbs_skips_depth_stereo(self):
        """STAGE_ORDER_SBS should not contain 'depth' or 'stereo'."""
        from scripts.run_pipeline import STAGE_ORDER_SBS

        assert "depth" not in STAGE_ORDER_SBS
        assert "stereo" not in STAGE_ORDER_SBS
        assert "equirect" in STAGE_ORDER_SBS
        assert "metadata" in STAGE_ORDER_SBS


class TestSBSPipelineIntegration:
    """Integration test: SBS input should skip depth/stereo and go to equirect."""

    def test_sbs_frame_split(self):
        """Test that SBS frames are correctly split into left/right."""
        # Create a synthetic SBS frame (left=red, right=blue)
        h, w = 480, 1920  # 4:1 SBS
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, : w // 2, 2] = 255  # left half = red
        frame[:, w // 2 :, 0] = 255  # right half = blue

        mid = w // 2
        left = frame[:, :mid, :]
        right = frame[:, mid:, :]

        assert left.shape == (h, mid, 3)
        assert right.shape == (h, mid, 3)
        # Left should be predominantly red
        assert left[:, :, 2].mean() > 200
        # Right should be predominantly blue
        assert right[:, :, 0].mean() > 200
