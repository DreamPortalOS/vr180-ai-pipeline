"""Tests for ``scripts/dome_frame_tools.py`` — D-3 (#408) dome frame geometry tools.

The three subcommands exist because Gemini (free tier) only emits 16:9/9:16
while a domemaster is a 1:1 canvas inside an inscribed circle, and because its
dome frames come back with a 10 %-of-centre rim.  These tests pin the three
guarantees the card asks for:

* ``pad169`` — 1024² → 1820×1024, bars exactly 0, centre **bit**-identical;
* ``crop11`` — the padded frame round-trips back to 1024² with the circle
  interior intact and the interior/exterior split taken from the same radius
  map the QA gate (``scripts/dome_qa.py``) uses, on both the OpenCV still path
  (bit-exact) and the ffmpeg video path (real 1 s lavfi clip in ``tmp_path``);
* ``rimlift`` — a 10 %-rim disc is lifted to ≥ 40 % of the centre, ``r < start``
  is untouched, the surround stays pure black, and the lift is driven by *ring*
  statistics with a Gaussian-smoothed gain map.

Measured on this box (OpenCV 4.11, ffmpeg N-125258) — the assertions are
calibrated against these, not against round numbers:

======================================  ==================================
measurement                             value
======================================  ==================================
pad169 1024² → width                    1820 (even), bars exactly 0
crop11 PNG round-trip                   interior bit-identical, exterior 0
crop11 JPEG vs the original (interior)  mean |Δ| 0.53, p99 2 (card budget ≤ 2)
crop11 JPEG, outside mean               0.086 (JPEG ringing at the masked edge)
crop11 PNG, interior vs its own input   |Δ| 0 (the tool adds no error)
crop11 video 180², interior vs source   mean |Δ| 2.0 (h264 crf 18)
crop11 video 180², outside mean         < 1.4 (gate threshold is 8)
rimlift 10 % rim (max-gain 4)           rim 20 → 80 = exactly 0.40 × centre
rimlift striped rim, gain oscillation   std 0.013 smoothed vs 0.606 raw
======================================  ==================================

Everything runs from ``tmp_path``: no network, no models, and the only
subprocess calls are ffmpeg/ffprobe (skipped when they are not on PATH).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from scripts import dome_frame_tools as dft

from studio.coverage import _radial_map

#: The card's acceptance size for pad169 / crop11.
_SIZE = 1024
#: Synthetic video: 16:9 (320×180) so ``crop=ih:ih`` has something to cut.
_VIDEO_W, _VIDEO_H = 320, 180
#: Rim-light fixtures (σ ≈ 15 px is specified at 1024², so fixtures are 1024²).
_RIM_SIZE = 1024

_FFMPEG = shutil.which("ffmpeg")
_FFPROBE = shutil.which("ffprobe")
requires_ffmpeg = pytest.mark.skipif(not (_FFMPEG and _FFPROBE), reason="ffmpeg/ffprobe not on PATH")

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"


# --------------------------------------------------------------------------- #
# fixtures / helpers
# --------------------------------------------------------------------------- #


def _smooth_square(size: int = _SIZE) -> np.ndarray:
    """A 1:1 BGR test image with low-frequency structure (a plausible dome frame).

    Deliberately free of per-pixel noise: the JPEG tolerance assertions in
    ``TestCrop11Stills`` measure the *codec*, and a noisy fixture would measure
    grain instead (uniform ±12 grain on this same fixture costs mean |Δ| 2.7).
    """
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    base = 120.0 + 80.0 * np.sin(xx / 90.0) + 40.0 * np.cos(yy / 55.0)
    img = np.stack([base, np.roll(base, 17, axis=1), np.roll(base, 31, axis=0)], axis=2)
    return np.clip(img, 0, 255).astype(np.uint8)


def _radius(size: int = _RIM_SIZE) -> np.ndarray:
    """The tool's own r map (shared with ``dome_qa.py``) so tests assert in its r."""
    return _radial_map(size, size)


def _disc(size: int = _RIM_SIZE, center_level: int = 200, rim_level: int = 20, start: float = 0.5) -> np.ndarray:
    """A domemaster-like disc: ``center_level`` inside ``r < start``, black outside 1 R.

    Defaults are the card's S3/S4 defect — a rim at 10 % of the centre.
    """
    r = _radius(size)
    level = np.where(r < start, center_level, rim_level).astype(np.uint8)
    return np.stack([np.where(r <= 1.0, level, 0).astype(np.uint8)] * 3, axis=2)


def _read_frames(path: str | Path) -> list[np.ndarray]:
    """All decoded BGR frames of ``path`` (short synthetic clips only)."""
    cap = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()
    return frames


def _ffprobe_streams(path: str | Path) -> list[dict]:
    """``ffprobe -show_streams`` as JSON (list form, no shell)."""
    proc = subprocess.run(
        [str(_FFPROBE), "-v", "error", "-show_streams", "-of", "json", str(path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, f"ffprobe failed: {proc.stderr[-600:]}"
    return json.loads(proc.stdout)["streams"]


# --------------------------------------------------------------------------- #
# pad169 — 1:1 → 16:9
# --------------------------------------------------------------------------- #


class TestPad169:
    """1024² → 1820×1024 with exact black bars and a bit-identical centre."""

    def test_1024_square_becomes_1820x1024_with_black_bars(self):
        img = _smooth_square()
        padded = dft.pad169_image(img)

        assert padded.shape == (1024, 1820, 3)
        x0 = (1820 - 1024) // 2  # 398
        assert (padded[:, :x0] == 0).all()
        assert (padded[:, x0 + 1024 :] == 0).all()

    def test_centre_is_bit_identical_to_the_source(self):
        img = _smooth_square()
        padded = dft.pad169_image(img)

        x0 = (padded.shape[1] - _SIZE) // 2
        assert np.array_equal(padded[:, x0 : x0 + _SIZE], img)
        assert padded.dtype == img.dtype == np.uint8

    @pytest.mark.parametrize(
        ("height", "expected_width"),
        [(57, 102), (360, 640), (361, 642), (720, 1280), (1024, 1820), (1080, 1920)],
    )
    def test_width_is_the_16_9_round_trip_rounded_to_even(self, height: int, expected_width: int):
        assert dft.pad169_width(height) == expected_width
        assert dft.pad169_width(height) % 2 == 0
        assert abs(dft.pad169_width(height) - height * 16 / 9) <= 1.5

    def test_rejects_a_non_square_image(self):
        with pytest.raises(ValueError, match="1:1"):
            dft.pad169_image(np.zeros((100, 200, 3), dtype=np.uint8))

    def test_file_path_writes_a_readable_png(self, tmp_path):
        src = tmp_path / "square.png"
        assert cv2.imwrite(str(src), _smooth_square())

        written = dft.pad169_file(src, tmp_path / "wide.png")

        assert Path(written) == tmp_path / "wide.png"
        assert cv2.imread(written, cv2.IMREAD_COLOR).shape == (1024, 1820, 3)

    def test_missing_input_names_the_file(self, tmp_path):
        with pytest.raises(ValueError, match="cannot read image"):
            dft.pad169_file(tmp_path / "nope.png", tmp_path / "out.png")


# --------------------------------------------------------------------------- #
# crop11 — 16:9 → 1:1
# --------------------------------------------------------------------------- #


class TestCrop11Stills:
    """The OpenCV path: exact round-trip, exact black surround."""

    def test_png_round_trip_restores_the_square_exactly(self, tmp_path):
        img = _smooth_square()
        src = tmp_path / "square.png"
        assert cv2.imwrite(str(src), img)

        padded = dft.pad169_file(src, tmp_path / "padded.png")
        back = dft.crop11_file(padded, tmp_path / "back.png")
        got = cv2.imread(str(back), cv2.IMREAD_COLOR)

        assert got.shape == (1024, 1024, 3)
        inside = dft.circle_mask(1024, 1024)
        assert np.array_equal(got[inside], img[inside])
        assert int(got[~inside].max()) == 0

    def test_jpeg_round_trip_stays_within_the_two_level_budget(self, tmp_path):
        """Card budget: the padded-then-cropped original within JPEG error ≤ 2."""
        img = _smooth_square()
        src = tmp_path / "square.png"
        assert cv2.imwrite(str(src), img)

        padded = dft.pad169_file(src, tmp_path / "padded.jpg")
        back = dft.crop11_file(padded, tmp_path / "back.jpg")
        got = cv2.imread(str(back), cv2.IMREAD_COLOR)

        inside = dft.circle_mask(1024, 1024)
        delta = np.abs(got.astype(np.int16) - img.astype(np.int16))[inside]
        assert delta.mean() <= 2.0  # measured 0.53 over both JPEG passes
        assert float(np.percentile(delta, 99)) <= 4.0  # measured 2.0

        # The surround stays black to within the encoder's own ringing: JPEG
        # spreads a hard masked edge over the 8×8 blocks that straddle it
        # (measured |max| 38 in the first 4 px, 2 beyond 16 px), so the
        # *pixel-exact* "outside is 0" claim is the lossless PNG path's, and
        # here the outside mean is the honest measure.
        assert float(got[~inside].mean()) < 1.0  # measured 0.086

    def test_still_path_adds_no_error_of_its_own(self, tmp_path):
        """crop11 ≠ resample: in a lossless container the interior is bit-identical."""
        src = tmp_path / "square.png"
        assert cv2.imwrite(str(src), _smooth_square())
        padded = dft.pad169_file(src, tmp_path / "padded.png")

        padded_arr = cv2.imread(str(padded), cv2.IMREAD_COLOR)
        x0 = (padded_arr.shape[1] - padded_arr.shape[0]) // 2
        centre = padded_arr[:, x0 : x0 + padded_arr.shape[0]]

        got = cv2.imread(dft.crop11_file(padded, tmp_path / "back.png"), cv2.IMREAD_COLOR)
        inside = dft.circle_mask(1024, 1024)
        assert np.array_equal(got[inside], centre[inside])

    def test_centre_square_matches_ffmpeg_crop_defaults(self):
        """``crop``'s default x/y is ``(in - out) / 2`` in ints; side is the height."""
        assert dft.center_square_bounds(1820, 1024) == (398, 0, 1024)
        assert dft.center_square_bounds(1024, 1024) == (0, 0, 1024)
        assert dft.center_square_bounds(1920, 1080) == (420, 0, 1080)
        with pytest.raises(ValueError, match="landscape"):
            dft.center_square_bounds(1080, 1920)

    def test_circle_mask_is_the_inscribed_circle(self):
        mask = dft.circle_mask(256, 256)
        centre = 255 / 2.0
        assert mask[int(centre), int(centre)]
        assert mask[int(centre), int(centre + 127)]  # 127 < 128 = R
        assert not mask[0, 0]
        # r = 1 exactly at the edge midpoints, and the mask matches r <= 1.
        assert np.array_equal(mask, _radial_map(256, 256) <= 1.0)

    def test_video_suffixes_route_to_ffmpeg(self):
        assert dft.is_video("gemini.MP4") and dft.is_video("clip.mkv")
        assert not dft.is_video("frame.png") and not dft.is_video("frame.jpg")


class _Completed:
    """Minimal ``subprocess.CompletedProcess`` stand-in for the fake-runner tests."""

    def __init__(self, returncode: int = 0, stderr: str = "") -> None:
        self.returncode = returncode
        self.stderr = stderr
        self.stdout = ""


def _make_test_clip(path: Path, *, with_audio: bool = True) -> None:
    """Write a 1 s 16:9 lavfi clip into ``tmp_path`` (list form, no shell).

    ``testsrc2`` carries saturated chroma on purpose: 4:2:0 subsampling is what
    puts a halo on a hard circle edge, and a gray source would hide exactly the
    encoder behaviour the surround assertions below have to tolerate.
    """
    lavfi = f"testsrc2=size={_VIDEO_W}x{_VIDEO_H}:rate=10:duration=1"
    cmd = [_FFMPEG, "-y", "-v", "error", "-f", "lavfi", "-i", lavfi]
    if with_audio:
        cmd += ["-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-shortest", "-c:a", "aac"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=120)


class TestCrop11Video:
    """The ffmpeg path: list-form argv, ``crop=ih:ih`` + ``geq`` circle, audio copy."""

    def test_command_is_list_form_with_crop_mask_and_audio_copy(self, tmp_path):
        cmd = dft.build_crop11_command(tmp_path / "in.mp4", tmp_path / "out.mp4")

        assert isinstance(cmd, list)
        assert all(isinstance(part, str) for part in cmd)
        graph = cmd[cmd.index("-filter_complex") + 1]
        assert "crop=ih:ih" in graph
        assert "geq=" in graph
        assert dft.geq_circle_expression() in graph
        # geometry comes from the frame itself (W/H inside geq), never from a probe
        assert "format=gbrp" in graph and dft.CROP11_PIX_FMT in graph
        assert cmd[cmd.index("-map") + 1].startswith("[")  # the masked 1:1 stream
        assert "0:a:0?" in cmd  # optional: a silent source still converts
        assert cmd[cmd.index("-c:a") + 1] == "copy"

    def test_runner_never_uses_shell_and_forwards_failure(self, tmp_path, monkeypatch):
        seen: dict = {}

        def fake_run(cmd, **kwargs):
            seen["cmd"], seen["kwargs"] = cmd, kwargs
            return _Completed(returncode=0)

        monkeypatch.setattr(dft.subprocess, "run", fake_run)
        out = dft.crop11_video(tmp_path / "in.mp4", tmp_path / "out.mp4")

        assert Path(out) == tmp_path / "out.mp4"
        assert isinstance(seen["cmd"], list)
        assert not seen["kwargs"].get("shell")

        monkeypatch.setattr(dft.subprocess, "run", lambda cmd, **kw: _Completed(returncode=1, stderr="boom"))
        with pytest.raises(RuntimeError, match="boom"):
            dft.crop11_video(tmp_path / "in.mp4", tmp_path / "out.mp4")

    @requires_ffmpeg
    def test_real_clip_is_squared_circle_masked_and_keeps_its_audio(self, tmp_path):
        src = tmp_path / "src169.mp4"
        _make_test_clip(src)

        out = dft.crop11_file(src, tmp_path / "out11.mp4")

        src_audio = next(s for s in _ffprobe_streams(src) if s["codec_type"] == "audio")
        streams = _ffprobe_streams(out)
        video = next(s for s in streams if s["codec_type"] == "video")
        audio = next(s for s in streams if s["codec_type"] == "audio")
        assert (video["width"], video["height"]) == (_VIDEO_H, _VIDEO_H)  # side = input height
        assert audio["codec_name"] == src_audio["codec_name"]  # -c:a copy

        src_frames = _read_frames(src)
        out_frames = _read_frames(out)
        assert len(out_frames) >= 5
        assert abs(len(out_frames) - len(src_frames)) <= 1  # decoder slack, not a re-encode
        inside = dft.circle_mask(_VIDEO_H, _VIDEO_H)
        x0 = (_VIDEO_W - _VIDEO_H) // 2

        # the interior is the *centred* crop of the source (a shifted crop or a
        # stray scale would blow this up immediately): measured mean |Δ| 2.0
        deltas = []
        inside_means = []
        for src_frame, out_frame in zip(src_frames, out_frames, strict=False):
            reference = src_frame[:, x0 : x0 + _VIDEO_H]
            deltas.append(float(np.abs(out_frame.astype(np.int16) - reference.astype(np.int16))[inside].mean()))
            inside_means.append(float(out_frame[inside].mean()))
        assert float(np.mean(deltas)) <= 8.0
        assert min(inside_means) > 20.0  # content survived the mask

        # the surround is black: the mean sits far under the QA gate's own
        # outside threshold (8).  A 1–3 px chroma/ringing halo hugs the circle
        # edge — that is what 4:2:0 does to a hard edge, and why the pixel-exact
        # "outside is 0" guarantee belongs to the still path above.
        outside_means = [float(frame[~inside].mean()) for frame in out_frames]
        assert max(outside_means) < 8.0  # measured 1.4

    @requires_ffmpeg
    def test_silent_clip_still_converts(self, tmp_path):
        """``-map 0:a:0?`` — no audio track must not fail the run."""
        src = tmp_path / "silent169.mp4"
        _make_test_clip(src, with_audio=False)

        out = dft.crop11_file(src, tmp_path / "silent11.mp4")

        streams = _ffprobe_streams(out)
        assert [s["codec_type"] for s in streams] == ["video"]


# --------------------------------------------------------------------------- #
# rimlift — radial brightening
# --------------------------------------------------------------------------- #


class TestRimlift:
    """A 10 %-rim disc must reach ≥ 40 % of the centre, leaving the centre alone."""

    def test_rim_reaches_forty_percent_of_the_centre(self):
        img = _disc(rim_level=20)  # 20 = 10 % of the centre's 200
        r = _radius()

        lifted, profile = dft.rimlift_image(img)

        rim_before = float(img[(r >= 0.9) & (r <= 1.0)].mean())
        rim_after = float(lifted[(r >= 0.9) & (r <= 1.0)].mean())
        centre = float(lifted[r < 0.3].mean())
        assert rim_before == pytest.approx(0.10 * 200.0, rel=0.05)  # the card's defect
        assert rim_after >= 0.40 * centre  # 20 × max_gain 4 = exactly the 40 % bar
        assert rim_after <= 0.55 * centre  # ... and never past --target
        assert profile.target_level == pytest.approx(0.55 * 200.0, rel=1e-6)

    def test_region_below_start_is_bit_exact(self):
        img = _disc()
        r = _radius()

        lifted, _ = dft.rimlift_image(img, start=0.5)

        assert np.array_equal(lifted[r < 0.5], img[r < 0.5])
        # and the same holds for a different start — it is a parameter, not luck
        lifted_7, _ = dft.rimlift_image(img, start=0.7)
        assert np.array_equal(lifted_7[r < 0.7], img[r < 0.7])

    def test_outside_the_circle_is_forced_black(self):
        img = _disc()
        img[~(_radius() <= 1.0)] = 255  # bright corners, as a padded frame has
        r = _radius()

        lifted, _ = dft.rimlift_image(img)

        assert (lifted[r > 1.0] == 0).all()
        clean, _ = dft.rimlift_image(_disc())
        assert np.array_equal(lifted[r <= 1.0], clean[r <= 1.0])  # corners never leak in

    def test_max_gain_caps_the_lift(self):
        """A 5 %-rim asking for gain 11 is capped at 4 → still under the 40 % bar."""
        img = _disc(rim_level=10)
        r = _radius()

        lifted, profile = dft.rimlift_image(img)

        assert max(profile.ring_gains) == pytest.approx(4.0)
        rim_after = float(lifted[(r >= 0.9) & (r <= 1.0)].mean())
        assert rim_after == pytest.approx(4 * 10, abs=1.5)
        assert rim_after < 0.40 * 200.0

    def test_an_already_bright_disc_is_never_darkened(self):
        r = _radius()
        img = np.full((_RIM_SIZE, _RIM_SIZE, 3), 200, dtype=np.uint8)

        lifted, profile = dft.rimlift_image(img)

        assert max(profile.ring_gains) == pytest.approx(1.0)  # gain is clamped at 1
        assert np.array_equal(lifted[r <= 1.0], img[r <= 1.0])

    def test_black_frame_is_left_alone(self):
        lifted, profile = dft.rimlift_image(np.zeros((_RIM_SIZE, _RIM_SIZE, 3), dtype=np.uint8))

        assert profile.center_level == 0.0
        assert int(lifted.max()) == 0

    def test_lift_is_per_ring_not_per_pixel(self):
        """A dark wedge is scaled by its *ring* gain, so relative structure survives.

        Per-pixel normalisation would flatten the wedge to the ring's mean; the
        card asks for a radius **profile**.
        """
        size = _RIM_SIZE
        r = _radius(size)
        yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
        theta = np.arctan2(yy - (size - 1) / 2, xx - (size - 1) / 2)
        rim = (r >= 0.9) & (r <= 1.0)
        wedge = rim & (np.abs(theta) < 0.3)
        img = _disc(size=size)
        img[wedge] = 10  # half the surrounding rim level

        lifted, profile = dft.rimlift_image(img)

        before = float(img[wedge].mean() / img[rim & ~wedge].mean())
        after = float(lifted[wedge].mean() / lifted[rim & ~wedge].mean())
        assert before == pytest.approx(0.5, rel=0.02)
        assert after == pytest.approx(before, rel=0.05)
        assert float(lifted[rim & ~wedge].mean()) >= 0.40 * profile.center_level

    def test_gain_map_is_gaussian_smoothed(self):
        """σ ≈ 15 px @1024 flattens the 32-ring staircase a raw profile would show.

        Measured on a striped rim: radial gain oscillation std 0.013 smoothed
        vs 0.606 raw — that ratio is what keeps the annuli from banding.
        """
        size = _RIM_SIZE
        r = _radius(size)
        stripes = np.floor(r / (0.5 / 32)) % 2 == 0
        level = np.where(stripes, 200, 20).astype(np.uint8)
        img = np.stack([np.where(r <= 1.0, level, 0).astype(np.uint8)] * 3, axis=2)

        gray = img.astype(np.float32).mean(axis=2)
        profile = dft.radius_profile_gains(gray, r, start=0.5)
        smooth = dft.gain_map_from_profile(r, profile, start=0.5, sigma_px=dft.RIM_SIGMA_PX)
        raw = dft.gain_map_from_profile(r, profile, start=0.5, sigma_px=0.0)

        row = size // 2
        centre = (size - 1) / 2
        xs = np.arange(int(centre) + 1, size)
        radius_of_x = (xs - centre) / (size / 2)
        window = (radius_of_x >= 0.55) & (radius_of_x <= 0.95)
        raw_std = float(raw[row, xs][window].std())
        smooth_std = float(smooth[row, xs][window].std())
        assert raw_std > 0.5  # the staircase really is in the fixture
        assert smooth_std < 0.1 * raw_std  # measured 0.013 vs 0.606

    def test_gain_map_pins_the_untouched_region_to_one(self):
        img = _disc()
        r = _radius()
        gray = img.astype(np.float32).mean(axis=2)

        gain = dft.gain_map_from_profile(r, dft.radius_profile_gains(gray, r, start=0.5), start=0.5)

        assert (gain[r < 0.5] == 1.0).all()  # smoothing may not bleed inward
        assert (gain[(r >= 0.7) & (r <= 0.95)] > 1.0).all()

    def test_rejects_non_square_and_bad_arguments(self, tmp_path):
        with pytest.raises(ValueError, match="1:1"):
            dft.rimlift_image(np.zeros((100, 200, 3), dtype=np.uint8))
        with pytest.raises(ValueError, match="start"):
            dft.rimlift_image(_disc(), start=1.0)
        with pytest.raises(ValueError, match="target"):
            dft.rimlift_image(_disc(), target=-0.1)
        with pytest.raises(ValueError, match="max-gain"):
            dft.rimlift_image(_disc(), max_gain=0.5)
        with pytest.raises(ValueError, match="still images"):
            dft.rimlift_file(tmp_path / "clip.mp4", tmp_path / "out.png")

    def test_file_path_lifts_a_png(self, tmp_path):
        src = tmp_path / "dome.png"
        assert cv2.imwrite(str(src), _disc(rim_level=20))

        written = dft.rimlift_file(src, tmp_path / "dome_lifted.png")

        lifted = cv2.imread(str(written), cv2.IMREAD_COLOR)
        r = _radius()
        assert float(lifted[(r >= 0.9) & (r <= 1.0)].mean()) >= 0.40 * 200.0
        assert int(lifted[r > 1.0].max()) == 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


class TestCli:
    """``main`` wires the three subcommands; ``--help`` must work bare."""

    def test_pad_crop_rimlift_round_trip_through_argv(self, tmp_path):
        src = tmp_path / "square.png"
        assert cv2.imwrite(str(src), _smooth_square())
        padded, squared, lifted = tmp_path / "wide.png", tmp_path / "square_back.png", tmp_path / "lifted.png"

        assert dft.main(["pad169", str(src), str(padded)]) == 0
        assert dft.main(["crop11", str(padded), str(squared)]) == 0
        assert dft.main(["rimlift", str(squared), str(lifted), "--target", "0.6", "--start", "0.4"]) == 0

        assert cv2.imread(str(padded), cv2.IMREAD_COLOR).shape == (1024, 1820, 3)
        r = _radius()
        back = cv2.imread(str(squared), cv2.IMREAD_COLOR)
        assert np.array_equal(back[dft.circle_mask(1024, 1024)], _smooth_square()[dft.circle_mask(1024, 1024)])
        assert int(cv2.imread(str(lifted), cv2.IMREAD_COLOR)[r > 1.0].max()) == 0

    def test_failure_returns_exit_code_one(self, tmp_path, caplog):
        with caplog.at_level(logging.ERROR, logger="dome_frame_tools"):
            assert dft.main(["pad169", str(tmp_path / "missing.png"), str(tmp_path / "out.png")]) == 1
        assert "pad169 failed" in caplog.text

    def test_help_runs_without_pythonpath(self):
        """K-15 spirit: the script bootstraps the repo root itself for ``studio``."""
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        proc = subprocess.run(
            [sys.executable, str(_SCRIPTS_DIR / "dome_frame_tools.py"), "--help"],
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )
        assert proc.returncode == 0, f"--help crashed: {proc.stderr[-600:]}"
        assert "pad169" in proc.stdout and "crop11" in proc.stdout and "rimlift" in proc.stdout
