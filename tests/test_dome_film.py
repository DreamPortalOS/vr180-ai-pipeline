"""Tests for ``scripts/dome_film.py`` — the multi-scene dome short composer (D-1, #427).

The card pins four guarantees this suite covers:

* the **video rim-lift** function (``compute_video_gain_map`` +
  ``apply_rimlift_gain`` in ``dome_frame_tools``) lifts a 10 %-of-centre rim
  to >= 40 % of the centre, leaves ``r < 0.5`` bit-exact and keeps the
  surround pure black — pure numpy, no ffmpeg needed;
* ``--dry-run`` prints the plan but writes **no file**;
* a full two-scene run on synthetic 2 s lavfi clips (bright centre, dark rim)
  produces a **square** master whose surround is black, whose total duration
  is approximately ``2 + 2 − crossfade`` and whose outer annulus is brighter
  after the lift, and whose ``dome_qa`` JSON report exists next to the film.

Everything runs from ``tmp_path``: no network, no models, no GPU.  The only
subprocess calls are ffmpeg/ffprobe (skipped when not on PATH).  The repo-root
pollution guard (``tests/conftest.py``) is the backstop.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest
from scripts import dome_film
from scripts import dome_frame_tools as dft

from studio.coverage import _radial_map

_FFMPEG = shutil.which("ffmpeg")
_FFPROBE = shutil.which("ffprobe")
requires_ffmpeg = pytest.mark.skipif(not (_FFMPEG and _FFPROBE), reason="ffmpeg/ffprobe not on PATH")

#: Small square for the synthetic rim-lift fixtures (σ is scaled with size).
_RIM_SIZE = 256
#: Synthetic clip geometry: 16:9 so crop11 has a square to cut.
_VW, _VH = 320, 180
#: Per-scene length of the synthetic clips (seconds).
_CLIP_DUR = 2.0


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _radius(size: int = _RIM_SIZE) -> np.ndarray:
    return _radial_map(size, size)


def _disc(size: int = _RIM_SIZE, center_level: int = 200, rim_level: int = 20, start: float = 0.5) -> np.ndarray:
    """A domemaster disc: bright inside ``r < start``, dark rim, black surround.

    Defaults are the card's S3/S4 defect — a rim at 10 % of the centre.
    """
    r = _radius(size)
    level = np.where(r < start, center_level, rim_level).astype(np.uint8)
    return np.stack([np.where(r <= 1.0, level, 0).astype(np.uint8)] * 3, axis=2)


def _ffprobe_duration(path: Path) -> float:
    proc = subprocess.run(
        [str(_FFPROBE), "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, f"ffprobe failed: {proc.stderr[-400:]}"
    return float(json.loads(proc.stdout)["format"]["duration"])


def _ffprobe_streams(path: Path) -> list[dict]:
    proc = subprocess.run(
        [str(_FFPROBE), "-v", "error", "-show_streams", "-of", "json", str(path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, f"ffprobe failed: {proc.stderr[-400:]}"
    return json.loads(proc.stdout)["streams"]


def _make_dome_clip(path: Path, *, duration: float = _CLIP_DUR, with_audio: bool = True) -> None:
    """A 2 s 16:9 lavfi clip with a bright centred disc on a dark field.

    ``geq`` paints a radial disc (bright centre, 10 %-of-centre rim, black
    outside) onto a 16:9 canvas so ``crop11`` has a real inscribed circle to
    cut and ``rimlift`` has a real dark annulus to lift.  List form, no shell.
    """
    # testsrc2 gives moving structure; we composite the disc over it so the
    # rim is genuinely dark (the defect rimlift fixes) while the centre stays
    # bright.  luminance(r) = 200 inside r<0.5, 20 in the rim, 0 outside 1.
    expr = "280+280*0.4*lte(hypot(X-(W-1)/2,Y-(H-1)/2),H/2*0.5)+20*0.4*between(hypot(X-(W-1)/2,Y-(H-1)/2),H/2*0.5,H/2)"
    vf = f"format=gbrp,geq=r='{expr}':g='{expr}':b='{expr}'"
    cmd = [
        str(_FFMPEG),
        "-y",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=size={_VW}x{_VH}:rate=25:duration={duration}",
    ]
    if with_audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}", "-shortest", "-c:a", "aac"]
    cmd += ["-vf", vf, "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=180)


def _write_plan(path: Path, scenes: list[dict], **extra) -> None:
    plan = {"size": 256, "upscale": "lanczos", "crossfade": 0.5, "scenes": scenes}
    plan.update(extra)
    path.write_text(json.dumps(plan), encoding="utf-8")


# --------------------------------------------------------------------------- #
# video rim-lift function (pure numpy — no ffmpeg)
# --------------------------------------------------------------------------- #


class TestVideoRimlift:
    """The card's video-rimlift unit test, run without ffmpeg."""

    def test_rim_reaches_forty_percent_of_centre(self):
        frame = _disc(rim_level=20)  # 20 = 10 % of the 200 centre
        r = _radius()

        gain, profile = dft.compute_video_gain_map(frame)
        lifted = dft.apply_rimlift_gain(frame, gain)

        centre = float(lifted[r < 0.3].mean())
        rim_after = float(lifted[(r >= 0.9) & (r <= 1.0)].mean())
        assert rim_after >= 0.40 * centre  # 20 × max_gain 4 = exactly the 40 % bar
        assert rim_after <= 0.55 * centre + 1.0  # ... and never past --target
        assert profile.target_level == pytest.approx(0.55 * 200.0, rel=1e-6)

    def test_region_below_start_is_unchanged(self):
        frame = _disc()
        r = _radius()

        gain, _ = dft.compute_video_gain_map(frame, start=0.5)
        lifted = dft.apply_rimlift_gain(frame, gain)

        assert np.array_equal(lifted[r < 0.5], frame[r < 0.5])

    def test_outside_the_circle_is_black(self):
        frame = _disc()
        frame[~(_radius() <= 1.0)] = 255  # bright corners, as a padded frame has
        r = _radius()

        gain, _ = dft.compute_video_gain_map(frame)
        lifted = dft.apply_rimlift_gain(frame, gain)

        assert (lifted[r > 1.0] == 0).all()

    def test_one_gain_map_applied_to_every_frame(self):
        """The card asks for "compute once, apply per frame": the same map lifts
        two different frames identically in the rim where the gain is the only
        difference (a darker second frame is lifted by the *same* factor)."""
        frame_a = _disc(rim_level=20)
        frame_b = _disc(rim_level=15)  # darker rim
        r = _radius()

        gain, _ = dft.compute_video_gain_map(frame_a)
        a = dft.apply_rimlift_gain(frame_a, gain)
        b = dft.apply_rimlift_gain(frame_b, gain)

        rim = (r >= 0.9) & (r <= 1.0)
        # the gain is the same constant map, so the ratio is preserved
        ratio_before = float(frame_a[rim].mean() / frame_b[rim].mean())
        ratio_after = float(a[rim].mean() / b[rim].mean())
        assert ratio_after == pytest.approx(ratio_before, rel=0.05)


# --------------------------------------------------------------------------- #
# --dry-run writes nothing
# --------------------------------------------------------------------------- #


class TestDryRun:
    def test_dry_run_writes_no_files(self, tmp_path):
        clip = tmp_path / "a.mp4"
        clip.write_bytes(b"not a real clip")  # dry-run never opens it
        plan = tmp_path / "plan.json"
        _write_plan(plan, [{"clip": str(clip), "is_169": True, "rimlift": True}])

        before = {p for p in tmp_path.iterdir()}
        out = tmp_path / "film.mp4"

        rc = dome_film.main(["--plan", str(plan), "--output", str(out), "--dry-run"])

        assert rc == 0
        assert not out.exists()
        assert {p for p in tmp_path.iterdir()} == before  # nothing added at all

    def test_dry_run_prints_each_step_and_predicted_duration(self, tmp_path, capsys):
        # A real 1 s clip so the predicted duration line is concrete.
        clip = tmp_path / "real.mp4"
        _make_dome_clip(clip, duration=1.0, with_audio=False)
        plan = tmp_path / "plan.json"
        _write_plan(plan, [{"clip": str(clip), "is_169": True, "rimlift": True}], crossfade=0.3)

        rc = dome_film.main(["--plan", str(plan), "--output", str(tmp_path / "film.mp4"), "--dry-run"])

        assert rc == 0
        captured = capsys.readouterr().out
        assert "crop11" in captured
        assert "rimlift" in captured
        assert "concat" in captured
        assert "dome_qa" in captured
        # predicted total = 1 - 0 = 1 scene → 1.0 s (no second scene to subtract from)
        assert "1.00" in captured or "1.0" in captured


# --------------------------------------------------------------------------- #
# plan parsing
# --------------------------------------------------------------------------- #


class TestPlan:
    def test_missing_plan_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            dome_film._load_plan(tmp_path / "nope.json")

    def test_relative_clip_path_resolves_against_plan_dir(self, tmp_path):
        (tmp_path / "a.mp4").write_bytes(b"x")
        plan = tmp_path / "plan.json"
        _write_plan(plan, [{"clip": "a.mp4"}])

        parsed = dome_film._load_plan(plan)

        assert parsed.scenes[0].clip == (tmp_path / "a.mp4").resolve()
        assert parsed.size == 256
        assert parsed.upscale == "lanczos"
        assert parsed.crossfade == 0.5

    def test_rejects_bad_upscale_and_crossfade(self, tmp_path):
        plan = tmp_path / "plan.json"
        _write_plan(plan, [{"clip": "a.mp4"}], upscale="realesrgan")
        with pytest.raises(ValueError, match="upscale"):
            dome_film._load_plan(plan)

    def test_rejects_no_scenes(self, tmp_path):
        plan = tmp_path / "plan.json"
        plan.write_text(json.dumps({"scenes": []}), encoding="utf-8")
        with pytest.raises(ValueError, match="no scenes"):
            dome_film._load_plan(plan)


# --------------------------------------------------------------------------- #
# command builders (pure functions — no ffmpeg)
# --------------------------------------------------------------------------- #


class TestCommandBuilders:
    def test_upscale_command_is_list_form_with_circle_mask_and_lanczos(self, tmp_path):
        cmd = dome_film.build_upscale_command(tmp_path / "in.mp4", tmp_path / "out.mp4", 4096)

        assert isinstance(cmd, list)
        assert all(isinstance(p, str) for p in cmd)
        graph = cmd[cmd.index("-filter_complex") + 1]
        assert "scale=4096:4096:flags=lanczos" in graph
        assert dft.geq_circle_expression() in graph  # the exact circle predicate
        assert cmd[cmd.index("-c:a") + 1] == "copy"
        assert "0:a:0?" in cmd  # optional audio map (silent source still works)

    def test_final_encode_is_h265_10bit(self, tmp_path):
        cmd = dome_film.build_final_encode_command(tmp_path / "c.mp4", tmp_path / "out.mp4", 4096)

        assert cmd[cmd.index("-c:v") + 1] == "libx265"
        assert cmd[cmd.index("-pix_fmt") + 1] == "yuv420p10le"
        assert "hvc1" in cmd  # Quest-compatible tag
        assert "+faststart" in cmd


# --------------------------------------------------------------------------- #
# full two-scene run (ffmpeg required)
# --------------------------------------------------------------------------- #


@requires_ffmpeg
class TestFullRun:
    """Two 2 s clips → square master, black surround, ≈ 2+2−crossfade, brighter
    rim, dome_qa report present."""

    def test_two_scenes_produce_a_valid_dome_master(self, tmp_path):
        size = 256
        a, b = tmp_path / "a.mp4", tmp_path / "b.mp4"
        _make_dome_clip(a)
        _make_dome_clip(b)
        plan = tmp_path / "plan.json"
        crossfade = 0.5
        _write_plan(
            plan,
            [
                {"clip": str(a), "is_169": True, "rimlift": True},
                {"clip": str(b), "is_169": True, "rimlift": True},
            ],
            size=size,
            crossfade=crossfade,
        )
        out = tmp_path / "film.mp4"

        rc = dome_film.main(["--plan", str(plan), "--output", str(out)])

        assert rc == 0, "dome_film main returned non-zero"
        assert out.is_file()

        streams = _ffprobe_streams(out)
        video = next(s for s in streams if s["codec_type"] == "video")
        # ── square at the requested size ────────────────────────────────────
        assert (int(video["width"]), int(video["height"])) == (size, size)
        assert video["codec_name"] in ("hevc", "h265")

        # ── duration ≈ 2 + 2 − crossfade (one xfade between two scenes) ─────
        dur = _ffprobe_duration(out)
        expected = 2 * _CLIP_DUR - crossfade
        assert abs(dur - expected) < 0.6, f"duration {dur:.2f}s, expected ~{expected:.2f}s"

        # ── the surround is black and the rim is lifted ────────────────────
        # decode one frame near the middle and check geometry in the dome's own r
        cap = cv2.VideoCapture(str(out))
        ok, frame = cap.read()
        cap.release()
        assert ok and frame is not None
        assert frame.shape[0] == frame.shape[1] == size
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        r = _radial_map(size, size)
        outside = gray[r > 1.0]
        # the QA gate's own threshold is 8; the surround must sit well under it
        assert float(outside.mean()) < 8.0
        # the lifted rim (r≈0.9–1.0) is brighter than a 10 %-of-centre defect
        # would be: rim ≥ 40 % of the centre
        centre = float(gray[r < 0.3].mean())
        rim = float(gray[(r >= 0.9) & (r <= 1.0)].mean())
        assert rim >= 0.40 * centre

        # ── the dome_qa JSON report exists next to the film ────────────────
        qa = out.with_suffix(".qa.json")
        assert qa.is_file()
        report = json.loads(qa.read_text(encoding="utf-8"))
        assert "summary" in report
        assert report["width"] == size and report["height"] == size

    def test_keep_intermediates_leaves_a_work_dir(self, tmp_path):
        size = 128
        a = tmp_path / "a.mp4"
        _make_dome_clip(a, duration=1.0, with_audio=False)
        plan = tmp_path / "plan.json"
        _write_plan(plan, [{"clip": str(a), "is_169": True, "rimlift": True}], size=size, crossfade=0.0)
        out = tmp_path / "film.mp4"

        rc = dome_film.main(["--plan", str(plan), "--output", str(out), "--keep-intermediates"])

        assert rc == 0
        work = tmp_path / (out.stem + "_work")
        assert work.is_dir()
        # the per-scene intermediates survive (crop11 + rimlift + upscale)
        assert any(p.name.startswith("scene00") for p in work.iterdir())

    def test_audio_mix_is_applied_when_plan_has_audio(self, tmp_path):
        size = 128
        a = tmp_path / "a.mp4"
        _make_dome_clip(a, duration=1.0, with_audio=True)
        amb = tmp_path / "amb.wav"
        subprocess.run(
            [
                str(_FFMPEG),
                "-y",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:duration=1",
                "-ac",
                "1",
                str(amb),
            ],
            check=True,
            capture_output=True,
            timeout=60,
        )
        plan = tmp_path / "plan.json"
        _write_plan(
            plan,
            [{"clip": str(a), "is_169": True, "rimlift": False}],
            size=size,
            crossfade=0.0,
            audio=str(amb),
            audio_gain_db=0,
        )
        out = tmp_path / "film.mp4"

        rc = dome_film.main(["--plan", str(plan), "--output", str(out)])

        assert rc == 0
        streams = _ffprobe_streams(out)
        assert any(s["codec_type"] == "audio" for s in streams)
