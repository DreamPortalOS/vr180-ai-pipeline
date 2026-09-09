"""Tests for pipeline.fulldome_mapper — FulldomeMapper class.

Unit tests mock ffmpeg/ffprobe. The D-1 (#334) geometry tests render a
synthetic target through the **real** ffmpeg and measure the result, because
the four defects #334 fixed were all invisible to string assertions:

=====================================  =====================  =====================
measurement (512² ring target)          before #334            after #334
=====================================  =====================  =====================
corners (outside the image circle)      0.00000% pure black    100.00000% pure black
outermost content radius, dome_fov=180  1.4115 R (= √2)        0.7541 R (the honest
                                                               120°-patch edge)
outermost content radius, dome_fov=90   1.4115 R               1.0000 R (the rim)
``iv_fov`` for a 16:9 source            67.5° (linear ratio)   88.51° (pinhole)
audio streams in the output             none (``-an``)         copied through
=====================================  =====================  =====================

The same table at the production size (4096², dome_fov=180) reads 0.00000% /
1.4139 R before and 100.00000% / 0.7530 R after.

Every number above was measured first and only then written down as an
assertion.  ``TestBlackFillMutations`` re-renders the graph with one element
surgically removed and asserts each measurement goes back to its "before"
value, so neither the black composite nor the circle mask can rot into a
decoration that no test would miss.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import unittest.mock
from pathlib import Path

import numpy as np
import pytest

from pipeline.equirectangular_mapper import EquirectangularMapper
from pipeline.fulldome_mapper import FulldomeMapper


def _ffmpeg_v360_available() -> bool:
    if not shutil.which("ffmpeg"):
        return False
    try:
        out = subprocess.run(["ffmpeg", "-filters"], capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return False
    return "v360" in out and "geq" in out


_FFMPEG = pytest.mark.skipif(not _ffmpeg_v360_available(), reason="ffmpeg with v360+geq unavailable")

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def dummy_input(tmp_path: Path) -> str:
    """Create a minimal valid-looking input path (file does NOT exist)."""
    return str(tmp_path / "nonexistent.mp4")


@pytest.fixture
def real_input(tmp_path: Path) -> str:
    """Create a real tiny test video with ffmpeg for integration tests."""
    mp4 = tmp_path / "test_src.mp4"
    # Generate a 1-second 320×240 test video (color bars, no audio)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=1:size=320x240:rate=30",
            "-frames:v",
            "5",
            "-pix_fmt",
            "yuv420p",
            str(mp4),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return str(mp4)


# ---------------------------------------------------------------------------
# Unit tests — construction & defaults
# ---------------------------------------------------------------------------


class TestFulldomeMapperConstruction:
    def test_default_params(self):
        mapper = FulldomeMapper()
        assert mapper.dome_fov == 180.0
        assert mapper.coverage_h_fov == 120.0
        assert mapper.coverage_v_fov is None
        assert mapper.output_size == 4096
        assert mapper.codec == "h264"
        assert mapper.crf == 18

    def test_custom_params(self):
        mapper = FulldomeMapper(
            dome_fov=200.0,
            coverage_h_fov=100.0,
            coverage_v_fov=60.0,
            output_size=2048,
            codec="h265",
            crf=22,
        )
        assert mapper.dome_fov == 200.0
        assert mapper.coverage_h_fov == 100.0
        assert mapper.coverage_v_fov == 60.0
        assert mapper.output_size == 2048
        assert mapper.codec == "h265"
        assert mapper.crf == 22

    def test_odd_output_size_is_rounded_up(self):
        """ffmpeg requires even dimensions; odd should be bumped by +1."""
        mapper = FulldomeMapper(output_size=2047)
        assert mapper.output_size == 2048

    def test_even_output_size_stays_unchanged(self):
        mapper = FulldomeMapper(output_size=4096)
        assert mapper.output_size == 4096


# ---------------------------------------------------------------------------
# Unit tests — input validation
# ---------------------------------------------------------------------------


class TestFulldomeMapperInputValidation:
    def test_nonexistent_input_raises(self):
        mapper = FulldomeMapper()
        with pytest.raises(FileNotFoundError, match="not found"):
            mapper.convert("C:\\nonexistent\\file.mp4", "out.mp4")


# ---------------------------------------------------------------------------
# Unit tests — _probe_coverage_v_fov (mocked ffprobe)
# ---------------------------------------------------------------------------


def _probe(mapper: FulldomeMapper, width: int, height: int) -> float:
    """Run ``_probe_coverage_v_fov`` against a mocked ffprobe of *width*×*height*."""
    with unittest.mock.patch("subprocess.run") as mock_run:
        mock_run.return_value = unittest.mock.Mock(
            returncode=0,
            stdout=json.dumps({"streams": [{"width": width, "height": height}]}),
        )
        return mapper._probe_coverage_v_fov("dummy.mp4")


class TestProbeCoverageVFov:
    """D-1 (#334) defect 3: ``iv_fov`` must be the pinhole solve, not a ratio."""

    def test_16_9_source(self):
        """The headline regression: 1920×1080 at ``ih_fov=120``.

        ``2*atan(tan(60°) * 1080/1920)`` = 88.51°.  The old linear rule gave
        ``120 * 0.5625`` = 67.5° — 21° of vertical field silently squashed.
        """
        result = _probe(FulldomeMapper(coverage_h_fov=120.0), 1920, 1080)
        assert abs(result - 88.5) < 0.5, f"expected the pinhole 88.5°, got {result}"
        assert abs(result - 67.5) > 1.0, "still using the linear ih_fov × aspect rule"

    def test_4_3_source(self):
        """640×480: pinhole 104.82°, the old linear rule said 90.0°."""
        result = _probe(FulldomeMapper(coverage_h_fov=120.0), 640, 480)
        assert abs(result - 104.82) < 0.01
        assert abs(result - 90.0) > 1.0

    def test_square_source_is_the_horizontal_fov_itself(self):
        """``h/w == 1`` makes both formulas agree — which is exactly why this
        bug survived: every house source is 1:1.
        """
        assert abs(_probe(FulldomeMapper(coverage_h_fov=120.0), 2880, 2880) - 120.0) < 1e-9

    def test_portrait_source(self):
        """9:16: pinhole ``2*atan(tan(60°)*16/9)`` = 144.02° (linear: 213.3°,
        which is not even a legal pinhole field).
        """
        assert abs(_probe(FulldomeMapper(coverage_h_fov=120.0), 1080, 1920) - 144.02) < 0.01

    @pytest.mark.parametrize("w,h", [(1920, 1080), (640, 480), (2880, 2880), (1080, 1920), (1280, 720)])
    def test_matches_the_vr180_implementation_exactly(self, w, h):
        """Reuse, not a copy: the dome number must be *bit-identical* to the
        VR180 one, so the two cannot drift the way #334 found them drifted.
        """
        h_fov = 120.0
        expected = EquirectangularMapper(src_hfov=h_fov, use_ffmpeg=False)._calc_vertical_fov(w, h)
        assert _probe(FulldomeMapper(coverage_h_fov=h_fov), w, h) == expected

    def test_ffprobe_failure_falls_back_to_90(self):
        mapper = FulldomeMapper(coverage_h_fov=120.0)
        with unittest.mock.patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("ffprobe not found")
            result = mapper._probe_coverage_v_fov("dummy.mp4")
            assert result == 90.0

    def test_malformed_ffprobe_output_falls_back(self):
        mapper = FulldomeMapper(coverage_h_fov=120.0)
        with unittest.mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = unittest.mock.Mock(
                returncode=0,
                stdout="not valid json",
            )
            result = mapper._probe_coverage_v_fov("dummy.mp4")
            assert result == 90.0

    def test_empty_streams_falls_back(self):
        mapper = FulldomeMapper(coverage_h_fov=120.0)
        with unittest.mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = unittest.mock.Mock(
                returncode=0,
                stdout=json.dumps({"streams": []}),
            )
            result = mapper._probe_coverage_v_fov("dummy.mp4")
            assert result == 90.0


# ---------------------------------------------------------------------------
# Unit tests — convert() with mocked subprocess
# ---------------------------------------------------------------------------


class TestConvertMocked:
    def test_convert_success(self, tmp_path: Path):
        """Verify the ffmpeg command is constructed correctly."""
        src = tmp_path / "input.mp4"
        src.write_text("fake video content")
        out = str(tmp_path / "output_dome.mp4")

        mapper = FulldomeMapper(
            coverage_h_fov=120.0,
            coverage_v_fov=75.0,  # explicit, no probing needed
            output_size=2048,
            codec="h264",
            crf=18,
        )

        with unittest.mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = unittest.mock.Mock(returncode=0, stdout="", stderr="")

            result = mapper.convert(str(src), out)

            assert result == out

            # Verify the subprocess command
            call_args = mock_run.call_args[0][0]
            assert "ffmpeg" in call_args[0]
            assert "-i" in call_args
            assert str(src) in call_args
            assert (
                "v360=input=flat:output=fisheye:ih_fov=120:iv_fov=75:h_fov=180:v_fov=180"
                ":w=2048:h=2048:interp=lanczos:alpha_mask=1" in " ".join(call_args)
            )
            assert out in call_args

    def test_audio_is_copied_not_dropped(self, tmp_path: Path):
        """D-1 (#334) defect 2: ``-an`` made every dome master silent."""
        src = tmp_path / "input.mp4"
        src.write_text("fake")
        out = str(tmp_path / "out.mp4")

        with unittest.mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = unittest.mock.Mock(returncode=0, stdout="", stderr="")
            FulldomeMapper(coverage_v_fov=75.0).convert(str(src), out)

        cmd = mock_run.call_args[0][0]
        assert "-an" not in cmd
        assert "0:a:0?" in cmd, "the source audio stream is never mapped"
        assert cmd[cmd.index("-c:a") + 1] == "copy", "audio must be a lossless passthrough"

    def test_the_video_stream_comes_from_the_filtergraph(self, tmp_path: Path):
        """Adding ``-map`` for audio makes the *video* map mandatory too — an
        unmapped filter output would silently ship the raw input instead.
        """
        src = tmp_path / "input.mp4"
        src.write_text("fake")
        out = str(tmp_path / "out.mp4")

        with unittest.mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = unittest.mock.Mock(returncode=0, stdout="", stderr="")
            mapper = FulldomeMapper(coverage_v_fov=75.0)
            mapper.convert(str(src), out)

        cmd = mock_run.call_args[0][0]
        assert f"[{mapper.OUT_LABEL}]" in cmd
        assert cmd[cmd.index("-filter_complex") + 1] == mapper._filter_complex(75.0)

    def test_convert_success_with_probe(self, tmp_path: Path):
        """When coverage_v_fov is None, the mapper should probe first."""
        src = tmp_path / "input.mp4"
        src.write_text("fake video content")
        out = str(tmp_path / "output_dome.mp4")

        mapper = FulldomeMapper(
            coverage_h_fov=120.0,
            coverage_v_fov=None,  # will probe
            output_size=2048,
        )

        with unittest.mock.patch("subprocess.run") as mock_run:
            # First call is ffprobe → return 16:9 source
            probe_result = unittest.mock.Mock(
                returncode=0,
                stdout=json.dumps({"streams": [{"width": 1920, "height": 1080}]}),
                stderr="",
            )
            # Second call is ffmpeg conversion
            ffmpeg_result = unittest.mock.Mock(returncode=0, stdout="", stderr="")
            mock_run.side_effect = [probe_result, ffmpeg_result]

            result = mapper.convert(str(src), out)
            assert result == out

            # The auto-computed iv_fov is the pinhole solve (88.51°), not the
            # linear 120 × 9/16 = 67.5° the mapper used before #334.
            ffmpeg_call = mock_run.call_args_list[1].args[0]
            ffmpeg_str = " ".join(ffmpeg_call)
            assert "iv_fov=88.5072" in ffmpeg_str
            assert "iv_fov=67.5" not in ffmpeg_str

    def test_convert_ffmpeg_failure(self, tmp_path: Path):
        src = tmp_path / "input.mp4"
        src.write_text("fake video content")
        out = str(tmp_path / "output.mp4")

        mapper = FulldomeMapper(coverage_v_fov=75.0)

        with unittest.mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = unittest.mock.Mock(returncode=1, stdout="", stderr="ffmpeg error occurred")

            with pytest.raises(RuntimeError, match="ffmpeg v360 conversion failed"):
                mapper.convert(str(src), out)

    def test_convert_h265_codec(self, tmp_path: Path):
        """Verify libx265 is used when codec='h265'."""
        src = tmp_path / "input.mp4"
        src.write_text("fake")
        out = str(tmp_path / "out.mp4")

        mapper = FulldomeMapper(coverage_v_fov=75.0, codec="h265")

        with unittest.mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = unittest.mock.Mock(returncode=0, stdout="", stderr="")
            mapper.convert(str(src), out)

            call_args = mock_run.call_args[0][0]
            assert "libx265" in call_args

    def test_subprocess_uses_list_no_shell(self, tmp_path: Path):
        """Critical: subprocess.run must receive a list, never shell=True."""
        src = tmp_path / "input.mp4"
        src.write_text("fake")
        out = str(tmp_path / "out.mp4")

        mapper = FulldomeMapper(coverage_v_fov=75.0)

        with unittest.mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = unittest.mock.Mock(returncode=0, stdout="", stderr="")
            mapper.convert(str(src), out)

            # subprocess.run should have been called with a list as first positional arg
            call_kwargs = mock_run.call_args.kwargs
            # Ensure shell is NOT True
            assert call_kwargs.get("shell") is not True, "shell=True is forbidden for security"
            # First positional arg must be a list
            call_args = mock_run.call_args[0][0]
            assert isinstance(call_args, list)


# ---------------------------------------------------------------------------
# D-1 (#334) — the filtergraph must say exactly what the geometry needs
# ---------------------------------------------------------------------------


def _v360_args(graph: str) -> dict[str, str]:
    """Parse the ``v360=k=v:...`` head of the graph into a dict.

    Parsing beats substring matching: ``"h_fov" in graph`` is *always* true
    because ``ih_fov`` contains it, so a naive "has h_fov" assertion could
    never fail.
    """
    head = graph.split("[0:v]", 1)[1].split(",")[0]
    assert head.startswith("v360="), head
    return dict(part.split("=", 1) for part in head[len("v360=") :].split(":"))


class TestFilterGraph:
    def test_uses_lanczos_not_the_default_bilinear(self):
        """D-1 (#334) defect 4 — ffmpeg's v360 default is ``line``."""
        assert _v360_args(FulldomeMapper()._filter_complex(90.0))["interp"] == "lanczos"

    def test_requests_the_out_of_fov_alpha_mask(self):
        assert _v360_args(FulldomeMapper()._filter_complex(90.0))["alpha_mask"] == "1"

    def test_reuses_the_vr180_black_composite_verbatim(self):
        """D-1 (#334) defect 1, half one.  Not "a black composite" — *the* one
        (#258).  A second copy would be a second thing to keep in sync.
        """
        assert FulldomeMapper._BLACK_COMPOSITE is EquirectangularMapper._BLACK_COMPOSITE
        assert FulldomeMapper._BLACK_COMPOSITE in FulldomeMapper()._filter_complex(90.0)

    def test_black_composite_runs_before_the_frame_is_encoded(self):
        """``alpha_mask=1`` only zeroes *alpha*, and ``yuv420p`` carries no
        alpha (#255) — so the composite has to sit between the two.
        """
        graph = FulldomeMapper()._filter_complex(90.0)
        assert graph.index("alpha_mask=1") < graph.index(FulldomeMapper._BLACK_COMPOSITE)
        assert graph.index(FulldomeMapper._BLACK_COMPOSITE) < graph.index("format=yuv420p")

    def test_circle_mask_is_multiplied_in(self):
        """D-1 (#334) defect 1, half two: the inscribed-circle mask."""
        graph = FulldomeMapper(output_size=2048)._filter_complex(90.0)
        assert "geq=" in graph
        assert "blend=all_mode=multiply" in graph
        assert "color=c=black:s=2048x2048" in graph

    @pytest.mark.parametrize("size", [512, 2048, 4096])
    def test_circle_mask_radius_is_the_inscribed_circle(self, size):
        """Half the canvas edge — not the diagonal (which is what the missing
        mask effectively amounted to: content out to 1.414 R).
        """
        assert f",{size / 2:g}),255,0)" in FulldomeMapper(output_size=size)._circle_mask_filter()

    def test_mask_labels_do_not_collide_with_the_black_composite(self):
        """``_BLACK_COMPOSITE`` owns ``_fg``/``_bgsrc``/``_bg``; reusing one of
        those names would silently rewire the graph.
        """
        owned = {"_fg", "_bgsrc", "_bg"}
        assert owned.isdisjoint({FulldomeMapper.DOME_LABEL, FulldomeMapper.CIRCLE_LABEL, FulldomeMapper.OUT_LABEL})

    def test_mask_source_emits_a_single_frame(self):
        """``geq`` is the expensive filter here; ``r=1:d=1`` keeps it to one
        frame per render instead of one per output frame.
        """
        assert ":r=1:d=1" in FulldomeMapper()._circle_mask_filter()
        assert "repeatlast=1" in FulldomeMapper()._filter_complex(90.0)


# ---------------------------------------------------------------------------
# D-1 (#334) — measured geometry, through the real ffmpeg
# ---------------------------------------------------------------------------

_SIZE = 512  # synthetic target *and* domemaster edge


def _ring_target(path: Path, size: int = _SIZE) -> str:
    """A full-frame concentric-ring target: greys 120/220, **never black**.

    Nothing in the frame is black, so any black pixel downstream can only have
    come from the out-of-FOV fill or the circle mask (the #254 trap).  Both
    tones are achromatic, which keeps 4:2:0 chroma out of the measurement.
    """
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    r = np.hypot(xx + 0.5 - size / 2, yy + 0.5 - size / 2)
    rgb = np.dstack([np.where(((r // (size / 21)) % 2) == 0, 120, 220).astype(np.uint8)] * 3)
    assert rgb.sum(axis=2).min() > 0, "the target must not contain black, or every assertion below is vacuous"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{size}x{size}",
            "-i",
            "-",
            "-frames:v",
            "1",
            str(path),
        ],
        input=rgb.tobytes(),
        capture_output=True,
        check=True,
        timeout=60,
    )
    return str(path)


def _first_frame_rgb(path: str, size: int = _SIZE) -> np.ndarray:
    """Decode frame 0 of *path* as packed RGB (no cv2 codec surprises)."""
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True,
        check=True,
        timeout=120,
    )
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(size, size, 3)


def _radii(size: int = _SIZE) -> np.ndarray:
    """Per-pixel radius in units of R = half the canvas edge."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    return np.hypot(xx + 0.5 - size / 2, yy + 0.5 - size / 2) / (size / 2)


def _corner_black_fraction(rgb: np.ndarray) -> float:
    """Fraction of the pixels **outside** the image circle that are pure black."""
    outside = _radii(rgb.shape[0]) > 1.0
    return float((rgb[outside].sum(axis=1) == 0).mean())


def _content_radius(rgb: np.ndarray) -> float:
    """Radius of the outermost non-black pixel, in units of R."""
    lit = rgb.sum(axis=2) > 30
    assert lit.any(), "the whole frame is black — the render failed, not the geometry"
    return float(_radii(rgb.shape[0])[lit].max())


def _render(mapper: FulldomeMapper, src: str, out_path: Path, graph: str | None = None) -> np.ndarray:
    """Convert *src* and return frame 0.  *graph* swaps in a mutated filtergraph.

    ``crf=0`` (lossless) is set by the caller: the measurements below are about
    geometry, and an achromatic target through a lossless encode leaves black
    exactly black.
    """
    if graph is None:
        mapper.convert(src, str(out_path))
    else:
        with unittest.mock.patch.object(FulldomeMapper, "_filter_complex", lambda self, v: graph):
            mapper.convert(src, str(out_path))
    return _first_frame_rgb(str(out_path), mapper.output_size)


def _mapper(dome_fov: float = 180.0, size: int = _SIZE) -> FulldomeMapper:
    return FulldomeMapper(
        dome_fov=dome_fov,
        coverage_h_fov=120.0,
        coverage_v_fov=120.0,  # explicit: these tests are about geometry, not probing
        output_size=size,
        crf=0,
    )


@_FFMPEG
class TestMeasuredDomemasterGeometry:
    """The acceptance gates of #334, measured rather than asserted by string."""

    def test_corners_outside_the_image_circle_are_pure_black(self, tmp_path: Path):
        """Gate: ≥ 99.5% pure black outside the circle.  Measured 100.00000%;
        before #334 it was 0.00000% (a flat RGB(45,39,34) smear on the real
        4096² run) because there was no image circle at all.
        """
        rgb = _render(_mapper(), _ring_target(tmp_path / "src.png"), tmp_path / "dome.mp4")
        assert _corner_black_fraction(rgb) >= 0.995

    def test_content_circle_radius_is_exactly_the_inscribed_circle(self, tmp_path: Path):
        """Gate: content radius = 1.00 ± 0.02 R (was 1.414 R = the diagonal).

        ``dome_fov=90`` is the configuration that makes the mask load-bearing:
        every pixel of the square canvas — corners included — then maps to a
        ray in front of the pinhole, so the source really does reach 1.414 R
        and only the circle mask cuts it back to the rim.  Measured 1.0000 R.
        """
        rgb = _render(_mapper(dome_fov=90.0), _ring_target(tmp_path / "src.png"), tmp_path / "dome.mp4")
        assert abs(_content_radius(rgb) - 1.0) <= 0.02

    def test_content_actually_reaches_the_rim(self, tmp_path: Path):
        """The other half of "= 1.00 R": a mask that cropped to, say, 0.9 R
        would pass the bound above while quietly throwing away real content.
        """
        rgb = _render(_mapper(dome_fov=90.0), _ring_target(tmp_path / "src.png"), tmp_path / "dome.mp4")
        r = _radii(rgb.shape[0])
        rim = (r > 0.98) & (r <= 1.0)
        assert (rgb.sum(axis=2) > 30)[rim].mean() >= 0.95

    def test_out_of_fov_annulus_inside_the_circle_is_black_too(self, tmp_path: Path):
        """The black composite's own job, and the bigger half of defect 1.

        A 120°×120° flat patch covers the dome out to 0.75 R; the annulus from
        there to the rim has no source behind it.  Before #334 that 44% of the
        image circle was v360's edge-clamped smear — the "brown ring".
        """
        rgb = _render(_mapper(), _ring_target(tmp_path / "src.png"), tmp_path / "dome.mp4")
        assert _content_radius(rgb) <= 0.80

    def test_the_dome_master_is_square(self, tmp_path: Path):
        rgb = _render(_mapper(), _ring_target(tmp_path / "src.png"), tmp_path / "dome.mp4")
        assert rgb.shape[0] == rgb.shape[1] == _SIZE


@_FFMPEG
class TestBlackFillMutations:
    """Mutation checks: remove one element, watch the matching gate go red.

    Without these, both halves of defect 1 could rot into decoration — the two
    mechanisms overlap outside the circle, so the corner gate alone does *not*
    pin the black composite (the mask covers the corners on its own), and the
    annulus gate alone does not pin the mask.
    """

    def test_dropping_the_black_composite_puts_the_smear_back(self, tmp_path: Path):
        mapper = _mapper()
        graph = mapper._filter_complex(120.0)
        mutated = graph.replace("," + FulldomeMapper._BLACK_COMPOSITE, "")
        assert mutated != graph, "the mutation did not apply — the check below would be vacuous"
        assert FulldomeMapper._BLACK_COMPOSITE not in mutated

        rgb = _render(mapper, _ring_target(tmp_path / "src.png"), tmp_path / "mutated.mp4", graph=mutated)
        # Exactly the gate test_out_of_fov_annulus_inside_the_circle_is_black_too
        # asserts — the smear fills the image circle out to the rim again.
        assert _content_radius(rgb) > 0.80, "expected the out-of-FOV smear back"
        with pytest.raises(AssertionError):
            assert _content_radius(rgb) <= 0.80

    def test_dropping_the_circle_mask_lets_the_corners_leak(self, tmp_path: Path):
        mapper = _mapper(dome_fov=90.0)
        mutated = (
            f"[0:v]{mapper._v360_filter(120.0)},{FulldomeMapper._BLACK_COMPOSITE}"
            f",format=yuv420p[{FulldomeMapper.OUT_LABEL}]"
        )
        assert "geq=" not in mutated and "blend=" not in mutated

        rgb = _render(mapper, _ring_target(tmp_path / "src.png"), tmp_path / "mutated.mp4", graph=mutated)
        # Exactly the gates TestMeasuredDomemasterGeometry asserts.
        assert _corner_black_fraction(rgb) < 0.995, "expected the corners to leak without the mask"
        assert abs(_content_radius(rgb) - 1.0) > 0.02, "expected content back out at the 1.414 R diagonal"
        with pytest.raises(AssertionError):
            assert _corner_black_fraction(rgb) >= 0.995


@_FFMPEG
class TestAudioSurvivesTheConversion:
    """D-1 (#334) defect 2, end to end through the real encoder."""

    @staticmethod
    def _with_audio(tmp_path: Path) -> str:
        src = tmp_path / "with_audio.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=320x180:rate=24:duration=1",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:duration=1",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-shortest",
                str(src),
            ],
            capture_output=True,
            check=True,
            timeout=120,
        )
        return str(src)

    @staticmethod
    def _codec_types(path: str) -> list[str]:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "json", path],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
        return [s["codec_type"] for s in json.loads(probe.stdout)["streams"]]

    def test_the_source_audio_track_reaches_the_dome_master(self, tmp_path: Path):
        out = tmp_path / "dome.mp4"
        FulldomeMapper(output_size=256, crf=28).convert(self._with_audio(tmp_path), str(out))
        assert "audio" in self._codec_types(str(out))
        assert "video" in self._codec_types(str(out))

    def test_a_silent_source_still_converts(self, tmp_path: Path):
        """``-map 0:a:0?`` — the ``?`` is what keeps a track-less source working."""
        src = tmp_path / "silent.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=320x180:rate=24:duration=1",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(src),
            ],
            capture_output=True,
            check=True,
            timeout=120,
        )
        out = tmp_path / "dome.mp4"
        FulldomeMapper(output_size=256, crf=28).convert(str(src), str(out))
        assert self._codec_types(str(out)) == ["video"]


# ---------------------------------------------------------------------------
# Integration test — actual ffmpeg (marked slow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestFulldomeMapperIntegration:
    """These tests use real ffmpeg to generate input and verify output.

    Run with:  pytest tests/test_fulldome_mapper.py -v -m slow
    """

    def test_output_is_square(self, real_input: str, tmp_path: Path):
        """The domemaster output must be square (width == height)."""
        out = str(tmp_path / "dome.mp4")
        mapper = FulldomeMapper(
            coverage_h_fov=120.0,
            coverage_v_fov=75.0,
            output_size=512,
            crf=28,
        )
        mapper.convert(real_input, out)
        assert Path(out).exists()

        # Probe output dimensions with ffprobe
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "json",
                out,
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
        info = json.loads(probe.stdout)
        w = info["streams"][0]["width"]
        h = info["streams"][0]["height"]
        assert w == h, f"Domemaster must be square, got {w}×{h}"

    def test_output_has_no_stereo_boxes(self, real_input: str, tmp_path: Path):
        """Fulldome output must NOT contain sv3d or st3d metadata boxes."""
        out = str(tmp_path / "dome.mp4")
        mapper = FulldomeMapper(
            coverage_h_fov=120.0,
            coverage_v_fov=75.0,
            output_size=512,
            crf=28,
        )
        mapper.convert(real_input, out)

        # Check for spherical video metadata boxes with ffprobe
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "stream=codec_name,width,height",
                "-of",
                "json",
                out,
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
        info = json.loads(probe.stdout)
        # Just having a valid video stream with no spherical metadata is sufficient
        assert len(info.get("streams", [])) >= 1

        # Also verify no sv3d/st3d in binary by grepping ffprobe side data output
        side_data = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream_side_data_list",
                "-of",
                "json",
                out,
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
        # If there are no side data entries, the output is clean
        side_json = json.loads(side_data.stdout) if side_data.stdout.strip() else {}
        streams = side_json.get("streams", [])
        if streams and "side_data_list" in streams[0]:
            for sd in streams[0]["side_data_list"]:
                assert "sv3d" not in json.dumps(sd).lower()
                assert "st3d" not in json.dumps(sd).lower()
