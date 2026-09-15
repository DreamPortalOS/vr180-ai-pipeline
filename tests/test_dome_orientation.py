"""D-2 (#371): the dome route's ``--dome-input-projection`` and ``--dome-pitch/yaw/roll``.

The dome route shipped with exactly one knob for the source (``--dome-coverage-h``)
and none at all for where the source sits on the dome: every master put the
frame's centre at the zenith, which is the one place a seated audience is not
looking.  D-2 adds the projection selector and the three angles, and
:class:`~pipeline.fulldome_mapper.FulldomeMapper` implements neither itself —
both delegate to the VR180 side (``_vr180_geometry``), so what these tests have
to prove is that the delegation lands where the geometry says it should and
that it costs an existing dome run nothing.

**The landing points are a closed form, not a recording.**  The output of the
dome graph is ``output=fisheye`` spanning ``dome_fov``, i.e. equidistant:
a direction ``θ`` off the optical axis lands at radius
``θ / (dome_fov/2) · R`` px from the canvas centre, at the azimuth of the
direction.  :func:`_dome_pixel` is that one line, and the rotation in front of
it is :func:`~tests.test_sphere_orientation._reference_rotation` — the
independent re-derivation of v360's measured ``rorder=ypr`` convention (#324),
which imports nothing from the mapper.  Neither function has an ffmpeg opinion
in it.

Measured against the real filter at 512², ``dome_fov=180``, over both
projections and the angle sets below, the closed form and the render agree to

=========================  ==========  ==========
marker                      worst |Δ|   worst |Δ|
                            (rect.)     (fisheye)
=========================  ==========  ==========
``forward`` (on axis)       0.39 px     0.10 px
``high`` (30° up)           0.56 px     0.28 px
``off_axis`` (35°, 30° az)  0.32 px     0.40 px
=========================  ==========  ==========

— so the 2 px gate the tests assert is ~4× the worst residual, comfortably
above sub-pixel encode noise and far below the tens of pixels a sign flip or a
reordering of the three rotations would cost.  ``TestOrientationMutations``
strips the orientation tail back out of the filtergraph and asserts the same
measurements go red, so the pass-through cannot rot into a no-op.

``TestZeroRegressionAtZero`` is the other half: at the default the emitted
filtergraph must be the **literal** ``origin/main`` one, pinned below as a
string rather than rebuilt from the code under test.
"""

from __future__ import annotations

import hashlib
import math
import shutil
import subprocess
import unittest.mock
from pathlib import Path

import numpy as np
import pytest

from pipeline.fulldome_mapper import FulldomeMapper

# The VR180 orientation suite already owns an independent reference rotation
# and a pair of synthetic sources carrying markers at known directions (#323).
# Importing them keeps *one* re-derivation of v360's convention in the repo:
# a second copy here could drift from the first and both could still be wrong
# together.  Nothing imported below touches the mapper.
from tests.test_sphere_orientation import (
    _MARKERS,
    _RECT_HFOV,
    _centroid,
    _direction,
    _fisheye_source,
    _rectilinear_source,
    _reference_rotation,
)


def _ffmpeg_v360_available() -> bool:
    if not shutil.which("ffmpeg"):
        return False
    try:
        out = subprocess.run(["ffmpeg", "-filters"], capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return False
    return "v360" in out and "geq" in out


_FFMPEG = pytest.mark.skipif(not _ffmpeg_v360_available(), reason="ffmpeg with v360+geq unavailable")

_SIZE = 512  # synthetic source *and* domemaster edge
_DOME_FOV = 180.0  # the inscribed circle spans the whole hemisphere
_TOLERANCE_PX = 2.0  # the acceptance gate; worst measured residual is 0.56 px

#: ``--dome-input-projection`` → (source builder, the mapper kwargs that read it).
#: ``coverage_v_fov`` is given explicitly so these tests measure the rotation
#: and not ``_probe_coverage_v_fov`` (which has its own tests in
#: ``test_fulldome_mapper.py``).  The spans match how each source was actually
#: drawn: a 90° pinhole patch, and a 180° equidistant image circle.
_SOURCES: dict[str, tuple] = {
    "rectilinear": (
        _rectilinear_source,
        {"input_projection": "rectilinear", "coverage_h_fov": _RECT_HFOV, "coverage_v_fov": _RECT_HFOV},
    ),
    "fisheye": (
        _fisheye_source,
        {"input_projection": "fisheye", "coverage_h_fov": 180.0, "coverage_v_fov": 180.0},
    ),
}

_PROJECTIONS = pytest.mark.parametrize("projection", sorted(_SOURCES))


# --------------------------------------------------------------------------- #
# The closed form — where a known source direction must land on the dome
# --------------------------------------------------------------------------- #


def _dome_pixel(direction: np.ndarray, size: int = _SIZE, dome_fov: float = _DOME_FOV) -> tuple[float, float]:
    """Where a unit *output* direction lands on an equidistant domemaster.

    ``output=fisheye`` is equidistant over ``dome_fov``: radius is proportional
    to the polar angle off the optical axis, with the rim (``dome_fov/2``) at
    ``R = size/2``.  Image ``y`` runs down, so a direction pointing up the
    sphere sits *above* the centre — hence the minus.
    """
    x, y, z = (float(c) for c in direction)
    polar = math.degrees(math.acos(max(-1.0, min(1.0, z))))
    azimuth = math.atan2(y, x)
    r = polar / (dome_fov / 2.0) * (size / 2.0)
    return size / 2.0 + r * math.cos(azimuth), size / 2.0 - r * math.sin(azimuth)


def _expected_landing(marker: str, *, yaw: float = 0.0, pitch: float = 0.0, roll: float = 0.0) -> tuple[float, float]:
    """Output ``(x, y)`` px of *marker*, whose source direction is known exactly.

    The rotation maps output→source, so a given source direction is seen at
    ``R⁻¹ d`` — and ``R`` is orthonormal, so ``R⁻¹ == Rᵀ`` (the same step
    ``test_sphere_orientation`` takes for the hequirect frame).
    """
    source_dir = next(_direction(p, a) for name, _rgb, p, a in _MARKERS if name == marker)
    return _dome_pixel(_reference_rotation(yaw, pitch, roll).T @ source_dir)


def _marker_colour(marker: str) -> tuple[int, int, int]:
    return next(rgb for name, rgb, _p, _a in _MARKERS if name == marker)


# --------------------------------------------------------------------------- #
# Rendering through the real ffmpeg
# --------------------------------------------------------------------------- #


def _write_source(img: np.ndarray, path: Path) -> str:
    """Write a synthetic source frame out as a single-frame PNG."""
    h, w = img.shape[:2]
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-i", "-",
        "-frames:v", "1", str(path),
    ]  # fmt: skip
    subprocess.run(cmd, input=img.tobytes(), capture_output=True, check=True, timeout=60)
    return str(path)


def _first_frame_rgb(path: Path, size: int = _SIZE) -> np.ndarray:
    """Decode frame 0 of *path* as packed RGB."""
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True,
        check=True,
        timeout=120,
    ).stdout
    return np.frombuffer(out, dtype=np.uint8).reshape(size, size, 3)


def _render_graph(src: str, graph: str, out: Path) -> bytes:
    """Run *src* through *graph* and return the raw rgb24 bytes it produces.

    Deliberately not via :meth:`FulldomeMapper.convert`: the zero-regression
    comparison has to be able to run ``origin/main``'s literal graph, which no
    longer has any code to run it.  Raw rgb24 out (no encoder) keeps the
    comparison about the filter chain.
    """
    cmd = [
        "ffmpeg", "-y", "-v", "error", "-i", src,
        "-filter_complex", graph, "-map", f"[{FulldomeMapper.OUT_LABEL}]",
        "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", str(out),
    ]  # fmt: skip
    subprocess.run(cmd, capture_output=True, check=True, timeout=120)
    return out.read_bytes()


def _dome_mapper(projection: str, **angles: float) -> FulldomeMapper:
    _build, kw = _SOURCES[projection]
    return FulldomeMapper(dome_fov=_DOME_FOV, output_size=_SIZE, crf=0, **kw, **angles)


@pytest.fixture(scope="module")
def dome_render(tmp_path_factory):
    """``(projection, **angles) -> frame 0``, memoised for the module.

    Each render is a real 512² ffmpeg pass; the angle sets below are shared by
    several assertions each, so caching turns ~30 renders into 8.  ``crf=0``
    keeps the encode lossless — the markers are chroma-heavy and the
    measurement is sub-pixel.
    """
    root = tmp_path_factory.mktemp("dome_orientation")
    sources: dict[str, str] = {}
    frames: dict[tuple, np.ndarray] = {}

    def _render(projection: str, **angles: float) -> np.ndarray:
        key = (projection, tuple(sorted(angles.items())))
        if key not in frames:
            if projection not in sources:
                build, _kw = _SOURCES[projection]
                sources[projection] = _write_source(build(), root / f"src_{projection}.png")
            tag = "_".join(f"{k}{v:g}" for k, v in sorted(angles.items())) or "zero"
            out = root / f"{projection}_{tag}.mp4"
            _dome_mapper(projection, **angles).convert(sources[projection], str(out))
            frames[key] = _first_frame_rgb(out)
        return frames[key]

    return _render


def _landing_error(frame: np.ndarray, marker: str, **angles: float) -> float:
    """Distance in px between where *marker* actually landed and the closed form."""
    got = _centroid(frame, _marker_colour(marker))
    assert got is not None, f"the {marker} marker is not in the rendered dome master at all"
    want = _expected_landing(marker, **angles)
    return math.hypot(got[0] - want[0], got[1] - want[1])


# --------------------------------------------------------------------------- #
# 1. Geometry — the markers land where the closed form says, for both inputs
# --------------------------------------------------------------------------- #

#: Angle sets exercised end to end.  Each isolates one axis, and the last one
#: fires all three at once so a composition order that happens to agree on the
#: single-axis cases (three of the six do, on one axis or another) still fails.
_ANGLE_SETS = [
    {"pitch": 20.0},
    {"pitch": -15.0},
    {"yaw": 15.0},
    {"roll": 30.0},
    {"pitch": 20.0, "yaw": 15.0, "roll": 10.0},
]


@_FFMPEG
class TestDomeOrientationLandsOnTheClosedForm:
    """The acceptance gate of D-2: ``--dome-pitch 20`` really tilts the dome."""

    @_PROJECTIONS
    def test_the_unrotated_master_already_matches_the_closed_form(self, projection, dome_render):
        """Guard the guard: if the closed form were wrong about the *output*
        projection, every rotated assertion below would be measuring the same
        error twice and could still pass.  Pinning the 0/0/0 render first
        separates "the fisheye formula is right" from "the rotation is right".
        """
        frame = dome_render(projection)
        for marker, _rgb, _p, _a in _MARKERS:
            assert _landing_error(frame, marker) < _TOLERANCE_PX

    @_PROJECTIONS
    @pytest.mark.parametrize("angles", _ANGLE_SETS, ids=lambda a: "_".join(f"{k}{v:g}" for k, v in sorted(a.items())))
    def test_every_marker_lands_on_the_closed_form(self, projection, angles, dome_render):
        """The whole card, measured: three markers × five angle sets × both
        source projections, each within 2 px of pure projection arithmetic.
        """
        frame = dome_render(projection, **angles)
        for marker, _rgb, _p, _a in _MARKERS:
            error = _landing_error(frame, marker, **angles)
            assert error < _TOLERANCE_PX, f"{projection}/{marker} at {angles} missed by {error:.2f} px"

    @_PROJECTIONS
    def test_pitch_20_drops_the_source_centre_by_a_fifth_of_the_radius(self, projection, dome_render):
        """The documented meaning of the flag, stated without the matrix.

        ``dome_fov=180`` puts 90° of polar angle across ``R``, so a 20° tilt has
        to move the on-axis marker ``20/90 · R`` = 56.9 px straight down the
        centre column — and nowhere sideways.
        """
        got = _centroid(dome_render(projection, pitch=20.0), _marker_colour("forward"))
        assert got is not None
        assert abs(got[0] - _SIZE / 2.0) < _TOLERANCE_PX, "pitch moved the marker sideways"
        assert abs(got[1] - (_SIZE / 2.0 + 20.0 / 90.0 * (_SIZE / 2.0))) < _TOLERANCE_PX

    @_PROJECTIONS
    def test_roll_leaves_the_on_axis_marker_alone(self, projection, dome_render):
        """``roll`` spins the domemaster about its own centre, so the one point
        it may not move is the centre itself — the cheapest way to catch a
        roll/pitch or roll/yaw mix-up in the delegation.
        """
        got = _centroid(dome_render(projection, roll=30.0), _marker_colour("forward"))
        assert got is not None
        assert math.hypot(got[0] - _SIZE / 2.0, got[1] - _SIZE / 2.0) < _TOLERANCE_PX


@_FFMPEG
class TestOrientationMutations:
    """Strip the orientation tail back out; the measurements must go red.

    Without this the class above would still pass if ``_orientation_terms``
    quietly returned ``""`` for *every* angle — every marker would land on its
    unrotated spot and only the tests that expect movement would notice.  So
    the mutation is performed for real: the delegation is replaced by the empty
    string it emits at 0/0/0, which is exactly what "the flag never reached
    v360" looks like.

    Run against the shipped code with the mutation applied to the module (not
    just patched here), 12 of the 16 tests above fail; the 4 that survive are
    the 0/0/0 pin and ``roll`` leaving the centre alone, both of which are
    *supposed* to be insensitive to it.
    """

    @staticmethod
    def _rendered_without_orientation(projection: str, tmp_path: Path, **angles: float) -> np.ndarray:
        build, _kw = _SOURCES[projection]
        src = _write_source(build(), tmp_path / f"src_{projection}.png")
        out = tmp_path / "mutant.mp4"
        with unittest.mock.patch.object(FulldomeMapper, "_orientation_terms", lambda self: ""):
            _dome_mapper(projection, **angles).convert(src, str(out))
        return _first_frame_rgb(out)

    @_PROJECTIONS
    def test_dropping_the_orientation_tail_misses_the_pitch_landing(self, projection, tmp_path: Path):
        """``--dome-pitch 20`` has to move the on-axis marker 56.9 px, so the
        un-tilted render misses its closed form by ~28× the 2 px gate.
        """
        frame = self._rendered_without_orientation(projection, tmp_path, pitch=20.0)
        assert _landing_error(frame, "forward", pitch=20.0) > 10 * _TOLERANCE_PX

    @_PROJECTIONS
    def test_dropping_the_orientation_tail_misses_the_three_axis_landing(self, projection, tmp_path: Path):
        """... and the same for all three angles at once, so a mutation that
        only broke ``pitch`` could not hide behind ``yaw``/``roll``.
        """
        angles = {"pitch": 20.0, "yaw": 15.0, "roll": 10.0}
        frame = self._rendered_without_orientation(projection, tmp_path, **angles)
        assert max(_landing_error(frame, m, **angles) for m, _rgb, _p, _a in _MARKERS) > 10 * _TOLERANCE_PX


# --------------------------------------------------------------------------- #
# 2. Zero regression — the default filtergraph is still origin/main's, byte for
#    byte
# --------------------------------------------------------------------------- #

#: ``origin/main``'s (pre-#371) filtergraph at 512² with an explicit 120°×120°
#: coverage — captured from ``git show origin/main:pipeline/fulldome_mapper.py``
#: and written out **as a literal**, not rebuilt from the code under test.  The
#: point of the pin is that it survives a change to the code that generates it.
_MAIN_512_GRAPH = (
    "[0:v]v360=input=flat:output=fisheye:ih_fov=120:iv_fov=120:h_fov=180:v_fov=180"
    ":w=512:h=512:interp=lanczos:alpha_mask=1"
    ",split[_fg][_bgsrc];[_bgsrc]lutrgb=r=0:g=0:b=0[_bg]"
    ";[_bg][_fg]overlay=format=rgb:eof_action=endall,format=gbrp[_dome]"
    ";color=c=black:s=512x512:r=1:d=1,format=gbrp"
    ",geq=r='if(lte(hypot(X-255.5,Y-255.5),256),255,0)'"
    ":g='if(lte(hypot(X-255.5,Y-255.5),256),255,0)'"
    ":b='if(lte(hypot(X-255.5,Y-255.5),256),255,0)'[_circle]"
    ";[_dome][_circle]blend=all_mode=multiply:repeatlast=1,format=yuv420p[_domemaster]"
)

#: SHA-256 of ``origin/main``'s graph for the **shipping** configuration — the
#: 4096² master with the pinhole ``iv_fov`` a 16:9 source probes to (#334).
#: A digest rather than a 525-character literal: the 4096² ``geq`` expression
#: is three copies of the same unreadable line, and the 512² literal above
#: already documents the shape.
_MAIN_4096_GRAPH_SHA256 = "9083eb2e4ae8598a10b50181cfcc786263b6043302de7b9005142dfa601f58da"

#: The ``iv_fov`` a 1920×1080 source auto-probes to on the pinhole rule.
_PINHOLE_IV_FOV_16_9 = 88.51


def _graph_sha256(graph: str) -> str:
    return hashlib.sha256(graph.encode()).hexdigest()


class TestZeroRegressionAtZero:
    """D-2 may not cost an existing dome run a single pixel.

    The dome route's whole output is one filtergraph, so "byte-identical to
    main" is a statement about that string — and the string is pinned twice
    over, as a literal and as a digest of the 4096² production one.
    """

    def test_the_default_graph_is_the_literal_main_one(self):
        got = FulldomeMapper(output_size=512, coverage_h_fov=120.0, coverage_v_fov=120.0)._filter_complex(120.0)
        assert got == _MAIN_512_GRAPH

    def test_the_production_graph_hashes_to_mains(self):
        """The one that actually ships: every default, 4096², 16:9 source."""
        got = FulldomeMapper()._filter_complex(_PINHOLE_IV_FOV_16_9)
        assert _graph_sha256(got) == _MAIN_4096_GRAPH_SHA256

    def test_explicit_zeros_are_indistinguishable_from_omitting_the_flags(self):
        """``yaw=0:pitch=0:roll=0`` renders the same but *changes the string*,
        and the string is what #334's geometry tests pin.
        """
        default = FulldomeMapper()._filter_complex(_PINHOLE_IV_FOV_16_9)
        zeros = FulldomeMapper(pitch=0.0, yaw=0.0, roll=0.0)._filter_complex(_PINHOLE_IV_FOV_16_9)
        assert _graph_sha256(zeros) == _MAIN_4096_GRAPH_SHA256
        assert zeros == default
        assert ":pitch=" not in zeros and ":yaw=" not in zeros and ":roll=" not in zeros

    def test_the_default_input_projection_is_still_input_flat(self):
        """``rectilinear`` is a new *name* for the behaviour that shipped; it
        must still emit v360's ``input=flat`` token and nothing else.
        """
        explicit = FulldomeMapper(input_projection="rectilinear")._filter_complex(_PINHOLE_IV_FOV_16_9)
        assert _graph_sha256(explicit) == _MAIN_4096_GRAPH_SHA256
        assert "input=flat:" in explicit
        assert "input=rectilinear" not in explicit

    def test_the_orientation_tail_is_empty_at_the_default(self):
        assert FulldomeMapper()._orientation_terms() == ""

    @pytest.mark.parametrize("angle", ["pitch", "yaw", "roll"])
    def test_a_non_zero_angle_really_does_change_the_graph(self, angle):
        """Guard the guard: the four pins above must not pass because the
        orientation terms are never emitted at all.
        """
        rotated = FulldomeMapper(**{angle: 20.0})._filter_complex(_PINHOLE_IV_FOV_16_9)
        assert _graph_sha256(rotated) != _MAIN_4096_GRAPH_SHA256
        head = dict(part.split("=", 1) for part in rotated.split(",")[0][len("[0:v]v360=") :].split(":"))
        assert head[angle] == "20"
        assert all(head[other] == "0" for other in {"pitch", "yaw", "roll"} - {angle})

    @pytest.mark.parametrize("projection", ["fisheye", "equirect"])
    def test_a_non_default_projection_really_does_change_the_graph(self, projection):
        """The same guard for the other new flag."""
        graph = FulldomeMapper(input_projection=projection)._filter_complex(_PINHOLE_IV_FOV_16_9)
        assert _graph_sha256(graph) != _MAIN_4096_GRAPH_SHA256
        assert f"input={projection}:" in graph

    @_FFMPEG
    def test_the_default_renders_byte_identically_to_mains_graph(self, tmp_path: Path):
        """The end-to-end claim, not just the string: main's literal graph and
        the shipped default graph produce the same SHA-256 over the raw rgb24
        pixels of a real render.
        """
        src = _write_source(_rectilinear_source(), tmp_path / "src.png")
        shipped = FulldomeMapper(output_size=512, coverage_h_fov=120.0, coverage_v_fov=120.0)._filter_complex(120.0)
        reference = _render_graph(src, _MAIN_512_GRAPH, tmp_path / "main.rgb")
        assert len(reference) == _SIZE * _SIZE * 3, f"short render: {len(reference)} bytes"
        assert hashlib.sha256(_render_graph(src, shipped, tmp_path / "ship.rgb")).hexdigest() == (
            hashlib.sha256(reference).hexdigest()
        )
