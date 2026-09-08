"""W-8 (#323): ``--pitch`` / ``--yaw`` / ``--roll`` sphere orientation.

Why the card exists: on the owner's Quest test, a fisheye source stretched to
180° was the most immersive option by a distance ("completely wrapped, can't
see the edge") but came with *"the viewing angle is too low, everything is up
above, I feel like I'm underground"*.  The pipeline had ``--src-hfov`` and
nothing else — no way to tilt the sphere at all.  ``--pitch`` is that way.

Two things make the assertions here worth something.

**The landing points are theory, not a recording.**  A 180° ``hequirect``
frame spans ±90° across its width *and* its height, so an angle ``a`` off the
forward axis sits ``a / 180`` of the frame from centre.  A pitch of ``p``
therefore has to move a marker exactly ``output_height * p / 180`` px down the
centre column — pure projection arithmetic, computed in
:func:`_hequirect_pixel` / :func:`_reference_rotation` from the convention
alone, with no ffmpeg opinion in it and nothing imported from the code under
test.

**The convention itself was measured, not assumed.**  ``v360``'s rotation
sign and its ``rorder=ypr`` composition were recovered by rendering a target
with dots at known directions through the real filter, converting each
landing back to a direction, and solving for the 3×3 that maps output
directions to source directions.  Against that measured matrix the six
possible orderings separate cleanly:

===================  ==================================================
composition          max |Δ| vs the measured matrix (yaw 25/pitch 15/roll 10)
===================  ==================================================
``Ry @ Rp @ Rr``     0.003   ← ships
``Ry @ Rr @ Rp``     0.041
``Rr @ Ry @ Rp``     0.074
``Rp @ Ry @ Rr``     0.111
``Rr @ Rp @ Ry``     0.118
``Rp @ Rr @ Ry``     0.124
===================  ==================================================

:func:`_reference_rotation` below re-derives that convention independently of
the mapper, and ``test_orientation_matrix_matches_the_independent_reference``
holds the two together.

Zero regression is the load-bearing property (``TestZeroRegressionAtZero``):
at the all-zero default the v360 head must be the *literal* pre-#323 string
(pinned below, copied from ``origin/main``), the OpenCV mesh must be bit-equal
to an independently rebuilt pre-#323 mesh, and the rendered bytes must hash
identically.  Nothing about this feature may cost an existing run a pixel.
"""

from __future__ import annotations

import hashlib
import math
import shutil
import subprocess

import numpy as np
import pytest

from pipeline.equirectangular_mapper import EquirectangularMapper


def _ffmpeg_v360_available() -> bool:
    if not shutil.which("ffmpeg"):
        return False
    try:
        out = subprocess.run(["ffmpeg", "-filters"], capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return False
    return "v360" in out


_FFMPEG = pytest.mark.skipif(not _ffmpeg_v360_available(), reason="ffmpeg v360 unavailable")

_S = 512  # square source *and* output edge, so 1 output px = 180/512 deg
_R = _S / 2.0
_K = _R / 90.0  # px per degree of a 180° equidistant image circle
_GREY = (128, 128, 128)
_BOTH_PATHS = pytest.mark.parametrize("use_ffmpeg", [pytest.param(True, marks=_FFMPEG), False])


# --------------------------------------------------------------------------- #
# Independent geometry reference — nothing here imports from the mapper
# --------------------------------------------------------------------------- #


def _reference_rotation(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """The output→source rotation, re-derived from the measured v360 convention.

    Deliberately a second implementation: the mapper's own
    ``_orientation_matrix`` is checked *against* this, so a sign flip or a
    reordering in the shipped code cannot make its own test pass.
    """
    y, p, r = math.radians(yaw), math.radians(pitch), math.radians(roll)
    r_yaw = np.array([[math.cos(y), 0.0, math.sin(y)], [0.0, 1.0, 0.0], [-math.sin(y), 0.0, math.cos(y)]])
    r_pitch = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(p), math.sin(p)], [0.0, -math.sin(p), math.cos(p)]])
    r_roll = np.array([[math.cos(r), math.sin(r), 0.0], [-math.sin(r), math.cos(r), 0.0], [0.0, 0.0, 1.0]])
    return r_yaw @ r_pitch @ r_roll


def _hequirect_pixel(direction: np.ndarray) -> tuple[float, float]:
    """Where a unit *output* direction lands in a square 180° hequirect frame.

    Inverse of the ``theta``/``phi`` grid the mapper builds: ``phi`` (0 at the
    top pole) is ``acos(y)`` and the azimuth is ``atan2(x, z)``, each spanning
    the full frame over 180°.
    """
    x, y, z = (float(c) for c in direction)
    phi = math.acos(max(-1.0, min(1.0, y)))
    theta = math.atan2(x, z)
    return _S * (theta / math.pi + 0.5), _S * (phi / math.pi)


def _expected_landing(source_dir: np.ndarray, *, yaw=0.0, pitch=0.0, roll=0.0) -> tuple[float, float]:
    """Output ``(x, y)`` px of a marker whose *source* direction is known.

    The rotation maps output→source, so the output direction of a given source
    direction is ``R⁻¹ d`` — and ``R`` is orthonormal, so ``R⁻¹ == Rᵀ``.
    """
    rot = _reference_rotation(yaw, pitch, roll)
    return _hequirect_pixel(rot.T @ source_dir)


def _direction(polar_deg: float, azimuth_deg: float) -> np.ndarray:
    """Unit direction ``polar_deg`` off forward, ``azimuth_deg`` CCW from "right"."""
    p, a = math.radians(polar_deg), math.radians(azimuth_deg)
    return np.array([math.sin(p) * math.cos(a), math.sin(p) * math.sin(a), math.cos(p)])


# --------------------------------------------------------------------------- #
# Synthetic sources — one per --input-projection, each with markers whose
# source direction is known exactly
# --------------------------------------------------------------------------- #

#: Markers carried by both sources: (colour name, RGB, polar°, azimuth°).
#: ``forward`` pins the simple centre-column arithmetic; ``high`` is 30° up so
#: a pitch cannot be faked by a crop; ``off_axis`` is off *both* axes so a
#: genuine rotation is distinguishable from a row/column translation.
_MARKERS = (
    ("forward", (0, 255, 0), 0.0, 0.0),
    ("high", (255, 0, 255), 30.0, 90.0),
    ("off_axis", (255, 255, 0), 35.0, 30.0),
)


def _marker_colour(name: str) -> tuple[int, int, int]:
    return next(rgb for n, rgb, _p, _a in _MARKERS if n == name)


def _marker_direction(name: str) -> np.ndarray:
    return next(_direction(p, a) for n, _rgb, p, a in _MARKERS if n == name)


def _fisheye_source() -> np.ndarray:
    """A 180° equidistant fisheye carrying the markers above.

    Built in numpy from ``r = k·θ`` so the marker directions are ground truth,
    not a second opinion from ffmpeg.  Nothing in the frame is black, so a
    black pixel downstream can only be the out-of-FOV fill (the #254 trap).
    """
    yy, xx = np.mgrid[0:_S, 0:_S].astype(np.float64)
    dx, dy = (xx + 0.5) - _R, (yy + 0.5) - _R
    img = np.full((_S, _S, 3), _GREY, dtype=np.uint8)
    img[np.hypot(dx, dy) <= _R] = (255, 255, 255)
    for _name, rgb, polar, azimuth in _MARKERS:
        # image y is down, so a marker "up" on the sphere sits at negative dy
        cx, cy = polar * _K * math.cos(math.radians(azimuth)), -polar * _K * math.sin(math.radians(azimuth))
        img[np.hypot(dx - cx, dy - cy) <= 5] = rgb
    return img


#: Pinhole source span.  90° keeps the 35° off-axis marker comfortably inside
#: the frame while staying a realistic ``--src-hfov``.
_RECT_HFOV = 90.0


def _rectilinear_source() -> np.ndarray:
    """A square pinhole frame of :data:`_RECT_HFOV` carrying the same markers.

    ``x = f·tan(θ)`` with ``f = W / (2·tan(hfov/2))`` — the model
    ``_build_mesh`` inverts, written forwards here.
    """
    focal = _S / (2.0 * math.tan(math.radians(_RECT_HFOV) / 2.0))
    yy, xx = np.mgrid[0:_S, 0:_S].astype(np.float64)
    dx, dy = (xx + 0.5) - _R, (yy + 0.5) - _R
    img = np.full((_S, _S, 3), _GREY, dtype=np.uint8)
    for _name, rgb, polar, azimuth in _MARKERS:
        d = _direction(polar, azimuth)
        cx, cy = focal * d[0] / d[2], -focal * d[1] / d[2]
        img[np.hypot(dx - cx, dy - cy) <= 5] = rgb
    return img


_SOURCES = {
    "fisheye": (_fisheye_source, {"input_projection": "fisheye", "fisheye_fov": 180.0}),
    "rectilinear": (_rectilinear_source, {"input_projection": "rectilinear", "src_hfov": _RECT_HFOV}),
}


def test_synthetic_sources_carry_no_black_pixels():
    """Guard the guard: a black marker would make the out-of-FOV assertions vacuous."""
    for build, _kw in _SOURCES.values():
        assert not np.any(build().sum(axis=2) == 0)


def _mapper(projection: str, use_ffmpeg: bool, **angles) -> EquirectangularMapper:
    _build, kw = _SOURCES[projection]
    return EquirectangularMapper(_S, _S, use_ffmpeg=use_ffmpeg, **kw, **angles)


def _centroid(out: np.ndarray, rgb: tuple[int, int, int]) -> tuple[float, float] | None:
    """Sub-pixel centre of the marker of colour *rgb*, in pixel-centre coords.

    Only the **largest connected** run of matching pixels counts.  A plain mean
    over every match is not safe here: Lanczos ringing around one marker
    overshoots into the complementary channels, so a stray pixel or two beside
    the *green* dot reads as magenta and, sitting 70 rows away, drags a naive
    centroid ~4 px off — enough to fail a 2 px bound for a reason that has
    nothing to do with the geometry under test.  Measured on the rectilinear
    source at ``roll=30``: 27 real magenta pixels plus 1 ringing pixel.
    """
    import cv2

    a = out[:, :, :3].astype(int)
    hit = np.ones(a.shape[:2], bool)
    for i, c in enumerate(rgb):
        hit &= (a[:, :, i] > 150) if c == 255 else (a[:, :, i] < 110)
    if not hit.any():
        return None
    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(hit.astype(np.uint8), connectivity=8)
    if count < 2:  # pragma: no cover - `hit.any()` guarantees one component
        return None
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y = centroids[biggest]
    return float(x) + 0.5, float(y) + 0.5


# --------------------------------------------------------------------------- #
# The rotation convention
# --------------------------------------------------------------------------- #


class TestOrientationConvention:
    @pytest.mark.parametrize(
        "angles", [(20.0, 0.0, 0.0), (0.0, 15.0, 0.0), (0.0, 0.0, 30.0), (25.0, 15.0, 10.0), (-20.0, 20.0, -30.0)]
    )
    def test_orientation_matrix_matches_the_independent_reference(self, angles):
        yaw, pitch, roll = angles
        m = EquirectangularMapper(_S, _S, yaw=yaw, pitch=pitch, roll=roll)
        np.testing.assert_allclose(m._orientation_matrix(), _reference_rotation(yaw, pitch, roll), atol=1e-12)

    def test_no_matrix_is_built_at_the_default(self):
        assert EquirectangularMapper(_S, _S)._orientation_matrix() is None
        assert EquirectangularMapper(_S, _S)._orientation_active() is False

    @pytest.mark.parametrize("angles", [{"pitch": 1.0}, {"yaw": -1.0}, {"roll": 0.5}])
    def test_any_single_non_zero_angle_activates_the_rotation(self, angles):
        m = EquirectangularMapper(_S, _S, **angles)
        assert m._orientation_active() is True
        assert m._orientation_matrix() is not None

    def test_the_matrix_is_a_rotation(self):
        """Orthonormal with det +1 — otherwise it would scale or mirror the sphere."""
        rot = EquirectangularMapper(_S, _S, yaw=25.0, pitch=15.0, roll=10.0)._orientation_matrix()
        np.testing.assert_allclose(rot @ rot.T, np.eye(3), atol=1e-12)
        assert float(np.linalg.det(rot)) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Geometry — the acceptance criterion, on both projections and both paths
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("projection", ["fisheye", "rectilinear"])
@_BOTH_PATHS
class TestPitchLandsOnTheory:
    """``--pitch 20`` must move a marker ``H·20/180`` px **down**, ±2 px."""

    @pytest.mark.parametrize("pitch", [10.0, 20.0, -20.0])
    def test_forward_marker_row_moves_by_the_theoretical_amount(self, projection, use_ffmpeg, pitch):
        build, _kw = _SOURCES[projection]
        out = _mapper(projection, use_ffmpeg, pitch=pitch).map_single(build())
        got = _centroid(out, _marker_colour("forward"))
        assert got is not None, "the forward marker fell out of the frame"
        # Closed form, independent of _reference_rotation: the source's forward
        # direction is pitch degrees below the output's forward direction.
        expected_y = _S * (0.5 + pitch / 180.0)
        assert abs(got[1] - expected_y) <= 2.0, f"row {got[1]:.2f}, expected {expected_y:.2f}"
        assert abs(got[0] - _S * 0.5) <= 2.0, f"pitch moved the column to {got[0]:.2f}"

    def test_pitch_zero_leaves_the_forward_marker_dead_centre(self, projection, use_ffmpeg):
        """Guard the guard: the shift above must be caused by --pitch."""
        build, _kw = _SOURCES[projection]
        got = _centroid(_mapper(projection, use_ffmpeg, pitch=0.0).map_single(build()), _marker_colour("forward"))
        assert abs(got[0] - _S * 0.5) <= 2.0 and abs(got[1] - _S * 0.5) <= 2.0, got

    def test_an_elevated_marker_moves_by_the_same_amount(self, projection, use_ffmpeg):
        """A 30°-up marker under ``pitch 20`` must sit at 10° up, i.e. the shift
        is a rotation of the whole sphere and not a crop of the top.
        """
        build, _kw = _SOURCES[projection]
        out = _mapper(projection, use_ffmpeg, pitch=20.0).map_single(build())
        got = _centroid(out, _marker_colour("high"))
        assert got is not None, "the 30°-up marker fell out of the frame"
        expected_y = _S * (0.5 - (30.0 - 20.0) / 180.0)
        assert abs(got[1] - expected_y) <= 2.0, f"row {got[1]:.2f}, expected {expected_y:.2f}"

    def test_an_off_axis_marker_follows_the_full_rotation(self, projection, use_ffmpeg):
        """Off both axes a pitch is *not* a row translation — the column moves
        too.  Only a real spherical rotation lands here.
        """
        build, _kw = _SOURCES[projection]
        out = _mapper(projection, use_ffmpeg, pitch=20.0).map_single(build())
        got = _centroid(out, _marker_colour("off_axis"))
        assert got is not None, "the off-axis marker fell out of the frame"
        want = _expected_landing(_marker_direction("off_axis"), pitch=20.0)
        assert abs(got[0] - want[0]) <= 2.0 and abs(got[1] - want[1]) <= 2.0, f"{got} vs theory {want}"


@pytest.mark.parametrize("projection", ["fisheye", "rectilinear"])
@_BOTH_PATHS
class TestYawAndRollLandOnTheory:
    """Same rotation, the other two axes.  Documented as rarely needed; still
    has to be *right*, because a wrong sign here would only ever surface in a
    headset.
    """

    @pytest.mark.parametrize("yaw", [20.0, -20.0])
    def test_yaw_moves_the_forward_marker_along_the_row(self, projection, use_ffmpeg, yaw):
        build, _kw = _SOURCES[projection]
        got = _centroid(_mapper(projection, use_ffmpeg, yaw=yaw).map_single(build()), _marker_colour("forward"))
        assert got is not None
        expected_x = _S * (0.5 - yaw / 180.0)  # positive yaw looks right ⇒ content moves left
        assert abs(got[0] - expected_x) <= 2.0, f"column {got[0]:.2f}, expected {expected_x:.2f}"
        assert abs(got[1] - _S * 0.5) <= 2.0, f"yaw moved the row to {got[1]:.2f}"

    def test_roll_turns_an_elevated_marker_about_the_forward_axis(self, projection, use_ffmpeg):
        build, _kw = _SOURCES[projection]
        out = _mapper(projection, use_ffmpeg, roll=30.0).map_single(build())
        got = _centroid(out, _marker_colour("high"))
        assert got is not None
        want = _expected_landing(_marker_direction("high"), roll=30.0)
        assert abs(got[0] - want[0]) <= 2.0 and abs(got[1] - want[1]) <= 2.0, f"{got} vs theory {want}"

    def test_roll_leaves_the_forward_marker_where_it_was(self, projection, use_ffmpeg):
        """Roll is about the forward axis, so the forward direction is its fixed
        point — a rolled *pitch* (wrong composition order) would move it.
        """
        build, _kw = _SOURCES[projection]
        got = _centroid(_mapper(projection, use_ffmpeg, roll=30.0).map_single(build()), _marker_colour("forward"))
        assert abs(got[0] - _S * 0.5) <= 2.0 and abs(got[1] - _S * 0.5) <= 2.0, got

    def test_all_three_together_land_on_theory(self, projection, use_ffmpeg):
        """The composition order (``ypr``) only shows up when all three are set."""
        build, _kw = _SOURCES[projection]
        angles = {"yaw": 25.0, "pitch": 15.0, "roll": 10.0}
        out = _mapper(projection, use_ffmpeg, **angles).map_single(build())
        for name in ("forward", "high"):
            got = _centroid(out, _marker_colour(name))
            assert got is not None, f"{name} marker fell out of the frame"
            want = _expected_landing(_marker_direction(name), **angles)
            assert abs(got[0] - want[0]) <= 2.0 and abs(got[1] - want[1]) <= 2.0, f"{name}: {got} vs theory {want}"


# --------------------------------------------------------------------------- #
# The two paths must not drift apart (the #303 rule)
# --------------------------------------------------------------------------- #


def _smooth_source(projection: str) -> np.ndarray:
    """A ramp-only source: no hard edges, so any disagreement between the two
    Lanczos implementations is a genuine *geometry* disagreement rather than
    ringing (the #294 fixture, one per projection).
    """
    yy, xx = np.mgrid[0:_S, 0:_S].astype(np.float64)
    dx, dy = (xx + 0.5) - _R, (yy + 0.5) - _R
    img = np.zeros((_S, _S, 3), np.uint8)
    if projection == "fisheye":
        r, az = np.hypot(dx, dy), np.arctan2(dy, dx)
        inside = r <= _R
        img[..., 0] = np.where(inside, r / _R * 200 + 30, 0)
        img[..., 1] = np.where(inside, (np.sin(az) * 0.5 + 0.5) * 200 + 30, 0)
        img[..., 2] = np.where(inside, (np.cos(az) * 0.5 + 0.5) * 200 + 30, 0)
    else:
        img[..., 0] = (dx / _S + 0.5) * 200 + 30
        img[..., 1] = (dy / _S + 0.5) * 200 + 30
        img[..., 2] = ((dx + dy) / (2 * _S) + 0.5) * 200 + 30
    return img


@_FFMPEG
@pytest.mark.parametrize("projection", ["fisheye", "rectilinear"])
@pytest.mark.parametrize(
    "angles",
    [
        {"pitch": 20.0},
        {"pitch": -20.0},
        {"yaw": 25.0},
        {"roll": 30.0},
        {"yaw": 25.0, "pitch": 15.0, "roll": 10.0},
    ],
)
def test_ffmpeg_and_opencv_paths_agree_under_rotation(projection, angles):
    """Both paths must implement the *same* rotation, not merely *a* rotation.

    Bounds are the #294/#302 ones (mean < 4, p99 < 30), which is interpolation
    noise: a sign flip on any axis moves content ~57 px at 20° and blows
    straight through them.  Measured with these angles: mean 0.7–1.5, p99 2–23
    — i.e. no worse than the unrotated baseline.
    """
    src = _smooth_source(projection)
    _build, kw = _SOURCES[projection]
    a = EquirectangularMapper(_S, _S, use_ffmpeg=True, **kw, **angles).map_single(src).astype(int)
    b = EquirectangularMapper(_S, _S, use_ffmpeg=False, **kw, **angles).map_single(src).astype(int)
    diff = np.abs(a - b)
    assert float(diff.mean()) < 4.0, f"mean |diff| = {diff.mean()}"
    assert float(np.percentile(diff, 99)) < 30.0, f"p99 |diff| = {np.percentile(diff, 99)}"


@_FFMPEG
@pytest.mark.parametrize("projection", ["fisheye", "rectilinear"])
def test_both_paths_put_the_markers_within_2px_of_each_other(projection):
    """The ≤2 px acceptance bound, stated directly between the two paths."""
    build, _kw = _SOURCES[projection]
    src = build()
    angles = {"yaw": 25.0, "pitch": 15.0, "roll": 10.0}
    ff = _mapper(projection, True, **angles).map_single(src)
    cv = _mapper(projection, False, **angles).map_single(src)
    for name, rgb, _p, _a in _MARKERS:
        a, b = _centroid(ff, rgb), _centroid(cv, rgb)
        assert a is not None and b is not None, f"{name} marker missing on one path"
        assert abs(a[0] - b[0]) <= 2.0 and abs(a[1] - b[1]) <= 2.0, f"{name}: ffmpeg {a} vs opencv {b}"


# --------------------------------------------------------------------------- #
# Zero regression — the precondition for merging this at all
# --------------------------------------------------------------------------- #

#: The v360 head ``origin/main`` (a7cc891) emits, copied verbatim.  These are
#: what "byte-identical to main" *means* for the ffmpeg path, so they are
#: written out rather than rebuilt from the code under test.
_MAIN_RECT_HEAD = "v360=input=flat:output=hequirect:ih_fov=90.0:iv_fov=58.72:w=512:h=512:alpha_mask=1"
_MAIN_FISH_HEAD = "v360=input=fisheye:output=hequirect:ih_fov=180:iv_fov=180:w=512:h=512:interp=lanczos:alpha_mask=1"


def _main_filter(head: str) -> str:
    return f"{head},{EquirectangularMapper._BLACK_COMPOSITE},format=rgb24"


def _render_rgb24(vfilter: str, src: np.ndarray) -> bytes:
    """Run one frame through *vfilter* and return the raw rgb24 output bytes."""
    h, w = src.shape[:2]
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostats",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-i", "pipe:0",
        "-vf", vfilter, "-frames:v", "1",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ]  # fmt: skip
    return subprocess.run(cmd, input=src.tobytes(), capture_output=True, timeout=60, check=True).stdout


def _reference_mesh_pre_323(mapper: EquirectangularMapper, src_w: int, src_h: int):
    """Rebuild ``origin/main``'s ``_build_mesh`` — i.e. with no rotation step.

    Transcribed from the pre-#323 source so it is a genuine reference rather
    than a re-run of the code under test.
    """
    w_out, h_out = mapper.output_width, mapper.output_height
    u, v = np.meshgrid(np.arange(w_out), np.arange(h_out))
    u, v = u.astype(np.float32), v.astype(np.float32)
    theta = (u / w_out - 0.5) * np.pi
    phi = (v / h_out) * np.pi
    ray_x = np.sin(theta) * np.sin(phi)
    ray_y = np.cos(phi)
    ray_z = np.cos(theta) * np.sin(phi)

    if mapper.input_projection == "fisheye":
        sx, sy = mapper._fisheye_source_coords(ray_x, ray_y, ray_z, src_w, src_h)
        in_bounds = (sx >= 0) & (sx < src_w) & (sy >= 0) & (sy < src_h)
    else:
        fx = src_w / (2.0 * math.tan(math.radians(mapper.src_hfov) / 2.0))
        cx, cy = src_w / 2.0, src_h / 2.0
        valid = ray_z > 0.01
        sx = np.where(valid, fx * ray_x / np.maximum(ray_z, 1e-6) + cx, -1.0)
        sy = np.where(valid, -fx * ray_y / np.maximum(ray_z, 1e-6) + cy, -1.0)
        in_bounds = valid & (sx >= 0) & (sx < src_w) & (sy >= 0) & (sy < src_h)
    sx = np.where(in_bounds, sx, -1.0)
    sy = np.where(in_bounds, sy, -1.0)
    return sx.astype(np.float32), sy.astype(np.float32)


class TestZeroRegressionAtZero:
    """``--pitch 0`` (the default) must be byte-identical to ``origin/main``."""

    def test_rectilinear_filter_string_is_the_literal_main_one(self):
        got = EquirectangularMapper(_S, _S, src_hfov=90.0)._v360_filter(640, 360)
        assert got == _main_filter(_MAIN_RECT_HEAD)

    def test_fisheye_filter_string_is_the_literal_main_one(self):
        got = EquirectangularMapper(_S, _S, input_projection="fisheye", fisheye_fov=180.0)._v360_filter(_S, _S)
        assert got == _main_filter(_MAIN_FISH_HEAD)

    @pytest.mark.parametrize("projection", ["fisheye", "rectilinear"])
    def test_explicit_zeros_emit_no_orientation_terms(self, projection):
        """Passing the flags at 0 must be indistinguishable from omitting them —
        ``yaw=0:pitch=0:roll=0`` would render the same but *change the string*,
        and the string is what #294/#301/#302 pin.
        """
        _build, kw = _SOURCES[projection]
        default = EquirectangularMapper(_S, _S, **kw)._v360_filter(_S, _S)
        zeros = EquirectangularMapper(_S, _S, **kw, pitch=0.0, yaw=0.0, roll=0.0)._v360_filter(_S, _S)
        assert zeros == default
        assert ":pitch=" not in zeros and ":yaw=" not in zeros and ":roll=" not in zeros

    @pytest.mark.parametrize("projection", ["fisheye", "rectilinear"])
    def test_a_non_zero_angle_does_reach_the_filter(self, projection):
        """Guard the guard: the three assertions above must not pass because the
        terms are never emitted at all.
        """
        _build, kw = _SOURCES[projection]
        flt = EquirectangularMapper(_S, _S, **kw, pitch=20.0)._v360_filter(_S, _S)
        head = dict(part.split("=", 1) for part in flt.split(",")[0][len("v360=") :].split(":"))
        assert head["pitch"] == "20" and head["yaw"] == "0" and head["roll"] == "0"

    @_FFMPEG
    @pytest.mark.parametrize("projection", ["fisheye", "rectilinear"])
    def test_rendered_bytes_hash_identically_to_mains_filter(self, projection):
        """The end-to-end claim: main's literal chain and the shipped default
        chain produce the same SHA-256 over the raw rgb24 output.
        """
        src = _smooth_source(projection)
        head = _MAIN_FISH_HEAD if projection == "fisheye" else _MAIN_RECT_HEAD
        if projection == "rectilinear":
            # the pinned head is the 640x360 one; render the square source
            # through the mapper's own 512² head instead.
            head = head.replace("iv_fov=58.72", "iv_fov=90.00")
        _build, kw = _SOURCES[projection]
        shipped = EquirectangularMapper(_S, _S, **kw).map_single(src)
        reference = _render_rgb24(_main_filter(head), src)
        assert len(reference) == _S * _S * 3, f"short render: {len(reference)} bytes"
        assert hashlib.sha256(shipped.tobytes()).hexdigest() == hashlib.sha256(reference).hexdigest()

    @pytest.mark.parametrize("projection", ["fisheye", "rectilinear"])
    def test_opencv_mesh_is_bit_identical_to_the_pre_323_mesh(self, projection):
        """The fallback path's zero-regression proof: the rotation branch must
        not perturb a single mesh coordinate at the default.
        """
        _build, kw = _SOURCES[projection]
        m = EquirectangularMapper(_S, _S, use_ffmpeg=False, **kw)
        m._build_mesh(_S, _S)
        for got, want in zip(m._mesh, _reference_mesh_pre_323(m, _S, _S), strict=True):
            assert got.dtype == want.dtype
            assert np.array_equal(got, want), "the pitch/yaw/roll branch moved the default mesh"

    @pytest.mark.parametrize("projection", ["fisheye", "rectilinear"])
    def test_opencv_output_hashes_identically_to_the_pre_323_mesh(self, projection):
        """... and the pixels that mesh produces, end to end."""
        build, kw = _SOURCES[projection]
        src = build()
        shipped = EquirectangularMapper(_S, _S, use_ffmpeg=False, **kw).map_single(src)
        reference_mapper = EquirectangularMapper(_S, _S, use_ffmpeg=False, **kw)
        reference_mapper._mesh = _reference_mesh_pre_323(reference_mapper, _S, _S)
        reference = reference_mapper.map_single(src)
        assert hashlib.sha256(shipped.tobytes()).hexdigest() == hashlib.sha256(reference.tobytes()).hexdigest()

    @pytest.mark.parametrize("projection", ["fisheye", "rectilinear"])
    def test_a_non_zero_pitch_really_does_change_the_opencv_mesh(self, projection):
        """Guard the guard for the two tests above."""
        _build, kw = _SOURCES[projection]
        m = EquirectangularMapper(_S, _S, use_ffmpeg=False, **kw, pitch=20.0)
        m._build_mesh(_S, _S)
        assert not np.array_equal(m._mesh[1], _reference_mesh_pre_323(m, _S, _S)[1])


# --------------------------------------------------------------------------- #
# Constructor validation
# --------------------------------------------------------------------------- #


class TestOrientationValidation:
    def test_defaults_are_zero(self):
        m = EquirectangularMapper(64, 64)
        assert (m.pitch, m.yaw, m.roll) == (0.0, 0.0, 0.0)

    @pytest.mark.parametrize("name", ["pitch", "yaw", "roll"])
    @pytest.mark.parametrize("bad", [180.5, -180.5, 361.0, float("nan"), float("inf")])
    def test_rejects_angles_v360_itself_would_reject(self, name, bad):
        """v360 declares ``yaw``/``pitch``/``roll`` as -180..180; accepting more
        on the OpenCV path would make the fallback silently disagree with the
        primary one instead of failing.
        """
        with pytest.raises(ValueError, match=name):
            EquirectangularMapper(64, 64, **{name: bad})

    @pytest.mark.parametrize("name", ["pitch", "yaw", "roll"])
    def test_accepts_the_bounds_themselves(self, name):
        for edge in (180.0, -180.0):
            assert getattr(EquirectangularMapper(64, 64, **{name: edge}), name) == edge

    @pytest.mark.parametrize("name", ["pitch", "yaw", "roll"])
    def test_integer_angles_are_stored_as_floats(self, name):
        assert isinstance(getattr(EquirectangularMapper(64, 64, **{name: 20}), name), float)


# --------------------------------------------------------------------------- #
# CLI — the flags exist, are validated, and reach every mapper the run builds
# --------------------------------------------------------------------------- #

_ANGLE_FLAGS = ("pitch", "yaw", "roll")


class TestCliSurface:
    def test_help_lists_all_three_flags(self, capsys):
        from scripts import run_pipeline as rp

        with pytest.raises(SystemExit):
            rp.parse_args(["--help"])
        out = capsys.readouterr().out
        for flag in _ANGLE_FLAGS:
            assert f"--{flag}" in out

    def test_defaults_are_zero(self):
        from scripts import run_pipeline as rp

        args = rp.parse_args(["--input", "s.mp4"])
        assert (args.pitch, args.yaw, args.roll) == (0.0, 0.0, 0.0)

    @pytest.mark.parametrize("flag", _ANGLE_FLAGS)
    def test_each_flag_is_parsed_as_a_float(self, flag):
        from scripts import run_pipeline as rp

        args = rp.parse_args(["--input", "s.mp4", f"--{flag}", "-12.5"])
        assert getattr(args, flag) == -12.5
        # ... and the other two stay at their default, so no flag aliases another
        for other in set(_ANGLE_FLAGS) - {flag}:
            assert getattr(args, other) == 0.0

    @pytest.mark.parametrize("flag", _ANGLE_FLAGS)
    @pytest.mark.parametrize("bad", ["181", "-181", "nan", "inf"])
    def test_out_of_range_angles_exit_non_zero(self, flag, bad, capsys):
        """v360's own bound, enforced at parse time rather than mid-render."""
        from scripts import run_pipeline as rp

        with pytest.raises(SystemExit) as excinfo:
            rp.parse_args(["--input", "s.mp4", f"--{flag}", bad])
        assert excinfo.value.code != 0
        assert f"--{flag}" in capsys.readouterr().err

    def test_the_cli_bound_is_the_mappers_bound(self):
        """One source of truth: a drift here is how the OpenCV fallback would
        start accepting angles the ffmpeg path refuses.
        """
        from scripts import run_pipeline as rp

        assert rp.ORIENTATION_LIMIT_DEG == EquirectangularMapper.ORIENTATION_LIMIT_DEG


class _MapperConstructedError(Exception):
    """Sentinel so a stage aborts as soon as the mapper kwargs are captured."""


def _capture_mapper_kwargs(monkeypatch):
    from scripts import run_pipeline as rp

    captured: dict = {}

    def _recorder(**kwargs):
        captured.update(kwargs)
        raise _MapperConstructedError

    monkeypatch.setattr(rp, "EquirectangularMapper", _recorder)
    return captured


class TestCliReachesEveryMapper:
    """#120 / #243 / #294 silent-drop defence, applied to the new flags.

    A ``--pitch`` that the projection stage quietly ignores would look exactly
    like the bug this card fixes, so every construction site is pinned.
    """

    def test_batch_projection_stage(self, monkeypatch):
        from scripts import run_pipeline as rp

        captured = _capture_mapper_kwargs(monkeypatch)
        args = rp.parse_args(
            ["--input", "s.mp4", "--output-width", "64", "--pitch", "20", "--yaw", "-5", "--roll", "3"]
        )
        args.output_height = 64
        with pytest.raises(_MapperConstructedError):
            rp.run_equirect_stage(args, [], [])
        assert (captured["pitch"], captured["yaw"], captured["roll"]) == (20.0, -5.0, 3.0)

    def test_batch_projection_stage_defaults_to_no_rotation(self, monkeypatch):
        """Guard the guard: the pass-through above must not hard-code angles."""
        from scripts import run_pipeline as rp

        captured = _capture_mapper_kwargs(monkeypatch)
        args = rp.parse_args(["--input", "s.mp4", "--output-width", "64"])
        args.output_height = 64
        with pytest.raises(_MapperConstructedError):
            rp.run_equirect_stage(args, [], [])
        assert (captured["pitch"], captured["yaw"], captured["roll"]) == (0.0, 0.0, 0.0)

    def test_chunked_fused_stage(self, monkeypatch):
        """The ``--chunk-size`` fused path builds its *own* mapper (#317)."""
        from scripts import run_pipeline as rp

        captured = _capture_mapper_kwargs(monkeypatch)
        args = rp.parse_args(["--input", "s.mp4", "--output-width", "64", "--chunk-size", "2", "--pitch", "12"])
        args.output_height = 64
        with pytest.raises(_MapperConstructedError):
            rp.run_chunked_fused_stage(args, [], [])
        assert captured["pitch"] == 12.0
        assert (captured["yaw"], captured["roll"]) == (0.0, 0.0)

    def test_streaming_pipeline_accepts_and_forwards_the_angles(self):
        """The stream is the default for ``--quality standard/high``, so a
        batch-only ``--pitch`` would be worse than none at all.
        """
        import inspect

        from pipeline.streaming_pipeline import StreamingPipeline

        sig = inspect.signature(StreamingPipeline.__init__)
        for flag in _ANGLE_FLAGS:
            assert sig.parameters[flag].default == 0.0
        src = inspect.getsource(StreamingPipeline.__init__)
        mapper_call = src[src.index("self.eq_mapper = EquirectangularMapper(") :]
        for flag in _ANGLE_FLAGS:
            assert f"{flag}={flag}," in mapper_call, f"{flag} never reaches the stream's mapper"

    def test_streaming_pipeline_hands_the_angles_to_a_real_mapper(self, monkeypatch):
        """Signature parity is not enough — build one and read the mapper back."""
        import pipeline.streaming_pipeline as sp

        monkeypatch.setattr(sp, "_resolve_hw_encoder", lambda *a, **k: False)
        monkeypatch.setattr(sp, "DepthEstimator", lambda **k: object())
        monkeypatch.setattr(sp, "StereoRenderer", lambda **k: object())
        pipe = sp.StreamingPipeline(output_width=64, output_height=64, pitch=20.0, yaw=-5.0, roll=3.0)
        assert (pipe.eq_mapper.pitch, pipe.eq_mapper.yaw, pipe.eq_mapper.roll) == (20.0, -5.0, 3.0)
        assert pipe.eq_mapper._orientation_active() is True

    def test_streaming_pipeline_defaults_leave_the_mapper_unrotated(self, monkeypatch):
        import pipeline.streaming_pipeline as sp

        monkeypatch.setattr(sp, "_resolve_hw_encoder", lambda *a, **k: False)
        monkeypatch.setattr(sp, "DepthEstimator", lambda **k: object())
        monkeypatch.setattr(sp, "StereoRenderer", lambda **k: object())
        pipe = sp.StreamingPipeline(output_width=64, output_height=64)
        assert pipe.eq_mapper._orientation_active() is False


class TestCliBookkeeping:
    def test_the_drop_detector_does_not_name_the_new_flags(self, caplog):
        """#243 defence: they *are* threaded into StreamingPipeline, so the
        swallowed-flag warning must stay quiet about them.
        """
        from scripts import run_pipeline as rp

        for flag in _ANGLE_FLAGS:
            assert flag in rp._STREAMING_SUPPORTED

        # --tile-size is deliberately NOT supported, so it is the positive
        # control: without it named, the assertions below would be vacuous.
        args = rp.parse_args(
            [
                "--input", "s.mp4", "--streaming",
                "--pitch", "20", "--yaw", "-5", "--roll", "3",
                "--tile-size", "256",
            ]
        )  # fmt: skip
        rp._warn_streaming_unsupported_args(args)

        assert "--tile-size" in caplog.text, "detector did not run; the assertions below prove nothing"
        for flag in _ANGLE_FLAGS:
            assert f"--{flag}" not in caplog.text

    def test_the_project_stage_manifest_records_the_angles(self, tmp_path):
        """A resumed ``project`` stage must be invalidated by a changed sphere
        orientation, exactly like a changed ``--src-hfov``.
        """
        from scripts import run_pipeline as rp

        # ``_stage_artifacts`` resolves (and creates) the stage temp dirs, so
        # the input must be absolute under tmp_path — a relative "s.mp4" would
        # drop ``s_vr180_temp/`` into the repo root (#318's guard catches it).
        args = rp.parse_args(
            [
                "--input", str(tmp_path / "s.mp4"),
                "--temp-dir", str(tmp_path / "work"),
                "--output-width", "64", "--pitch", "20",
            ]
        )  # fmt: skip
        args.output_height = 64
        _inputs, _outputs, params = rp._stage_artifacts(args, "project")
        assert params["pitch"] == 20.0
        assert params["yaw"] == 0.0 and params["roll"] == 0.0

    def test_the_sidecar_records_the_angles(self, tmp_path, monkeypatch):
        from types import SimpleNamespace

        from scripts import run_pipeline as rp

        import pipeline.sidecar as sc

        captured: dict = {}
        monkeypatch.setattr(sc, "write_sidecar", lambda p, *, immersive, generation: captured.update(g=generation))
        args = SimpleNamespace(
            output_width=2880, output_height=2880, preset=None, input_projection="fisheye", fisheye_fov=180.0,
            pitch=20.0, yaw=-5.0, roll=3.0,
        )  # fmt: skip

        rp._write_sidecar_from_args(str(tmp_path / "out.mp4"), "vr180", args)

        assert captured["g"]["sphere_orientation"] == {"pitch": 20.0, "yaw": -5.0, "roll": 3.0}

    def test_the_sidecar_records_zeros_rather_than_omitting_them(self, tmp_path, monkeypatch):
        """An absent key would be ambiguous: "level" or "run predates #323"?"""
        from types import SimpleNamespace

        from scripts import run_pipeline as rp

        import pipeline.sidecar as sc

        captured: dict = {}
        monkeypatch.setattr(sc, "write_sidecar", lambda p, *, immersive, generation: captured.update(g=generation))
        # Deliberately a namespace with no pitch/yaw/roll at all — the #294
        # sidecar tests build args like this, and the writer must tolerate it.
        args = SimpleNamespace(output_width=2880, output_height=2880, preset=None, input_projection="rectilinear")

        rp._write_sidecar_from_args(str(tmp_path / "out.mp4"), "vr180", args)

        assert captured["g"]["sphere_orientation"] == {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}
