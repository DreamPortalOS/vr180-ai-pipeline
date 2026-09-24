"""Python mirror of ``studio/static/dome_proj.js`` — the Studio 3D dome's projection math.

Issue #416 moved the Studio's domemaster projection into a dependency-free pure JS
module (``dome_proj.js``) so the 3D preview, the yaw/pitch/roll tools and the
virtual-camera viewport share one set of formulas.  This file is a line-for-line
Python port of the *same* functions: ``tests/test_studio_dome_orientation.py``
asserts the two agree (对拍) and that they agree with the convention the shipped
fulldome route already pins — circle centre = zenith, rim = horizon, the **bottom**
of the circle = directly in front of the audience (DECISION_DOME), and v360's
``rorder=ypr`` orientation (#323/#324).

It is a mirror, not an independent implementation, on purpose: a third copy of the
spherical math in the repo would drift from the other two and could still be wrong
along with them.  It is stdlib-only (``math``), has no numpy / opencv / ffmpeg /
network dependency, and imports nothing from ``studio`` or ``pipeline`` — the same
"no models, CPU-only, no network" bar the pipeline tests run under.  The test gate
imports this module and cross-checks it against ``pipeline.equirectangular_mapper``'s
measured v360 matrix (the ground truth) and, where Node is available, against the
JS module byte-for-byte.

Change a formula in ``dome_proj.js`` and mirror it here.

Sign conventions (must stay identical to the JS):

* Frame: ``x`` right, ``y`` up, ``z`` forward (the audience's front).
* :func:`orient_matrix` is the **output → source** matrix, returned row-major as a
  flat 9-sequence.  ``R = R_yaw @ R_pitch @ R_roll`` (v360 ``rorder=ypr``);
  ``+pitch`` drops content toward the horizon — the same sign as ``--dome-pitch``.
* :func:`dir_to_master_uv` puts the audience's front at the **bottom** of the
  circle by default (``front_is_bottom=True``); ``False`` flips it to the top.
* :func:`camera_ray` is the *camera's* yaw/pitch, positive-up/right, so ``+pitch``
  raises the camera toward the zenith.  That is the opposite hand from the dome
  ``+pitch`` above, which is documented next to the sliders and is what makes
  "camera pointing at the zenith" the natural ``cam_pitch = 90``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

HALF_PI = math.pi / 2
DEG = math.pi / 180

#: A dome direction is a plain 3-element sequence ``(x, y, z)``, row-major.
Vec3 = tuple[float, float, float]
Mat3 = tuple[float, ...]

#: The identity matrix, as returned by :func:`orient_matrix` at 0/0/0.
IDENTITY: Mat3 = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


def dir_from_theta_phi(theta: float, phi: float) -> Vec3:
    """Direction on the dome from colatitude ``theta`` and azimuth ``phi``.

    ``theta`` = 0 is the zenith, ``HALF_PI`` the horizon; ``phi`` = 0 is the
    audience's front (``+z``), ``pi/2`` is to the right (``+x``).
    """

    st = math.sin(theta)
    return (st * math.sin(phi), math.cos(theta), st * math.cos(phi))


def mat_mul3(a: Sequence[float], b: Sequence[float]) -> Mat3:
    """Row-major 3x3 product of the two flat 9-sequences."""

    o: list[float] = [0.0] * 9
    for i in range(3):
        for j in range(3):
            o[i * 3 + j] = a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j]
    return tuple(o)


def mv_mul3(m: Sequence[float], v: Vec3) -> Vec3:
    """Row-major matrix · vector."""

    return (
        m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
        m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
        m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
    )


def orient_matrix(orient: Mapping[str, float]) -> Mat3:
    """Orientation rotation ``R = R_yaw @ R_pitch @ R_roll`` (v360 ``rorder=ypr``).

    ``orient`` is ``{"yaw", "pitch", "roll"}`` in degrees.  Returns the
    **output → source** matrix row-major as a flat 9-tuple, so a direction ``v``
    maps to ``source = R · v``.  This is the matrix the browser uploads as the
    ``uOrient`` uniform (column-major after a transpose in ``preview3d.js``).
    """

    y = (orient.get("yaw") or 0) * DEG
    p = (orient.get("pitch") or 0) * DEG
    r = (orient.get("roll") or 0) * DEG
    cy, sy = math.cos(y), math.sin(y)
    cp, sp = math.cos(p), math.sin(p)
    cr, sr = math.cos(r), math.sin(r)
    # R_yaw = [[cy,0,sy],[0,1,0],[-sy,0,cy]]
    # R_pitch= [[1,0,0],[0,cp,sp],[0,-sp,cp]]
    # R_roll = [[cr,sr,0],[-sr,cr,0],[0,0,1]]
    ry_rp = mat_mul3((cy, 0.0, sy, 0.0, 1.0, 0.0, -sy, 0.0, cy), (1.0, 0.0, 0.0, 0.0, cp, sp, 0.0, -sp, cp))
    return mat_mul3(ry_rp, (cr, sr, 0.0, -sr, cr, 0.0, 0.0, 0.0, 1.0))


def _transposed(m: Sequence[float]) -> Mat3:
    """Transpose of a row-major flat 9-sequence (returns row-major again)."""

    return (m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8])


def source_to_output(src: Vec3, orient: Mapping[str, float]) -> Vec3:
    """Rotate a *source* dome direction into the *output* frame: ``out = R⁻¹ · src``.

    ``R`` is orthonormal, so ``R⁻¹ == Rᵀ`` — one transpose, no solve.
    """

    return mv_mul3(_transposed(orient_matrix(orient)), src)


def output_to_source(out: Vec3, orient: Mapping[str, float]) -> Vec3:
    """Rotate an *output* dome direction into the *source* frame: ``src = R · out``."""

    return mv_mul3(orient_matrix(orient), out)


def _unit(v: Vec3) -> Vec3:
    length = math.hypot(v[0], v[1], v[2]) or 1.0
    return (v[0] / length, v[1] / length, v[2] / length)


def dir_to_master_uv(d: Vec3, orient: Mapping[str, float], front_is_bottom: bool = True) -> dict[str, Any]:
    """Map a direction in the OUTPUT frame to domemaster texture UV.

    ``front_is_bottom=True`` (the default, and the shipped convention) puts the
    audience's front at the **bottom** of the circle.  ``False`` moves it to the
    top.  Returns ``{"u", "v", "inside"}``; ``inside`` is ``False`` for a
    direction below the horizon (``theta > 90°``, outside the inscribed circle —
    pure black on a legal domemaster, D-1/#334).
    """

    src = output_to_source(d, orient)
    nx, ny, nz = _unit(src)
    theta = math.acos(max(-1.0, min(1.0, ny)))  # colatitude from the zenith
    if theta > HALF_PI + 1e-6:
        return {"u": 0.5, "v": 0.5, "inside": False}
    r = theta / HALF_PI  # azimuthal-equidistant radius 0..1
    horiz = math.hypot(nx, nz)
    if horiz < 1e-6:
        return {"u": 0.5, "v": 0.5, "inside": True}
    # azimuth measured from +z (front) toward +x (right): phi = atan2(x, z).
    # front (phi=0) -> bottom centre, back (phi=pi) -> top, right (phi=pi/2)
    # -> right, left (-pi/2) -> left: u tracks sin(phi), v tracks cos(phi).
    phi = math.atan2(nx, nz)
    u = 0.5 + 0.5 * r * math.sin(phi)
    v = 0.5 + 0.5 * r * math.cos(phi)
    if not front_is_bottom:
        v = 1.0 - v  # flip the front to the top of the circle
    return {"u": u, "v": v, "inside": True}


def master_uv_to_dir(u: float, v: float, orient: Mapping[str, float], front_is_bottom: bool = True) -> Vec3 | None:
    """Inverse of :func:`dir_to_master_uv`: a domemaster UV -> OUTPUT-frame direction.

    The camera viewport uses this to turn a master pixel back into the dome
    direction it depicts, then into a camera ray.  ``None`` outside the inscribed
    circle.
    """

    dx = u - 0.5
    dy = v - 0.5
    r = math.hypot(dx, dy)
    if r > 0.5 + 1e-6:
        return None  # outside the inscribed circle
    theta = min(1.0, r / 0.5) * HALF_PI  # r in [0,0.5] -> theta in [0,90]
    # inverse of dir_to_master_uv: sin(phi) = dx/s, cos(phi) = dy/s with s = 0.5,
    # so phi = atan2(sin, cos) = atan2(dx, dy).
    cy = dy
    if not front_is_bottom:
        cy = -dy  # undo the front-to-top flip
    phi = 0.0 if r < 1e-9 else math.atan2(dx, cy)
    st = math.sin(theta)
    src = (st * math.sin(phi), math.cos(theta), st * math.cos(phi))
    return source_to_output(src, orient)


def camera_ray(nx: float, ny: float, cam_yaw: float, cam_pitch: float, half_fov_deg: float) -> Vec3:
    """Camera ray (in the OUTPUT dome frame) for fragment ``(nx, ny)`` in ``[-1, 1]``.

    ``nx``/``ny`` are normalised frame coords at the near plane.  ``cam_yaw`` and
    ``cam_pitch`` are degrees about the dome's right / vertical axes, and
    ``half_fov_deg`` is half the field of view — the *vertical* half, so the
    horizontal half follows the aspect in the shader.

    Convention: a level camera looks at the audience's front (``+z``), so
    ``(yaw, pitch) = (0, 0)`` is straight ahead.  ``+yaw`` turns it toward ``+x``
    (right) and ``+pitch`` raises it toward the zenith — both positive-up/right,
    like the sliders.  That is the opposite hand from :func:`orient_matrix`'s
    ``+pitch``, which *drops* dome content; the split is why ``cam_pitch = 90`` is
    "pointing at the zenith".
    """

    h = half_fov_deg * DEG
    ray = _unit((math.tan(h) * nx, math.tan(h) * ny, 1.0))
    p = cam_pitch * DEG
    y = cam_yaw * DEG
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    # pitch about +x (right): +pitch raises the look direction toward +y
    r1 = (ray[0], ray[1] * cp + ray[2] * sp, -ray[1] * sp + ray[2] * cp)
    # yaw about +y (up): +yaw turns the look direction from +z toward +x
    return (r1[0] * cy + r1[2] * sy, r1[1], -r1[0] * sy + r1[2] * cy)


def camera_frustum_dirs(cam_yaw: float, cam_pitch: float, half_fov_deg: float) -> list[Vec3]:
    """The four frustum corner rays plus the centre look direction.

    In the OUTPUT dome frame, for drawing the camera's view-volume wireframe in
    the 3D view (the camera sits at the dome centre, looking outward).  Corner
    order matches the JS: (-1,-1), (1,-1), (1,1), (-1,1), then the centre.
    """

    corners = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
    out = [camera_ray(c[0], c[1], cam_yaw, cam_pitch, half_fov_deg) for c in corners]
    out.append(camera_ray(0.0, 0.0, cam_yaw, cam_pitch, half_fov_deg))
    return out


def _round_half_up(x: float) -> int:
    """Round half away from zero — i.e. JavaScript's ``Math.round``.

    Python's builtin ``round`` is banker's (round-half-to-even), so
    ``round(0.5) == 0`` while ``Math.round(0.5) == 1``.  The browser is the
    source of truth for what the operator sees, so this mirror rounds like it —
    the 对拍 test in ``tests/test_studio_dome_orientation.py`` pins the half
    degree both ways so the two cannot drift.
    """

    return math.floor(x + 0.5) if x >= 0 else math.ceil(x - 0.5)


def export_cli_params(orient: Mapping[str, float]) -> dict[str, int]:
    """The orientation the UI is showing, as the CLI-named params the
    ``convert.dome`` node / ``--dome-*`` flags consume.

    Same names, same signs as the CLI: ``dome_pitch`` / ``dome_yaw`` /
    ``dome_roll``.  Rounded to whole degrees — the slider step is 1°.
    """

    return {
        "dome_pitch": _round_half_up(orient.get("pitch") or 0),
        "dome_yaw": _round_half_up(orient.get("yaw") or 0),
        "dome_roll": _round_half_up(orient.get("roll") or 0),
    }


#: The public surface, mirrored onto ``window.DomeProj`` in the JS module.
__all__ = [
    "DEG",
    "HALF_PI",
    "IDENTITY",
    "camera_frustum_dirs",
    "camera_ray",
    "dir_from_theta_phi",
    "dir_to_master_uv",
    "export_cli_params",
    "master_uv_to_dir",
    "mat_mul3",
    "mv_mul3",
    "orient_matrix",
    "output_to_source",
    "source_to_output",
]
