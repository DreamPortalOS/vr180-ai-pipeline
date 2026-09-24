"""Studio dome orientation — cross-check (对拍) of the projection math.

Issue #416 extracted the Studio 3D dome's domemaster projection into a
dependency-free pure-JS module (``studio/static/dome_proj.js``) and a stdlib-only
Python mirror (``studio/dome_proj.py``), so the 3D preview, the yaw/pitch/roll
tools and the virtual-camera viewport share one set of formulas.  This module
pins the contract those formulas must hold:

* the domemaster convention (DECISION_DOME / ``DOME_PROMPT.txt``): circle centre
  = zenith, rim = horizon, the BOTTOM of the circle = directly in front of the
  audience;
* v360's ``rorder=ypr`` orientation with the same signs as the CLI's
  ``--dome-yaw/--dome-pitch/--dome-roll`` — recovered by measurement in
  ``pipeline/equirectangular_mapper.py::_orientation_matrix`` (the ground truth
  this test cross-checks the mirror against, so a drift in either is caught);
* the camera-zenith round-trip (a camera pointed at the zenith sees the master's
  circle centre) the acceptance checklist requires.

The Python assertions run everywhere (numpy is a pipeline dependency; the
mirror itself is stdlib-only).  The JS↔Python agreement is checked where Node
is available (``node --test`` over ``tests/static/js/dome_proj.test.js``); CI is
ubuntu + CPU-only and does not install Node, so that part is a *skip* there, not
a failure — but locally it runs, and the Python side already enforces the math
against the measured v360 matrix regardless.
"""

from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from studio.dome_proj import (
    camera_frustum_dirs,
    camera_ray,
    dir_from_theta_phi,
    dir_to_master_uv,
    export_cli_params,
    master_uv_to_dir,
    orient_matrix,
    output_to_source,
    source_to_output,
)

ROOT = Path(__file__).resolve().parent.parent
JS_MODULE = ROOT / "studio" / "static" / "dome_proj.js"
JS_TEST = ROOT / "tests" / "static" / "js" / "dome_proj.test.js"
DEG = math.pi / 180
HALF_PI = math.pi / 2
ZERO = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}


def _v360_ground_truth(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """The measured v360 output→source matrix, rebuilt from the documented
    elementary rotations in ``equirectangular_mapper._orientation_matrix``.

    ``R = R_yaw @ R_pitch @ R_roll`` (v360 ``rorder=ypr``), frame
    ``x`` right / ``y`` up / ``z`` forward.  This is the convention recovered by
    rendering dots through the real ffmpeg filter (#323/#324), so matching it is
    matching the shipped pipeline — the point of the 对拍.
    """

    y, p, r = (a * DEG for a in (yaw, pitch, roll))
    cy, sy = math.cos(y), math.sin(y)
    cp, sp = math.cos(p), math.sin(p)
    cr, sr = math.cos(r), math.sin(r)
    r_yaw = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    r_pitch = np.array([[1.0, 0.0, 0.0], [0.0, cp, sp], [0.0, -sp, cp]])
    r_roll = np.array([[cr, sr, 0.0], [-sr, cr, 0.0], [0.0, 0.0, 1.0]])
    return r_yaw @ r_pitch @ r_roll


def _close(a: float, b: float, tol: float = 1e-9, msg: str = "") -> None:
    assert abs(a - b) <= tol, f"{msg}: |{a}| vs |{b}| (tol {tol})"


def _close_vec(a, b, tol: float = 1e-9, msg: str = "") -> None:
    assert len(a) == len(b), f"{msg}: length {len(a)} != {len(b)}"
    for i, (x, y) in enumerate(zip(a, b, strict=True)):
        assert abs(x - y) <= tol, f"{msg}[{i}]: |{x}| vs |{y}| (tol {tol})"


def _unit(v):
    length = math.hypot(v[0], v[1], v[2]) or 1.0
    return (v[0] / length, v[1] / length, v[2] / length)


# ----------------------------------------------------------------- v360 对拍
@pytest.mark.parametrize(
    "yaw,pitch,roll",
    [(0, 0, 0), (30, 20, -10), (-70, 5, 45), (37, -18, 62), (120, -90, 0), (0, 89, 0)],
)
def test_orient_matrix_matches_measured_v360(yaw, pitch, roll) -> None:
    """The mirror's orient_matrix is the measured v360 output→source matrix."""
    o = {"yaw": yaw, "pitch": pitch, "roll": roll}
    M = np.array(orient_matrix(o)).reshape(3, 3)
    expected = _v360_ground_truth(yaw, pitch, roll)
    np.testing.assert_allclose(M, expected, atol=1e-9, err_msg=f"ypr {yaw}/{pitch}/{roll}")


def test_orient_matrix_is_orthogonal() -> None:
    """output→source is a rotation, so R @ Rᵀ == I and R⁻¹ == Rᵀ."""
    o = {"yaw": 40.0, "pitch": -25.0, "roll": 70.0}
    M = np.array(orient_matrix(o)).reshape(3, 3)
    np.testing.assert_allclose(M @ M.T, np.eye(3), atol=1e-9)
    for v in ([0.2, 0.5, -0.83], [1, 0, 0], [0, 0, 1], [-0.6, 0.6, 0.53]):
        v = np.array(v, dtype=float)
        out = np.array(source_to_output(tuple(v), o))
        back = np.array(output_to_source(tuple(out), o))
        np.testing.assert_allclose(back, v, atol=1e-9)


def test_pitch_drops_content_toward_horizon_like_dome_pitch() -> None:
    """An output ray at elevation E samples the source at E + pitch, so a marker
    at source elevation e lands at output elevation e - pitch — the same sign
    as ``--dome-pitch`` (content moves DOWN for +pitch)."""
    src_elev = 60.0
    d = dir_from_theta_phi(HALF_PI - src_elev * DEG, 0.0)
    for pitch in (-30.0, 0.0, 30.0, 90.0):
        out = source_to_output(d, {"yaw": 0.0, "pitch": pitch, "roll": 0.0})
        elev = math.degrees(math.asin(max(-1.0, min(1.0, _unit(out)[1]))))
        _close(elev, src_elev - pitch, 1e-6, f"pitch {pitch}")


# ------------------------------------------------------------- the convention
def test_identity_centre_zenith_rim_horizon_bottom_front() -> None:
    """DECISION_DOME: centre = zenith, rim = horizon, bottom = front."""
    z = dir_to_master_uv((0, 1, 0), ZERO)
    assert z["inside"] is True
    _close(z["u"], 0.5)
    _close(z["v"], 0.5)

    front = dir_to_master_uv((0, 0, 1), ZERO)
    assert front["inside"] is True
    _close(front["u"], 0.5)
    _close(front["v"], 1.0, msg="front at BOTTOM")

    _close(dir_to_master_uv((0, 0, -1), ZERO)["v"], 0.0, msg="back at top")
    _close(dir_to_master_uv((1, 0, 0), ZERO)["u"], 1.0, msg="right")
    _close(dir_to_master_uv((-1, 0, 0), ZERO)["u"], 0.0, msg="left")

    # horizon ring -> circle rim (r == 1, azimuthal-equidistant)
    for phi in (0, math.pi / 4, math.pi / 2, math.pi, 3 * math.pi / 4):
        uv = dir_to_master_uv(dir_from_theta_phi(HALF_PI, phi), ZERO)
        _close(math.hypot(uv["u"] - 0.5, uv["v"] - 0.5), 0.5, 1e-9)


def test_equidistant_radius_scales_with_colatitude() -> None:
    """r = colatitude / 90°, the azimuthal-equidistant mapping.

    ``dir_from_theta_phi`` takes the colatitude ``theta`` (0 = zenith), so a
    source at *elevation* ``elev`` sits at colatitude ``90 - elev`` and its
    master radius is ``(90 - elev) / 90``."""
    for elev_deg in (10, 33, 60, 80):
        uv = dir_to_master_uv(dir_from_theta_phi(HALF_PI - elev_deg * DEG, math.pi / 3), ZERO)
        _close(2 * math.hypot(uv["u"] - 0.5, uv["v"] - 0.5), (90 - elev_deg) / 90.0, 1e-6)


def test_below_horizon_is_outside_inscribed_circle() -> None:
    for theta in (1.58, 1.7, 2.0):
        assert dir_to_master_uv(dir_from_theta_phi(theta, 0.3), ZERO)["inside"] is False


def test_master_uv_to_dir_inverts_dir_to_master_uv() -> None:
    for elev_deg in (5, 25, 45, 67, 88):
        for phi_deg in (-150, -45, 0, 40, 90, 170):
            d = dir_from_theta_phi(HALF_PI - elev_deg * DEG, phi_deg * DEG)
            uv = dir_to_master_uv(d, ZERO)
            back = _unit(master_uv_to_dir(uv["u"], uv["v"], ZERO))
            _close_vec(back, _unit(d), 1e-6, f"round-trip {elev_deg}/{phi_deg}")


def test_master_uv_to_dir_null_outside_circle() -> None:
    assert master_uv_to_dir(0.95, 0.95, ZERO) is None
    assert master_uv_to_dir(-0.1, 0.5, ZERO) is None


def test_front_flip_toggle_moves_front_to_top() -> None:
    top = dir_to_master_uv((0, 0, 1), ZERO, front_is_bottom=False)
    _close(top["u"], 0.5)
    _close(top["v"], 0.0, msg="front at top")
    _close(dir_to_master_uv((0, 0, -1), ZERO, front_is_bottom=False)["v"], 1.0)
    d = dir_from_theta_phi(HALF_PI - math.radians(45), -math.pi / 4)
    uv = dir_to_master_uv(d, ZERO, front_is_bottom=False)
    back = _unit(master_uv_to_dir(uv["u"], uv["v"], ZERO, front_is_bottom=False))
    _close_vec(back, _unit(d), 1e-6, "front-top round-trip")


# -------------------------------------------------------------------- camera
def test_camera_level_looks_front_yaw_right_pitch_up() -> None:
    _close_vec(camera_ray(0, 0, 0, 0, 45), (0, 0, 1))
    right = camera_ray(0, 0, 90, 0, 45)
    _close(right[0], 1.0)
    _close(right[2], 0.0)
    up = camera_ray(0, 0, 0, 45, 45)
    _close(up[1], math.sin(45 * DEG))
    for fov in (10, 45, 85):
        for ndc in ((-1, -1), (0.7, -0.3), (1, 1)):
            ray = camera_ray(ndc[0], ndc[1], 20, -15, fov)
            _close(math.hypot(ray[0], ray[1], ray[2]), 1.0, 1e-9, f"unit fov {fov}")


def test_camera_at_zenith_sees_master_circle_centre() -> None:
    """Acceptance: camera正对天顶时 2D 画面中心 = 母版圆心 (identity orientation).

    With a non-zero yaw/pitch/roll the dome is deliberately tilted so its zenith
    maps elsewhere — that is what the orientation tool is for; the identity case
    is the one the checklist pins."""
    for yaw, pitch in [(0, 90), (40, 90), (-60, 90), (120, 90)]:
        ray = camera_ray(0, 0, yaw, pitch, 45)
        _close(ray[1], 1.0, 1e-9, f"look dir of cam {yaw}")
        uv = dir_to_master_uv(ray, ZERO)
        assert uv["inside"] is True
        _close(uv["u"], 0.5, 1e-9, "u at zenith")
        _close(uv["v"], 0.5, 1e-9, "v at zenith")


def test_camera_frustum_dirs_five_unit_rays() -> None:
    dirs = camera_frustum_dirs(30, -20, 50)
    assert len(dirs) == 5
    for d in dirs:
        _close(math.hypot(d[0], d[1], d[2]), 1.0, 1e-9, "unit")
    _close_vec(dirs[4], camera_ray(0, 0, 30, -20, 50), msg="centre = look dir")
    _close_vec(dirs[0], camera_ray(-1, -1, 30, -20, 50), msg="corner 0")


# ----------------------------------------------------------- export / CLI signs
def test_export_cli_params_names_signs_half_up() -> None:
    """The UI export uses the CLI's --dome-* names and signs, rounded half-up."""
    cli = export_cli_params({"yaw": 45.6, "pitch": -33.4, "roll": 2.5})
    assert cli == {"dome_pitch": -33, "dome_yaw": 46, "dome_roll": 3}
    # half-up, not banker's: round(2.5) == 3 (Python's round() would say 2)
    assert cli["dome_roll"] == 3
    assert export_cli_params({}) == {"dome_pitch": 0, "dome_yaw": 0, "dome_roll": 0}
    assert export_cli_params({"yaw": 90, "pitch": 0, "roll": 0}) == {
        "dome_pitch": 0,
        "dome_yaw": 90,
        "dome_roll": 0,
    }
    assert export_cli_params({"pitch": -180}) == {"dome_pitch": -180, "dome_yaw": 0, "dome_roll": 0}


def test_export_cli_params_identity_on_integers() -> None:
    for yaw in range(-180, 181, 7):
        for pitch in range(-180, 181, 13):
            cli = export_cli_params({"yaw": yaw, "pitch": pitch, "roll": yaw - pitch})
            assert cli["dome_yaw"] == yaw
            assert cli["dome_pitch"] == pitch
            assert cli["dome_roll"] == yaw - pitch


def test_ui_emits_cli_flag_names() -> None:
    """preview3d.js must echo the exact --dome-* flag names the pipeline reads."""
    ui = (ROOT / "studio" / "static" / "preview3d.js").read_text(encoding="utf-8")
    for flag in ("--dome-pitch", "--dome-yaw", "--dome-roll"):
        assert flag in ui, f"preview3d.js does not emit {flag}"


# ----------------------------------------------------- JS ↔ Python 对拍 (node)
def _node_bin() -> str | None:
    return shutil.which("node")


@pytest.mark.skipif(_node_bin() is None, reason="node not installed; JS tests run where Node is available")
def test_js_module_matches_python_mirror(tmp_path) -> None:
    """Where Node is available, the JS module and the Python mirror agree on the
    numbers for the same inputs — the true JS↔Python 对拍.  Skipped on CI
    (ubuntu + no Node); the Python side already cross-checks against the
    measured v360 matrix above, so the contract holds there regardless."""
    assert JS_MODULE.is_file()
    # Sample a grid of (yaw, pitch, roll, dir) and compare dirToMasterUV.
    cases = []
    for yaw in (0, 30, -70, 120):
        for pitch in (0, 20, -18, 89):
            for roll in (0, -10, 45, 62):
                for elev in (0, 30, 60, 85):
                    for az in (0, 45, 90, 170, -150):
                        d = dir_from_theta_phi(HALF_PI - elev * DEG, az * DEG)
                        uv = dir_to_master_uv(d, {"yaw": yaw, "pitch": pitch, "roll": roll})
                        cases.append((yaw, pitch, roll, d[0], d[1], d[2], uv["u"], uv["v"], int(uv["inside"])))
    payload = "\n".join(",".join(str(c) for c in row) for row in cases)
    script = (
        "const D=require(process.argv[1]);"
        "const lines=require('fs').readFileSync(0,'utf8').trim().split('\\n');"
        "const out=[];"
        "for(const l of lines){const[y,p,r,x,y2,z]=l.split(',').map(Number);"
        "const o={yaw:y,pitch:p,roll:r};"
        "const uv=D.dirToMasterUV([x,y2,z],o);"
        "out.push(uv.u+','+uv.v+','+(uv.inside?1:0));}"
        "process.stdout.write(out.join('\\n'));"
    )
    proc = subprocess.run(
        [_node_bin(), "-e", script, str(JS_MODULE)],
        input=payload,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, f"node failed:\n{proc.stderr}"
    js_lines = proc.stdout.strip().split("\n")
    assert len(js_lines) == len(cases), f"row count {len(js_lines)} != {len(cases)}"
    for row, js in zip(cases, js_lines, strict=True):
        exp_u, exp_v, exp_in = row[6], row[7], row[8]
        got_u, got_v, got_in = (float(x) if i < 2 else int(x) for i, x in enumerate(js.split(",")))
        assert got_in == exp_in, f"inside mismatch {row}: js={got_in} py={exp_in}"
        _close(got_u, exp_u, 1e-9, f"u {row}")
        _close(got_v, exp_v, 1e-9, f"v {row}")


@pytest.mark.skipif(_node_bin() is None, reason="node not installed; JS tests run where Node is available")
def test_js_node_test_runner_passes(tmp_path) -> None:
    """Run ``node --test`` over the JS self-check suite (skipped on CI)."""
    assert JS_TEST.is_file()
    proc = subprocess.run(
        [_node_bin(), "--test", str(JS_TEST)],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
        timeout=60,
    )
    assert proc.returncode == 0, f"node --test failed (rc={proc.returncode}):\n{proc.stdout}\n{proc.stderr}"
