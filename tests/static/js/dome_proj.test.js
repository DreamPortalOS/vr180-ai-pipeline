/* Projection-math cross-check (对拍) for studio/static/dome_proj.js.
 *
 * Run from the repo root (see run_js_check in tests/test_studio_dome_orientation.py):
 *   node --test tests/static/js/dome_proj.test.js
 *
 * Node's built-in test runner keeps this dependency-free: no package.json, no
 * install, nothing to add to requirements.txt.  These assert the *contract*
 * the Studio 3D dome previews against, not the internals:
 *
 *   - the domemaster convention (DECISION_DOME): circle centre = zenith,
 *     rim = horizon, the BOTTOM of the circle = directly in front of the
 *     audience.  A mapping that landed the front at the top is wrong.
 *   - yaw/pitch/roll == v360's `rorder=ypr`, the same signs as the CLI's
 *     --dome-yaw/--dome-pitch/--dome-roll (recovered by measurement in
 *     pipeline/equirectangular_mapper.py::_orientation_matrix).
 *   - the inverse of dirToMasterUV round-trips.
 *   - a camera pointed at the zenith sees the master's circle centre.
 *
 * The 45-degree half-angles here are deliberate: `Math.round` is round-half-up,
 * and one of them lands exactly on the boundary so a drift to banker's
 * round-half-to-even shows up as a failure.
 */
const assert = require("node:assert/strict");
const { test } = require("node:test");

const DP = require("../../../studio/static/dome_proj.js");

const TOL = 1e-9;

function close(a, b, tol = TOL, msg) {
  assert.ok(Math.abs(a - b) <= tol, `${msg || ""} |${a}| vs |${b}| (tol ${tol})`);
}

function closeVec(v, w, tol = TOL, msg) {
  assert.equal(v.length, 3, `${msg || "vector length"}`);
  for (let i = 0; i < 3; i++) close(v[i], w[i], tol, `${msg || ""}[${i}]`);
}

function unit(v) {
  const l = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / l, v[1] / l, v[2] / l];
}

const ZERO = { yaw: 0, pitch: 0, roll: 0 };

// ---------------------------------------------------------------- the contract
test("identity orientation: centre=zenith, rim=horizon, bottom=front", () => {
  // zenith -> circle centre
  const zenith = DP.dirToMasterUV([0, 1, 0], ZERO);
  assert.equal(zenith.inside, true);
  close(zenith.u, 0.5);
  close(zenith.v, 0.5);

  // front -> bottom centre of the circle
  const front = DP.dirToMasterUV([0, 0, 1], ZERO);
  assert.equal(front.inside, true);
  close(front.u, 0.5);
  close(front.v, 1);

  // back -> top; right -> right; left -> left (phi measured from +z toward +x)
  close(DP.dirToMasterUV([0, 0, -1], ZERO).v, 0);
  close(DP.dirToMasterUV([1, 0, 0], ZERO).u, 1);
  close(DP.dirToMasterUV([-1, 0, 0], ZERO).u, 0);

  // horizon ring -> circle rim (r == 1 at the rim)
  for (const phi of [0, Math.PI / 4, Math.PI / 2, Math.PI, (3 * Math.PI) / 4]) {
    const d = DP.dirFromThetaPhi(Math.PI / 2, phi);
    const uv = DP.dirToMasterUV(d, ZERO);
    close(Math.hypot(uv.u - 0.5, uv.v - 0.5), 0.5);
  }
});

test("below the horizon is outside the inscribed circle -> pure black", () => {
  for (const theta of [1.58, 1.7, 2.0]) {
    const uv = DP.dirToMasterUV(DP.dirFromThetaPhi(theta, 0.3), ZERO);
    assert.equal(uv.inside, false);
  }
});

test("dirToMasterUV is radius-1 at the rim and theta/90deg inside (equidistant)", () => {
  for (const thetaDeg of [10, 33, 60, 80]) {
    const theta = (thetaDeg * Math.PI) / 180;
    const uv = DP.dirToMasterUV(DP.dirFromThetaPhi(theta, Math.PI / 3), ZERO);
    // radius scales with angle: 60 deg sits at 60/90 = 0.6667 of R.
    close(2 * Math.hypot(uv.u - 0.5, uv.v - 0.5), thetaDeg / 90, 1e-6);
  }
});

test("masterUVToDir inverts dirToMasterUV", () => {
  for (const thetaDeg of [5, 25, 45, 67, 88]) {
    for (const phiDeg of [-150, -45, 0, 40, 90, 170]) {
      const d = DP.dirFromThetaPhi((thetaDeg * Math.PI) / 180, (phiDeg * Math.PI) / 180);
      const back = unit(DP.masterUVToDir(DP.dirToMasterUV(d, ZERO).u, DP.dirToMasterUV(d, ZERO).v, ZERO));
      closeVec(back, unit(d), 1e-6, `round-trip ${thetaDeg}/${phiDeg}`);
    }
  }
});

test("masterUVToDir returns null outside the inscribed circle", () => {
  assert.equal(DP.masterUVToDir(0.95, 0.95, ZERO), null);
  assert.equal(DP.masterUVToDir(-0.1, 0.5, ZERO), null);
});

test("the front-flip toggle moves the audience front to the top of the circle", () => {
  const top = DP.dirToMasterUV([0, 0, 1], ZERO, false);
  close(top.u, 0.5);
  close(top.v, 0);
  const back = DP.dirToMasterUV([0, 0, -1], ZERO, false);
  close(back.v, 1);
  // and it round-trips as well
  const d = DP.dirFromThetaPhi(Math.PI / 3, -Math.PI / 4);
  const uv = DP.dirToMasterUV(d, ZERO, false);
  closeVec(unit(DP.masterUVToDir(uv.u, uv.v, ZERO, false)), unit(d), 1e-6);
});

// ---------------------------------------------------------------- orientation
test("zero orientation is the identity transform", () => {
  // outputToSource(v, ZERO) == v and sourceToOutput(v, ZERO) == v.
  for (const v of [[0.3, -0.4, 0.9], [1, 0, 0], [0, 0, 1], [-0.5, 0.5, 0.6]]) {
    closeVec(DP.outputToSource(v, ZERO), v, 1e-9, "outputToSource identity");
    closeVec(DP.sourceToOutput(v, ZERO), v, 1e-9, "sourceToOutput identity");
  }
});

test("orientMatrix is orthogonal and inverts via transpose", () => {
  const o = { yaw: 40, pitch: -25, roll: 70 };
  const M = DP.orientMatrix(o);
  // M @ M^T == I
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      let dot = 0;
      for (let k = 0; k < 3; k++) dot += M[i * 3 + k] * M[j * 3 + k];
      close(dot, i === j ? 1 : 0, 1e-9, `M @ M^T [${i}][${j}]`);
    }
  }
  // sourceToOutput(outputToSource(v)) == v, for arbitrary v
  for (const v of [[0.2, 0.5, -0.83], [1, 0, 0], [0, 0, 1], [-0.6, 0.6, 0.53]]) {
    closeVec(DP.sourceToOutput(DP.outputToSource(v, o), o), v, 1e-9);
  }
});

test("yaw rotates about the vertical axis and keeps elevation", () => {
  const d = unit([0.6, 0.2, 0.775]);
  for (const yaw of [-120, -30, 0, 45, 90, 180]) {
    const out = DP.sourceToOutput(d, { yaw, pitch: 0, roll: 0 });
    close(out[1], d[1], 1e-9, `elevation kept at yaw ${yaw}`);
    close(Math.hypot(out[0], out[2]), Math.hypot(d[0], d[2]), 1e-9);
  }
});

test("roll rotates about the forward axis and keeps the forward component", () => {
  const d = unit([0.6, 0.2, 0.775]);
  for (const roll of [-90, -30, 0, 30, 90]) {
    const out = DP.sourceToOutput(d, { yaw: 0, pitch: 0, roll });
    close(out[2], d[2], 1e-9, `forward kept at roll ${roll}`);
    close(Math.hypot(out[0], out[1]), Math.hypot(d[0], d[1]), 1e-9);
  }
});

test("rorder=ypr: yaw/pitch/roll are NOT commutative, only this order matches", () => {
  const o = { yaw: 60, pitch: 45, roll: 30 };
  const DEG = Math.PI / 180;
  const mul = (a, b) => {
    const r = [];
    for (let i = 0; i < 3; i++)
      for (let j = 0; j < 3; j++) r.push(a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j]);
    return r;
  };
  const y = 60 * DEG,
    p = 45 * DEG,
    r = 30 * DEG;
  const Ry = [Math.cos(y), 0, Math.sin(y), 0, 1, 0, -Math.sin(y), 0, Math.cos(y)];
  const Rp = [1, 0, 0, 0, Math.cos(p), Math.sin(p), 0, -Math.sin(p), Math.cos(p)];
  const Rr = [Math.cos(r), Math.sin(r), 0, -Math.sin(r), Math.cos(r), 0, 0, 0, 1];
  // the shipped convention, straight from equirectangular_mapper.py
  const expected = mul(mul(Ry, Rp), Rr);
  for (let i = 0; i < 9; i++) close(DP.orientMatrix(o)[i], expected[i], 1e-9, `ypr[${i}]`);
  // every other ordering disagrees — so a silent reorder would be caught
  const others = { ypr_alt: mul(mul(Ry, Rr), Rp), pyr: mul(mul(Rp, Ry), Rr), xpr: mul(Rr, mul(Rp, Ry)) };
  for (const [name, m] of Object.entries(others)) {
    const diff = m.reduce((acc, v, i) => acc + Math.abs(v - expected[i]), 0);
    assert.ok(diff > 0.05, `${name} should differ from rorder=ypr`);
  }
});

test("+pitch drops content toward the horizon, like --dome-pitch", () => {
  // output elevation E samples the source at E + pitch, so a marker at source
  // elevation e lands at output elevation e - pitch.
  const srcElevDeg = 60;
  const d = DP.dirFromThetaPhi(Math.PI / 2 - (srcElevDeg * Math.PI) / 180, 0);
  for (const pitch of [-30, 0, 30, 90]) {
    const out = DP.sourceToOutput(d, { yaw: 0, pitch, roll: 0 });
    const elev = (Math.asin(Math.max(-1, Math.min(1, out[1]))) * 180) / Math.PI;
    close(elev, srcElevDeg - pitch, 1e-6, `pitch ${pitch}`);
  }
});

test("the browser's WebGL uOrient upload makes m*vec == outputToSource", () => {
  // preview3d.js uploads transpose(orientMatrix) as column-major with
  // uniformMatrix3fv(..., transpose=false, value).  In GLSL, mat3·vec3 treats
  // the vector as a column, so result[i] = sum_j mat[col=j][row=i] * v[j].
  // The column-major flat array T passed by _orientMat stores column j in
  // T[j*3..j*3+2], so mat[col=j][row=i] == T[j*3+i].  Substituting, the
  // effective math matrix m[row=i][col=j] == T[j*3+i], which is exactly
  // orientMatrix's row-major M[i][j] == M[i*3+j] == T[j*3+i].  So the shader's
  // m*vec must equal the pure-math M·d that dirToMasterUV(outputToSource)
  // assumes — this test pins that, so a future "simplification" that drops the
  // transpose (or switches to transpose=true) flips it red.
  const o = { yaw: 37, pitch: -18, roll: 62 };
  const M = DP.orientMatrix(o);
  const T = [M[0], M[3], M[6], M[1], M[4], M[7], M[2], M[5], M[8]]; // _orientMat()
  // GLSL mat3·vec3 with column-major T: result[i] = sum_j T[j*3+i]*v[j]
  const glslMv = (v) => {
    const r = [0, 0, 0];
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) r[i] += T[j * 3 + i] * v[j];
    return r;
  };
  for (const d of [[0.3, 0.5, -0.8], [1, 0, 0], [0, 0, 1], [-0.6, 0.6, 0.53]]) {
    closeVec(glslMv(d), DP.outputToSource(d, o), 1e-9, "shader == outputToSource");
  }
});

// ---------------------------------------------------------------- the camera
test("a camera at (0,0) looks at the audience front; +yaw turns right, +pitch up", () => {
  closeVec(DP.cameraRay(0, 0, 0, 0, 45), [0, 0, 1]);
  const right = DP.cameraRay(0, 0, 90, 0, 45);
  close(right[0], 1, 1e-9);
  close(right[2], 0, 1e-9);
  const up = DP.cameraRay(0, 0, 0, 45, 45);
  close(up[1], Math.sin((45 * Math.PI) / 180), 1e-9);
  // rays stay unit length at any FOV
  for (const fov of [10, 45, 85]) {
    for (const ndc of [[-1, -1], [0.7, -0.3], [1, 1]]) {
      close(Math.hypot(...DP.cameraRay(ndc[0], ndc[1], 20, -15, fov)), 1, 1e-9, `unit fov ${fov}`);
    }
  }
});

test("camera pointed at the zenith sees the master's circle centre (identity orientation)", () => {
  // At the default orientation the dome's zenith IS the master's circle
  // centre, so a camera ray straight up samples (0.5, 0.5).  (With a non-zero
  // yaw/pitch/roll the dome is deliberately tilted, so its zenith maps to a
  // different master pixel — that is what the orientation tool is for.)
  for (const [yaw, pitch] of [[0, 90], [40, 90], [-60, 90], [120, 90]]) {
    const ray = DP.cameraRay(0, 0, yaw, pitch, 45);
    close(ray[1], 1, 1e-9, `look dir of cam ${yaw}`);
    const uv = DP.dirToMasterUV(ray, ZERO);
    assert.equal(uv.inside, true);
    close(uv.u, 0.5, 1e-9, "u at zenith");
    close(uv.v, 0.5, 1e-9, "v at zenith");
  }
});

test("camera viewport uses the same orientation transform the dome shader does", () => {
  // The 2D viewport and the 3D dome must agree on where a direction samples the
  // master, so moving the camera over a tilted dome shows the same pixel the
  // 3D surface would render for that ray.  Both call dirToMasterUV(ray, orient),
  // so feeding the same ray + orient must return the same UV — this pins that
  // the CameraView shader (which inlines the formulas) matches the pure module.
  for (const o of [
    { yaw: 0, pitch: 0, roll: 0 },
    { yaw: 30, pitch: 20, roll: -10 },
    { yaw: -70, pitch: 5, roll: 45 },
  ]) {
    for (const [cy, cp] of [[0, 0], [30, -20], [-40, 60], [0, 89]]) {
      const ray = DP.cameraRay(0, 0, cy, cp, 45);
      const uv = DP.dirToMasterUV(ray, o);
      if (!uv.inside) continue; // ray below the horizon: no master pixel to invert
      // round-trip: the UV the camera renders must invert back to the same ray
      // (up to the direction the dome would show at that point).
      const back = DP.masterUVToDir(uv.u, uv.v, o);
      if (back) closeVec(unit(back), unit(ray), 1e-6, `cam ${cy}/${cp} orient ${o.yaw}`);
    }
  }
});

test("the 2D viewport centre maps the camera's look direction onto the master", () => {
  // Every centre fragment has NDC (0,0), so whatever it samples is cameraRay(0,0)
  // through dirToMasterUV — which must equal the same ray fed a full FOV.
  for (const fov of [15, 45, 85]) {
    const r = DP.cameraRay(0, 0, -70, 30, fov);
    const r2 = DP.cameraRay(0, 0, -70, 30, 60);
    closeVec(r, r2, 1e-9, `fov ${fov} independent of centre ray`);
  }
});

test("cameraFrustumDirs: 5 unit rays, centre = look direction, corners are wider", () => {
  const dirs = DP.cameraFrustumDirs(30, -20, 50);
  assert.equal(dirs.length, 5);
  dirs.forEach((d) => close(Math.hypot(d[0], d[1], d[2]), 1, 1e-9, "unit"));
  closeVec(dirs[4], DP.cameraRay(0, 0, 30, -20, 50));
  // corner 0 is (-1,-1): the lowest, leftmost ray
  closeVec(dirs[0], DP.cameraRay(-1, -1, 30, -20, 50));
  const spread = Math.acos(
    Math.max(-1, Math.min(1, dirs[0][0] * dirs[4][0] + dirs[0][1] * dirs[4][1] + dirs[0][2] * dirs[4][2])),
  );
  assert.ok(spread > (40 * Math.PI) / 180, "a corner sits well off the look direction");
});

test("exportCliParams keeps the CLI names and signs, rounding half up", () => {
  const cli = DP.exportCliParams({ yaw: 45.6, pitch: -33.4, roll: 2.5 });
  assert.deepEqual(cli, { dome_pitch: -33, dome_yaw: 46, dome_roll: 3 });
  // half-up, not banker's: Math.round(2.5) === 3 (Python's round would say 2)
  assert.equal(cli.dome_roll, 3);
  assert.deepEqual(DP.exportCliParams({}), { dome_pitch: 0, dome_yaw: 0, dome_roll: 0 });
  assert.deepEqual(DP.exportCliParams({ yaw: 90, pitch: 0, roll: 0 }), {
    dome_pitch: 0,
    dome_yaw: 90,
    dome_roll: 0,
  });
  assert.deepEqual(DP.exportCliParams({ pitch: -180 }), { dome_pitch: -180, dome_yaw: 0, dome_roll: 0 });
});

test("exportCliParams is the identity of the slider range (integer input)", () => {
  for (let yaw = -180; yaw <= 180; yaw += 7) {
    for (let pitch = -180; pitch <= 180; pitch += 13) {
      const cli = DP.exportCliParams({ yaw, pitch, roll: yaw - pitch });
      assert.equal(cli.dome_yaw, yaw);
      assert.equal(cli.dome_pitch, pitch);
      assert.equal(cli.dome_roll, yaw - pitch);
    }
  }
});
