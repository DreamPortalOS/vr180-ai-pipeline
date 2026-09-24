/* dome_proj.js — pure domemaster projection math, no dependencies.
 *
 * Browser: loaded as a classic <script>; exports onto window.DomeProj.
 * Node:   require("./dome_proj.js") returns the same object (module.exports).
 *
 * Kept deliberately small and dependency-free so the *same* formulas can be
 * replicated in studio/dome_proj.py for a cross-check (对拍) test — see
 * tests/test_studio_dome_orientation.py.  If you change a formula here, mirror
 * it there.
 *
 * Convention (DECISION_DOME / docs/FULLDOME_USAGE.md / DOME_PROMPT.txt):
 *   domemaster circle centre = zenith (头顶), circle rim = horizon,
 *   the BOTTOM of the circle = directly in front of the audience.
 *   r = colatitude / 90°, measured outward from the centre (azimuthal-
 *   equidistant).  Outside the inscribed circle is pure black.
 *
 * Orientation (v360 rorder=ypr, recovered by measurement in
 * pipeline/equirectangular_mapper.py::_orientation_matrix):
 *   R = R_yaw @ R_pitch @ R_roll   in the frame  x=right, y=up, z=forward
 *   pitch = 90° − elevation of the source centre on the dome, i.e. +pitch
 *           drops content toward the horizon (matches --dome-pitch).
 *   yaw swings content about the vertical axis; roll spins it about forward.
 *   The CLI flags are --dome-pitch / --dome-yaw / --dome-roll, same signs.
 */
(function (root) {
  "use strict";

  var HALF_PI = Math.PI / 2;
  var DEG = Math.PI / 180;

  /** Direction on the dome from colatitude theta (0=zenith, HALF_PI=horizon)
   * and azimuth phi (0=front/audience, +pi/2=right).  Frame: x right, y up,
   * z forward, so phi=0 → (0,0,+1) front, phi=pi/2 → (1,0,0) right. */
  function dirFromThetaPhi(theta, phi) {
    var st = Math.sin(theta);
    return [st * Math.sin(phi), Math.cos(theta), st * Math.cos(phi)];
  }

  /** Orientation rotation R = R_yaw @ R_pitch @ R_roll (v360 rorder=ypr).
   * orient = {yaw,pitch,roll} in degrees.  Returns the output→source matrix
   * as a flat 9-array (row-major) so a 3-vector v maps to source = M·v. */
  function orientMatrix(orient) {
    var y = (orient.yaw || 0) * DEG;
    var p = (orient.pitch || 0) * DEG;
    var r = (orient.roll || 0) * DEG;
    var cy = Math.cos(y), sy = Math.sin(y);
    var cp = Math.cos(p), sp = Math.sin(p);
    var cr = Math.cos(r), sr = Math.sin(r);
    // R_yaw = [[cy,0,sy],[0,1,0],[-sy,0,cy]]
    // R_pitch= [[1,0,0],[0,cp,sp],[0,-sp,cp]]
    // R_roll = [[cr,sr,0],[-sr,cr,0],[0,0,1]]
    var RyRp = matMul3(
      [cy, 0, sy, 0, 1, 0, -sy, 0, cy],
      [1, 0, 0, 0, cp, sp, 0, -sp, cp],
    );
    return matMul3(RyRp, [cr, sr, 0, -sr, cr, 0, 0, 0, 1]);
  }

  function matMul3(a, b) {
    var o = new Array(9);
    for (var i = 0; i < 3; i++) {
      for (var j = 0; j < 3; j++) {
        o[i * 3 + j] =
          a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j];
      }
    }
    return o;
  }

  function mvMul3(m, v) {
    return [
      m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
      m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
      m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
    ];
  }

  /** Rotate a *source* dome direction into the *output* frame: out = M⁻¹·src.
   * Since R is orthogonal, M⁻¹ = Mᵀ.  orient in degrees. */
  function sourceToOutput(src, orient) {
    var M = orientMatrix(orient);
    // transpose for the inverse (output = M^-1 @ source)
    var t = [M[0], M[3], M[6], M[1], M[4], M[7], M[2], M[5], M[8]];
    return mvMul3(t, src);
  }

  /** Rotate an *output* dome direction into the *source* frame: src = M·out. */
  function outputToSource(out, orient) {
    return mvMul3(orientMatrix(orient), out);
  }

  /** Map a direction in the OUTPUT frame to domemaster texture UV.
   * frontIsBottom=true (default) → the audience front sits at the bottom of
   * the circle (the standard convention).  false → front at the top.
   * Returns {u, v, inside} where inside is false for directions below the
   * horizon (theta > 90°, outside the inscribed circle → pure black). */
  function dirToMasterUV(d, orient, frontIsBottom) {
    if (frontIsBottom === undefined) frontIsBottom = true;
    // The output direction is what the viewer looks at; sample the source at
    // the matching place: out -> source via M.
    var src = outputToSource(d, orient);
    var nx = src[0],
      ny = src[1],
      nz = src[2];
    var len = Math.hypot(nx, ny, nz) || 1;
    nx /= len;
    ny /= len;
    nz /= len;
    var theta = Math.acos(Math.max(-1, Math.min(1, ny))); // colatitude from zenith
    if (theta > HALF_PI + 1e-6) return { u: 0.5, v: 0.5, inside: false };
    var r = theta / HALF_PI; // azimuthal-equidistant radius 0..1
    var horiz = Math.hypot(nx, nz);
    var phi, u, v;
    if (horiz < 1e-6) {
      u = 0.5;
      v = 0.5;
    } else {
      // azimuth measured from +z (front) toward +x (right): phi = atan2(x, z).
      // front (phi=0) → bottom centre, back (phi=pi) → top, right (phi=pi/2)
      // → right, left (−pi/2) → left: u tracks sin(phi), v tracks cos(phi).
      phi = Math.atan2(nx, nz);
      u = 0.5 + 0.5 * r * Math.sin(phi);
      v = 0.5 + 0.5 * r * Math.cos(phi);
      if (!frontIsBottom) v = 1 - v; // flip front to the top of the circle
    }
    return { u: u, v: v, inside: true };
  }

  /** Inverse of dirToMasterUV: a domemaster UV → direction in the OUTPUT
   * frame.  Used by the camera viewport to turn a master pixel back into the
   * dome direction it depicts, then into a camera ray. */
  function masterUVToDir(u, v, orient, frontIsBottom) {
    if (frontIsBottom === undefined) frontIsBottom = true;
    var dx = u - 0.5,
      dy = v - 0.5;
    var r = Math.hypot(dx, dy);
    if (r > 0.5 + 1e-6) return null; // outside the inscribed circle
    var theta = Math.min(1, r / 0.5) * HALF_PI; // r in [0,0.5] -> theta in [0,90]
    // inverse of dirToMasterUV: u = 0.5 + 0.5 r sin(phi), v = 0.5 + 0.5 r cos(phi)
    // (frontIsBottom); so sin(phi) = dx/s, cos(phi) = dy/s with s = 0.5, and
    // phi = atan2(sin, cos) = atan2(dx, dy).
    var cy = dy;
    if (!frontIsBottom) cy = -dy; // undo the front-to-top flip
    var phi;
    if (r < 1e-9) {
      phi = 0;
    } else {
      phi = Math.atan2(dx, cy);
    }
    var st = Math.sin(theta);
    var src = [st * Math.sin(phi), Math.cos(theta), st * Math.cos(phi)];
    return sourceToOutput(src, orient);
  }

  /** Build a camera ray direction (in the OUTPUT dome frame) for a fragment
   * at normalised coords (nx,ny in [-1,1]) given camera yaw/pitch (degrees,
   * about the dome's vertical / right axes) and half-FOV (degrees).  The
   * camera sits at the dome centre looking outward.
   *
   * Convention: a level camera looking horizontally has (yaw,pitch)=(0,0) →
   * the audience's front (+z).  +yaw turns the camera right (toward +x),
   * +pitch raises it toward the zenith.  Both are positive-up/right, like the
   * sliders, which is the opposite hand from the dome-pitch knob above (that
   * one drops content); the signs are documented next to the sliders. */
  function cameraRay(nx, ny, camYaw, camPitch, halfFovDeg) {
    var h = halfFovDeg * DEG;
    var tx = Math.tan(h) * nx;
    var ty = Math.tan(h) * ny;
    var ray = norm([tx, ty, 1]);
    var p = camPitch * DEG;
    var y = camYaw * DEG;
    var cp = Math.cos(p),
      sp = Math.sin(p);
    var cy = Math.cos(y),
      sy = Math.sin(y);
    // pitch about +x (right): +pitch raises the look direction toward +y
    var r1 = [ray[0], ray[1] * cp + ray[2] * sp, -ray[1] * sp + ray[2] * cp];
    // yaw about +y (up): +yaw turns the look direction from +z toward +x
    var r2 = [r1[0] * cy + r1[2] * sy, r1[1], -r1[0] * sy + r1[2] * cy];
    return r2;
  }

  function norm(v) {
    var l = Math.hypot(v[0], v[1], v[2]) || 1;
    return [v[0] / l, v[1] / l, v[2] / l];
  }

  /** The five frustum corner rays (near plane centre + 4 corners) in the OUTPUT
   * frame, for drawing the camera's view-volume wireframe in the 3D view.
   * Returns an array of [origin, dir] segments expressed as direction vectors
   * from the dome centre (camera at centre). */
  function cameraFrustumDirs(camYaw, camPitch, halfFovDeg) {
    var corners = [
      [-1, -1],
      [1, -1],
      [1, 1],
      [-1, 1],
    ];
    var out = [];
    for (var i = 0; i < 4; i++) {
      out.push(cameraRay(corners[i][0], corners[i][1], camYaw, camPitch, halfFovDeg));
    }
    out.push(cameraRay(0, 0, camYaw, camPitch, halfFovDeg)); // centre / look dir
    return out;
  }

  /** Export the orientation the UI is showing, as the CLI-named params the
   * convert.dome node / --dome-* flags consume.  Same names, same signs. */
  function exportCliParams(orient) {
    return {
      dome_pitch: Math.round(orient.pitch || 0),
      dome_yaw: Math.round(orient.yaw || 0),
      dome_roll: Math.round(orient.roll || 0),
    };
  }

  var api = {
    HALF_PI: HALF_PI,
    DEG: DEG,
    dirFromThetaPhi: dirFromThetaPhi,
    orientMatrix: orientMatrix,
    sourceToOutput: sourceToOutput,
    outputToSource: outputToSource,
    dirToMasterUV: dirToMasterUV,
    masterUVToDir: masterUVToDir,
    cameraRay: cameraRay,
    cameraFrustumDirs: cameraFrustumDirs,
    exportCliParams: exportCliParams,
  };

  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  }
  if (typeof window !== "undefined") {
    window.DomeProj = api;
  }
})(typeof globalThis !== "undefined" ? globalThis : this);
