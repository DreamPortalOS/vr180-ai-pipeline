/* dome_xr_proj.js — the Quest dome viewer's view-direction → master-UV formula.
 *
 * issue #434: the XR viewer MUST render a dome that matches the 2D Studio
 * preview (preview3d.js) and the shipped fulldome route, so its fragment
 * shader cannot carry a third copy of the spherical math.  This module is the
 * pure, dependency-free version of that formula — the GLSL in dome_xr.js is
 * kept staging-identical to it, and the Python mirror in studio/dome_proj.py
 * must agree with it.  tests/test_studio_xr.py pins 5 directions (zenith /
 * front / right / back / 45°仰角) against dir_to_master_uv to < 1e-3:
 *
 *   - frame: x right, y up, z forward (the audience's front is +z);
 *   - domemaster convention (DECISION_DOME): circle centre = zenith,
 *     rim = horizon, the BOTTOM of the circle = directly in front of the
 *     audience;
 *   - r = colatitude / 90° (azimuthal-equidistant), mapped onto the
 *     view-direction's tangent frame as u = 0.5 + 0.5·r·cos(φ), v =
 *     0.5 + 0.5·r·sin(φ), where the screen-right/screen-up tangent pair is
 *     t = up×v (right), b = v×t (up), and φ = atan2(b.y, t.x).  At v = front
 *     (+z) this gives t = +x, b = +y, so φ = 0 (horizontal) lands at the
 *     BOTTOM of the circle — the same convention dirToMasterUV in dome_proj.js
 *     pins for the audience front.
 *   - directions below the horizon (colatitude > 90°, outside the inscribed
 *     circle) sample pure black — inside:false.
 *
 * Browser: classic <script> → window.DomeXRProj.  Node: require() returns the
 * same object.  The shader in dome_xr.js never inlines a different formula:
 * its body is this function written out in GLSL.
 */
(function (root) {
  "use strict";

  var HALF_PI = Math.PI / 2;

  /** Map a unit view direction (x right, y up, z forward) to domemaster UV.
   * Returns {u, v, inside}.  inside:false → below the horizon (pure black on
   * a legal domemaster).  The GLSL in dome_xr.js is kept identical to this.
   *
   * Derivation (must stay mirrored with dome_proj.py's dir_to_master_uv):
   * the view axis is v = (sinθ·sinφ0, cosθ, sinθ·cosφ0) (colatitude θ from
   * the zenith, azimuth φ0 from the audience front toward +x/right).  Screen
   * right is the same-colatitude circle walked +90° in azimuth: t = (cosφ0,
   * 0, −sinφ0); screen up is b = (cosθ·sinφ0, −sinθ, cosθ·cosφ0).  The dome
   * content at azimuth φ0 is the circle's bottom centre (u=0.5, v=1.0 — the
   * bottom is the audience front per DECISION_DOME), and ±azimuth walks the
   * circle horizontally: u − 0.5 = 0.5·r·sin(φ0), v − 0.5 = 0.5·r·cos(φ0). */
  function dirToUV(x, y, z) {
    var theta = Math.acos(Math.max(-1, Math.min(1, y))); // colatitude from zenith
    if (theta > HALF_PI + 1e-6) return { u: 0.5, v: 0.5, inside: false };
    var r = theta / HALF_PI; // azimuthal-equidistant radius 0..1
    // view azimuth from the audience front toward +x: sinφ0 = x/h, cosφ0 = z/h
    // where h = horizontal extent of v.  Pure zenith (x=z=0) -> φ0 = 0.
    var h = Math.hypot(x, z);
    var sinPhi = h > 1e-6 ? x / h : 0.0;
    var cosPhi = h > 1e-6 ? z / h : 1.0;
    var u = 0.5 + 0.5 * r * sinPhi;
    var v = 0.5 + 0.5 * r * cosPhi;
    return { u: u, v: v, inside: true };
  }

  var api = {
    dirToUV: dirToUV,
  };

  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  }
  if (typeof window !== "undefined") {
    window.DomeXRProj = api;
  }
})(typeof globalThis !== "undefined" ? globalThis : this);
