/* geometry.js — dome / grid / projector geometry in UNIT dome space.
 *
 * Convention used everywhere in this app:
 *   +Y  = zenith (up)
 *   -Z  = dome front (the direction the audience faces)
 *   y=0 = springline plane (the dome's horizon / 起拱线)
 *   Points on the hemisphere:
 *     theta = zenith angle (0 at zenith, PI/2 at the horizon)
 *     phi   = azimuth measured from the front, increasing to the right
 *     dir   = (sin(theta)*sin(phi), cos(theta), -sin(theta)*cos(phi))
 */
(function (global) {
  'use strict';

  var HALF_PI = Math.PI / 2;

  function dirAt(theta, phi) {
    var st = Math.sin(theta);
    return [st * Math.sin(phi), Math.cos(theta), -st * Math.cos(phi)];
  }

  /** Triangulated hemisphere (inner surface). Vertices are unit directions. */
  function hemisphere(rings, segments) {
    var pos = [], idx = [], i, j;
    for (i = 0; i <= rings; i++) {
      var theta = (i / rings) * HALF_PI;
      for (j = 0; j <= segments; j++) {
        var d = dirAt(theta, (j / segments) * Math.PI * 2);
        pos.push(d[0], d[1], d[2]);
      }
    }
    var stride = segments + 1;
    for (i = 0; i < rings; i++) {
      for (j = 0; j < segments; j++) {
        var a = i * stride + j, b = a + stride;
        idx.push(a, b, a + 1, a + 1, b, b + 1);
      }
    }
    return {
      positions: new Float32Array(pos),
      indices: new Uint16Array(idx),
      count: idx.length
    };
  }

  /** Latitude circle at a given zenith angle, as a GL_LINES vertex list. */
  function latitudeCircle(theta, segments, out) {
    out = out || [];
    for (var j = 0; j < segments; j++) {
      var a = dirAt(theta, (j / segments) * Math.PI * 2);
      var b = dirAt(theta, ((j + 1) / segments) * Math.PI * 2);
      out.push(a[0], a[1], a[2], b[0], b[1], b[2]);
    }
    return out;
  }

  /** Meridian arc from zenith to horizon at azimuth phi. */
  function meridian(phi, steps, out) {
    out = out || [];
    for (var i = 0; i < steps; i++) {
      var a = dirAt((i / steps) * HALF_PI, phi);
      var b = dirAt(((i + 1) / steps) * HALF_PI, phi);
      out.push(a[0], a[1], a[2], b[0], b[1], b[2]);
    }
    return out;
  }

  /** Latitude grid every 15 deg + meridians every 30 deg. */
  function grid() {
    var v = [], t, p;
    for (t = 15; t <= 90; t += 15) latitudeCircle((t * Math.PI) / 180, 96, v);
    for (p = 0; p < 360; p += 30) meridian((p * Math.PI) / 180, 24, v);
    return new Float32Array(v);
  }

  /** Floor disc: concentric rings + spokes, drawn on the springline plane. */
  function floorPlan() {
    var v = [], k, j, segs = 72;
    for (k = 1; k <= 4; k++) {
      var r = k / 4;
      for (j = 0; j < segs; j++) {
        var a0 = (j / segs) * Math.PI * 2, a1 = ((j + 1) / segs) * Math.PI * 2;
        v.push(r * Math.sin(a0), 0, -r * Math.cos(a0),
               r * Math.sin(a1), 0, -r * Math.cos(a1));
      }
    }
    for (j = 0; j < 12; j++) {
      var a = (j / 12) * Math.PI * 2;
      v.push(0, 0, 0, Math.sin(a), 0, -Math.cos(a));
    }
    return new Float32Array(v);
  }

  /** A "front of dome" marker: an arrow on the floor pointing at -Z. */
  function frontMarker() {
    var v = [];
    v.push(0, 0.002, -0.55, 0, 0.002, -0.95);
    v.push(0, 0.002, -0.95, -0.07, 0.002, -0.83);
    v.push(0, 0.002, -0.95, 0.07, 0.002, -0.83);
    return new Float32Array(v);
  }

  /**
   * Schematic projector rig. Each unit is placed on the rim (tower) and
   * cross-fires at the opposite side of the dome, which is the usual layout
   * for perimeter-mounted fulldome rigs. The vendor never published the real
   * layout, so this is deliberately a diagram, not a calibration.
   *
   * Returns GL_LINES vertices: a small body box + a frustum cone.
   */
  function projectorRig(count, azOffsetDeg, mountY, targetElevDeg, halfAngleDeg) {
    var v = [], k;
    var targetTheta = HALF_PI - (targetElevDeg * Math.PI) / 180;
    var half = (halfAngleDeg * Math.PI) / 180;

    for (k = 0; k < count; k++) {
      var phi = ((azOffsetDeg + (k * 360) / count) * Math.PI) / 180;
      var apex = [1.02 * Math.sin(phi), mountY, -1.02 * Math.cos(phi)];
      var target = dirAt(targetTheta, phi + Math.PI);

      var axis = normalize(subtract(target, apex));
      var dist = length(subtract(target, apex));
      var up = Math.abs(axis[1]) > 0.95 ? [0, 0, -1] : [0, 1, 0];
      var right = normalize(cross(axis, up));
      var realUp = cross(right, axis);
      var rad = dist * Math.tan(half);

      // body box (small cube at the apex)
      boxLines(v, apex, right, realUp, axis, 0.045, 0.035, 0.07);

      // frustum edges + far rim
      var segs = 20, prev = null, first = null;
      for (var j = 0; j <= segs; j++) {
        var a = (j / segs) * Math.PI * 2;
        var p = [
          apex[0] + axis[0] * dist + rad * (right[0] * Math.cos(a) + realUp[0] * Math.sin(a)),
          apex[1] + axis[1] * dist + rad * (right[1] * Math.cos(a) + realUp[1] * Math.sin(a)),
          apex[2] + axis[2] * dist + rad * (right[2] * Math.cos(a) + realUp[2] * Math.sin(a))
        ];
        if (prev) v.push(prev[0], prev[1], prev[2], p[0], p[1], p[2]);
        else first = p;
        prev = p;
        if (j % 5 === 0) v.push(apex[0], apex[1], apex[2], p[0], p[1], p[2]);
      }
      if (first && prev) v.push(prev[0], prev[1], prev[2], first[0], first[1], first[2]);
    }
    return new Float32Array(v);
  }

  function boxLines(v, c, ax, ay, az, sx, sy, sz) {
    var corners = [];
    for (var i = 0; i < 8; i++) {
      var dx = (i & 1 ? 1 : -1) * sx / 2;
      var dy = (i & 2 ? 1 : -1) * sy / 2;
      var dz = (i & 4 ? 1 : -1) * sz / 2;
      corners.push([
        c[0] + ax[0] * dx + ay[0] * dy + az[0] * dz,
        c[1] + ax[1] * dx + ay[1] * dy + az[1] * dz,
        c[2] + ax[2] * dx + ay[2] * dy + az[2] * dz
      ]);
    }
    var edges = [0, 1, 1, 3, 3, 2, 2, 0, 4, 5, 5, 7, 7, 6, 6, 4, 0, 4, 1, 5, 2, 6, 3, 7];
    for (var e = 0; e < edges.length; e += 2) {
      var a = corners[edges[e]], b = corners[edges[e + 1]];
      v.push(a[0], a[1], a[2], b[0], b[1], b[2]);
    }
  }

  function subtract(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
  function length(a) { return Math.hypot(a[0], a[1], a[2]); }
  function normalize(a) { var l = length(a) || 1; return [a[0] / l, a[1] / l, a[2] / l]; }
  function cross(a, b) {
    return [
      a[1] * b[2] - a[2] * b[1],
      a[2] * b[0] - a[0] * b[2],
      a[0] * b[1] - a[1] * b[0]
    ];
  }

  global.DomeGeometry = {
    dirAt: dirAt,
    hemisphere: hemisphere,
    latitudeCircle: latitudeCircle,
    grid: grid,
    floorPlan: floorPlan,
    frontMarker: frontMarker,
    projectorRig: projectorRig
  };
})(window);
