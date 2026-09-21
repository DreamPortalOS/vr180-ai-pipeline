/* glkit.js — minimal WebGL + mat4 helpers (no dependencies, classic script). */
(function (global) {
  'use strict';

  // ---------------------------------------------------------------- mat4
  // Column-major 4x4 matrices, same memory layout GL expects.
  var mat4 = {
    identity: function () {
      return new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
    },

    multiply: function (a, b) {
      // returns a * b
      var o = new Float32Array(16);
      for (var c = 0; c < 4; c++) {
        for (var r = 0; r < 4; r++) {
          o[c * 4 + r] =
            a[r] * b[c * 4] +
            a[4 + r] * b[c * 4 + 1] +
            a[8 + r] * b[c * 4 + 2] +
            a[12 + r] * b[c * 4 + 3];
        }
      }
      return o;
    },

    perspective: function (fovyRad, aspect, near, far) {
      var f = 1 / Math.tan(fovyRad / 2);
      var nf = 1 / (near - far);
      return new Float32Array([
        f / aspect, 0, 0, 0,
        0, f, 0, 0,
        0, 0, (far + near) * nf, -1,
        0, 0, 2 * far * near * nf, 0
      ]);
    },

    lookAt: function (eye, center, up) {
      var z = norm3(sub3(eye, center));
      var x = norm3(cross3(up, z));
      var y = cross3(z, x);
      return new Float32Array([
        x[0], y[0], z[0], 0,
        x[1], y[1], z[1], 0,
        x[2], y[2], z[2], 0,
        -dot3(x, eye), -dot3(y, eye), -dot3(z, eye), 1
      ]);
    },

    rotationX: function (a) {
      var c = Math.cos(a), s = Math.sin(a);
      return new Float32Array([1, 0, 0, 0, 0, c, s, 0, 0, -s, c, 0, 0, 0, 0, 1]);
    },

    rotationY: function (a) {
      var c = Math.cos(a), s = Math.sin(a);
      return new Float32Array([c, 0, -s, 0, 0, 1, 0, 0, s, 0, c, 0, 0, 0, 0, 1]);
    },

    translation: function (x, y, z) {
      return new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1]);
    },

    scaling: function (s) {
      return new Float32Array([s, 0, 0, 0, 0, s, 0, 0, 0, 0, s, 0, 0, 0, 0, 1]);
    }
  };

  function sub3(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
  function dot3(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
  function cross3(a, b) {
    return [
      a[1] * b[2] - a[2] * b[1],
      a[2] * b[0] - a[0] * b[2],
      a[0] * b[1] - a[1] * b[0]
    ];
  }
  function norm3(a) {
    var l = Math.hypot(a[0], a[1], a[2]) || 1;
    return [a[0] / l, a[1] / l, a[2] / l];
  }

  // ---------------------------------------------------------------- shaders
  function compile(gl, type, src) {
    var sh = gl.createShader(type);
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
      var log = gl.getShaderInfoLog(sh);
      gl.deleteShader(sh);
      throw new Error('shader compile failed: ' + log);
    }
    return sh;
  }

  function createProgram(gl, vsSrc, fsSrc) {
    var p = gl.createProgram();
    gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, vsSrc));
    gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, fsSrc));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
      throw new Error('program link failed: ' + gl.getProgramInfoLog(p));
    }
    // Cache attribute / uniform locations by name.
    p.at = {};
    p.un = {};
    var n = gl.getProgramParameter(p, gl.ACTIVE_ATTRIBUTES), i;
    for (i = 0; i < n; i++) {
      var an = gl.getActiveAttrib(p, i).name;
      p.at[an] = gl.getAttribLocation(p, an);
    }
    n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
    for (i = 0; i < n; i++) {
      var un = gl.getActiveUniform(p, i).name.replace(/\[0\]$/, '');
      p.un[un] = gl.getUniformLocation(p, un);
    }
    return p;
  }

  function buffer(gl, data, target) {
    var b = gl.createBuffer();
    target = target || gl.ARRAY_BUFFER;
    gl.bindBuffer(target, b);
    gl.bufferData(target, data, gl.STATIC_DRAW);
    return b;
  }

  function createTexture(gl) {
    var t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    return t;
  }

  global.GLKit = {
    mat4: mat4,
    createProgram: createProgram,
    buffer: buffer,
    createTexture: createTexture
  };
})(window);
