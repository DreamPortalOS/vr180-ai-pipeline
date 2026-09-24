/* Studio 3D dome preview — offline WebGL, no CDN.
 * Geometry convention matches web/dome-preview and dome_proj.js:
 *   +Y zenith, -Z front (audience), y=0 springline, r = zenith/90°.
 *   domemaster circle centre = zenith, rim = horizon, BOTTOM of circle =
 *   directly in front of the audience (DECISION_DOME / docs/FULLDOME_USAGE.md).
 *   Orientation (yaw/pitch/roll) follows v360 rorder=ypr; see dome_proj.js.
 */
(() => {
  "use strict";

  const HALF_PI = Math.PI / 2;

  const VENUE = {
    diameter_m: 12,
    tilt_deg: 0,
    projectors_lower: 5,
    projectors_upper: 3,
    projector_model: "Optoma CUL80T",
    server: "HDX8",
    seats: 79,
    standees: 150,
    master: "4096×4096 domemaster (mono)",
    stereo: false,
    lux_est: 198,
    surface_m2: 226.2,
  };

  const RATIOS = [
    {
      id: "src_1x1",
      label: "AI 源片 1:1",
      w: 1,
      h: 1,
      note: "生成默认方图；1:1 利于 180° 铺开",
      color: "#3d9cf0",
    },
    {
      id: "vr180_eye",
      label: "VR180 单眼",
      w: 1,
      h: 1,
      note: "方图/眼（2880²–4096²）；SBS 总宽 2×",
      color: "#b388ff",
    },
    {
      id: "vr180_sbs",
      label: "VR180 SBS 并排",
      w: 2,
      h: 1,
      note: "双眼并排 + sv3d/st3d",
      color: "#9b7bff",
    },
    {
      id: "domemaster",
      label: "球幕 domemaster",
      w: 1,
      h: 1,
      note: "4096² 内切圆=180° 半球；角部纯黑",
      color: "#e6b450",
    },
    {
      id: "flat_169",
      label: "普通 16:9 素材",
      w: 16,
      h: 9,
      note: "直接映射会糊、需外绘/加大 hfov",
      color: "#8fa0b5",
    },
  ];

  // ------------------------------------------------------- small mat4 kit
  function mat4Identity() {
    return new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
  }
  function mat4Perspective(fovy, aspect, near, far) {
    const f = 1 / Math.tan(fovy / 2);
    const nf = 1 / (near - far);
    return new Float32Array([
      f / aspect, 0, 0, 0,
      0, f, 0, 0,
      0, 0, (far + near) * nf, -1,
      0, 0, 2 * far * near * nf, 0,
    ]);
  }
  function mat4LookAt(eye, center, up) {
    const zx = eye[0] - center[0], zy = eye[1] - center[1], zz = eye[2] - center[2];
    let len = Math.hypot(zx, zy, zz) || 1;
    const z = [zx / len, zy / len, zz / len];
    let xx = up[1] * z[2] - up[2] * z[1];
    let xy = up[2] * z[0] - up[0] * z[2];
    let xz = up[0] * z[1] - up[1] * z[0];
    len = Math.hypot(xx, xy, xz) || 1;
    const x = [xx / len, xy / len, xz / len];
    const y = [
      z[1] * x[2] - z[2] * x[1],
      z[2] * x[0] - z[0] * x[2],
      z[0] * x[1] - z[1] * x[0],
    ];
    return new Float32Array([
      x[0], y[0], z[0], 0,
      x[1], y[1], z[1], 0,
      x[2], y[2], z[2], 0,
      -(x[0] * eye[0] + x[1] * eye[1] + x[2] * eye[2]),
      -(y[0] * eye[0] + y[1] * eye[1] + y[2] * eye[2]),
      -(z[0] * eye[0] + z[1] * eye[1] + z[2] * eye[2]),
      1,
    ]);
  }
  function mat4Mul(a, b) {
    const o = new Float32Array(16);
    for (let c = 0; c < 4; c++) {
      for (let r = 0; r < 4; r++) {
        o[c * 4 + r] =
          a[r] * b[c * 4] +
          a[4 + r] * b[c * 4 + 1] +
          a[8 + r] * b[c * 4 + 2] +
          a[12 + r] * b[c * 4 + 3];
      }
    }
    return o;
  }

  function dirAt(theta, phi) {
    const st = Math.sin(theta);
    return [st * Math.sin(phi), Math.cos(theta), -st * Math.cos(phi)];
  }

  function hemisphereMesh(rings, segments) {
    const pos = [];
    const idx = [];
    for (let i = 0; i <= rings; i++) {
      const theta = (i / rings) * HALF_PI;
      for (let j = 0; j <= segments; j++) {
        const d = dirAt(theta, (j / segments) * Math.PI * 2);
        pos.push(d[0], d[1], d[2]);
      }
    }
    const stride = segments + 1;
    for (let i = 0; i < rings; i++) {
      for (let j = 0; j < segments; j++) {
        const a = i * stride + j;
        const b = a + stride;
        idx.push(a, b, a + 1, a + 1, b, b + 1);
      }
    }
    return { positions: new Float32Array(pos), indices: new Uint16Array(idx) };
  }

  function latitudeCircle(theta, segments) {
    const v = [];
    for (let j = 0; j < segments; j++) {
      const a = dirAt(theta, (j / segments) * Math.PI * 2);
      const b = dirAt(theta, ((j + 1) / segments) * Math.PI * 2);
      v.push(a[0], a[1], a[2], b[0], b[1], b[2]);
    }
    return new Float32Array(v);
  }

  function meridian(phi, steps) {
    const v = [];
    for (let i = 0; i < steps; i++) {
      const a = dirAt((i / steps) * HALF_PI, phi);
      const b = dirAt(((i + 1) / steps) * HALF_PI, phi);
      v.push(a[0], a[1], a[2], b[0], b[1], b[2]);
    }
    return new Float32Array(v);
  }

  function gridLines() {
    const parts = [];
    for (let t = 15; t <= 90; t += 15) parts.push(latitudeCircle((t * Math.PI) / 180, 72));
    for (let p = 0; p < 360; p += 45) parts.push(meridian((p * Math.PI) / 180, 18));
    // flatten
    let n = 0;
    parts.forEach((p) => (n += p.length));
    const out = new Float32Array(n);
    let o = 0;
    parts.forEach((p) => {
      out.set(p, o);
      o += p.length;
    });
    return out;
  }

  function compile(gl, vsSrc, fsSrc) {
    function sh(type, src) {
      const s = gl.createShader(type);
      gl.shaderSource(s, src);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
        throw new Error(gl.getShaderInfoLog(s) || "shader compile failed");
      }
      return s;
    }
    const p = gl.createProgram();
    gl.attachShader(p, sh(gl.VERTEX_SHADER, vsSrc));
    gl.attachShader(p, sh(gl.FRAGMENT_SHADER, fsSrc));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
      throw new Error(gl.getProgramInfoLog(p) || "link failed");
    }
    return p;
  }

  class DomePreview {
    constructor(canvas, opts = {}) {
      this.canvas = canvas;
      this.gl = canvas.getContext("webgl", { antialias: true, alpha: false });
      if (!this.gl) throw new Error("WebGL unavailable");
      const gl = this.gl;
      this.coverageR = opts.coverageR != null ? opts.coverageR : -1;
      this.hasTexture = false;
      this.tex = gl.createTexture();
      this.orbit = { az: 0.7, el: 0.55, dist: 2.85 };
      this.drag = null;
      this.raf = 0;
      // orientation the dome content is shown at (yaw/pitch/roll, degrees) plus
      // the quick flips.  frontIsBottom is the standard convention (audience
      // front = bottom of the circle); uFlip/vFlip are ±1 multipliers that
      // implement 水平/垂直翻转 and the 正前方=圆上方 toggle.
      this.orient = { yaw: 0, pitch: 0, roll: 0 };
      this.frontIsBottom = true;
      this.uFlip = 1;
      this.vFlip = 1;
      // virtual camera for the 2D viewport (yaw/pitch deg, half-FOV deg).
      this.camera = { yaw: 0, pitch: 45, halfFov: 45 };
      this.cameraShow = false;
      this.onOrientChange = null;
      this._initPrograms();
      this._initBuffers();
      this._bindEvents();
      this._loop();
    }

    /** Build the output→source mat3 for uOrient as a column-major
     * Float32Array[9] (what WebGL's uniformMatrix3fv expects with
     * transpose=false).  dome_proj.js's orientMatrix is row-major, so we
     * transpose here — the math is otherwise identical. */
    _orientMat() {
      const DP = window.DomeProj;
      if (!DP) return new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
      const m = DP.orientMatrix(this.orient); // row-major [0..8]
      // transpose: column-major = [m0,m3,m6, m1,m4,m7, m2,m5,m8]
      return new Float32Array([m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]]);
    }

    /** The vertical-flip multiplier the dome and camera shaders upload as
     * uVFlip, composing frontIsBottom with the quick vFlip — the tested pure
     * formula in DomeProj.effectiveVFlip, with the inline fallback matching it
     * should the module not have loaded (it always loads before preview3d.js). */
    _effVFlip() {
      const DP = window.DomeProj;
      return DP
        ? DP.effectiveVFlip(this.frontIsBottom, this.vFlip)
        : this.frontIsBottom
          ? this.vFlip
          : -this.vFlip;
    }

    /** Set yaw/pitch/roll (degrees) and recompute; emits onOrientChange. */
    setOrientation(o) {
      this.orient.yaw = o.yaw != null ? o.yaw : this.orient.yaw;
      this.orient.pitch = o.pitch != null ? o.pitch : this.orient.pitch;
      this.orient.roll = o.roll != null ? o.roll : this.orient.roll;
      if (o.frontIsBottom != null) {
        // Do NOT fold frontIsBottom into vFlip's sign here: the shaders compose
        // the two via DomeProj.effectiveVFlip(frontIsBottom, vFlip), and mutating
        // vFlip too double-negated the front-to-top toggle into a no-op.  Keep
        // the two states independent — the 「正前方=圆下方/圆上方」 toggle and the
        // quick 垂直翻转 button each flip on their own.
        this.frontIsBottom = o.frontIsBottom;
      }
      if (o.uFlip != null) this.uFlip = o.uFlip;
      if (o.vFlip != null) this.vFlip = o.vFlip;
      if (this.onOrientChange) this.onOrientChange(this.getOrientation());
    }

    getOrientation() {
      return {
        yaw: this.orient.yaw,
        pitch: this.orient.pitch,
        roll: this.orient.roll,
        frontIsBottom: this.frontIsBottom,
        uFlip: this.uFlip,
        vFlip: this.vFlip,
        cli: window.DomeProj ? window.DomeProj.exportCliParams(this.orient) : null,
      };
    }

    setCamera(c) {
      if (c.yaw != null) this.camera.yaw = c.yaw;
      if (c.pitch != null) this.camera.pitch = c.pitch;
      if (c.halfFov != null) this.camera.halfFov = c.halfFov;
      if (this.onCamChange) this.onCamChange();
    }

    _initPrograms() {
      const gl = this.gl;
      this.domeProg = compile(
        gl,
        [
          "attribute vec3 aDir;",
          "uniform mat4 uMVP;",
          "uniform float uRadius;",
          "varying vec3 vDir;",
          "void main(){ vDir=aDir; gl_Position=uMVP*vec4(aDir*uRadius,1.0); }",
        ].join("\n"),
        [
          "precision highp float;",
          "varying vec3 vDir;",
          "uniform sampler2D uTex;",
          "uniform float uHasTex;",
          "uniform float uCovR;",
          "uniform mat3 uOrient;", // output→source (v360 rorder=ypr), front=+z
          "uniform float uUFlip;", // quick horizontal flip (-1/1)
          "uniform float uVFlip;", // quick vertical flip / front-on-top (-1/1)
          "void main(){",
          "  vec3 d=normalize(uOrient*vec3(vDir.x, vDir.y, -vDir.z));",
          "  float theta=acos(clamp(d.y,-1.0,1.0));",
          "  float r=clamp(theta/1.5707963267,0.0,1.0);",
          "  float h=length(d.xz);",
          "  vec2 uv=vec2(0.5,0.5);",
          "  float phi=0.0;",
          // front = +z (matches dome_proj.js): phi = atan(d.x, d.z).
          // front -> BOTTOM of circle: u tracks sin(phi), v tracks cos(phi).
          "  if(h>1e-5){ phi=atan(d.x, d.z);",
          "    uv=vec2(0.5+0.5*r*sin(phi)*uUFlip, 0.5+0.5*r*cos(phi)*uVFlip); }",
          "  vec3 col;",
          "  if(uHasTex>0.5){ col=texture2D(uTex,uv).rgb; }",
          "  else {",
          "    float band=mod(floor(r*6.0)+floor((phi+3.14159265)/0.7853981634),2.0);",
          "    col=mix(vec3(0.10,0.12,0.16), vec3(0.16,0.19,0.25), band);",
          "  }",
          "  if(uCovR>=0.0 && r>uCovR+0.002){",
          "    float stripe=step(0.5,fract((r-uCovR)*24.0));",
          "    col=mix(col, vec3(1.0,0.40,0.26), 0.22+0.16*stripe);",
          "  }",
          "  if(uCovR>=0.0 && abs(r-uCovR)<0.006){ col=vec3(1.0,0.76,0.26); }",
          "  gl_FragColor=vec4(col,1.0);",
          "}",
        ].join("\n"),
      );
      this.lineProg = compile(
        gl,
        [
          "attribute vec3 aPos;",
          "uniform mat4 uMVP;",
          "uniform float uRadius;",
          "void main(){ gl_Position=uMVP*vec4(aPos*uRadius,1.0); }",
        ].join("\n"),
        [
          "precision mediump float;",
          "uniform vec4 uColor;",
          "void main(){ gl_FragColor=uColor; }",
        ].join("\n"),
      );
      this.uDome = {
        mvp: gl.getUniformLocation(this.domeProg, "uMVP"),
        radius: gl.getUniformLocation(this.domeProg, "uRadius"),
        tex: gl.getUniformLocation(this.domeProg, "uTex"),
        hasTex: gl.getUniformLocation(this.domeProg, "uHasTex"),
        covR: gl.getUniformLocation(this.domeProg, "uCovR"),
        orient: gl.getUniformLocation(this.domeProg, "uOrient"),
        uFlip: gl.getUniformLocation(this.domeProg, "uUFlip"),
        vFlip: gl.getUniformLocation(this.domeProg, "uVFlip"),
        aDir: gl.getAttribLocation(this.domeProg, "aDir"),
      };
      this.uLine = {
        mvp: gl.getUniformLocation(this.lineProg, "uMVP"),
        radius: gl.getUniformLocation(this.lineProg, "uRadius"),
        color: gl.getUniformLocation(this.lineProg, "uColor"),
        aPos: gl.getAttribLocation(this.lineProg, "aPos"),
      };
    }

    _initBuffers() {
      const gl = this.gl;
      const mesh = hemisphereMesh(24, 48);
      this.domePos = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, this.domePos);
      gl.bufferData(gl.ARRAY_BUFFER, mesh.positions, gl.STATIC_DRAW);
      this.domeIdx = gl.createBuffer();
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.domeIdx);
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, mesh.indices, gl.STATIC_DRAW);
      this.domeCount = mesh.indices.length;

      this.gridBuf = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, this.gridBuf);
      gl.bufferData(gl.ARRAY_BUFFER, gridLines(), gl.STATIC_DRAW);
      this.gridCount = gridLines().length / 3;

      // coverage edge circle at current radius (rebuilt on setCoverage)
      this.covBuf = gl.createBuffer();
      this.covCount = 0;

      // projector dots (unit dirs)
      const lower = [];
      for (let i = 0; i < VENUE.projectors_lower; i++) {
        const az = (i / VENUE.projectors_lower) * Math.PI * 2;
        lower.push(dirAt((70 * Math.PI) / 180, az));
      }
      const upper = [];
      for (let i = 0; i < VENUE.projectors_upper; i++) {
        const az = ((i + 0.5) / VENUE.projectors_upper) * Math.PI * 2;
        upper.push(dirAt((35 * Math.PI) / 180, az));
      }
      const pts = [];
      lower.forEach((d) => pts.push(d[0], d[1], d[2]));
      upper.forEach((d) => pts.push(d[0], d[1], d[2]));
      this.projBuf = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, this.projBuf);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(pts), gl.STATIC_DRAW);
      this.projLower = VENUE.projectors_lower;
      this.projUpper = VENUE.projectors_upper;
    }

    setCoverage(r) {
      this.coverageR = typeof r === "number" && r >= 0 ? Math.min(1, r) : -1;
      if (this.coverageR < 0) {
        this.covCount = 0;
        return;
      }
      const v = latitudeCircle(this.coverageR * HALF_PI, 96);
      const gl = this.gl;
      gl.bindBuffer(gl.ARRAY_BUFFER, this.covBuf);
      gl.bufferData(gl.ARRAY_BUFFER, v, gl.STATIC_DRAW);
      this.covCount = v.length / 3;
    }

    setImage(img) {
      const gl = this.gl;
      if (!img) {
        this.hasTexture = false;
        return;
      }
      gl.bindTexture(gl.TEXTURE_2D, this.tex);
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      this.hasTexture = true;
    }

    _bindEvents() {
      const c = this.canvas;
      c.addEventListener("mousedown", (e) => {
        this.drag = { x: e.clientX, y: e.clientY, az: this.orbit.az, el: this.orbit.el };
      });
      window.addEventListener("mouseup", () => {
        this.drag = null;
      });
      window.addEventListener("mousemove", (e) => {
        if (!this.drag) return;
        this.orbit.az = this.drag.az + (e.clientX - this.drag.x) * 0.01;
        this.orbit.el = Math.max(0.05, Math.min(1.35, this.drag.el + (e.clientY - this.drag.y) * 0.01));
      });
      c.addEventListener(
        "wheel",
        (e) => {
          e.preventDefault();
          this.orbit.dist = Math.max(1.6, Math.min(5, this.orbit.dist + e.deltaY * 0.002));
        },
        { passive: false },
      );
    }

    _mvp() {
      const w = this.canvas.clientWidth || this.canvas.width;
      const h = this.canvas.clientHeight || this.canvas.height;
      const proj = mat4Perspective((55 * Math.PI) / 180, w / Math.max(1, h), 0.1, 20);
      const az = this.orbit.az;
      const el = this.orbit.el;
      const d = this.orbit.dist;
      const eye = [
        d * Math.cos(el) * Math.sin(az),
        d * Math.sin(el) + 0.15,
        d * Math.cos(el) * Math.cos(az),
      ];
      const view = mat4LookAt(eye, [0, 0.15, 0], [0, 1, 0]);
      return mat4Mul(proj, view);
    }

    _draw() {
      const gl = this.gl;
      const w = this.canvas.clientWidth || 300;
      const h = this.canvas.clientHeight || 200;
      const dpr = window.devicePixelRatio || 1;
      const pw = Math.floor(w * dpr);
      const ph = Math.floor(h * dpr);
      if (this.canvas.width !== pw || this.canvas.height !== ph) {
        this.canvas.width = pw;
        this.canvas.height = ph;
      }
      gl.viewport(0, 0, pw, ph);
      gl.clearColor(0.04, 0.055, 0.08, 1);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.enable(gl.DEPTH_TEST);
      gl.disable(gl.CULL_FACE);
      const mvp = this._mvp();

      // dome
      gl.useProgram(this.domeProg);
      gl.uniformMatrix4fv(this.uDome.mvp, false, mvp);
      gl.uniform1f(this.uDome.radius, 1.0);
      gl.uniform1f(this.uDome.hasTex, this.hasTexture ? 1 : 0);
      gl.uniform1f(this.uDome.covR, this.coverageR);
      gl.uniformMatrix3fv(this.uDome.orient, false, this._orientMat());
      gl.uniform1f(this.uDome.uFlip, this.uFlip);
      gl.uniform1f(this.uDome.vFlip, this._effVFlip());
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.tex);
      gl.uniform1i(this.uDome.tex, 0);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.domePos);
      gl.enableVertexAttribArray(this.uDome.aDir);
      gl.vertexAttribPointer(this.uDome.aDir, 3, gl.FLOAT, false, 0, 0);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.domeIdx);
      gl.drawElements(gl.TRIANGLES, this.domeCount, gl.UNSIGNED_SHORT, 0);

      // camera frustum wireframe in the 3D view
      this._drawCameraFrustum(mvp);

      // lines
      gl.useProgram(this.lineProg);
      gl.uniformMatrix4fv(this.uLine.mvp, false, mvp);
      gl.uniform1f(this.uLine.radius, 1.01);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.gridBuf);
      gl.enableVertexAttribArray(this.uLine.aPos);
      gl.vertexAttribPointer(this.uLine.aPos, 3, gl.FLOAT, false, 0, 0);
      gl.uniform4f(this.uLine.color, 0.35, 0.45, 0.58, 1);
      gl.drawArrays(gl.LINES, 0, this.gridCount);

      if (this.covCount > 0) {
        gl.uniform1f(this.uLine.radius, 1.02);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.covBuf);
        gl.vertexAttribPointer(this.uLine.aPos, 3, gl.FLOAT, false, 0, 0);
        gl.uniform4f(this.uLine.color, 1.0, 0.76, 0.26, 1);
        gl.drawArrays(gl.LINES, 0, this.covCount);
      }

      // projectors as small squares via LINES cross
      // draw points with gl.POINTS
      gl.bindBuffer(gl.ARRAY_BUFFER, this.projBuf);
      gl.vertexAttribPointer(this.uLine.aPos, 3, gl.FLOAT, false, 0, 0);
      gl.uniform1f(this.uLine.radius, 0.98);
      // lower ring
      gl.uniform4f(this.uLine.color, 0.3, 0.85, 0.55, 1);
      // reuse points: draw as points
      gl.drawArrays(gl.POINTS, 0, this.projLower + this.projUpper);
    }

    /** The camera's frustum as a wireframe drawn from the dome centre out
     * toward its corners — 4 corner rays + the 4 face edges + the near-plane
     * quad, rebuilt whenever the camera yaw/pitch/FOV change.
     *
     * The corner rays are walked around the near-plane perimeter (-1,-1),
     * (1,-1), (1,1), (-1,1) so the quad's edges are adjacent corners, not
     * diagonals.  They come back in the projection frame (+z = audience front)
     * and are bridged into the scene's geometry frame (-z = front) before being
     * drawn — see DomeProj.domeRayToSceneDir; without the bridge the wireframe
     * renders 180° out in yaw against the dome content it sits on. */
    _buildFrustum() {
      const DP = window.DomeProj;
      if (!DP) return [];
      const { yaw, pitch, halfFov } = this.camera;
      const corners = [
        [-1, -1],
        [1, -1],
        [1, 1],
        [-1, 1],
      ];
      const bridge = DP.domeRayToSceneDir
        ? (d) => DP.domeRayToSceneDir(d)
        : (d) => [d[0], d[1], -d[2]]; // fallback = the same negation
      const c = corners.map((k) => bridge(DP.cameraRay(k[0], k[1], yaw, pitch, halfFov)));
      const look = bridge(DP.cameraRay(0, 0, yaw, pitch, halfFov));
      const L = 1.15; // frustum length (just past the dome surface)
      const segs = [];
      c.forEach((d) => {
        segs.push(0, 0, 0, d[0] * L, d[1] * L, d[2] * L);
      });
      // near-plane quad edges, corner i -> corner i+1 (wrapping), i.e. the
      // perimeter of the near plane rather than its diagonals
      for (let i = 0; i < 4; i++) {
        const a = c[i];
        const b = c[(i + 1) % 4];
        segs.push(a[0] * L, a[1] * L, a[2] * L, b[0] * L, b[1] * L, b[2] * L);
      }
      segs.push(0, 0, 0, look[0] * L, look[1] * L, look[2] * L); // look direction
      return segs;
    }

    _drawCameraFrustum(mvp) {
      if (!this.cameraShow) return;
      const gl = this.gl;
      const segs = this._buildFrustum();
      if (!segs.length) return;
      const data = new Float32Array(segs);
      if (!this.frustumBuf) this.frustumBuf = gl.createBuffer();
      gl.useProgram(this.lineProg);
      gl.uniformMatrix4fv(this.uLine.mvp, false, mvp);
      gl.uniform1f(this.uLine.radius, 1.0);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.frustumBuf);
      gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_DRAW);
      gl.enableVertexAttribArray(this.uLine.aPos);
      gl.vertexAttribPointer(this.uLine.aPos, 3, gl.FLOAT, false, 0, 0);
      gl.uniform4f(this.uLine.color, 0.24, 0.95, 0.7, 1);
      gl.drawArrays(gl.LINES, 0, segs.length / 3);
    }

    _loop() {
      const tick = () => {
        this._draw();
        this.raf = requestAnimationFrame(tick);
      };
      tick();
    }

    destroy() {
      cancelAnimationFrame(this.raf);
    }
  }

  // ------------------------------------------------------- camera 2D viewport
  /* Renders the perspective 2D frame a virtual camera at the dome centre
   * would see, by inverse-projecting each fragment's camera ray back onto the
   * domemaster texture (dome_proj.js).  Shares the DomePreview's WebGL texture
   * so a freshly loaded master is visible immediately. */
  class CameraView {
    constructor(canvas, preview) {
      this.canvas = canvas;
      this.preview = preview;
      this.gl = canvas.getContext("webgl", { antialias: true, alpha: false });
      if (!this.gl) throw new Error("WebGL unavailable");
      this.raf = 0;
      this._init();
      this._loop();
    }

    _init() {
      const gl = this.gl;
      this.prog = compile(
        gl,
        [
          "attribute vec2 aPos;",
          "varying vec2 vNdc;",
          "void main(){ vNdc=aPos; gl_Position=vec4(aPos,0.0,1.0); }",
        ].join("\n"),
        [
          "precision highp float;",
          "varying vec2 vNdc;",
          "uniform sampler2D uTex;",
          "uniform float uHasTex;",
          "uniform mat3 uOrient;",
          "uniform float uUFlip;",
          "uniform float uVFlip;",
          "uniform vec3 uCam;", // (yaw, pitch deg, halfFov deg)
          "const float PI=3.14159265358979;",
          "const float HPI=1.57079632679490;",
          // dirToMasterUV, inlined to match dome_proj.js exactly.
          "vec2 dirToUV(vec3 d){",
          "  d=normalize(uOrient*d);",
          "  float theta=acos(clamp(d.y,-1.0,1.0));",
          "  if(theta>HPI+1e-3) return vec2(-1.0);",
          "  float r=clamp(theta/HPI,0.0,1.0);",
          "  float h=length(d.xz);",
          "  if(h<1e-5) return vec2(0.5,0.5);",
          "  float phi=atan(d.x, d.z);",
          "  return vec2(0.5+0.5*r*sin(phi)*uUFlip, 0.5+0.5*r*cos(phi)*uVFlip);",
          "}",
          // cameraRay, inlined (positive-up/right, +yaw->+x, +pitch->+y).
          "vec3 camRay(vec2 ndc, float yaw, float pitch, float halfFov){",
          "  float hh=halfFov*PI/180.0;",
          "  vec3 ray=normalize(vec3(tan(hh)*ndc.x, tan(hh)*ndc.y, 1.0));",
          "  float p=pitch*PI/180.0, y=yaw*PI/180.0;",
          "  float cp=cos(p), sp=sin(p), cy=cos(y), sy=sin(y);",
          "  vec3 r1=vec3(ray.x, ray.y*cp+ray.z*sp, -ray.y*sp+ray.z*cp);",
          "  return vec3(r1.x*cy+r1.z*sy, r1.y, -r1.x*sy+r1.z*cy);",
          "}",
          "void main(){",
          "  vec3 ray=camRay(vNdc, uCam.x, uCam.y, uCam.z);",
          "  vec2 uv=dirToUV(ray);",
          "  if(uv.x<0.0){ gl_FragColor=vec4(0.02,0.03,0.05,1.0); return; }",
          "  vec3 col=uHasTex>0.5 ? texture2D(uTex,uv).rgb",
          "    : vec3(0.08,0.10,0.13)*(0.5+0.5*ray.y);",
          // subtle vignette so the camera frame reads as a view, not the master
          "  float vig=1.0-0.18*dot(vNdc,vNdc);",
          "  gl_FragColor=vec4(col*vig,1.0);",
          "}",
        ].join("\n"),
      );
      this.u = {
        tex: gl.getUniformLocation(this.prog, "uTex"),
        hasTex: gl.getUniformLocation(this.prog, "uHasTex"),
        orient: gl.getUniformLocation(this.prog, "uOrient"),
        uFlip: gl.getUniformLocation(this.prog, "uUFlip"),
        vFlip: gl.getUniformLocation(this.prog, "uVFlip"),
        cam: gl.getUniformLocation(this.prog, "uCam"),
        aPos: gl.getAttribLocation(this.prog, "aPos"),
      };
      const q = new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]);
      this.quad = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, this.quad);
      gl.bufferData(gl.ARRAY_BUFFER, q, gl.STATIC_DRAW);
    }

    _draw() {
      const gl = this.gl;
      const w = this.canvas.clientWidth || 256;
      const h = this.canvas.clientHeight || 160;
      const dpr = window.devicePixelRatio || 1;
      const pw = Math.floor(w * dpr);
      const ph = Math.floor(h * dpr);
      if (this.canvas.width !== pw || this.canvas.height !== ph) {
        this.canvas.width = pw;
        this.canvas.height = ph;
      }
      gl.viewport(0, 0, pw, ph);
      gl.clearColor(0.02, 0.03, 0.05, 1);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.useProgram(this.prog);
      const p = this.preview;
      gl.uniform1f(this.u.hasTex, p.hasTexture ? 1 : 0);
      gl.uniformMatrix3fv(this.u.orient, false, p._orientMat());
      gl.uniform1f(this.u.uFlip, p.uFlip);
      gl.uniform1f(this.u.vFlip, p._effVFlip());
      gl.uniform3f(this.u.cam, p.camera.yaw, p.camera.pitch, p.camera.halfFov);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, p.tex);
      gl.uniform1i(this.u.tex, 0);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.quad);
      gl.enableVertexAttribArray(this.u.aPos);
      gl.vertexAttribPointer(this.u.aPos, 2, gl.FLOAT, false, 0, 0);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }

    _loop() {
      const tick = () => {
        this._draw();
        this.raf = requestAnimationFrame(tick);
      };
      tick();
    }

    destroy() {
      cancelAnimationFrame(this.raf);
    }
  }

  // ------------------------------------------------------------ analysis
  // Coverage is measured by the SERVER, not in the browser. The 3D panel POSTs
  // its image to /api/coverage, which runs studio.coverage.analyze_frame — the
  // one shared full-ring-fill scan used by scripts/dome_qa.py and the Studio
  // qa.dome_coverage node — so the panel, the node and the release gate read
  // the same number for the same master. The old client-side scan
  // (analyzeImage / edgeAt(0.98), a third algorithm) is gone: it could disagree
  // with the server on the same frame (issue #405). The frontend now only draws
  // the coverage circle; the readout text is the server's.

  function canvasToBlob(canvas) {
    return new Promise((resolve, reject) => {
      canvas.toBlob(
        (blob) => (blob ? resolve(blob) : reject(new Error("canvas.toBlob returned no blob"))),
        "image/png",
        90,
      );
    });
  }

  /** Draw *img* onto a capped offscreen canvas (keeps the upload small; the
   * server downscales to its own ANALYZE_MAX_DIM, coverage r/R is scale-invariant). */
  function imageToCanvas(img, maxDim = 512) {
    const scale = Math.min(1, maxDim / Math.max(img.width || 1, img.height || 1));
    const w = Math.max(1, Math.round((img.width || 1) * scale));
    const h = Math.max(1, Math.round((img.height || 1) * scale));
    const c = document.createElement("canvas");
    c.width = w;
    c.height = h;
    const ctx = c.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(img, 0, 0, w, h);
    return c;
  }

  /** tiny deterministic PRNG so preset domes are textured (not a flat pedestal
   * the server's texture-energy scan would ignore) and stable across clicks. */
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  /** Synthetic domemaster filled out to r=fillR with random texture; outside
   * the fill and outside the inscribed circle is pure black. The server reads
   * this image, so a preset's verdict comes from the shared scan, not a number
   * hardcoded in the button. */
  function makeDomeCanvas(fillR, size = 256, seed = 0xfeed) {
    const c = document.createElement("canvas");
    c.width = size;
    c.height = size;
    const ctx = c.getContext("2d", { willReadFrequently: true });
    const cx = (size - 1) / 2;
    const cy = (size - 1) / 2;
    const R = size / 2;
    const rng = mulberry32(seed);
    const img = ctx.createImageData(size, size);
    const d = img.data;
    for (let y = 0; y < size; y++) {
      const dy = y - cy;
      for (let x = 0; x < size; x++) {
        const i = (y * size + x) * 4;
        const rr = Math.sqrt((x - cx) * (x - cx) + dy * dy) / R;
        if (rr <= 1.0 && rr <= fillR) {
          const base = rng() * 255;
          d[i] = base;
          d[i + 1] = (base + rng() * 64) % 256;
          d[i + 2] = (base + rng() * 128) % 256;
          d[i + 3] = 255;
        } else {
          d[i] = 0;
          d[i + 1] = 0;
          d[i + 2] = 0;
          d[i + 3] = 255;
        }
      }
    }
    ctx.putImageData(img, 0, 0);
    return c;
  }

  /** Map the server's snake_case CoverageStats to the camelCase shape the rest
   * of the UI (setCoverageUi / applyStats) reads — the same shape the old
   * client-side analyzeImage returned, so no other call site changes. */
  function normalizeCoverage(d) {
    return {
      coverageRadius: d.coverage_radius,
      coverageDeg: d.coverage_deg,
      softRadius: d.soft_radius,
      outerFill: d.outer_fill,
      solidAngleFrac: d.solid_angle_frac,
      level: d.level,
      text: d.text,
    };
  }

  /** POST a canvas image to the server and return its coverage stats
   * (camelCase). Resolves to {error} on any failure so the UI degrades. */
  async function analyzeOnServer(canvas) {
    let blob;
    try {
      blob = await canvasToBlob(canvas);
    } catch (err) {
      return { error: err.message || "toBlob failed" };
    }
    try {
      const res = await fetch("/api/coverage", {
        method: "POST",
        headers: { "Content-Type": "image/png" },
        body: blob,
      });
      if (!res.ok) {
        const txt = await res.text().catch(() => "");
        return { error: `HTTP ${res.status}${txt ? ": " + txt.slice(0, 120) : ""}` };
      }
      return normalizeCoverage(await res.json());
    } catch (err) {
      return { error: err.message || "network" };
    }
  }

  // ---------------------------------------------------------- UI wiring
  function el(id) {
    return document.getElementById(id);
  }

  function renderRatios(host) {
    host.innerHTML = "";
    RATIOS.forEach((r) => {
      const card = document.createElement("div");
      card.className = "ratio-card";
      const maxW = 72;
      const maxH = 40;
      let w = maxW;
      let h = maxW * (r.h / r.w);
      if (h > maxH) {
        h = maxH;
        w = maxH * (r.w / r.h);
      }
      card.innerHTML = `
        <div class="ratio-box" style="width:${w}px;height:${h}px;border-color:${r.color};color:${r.color}">
          ${r.w}:${r.h}
        </div>
        <div class="ratio-meta">
          <strong>${r.label}</strong>
          <span>${r.note}</span>
        </div>`;
      host.appendChild(card);
    });
  }

  function renderVenue(host) {
    host.innerHTML = `
      <div class="venue-grid">
        <div><span>直径</span><b>${VENUE.diameter_m} m</b></div>
        <div><span>倾角</span><b>${VENUE.tilt_deg}°（推断）</b></div>
        <div><span>投影</span><b>${VENUE.projectors_lower}+${VENUE.projectors_upper} · ${VENUE.projector_model}</b></div>
        <div><span>服务器</span><b>${VENUE.server}</b></div>
        <div><span>母版</span><b>${VENUE.master}</b></div>
        <div><span>立体</span><b>${VENUE.stereo ? "双目" : "单目 2D"}</b></div>
        <div><span>坐席</span><b>${VENUE.seats} 坐 / ${VENUE.standees} 站</b></div>
        <div><span>照度估算</span><b>≈${VENUE.lux_est} lux</b></div>
        <div><span>半球面积</span><b>${VENUE.surface_m2} ㎡</b></div>
      </div>
      <p class="hint">warp/blend 由场馆 HDX8 标定；我们只交整张 4096² 未变形母版。</p>`;
  }

  function setCoverageUi(stats) {
    const box = el("covReadout");
    if (!box) return;
    if (!stats || stats.error) {
      const why = stats && stats.error ? stats.error : "无数据";
      box.innerHTML = `<div class="cov-bad">无法测量（${why}）</div>`;
      return;
    }
    const cls = stats.level === "ok" ? "cov-ok" : stats.level === "warn" ? "cov-warn" : "cov-bad";
    box.innerHTML = `
      <div class="cov-main ${cls}">
        <div class="cov-deg">${stats.coverageDeg.toFixed(1)}°</div>
        <div class="cov-sub">实心天顶角 · r=${stats.coverageRadius.toFixed(2)}</div>
      </div>
      <div class="cov-rows">
        <div><span>含羽化外沿</span><b>${(stats.softRadius * 90).toFixed(1)}°</b></div>
        <div><span>外圈 0.80–0.98 有内容</span><b>${(stats.outerFill * 100).toFixed(1)}%</b></div>
        <div><span>覆盖立体角</span><b>${(stats.solidAngleFrac * 100).toFixed(1)}%</b></div>
      </div>
      <p class="hint">${stats.text}</p>`;
  }

  function setCoverageLoading(msg = "服务端测量中…") {
    const box = el("covReadout");
    if (box) box.innerHTML = `<div class="cov-main cov-warn"><div class="cov-deg">…</div><div class="cov-sub">${msg}</div></div>`;
  }

  // ---- draggable right panel width, persisted in localStorage (#416) ----
  function initRailResizer() {
    const rail = el("rightRail");
    const handle = el("railResizer");
    if (!rail || !handle) return;
    const KEY = "studio.railWidth";
    const applyPx = (px) => {
      rail.style.width = `${px}px`;
      rail.style.flex = "0 0 auto";
    };
    const clamp = (px) => {
      const min = 280;
      const maxVw = window.innerWidth * 0.7;
      return Math.max(min, Math.min(maxVw, px));
    };
    const saved = Number(localStorage.getItem(KEY));
    if (saved && saved >= 280) applyPx(clamp(saved));
    handle.addEventListener("mousedown", (e) => {
      e.preventDefault();
      const startX = e.clientX;
      const startW = rail.getBoundingClientRect().width;
      const move = (ev) => {
        // dragging the LEFT edge rightward (toward canvas) makes the panel
        // narrower; leftward (toward screen edge) makes it wider. Screen-space
        // width grows as the handle moves left.
        const w = clamp(startW - (ev.clientX - startX));
        applyPx(w);
      };
      const up = () => {
        window.removeEventListener("mousemove", move);
        window.removeEventListener("mouseup", up);
        localStorage.setItem(KEY, String(Math.round(rail.getBoundingClientRect().width)));
      };
      window.addEventListener("mousemove", move);
      window.addEventListener("mouseup", up);
    });
  }

  function bootStudioPanel() {
    initRailResizer();
    const canvas = el("domeCanvas");
    if (!canvas) return;
    let preview;
    try {
      preview = new DomePreview(canvas);
    } catch (err) {
      const host = el("covReadout");
      if (host) host.innerHTML = `<div class="cov-bad">${err.message}</div>`;
      return;
    }

    renderRatios(el("ratioList"));
    renderVenue(el("venueFacts"));

    // default demo coverage: historically bad master ~0.67 (60°)
    const demo = { coverageRadius: 0.668, coverageDeg: 60.1, softRadius: 0.72, outerFill: 0.088, solidAngleFrac: 0.502, level: "bad", text: "示例：历史坏母版（内容只到 ~60°）。加载你的分镜图/母版可替换。" };
    preview.setCoverage(demo.coverageRadius);
    setCoverageUi(demo);

    const file = el("domeFile");
    const applyStats = (stats, img) => {
      if (img) preview.setImage(img);
      if (stats && stats.coverageRadius != null) preview.setCoverage(stats.coverageRadius);
      setCoverageUi(stats);
    };

    if (file) {
      file.addEventListener("change", () => {
        const f = file.files && file.files[0];
        if (!f) return;
        const url = URL.createObjectURL(f);
        const img = new Image();
        img.onload = async () => {
          // Frontend only draws the circle; the reading is the server's
          // (issue #405).
          const stats = await analyzeOnServer(imageToCanvas(img));
          applyStats(stats, img);
          URL.revokeObjectURL(url);
        };
        img.onerror = () => {
          URL.revokeObjectURL(url);
          setCoverageUi({ error: "图片解码失败" });
        };
        img.src = url;
      });
    }

    // buttons for known presets: the image is still generated on the frontend,
    // but the reading is measured by the server (issue #405).
    const presetCoverage = async (fillR) => {
      preview.setImage(null);
      preview.hasTexture = false;
      preview.setCoverage(-1);
      setCoverageLoading();
      const canvas = makeDomeCanvas(fillR);
      const stats = await analyzeOnServer(canvas);
      preview.setImage(canvas);
      applyStats(stats, null);
    };
    el("btnCovGood") && el("btnCovGood").addEventListener("click", () => presetCoverage(0.98));
    el("btnCovBad") && el("btnCovBad").addEventListener("click", () => presetCoverage(0.6));

    // ---------------- orientation tools (yaw/pitch/roll + quick transforms)
    const sliderRow = (key, host, label, hint) => {
      if (!host) return;
      host.innerHTML = `
        <div class="sl-row">
          <span class="sl-lab">${label}</span>
          <input class="sl" id="sl-${key}" type="range" min="-180" max="180" step="1" value="0" />
          <input class="sl-num" id="num-${key}" type="number" min="-180" max="180" step="1" value="0" />
        </div>
        <p class="hint">${hint}</p>`;
      const sl = el(`sl-${key}`);
      const num = el(`num-${key}`);
      const push = (v) => {
        const c = Math.max(-180, Math.min(180, Math.round(Number(v) || 0)));
        sl.value = String(c);
        num.value = String(c);
        preview.setOrientation({ [key]: c });
      };
      sl.addEventListener("input", () => push(sl.value));
      num.addEventListener("change", () => push(num.value));
    };
    sliderRow("yaw", el("orientYawRow"), "yaw 旋转", "绕竖直轴转动母版（对应 --dome-yaw）");
    sliderRow("pitch", el("orientPitchRow"), "pitch 俯仰", "把内容从天顶压低（对应 --dome-pitch，正值向地平线）");
    sliderRow("roll", el("orientRollRow"), "roll 侧滚", "绕中心旋转母版（对应 --dome-roll）");

    const refreshOrientUi = () => {
      const o = preview.getOrientation();
      ["yaw", "pitch", "roll"].forEach((k) => {
        const sl = el(`sl-${k}`);
        const num = el(`num-${k}`);
        if (sl && document.activeElement !== sl) sl.value = String(o[k]);
        if (num && document.activeElement !== num) num.value = String(o[k]);
      });
      const cb = el("frontIsBottom");
      if (cb && cb.checked !== o.frontIsBottom) cb.checked = o.frontIsBottom;
      const code = el("orientExport");
      if (code) {
        const cli = o.cli || { dome_pitch: o.pitch, dome_yaw: o.yaw, dome_roll: o.roll };
        code.textContent =
          `--dome-pitch ${cli.dome_pitch}\n--dome-yaw ${cli.dome_yaw}\n--dome-roll ${cli.dome_roll}`;
      }
    };
    preview.onOrientChange = refreshOrientUi;

    const btnReset = el("btnOrientReset");
    btnReset &&
      btnReset.addEventListener("click", () => {
        preview.orient = { yaw: 0, pitch: 0, roll: 0 };
        preview.frontIsBottom = true;
        preview.uFlip = 1;
        preview.vFlip = 1;
        preview.setOrientation({});
      });
    const rot90 = (d) => () =>
      preview.setOrientation({ yaw: ((preview.orient.yaw + d + 540) % 360) - 180 });
    el("btnRot90cw") && el("btnRot90cw").addEventListener("click", rot90(90));
    el("btnRot90ccw") && el("btnRot90ccw").addEventListener("click", rot90(-90));
    const flip = (axis) => () =>
      preview.setOrientation({ [axis]: -preview[axis] });
    el("btnFlipH") && el("btnFlipH").addEventListener("click", flip("uFlip"));
    el("btnFlipV") && el("btnFlipV").addEventListener("click", flip("vFlip"));
    const cbFront = el("frontIsBottom");
    cbFront &&
      cbFront.addEventListener("change", () => preview.setOrientation({ frontIsBottom: cbFront.checked }));

    // export into the selected convert.dome node so the run honours the preview.
    // Reads the inspector DOM (app.js keeps state private) to find a
    // convert.dome node and its pitch/yaw/roll number fields, then sets them —
    // the inspector's own change handler writes the value back to node.params.
    function applyOrientationToNode(cli) {
      const insp = el("inspectorBody");
      if (!insp) return false;
      const typeEl = insp.querySelector(".node-type");
      if (!typeEl || typeEl.textContent !== "convert.dome") return false;
      const labels = Array.from(insp.querySelectorAll("label"));
      let n = 0;
      const setField = (suffix, value) => {
        const lab = labels.find((l) => (l.textContent || "").trim().toLowerCase().startsWith(suffix));
        if (!lab) return;
        const f = lab.querySelector("input[type=number]");
        if (!f) return;
        f.value = String(value);
        f.dispatchEvent(new Event("change", { bubbles: true }));
        n++;
      };
      setField("pitch", cli.dome_pitch);
      setField("yaw", cli.dome_yaw);
      setField("roll", cli.dome_roll);
      return n > 0;
    }

    const btnExport = el("btnOrientExport");
    btnExport &&
      btnExport.addEventListener("click", () => {
        const o = preview.getOrientation();
        const cli = o.cli;
        if (!cli) return;
        const applied = applyOrientationToNode(cli);
        const st = el("orientExportStatus");
        if (st) {
          st.textContent = applied
            ? "已写入选中的 convert.dome 节点 pitch/yaw/roll"
            : "未选中 convert.dome 节点（参数已复制到剪贴板）";
          st.className = `hint ${applied ? "ok" : ""}`;
        }
        try {
          navigator.clipboard &&
            navigator.clipboard.writeText(
              `--dome-pitch ${cli.dome_pitch} --dome-yaw ${cli.dome_yaw} --dome-roll ${cli.dome_roll}`,
            );
        } catch (_) {
          /* clipboard unavailable; the readout text stays */
        }
      });

    // ---------------- camera viewport (virtual camera + 2D perspective)
    const camCanvas = el("camCanvas");
    let cam2d = null;
    if (camCanvas) {
      try {
        cam2d = new CameraView(camCanvas, preview);
      } catch (err) {
        const host = el("camReadout");
        if (host) host.innerHTML = `<div class="cov-bad">${err.message}</div>`;
      }
    }
    const camSlider = (key, label, min, max, hint, deg) => {
      const host = el(`cam${key}Row`);
      if (!host) return;
      host.innerHTML = `
        <div class="sl-row">
          <span class="sl-lab">${label}</span>
          <input class="sl" id="cam-${key}" type="range" min="${min}" max="${max}" step="1" value="${deg != null ? deg : min}" />
          <input class="sl-num" id="camnum-${key}" type="number" min="${min}" max="${max}" step="1" value="${deg != null ? deg : min}" />
        </div>
        ${hint ? `<p class="hint">${hint}</p>` : ""}`;
      const sl = el(`cam-${key}`);
      const num = el(`camnum-${key}`);
      const push = (v) => {
        const n = Math.max(min, Math.min(max, Math.round(Number(v) || min)));
        if (sl && document.activeElement !== sl) sl.value = String(n);
        if (num && document.activeElement !== num) num.value = String(n);
        preview.setCamera({ [key]: n });
      };
      sl && sl.addEventListener("input", () => push(sl.value));
      num && num.addEventListener("change", () => push(num.value));
    };
    camSlider("yaw", "cam-yaw", -180, 180, "相机水平朝向", preview.camera.yaw);
    camSlider("pitch", "cam-pitch", -89, 90, "相机俯仰（+90 = 正对天顶）", preview.camera.pitch);
    camSlider("halfFov", "cam-halfFov", 15, 85, "相机半视场角", preview.camera.halfFov);

    const camReadout = el("camReadout");
    const refreshCam = () => {
      if (!camReadout) return;
      const DP = window.DomeProj;
      const ray = DP ? DP.cameraRay(0, 0, preview.camera.yaw, preview.camera.pitch, preview.camera.halfFov) : null;
      if (!ray) return;
      const elev = (Math.asin(Math.max(-1, Math.min(1, ray[1]))) * 180) / Math.PI;
      camReadout.innerHTML = `<p class="hint">视线仰角 <b>${elev.toFixed(0)}°</b>（90°=天顶，0°=地平线，&lt;0=地面）<br/>相机正对天顶时 2D 画面中心 = 母版圆心。</p>`;
    };
    preview.onCamChange = refreshCam;

    const camBtnReset = el("btnCamReset");
    camBtnReset &&
      camBtnReset.addEventListener("click", () => {
        preview.setCamera({ yaw: 0, pitch: 45, halfFov: 45 });
        ["yaw", "pitch", "halfFov"].forEach((k) => {
          const sl = el(`cam-${k}`);
          const num = el(`camnum-${k}`);
          if (sl) sl.value = String(preview.camera[k]);
          if (num) num.value = String(preview.camera[k]);
        });
        refreshCam();
      });
    refreshCam();

    // drag the camera in the 3D view: left = yaw, up = pitch
    const dragCam = { active: false };
    const camToggle = el("btnCamDrag");
    if (camToggle) {
      camToggle.addEventListener("click", () => {
        dragCam.active = !dragCam.active;
        preview.cameraShow = dragCam.active;
        camToggle.textContent = dragCam.active ? "● 机位拖拽开" : "机位拖拽";
        camToggle.className = `button ${dragCam.active ? "primary" : ""}`;
      });
    }
    const dragStart = (e) => {
      if (!dragCam.active) return;
      e.preventDefault();
      const sx = e.clientX, sy = e.clientY;
      const oYaw = preview.camera.yaw, oPitch = preview.camera.pitch;
      const move = (ev) => {
        preview.setCamera({
          yaw: Math.max(-180, Math.min(180, oYaw + (ev.clientX - sx) * 0.5)),
          pitch: Math.max(-89, Math.min(90, oPitch - (ev.clientY - sy) * 0.5)),
        });
        ["yaw", "pitch"].forEach((k) => {
          const sl = el(`cam-${k}`);
          const num = el(`camnum-${k}`);
          if (sl && document.activeElement !== sl) sl.value = String(preview.camera[k]);
          if (num && document.activeElement !== num) num.value = String(preview.camera[k]);
        });
        refreshCam();
      };
      const up = () => {
        window.removeEventListener("mousemove", move);
        window.removeEventListener("mouseup", up);
      };
      window.addEventListener("mousemove", move);
      window.addEventListener("mouseup", up);
    };
    el("domeCanvas") && el("domeCanvas").addEventListener("mousedown", dragStart);

    // accept coverage report from studio run via global event
    window.addEventListener("studio:coverage", (ev) => {
      const d = ev.detail || {};
      if (d.coverageRadius != null || d.coverage_deg != null) {
        const r = d.coverageRadius != null ? d.coverageRadius : d.coverage_deg / 90;
        const deg = d.coverageDeg != null ? d.coverageDeg : d.coverage_deg;
        preview.setCoverage(r);
        setCoverageUi({
          coverageRadius: r,
          coverageDeg: deg,
          softRadius: d.softRadius != null ? d.softRadius : r,
          outerFill: d.outerFill != null ? d.outerFill : d.outer_fill,
          solidAngleFrac: d.solidAngleFrac != null ? d.solidAngleFrac : d.solid_angle_frac,
          level: d.level || (deg >= 85 ? "ok" : deg >= 75 ? "warn" : "bad"),
          text: d.text || "来自最近一次 Studio 运行的覆盖度报告。",
        });
      }
    });

    window.StudioDomePreview = preview;
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", bootStudioPanel);
  } else {
    bootStudioPanel();
  }

  // expose constructors for reuse / headless verification
  window.DomePreview = DomePreview;
  window.CameraView = CameraView;
})();
