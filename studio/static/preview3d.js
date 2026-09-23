/* Studio 3D dome preview — offline WebGL, no CDN.
 * Geometry convention matches web/dome-preview:
 *   +Y zenith, -Z front (audience), y=0 springline, r = zenith/90°.
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
      this._initPrograms();
      this._initBuffers();
      this._bindEvents();
      this._loop();
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
          "void main(){",
          "  vec3 d=normalize(vDir);",
          "  float theta=acos(clamp(d.y,-1.0,1.0));",
          "  float r=clamp(theta/1.5707963267,0.0,1.0);",
          "  float h=length(d.xz);",
          "  vec2 uv=vec2(0.5,0.5);",
          "  float phi=0.0;",
          "  if(h>1e-5){ uv=vec2(0.5+0.5*r*d.x/h, 0.5+0.5*r*(-d.z)/h); phi=atan(d.x,-d.z); }",
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
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.tex);
      gl.uniform1i(this.uDome.tex, 0);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.domePos);
      gl.enableVertexAttribArray(this.uDome.aDir);
      gl.vertexAttribPointer(this.uDome.aDir, 3, gl.FLOAT, false, 0, 0);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.domeIdx);
      gl.drawElements(gl.TRIANGLES, this.domeCount, gl.UNSIGNED_SHORT, 0);

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

  /** POST a canvas image to the server and return its coverage stats.
   * Resolves to {error} on any failure so the UI can degrade gracefully. */
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
      return await res.json();
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

  function bootStudioPanel() {
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
})();
