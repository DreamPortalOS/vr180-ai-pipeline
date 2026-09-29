/* dome_xr.js — Quest dome VR simulator (issue #434).
 *
 * A single dependency-free page (no CDN) that lets the owner sit inside a
 * virtual dome and inspect how a domemaster master (image or video) maps to
 * the surface: zenith position, front, outer-ring squish, edge black, stretch.
 *
 * Three render paths share one WebGL2 context + the same shader:
 *   1. WebXR immersive-vr (Quest 3) — the headset pose drives the view;
 *      controllers/hands drive play-pause / tilt / overlay / seat.
 *   2. 2D fallback (desktop / headless) — mouse-drag orbit from the seat, so
 *      the lead can do automated acceptance and preview without a headset.
 *   3. static — the page still serves and the picker works without any GL.
 *
 * The fragment shader is kept staging-identical to dome_xr_proj.js's dirToUV
 * (and thus to studio/dome_proj.py's dir_to_master_uv — the 对拍 test pins 5
 * directions).  The geometry frame and the vDir bridge match preview3d.js so
 * the dome the Quest shows is the dome the 2D Studio preview shows.
 */
(function () {
  "use strict";

  // ---------------------------------------------------------- small mat4 kit
  // Mirrored from preview3d.js so the dome is placed with the same convention.
  function mat4Identity() {
    return new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
  }
  function mat4Perspective(fovy, aspect, near, far) {
    var f = 1 / Math.tan(fovy / 2);
    var nf = 1 / (near - far);
    return new Float32Array([
      f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) * nf, -1, 0, 0,
      2 * far * near * nf, 0,
    ]);
  }
  function mat4LookAt(eye, center, up) {
    var zx = eye[0] - center[0], zy = eye[1] - center[1], zz = eye[2] - center[2];
    var len = Math.hypot(zx, zy, zz) || 1;
    var z = [zx / len, zy / len, zz / len];
    var xx = up[1] * z[2] - up[2] * z[1];
    var xy = up[2] * z[0] - up[0] * z[2];
    var xz = up[0] * z[1] - up[1] * z[0];
    len = Math.hypot(xx, xy, xz) || 1;
    var x = [xx / len, xy / len, xz / len];
    var y = [
      z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2],
      z[0] * x[1] - z[1] * x[0],
    ];
    return new Float32Array([
      x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0,
      -(x[0] * eye[0] + x[1] * eye[1] + x[2] * eye[2]),
      -(y[0] * eye[0] + y[1] * eye[1] + y[2] * eye[2]),
      -(z[0] * eye[0] + z[1] * eye[1] + z[2] * eye[2]), 1,
    ]);
  }
  function mat4Mul(a, b) {
    var o = new Float32Array(16);
    for (var c = 0; c < 4; c++) {
      for (var r = 0; r < 4; r++) {
        o[c * 4 + r] =
          a[0 * 4 + r] * b[c * 4 + 0] +
          a[1 * 4 + r] * b[c * 4 + 1] +
          a[2 * 4 + r] * b[c * 4 + 2] +
          a[3 * 4 + r] * b[c * 4 + 3];
      }
    }
    return o;
  }
  function mat4RotateX(rad) {
    var c = Math.cos(rad), s = Math.sin(rad);
    return new Float32Array([1, 0, 0, 0, 0, c, s, 0, 0, -s, c, 0, 0, 0, 0, 1]);
  }
  function mat4Translate(x, y, z) {
    return new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1]);
  }

  var HALF_PI = Math.PI / 2;
  var DEG = Math.PI / 180;

  // Geometry frame (matches preview3d.js dirAt): +y zenith, -z audience front.
  function dirAt(theta, phi) {
    var st = Math.sin(theta);
    return [st * Math.sin(phi), Math.cos(theta), -st * Math.cos(phi)];
  }
  function hemisphereMesh(rings, segments) {
    var pos = [], idx = [];
    for (var i = 0; i <= rings; i++) {
      var theta = (i / rings) * HALF_PI;
      for (var j = 0; j <= segments; j++) {
        var d = dirAt(theta, (j / segments) * Math.PI * 2);
        pos.push(d[0], d[1], d[2]);
      }
    }
    var stride = segments + 1;
    for (var i2 = 0; i2 < rings; i2++) {
      for (var j2 = 0; j2 < segments; j2++) {
        var a = i2 * stride + j2;
        var b = a + stride;
        idx.push(a, b, a + 1, a + 1, b, b + 1);
      }
    }
    return { positions: new Float32Array(pos), indices: new Uint16Array(idx) };
  }
  function latCircle(theta, segs) {
    var v = [];
    for (var j = 0; j < segs; j++) {
      var a = dirAt(theta, (j / segs) * Math.PI * 2);
      var b = dirAt(theta, ((j + 1) / segs) * Math.PI * 2);
      v.push(a[0], a[1], a[2], b[0], b[1], b[2]);
    }
    return new Float32Array(v);
  }
  function meridian(phi, steps) {
    var v = [];
    for (var i = 0; i < steps; i++) {
      var a = dirAt((i / steps) * HALF_PI, phi);
      var b = dirAt(((i + 1) / steps) * HALF_PI, phi);
      v.push(a[0], a[1], a[2], b[0], b[1], b[2]);
    }
    return new Float32Array(v);
  }
  // Floor disc at y=0, radius R, dark gray — the "below the horizon" ground.
  function floorDisc(segs) {
    var pos = [0, 0, 0], idx = [];
    for (var j = 0; j < segs; j++) {
      var a = (j / segs) * Math.PI * 2;
      pos.push(Math.cos(a), 0, Math.sin(a));
      if (j > 0) idx.push(0, j, j + 1);
    }
    idx.push(0, segs, 1);
    return { positions: new Float32Array(pos), indices: new Uint16Array(idx) };
  }
  // Seat ring: small quads around the back half at radius rr.
  function seatRing(count, rr, w) {
    var pos = [], idx = [];
    for (var k = 0; k < count; k++) {
      // back half only: phi in [pi/2, 3pi/2]
      var phi = Math.PI / 2 + (k / (count - 1)) * Math.PI;
      var cx = Math.cos(phi) * rr, cz = Math.sin(phi) * rr;
      var p = [cx - w, 0, cz - w, cx + w, 0, cz - w, cx + w, 0, cz + w, cx - w, 0, cz + w];
      var base = k * 4;
      for (var t = 0; t < 4; t++) pos.push(p[t * 3], p[t * 3 + 1], p[t * 3 + 2]);
      idx.push(base + 0, base + 1, base + 2, base + 0, base + 2, base + 3);
    }
    return { positions: new Float32Array(pos), indices: new Uint16Array(idx) };
  }
  // Comfortable-viewing band on the dome surface (elevation elLo..elHi),
  // triangulated so the shader can paint it translucent over the master.
  function comfortBand(elLo, elHi, segs) {
    var tLo = HALF_PI - elLo * DEG, tHi = HALF_PI - elHi * DEG;
    var pos = [], idx = [];
    for (var j = 0; j <= segs; j++) {
      var ph = (j / segs) * Math.PI * 2;
      var a = dirAt(tLo, ph), b = dirAt(tHi, ph);
      pos.push(a[0], a[1], a[2], b[0], b[1], b[2]);
    }
    for (var j2 = 0; j2 < segs; j2++) {
      var a0 = j2 * 2, a1 = a0 + 1, b0 = a0 + 2, b1 = b0 + 1;
      idx.push(a0, b0, a1, a1, b0, b1);
    }
    return { positions: new Float32Array(pos), indices: new Uint16Array(idx) };
  }

  // ------------------------------------------------------------- shaders
  // Dome surface: vDir is the dome-LOCAL direction (zenith +y, front -z_world).
  // The fragment bridges to the proj frame (front +z) the way preview3d.js does
  // and inlines dirToUV — staging-identical to dome_xr_proj.js.
  var DOME_VS = [
    "attribute vec3 aDir;",
    "uniform mat4 uMVP;",
    "uniform mat4 uModel;",
    "uniform float uRadius;",
    "varying vec3 vDir;",
    "void main(){ vDir=aDir; gl_Position=uMVP*uModel*vec4(aDir*uRadius,1.0); }",
  ].join("\n");
  var DOME_FS = [
    "precision highp float;",
    "varying vec3 vDir;",
    "uniform sampler2D uTex;",
    "uniform float uHasTex;",
    "void main(){",
    "  vec3 d=normalize(vec3(vDir.x, vDir.y, -vDir.z));", // geometry->proj (front +z)
    "  float theta=acos(clamp(d.y,-1.0,1.0));",
    "  float r=clamp(theta/1.5707963267,0.0,1.0);",
    "  float h=length(d.xz);",
    "  vec2 uv=vec2(0.5,0.5);",
    "  if(h>1e-5){ float sx=d.x/h, cz=d.z/h;",
    "    uv=vec2(0.5+0.5*r*sx, 0.5+0.5*r*cz); }", // dirToUV: front(+z) -> v=1 bottom
    "  vec3 col;",
    "  if(uHasTex>0.5){ col=texture2D(uTex,uv).rgb; }",
    "  else {",
    "    float band=mod(floor(r*6.0)+floor((atan(d.x,d.z)+3.14159265)/0.7853981634),2.0);",
    "    col=mix(vec3(0.10,0.12,0.16), vec3(0.16,0.19,0.25), band);",
    "  }",
    "  gl_FragColor=vec4(col,1.0);",
    "}",
  ].join("\n");
  var LINE_VS = [
    "attribute vec3 aPos;",
    "uniform mat4 uMVP;",
    "uniform mat4 uModel;",
    "uniform float uRadius;",
    "void main(){ gl_Position=uMVP*uModel*vec4(aPos*uRadius,1.0); }",
  ].join("\n");
  var LINE_FS = ["precision highp float;", "uniform vec4 uColor;", "void main(){ gl_FragColor=uColor; }"].join("\n");
  var FLOOR_FS = [
    "precision highp float;",
    "uniform vec4 uColor;",
    "varying float vDist;",
    "void main(){ float a=clamp(1.0-vDist,0.0,1.0); gl_FragColor=vec4(uColor.rgb, uColor.a*a); }",
  ].join("\n");
  // floor needs a distance varying; use a second vs for it.
  var FLOOR_VS = [
    "attribute vec3 aPos;",
    "uniform mat4 uMVP;",
    "varying float vDist;",
    "void main(){ vDist=clamp(length(aPos.xz)*0.5,0.0,1.0); gl_Position=uMVP*vec4(aPos,1.0); }",
  ].join("\n");

  function compile(gl, type, src) {
    var sh = gl.createShader(type);
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
      var log = gl.getShaderInfoLog(sh) || "(no log)";
      gl.deleteShader(sh);
      throw new Error("shader compile: " + log);
    }
    return sh;
  }
  function program(gl, vs, fs) {
    var p = gl.createProgram();
    gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, vs));
    gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
      throw new Error("program link: " + (gl.getProgramInfoLog(p) || "(no log)"));
    }
    return p;
  }

  // ------------------------------------------------------------- state
  var state = {
    source: null, // {path,kind,url,name} the picked master
    tilt: 30, // degrees (dome physical tilt)
    seat: "center", // center | rear | front
    radius: 8, // metres
    overlay: true, // elevation/azimuth overlay
    playing: false,
    // GL
    gl: null,
    domeProg: null, lineProg: null, floorProg: null,
    domeMesh: null, overlayMeshes: null, floorMesh: null, seatMesh: null, bandMesh: null,
    tex: null, texKind: null, video: null,
    // XR
    xrSession: null, xrSpace: null, xrGL: null, rafHandle: null,
    // 2D fallback
    mode2d: false, yaw: 0, pitch: 30, // orbit look (degrees)
    dragging: false, lastX: 0, lastY: 0,
    // input edges
    prevBtn: {}, prevAxes: {},
  };

  function $(id) {
    return document.getElementById(id);
  }

  // ------------------------------------------------------------- sources
  async function loadSources() {
    var srcParam = new URLSearchParams(location.search).get("src");
    try {
      var res = await fetch("/api/xr/sources" + (srcParam ? "?src=" + encodeURIComponent(srcParam) : ""));
      var data = await res.json(); // read the body ONCE
      renderSourceList(data.sources || []);
      if (data.sources && data.sources.length === 1 && srcParam) pickSource(data.sources[0]);
    } catch (err) {
      setStatus("母版列表加载失败: " + err.message, true);
    }
  }

  function renderSourceList(list) {
    var host = $("xrSourceList");
    host.innerHTML = "";
    list.forEach(function (entry) {
      var li = document.createElement("li");
      li.dataset.path = entry.path;
      if (state.source && state.source.path === entry.path) li.classList.add("selected");
      var thumb = document.createElement("img");
      thumb.className = "thumb";
      thumb.src = entry.kind === "image" ? entry.media_url : "";
      thumb.alt = "";
      var meta = document.createElement("div");
      meta.className = "meta";
      var name = document.createElement("div");
      name.className = "name";
      name.textContent = entry.name;
      var kind = document.createElement("div");
      kind.className = "kind";
      kind.textContent = entry.kind === "image" ? "图片" : "视频";
      meta.appendChild(name);
      meta.appendChild(kind);
      li.appendChild(thumb);
      li.appendChild(meta);
      li.addEventListener("click", function () {
        pickSource(entry);
      });
      host.appendChild(li);
    });
    if (!list.length) {
      var empty = document.createElement("li");
      empty.style.cursor = "default";
      empty.textContent = "work_root 内暂无母版。上传一个或用 ?src= 指定路径。";
      host.appendChild(empty);
    }
  }

  function pickSource(entry) {
    state.source = entry;
    var host = $("xrSourceList");
    var items = host ? host.children : [];
    for (var i = 0; i < items.length; i++) {
      if (items[i].dataset) {
        if (items[i].dataset.path === entry.path) items[i].classList.add("selected");
        else items[i].classList.remove("selected");
      }
    }
    $("btnEnterVR").disabled = false; // a source is picked — enable Enter VR
    setStatus("已选: " + entry.name + " — 进入 VR 或在 2D 模式预览");
    loadTexture();
  }

  async function uploadFile(file) {
    if (!file) return;
    setStatus("上传中…");
    var fd = new FormData();
    fd.append("file", file);
    try {
      var res = await fetch("/api/upload", { method: "POST", body: fd });
      var meta = await res.json(); // read the body ONCE only
      if (!res.ok || !meta || !meta.path) {
        throw new Error((meta && meta.detail) || "上传失败");
      }
      var entry = {
        path: meta.path,
        name: file.name,
        kind: meta.kind || (file.type.indexOf("video") === 0 ? "video" : "image"),
        media_url: "/api/media?path=" + encodeURIComponent(meta.path),
      };
      await loadSources();
      pickSource(entry);
    } catch (err) {
      setStatus("上传失败: " + err.message, true);
    }
  }

  // ------------------------------------------------------------- textures
  function loadTexture() {
    var gl = state.gl;
    if (!gl || !state.source) return;
    if (state.source.kind === "image") {
      var img = new Image();
      img.onload = function () {
        gl.bindTexture(gl.TEXTURE_2D, state.tex);
        gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false); // matches preview3d.js
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        state.texKind = "image";
      };
      img.onerror = function () {
        setStatus("母版图片加载失败", true);
      };
      img.src = state.source.media_url;
    } else {
      // video texture — update each frame in draw()
      if (!state.video) {
        var v = document.createElement("video");
        v.crossOrigin = "anonymous";
        v.loop = true;
        v.preload = "auto";
        v.playsInline = true;
        state.video = v;
      }
      state.video.src = state.source.media_url;
      state.video.muted = true; // autoplay policies: muted plays inline
      state.video.play().then(
        function () {
          state.playing = true;
        },
        function () {
          state.playing = false;
        }
      );
      state.texKind = "video";
    }
  }

  function syncVideoTexture(gl) {
    if (state.texKind !== "video" || !state.video) return;
    if (state.video.readyState >= 2) {
      gl.bindTexture(gl.TEXTURE_2D, state.tex);
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, state.video);
    }
  }

  // ------------------------------------------------------------- renderer setup
  function initGL(canvas) {
    var gl = canvas.getContext("webgl2", { xrCompatible: true, antialias: true });
    if (!gl) {
      // fall back to webgl1 for the 2D preview (no WebXR without webgl2)
      gl = canvas.getContext("webgl", { antialias: true });
    }
    if (!gl) {
      // no WebGL at all (a stripped headless runner): degrade to the picker
      // only instead of throwing and blanking the whole page.
      setStatus("此浏览器无 WebGL — 仅显示母版列表。", true);
      return null;
    }
    state.gl = gl;
    state.domeProg = program(gl, DOME_VS, DOME_FS);
    state.lineProg = program(gl, LINE_VS, LINE_FS);
    state.floorProg = program(gl, FLOOR_VS, FLOOR_FS);
    state.domeMesh = hemisphereMesh(24, 48);
    state.floorMesh = floorDisc(48);
    state.seatMesh = seatRing(9, 0.78, 0.04);
    state.bandMesh = comfortBand(20, 60, 48);
    state.overlayMeshes = buildOverlay();
    state.tex = gl.createTexture();
    // attribute/uniform locations cached on use to keep this small.
    return gl;
  }

  function buildOverlay() {
    // Elevation circles 15/30/45/60/75 + azimuth lines every 30° + front red.
    var parts = [];
    var elevations = [15, 30, 45, 60, 75];
    for (var i = 0; i < elevations.length; i++) {
      parts.push({ verts: latCircle(HALF_PI - elevations[i] * DEG, 96), color: [0.3, 0.34, 0.42, 0.9] });
    }
    for (var a = 0; a < 360; a += 30) {
      parts.push({ verts: meridian(a * DEG, 24), color: [0.3, 0.34, 0.42, 0.9] });
    }
    // front meridian (azimuth 0 = +z proj = -z geom) in red
    parts.push({ verts: meridian(0 * DEG, 24), color: [1.0, 0.35, 0.3, 1.0] });
    return parts;
  }

  // ------------------------------------------------------------- drawing
  function seatOffset() {
    var r = state.radius;
    if (state.seat === "rear") return [0, 1.5, -r / 3];
    if (state.seat === "front") return [0, 1.5, r / 3];
    return [0, 1.5, 0];
  }
  function tiltModel() {
    // +tilt leans the dome zenith toward -z (back) — a tilted cinema.
    return mat4RotateX(-state.tilt * DEG);
  }

  // 2D path: set the full-canvas viewport, clear once, then draw the shared
  // scene body.  Kept as a thin wrapper over drawSceneBody so the 2D preview
  // and the VR scene can never diverge on geometry.
  function drawScene(gl, view, proj) {
    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
    gl.clearColor(0.04, 0.05, 0.08, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST);
    gl.disable(gl.CULL_FACE);
    drawSceneBody(gl, mat4Mul(proj, view));
  }

  function buf(gl, data) {
    var b = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, b);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
    return b;
  }
  function idxbuf(gl, data) {
    var b = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, b);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, data, gl.STATIC_DRAW);
    return b;
  }
  function scaledPositions(arr, s) {
    var out = new Float32Array(arr.length);
    for (var i = 0; i < arr.length; i++) out[i] = arr[i] * s;
    return out;
  }

  // ------------------------------------------------------------- 2D fallback
  function start2D() {
    state.mode2d = true;
    var canvas = $("xrCanvas");
    initGL(canvas);
    if (!state.gl) return; // picker-only degrade
    resize2D();
    window.addEventListener("resize", resize2D);
    bindMouse(canvas);
    setHint("2D 环视模式：拖拽环顾。B 切换叠加层；空格 播放/暂停；↑↓ 倾角 ±5°。");
    loop2D();
  }
  function resize2D() {
    var canvas = $("xrCanvas");
    var dpr = window.devicePixelRatio || 1;
    canvas.width = Math.floor((canvas.clientWidth || 800) * dpr);
    canvas.height = Math.floor((canvas.clientHeight || 600) * dpr);
  }
  function loop2D() {
    if (state.xrSession) return; // VR owns the loop once active
    draw2D();
    requestAnimationFrame(loop2D);
  }
  function draw2D() {
    var gl = state.gl;
    if (!gl) return;
    var canvas = gl.canvas;
    var aspect = (canvas.width || 1) / Math.max(1, canvas.height || 1);
    var proj = mat4Perspective((70 * Math.PI) / 180, aspect, 0.05, 100);
    var eye = seatOffset();
    // orbit the look direction by yaw/pitch around the seat
    var yaw = state.yaw * DEG, pitch = state.pitch * DEG;
    var dir = [
      Math.cos(pitch) * Math.sin(yaw),
      Math.sin(pitch),
      -Math.cos(pitch) * Math.cos(yaw), // -z is front in geom frame
    ];
    var center = [eye[0] + dir[0], eye[1] + dir[1], eye[2] + dir[2]];
    var view = mat4LookAt(eye, center, [0, 1, 0]);
    drawScene(gl, view, proj);
  }
  function bindMouse(canvas) {
    canvas.addEventListener("mousedown", function (e) {
      state.dragging = true;
      state.lastX = e.clientX;
      state.lastY = e.clientY;
    });
    window.addEventListener("mouseup", function () {
      state.dragging = false;
    });
    window.addEventListener("mousemove", function (e) {
      if (!state.dragging) return;
      var dx = e.clientX - state.lastX;
      var dy = e.clientY - state.lastY;
      state.lastX = e.clientX;
      state.lastY = e.clientY;
      state.yaw += dx * 0.25;
      state.pitch = Math.max(-80, Math.min(90, state.pitch + dy * 0.25));
    });
    // touch
    canvas.addEventListener("touchstart", function (e) {
      if (!e.touches.length) return;
      state.dragging = true;
      state.lastX = e.touches[0].clientX;
      state.lastY = e.touches[0].clientY;
    }, { passive: true });
    canvas.addEventListener("touchmove", function (e) {
      if (!state.dragging || !e.touches.length) return;
      var dx = e.touches[0].clientX - state.lastX;
      var dy = e.touches[0].clientY - state.lastY;
      state.lastX = e.touches[0].clientX;
      state.lastY = e.touches[0].clientY;
      state.yaw += dx * 0.25;
      state.pitch = Math.max(-80, Math.min(90, state.pitch + dy * 0.25));
    }, { passive: true });
    canvas.addEventListener("touchend", function () {
      state.dragging = false;
    });
  }

  // ------------------------------------------------------------- WebXR
  function xrSupported() {
    return !!navigator.xr && !!navigator.xr.isSessionSupported;
  }
  async function probeXR() {
    if (!xrSupported()) return false;
    try {
      return await navigator.xr.isSessionSupported("immersive-vr");
    } catch (e) {
      return false;
    }
  }
  async function enterVR() {
    if (!state.source) {
      setStatus("先选择一个母版", true);
      return;
    }
    if (!navigator.xr) {
      setStatus("此浏览器无 WebXR — 用 2D 环视模式", true);
      return;
    }
    try {
      var session = await navigator.xr.requestSession("immersive-vr", {
        requiredFeatures: ["local"],
      });
      state.xrSession = session;
      var gl = state.gl;
      await gl.makeXRCompatible();
      state.xrGL = new XRWebGLLayer(session, gl);
      await session.updateRenderState({ baseLayer: state.xrGL });
      state.xrSpace = await session.requestReferenceSpace("local");
      session.addEventListener("end", onXRSessionEnd);
      session.addEventListener("inputsourceschange", onInputSourcesChange);
      // hand pinch / bare-source select → play-pause (controllers are polled)
      session.addEventListener("selectstart", onSelectStart);
      $("btnEnterVR").textContent = "退出 VR";
      setHint("扳机/A 播放暂停 · 摇杆左右 ±5s · 上下 倾角 ±5° · B 叠加层 · 握把 切座位");
      state.rafHandle = session.requestAnimationFrame(onXRFrame);
    } catch (err) {
      setStatus("进入 VR 失败: " + err.message, true);
      state.xrSession = null;
    }
  }
  function endVR() {
    if (state.xrSession) {
      var s = state.xrSession;
      state.xrSession = null;
      s.end();
    }
  }
  function onXRSessionEnd() {
    // session is already ending; just reset local state.
    state.xrSession = null;
    state.xrSpace = null;
    $("btnEnterVR").textContent = "进入 VR";
    setHint("2D 环视模式：拖拽环顾。B 切换叠加层。");
    loop2D(); // resume the fallback loop
  }

  function onXRFrame(t, frame) {
    var session = state.xrSession;
    if (!session) return;
    var pose = frame.getViewerPose(state.xrSpace);
    var gl = state.gl;
    if (!pose || !pose.views || !pose.views.length) {
      state.rafHandle = session.requestAnimationFrame(onXRFrame);
      return;
    }
    handleInputs(frame);
    var layer = state.xrGL;
    gl.bindFramebuffer(gl.FRAMEBUFFER, layer.framebuffer);
    // Clear the colour buffer once; per-eye passes clear only depth.
    gl.clearColor(0.04, 0.05, 0.08, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST);
    gl.disable(gl.CULL_FACE);
    for (var i = 0; i < pose.views.length; i++) {
      var view = pose.views[i];
      var vp = layer.getViewport(view);
      if (vp) gl.viewport(vp.x, vp.y, vp.width, vp.height);
      gl.clear(gl.DEPTH_BUFFER_BIT);
      drawSceneBody(gl, mat4Mul(view.projectionMatrix, xrView(view)));
    }
    state.rafHandle = session.requestAnimationFrame(onXRFrame);
  }

  // View matrix for one eye: the seat offset composed with the headset pose.
  // XRView.transform.matrix is the reference-space pose (column-major); the
  // seat is a translation of the reference space origin, so the eye is
  // seat + headsetTransform.  Applying the seat as a translate to the pose
  // (T(seat) · pose) keeps the headset's local motion intact.
  function xrView(view) {
    var m = view.transform.matrix;
    var t = new Float32Array(16);
    for (var i = 0; i < 16; i++) t[i] = m[i];
    var eye = seatOffset();
    t[12] += eye[0];
    t[13] += eye[1];
    t[14] += eye[2];
    return t;
  }

  // The scene body without the per-frame clear/viewport setup — shared by 2D
  // and XR.  Kept as one function so the two paths cannot diverge on geometry.
  function drawSceneBody(gl, mvp) {
    var model = tiltModel();
    var radius = Math.max(1, state.radius);
    // floor
    gl.useProgram(state.floorProg);
    gl.uniformMatrix4fv(gl.getUniformLocation(state.floorProg, "uMVP"), false, mvp);
    gl.uniform4f(gl.getUniformLocation(state.floorProg, "uColor"), 0.1, 0.11, 0.14, 1.0);
    var fp = scaledPositions(state.floorMesh.positions, 1.5 * radius);
    gl.bindBuffer(gl.ARRAY_BUFFER, state._floorBuf2 || (state._floorBuf2 = buf(gl, fp)));
    gl.bufferData(gl.ARRAY_BUFFER, fp, gl.STATIC_DRAW);
    var aPosFloor = gl.getAttribLocation(state.floorProg, "aPos");
    gl.enableVertexAttribArray(aPosFloor);
    gl.vertexAttribPointer(aPosFloor, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, state._floorIdx || (state._floorIdx = idxbuf(gl, state.floorMesh.indices)));
    gl.drawElements(gl.TRIANGLES, state.floorMesh.indices.length, gl.UNSIGNED_SHORT, 0);
    // dome
    if (state.texKind === "video") syncVideoTexture(gl);
    gl.useProgram(state.domeProg);
    gl.uniformMatrix4fv(gl.getUniformLocation(state.domeProg, "uMVP"), false, mvp);
    gl.uniformMatrix4fv(gl.getUniformLocation(state.domeProg, "uModel"), false, model);
    gl.uniform1f(gl.getUniformLocation(state.domeProg, "uRadius"), radius);
    gl.uniform1f(gl.getUniformLocation(state.domeProg, "uHasTex"), state.texKind ? 1 : 0);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, state.tex);
    gl.uniform1i(gl.getUniformLocation(state.domeProg, "uTex"), 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, state._domeBuf || (state._domeBuf = buf(gl, state.domeMesh.positions)));
    var aDir = gl.getAttribLocation(state.domeProg, "aDir");
    gl.enableVertexAttribArray(aDir);
    gl.vertexAttribPointer(aDir, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, state._domeIdx || (state._domeIdx = idxbuf(gl, state.domeMesh.indices)));
    gl.drawElements(gl.TRIANGLES, state.domeMesh.indices.length, gl.UNSIGNED_SHORT, 0);
    // overlay
    if (state.overlay) drawOverlay(gl, mvp, model, radius);
    // seats
    gl.useProgram(state.lineProg);
    gl.uniformMatrix4fv(gl.getUniformLocation(state.lineProg, "uMVP"), false, mvp);
    gl.uniformMatrix4fv(gl.getUniformLocation(state.lineProg, "uModel"), false, mat4Identity());
    gl.uniform1f(gl.getUniformLocation(state.lineProg, "uRadius"), radius);
    gl.uniform4f(gl.getUniformLocation(state.lineProg, "uColor"), 0.16, 0.18, 0.22, 1.0);
    gl.bindBuffer(gl.ARRAY_BUFFER, state._seatBuf || (state._seatBuf = buf(gl, scaledPositions(state.seatMesh.positions, 1))));
    var aPosLine = gl.getAttribLocation(state.lineProg, "aPos");
    gl.enableVertexAttribArray(aPosLine);
    gl.vertexAttribPointer(aPosLine, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, state._seatIdx || (state._seatIdx = idxbuf(gl, state.seatMesh.indices)));
    gl.drawElements(gl.TRIANGLES, state.seatMesh.indices.length, gl.UNSIGNED_SHORT, 0);
  }

  function drawOverlay(gl, mvp, model, radius) {
    gl.useProgram(state.lineProg);
    gl.uniformMatrix4fv(gl.getUniformLocation(state.lineProg, "uMVP"), false, mvp);
    gl.uniformMatrix4fv(gl.getUniformLocation(state.lineProg, "uModel"), false, model);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    // comfort band
    gl.uniform4f(gl.getUniformLocation(state.lineProg, "uColor"), 0.2, 0.7, 0.45, 0.18);
    gl.uniform1f(gl.getUniformLocation(state.lineProg, "uRadius"), radius * 1.002);
    gl.bindBuffer(gl.ARRAY_BUFFER, state._bandBuf || (state._bandBuf = buf(gl, state.bandMesh.positions)));
    var aPos = gl.getAttribLocation(state.lineProg, "aPos");
    gl.enableVertexAttribArray(aPos);
    gl.vertexAttribPointer(aPos, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, state._bandIdx || (state._bandIdx = idxbuf(gl, state.bandMesh.indices)));
    gl.drawElements(gl.TRIANGLES, state.bandMesh.indices.length, gl.UNSIGNED_SHORT, 0);
    gl.uniform1f(gl.getUniformLocation(state.lineProg, "uRadius"), radius * 1.004);
    state._ovBuf = state._ovBuf || [];
    for (var i = 0; i < state.overlayMeshes.length; i++) {
      var m = state.overlayMeshes[i];
      gl.uniform4fv(gl.getUniformLocation(state.lineProg, "uColor"), new Float32Array(m.color));
      if (!state._ovBuf[i]) state._ovBuf[i] = buf(gl, m.verts);
      gl.bindBuffer(gl.ARRAY_BUFFER, state._ovBuf[i]);
      gl.enableVertexAttribArray(aPos);
      gl.vertexAttribPointer(aPos, 3, gl.FLOAT, false, 0, 0);
      gl.drawArrays(gl.LINES, 0, m.verts.length / 3);
    }
    gl.disable(gl.BLEND);
  }

  // ------------------------------------------------------------- inputs
  // Controllers with a gamepad are polled per frame (trigger/A = play-pause,
  // stick L/R = ±5 s, stick U/D = tilt ±5°, B = overlay, grip = seat).  Hands
  // and bare input sources have no gamepad, so the session-level `selectstart`
  // event (a pinch / tap) drives play-pause instead — that is the documented
  // WebXR hand-tracking gesture, not a per-source gamepad field.
  function handleInputs(frame) {
    var session = state.xrSession;
    if (!session) return;
    var sources = session.inputSources || [];
    for (var i = 0; i < sources.length; i++) {
      var gp = sources[i].gamepad;
      if (!gp) continue; // hands → handled by sessions' selectstart
      var btns = gp.buttons || [];
      // buttons[0] = trigger/A, buttons[1] = squeeze/grip, buttons[5] = B/Y
      var trigger = btns[0] && btns[0].pressed;
      var grip = btns[1] && btns[1].pressed;
      var bBtn = btns[5] && btns[5].pressed;
      if (trigger && !state.prevBtn["t" + i]) togglePlay();
      if (grip && !state.prevBtn["g" + i]) cycleSeat();
      if (bBtn && !state.prevBtn["b" + i]) state.overlay = !state.overlay;
      state.prevBtn["t" + i] = !!trigger;
      state.prevBtn["g" + i] = !!grip;
      state.prevBtn["b" + i] = !!bBtn;
      // thumbstick axes: [x,y] ±1; edge-trigger on |axis|>0.7
      var axes = gp.axes || [];
      var ax = axes[0] || 0, ay = axes[1] || 0;
      if (Math.abs(ax) > 0.7 && !state.prevAxes["x" + i]) {
        seekVideo(ax > 0 ? 5 : -5);
      }
      if (Math.abs(ay) > 0.7 && !state.prevAxes["y" + i]) {
        state.tilt = Math.max(-45, Math.min(45, state.tilt + (ay < 0 ? 5 : -5)));
        applyTiltUI();
      }
      state.prevAxes["x" + i] = Math.abs(ax) > 0.7;
      state.prevAxes["y" + i] = Math.abs(ay) > 0.7;
    }
  }
  function onInputSourcesChange() {
    // presence change only; per-frame polling happens in onXRFrame
  }
  // A pinch / tap on a hand input (or a controller without a gamepad).
  function onSelectStart() {
    togglePlay();
  }
  function togglePlay() {
    if (!state.video) return;
    if (state.playing) {
      state.video.pause();
      state.playing = false;
    } else {
      state.video.play();
      state.playing = true;
    }
  }
  function seekVideo(delta) {
    if (!state.video) return;
    try {
      state.video.currentTime = Math.max(0, Math.min((state.video.duration || 0), state.video.currentTime + delta));
    } catch (e) {
      // seeking can throw before metadata is ready — ignore
    }
  }
  function cycleSeat() {
    state.seat = state.seat === "center" ? "rear" : state.seat === "rear" ? "front" : "center";
    $("xrSeat").value = state.seat;
  }
  function applyTiltUI() {
    var sel = $("xrTilt");
    if (!sel) return;
    var match = ["0", "15", "30"].indexOf(String(state.tilt));
    if (match >= 0) {
      sel.value = String(state.tilt);
      $("xrTiltCustomWrap").hidden = true;
    } else {
      sel.value = "custom";
      $("xrTiltCustomWrap").hidden = false;
      $("xrTiltCustom").value = state.tilt;
    }
  }

  // ------------------------------------------------------------- keyboard
  function onKeyDown(e) {
    // never steal keys while typing in a field
    var tag = (e.target && e.target.tagName) || "";
    if (tag === "INPUT" || tag === "TEXTAREA" || (e.target && e.target.isContentEditable)) return;
    var key = e.key.toLowerCase();
    if (key === "b") {
      state.overlay = !state.overlay;
    } else if (key === " ") {
      e.preventDefault();
      togglePlay();
    } else if (key === "arrowleft") {
      seekVideo(-5);
    } else if (key === "arrowright") {
      seekVideo(5);
    } else if (key === "arrowup") {
      state.tilt = Math.min(45, state.tilt + 5);
      applyTiltUI();
    } else if (key === "arrowdown") {
      state.tilt = Math.max(-45, state.tilt - 5);
      applyTiltUI();
    }
  }

  // ------------------------------------------------------------- status
  function setStatus(msg, isErr) {
    var el = $("xrStatus");
    if (!el) return;
    el.textContent = msg;
    el.className = isErr ? "err" : "";
  }
  function setHint(msg) {
    var el = $("xrHint");
    if (el) el.textContent = msg;
  }

  // ------------------------------------------------------------- init
  function bindUI() {
    var tilt = $("xrTilt");
    var tiltC = $("xrTiltCustom");
    var tiltWrap = $("xrTiltCustomWrap");
    tilt.addEventListener("change", function () {
      if (tilt.value === "custom") {
        tiltWrap.hidden = false;
        state.tilt = parseFloat(tiltC.value) || 0;
      } else {
        tiltWrap.hidden = true;
        state.tilt = parseFloat(tilt.value) || 0;
      }
    });
    tiltC.addEventListener("change", function () {
      state.tilt = parseFloat(tiltC.value) || 0;
    });
    $("xrSeat").addEventListener("change", function (e) {
      state.seat = e.target.value;
    });
    $("xrRadius").addEventListener("change", function (e) {
      state.radius = Math.max(3, parseFloat(e.target.value) || 8);
    });
    $("xrFile").addEventListener("change", function (e) {
      if (e.target.files && e.target.files[0]) uploadFile(e.target.files[0]);
    });
    $("btnEnterVR").addEventListener("click", function () {
      if (state.xrSession) endVR();
      else enterVR();
    });
    window.addEventListener("keydown", onKeyDown);
  }

  async function init() {
    // guard element existence so the page degrades gracefully if served
    // partially (a missing id used to throw and blank the whole page).
    if (!$("xrCanvas") || !$("btnEnterVR")) return;
    state._ovBuf = [];
    bindUI();
    await loadSources();
    // start 2D fallback immediately; VR is layered on top when entered.
    start2D();
    var ok = await probeXR();
    if (ok) {
      $("btnEnterVR").disabled = !state.source;
      setHint("支持 VR：选择母版后点「进入 VR」。当前为 2D 环视预览。");
    } else {
      setHint("此浏览器无 WebXR — 2D 环视模式（Quest 上请用 adb reverse 打开本页进入 VR）。");
    }
    var srcParam = new URLSearchParams(location.search).get("src");
    if (srcParam) setHint("src=" + srcParam + " — 选择母版后进入 VR。");
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }

  // exposed for headless acceptance / source-level guards (no globals leaked
  // besides this handle).
  window.DomeXR = { state: state, init: init, pickSource: pickSource, loadSources: loadSources };
})();
