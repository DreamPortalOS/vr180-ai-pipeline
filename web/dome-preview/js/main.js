/* main.js — the dome coverage previewer application. */
(function () {
  'use strict';

  var mat4 = GLKit.mat4;
  var G = DomeGeometry;

  // ------------------------------------------------------------------ shaders
  var DOME_VS = [
    'attribute vec3 aDir;',
    'uniform mat4 uProj;',
    'uniform mat4 uView;',
    'uniform mat4 uModel;',
    'uniform float uRadius;',
    'varying vec3 vDir;',
    'void main() {',
    '  vDir = aDir;',
    '  gl_Position = uProj * uView * uModel * vec4(aDir * uRadius, 1.0);',
    '}'
  ].join('\n');

  var DOME_FS = [
    'precision highp float;',
    'varying vec3 vDir;',
    'uniform sampler2D uTex;',
    'uniform float uHasTex;',
    'uniform float uCovR;',   // normalised coverage radius, <0 = unknown
    'uniform float uMark;',
    'void main() {',
    '  vec3 d = normalize(vDir);',
    '  float theta = acos(clamp(d.y, -1.0, 1.0));',
    '  float r = clamp(theta / 1.5707963267, 0.0, 1.0);',
    '  float h = length(d.xz);',
    '  vec2 uv = vec2(0.5, 0.5);',
    '  float phi = 0.0;',
    '  if (h > 1e-5) {',
    '    uv = vec2(0.5 + 0.5 * r * d.x / h, 0.5 + 0.5 * r * (-d.z) / h);',
    '    phi = atan(d.x, -d.z);',
    '  }',
    '  vec3 col;',
    '  if (uHasTex > 0.5) {',
    '    col = texture2D(uTex, uv).rgb;',
    '  } else {',
    '    float band = mod(floor(r * 6.0) + floor((phi + 3.14159265) / 0.5235988), 2.0);',
    '    col = mix(vec3(0.10, 0.12, 0.16), vec3(0.17, 0.20, 0.26), band);',
    '    if (abs(phi) < 0.13 && r > 0.80) col = vec3(0.38, 0.17, 0.17);',
    '  }',
    '  if (uMark > 0.5 && uCovR >= 0.0 && r > uCovR + 0.002) {',
    '    float stripe = step(0.5, fract((r - uCovR) * 24.0));',
    '    col = mix(col, vec3(1.0, 0.40, 0.26), 0.20 + 0.16 * stripe);',
    '  }',
    '  if (uMark > 0.5 && uCovR >= 0.0 && abs(r - uCovR) < 0.005) {',
    '    col = vec3(1.0, 0.76, 0.26);',
    '  }',
    '  gl_FragColor = vec4(col, 1.0);',
    '}'
  ].join('\n');

  var LINE_VS = [
    'attribute vec3 aPos;',
    'uniform mat4 uProj;',
    'uniform mat4 uView;',
    'uniform mat4 uModel;',
    'uniform float uRadius;',
    'void main() {',
    '  gl_Position = uProj * uView * uModel * vec4(aPos * uRadius, 1.0);',
    '}'
  ].join('\n');

  var LINE_FS = [
    'precision mediump float;',
    'uniform vec4 uColor;',
    'void main() { gl_FragColor = uColor; }'
  ].join('\n');

  // ------------------------------------------------------------------ state
  var params = {
    diameter: 12,      // m — Fulldome.pro 12 m venue
    tilt: 0,           // deg — not documented by the vendor, assumed 0
    eyeHeight: 1.2,    // m above the springline plane (seated audience)
    fov: 75,
    lumaThr: 8,
    showGrid: true,
    showMark: true,
    showProjectors: true,
    mode: 'audience'
  };

  var view = {
    yaw: 0,
    pitch: 0.35,
    orbitAz: 0.6,
    orbitEl: 0.55,
    orbitDist: 2.9
  };

  var stats = null;
  var mediaKind = null;   // 'image' | 'video' | null
  var lastAnalyze = 0;

  var canvas = document.getElementById('gl');
  var video = document.getElementById('video');
  var gl = canvas.getContext('webgl', { antialias: true, alpha: false }) ||
           canvas.getContext('experimental-webgl');
  if (!gl) {
    document.getElementById('stage').innerHTML =
      '<p style="padding:24px;color:#ff6b6b">这个浏览器没有可用的 WebGL 上下文。</p>';
    return;
  }

  var domeProg = GLKit.createProgram(gl, DOME_VS, DOME_FS);
  var lineProg = GLKit.createProgram(gl, LINE_VS, LINE_FS);

  var mesh = G.hemisphere(72, 144);
  var domeBuf = GLKit.buffer(gl, mesh.positions);
  var domeIdx = GLKit.buffer(gl, mesh.indices, gl.ELEMENT_ARRAY_BUFFER);

  var lines = {
    grid: makeLines(G.grid()),
    floor: makeLines(G.floorPlan()),
    front: makeLines(G.frontMarker()),
    lower: makeLines(G.projectorRig(5, 0, -0.02, 22, 26)),
    upper: makeLines(G.projectorRig(3, 36, 0.12, 62, 22))
  };

  var tex = GLKit.createTexture(gl);
  var hasTex = 0;
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);

  function makeLines(arr) {
    return { buf: GLKit.buffer(gl, arr), count: arr.length / 3 };
  }

  // ------------------------------------------------------------------ render
  function resize() {
    var dpr = Math.min(window.devicePixelRatio || 1, 2);
    var w = Math.max(1, Math.floor(canvas.clientWidth * dpr));
    var h = Math.max(1, Math.floor(canvas.clientHeight * dpr));
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
  }

  function cameras() {
    var R = params.diameter / 2;
    var proj, viewM;
    var aspect = canvas.width / Math.max(1, canvas.height);

    if (params.mode === 'audience') {
      var eye = [0, params.eyeHeight, 0];
      var cp = Math.cos(view.pitch);
      var fwd = [
        cp * Math.sin(view.yaw),
        Math.sin(view.pitch),
        -cp * Math.cos(view.yaw)
      ];
      viewM = mat4.lookAt(eye, [eye[0] + fwd[0], eye[1] + fwd[1], eye[2] + fwd[2]], [0, 1, 0]);
      proj = mat4.perspective((params.fov * Math.PI) / 180, aspect, 0.02, R * 20 + 50);
    } else {
      var d = view.orbitDist * R;
      var ce = Math.cos(view.orbitEl);
      var eye2 = [
        d * ce * Math.sin(view.orbitAz),
        d * Math.sin(view.orbitEl),
        -d * ce * Math.cos(view.orbitAz)
      ];
      var target = [0, R * 0.25, 0];
      eye2[1] += target[1];
      viewM = mat4.lookAt(eye2, target, [0, 1, 0]);
      proj = mat4.perspective((45 * Math.PI) / 180, aspect, R * 0.02, d * 6 + 100);
    }
    return { proj: proj, view: viewM };
  }

  function drawLines(set, color, model, radius, cam) {
    if (!set.count) return;
    gl.useProgram(lineProg);
    gl.bindBuffer(gl.ARRAY_BUFFER, set.buf);
    gl.enableVertexAttribArray(lineProg.at.aPos);
    gl.vertexAttribPointer(lineProg.at.aPos, 3, gl.FLOAT, false, 0, 0);
    gl.uniformMatrix4fv(lineProg.un.uProj, false, cam.proj);
    gl.uniformMatrix4fv(lineProg.un.uView, false, cam.view);
    gl.uniformMatrix4fv(lineProg.un.uModel, false, model);
    gl.uniform1f(lineProg.un.uRadius, radius);
    gl.uniform4fv(lineProg.un.uColor, color);
    gl.drawArrays(gl.LINES, 0, set.count);
  }

  function render() {
    resize();
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clearColor(0.04, 0.05, 0.07, 1);
    gl.enable(gl.DEPTH_TEST);
    gl.disable(gl.CULL_FACE);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    var R = params.diameter / 2;
    var model = mat4.rotationX((-params.tilt * Math.PI) / 180);
    var cam = cameras();

    // dome
    gl.useProgram(domeProg);
    gl.bindBuffer(gl.ARRAY_BUFFER, domeBuf);
    gl.enableVertexAttribArray(domeProg.at.aDir);
    gl.vertexAttribPointer(domeProg.at.aDir, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, domeIdx);
    gl.uniformMatrix4fv(domeProg.un.uProj, false, cam.proj);
    gl.uniformMatrix4fv(domeProg.un.uView, false, cam.view);
    gl.uniformMatrix4fv(domeProg.un.uModel, false, model);
    gl.uniform1f(domeProg.un.uRadius, R);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.uniform1i(domeProg.un.uTex, 0);
    gl.uniform1f(domeProg.un.uHasTex, hasTex);
    gl.uniform1f(domeProg.un.uCovR, stats && !stats.error ? stats.coverageRadius : -1);
    gl.uniform1f(domeProg.un.uMark, params.showMark ? 1 : 0);
    gl.drawElements(gl.TRIANGLES, mesh.count, gl.UNSIGNED_SHORT, 0);

    // overlays
    if (params.showGrid) {
      drawLines(lines.grid, [0.55, 0.72, 0.85, 0.28], model, R * 1.001, cam);
      drawLines(lines.floor, [0.4, 0.5, 0.6, 0.22], model, R, cam);
      drawLines(lines.front, [1.0, 0.45, 0.35, 0.9], model, R, cam);
    }
    if (params.showProjectors) {
      drawLines(lines.lower, [0.36, 0.88, 1.0, 0.55], model, R, cam);
      drawLines(lines.upper, [1.0, 0.82, 0.35, 0.55], model, R, cam);
    }

    if (mediaKind === 'video' && !video.paused && !video.ended && video.readyState >= 2) {
      uploadTexture(video);
      var now = performance.now();
      if (now - lastAnalyze > 500) {
        lastAnalyze = now;
        runAnalysis();
      }
    }
    requestAnimationFrame(render);
  }

  function uploadTexture(source) {
    gl.bindTexture(gl.TEXTURE_2D, tex);
    try {
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source);
      hasTex = 1;
    } catch (e) {
      hasTex = 0;
    }
  }

  // ------------------------------------------------------------------ readouts
  function el(id) { return document.getElementById(id); }

  function pct(x) { return (x * 100).toFixed(1) + '%'; }

  function runAnalysis() {
    var source = mediaKind === 'video' ? video : currentBitmap;
    if (!source) return;
    stats = Coverage.analyze(source, params.lumaThr);
    updateReadouts();
  }

  function updateReadouts() {
    var v = el('verdict');
    if (!stats) {
      el('zenithDeg').textContent = '—';
      return;
    }
    if (stats.error === 'tainted') {
      el('zenithDeg').textContent = '—';
      v.textContent = '浏览器拒绝读取画面像素（file:// 限制）。' +
        '请在仓库根目录运行 python -m http.server 8000 后用 http://localhost:8000/web/dome-preview/ 打开。';
      v.style.borderLeftColor = 'var(--bad)';
      return;
    }
    el('zenithDeg').textContent = stats.coverageDeg.toFixed(1) + '°';
    el('covR').textContent = stats.coverageRadius.toFixed(3);
    el('outerFill').textContent = pct(stats.outerFill);
    el('contentFrac').textContent = pct(stats.contentFrac);
    el('solidAngle').textContent = pct(stats.solidAngleFrac);

    var ver = Coverage.verdict(stats);
    v.textContent = ver.text;
    v.style.borderLeftColor =
      ver.level === 'ok' ? 'var(--ok)' : ver.level === 'warn' ? 'var(--warn)' : 'var(--bad)';

    el('outerFill').style.color =
      stats.outerFill < 0.5 ? 'var(--bad)' : 'var(--ok)';

    drawProfile();
  }

  function drawProfile() {
    var c = el('profile');
    var g = c.getContext('2d');
    var w = c.width, h = c.height;
    g.clearRect(0, 0, w, h);
    g.fillStyle = '#11151b';
    g.fillRect(0, 0, w, h);
    if (!stats || stats.error) return;

    // outer ring band the owner cares about
    g.fillStyle = 'rgba(255,107,107,0.12)';
    g.fillRect(stats.outerLo * w, 0, (stats.outerHi - stats.outerLo) * w, h);

    g.strokeStyle = 'rgba(255,255,255,0.10)';
    g.beginPath();
    for (var t = 0.2; t < 1; t += 0.2) {
      g.moveTo(t * w, 0); g.lineTo(t * w, h);
    }
    g.stroke();

    g.fillStyle = '#4fc3f7';
    for (var b = 0; b < stats.bins; b++) {
      var x = (b / stats.bins) * w;
      var bw = w / stats.bins + 0.6;
      var bh = stats.fill[b] * (h - 6);
      g.fillRect(x, h - bh, bw, bh);
    }

    var cx = stats.coverageRadius * w;
    g.strokeStyle = '#ffbf42';
    g.lineWidth = 1.5;
    g.beginPath();
    g.moveTo(cx, 0); g.lineTo(cx, h);
    g.stroke();
    g.fillStyle = '#ffbf42';
    g.font = '10px system-ui, sans-serif';
    g.fillText(stats.coverageDeg.toFixed(1) + '°', Math.min(cx + 4, w - 30), 11);
  }

  // ------------------------------------------------------------------ media
  var currentBitmap = null;

  function loadFile(file) {
    if (!file) return;
    stats = null;
    if (currentBitmap && currentBitmap.close) currentBitmap.close();
    currentBitmap = null;

    if (/^video\//.test(file.type) || /\.(mp4|mov|webm|mkv)$/i.test(file.name)) {
      mediaKind = 'video';
      video.src = URL.createObjectURL(file);
      video.load();
      video.onloadeddata = function () {
        el('fileInfo').textContent =
          file.name + ' · ' + video.videoWidth + '×' + video.videoHeight +
          ' · 视频（空格键播放/暂停）';
        uploadTexture(video);
        runAnalysis();
      };
      video.play().catch(function () { /* autoplay may be blocked; fine */ });
      return;
    }

    mediaKind = 'image';
    video.pause();
    video.removeAttribute('src');
    createImageBitmap(file).then(function (bmp) {
      currentBitmap = bmp;
      el('fileInfo').textContent =
        file.name + ' · ' + bmp.width + '×' + bmp.height + ' · 静帧';
      uploadTexture(bmp);
      runAnalysis();
    }).catch(function (err) {
      el('fileInfo').textContent = '无法解码这个文件：' + err.message;
    });
  }

  el('file').addEventListener('change', function (e) {
    loadFile(e.target.files && e.target.files[0]);
  });

  // drag & drop anywhere on the stage
  document.addEventListener('dragover', function (e) { e.preventDefault(); });
  document.addEventListener('drop', function (e) {
    e.preventDefault();
    if (e.dataTransfer && e.dataTransfer.files.length) loadFile(e.dataTransfer.files[0]);
  });

  el('analyzeBtn').addEventListener('click', function () {
    if (mediaKind === 'video') uploadTexture(video);
    runAnalysis();
  });

  document.addEventListener('keydown', function (e) {
    if (e.code === 'Space' && mediaKind === 'video') {
      e.preventDefault();
      if (video.paused) video.play(); else video.pause();
    }
  });

  // ------------------------------------------------------------------ controls
  function slider(id, key, fmt, after) {
    var input = el(id), out = el(id + 'Val');
    function apply() {
      params[key] = parseFloat(input.value);
      out.textContent = fmt(params[key]);
      if (after) after();
    }
    input.addEventListener('input', apply);
    apply();
  }

  slider('diameter', 'diameter', function (v) { return v.toFixed(1) + ' m'; });
  slider('tilt', 'tilt', function (v) { return v.toFixed(0) + '°'; });
  slider('eyeHeight', 'eyeHeight', function (v) { return v.toFixed(2) + ' m'; });
  slider('fov', 'fov', function (v) { return v.toFixed(0) + '°'; });
  slider('lumaThr', 'lumaThr', function (v) { return v.toFixed(0); }, function () {
    if (mediaKind) runAnalysis();
  });

  el('resetVenue').addEventListener('click', function () {
    el('diameter').value = 12; el('tilt').value = 0; el('eyeHeight').value = 1.2;
    ['diameter', 'tilt', 'eyeHeight'].forEach(function (id) {
      el(id).dispatchEvent(new Event('input'));
    });
  });

  function setMode(mode) {
    params.mode = mode;
    el('camAudience').classList.toggle('on', mode === 'audience');
    el('camExternal').classList.toggle('on', mode === 'external');
  }
  el('camAudience').addEventListener('click', function () { setMode('audience'); });
  el('camExternal').addEventListener('click', function () { setMode('external'); });

  function toggle(id, key) {
    el(id).addEventListener('click', function () {
      params[key] = !params[key];
      el(id).classList.toggle('on', params[key]);
    });
  }
  toggle('gridBtn', 'showGrid');
  toggle('markBtn', 'showMark');
  toggle('projBtn', 'showProjectors');

  // ------------------------------------------------------------------ input
  var dragging = false, lastX = 0, lastY = 0;
  canvas.addEventListener('pointerdown', function (e) {
    dragging = true; lastX = e.clientX; lastY = e.clientY;
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener('pointerup', function (e) {
    dragging = false;
    try { canvas.releasePointerCapture(e.pointerId); } catch (err) { /* ignore */ }
  });
  canvas.addEventListener('pointermove', function (e) {
    if (!dragging) return;
    var dx = e.clientX - lastX, dy = e.clientY - lastY;
    lastX = e.clientX; lastY = e.clientY;
    if (params.mode === 'audience') {
      view.yaw -= dx * 0.005;
      view.pitch = clamp(view.pitch + dy * 0.005, -1.45, 1.45);
    } else {
      view.orbitAz -= dx * 0.008;
      view.orbitEl = clamp(view.orbitEl + dy * 0.006, -0.2, 1.45);
    }
  });
  canvas.addEventListener('wheel', function (e) {
    e.preventDefault();
    if (params.mode === 'external') {
      view.orbitDist = clamp(view.orbitDist * Math.exp(e.deltaY * 0.0012), 0.2, 12);
    } else {
      el('fov').value = clamp(params.fov + e.deltaY * 0.03, 40, 110);
      el('fov').dispatchEvent(new Event('input'));
    }
  }, { passive: false });

  function clamp(v, lo, hi) { return v < lo ? lo : v > hi ? hi : v; }

  window.addEventListener('resize', resize);
  render();
})();
