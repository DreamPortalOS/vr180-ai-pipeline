/* Immersive Node Studio — offline-capable canvas (no CDN). */
(() => {
  "use strict";

  const NODE_W = 200;
  const NODE_H = 88;
  const PORT_R = 6;
  const HEAD_H = 24;

  /** Built-in fallback when API is unreachable (file:// preview). */
  const FALLBACK_TYPES = [
    {
      type: "script.storyboard",
      category: "script",
      label: "分镜脚本",
      inputs: [],
      outputs: [
        { name: "prompt", type: "text" },
        { name: "duration", type: "number" },
        { name: "storyboard", type: "json" },
      ],
      param_schema: [
        { name: "title", type: "string", default: "untitled", label: "标题" },
        { name: "prompt", type: "string", default: "", label: "画面描述" },
        { name: "duration", type: "number", default: 5, label: "时长(秒)" },
        { name: "aspect_ratio", type: "string", default: "1:1", label: "画幅" },
      ],
    },
    {
      type: "text.llm_polish",
      category: "script",
      label: "LLM 润色",
      inputs: [
        { name: "prompt", type: "text" },
        { name: "instruction", type: "text" },
      ],
      outputs: [
        { name: "prompt", type: "text" },
        { name: "meta", type: "json" },
      ],
      param_schema: [
        { name: "provider", type: "string", default: "mock", label: "provider" },
        { name: "target", type: "string", default: "vr180", label: "目标" },
      ],
    },
    {
      type: "video.mock",
      category: "generate",
      label: "Mock 视频",
      inputs: [
        { name: "prompt", type: "text" },
        { name: "duration", type: "number" },
      ],
      outputs: [
        { name: "video", type: "video" },
        { name: "meta", type: "json" },
      ],
      param_schema: [
        { name: "duration", type: "number", default: 2, label: "时长" },
        { name: "size", type: "number", default: 256, label: "短边" },
      ],
    },
    {
      type: "video.seedance",
      category: "generate",
      label: "Seedance 视频",
      inputs: [
        { name: "prompt", type: "text" },
        { name: "image", type: "image" },
        { name: "duration", type: "number" },
      ],
      outputs: [
        { name: "video", type: "video" },
        { name: "meta", type: "json" },
      ],
      param_schema: [
        { name: "provider", type: "string", default: "mock", label: "provider" },
        { name: "resolution", type: "string", default: "480p", label: "分辨率" },
        { name: "confirm_paid", type: "boolean", default: false, label: "确认付费" },
      ],
    },
    {
      type: "qa.source_quality",
      category: "qa",
      label: "源片质检",
      inputs: [
        { name: "video", type: "video" },
        { name: "image", type: "image" },
      ],
      outputs: [
        { name: "report", type: "json" },
        { name: "passed", type: "number" },
        { name: "video", type: "video" },
      ],
      param_schema: [{ name: "mode", type: "string", default: "auto", label: "mode" }],
    },
    {
      type: "convert.dome",
      category: "convert",
      label: "球幕转换",
      inputs: [{ name: "video", type: "video" }],
      outputs: [
        { name: "video", type: "video" },
        { name: "meta", type: "json" },
      ],
      param_schema: [
        { name: "size", type: "number", default: 256, label: "边长" },
        { name: "coverage_h", type: "number", default: 120, label: "覆盖°" },
      ],
    },
    {
      type: "qa.dome_coverage",
      category: "qa",
      label: "球幕覆盖度",
      inputs: [{ name: "video", type: "video" }],
      outputs: [
        { name: "report", type: "json" },
        { name: "passed", type: "number" },
        { name: "video", type: "video" },
      ],
      param_schema: [{ name: "min_deg", type: "number", default: 85, label: "合格下限°" }],
    },
    {
      type: "convert.vr180",
      category: "convert",
      label: "VR180 转换",
      inputs: [{ name: "video", type: "video" }],
      outputs: [
        { name: "video", type: "video" },
        { name: "meta", type: "json" },
      ],
      param_schema: [
        { name: "mode", type: "string", default: "mock", label: "mode" },
        { name: "eye_size", type: "number", default: 256, label: "每眼边长" },
      ],
    },
    {
      type: "tool.preview",
      category: "tool",
      label: "预览",
      inputs: [{ name: "value", type: "any" }],
      outputs: [
        { name: "value", type: "any" },
        { name: "summary", type: "json" },
      ],
      param_schema: [{ name: "label", type: "string", default: "preview", label: "标签" }],
    },
    {
      type: "export.bundle",
      category: "export",
      label: "导出",
      inputs: [
        { name: "video", type: "video" },
        { name: "prompt", type: "text" },
      ],
      outputs: [
        { name: "path", type: "text" },
        { name: "manifest", type: "json" },
      ],
      param_schema: [{ name: "filename", type: "string", default: "export.mp4", label: "文件名" }],
    },
  ];

  function fallbackDualProject() {
    return {
      version: 1,
      name: "production (offline fallback)",
      nodes: [
        { id: "n_brief", type: "script.project", pos: [20, 160], params: { title: "demo", theme: "canyon", style: "cinematic", total_seconds: 6 }, muted: false },
        { id: "n_shots", type: "script.shot_list", pos: [250, 100], params: { shot_texts: "open\npush", shot_durations: "3,3" }, muted: false },
        { id: "n_polish", type: "text.polish_shots", pos: [480, 40], params: { provider: "mock" }, muted: false },
        { id: "n_stills", type: "image.batch_stills", pos: [710, 40], params: { width: 160, height: 160 }, muted: false },
        { id: "n_review", type: "checkpoint.review", pos: [940, 40], params: { ack: true }, muted: false },
        { id: "n_clips", type: "video.from_stills", pos: [710, 260], params: { size: 128, fps: 10 }, muted: false },
        { id: "n_concat", type: "video.concat", pos: [940, 260], params: {}, muted: false },
        { id: "n_bgm", type: "audio.bgm_tone", pos: [940, 420], params: { duration: 6 }, muted: false },
        { id: "n_mux", type: "audio.mux", pos: [1170, 260], params: {}, muted: false },
        { id: "n_dome", type: "convert.dome", pos: [20, 400], params: { size: 128 }, muted: false },
        { id: "n_vr", type: "convert.vr180", pos: [250, 400], params: { mode: "mock" }, muted: false },
        { id: "n_export", type: "export.bundle", pos: [480, 400], params: { filename: "final.mp4" }, muted: false },
      ],
      edges: [
        { id: "e1", from: ["n_brief", "brief"], to: ["n_shots", "brief"] },
        { id: "e2", from: ["n_shots", "shots"], to: ["n_polish", "shots"] },
        { id: "e3", from: ["n_polish", "shots"], to: ["n_stills", "shots"] },
        { id: "e4", from: ["n_stills", "stills"], to: ["n_review", "stills"] },
        { id: "e5", from: ["n_review", "stills"], to: ["n_clips", "stills"] },
        { id: "e6", from: ["n_clips", "videos"], to: ["n_concat", "videos"] },
        { id: "e7", from: ["n_shots", "total_seconds"], to: ["n_bgm", "duration"] },
        { id: "e8", from: ["n_concat", "video"], to: ["n_mux", "video"] },
        { id: "e9", from: ["n_bgm", "audio"], to: ["n_mux", "audio"] },
        { id: "e10", from: ["n_mux", "video"], to: ["n_dome", "video"] },
        { id: "e11", from: ["n_mux", "video"], to: ["n_vr", "video"] },
        { id: "e12", from: ["n_vr", "video"], to: ["n_export", "video"] },
      ],
      settings: { workflow: "script→stills→review→video→concat→audio→export" },
    };
  }

  /** Normalize server edges {from:[n,p]} and canvas edges {from_node,...}. */
  function normalizeProject(raw) {
    const nodes = (raw.nodes || []).map((n) => ({
      id: n.id,
      type: n.type,
      pos: Array.isArray(n.pos) ? [Number(n.pos[0]) || 0, Number(n.pos[1]) || 0] : [0, 0],
      params: n.params || {},
      muted: !!n.muted,
    }));
    const edges = (raw.edges || []).map((e) => {
      if (e.from_node) {
        return {
          id: e.id,
          from_node: e.from_node,
          from_port: e.from_port,
          to_node: e.to_node,
          to_port: e.to_port,
        };
      }
      const src = e.from || [];
      const dst = e.to || [];
      return {
        id: e.id,
        from_node: src[0],
        from_port: src[1],
        to_node: dst[0],
        to_port: dst[1],
      };
    });
    return {
      version: raw.version || 1,
      name: raw.name || "untitled",
      nodes,
      edges,
      settings: raw.settings || {},
    };
  }

  function toServerProject(p) {
    return {
      version: p.version || 1,
      name: p.name,
      nodes: p.nodes.map((n) => ({
        id: n.id,
        type: n.type,
        pos: n.pos,
        params: n.params || {},
        muted: !!n.muted,
      })),
      edges: p.edges.map((e) => ({
        id: e.id,
        from: [e.from_node, e.from_port],
        to: [e.to_node, e.to_port],
      })),
      settings: p.settings || {},
    };
  }

  const state = {
    project: { version: 1, name: "untitled", nodes: [], edges: [], settings: {} },
    nodeTypes: Object.fromEntries(FALLBACK_TYPES.map((t) => [t.type, t])),
    selectedId: null,
    drag: null,
    linking: null,
    status: {},
    pan: { x: 20, y: 20 },
    spacePan: null,
    online: false,
    paletteFilter: "",
  };

  const canvas = document.getElementById("canvas");
  const ctx = canvas.getContext("2d");
  const paletteEl = document.getElementById("nodePalette");
  const inspectorEl = document.getElementById("inspectorBody");
  const runOutEl = document.getElementById("runOut");
  const statusBar = document.getElementById("statusBar");
  const projectNameEl = document.getElementById("projectName");
  const fileInput = document.getElementById("fileInput");
  const emptyOverlay = document.getElementById("canvasEmpty");
  const backendStatus = document.getElementById("backendStatus");
  const paletteSearch = document.getElementById("paletteSearch");

  function uid(prefix) {
    return `${prefix}_${Math.random().toString(36).slice(2, 9)}`;
  }

  function setStatus(msg) {
    statusBar.textContent = msg;
  }

  function setBackend(ok, detail) {
    state.online = ok;
    backendStatus.textContent = detail;
    backendStatus.className = "hint " + (ok ? "ok" : "err");
  }

  function resizeCanvas() {
    const wrap = canvas.parentElement;
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(320, wrap.clientWidth);
    const h = Math.max(240, wrap.clientHeight);
    canvas.width = Math.floor(w * dpr);
    canvas.height = Math.floor(h * dpr);
    canvas.style.width = w + "px";
    canvas.style.height = h + "px";
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    draw();
  }

  function viewSize() {
    return {
      w: canvas.clientWidth || canvas.width,
      h: canvas.clientHeight || canvas.height,
    };
  }

  async function api(path, options) {
    const res = await fetch(path, options);
    if (!res.ok) {
      let detail = res.statusText;
      try {
        const body = await res.json();
        detail = body.detail || JSON.stringify(body);
      } catch (_) {
        /* ignore */
      }
      throw new Error(detail);
    }
    return res.json();
  }

  function renderPalette() {
    const q = (state.paletteFilter || "").trim().toLowerCase();
    paletteEl.innerHTML = "";
    Object.values(state.nodeTypes)
      .slice()
      .sort((a, b) => (a.category + a.label).localeCompare(b.category + b.label, "zh"))
      .forEach((t) => {
        const hay = (t.type + " " + (t.label || "") + " " + (t.category || "")).toLowerCase();
        if (q && !hay.includes(q)) return;
        const btn = document.createElement("button");
        btn.type = "button";
        btn.innerHTML = `<span>${t.label || t.type}</span><span class="cat">${t.category || ""} · ${t.type}</span>`;
        btn.addEventListener("click", () => addNode(t.type));
        paletteEl.appendChild(btn);
      });
  }

  function addNode(type) {
    const meta = state.nodeTypes[type];
    if (!meta) return;
    const id = uid("n");
    const params = {};
    (meta.param_schema || []).forEach((p) => {
      params[p.name] = p.default;
    });
    const { w, h } = viewSize();
    const n = state.project.nodes.length;
    const x = 40 + ((n * 40) % Math.max(120, w - NODE_W - 80));
    const y = 40 + ((n * 36) % Math.max(100, h - NODE_H - 80));
    state.project.nodes.push({ id, type, pos: [x - state.pan.x, y - state.pan.y], params, muted: false });
    state.selectedId = id;
    updateEmpty();
    renderInspector();
    draw();
    setStatus(`已添加 ${meta.label || type}`);
  }

  function nodeById(id) {
    return state.project.nodes.find((n) => n.id === id);
  }

  function portPositions(node) {
    const meta = state.nodeTypes[node.type] || { inputs: [], outputs: [] };
    const inputs = (meta.inputs || []).map((p, i) => ({
      ...p,
      x: node.pos[0],
      y: node.pos[1] + HEAD_H + 18 + i * 20,
      side: "in",
    }));
    const outputs = (meta.outputs || []).map((p, i) => ({
      ...p,
      x: node.pos[0] + NODE_W,
      y: node.pos[1] + HEAD_H + 18 + i * 20,
      side: "out",
    }));
    return { inputs, outputs };
  }

  function hitPort(x, y) {
    for (let i = state.project.nodes.length - 1; i >= 0; i -= 1) {
      const node = state.project.nodes[i];
      const { inputs, outputs } = portPositions(node);
      for (const p of [...inputs, ...outputs]) {
        const dx = x - p.x;
        const dy = y - p.y;
        if (dx * dx + dy * dy <= (PORT_R + 5) ** 2) return { node, port: p };
      }
    }
    return null;
  }

  function hitNode(x, y) {
    for (let i = state.project.nodes.length - 1; i >= 0; i -= 1) {
      const n = state.project.nodes[i];
      if (x >= n.pos[0] && x <= n.pos[0] + NODE_W && y >= n.pos[1] && y <= n.pos[1] + NODE_H) return n;
    }
    return null;
  }

  function canvasPoint(evt) {
    const rect = canvas.getBoundingClientRect();
    return {
      x: evt.clientX - rect.left - state.pan.x,
      y: evt.clientY - rect.top - state.pan.y,
    };
  }

  function roundRect(x, y, w, h, r) {
    const rr = Math.min(r, w / 2, h / 2);
    ctx.beginPath();
    ctx.moveTo(x + rr, y);
    ctx.arcTo(x + w, y, x + w, y + h, rr);
    ctx.arcTo(x + w, y + h, x, y + h, rr);
    ctx.arcTo(x, y + h, x, y, rr);
    ctx.arcTo(x, y, x + w, y, rr);
    ctx.closePath();
  }

  function statusColor(st) {
    if (st === "ok") return "#1f6b4a";
    if (st === "error") return "#7a2e35";
    if (st === "running") return "#7a5a1d";
    if (st === "skipped") return "#3a4454";
    return "#2a3a52";
  }

  function portColor(type) {
    const colors = {
      text: "#9aa7b8",
      video: "#b388ff",
      image: "#4dd0c8",
      json: "#e6b450",
      number: "#5b9dff",
      any: "#cfd8e3",
    };
    return colors[type] || "#cfd8e3";
  }

  function drawWire(x1, y1, x2, y2, color) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.25;
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    const cx = (x1 + x2) / 2;
    ctx.bezierCurveTo(cx, y1, cx, y2, x2, y2);
    ctx.stroke();
  }

  function draw() {
    const { w, h } = viewSize();
    ctx.clearRect(0, 0, w, h);
    ctx.save();
    ctx.translate(state.pan.x, state.pan.y);

    state.project.edges.forEach((e) => {
      const a = nodeById(e.from_node);
      const b = nodeById(e.to_node);
      if (!a || !b) return;
      const pa = portPositions(a).outputs.find((p) => p.name === e.from_port);
      const pb = portPositions(b).inputs.find((p) => p.name === e.to_port);
      if (!pa || !pb) return;
      drawWire(pa.x, pa.y, pb.x, pb.y, "#6aa6d8");
    });

    if (state.linking) {
      drawWire(state.linking.x1, state.linking.y1, state.linking.x2, state.linking.y2, "#e6b450");
    }

    state.project.nodes.forEach((n) => {
      const meta = state.nodeTypes[n.type] || { label: n.type, inputs: [], outputs: [] };
      const selected = n.id === state.selectedId;
      const st = state.status[n.id] || "idle";

      ctx.fillStyle = selected ? "#27364d" : "#1e2a3a";
      ctx.strokeStyle = selected ? "#3d9cf0" : "#2a3545";
      ctx.lineWidth = selected ? 2 : 1;
      roundRect(n.pos[0], n.pos[1], NODE_W, NODE_H, 10);
      ctx.fill();
      ctx.stroke();

      ctx.fillStyle = statusColor(st);
      roundRect(n.pos[0], n.pos[1], NODE_W, HEAD_H, 10);
      ctx.fill();
      ctx.fillStyle = "#1a2330";
      ctx.fillRect(n.pos[0], n.pos[1] + HEAD_H - 8, NODE_W, 8);

      ctx.fillStyle = "#e8eef7";
      ctx.font = "600 12px 'Segoe UI', 'PingFang SC', sans-serif";
      const title = meta.label || n.type;
      ctx.fillText(title, n.pos[0] + 10, n.pos[1] + 16);

      ctx.fillStyle = "#8fa0b5";
      ctx.font = "10px ui-monospace, Consolas, monospace";
      ctx.fillText(n.id, n.pos[0] + 10, n.pos[1] + NODE_H - 8);

      if (n.muted) {
        ctx.fillStyle = "rgba(0,0,0,0.35)";
        roundRect(n.pos[0], n.pos[1], NODE_W, NODE_H, 10);
        ctx.fill();
      }

      const { inputs, outputs } = portPositions(n);
      inputs.forEach((p) => {
        ctx.fillStyle = portColor(p.type);
        ctx.beginPath();
        ctx.arc(p.x, p.y, PORT_R, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = "#0b1016";
        ctx.stroke();
        ctx.fillStyle = "#8fa0b5";
        ctx.font = "10px sans-serif";
        ctx.fillText(p.name, p.x + 9, p.y + 3);
      });
      outputs.forEach((p) => {
        ctx.fillStyle = portColor(p.type);
        ctx.beginPath();
        ctx.arc(p.x, p.y, PORT_R, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = "#0b1016";
        ctx.stroke();
        ctx.fillStyle = "#8fa0b5";
        ctx.font = "10px sans-serif";
        const tw = ctx.measureText(p.name).width;
        ctx.fillText(p.name, p.x - 9 - tw, p.y + 3);
      });
    });

    ctx.restore();
  }

  function updateEmpty() {
    emptyOverlay.classList.toggle("hidden", state.project.nodes.length > 0);
  }

  function renderInspector() {
    const node = nodeById(state.selectedId);
    if (!node) {
      inspectorEl.innerHTML = '<p class="hint">选中画布上的节点以编辑参数</p>';
      return;
    }
    const meta = state.nodeTypes[node.type] || { label: node.type, param_schema: [] };
    inspectorEl.innerHTML = "";

    const head = document.createElement("div");
    head.innerHTML = `<div style="font-weight:700">${meta.label || node.type}</div><div class="node-type">${node.type}</div>`;
    inspectorEl.appendChild(head);

    (meta.param_schema || []).forEach((p) => {
      const label = document.createElement("label");
      label.textContent = p.label || p.name;
      let field;
      if (p.type === "boolean") {
        field = document.createElement("input");
        field.type = "checkbox";
        field.checked = !!node.params[p.name];
        field.addEventListener("change", () => {
          node.params[p.name] = field.checked;
        });
      } else if (p.type === "string" && String(p.default || "").length > 40) {
        field = document.createElement("textarea");
        field.value = node.params[p.name] ?? p.default ?? "";
        field.addEventListener("change", () => {
          node.params[p.name] = field.value;
        });
      } else if (p.type === "number") {
        field = document.createElement("input");
        field.type = "number";
        field.value = node.params[p.name] ?? p.default ?? 0;
        field.addEventListener("change", () => {
          node.params[p.name] = Number(field.value);
        });
      } else {
        field = document.createElement("input");
        field.type = "text";
        field.value = node.params[p.name] ?? p.default ?? "";
        field.addEventListener("change", () => {
          node.params[p.name] = field.value;
        });
      }
      label.appendChild(field);
      inspectorEl.appendChild(label);
    });

    const del = document.createElement("button");
    del.type = "button";
    del.className = "danger";
    del.textContent = "删除节点";
    del.addEventListener("click", () => removeNode(node.id));
    inspectorEl.appendChild(del);
  }

  function removeNode(id) {
    state.project.nodes = state.project.nodes.filter((n) => n.id !== id);
    state.project.edges = state.project.edges.filter((e) => e.from_node !== id && e.to_node !== id);
    if (state.selectedId === id) state.selectedId = null;
    updateEmpty();
    renderInspector();
    draw();
  }

  function fitView() {
    const nodes = state.project.nodes;
    if (!nodes.length) {
      state.pan = { x: 20, y: 20 };
      draw();
      return;
    }
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    nodes.forEach((n) => {
      minX = Math.min(minX, n.pos[0]);
      minY = Math.min(minY, n.pos[1]);
      maxX = Math.max(maxX, n.pos[0] + NODE_W);
      maxY = Math.max(maxY, n.pos[1] + NODE_H);
    });
    const { w, h } = viewSize();
    const pad = 48;
    const contentW = Math.max(1, maxX - minX);
    const contentH = Math.max(1, maxY - minY);
    // Center without zoom for now; clamp so content is fully visible when possible.
    let ox = pad - minX;
    let oy = pad - minY;
    if (contentW + pad * 2 < w) ox = (w - contentW) / 2 - minX;
    if (contentH + pad * 2 < h) oy = (h - contentH) / 2 - minY;
    // If wider than viewport, keep left edge visible.
    if (contentW + pad * 2 >= w) ox = pad - minX;
    if (contentH + pad * 2 >= h) oy = pad - minY;
    state.pan = { x: ox, y: oy };
    draw();
  }

  function loadProjectObject(data) {
    state.project = normalizeProject(data);
    projectNameEl.textContent = state.project.name || "untitled";
    state.selectedId = null;
    state.status = {};
    updateEmpty();
    renderInspector();
    fitView();
  }

  function saveProject() {
    const serverShape = toServerProject(state.project);
    const blob = new Blob([JSON.stringify(serverShape, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${state.project.name || "project"}.studio.json`;
    a.click();
    URL.revokeObjectURL(a.href);
    setStatus("已下载 JSON");
  }

  function loadProjectFile(file) {
    const reader = new FileReader();
    reader.onload = () => {
      try {
        loadProjectObject(JSON.parse(String(reader.result)));
        setStatus("工程已打开");
      } catch (err) {
        setStatus("打开失败: " + err.message);
      }
    };
    reader.readAsText(file);
  }

  async function loadCatalogFromApi() {
    const types = await api("/api/node-types");
    if (Array.isArray(types) && types.length) {
      state.nodeTypes = Object.fromEntries(types.map((t) => [t.type, t]));
    }
    renderPalette();
  }

  async function loadDemo() {
    if (state.online) {
      const data = await api("/api/demo-project");
      loadProjectObject(data);
    } else {
      loadProjectObject(fallbackDualProject());
    }
    setStatus("已载入演示工程");
  }

  async function loadDualTemplate() {
    if (state.online) {
      const data = await api("/api/templates/production");
      loadProjectObject(data);
    } else {
      loadProjectObject(fallbackDualProject());
    }
    setStatus("已载入生产流程模板（脚本→分镜图→视频→拼合→配乐→导出）");
  }

  async function runProject() {
    if (!state.online) {
      runOutEl.textContent =
        "后端未连接，无法执行图。\n请在仓库根目录运行：\n  python -m studio.server\n然后打开 http://127.0.0.1:8787";
      setStatus("需要后端才能运行");
      return;
    }
    setStatus("运行中…");
    runOutEl.textContent = "…";
    try {
      const report = await api("/api/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ project: toServerProject(state.project) }),
      });
      state.status = Object.fromEntries(
        Object.entries(report.results || {}).map(([id, r]) => [id, r.status]),
      );
      runOutEl.textContent = JSON.stringify(report, null, 2);
      setStatus("运行完成");
      draw();
    } catch (err) {
      runOutEl.textContent = String(err.message || err);
      setStatus("运行失败");
      draw();
    }
  }

  // --- pointer ---
  canvas.addEventListener("mousedown", (evt) => {
    const p = canvasPoint(evt);
    if (evt.button === 1 || state.spacePan) {
      state.drag = { mode: "pan", sx: evt.clientX, sy: evt.clientY, px: state.pan.x, py: state.pan.y };
      return;
    }
    const port = hitPort(p.x, p.y);
    if (port && port.port.side === "out") {
      state.linking = {
        fromNode: port.node.id,
        fromPort: port.port.name,
        x1: port.port.x,
        y1: port.port.y,
        x2: p.x,
        y2: p.y,
      };
      draw();
      return;
    }
    const node = hitNode(p.x, p.y);
    if (node) {
      state.selectedId = node.id;
      state.drag = {
        mode: "node",
        id: node.id,
        ox: p.x - node.pos[0],
        oy: p.y - node.pos[1],
      };
      renderInspector();
      draw();
      return;
    }
    state.selectedId = null;
    state.drag = { mode: "pan", sx: evt.clientX, sy: evt.clientY, px: state.pan.x, py: state.pan.y };
    renderInspector();
    draw();
  });

  canvas.addEventListener("mousemove", (evt) => {
    const p = canvasPoint(evt);
    if (state.linking) {
      state.linking.x2 = p.x;
      state.linking.y2 = p.y;
      draw();
      return;
    }
    if (!state.drag) return;
    if (state.drag.mode === "pan") {
      state.pan.x = state.drag.px + (evt.clientX - state.drag.sx);
      state.pan.y = state.drag.py + (evt.clientY - state.drag.sy);
      draw();
      return;
    }
    const node = nodeById(state.drag.id);
    if (node) {
      node.pos = [p.x - state.drag.ox, p.y - state.drag.oy];
      draw();
    }
  });

  window.addEventListener("mouseup", (evt) => {
    if (state.linking) {
      const p = canvasPoint(evt);
      const target = hitPort(p.x, p.y);
      if (target && target.port.side === "in" && target.node.id !== state.linking.fromNode) {
        state.project.edges = state.project.edges.filter(
          (e) => !(e.to_node === target.node.id && e.to_port === target.port.name),
        );
        state.project.edges.push({
          id: uid("e"),
          from_node: state.linking.fromNode,
          from_port: state.linking.fromPort,
          to_node: target.node.id,
          to_port: target.port.name,
        });
        setStatus("已连接");
      }
      state.linking = null;
      draw();
    }
    state.drag = null;
  });

  canvas.addEventListener("dblclick", (evt) => {
    const p = canvasPoint(evt);
    const node = hitNode(p.x, p.y);
    if (node) {
      state.selectedId = node.id;
      renderInspector();
    }
  });

  canvas.addEventListener(
    "wheel",
    (evt) => {
      // simple pan with wheel; Shift+wheel pans Y
      evt.preventDefault();
      if (evt.shiftKey) state.pan.y -= evt.deltaY;
      else state.pan.x -= evt.deltaX || evt.deltaY;
      draw();
    },
    { passive: false },
  );

  function bindUi() {
    document.getElementById("btnDemo").addEventListener("click", () => loadDemo().catch((e) => setStatus(e.message)));
    document.getElementById("btnDual").addEventListener("click", () => loadDualTemplate().catch((e) => setStatus(e.message)));
    document.getElementById("btnEmptyDual").addEventListener("click", () => loadDualTemplate().catch((e) => setStatus(e.message)));
    document.getElementById("btnSave").addEventListener("click", saveProject);
    document.getElementById("btnLoad").addEventListener("click", () => fileInput.click());
    document.getElementById("btnClear").addEventListener("click", () => {
      state.project = { version: 1, name: "untitled", nodes: [], edges: [], settings: {} };
      state.selectedId = null;
      state.status = {};
      projectNameEl.textContent = "untitled";
      updateEmpty();
      renderInspector();
      draw();
    });
    document.getElementById("btnRun").addEventListener("click", () => runProject());
    fileInput.addEventListener("change", () => {
      if (fileInput.files && fileInput.files[0]) loadProjectFile(fileInput.files[0]);
      fileInput.value = "";
    });
    paletteSearch.addEventListener("input", () => {
      state.paletteFilter = paletteSearch.value;
      renderPalette();
    });

    window.addEventListener("keydown", (evt) => {
      if (evt.code === "Space") {
        state.spacePan = true;
        canvas.style.cursor = "grab";
      }
      if ((evt.ctrlKey || evt.metaKey) && evt.key === "Enter") {
        evt.preventDefault();
        runProject();
      }
      if ((evt.ctrlKey || evt.metaKey) && evt.key.toLowerCase() === "s") {
        evt.preventDefault();
        saveProject();
      }
      if (evt.key === "Delete" && state.selectedId) removeNode(state.selectedId);
      if (evt.key === "Escape") {
        state.linking = null;
        draw();
      }
    });
    window.addEventListener("keyup", (evt) => {
      if (evt.code === "Space") {
        state.spacePan = null;
        canvas.style.cursor = "default";
      }
    });
    window.addEventListener("resize", resizeCanvas);
  }

  async function boot() {
    bindUi();
    renderPalette();
    updateEmpty();
    resizeCanvas();
    try {
      const health = await api("/api/health");
      setBackend(true, `已连接 · ${health.work_root || ""}`);
      await loadCatalogFromApi();
    } catch (_) {
      setBackend(false, "离线模式：可编辑画布；运行需 python -m studio.server");
    }
    try {
      await loadDualTemplate();
    } catch (_) {
      loadProjectObject(fallbackDualProject());
    }
    resizeCanvas();
  }

  boot();
})();
