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
      type: "input.text",
      category: "input",
      label: "文本输入",
      inputs: [],
      outputs: [{ name: "text", type: "text" }],
      param_schema: [
        { name: "label", type: "string", default: "input", label: "标签" },
        { name: "text", type: "string", default: "", label: "文本" },
      ],
    },
    {
      type: "input.image",
      category: "input",
      label: "图片输入",
      inputs: [{ name: "value", type: "image" }],
      outputs: [
        { name: "image", type: "image" },
        { name: "meta", type: "json" },
      ],
      param_schema: [{ name: "path", type: "string", default: "", label: "图片路径" }],
    },
    {
      type: "input.video",
      category: "input",
      label: "视频输入",
      inputs: [{ name: "value", type: "video" }],
      outputs: [
        { name: "video", type: "video" },
        { name: "meta", type: "json" },
      ],
      param_schema: [{ name: "path", type: "string", default: "", label: "视频路径" }],
    },
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

  /** Palette group ordering — 输入 first so the drop-created nodes are visible. */
  const PALETTE_GROUPS = [
    ["input", "输入"],
    ["script", "脚本"],
    ["generate", "生成"],
    ["convert", "转换"],
    ["qa", "质检"],
    ["checkpoint", "审核"],
    ["tool", "工具"],
    ["export", "导出"],
  ];

  function renderPalette() {
    const q = (state.paletteFilter || "").trim().toLowerCase();
    paletteEl.innerHTML = "";
    const types = Object.values(state.nodeTypes);
    const hayFor = (t) => (t.type + " " + (t.label || "") + " " + (t.category || "")).toLowerCase();
    const groups = PALETTE_GROUPS.map(([key, label]) => [
      key,
      label,
      types.filter((t) => (t.category || "tool") === key),
    ]);
    // Anything with an unlisted category is appended as a final "其他" group.
    const known = new Set(PALETTE_GROUPS.map((g) => g[0]));
    const other = types.filter((t) => !known.has(t.category || "tool"));
    if (other.length) groups.push(["other", "其他", other]);

    groups.forEach(([key, label, items]) => {
      const rows = items.filter((t) => !q || hayFor(t).includes(q));
      if (!rows.length) return;
      const head = document.createElement("div");
      head.className = "palette-group-title";
      head.textContent = label;
      paletteEl.appendChild(head);
      rows
        .slice()
        .sort((a, b) => (a.type + a.label).localeCompare(b.type + b.label, "zh"))
        .forEach((t) => {
          const btn = document.createElement("button");
          btn.type = "button";
          btn.innerHTML = `<span>${t.label || t.type}</span><span class="cat">${t.category || ""} · ${t.type}</span>`;
          btn.addEventListener("click", () => addNode(t.type));
          paletteEl.appendChild(btn);
        });
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
      // input.image / input.video render a thumbnail or poster frame behind
      // the ports (input.text keeps the plain card layout).
      const thumb = inputThumb(n);
      if (thumb) {
        ctx.save();
        ctx.beginPath();
        ctx.rect(n.pos[0] + 4, n.pos[1] + HEAD_H + 4, NODE_W - 8, NODE_H - HEAD_H - 8);
        ctx.clip();
        ctx.fillStyle = "#0a0e14";
        ctx.fillRect(n.pos[0] + 4, n.pos[1] + HEAD_H + 4, NODE_W - 8, NODE_H - HEAD_H - 8);
        if (thumb.img && thumb.img.complete && thumb.img.naturalWidth) {
          const bx = n.pos[0] + 4;
          const by = n.pos[1] + HEAD_H + 4;
          const bw = NODE_W - 8;
          const bh = NODE_H - HEAD_H - 8;
          // cover-fit, centered
          const r = Math.max(bw / thumb.img.naturalWidth, bh / thumb.img.naturalHeight);
          const dw = thumb.img.naturalWidth * r;
          const dh = thumb.img.naturalHeight * r;
          ctx.drawImage(thumb.img, bx + (bw - dw) / 2, by + (bh - dh) / 2, dw, dh);
        } else {
          ctx.fillStyle = "#8fa0b5";
          ctx.font = "10px sans-serif";
          ctx.fillText("加载缩略图…", bx + 6, by + bh / 2 + 3);
        }
        ctx.restore();
        // Kind badge (duration / resolution) in the node corner.
        if (thumb.badge) {
          ctx.fillStyle = "rgba(0,0,0,0.62)";
          ctx.fillRect(n.pos[0] + NODE_W - 72, n.pos[1] + NODE_H - 16, 68, 13);
          ctx.fillStyle = "#e8eef7";
          ctx.font = "10px ui-monospace, Consolas, monospace";
          ctx.fillText(thumb.badge, n.pos[0] + NODE_W - 68, n.pos[1] + NODE_H - 6);
        }
      }

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

    // Drag-over affordance: a dashed "drop here" rectangle on the canvas.
    if (dropHover) {
      const { w, h } = viewSize();
      ctx.save();
      ctx.strokeStyle = "#3d9cf0";
      ctx.lineWidth = 2;
      ctx.setLineDash([8, 6]);
      ctx.strokeRect(6, 6, w - 12, h - 12);
      ctx.setLineDash([]);
      ctx.fillStyle = "#3d9cf0";
      ctx.font = "600 14px 'Segoe UI', 'PingFang SC', sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("松开以在此处新建输入节点", w / 2, 28);
      ctx.textAlign = "left";
      ctx.restore();
    }
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
      // Path params for input nodes get a file picker, not a raw text field —
      // typing an absolute path is brittle, and the picker uploads + de-dups.
      const isPathParam = p.type === "string" && p.name === "path";
      if (isPathParam && node.type.startsWith("input.")) {
        const row = document.createElement("div");
        row.className = "input-path-row";
        row.style.marginTop = "10px";
        const hint = document.createElement("div");
        hint.style.fontSize = "11px";
        hint.style.color = "var(--muted)";
        hint.textContent = "素材路径";
        row.appendChild(hint);
        const cur = node.params[p.name] || "";
        const out = document.createElement("div");
        out.style.cssText =
          "font-size:11px;font-family:ui-monospace,Consolas,monospace;word-break:break-all;margin-top:3px;color:#b8c7d9";
        out.textContent = cur || "未选择";
        row.appendChild(out);
        const picker = document.createElement("label");
        picker.type = "button";
        picker.className = "file-btn";
        picker.style.cssText = "display:block;margin-top:6px";
        picker.textContent = node.type === "input.video" ? "重新选择视频…" : "选择图片…";
        const inp = document.createElement("input");
        inp.type = "file";
        inp.accept =
          node.type === "input.video"
            ? "video/mp4,video/quicktime,.mp4,.mov"
            : "image/png,image/jpeg,image/webp,.png,.jpg,.jpeg,.webp";
        inp.hidden = true;
        inp.addEventListener("change", () => {
          if (inp.files && inp.files[0]) {
            reuploadIntoNode(inp.files[0], node);
          }
          inp.value = "";
        });
        picker.appendChild(inp);
        row.appendChild(picker);
        inspectorEl.appendChild(row);
        return;
      }
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
      } else if (node.type === "input.text" && p.name === "text") {
        // input.text edits its payload right on the card — no separate node.
        field = document.createElement("textarea");
        field.rows = 6;
        field.value = node.params[p.name] ?? p.default ?? "";
        field.placeholder = "在此直接输入文本 / prompt…";
        field.addEventListener("input", () => {
          node.params[p.name] = field.value;
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

    const dirty = document.createElement("button");
    dirty.type = "button";
    dirty.textContent = "仅重跑此节点下游";
    dirty.title = "POST /api/run with dirty_from=" + node.id;
    dirty.style.marginTop = "8px";
    dirty.addEventListener("click", () => runProject({ dirty_from: node.id }));
    inspectorEl.appendChild(dirty);

    const onlyAnc = document.createElement("button");
    onlyAnc.type = "button";
    onlyAnc.textContent = "仅跑此节点+上游";
    onlyAnc.style.marginTop = "6px";
    onlyAnc.addEventListener("click", () => runProject({ only_downstream_of: node.id }));
    inspectorEl.appendChild(onlyAnc);
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

  async function runProject(opts) {
    opts = opts || {};
    if (!state.online) {
      runOutEl.textContent =
        "后端未连接，无法执行图。\n请在仓库根目录运行：\n  python -m studio.server\n然后打开 http://127.0.0.1:8787";
      setStatus("需要后端才能运行");
      return;
    }
    const label = opts.dirty_from
      ? `重跑下游 ${opts.dirty_from}…`
      : opts.only_downstream_of
        ? `仅跑 ${opts.only_downstream_of}+上游…`
        : "运行中…";
    setStatus(label);
    runOutEl.textContent = "…";
    try {
      const payload = { project: toServerProject(state.project) };
      if (opts.dirty_from) payload.dirty_from = opts.dirty_from;
      if (opts.only_downstream_of) payload.only_downstream_of = opts.only_downstream_of;
      const report = await api("/api/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      state.status = Object.fromEntries(
        Object.entries(report.results || {}).map(([id, r]) => [id, r.status]),
      );
      runOutEl.textContent = JSON.stringify(report, null, 2);
      renderGallery(report.gallery);
      // Feed 3D panel if a coverage node ran
      const cov = (report.results && (report.results.n_cov || report.results.cov)) || null;
      if (cov && cov.outputs && cov.outputs.report) {
        const rep = cov.outputs.report;
        window.dispatchEvent(
          new CustomEvent("studio:coverage", {
            detail: {
              coverageRadius: rep.coverage_radius != null ? rep.coverage_radius : rep.coverageRadius,
              coverageDeg: rep.coverage_deg != null ? rep.coverage_deg : rep.coverageDeg,
              softRadius: rep.soft_radius != null ? rep.soft_radius : rep.softRadius,
              outerFill: rep.outer_fill != null ? rep.outer_fill : rep.outerFill,
              solidAngleFrac: rep.solid_angle_frac != null ? rep.solid_angle_frac : rep.solidAngleFrac,
              level: rep.level,
              text: rep.text,
            },
          }),
        );
        // auto-switch to dome tab when coverage arrives
        const domeTab = document.querySelector('.tab[data-tab="dome"]');
        if (domeTab) domeTab.click();
      }
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

  // --- drag & drop files from the desktop onto the canvas (issue #417) ---
  // LibTV-style: a file dropped on empty canvas area becomes an input node
  // at the drop point. Dropping onto an existing node is ignored so it does
  // not swallow a node drag.
  let dropHover = false;

  canvas.addEventListener("dragenter", (evt) => {
    if (!evt.dataTransfer) return;
    evt.preventDefault();
    dropHover = true;
    canvas.style.cursor = "copy";
    setStatus("松开以在此处新建输入节点");
    draw();
  });

  canvas.addEventListener("dragover", (evt) => {
    // preventDefault is what actually permits the drop.
    if (!evt.dataTransfer) return;
    evt.preventDefault();
    if (evt.dataTransfer.dropEffect !== undefined) evt.dataTransfer.dropEffect = "copy";
  });

  canvas.addEventListener("dragleave", (evt) => {
    // Only clear when leaving the canvas element itself (child bubbles).
    if (evt.target === canvas) {
      dropHover = false;
      canvas.style.cursor = "default";
    }
  });

  canvas.addEventListener("drop", async (evt) => {
    evt.preventDefault();
    dropHover = false;
    canvas.style.cursor = "default";
    const files = evt.dataTransfer && evt.dataTransfer.files;
    if (!files || !files.length) return;
    // All files land at the same drop point, staggered slightly so the
    // second one is not hidden behind the first.
    const p = canvasPoint(evt);
    for (let i = 0; i < files.length; i += 1) {
      await uploadToCanvas(files[i], { x: p.x + i * 24, y: p.y + i * 24 });
    }
  });

  function bindUi() {
    document.getElementById("btnDemo").addEventListener("click", () => loadDemo().catch((e) => setStatus(e.message)));
    document.getElementById("btnDual").addEventListener("click", () => loadDualTemplate().catch((e) => setStatus(e.message)));
    document.getElementById("btnEmptyDual").addEventListener("click", () => loadDualTemplate().catch((e) => setStatus(e.message)));
    document.getElementById("btnSave").addEventListener("click", saveProject);
    const btnSaveServer = document.getElementById("btnSaveServer");
    if (btnSaveServer) btnSaveServer.addEventListener("click", () => saveToServer());
    const btnProjects = document.getElementById("btnProjects");
    if (btnProjects) {
      btnProjects.addEventListener("click", () => {
        const t = document.querySelector('.tab[data-tab="projects"]');
        if (t) t.click();
        refreshProjects();
        loadSettings();
      });
    }
    const bSetLoad = document.getElementById("btnSettingsLoad");
    if (bSetLoad) bSetLoad.addEventListener("click", () => loadSettings());
    const bSetSave = document.getElementById("btnSettingsSave");
    if (bSetSave) bSetSave.addEventListener("click", () => saveSettings());
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
    const uploadInput = document.getElementById("uploadInput");
    if (uploadInput) {
      uploadInput.addEventListener("change", () => {
        if (uploadInput.files && uploadInput.files[0]) {
          uploadToCanvas(uploadInput.files[0]);
        }
        uploadInput.value = "";
      });
    }
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

  function mediaUrl(p) {
    if (!p) return "";
    return `/api/media?path=${encodeURIComponent(p)}`;
  }

  /**
   * Upload a File and add the matching input node at ``at`` (canvas coords).
   * Shared by the sidebar picker and the canvas drop handler.
   */
  /**
   * Re-upload into an existing input node (file picker on its card). Keeps the
   * node id and position; only the path + cached metadata change.
   */
  async function reuploadIntoNode(file, node) {
    const type = inputTypeForFile(file.name);
    if (!type || type !== node.type) {
      setStatus(
        `文件类型不匹配：${file.name} 需要 ${(state.nodeTypes[node.type] || {}).label || node.type}`
      );
      return;
    }
    if (!state.online) {
      setStatus("离线无法上传；请先启动 studio.server");
      return;
    }
    setStatus(`上传中 ${file.name}…`);
    try {
      const meta = await uploadFile(file);
      node.params.path = meta.path;
      state.inputMeta = state.inputMeta || {};
      state.inputMeta[node.id] = meta;
      renderInspector();
      draw();
      setStatus(`已更新素材：${meta.kind}`);
    } catch (err) {
      setStatus("上传失败: " + (err.message || err));
    }
  }

  async function uploadToCanvas(file, at) {
    const type = inputTypeForFile(file.name);
    if (!type) {
      setStatus(`不支持的文件类型：${file.name}（仅 png/jpg/jpeg/webp/mp4/mov）`);
      return;
    }
    if (!state.online) {
      setStatus("离线无法上传；请先启动 studio.server");
      return;
    }
    setStatus(`上传中 ${file.name}…`);
    try {
      const meta = await uploadFile(file);
      const p = at || { x: 60 + state.project.nodes.length * 44, y: 80 };
      addInputNodeAt(type, p.x, p.y, meta);
    } catch (err) {
      setStatus("上传失败: " + (err.message || err));
    }
  }

  function fmtDur(sec) {
    if (sec == null) return "";
    const s = Math.round(sec);
    const m = Math.floor(s / 60);
    return m > 0 ? `${m}:${String(s % 60).padStart(2, "0")}` : `${s}s`;
  }

  /**
   * Resolve the picture + badge for an input node so the canvas can paint a
   * thumbnail (images) or poster frame (videos). The image is cached in
   * ``state.thumbCache`` keyed by path so redraws stay synchronous.
   */
  function inputThumb(node) {
    if (!node || !node.type.startsWith("input.")) return null;
    const meta = (state.inputMeta && state.inputMeta[node.id]) || null;
    // Prefer the server's poster frame for videos; fall back to the raw path
    // for images (and videos whose poster failed to render).
    const src = meta
      ? (meta.kind === "video" && meta.poster ? meta.poster : node.params && node.params.path)
      : node.params && node.params.path;
    if (!src) return null;
    const url = mediaUrl(src);
    let img = state.thumbCache && state.thumbCache[url];
    if (!img) {
      img = new Image();
      img.onload = () => draw();
      img.onerror = () => {
        delete state.thumbCache[url];
        draw();
      };
      img.src = url;
      if (!state.thumbCache) state.thumbCache = {};
      state.thumbCache[url] = img;
    }
    let badge = "";
    if (meta && meta.kind === "video") {
      const bits = [];
      if (meta.width && meta.height) bits.push(`${meta.width}x${meta.height}`);
      if (meta.duration != null) bits.push(fmtDur(meta.duration));
      badge = bits.join(" ");
    } else if (meta && meta.kind === "image" && meta.width && meta.height) {
      badge = `${meta.width}x${meta.height}`;
    } else if (!meta) {
      // No server metadata yet (project loaded from disk, not uploaded).
      badge = node.type === "input.video" ? "▶" : "";
    }
    return { img, badge, url };
  }

  /** Upload one dropped file; returns the server's metadata reply. */
  async function uploadFile(file) {
    const form = new FormData();
    form.append("file", file, file.name);
    const res = await fetch("/api/upload", { method: "POST", body: form });
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || JSON.stringify(body);
    } catch (_) {
      /* ignore — non-JSON body */
    }
    if (!res.ok) throw new Error(detail);
    return res.json();
  }

  /** Pick the input node type for a dropped file from its extension. */
  function inputTypeForFile(name) {
    const ext = (name.split(".").pop() || "").toLowerCase();
    if (["png", "jpg", "jpeg", "webp"].includes(ext)) return "input.image";
    if (["mp4", "mov"].includes(ext)) return "input.video";
    return null;
  }

  /** Create an input node at canvas point (x, y) wired to the uploaded file. */
  function addInputNodeAt(type, x, y, meta) {
    const meta0 = state.nodeTypes[type] || { param_schema: [] };
    const params = {};
    (meta0.param_schema || []).forEach((p) => {
      params[p.name] = p.default;
    });
    params.path = meta.path;
    const id = uid("n");
    state.project.nodes.push({ id, type, pos: [x, y], params, muted: false });
    // Stash the server metadata so the canvas can show a thumbnail/poster
    // for this node before the graph is ever run.
    state.inputMeta = state.inputMeta || {};
    state.inputMeta[id] = meta;
    state.selectedId = id;
    updateEmpty();
    renderInspector();
    draw();
    const label = meta0.label || type;
    setStatus(`已添加 ${label}（${meta.kind}）`);
  }

  function renderGallery(gallery) {
    const box = document.getElementById("galleryBox");
    if (!box) return;
    if (!gallery || (!gallery.sheet && !(gallery.shots && gallery.shots.length))) {
      box.classList.add("hidden");
      box.innerHTML = "";
      return;
    }
    box.classList.remove("hidden");
    const shots = (gallery.shots || [])
      .map(
        (s) =>
          `<li><b>${s.id || "?"}</b> ${s.duration || ""}s — ${s.description || ""}` +
          (s.image
            ? `<br/><img class="shot-thumb" src="${mediaUrl(s.image)}" alt="${s.id || ""}" />`
            : "") +
          `<br/><span class="sheet-path">${s.image || ""}</span></li>`,
      )
      .join("");
    const sheetImg = gallery.sheet
      ? `<img class="sheet-thumb" src="${mediaUrl(gallery.sheet)}" alt="contact sheet" />`
      : "";
    box.innerHTML = `
      <div><strong>分镜图廊</strong>（${gallery.count || 0} 镜）</div>
      <div class="sheet-path">联络表：${gallery.sheet || "—"}</div>
      ${sheetImg}
      <ul>${shots}</ul>`;
  }

  async function saveToServer() {
    if (!state.online) {
      setStatus("离线无法存库；请先启动 studio.server");
      return;
    }
    try {
      const res = await api("/api/projects", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          project: toServerProject(state.project),
          project_id: state.project.name || undefined,
        }),
      });
      state.project.name = res.name || state.project.name;
      projectNameEl.textContent = state.project.name;
      setStatus(`已存库 ${res.id} · ${res.nodes} 节点`);
      await refreshProjects();
    } catch (err) {
      setStatus("存库失败: " + err.message);
    }
  }

  async function refreshProjects() {
    const listEl = document.getElementById("projectsList");
    if (!listEl) return;
    if (!state.online) {
      listEl.innerHTML = '<p class="hint">后端未连接</p>';
      return;
    }
    try {
      const items = await api("/api/projects");
      if (!items.length) {
        listEl.innerHTML = '<p class="hint">暂无已保存工程</p>';
        return;
      }
      listEl.innerHTML = "";
      items.forEach((it) => {
        const row = document.createElement("div");
        row.className = "project-row";
        row.innerHTML = `
          <div>
            <strong>${it.name}</strong>
            <div class="hint">${it.id} · ${it.nodes} 节点 · ${it.mtime_iso}</div>
          </div>
          <div class="project-actions">
            <button type="button" data-act="open">打开</button>
            <button type="button" data-act="del">删</button>
          </div>`;
        row.querySelector('[data-act="open"]').addEventListener("click", async () => {
          try {
            const data = await api(`/api/projects/${encodeURIComponent(it.id)}`);
            loadProjectObject(data);
            setStatus("已打开 " + it.id);
          } catch (e) {
            setStatus("打开失败: " + e.message);
          }
        });
        row.querySelector('[data-act="del"]').addEventListener("click", async () => {
          try {
            await api(`/api/projects/${encodeURIComponent(it.id)}`, { method: "DELETE" });
            await refreshProjects();
            setStatus("已删除 " + it.id);
          } catch (e) {
            setStatus("删除失败: " + e.message);
          }
        });
        listEl.appendChild(row);
      });
    } catch (err) {
      listEl.innerHTML = `<p class="hint err">${err.message}</p>`;
    }
    await loadSettings();
  }

  async function loadSettings() {
    if (!state.online) return;
    try {
      const s = await api("/api/settings");
      const set = (id, v) => {
        const el = document.getElementById(id);
        if (el) el.value = v || "";
      };
      set("setLlmUrl", s.litellm_base_url);
      set("setLlmModel", s.litellm_model);
      set("setSnUrl", s.sensenova_base_url);
      set("setSnModel", s.sensenova_model);
      const st = document.getElementById("settingsStatus");
      if (st) {
        st.textContent = s.litellm_api_key_set
          ? `已配置 LiteLLM（key ${s.litellm_api_key_masked}）· model=${s.litellm_model || "—"}`
          : "尚未写入 LiteLLM key；润色节点 provider=mock 仍可用";
      }
    } catch (_) {
      /* ignore */
    }
  }

  async function saveSettings() {
    if (!state.online) {
      setStatus("离线无法保存设置");
      return;
    }
    const val = (id) => {
      const el = document.getElementById(id);
      return el ? el.value.trim() : "";
    };
    const body = {
      litellm_base_url: val("setLlmUrl") || null,
      litellm_model: val("setLlmModel") || null,
      sensenova_base_url: val("setSnUrl") || null,
      sensenova_model: val("setSnModel") || null,
    };
    const key = val("setLlmKey");
    if (key) body.litellm_api_key = key;
    try {
      const s = await api("/api/settings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const st = document.getElementById("settingsStatus");
      if (st) st.textContent = `已保存 · base=${s.litellm_base_url || "—"} model=${s.litellm_model || "—"}`;
      setStatus("供应商设置已保存到本机");
    } catch (err) {
      setStatus("设置保存失败: " + err.message);
    }
  }

  function bindTabs() {
    document.querySelectorAll(".tab").forEach((btn) => {
      btn.addEventListener("click", () => {
        const name = btn.getAttribute("data-tab");
        document.querySelectorAll(".tab").forEach((b) => b.classList.toggle("active", b === btn));
        document.querySelectorAll(".tab-panel").forEach((p) => {
          p.classList.toggle("active", p.id === "tab-" + name);
        });
        // resize dome canvas when its tab becomes visible
        if (name === "dome" && window.StudioDomePreview) {
          window.dispatchEvent(new Event("resize"));
        }
      });
    });
  }

  async function boot() {
    bindTabs();
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
