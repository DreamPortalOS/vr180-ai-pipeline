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
    /** Latest server-side run results keyed by node id, for inline previews. */
    lastRun: null,
    /** Drawer open/closed state — persisted to localStorage. */
    drawerOpen: true,
    drawerHeight: 220,
    /** Ordered shot ids per node id after a drag-reorder (written back to JSON). */
    shotOrder: {},
    /** Checked shot ids per node id. */
    shotChecked: {},
    /** Picker: node id → index into node history (0 = latest). */
    historyIdx: {},
  };

  const LS_DRAWER = "studio.drawer.v1";

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
      // Body box geometry (shared by every branch so a variable declared in
      // one branch can't leak/ReferenceError another — the #417 redraw bug).
      const bx = n.pos[0] + 4;
      const by = n.pos[1] + HEAD_H + 4;
      const bw = NODE_W - 8;
      const bh = NODE_H - HEAD_H - 8;

      // input.text: draw the text payload inline on the card (issue #418).
      // Clicking the body opens an overlaid textarea to edit; blur saves.
      if (n.type === "input.text") {
        ctx.save();
        ctx.beginPath();
        ctx.rect(bx, by, bw, bh);
        ctx.clip();
        ctx.fillStyle = "#0a0e14";
        ctx.fillRect(bx, by, bw, bh);
        const txt = String(n.params && n.params.text != null ? n.params.text : "");
        ctx.fillStyle = txt ? "#cdd9e8" : "#5a6b80";
        ctx.font = "11px 'Segoe UI', 'PingFang SC', sans-serif";
        ctx.textBaseline = "top";
        const lines = txt ? txt.split("\n") : ["（点击输入文本）"];
        let ty = by + 4;
        for (let li = 0; li < lines.length && ty < by + bh - 4; li += 1) {
          const slice = lines[li].slice(0, 26);
          ctx.fillText(slice, bx + 6, ty);
          ty += 13;
        }
        ctx.textBaseline = "alphabetic";
        ctx.restore();
        drawHistoryArrows(n, bx, by, bw);
        drawPortsAndBadge(n, inputs, outputs, null, bx);
        return;
      }

      // image/video nodes (input or output) render a thumbnail or poster frame
      // behind the ports. nodeRunThumb covers input.* uploads AND run outputs.
      const thumb = nodeRunThumb(n);
      if (thumb) {
        ctx.save();
        ctx.beginPath();
        ctx.rect(bx, by, bw, bh);
        ctx.clip();
        ctx.fillStyle = "#0a0e14";
        ctx.fillRect(bx, by, bw, bh);
        if (thumb.img && thumb.img.complete && thumb.img.naturalWidth) {
          // cover-fit, centered
          const r = Math.max(bw / thumb.img.naturalWidth, bh / thumb.img.naturalHeight);
          const dw = thumb.img.naturalWidth * r;
          const dh = thumb.img.naturalHeight * r;
          ctx.drawImage(thumb.img, bx + (bw - dw) / 2, by + (bh - dh) / 2, dw, dh);
        } else if (thumb.placeholder) {
          ctx.fillStyle = "#5a6b80";
          ctx.font = "11px sans-serif";
          ctx.textAlign = "center";
          ctx.fillText(thumb.placeholder, bx + bw / 2, by + bh / 2 + 4);
          ctx.textAlign = "left";
        } else {
          ctx.fillStyle = "#5a6b80";
          ctx.font = "10px sans-serif";
          ctx.fillText("加载缩略图…", bx + 6, by + bh / 2 + 3);
        }
        ctx.restore();
        drawHistoryArrows(n, bx, by, bw);
        drawPortsAndBadge(n, inputs, outputs, thumb.badge, bx);
        return;
      }

      drawHistoryArrows(n, bx, by, bw);
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

  /** Draw the ◀▶ history arrows on a node card when history has ≥2 entries. */
  function drawHistoryArrows(n, bx, by, bw) {
    const rep = state.lastRun;
    const entries = (rep && rep.history && rep.history[n.id]) || [];
    if (entries.length < 2) return;
    ctx.save();
    const y = by + 6;
    for (const [glyph, x] of [["◀", bx + 4], ["▶", bx + bw - 14]]) {
      ctx.fillStyle = "rgba(0,0,0,0.55)";
      ctx.fillRect(x, y - 9, 12, 13);
      ctx.fillStyle = "#e8eef7";
      ctx.font = "9px sans-serif";
      ctx.fillText(glyph, x + 2, y + 1);
    }
    // History counter, so the lead can see 1/3 without reading the status bar.
    const idx = state.historyIdx[n.id];
    const nCur = idx == null ? entries.length : entries.length - idx;
    ctx.fillStyle = "rgba(0,0,0,0.55)";
    ctx.fillRect(bx + bw / 2 - 14, y - 9, 28, 13);
    ctx.fillStyle = "#e8eef7";
    ctx.font = "9px ui-monospace, Consolas, monospace";
    ctx.fillText(`${nCur}/${entries.length}`, bx + bw / 2 - 9, y + 1);
    ctx.restore();
  }

  /** Draw node ports + optional kind badge (factor of the old draw() block). */
  function drawPortsAndBadge(n, inputs, outputs, badge, bx) {
    if (badge) {
      ctx.fillStyle = "rgba(0,0,0,0.62)";
      ctx.fillRect(n.pos[0] + NODE_W - 72, n.pos[1] + NODE_H - 16, 68, 13);
      ctx.fillStyle = "#e8eef7";
      ctx.font = "10px ui-monospace, Consolas, monospace";
      ctx.fillText(badge, n.pos[0] + NODE_W - 68, n.pos[1] + NODE_H - 6);
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
  }

  /** Single global inline textarea for editing an input.text node on its card. */
  function editTextNodeInline(node) {
    if (!node || node.type !== "input.text") return;
    closeTextNodeEditor();
    const stage = document.getElementById("canvasStage");
    if (!stage) return;
    const ta = document.createElement("textarea");
    ta.id = "nodeTextEditor";
    ta.className = "node-text-editor";
    ta.value = String(node.params.text != null ? node.params.text : "");
    ta.placeholder = "输入文本 / prompt，失焦保存…";
    ta.addEventListener("blur", () => {
      node.params.text = ta.value;
      ta.remove();
      draw();
      renderInspector();
      setStatus("已保存文本节点");
    });
    ta.addEventListener("keydown", (evt) => {
      if (evt.key === "Escape") {
        evt.preventDefault();
        ta.blur();
      }
      evt.stopPropagation();
    });
    const rect = canvas.getBoundingClientRect();
    const scale = rect.width / (canvas.width / (window.devicePixelRatio || 1)) || 1;
    const left = rect.left + state.pan.x + node.pos[0] + 4;
    const top = rect.top + state.pan.y + node.pos[1] + HEAD_H + 4;
    ta.style.left = left + "px";
    ta.style.top = top + "px";
    ta.style.width = (NODE_W - 8) * scale + "px";
    ta.style.height = (NODE_H - HEAD_H - 8) * scale + "px";
    stage.appendChild(ta);
    ta.focus();
    ta.select();
  }

  function closeTextNodeEditor() {
    const ta = document.getElementById("nodeTextEditor");
    if (ta) ta.remove();
  }

  /** Click-vs-drag hit test for inline text editing + history arrows. */
  function nodeCardClickTarget(node, p) {
    if (!node) return null;
    const bx = node.pos[0] + 4;
    const by = node.pos[1] + HEAD_H + 4;
    const bw = NODE_W - 8;
    const rep = state.lastRun;
    const entries = (rep && rep.history && rep.history[node.id]) || [];
    if (entries.length >= 2) {
      const ay = by + 6 - 9;
      const ah = 13;
      if (p.y >= ay && p.y <= ay + ah) {
        if (p.x >= bx + 2 && p.x <= bx + 18) return { kind: "history", delta: -1 };
        if (p.x >= bx + bw - 18 && p.x <= bx + bw - 2) return { kind: "history", delta: 1 };
      }
    }
    if (node.type === "input.text") return { kind: "text" };
    return null;
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
    renderDrawer();
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
    state.lastRun = null;
    state.historyIdx = {};
    updateEmpty();
    renderInspector();
    renderDrawer();
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
      state.lastRun = report;
      runOutEl.textContent = JSON.stringify(report, null, 2);
      // Bottom drawer (#418) replaces the right-rail gallery.
      renderDrawer();
      // Inline node-card previews from this run's outputs.
      loadNodeRunPreviews(report);
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
      // History arrows (◀▶) on the card take a single click and don't drag.
      const click = nodeCardClickTarget(node, p);
      if (click && click.kind === "history") {
        state.selectedId = node.id;
        stepNodeHistory(node.id, click.delta);
        renderInspector();
        renderDrawer();
        return;
      }
      state.selectedId = node.id;
      state.drag = {
        mode: "node",
        id: node.id,
        ox: p.x - node.pos[0],
        oy: p.y - node.pos[1],
        startX: node.pos[0],
        startY: node.pos[1],
        moved: false,
      };
      renderInspector();
      renderDrawer();
      draw();
      return;
    }
    state.selectedId = null;
    state.drag = { mode: "pan", sx: evt.clientX, sy: evt.clientY, px: state.pan.x, py: state.pan.y };
    renderInspector();
    renderDrawer();
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
      const nx = p.x - state.drag.ox;
      const ny = p.y - state.drag.oy;
      if (Math.abs(nx - state.drag.startX) > 2 || Math.abs(ny - state.drag.startY) > 2) {
        state.drag.moved = true;
      }
      node.pos = [nx, ny];
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
    if (state.drag && state.drag.mode === "node" && !state.drag.moved) {
      const node = nodeById(state.drag.id);
      if (node && node.type === "input.text") {
        // A plain click (no drag) on a text node opens inline editing.
        editTextNodeInline(node);
      }
    }
    state.drag = null;
  });

  canvas.addEventListener("dblclick", (evt) => {
    const p = canvasPoint(evt);
    const node = hitNode(p.x, p.y);
    if (node) {
      state.selectedId = node.id;
      renderInspector();
      renderDrawer();
      if (node.type === "input.text") {
        editTextNodeInline(node);
        return;
      }
      // Double-click an image/video node card to toggle playback inline.
      toggleNodePlayback(node, p.x, p.y);
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
    bindDrawer();
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
      state.lastRun = null;
      state.historyIdx = {};
      closeTextNodeEditor();
      closeInlineVideo();
      projectNameEl.textContent = "untitled";
      updateEmpty();
      renderInspector();
      renderDrawer();
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
      if (evt.key.toLowerCase() === "b" && !evt.ctrlKey && !evt.metaKey && !evt.altKey) {
        // Don't steal the key when the user is typing in a text field.
        const ae = document.activeElement;
        const typing = ae && (ae.tagName === "INPUT" || ae.tagName === "TEXTAREA" || ae.isContentEditable);
        if (!typing) {
          evt.preventDefault();
          state.drawerOpen = !state.drawerOpen;
          applyDrawerState();
          saveDrawerState();
          if (state.drawerOpen) renderDrawer();
        }
      }
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
    // Read the body exactly once: a Response stream cannot be consumed twice,
    // and parsing it a second time on success used to throw "body stream already
    // read", so no input node was ever created (found in lead browser QA).
    let body = null;
    try {
      body = await res.json();
    } catch (_) {
      /* non-JSON body */
    }
    if (!res.ok) throw new Error((body && body.detail) || res.statusText);
    if (!body) throw new Error("upload returned no JSON metadata");
    return body;
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

  /**
   * Collect the shots array for a storyboard-class node from the latest run
   * result (or a passed run). Returns [] when the node has no shots yet so
   * the drawer renders grey "not generated" cards only when the node carries
   * shot descriptions (issue #418).
   */
  function shotsForNode(nodeId, report) {
    const rep = report || state.lastRun;
    if (!rep || !rep.results) return [];
    const res = rep.results[nodeId];
    if (!res || !res.outputs) return [];
    const outs = res.outputs;
    // batch_stills / review: { stills: { shots: [...] } }
    const stills = outs.stills;
    if (stills && Array.isArray(stills.shots)) return stills.shots;
    // shot_list / polish: { shots: { shots: [...] } }
    const shotsWrap = outs.shots;
    if (shotsWrap && Array.isArray(shotsWrap.shots)) return shotsWrap.shots;
    // storyboard node: { storyboard: {...} } (single shot)
    const sb = outs.storyboard;
    if (sb && (sb.description || sb.prompt)) return [sb];
    return [];
  }

  /**
   * Pull shot descriptions from the *current params* of a shot_list node so
   * the drawer can list grey "未生成" cards even before the first run (the
   * card read-out only happens after /api/run). Falls back to [].
   */
  function shotsFromParams(node) {
    if (!node) return [];
    if (node.type === "script.shot_list") {
      const lines = String(node.params.shot_texts || "").split("\n").map((s) => s.trim()).filter(Boolean);
      const durs = String(node.params.shot_durations || "").split(",").map((s) => s.trim());
      const motions = String(node.params.motions || "").split(",").map((s) => s.trim());
      return lines.map((line, i) => ({
        id: `shot_${String(i + 1).padStart(2, "0")}`,
        index: i,
        description: line,
        motion: motions[i] || "dolly_in",
        duration: durs[i] ? Number(durs[i]) : 4,
      }));
    }
    if (node.type === "script.storyboard") {
      const sb = {
        id: "shot_01",
        index: 0,
        description: node.params.prompt || node.params.title || "",
        duration: Number(node.params.duration) || 5,
        aspect_ratio: node.params.aspect_ratio || "1:1",
      };
      return [sb];
    }
    return [];
  }

  /** Merge params-derived shot shells with run results so grey cards appear pre-run. */
  function drawerShots(nodeId) {
    const node = nodeById(nodeId);
    if (!node) return [];
    const runShots = shotsForNode(nodeId);
    const paramShots = shotsFromParams(node);
    // Prefer run shots (have image), but fall back to param shells so the
    // drawer lists every intended shot before the first run.
    if (runShots.length) return runShots;
    return paramShots;
  }

  /** Shot-order helpers — accept string ids, ignore null/undefined. */
  function shotOrderFor(nodeId) {
    return (state.shotOrder[nodeId] || []).filter((x) => x != null).map(String);
  }

  function reorderShots(shots, order) {
    if (!order || !order.length) return shots;
    const rank = new Map(order.map((id, i) => [id, i]));
    const known = shots.filter((s) => rank.has(String(s.id)));
    const missing = shots.filter((s) => !rank.has(String(s.id)));
    known.sort((a, b) => rank.get(String(a.id)) - rank.get(String(b.id)));
    return known.concat(missing);
  }

  function saveDrawerState() {
    try {
      localStorage.setItem(
        LS_DRAWER,
        JSON.stringify({
          open: state.drawerOpen,
          height: state.drawerHeight,
          shotOrder: state.shotOrder,
          shotChecked: state.shotChecked,
        }),
      );
    } catch (_) {
      /* localStorage may be disabled (file:// privacy); non-fatal. */
    }
  }

  function loadDrawerState() {
    try {
      const raw = localStorage.getItem(LS_DRAWER);
      if (!raw) return;
      const data = JSON.parse(raw);
      if (typeof data.open === "boolean") state.drawerOpen = data.open;
      if (Number.isFinite(data.height)) state.drawerHeight = Math.max(120, Math.min(520, data.height));
      if (data.shotOrder && typeof data.shotOrder === "object") state.shotOrder = data.shotOrder;
      if (data.shotChecked && typeof data.shotChecked === "object") state.shotChecked = data.shotChecked;
    } catch (_) {
      /* corrupt entry — ignore and use defaults. */
    }
  }

  const drawerEl = document.getElementById("shotDrawer");
  const drawerBodyEl = document.getElementById("drawerBody");
  const drawerCardsEl = document.getElementById("drawerCards");
  const drawerTableWrap = document.getElementById("drawerTableWrap");
  const drawerTableEl = document.getElementById("drawerTable");
  const drawerEmptyEl = document.getElementById("drawerEmpty");
  const drawerToggleBtn = document.getElementById("drawerToggle");
  const drawerCountEl = document.getElementById("drawerCount");
  const drawerTitleEl = document.getElementById("drawerTitle");

  let drawerCardsView = "cards"; // cards | table

  function applyDrawerState() {
    if (!drawerEl) return;
    drawerEl.classList.toggle("collapsed", !state.drawerOpen);
    drawerEl.style.setProperty("--drawer-h", state.drawerHeight + "px");
  }

  function bindDrawer() {
    loadDrawerState();
    applyDrawerState();
    const handle = document.getElementById("drawerHandle");
    if (handle) {
      handle.addEventListener("dblclick", () => {
        state.drawerOpen = !state.drawerOpen;
        applyDrawerState();
        saveDrawerState();
        if (state.drawerOpen) renderDrawer();
      });
    }
    if (drawerToggleBtn) {
      drawerToggleBtn.addEventListener("click", () => {
        state.drawerOpen = !state.drawerOpen;
        applyDrawerState();
        saveDrawerState();
        if (state.drawerOpen) renderDrawer();
      });
    }
    // Segment toggles for card/table view.
    const segCards = document.getElementById("drawerViewCards");
    const segTable = document.getElementById("drawerViewTable");
    if (segCards) {
      segCards.addEventListener("click", () => {
        drawerCardsView = "cards";
        segCards.classList.add("active");
        if (segTable) segTable.classList.remove("active");
        renderDrawer();
      });
    }
    if (segTable) {
      segTable.addEventListener("click", () => {
        drawerCardsView = "table";
        segTable.classList.add("active");
        if (segCards) segCards.classList.remove("active");
        renderDrawer();
      });
    }
    // Drag-to-resize the drawer handle.
    bindDrawerResize(handle);
  }

  function bindDrawerResize(handle) {
    if (!handle) return;
    let resizing = null;
    handle.addEventListener("mousedown", (evt) => {
      if (evt.button !== 0) return;
      // Avoid starting a resize on a button click inside the handle.
      if (evt.target.closest("button")) return;
      resizing = { sy: evt.clientY, sh: state.drawerHeight };
      document.body.style.cursor = "ns-resize";
      evt.preventDefault();
    });
    window.addEventListener("mousemove", (evt) => {
      if (!resizing) return;
      const delta = resizing.sy - evt.clientY; // drag up = taller
      state.drawerHeight = Math.max(120, Math.min(520, resizing.sh + delta));
      drawerEl.style.setProperty("--drawer-h", state.drawerHeight + "px");
    });
    window.addEventListener("mouseup", () => {
      if (!resizing) return;
      resizing = null;
      document.body.style.cursor = "";
      saveDrawerState();
    });
  }

  /**
   * Render the bottom drawer for the currently selected storyboard node.
   * Hidden/empty for non-storyboard selections (the drawer collapses to its
   * empty hint so the canvas stays usable).
   */
  function renderDrawer() {
    if (!drawerEl || !state.drawerOpen) return;
    const node = nodeById(state.selectedId);
    const isSb =
      node &&
      [
        "script.storyboard",
        "script.shot_list",
        "text.polish_shots",
        "image.batch_stills",
        "checkpoint.review",
      ].includes(node.type);
    let shots = [];
    let title = "分镜";
    let scope = null; // for shot-order/checkbox persistence
    if (node && isSb) {
      shots = drawerShots(node.id);
      title = (state.nodeTypes[node.type] || {}).label || node.type;
      scope = node.id;
    } else if (state.lastRun && state.lastRun.gallery && state.lastRun.gallery.shots) {
      // After a run with no storyboard node selected, surface the gallery so
      // the lead sees every shot card immediately ("运行生产模板后…显示全部镜头卡").
      shots = state.lastRun.gallery.shots;
      title = "运行结果分镜";
      scope = "__run_gallery__";
    }
    if (!scope) {
      drawerEmptyEl.classList.remove("hidden");
      drawerCardsEl.classList.add("hidden");
      if (drawerTableWrap) drawerTableWrap.classList.add("hidden");
      drawerTitleEl.textContent = "分镜";
      drawerCountEl.textContent = "选中脚本/分镜节点后显示镜头";
      return;
    }
    const order = shotOrderFor(scope);
    if (order.length) shots = reorderShots(shots, order);
    drawerTitleEl.textContent = title;
    const hasImg = shots.filter((s) => s.image).length;
    drawerCountEl.textContent = `${shots.length} 镜 · ${hasImg} 已生成`;
    if (!shots.length) {
      drawerEmptyEl.classList.remove("hidden");
      drawerCardsEl.classList.add("hidden");
      if (drawerTableWrap) drawerTableWrap.classList.add("hidden");
      return;
    }
    drawerEmptyEl.classList.add("hidden");
    if (drawerCardsView === "cards") {
      if (drawerTableWrap) drawerTableWrap.classList.add("hidden");
      drawerCardsEl.classList.remove("hidden");
      renderDrawerCards(scope, shots);
    } else {
      drawerCardsEl.classList.add("hidden");
      if (drawerTableWrap) drawerTableWrap.classList.remove("hidden");
      renderDrawerTable(scope, shots);
    }
  }

  function shotStatusClass(shot) {
    if (shot.error) return "err";
    if (shot.image) return "ok";
    return "";
  }

  function checkedShotIds(nodeId, shots) {
    const set = state.shotChecked[nodeId] || [];
    if (set.length) return new Set(set.map(String));
    // default: all checked
    return new Set(shots.map((s) => String(s.id)));
  }

  function toggleShotChecked(nodeId, shotId, shots) {
    const all = checkedShotIds(nodeId, shots);
    if (all.has(String(shotId))) all.delete(String(shotId));
    else all.add(String(shotId));
    state.shotChecked[nodeId] = shots.map((s) => String(s.id)).filter((id) => all.has(id));
    saveDrawerState();
    renderDrawer();
  }

  function renderDrawerCards(nodeId, shots) {
    drawerCardsEl.innerHTML = "";
    const checked = checkedShotIds(nodeId, shots);
    shots.forEach((shot, idx) => {
      const card = document.createElement("div");
      card.className = "shot-card";
      card.draggable = true;
      card.dataset.id = String(shot.id);
      card.dataset.idx = String(idx);

      const thumb = document.createElement("div");
      thumb.className = "thumb" + (shot.image ? "" : " placeholder");
      if (shot.image) {
        const img = document.createElement("img");
        img.src = mediaUrl(shot.image);
        img.alt = shot.id || "";
        img.loading = "lazy";
        img.onerror = () => {
          thumb.classList.add("placeholder");
          thumb.innerHTML = "";
        };
        thumb.appendChild(img);
      }
      const dot = document.createElement("span");
      dot.className = "status-dot " + shotStatusClass(shot);
      thumb.appendChild(dot);
      const pick = document.createElement("span");
      pick.className = "pick" + (checked.has(String(shot.id)) ? " on" : "");
      pick.title = "勾选参与生成";
      pick.addEventListener("click", (e) => {
        e.stopPropagation();
        toggleShotChecked(nodeId, shot.id, shots);
      });
      thumb.appendChild(pick);
      card.appendChild(thumb);

      const meta = document.createElement("div");
      meta.className = "meta";
      const no = document.createElement("div");
      no.innerHTML = `<span class="no">${shot.id || idx + 1}</span> <span class="dur">${fmtDur(shot.duration)}</span>`;
      meta.appendChild(no);
      const desc = document.createElement("div");
      desc.className = "desc";
      desc.textContent = shot.description || shot.motion || "";
      desc.title = shot.description || "";
      meta.appendChild(desc);
      card.appendChild(meta);

      bindDragReorder(card, nodeId, shots);
      drawerCardsEl.appendChild(card);
    });
  }

  function renderDrawerTable(nodeId, shots) {
    if (!drawerTableEl) return;
    drawerTableEl.innerHTML = "";
    const thead = document.createElement("thead");
    thead.innerHTML =
      "<tr><th>#</th><th>状态</th><th>镜头</th><th class='desc'>描述</th><th>时长</th><th>运动</th></tr>";
    drawerTableEl.appendChild(thead);
    const checked = checkedShotIds(nodeId, shots);
    const tbody = document.createElement("tbody");
    shots.forEach((shot, idx) => {
      const tr = document.createElement("tr");
      tr.draggable = true;
      tr.dataset.id = String(shot.id);
      tr.dataset.idx = String(idx);
      const cells = [
        `<td>${shot.id || idx + 1}</td>`,
        `<td><span class="status-dot ${shotStatusClass(shot)}"></span></td>`,
        `<td><span class="pick ${checked.has(String(shot.id)) ? "on" : ""}" title="勾选"></span></td>`,
        `<td class="desc">${escapeHtml(shot.description || "")}</td>`,
        `<td>${fmtDur(shot.duration)}</td>`,
        `<td>${escapeHtml(shot.motion || "")}</td>`,
      ];
      tr.innerHTML = cells.join("");
      const pick = tr.querySelector(".pick");
      if (pick) pick.addEventListener("click", (e) => {
        e.stopPropagation();
        toggleShotChecked(nodeId, shot.id, shots);
      });
      bindDragReorder(tr, nodeId, shots);
      tbody.appendChild(tr);
    });
    drawerTableEl.appendChild(tbody);
  }

  function escapeHtml(s) {
    return String(s == null ? "" : s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  /** HTML5 drag-reorder for drawer cards/rows; writes back to state.shotOrder. */
  function bindDragReorder(el, nodeId, shots) {
    let dragId = null;
    el.addEventListener("dragstart", (evt) => {
      dragId = el.dataset.id;
      el.classList.add("dragging");
      if (evt.dataTransfer) {
        evt.dataTransfer.effectAllowed = "move";
        try {
          evt.dataTransfer.setData("text/plain", dragId);
        } catch (_) {
          /* some browsers reject setData under file://; rely on closure var */
        }
      }
    });
    el.addEventListener("dragend", () => {
      el.classList.remove("dragging");
      drawerCardsEl.querySelectorAll(".drag-over").forEach((n) => n.classList.remove("drag-over"));
    });
    el.addEventListener("dragover", (evt) => {
      evt.preventDefault();
      if (evt.dataTransfer) evt.dataTransfer.dropEffect = "move";
    });
    el.addEventListener("dragenter", () => el.classList.add("drag-over"));
    el.addEventListener("dragleave", () => el.classList.remove("drag-over"));
    el.addEventListener("drop", (evt) => {
      evt.preventDefault();
      el.classList.remove("drag-over");
      const targetId = el.dataset.id;
      if (!dragId || dragId === targetId) return;
      const order = shots.map((s) => String(s.id));
      const from = order.indexOf(dragId);
      const to = order.indexOf(targetId);
      if (from < 0 || to < 0) return;
      const [moved] = order.splice(from, 1);
      order.splice(to, 0, moved);
      state.shotOrder[nodeId] = order;
      saveDrawerState();
      renderDrawer();
    });
  }

  /**
   * Resolve the inline preview artefact (image path or video poster) for an
   * output-producing node after a run. Returns null for nodes without a
   * previewable result. Supports the ◀▶ history switcher: when ``idx`` is
   * set, it picks that entry from the node's history (0 = newest).
   */
  function nodeRunThumb(node, idx) {
    if (!node) return null;
    // input.* nodes keep using their upload-derived thumbnail (inputThumb).
    if (node.type.startsWith("input.")) return inputThumb(node);
    const rep = state.lastRun;
    if (!rep || !rep.results) return null;
    // When the ◀▶ switcher is engaged, read from the history entries.
    let outputs = null;
    const histIdx = state.historyIdx[node.id];
    if (histIdx != null) {
      const entries = (rep.history && rep.history[node.id]) || [];
      const entry = entries[entries.length - 1 - histIdx];
      if (entry) outputs = entry.outputs;
    }
    if (!outputs) outputs = (rep.results[node.id] || {}).outputs || {};
    // image.batch_stills → first still's image (or contact sheet).
    const stills = outputs.stills;
    if (stills && Array.isArray(stills.shots)) {
      const withImg = stills.shots.filter((s) => s.image);
      if (withImg.length) return { img: cachedImage(mediaUrl(withImg[0].image)), badge: `${withImg.length}/${stills.shots.length}` };
      if (stills.shots.length) return { placeholder: "未生成", badge: `0/${stills.shots.length}` };
    }
    // generic image output port
    for (const key of ["image", "sheet", "poster"]) {
      if (outputs[key]) return { img: cachedImage(mediaUrl(outputs[key])), badge: "" };
    }
    // video outputs: outputs.video may be a path or { path, poster }
    const v = outputs.video;
    if (v) {
      const path = typeof v === "string" ? v : v.path || v.url;
      const poster = typeof v === "object" ? v.poster : null;
      if (poster) return { img: cachedImage(mediaUrl(poster)), video: path, badge: "▶" };
      if (path) return { placeholder: "▶ 视频", video: path, badge: "▶" };
    }
    // convert/export nodes with manifest
    const path = outputs.path;
    if (typeof path === "string" && /\.(png|jpe?g|webp|mp4|mov)$/i.test(path)) {
      if (/\.(mp4|mov)$/i.test(path)) return { placeholder: "▶ 视频", video: path, badge: "▶" };
      return { img: cachedImage(mediaUrl(path)), badge: "" };
    }
    return null;
  }

  /** Cache + load an Image for a media path; returns the <img> (maybe not yet
   *  loaded) and triggers a redraw on load. */
  function cachedImage(url) {
    if (!url) return null;
    if (!state.thumbCache) state.thumbCache = {};
    let img = state.thumbCache[url];
    if (!img) {
      img = new Image();
      img.onload = () => draw();
      img.onerror = () => {
        delete state.thumbCache[url];
        draw();
      };
      img.src = url;
      state.thumbCache[url] = img;
    }
    return img;
  }

  /** After a run, force a redraw so node cards pick up new thumbnails. */
  function loadNodeRunPreviews(report) {
    draw();
  }

  /**
   * ◀▶ history switcher for the selected node. Steps the history index by
   * ``delta`` (-1 = ◀ older, +1 = ▶ newer) and clamps within the node's
   * recorded history length. Bounded so it can't run past the newest entry.
   */
  function stepNodeHistory(nodeId, delta) {
    const rep = state.lastRun;
    const entries = (rep && rep.history && rep.history[nodeId]) || [];
    if (entries.length < 2) {
      setStatus("该节点暂无更多历史结果");
      return;
    }
    let cur = state.historyIdx[nodeId] == null ? 0 : state.historyIdx[nodeId];
    cur = Math.max(0, Math.min(entries.length - 1, cur + delta));
    state.historyIdx[nodeId] = cur;
    draw();
    const human = entries.length - cur; // 1 = newest
    setStatus(`历史结果 ${human}/${entries.length}`);
  }

  /** Floating inline video player for double-click on a video-output node. */
  function toggleNodePlayback(node, x, y) {
    if (!node) return;
    // input.video nodes already have a poster; output video nodes need a path.
    let videoPath = null;
    const thumb = nodeRunThumb(node);
    if (thumb && thumb.video) videoPath = thumb.video;
    if (node.type === "input.video" && node.params && node.params.path) {
      videoPath = node.params.path;
    }
    if (!videoPath) return;
    openInlineVideo(node.id, videoPath);
  }

  function openInlineVideo(nodeId, path) {
    closeInlineVideo();
    const v = document.createElement("video");
    v.id = "inlineVideo_" + nodeId;
    v.src = mediaUrl(path);
    v.controls = true;
    v.autoplay = true;
    v.style.cssText =
      "position:fixed;right:342px;bottom:36px;width:min(42vw,520px);max-height:46vh;border-radius:10px;border:1px solid var(--border);box-shadow:var(--shadow);background:#000;z-index:20";
    const close = document.createElement("button");
    close.type = "button";
    close.textContent = "✕";
    close.style.cssText =
      "position:fixed;right:342px;bottom:calc(36px + min(46vh,420px) + 4px);z-index:21";
    close.addEventListener("click", closeInlineVideo);
    close.id = "inlineVideoClose";
    document.body.appendChild(v);
    document.body.appendChild(close);
    setStatus("双击节点已打开内联播放器");
  }

  function closeInlineVideo() {
    const v = document.querySelector("[id^='inlineVideo_']");
    const c = document.getElementById("inlineVideoClose");
    if (v) v.remove();
    if (c) c.remove();
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
