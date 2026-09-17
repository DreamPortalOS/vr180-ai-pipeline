/* Immersive Node Studio — offline canvas (no CDN). */
(() => {
  const NODE_W = 180;
  const NODE_H = 78;
  const PORT_R = 6;

  const state = {
    project: { version: 1, name: "untitled", nodes: [], edges: [], settings: {} },
    nodeTypes: {},
    selectedId: null,
    drag: null,
    linking: null,
    status: {},
    pan: { x: 0, y: 0 },
  };

  const canvas = document.getElementById("canvas");
  const ctx = canvas.getContext("2d");
  const paletteEl = document.getElementById("nodePalette");
  const inspectorEl = document.getElementById("inspectorBody");
  const runOutEl = document.getElementById("runOut");
  const statusBar = document.getElementById("statusBar");
  const projectNameEl = document.getElementById("projectName");
  const fileInput = document.getElementById("fileInput");

  function uid(prefix) {
    return `${prefix}_${Math.random().toString(36).slice(2, 9)}`;
  }

  function setStatus(msg) {
    statusBar.textContent = msg;
  }

  function resizeCanvas() {
    const wrap = canvas.parentElement;
    canvas.width = wrap.clientWidth;
    canvas.height = wrap.clientHeight;
    draw();
  }

  function projectToPayload() {
    return {
      project: JSON.parse(JSON.stringify(state.project)),
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

  async function loadCatalog() {
    const types = await api("/api/node-types");
    state.nodeTypes = Object.fromEntries(types.map((t) => [t.type, t]));
    paletteEl.innerHTML = "";
    types.forEach((t) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.textContent = `${t.label}`;
      btn.title = t.type;
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
    const offset = state.project.nodes.length * 24;
    state.project.nodes.push({
      id,
      type,
      pos: [80 + offset, 80 + offset],
      params,
      muted: false,
    });
    state.selectedId = id;
    setStatus(`added ${type}`);
    renderInspector();
    draw();
  }

  function nodeById(id) {
    return state.project.nodes.find((n) => n.id === id);
  }

  function portPositions(node) {
    const meta = state.nodeTypes[node.type] || { inputs: [], outputs: [] };
    const inputs = (meta.inputs || []).map((p, i) => ({
      ...p,
      x: node.pos[0],
      y: node.pos[1] + 28 + i * 18,
      side: "in",
    }));
    const outputs = (meta.outputs || []).map((p, i) => ({
      ...p,
      x: node.pos[0] + NODE_W,
      y: node.pos[1] + 28 + i * 18,
      side: "out",
    }));
    return { inputs, outputs };
  }

  function hitPort(x, y) {
    for (const node of state.project.nodes) {
      const { inputs, outputs } = portPositions(node);
      for (const p of [...inputs, ...outputs]) {
        const dx = x - p.x;
        const dy = y - p.y;
        if (dx * dx + dy * dy <= (PORT_R + 4) ** 2) {
          return { node, port: p };
        }
      }
    }
    return null;
  }

  function hitNode(x, y) {
    for (let i = state.project.nodes.length - 1; i >= 0; i -= 1) {
      const n = state.project.nodes[i];
      if (x >= n.pos[0] && x <= n.pos[0] + NODE_W && y >= n.pos[1] && y <= n.pos[1] + NODE_H) {
        return n;
      }
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

  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.save();
    ctx.translate(state.pan.x, state.pan.y);

    // edges
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

    // nodes
    state.project.nodes.forEach((n) => {
      const meta = state.nodeTypes[n.type] || { label: n.type, inputs: [], outputs: [] };
      const selected = n.id === state.selectedId;
      const st = state.status[n.id] || "idle";
      const headColor =
        st === "ok" ? "#1f6b4a" : st === "error" ? "#7a2e35" : st === "running" ? "#7a5a1d" : "#2c3a4f";
      ctx.fillStyle = selected ? "#2a3b55" : "#243041";
      ctx.strokeStyle = selected ? "#3d9cf0" : "#2a3545";
      ctx.lineWidth = selected ? 2 : 1;
      roundRect(n.pos[0], n.pos[1], NODE_W, NODE_H, 8);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = headColor;
      roundRect(n.pos[0], n.pos[1], NODE_W, 22, 8, true);
      ctx.fill();
      ctx.fillStyle = "#e7eef7";
      ctx.font = "12px sans-serif";
      ctx.fillText(meta.label || n.type, n.pos[0] + 10, n.pos[1] + 15);

      const { inputs, outputs } = portPositions(n);
      inputs.forEach((p) => drawPort(p.x, p.y, p.type, true, n.id, p.name));
      outputs.forEach((p) => drawPort(p.x, p.y, p.type, false, n.id, p.name));
    });

    ctx.restore();
  }

  function roundRect(x, y, w, h, r, topOnly) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, topOnly ? 0 : r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, topOnly ? 0 : r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  function drawWire(x1, y1, x2, y2, color) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    const cx = (x1 + x2) / 2;
    ctx.bezierCurveTo(cx, y1, cx, y2, x2, y2);
    ctx.stroke();
  }

  function drawPort(x, y, type, isInput, nodeId, portName) {
    const colors = {
      text: "#9aa7b8",
      video: "#b388ff",
      image: "#4dd0c8",
      json: "#e6b450",
      number: "#5b9dff",
      any: "#cfd8e3",
    };
    ctx.fillStyle = colors[type] || "#cfd8e3";
    ctx.beginPath();
    ctx.arc(x, y, PORT_R, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "#0f1419";
    ctx.stroke();
    ctx.fillStyle = "#8b9bb0";
    ctx.font = "10px sans-serif";
    if (isInput) {
      ctx.fillText(portName, x + 10, y + 3);
    } else {
      const label = portName;
      const w = ctx.measureText(label).width;
      ctx.fillText(label, x - 10 - w, y + 3);
    }
    void nodeId;
  }

  function renderInspector() {
    const node = nodeById(state.selectedId);
    if (!node) {
      inspectorEl.textContent = "选中一个节点以编辑参数";
      return;
    }
    const meta = state.nodeTypes[node.type] || { label: node.type, param_schema: [] };
    inspectorEl.innerHTML = "";
    const title = document.createElement("div");
    title.textContent = `${meta.label} (${node.type})`;
    title.style.fontWeight = "600";
    title.style.marginBottom = "6px";
    inspectorEl.appendChild(title);

    const idLabel = document.createElement("label");
    idLabel.textContent = "节点 ID";
    const idInput = document.createElement("input");
    idInput.value = node.id;
    idInput.disabled = true;
    idLabel.appendChild(idInput);
    inspectorEl.appendChild(idLabel);

    (meta.param_schema || []).forEach((p) => {
      const label = document.createElement("label");
      label.textContent = p.label || p.name;
      const field = document.createElement(p.type === "string" && String(p.default || "").length > 40 ? "textarea" : "input");
      if (field.tagName === "INPUT") {
        field.type = p.type === "number" ? "number" : "text";
      }
      field.value = node.params[p.name] ?? p.default ?? "";
      field.addEventListener("change", () => {
        const raw = field.value;
        node.params[p.name] = p.type === "number" ? Number(raw) : raw;
        draw();
      });
      label.appendChild(field);
      inspectorEl.appendChild(label);
    });

    const del = document.createElement("button");
    del.type = "button";
    del.textContent = "删除节点";
    del.style.marginTop = "12px";
    del.addEventListener("click", () => {
      state.project.nodes = state.project.nodes.filter((n) => n.id !== node.id);
      state.project.edges = state.project.edges.filter((e) => e.from_node !== node.id && e.to_node !== node.id);
      state.selectedId = null;
      renderInspector();
      draw();
    });
    inspectorEl.appendChild(del);
  }

  canvas.addEventListener("mousedown", (evt) => {
    const p = canvasPoint(evt);
    const port = hitPort(p.x, p.y);
    if (port && port.port.side === "out") {
      state.linking = { fromNode: port.node.id, fromPort: port.port.name, x1: port.port.x, y1: port.port.y, x2: p.x, y2: p.y };
      draw();
      return;
    }
    const node = hitNode(p.x, p.y);
    if (node) {
      state.selectedId = node.id;
      state.drag = { id: node.id, ox: p.x - node.pos[0], oy: p.y - node.pos[1] };
      renderInspector();
      draw();
      return;
    }
    state.selectedId = null;
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
    if (state.drag) {
      const node = nodeById(state.drag.id);
      if (node) {
        node.pos = [p.x - state.drag.ox, p.y - state.drag.oy];
        draw();
      }
    }
  });

  window.addEventListener("mouseup", (evt) => {
    if (state.linking) {
      const p = canvasPoint(evt);
      const target = hitPort(p.x, p.y);
      if (target && target.port.side === "in" && target.node.id !== state.linking.fromNode) {
        // one input, one edge
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
        setStatus("connected");
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

  function saveProject() {
    const blob = new Blob([JSON.stringify(state.project, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${state.project.name || "project"}.studio.json`;
    a.click();
    URL.revokeObjectURL(a.href);
    setStatus("saved JSON download");
  }

  function loadProjectFile(file) {
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const data = JSON.parse(String(reader.result));
        state.project = data;
        projectNameEl.textContent = data.name || "untitled";
        state.selectedId = null;
        renderInspector();
        draw();
        setStatus("project loaded");
      } catch (err) {
        setStatus(`load failed: ${err.message}`);
      }
    };
    reader.readAsText(file);
  }

  async function loadDemo() {
    const data = await api("/api/demo-project");
    state.project = data;
    projectNameEl.textContent = data.name;
    state.selectedId = null;
    state.status = {};
    renderInspector();
    draw();
    setStatus("demo project loaded");
  }

  async function runProject() {
    setStatus("running…");
    runOutEl.textContent = "…";
    try {
      const report = await api("/api/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(projectToPayload()),
      });
      state.status = Object.fromEntries(
        Object.entries(report.results || {}).map(([id, r]) => [id, r.status]),
      );
      runOutEl.textContent = JSON.stringify(report, null, 2);
      setStatus("run ok");
      draw();
    } catch (err) {
      state.status = {};
      runOutEl.textContent = String(err.message || err);
      setStatus("run failed");
      draw();
    }
  }

  document.getElementById("btnDemo").addEventListener("click", () => loadDemo().catch((e) => setStatus(e.message)));
  document.getElementById("btnSave").addEventListener("click", saveProject);
  document.getElementById("btnLoad").addEventListener("click", () => fileInput.click());
  document.getElementById("btnRun").addEventListener("click", () => runProject());
  fileInput.addEventListener("change", () => {
    if (fileInput.files && fileInput.files[0]) loadProjectFile(fileInput.files[0]);
    fileInput.value = "";
  });

  window.addEventListener("keydown", (evt) => {
    if ((evt.ctrlKey || evt.metaKey) && evt.key === "Enter") {
      evt.preventDefault();
      runProject();
    }
    if ((evt.ctrlKey || evt.metaKey) && evt.key.toLowerCase() === "s") {
      evt.preventDefault();
      saveProject();
    }
    if (evt.key === "Delete" && state.selectedId) {
      const id = state.selectedId;
      state.project.nodes = state.project.nodes.filter((n) => n.id !== id);
      state.project.edges = state.project.edges.filter((e) => e.from_node !== id && e.to_node !== id);
      state.selectedId = null;
      renderInspector();
      draw();
    }
  });

  window.addEventListener("resize", resizeCanvas);

  loadCatalog()
    .then(() => loadDemo())
    .catch((err) => setStatus(err.message))
    .finally(resizeCanvas);
})();
