/**
 * VR180 Studio — Frontend Application
 * Single-page app for video conversion, task management, and result download.
 */

const API_BASE = '/api/v1';

// ===== STATE =====
const state = {
  currentView: 'convert',
  userId: 'default-user',
  tasks: [],
  results: [],
  quota: null,
};

// ===== DOM REFS =====
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

// ===== NAVIGATION =====
$$('.nav-link, [data-view]').forEach(link => {
  link.addEventListener('click', (e) => {
    e.preventDefault();
    const view = link.dataset.view;
    if (view) switchView(view);
  });
});

function switchView(view) {
  state.currentView = view;
  $$('.view').forEach(v => v.classList.remove('active'));
  $(`#view${capitalize(view)}`).classList.add('active');
  $$('.nav-link').forEach(n => {
    n.classList.toggle('active', n.dataset.view === view);
  });
  if (view === 'tasks') loadTasks();
  if (view === 'results') loadResults();
}

function capitalize(s) { return s.charAt(0).toUpperCase() + s.slice(1); }

// ===== FILE DROP =====
const fileDrop = $('#fileDrop');
const fileInput = $('#inputFile');
const fileDropContent = $('#fileDropContent');

fileDrop.addEventListener('click', () => fileInput.click());

fileDrop.addEventListener('dragover', (e) => {
  e.preventDefault();
  fileDrop.classList.add('dragover');
});

fileDrop.addEventListener('dragleave', () => {
  fileDrop.classList.remove('dragover');
});

fileDrop.addEventListener('drop', (e) => {
  e.preventDefault();
  fileDrop.classList.remove('dragover');
  if (e.dataTransfer.files.length) {
    fileInput.files = e.dataTransfer.files;
    showFileSelected(e.dataTransfer.files[0]);
  }
});

fileInput.addEventListener('change', () => {
  if (fileInput.files.length) {
    showFileSelected(fileInput.files[0]);
  }
});

function showFileSelected(file) {
  fileDrop.classList.add('has-file');
  const sizeMB = (file.size / (1024 * 1024)).toFixed(1);
  fileDropContent.innerHTML = `
    <div class="file-name">✅ ${escapeHtml(file.name)}</div>
    <div class="file-size">${sizeMB} MB</div>
  `;
}

// ===== CONVERT FORM =====
$('#convertForm').addEventListener('submit', async (e) => {
  e.preventDefault();

  const file = fileInput.files[0];
  if (!file) {
    toast('Please select a video file', 'error');
    return;
  }

  const btn = $('#submitBtn');
  const btnText = btn.querySelector('.btn-text');
  const btnSpinner = btn.querySelector('.btn-spinner');
  btn.disabled = true;
  btnText.textContent = 'Uploading...';
  btnSpinner.classList.remove('hidden');

  try {
    // Create task
    const outputFormat = $('#outputFormat').value;
    const resolution = $('#resolution').value;
    const codec = $('#codec').value;
    const upscale = $('#upscale').checked;
    const injectMetadata = $('#injectMetadata').checked;

    const formData = new FormData();
    formData.append('file', file);
    formData.append('output_format', outputFormat);
    formData.append('resolution', resolution);
    formData.append('codec', codec);
    formData.append('upscale', String(upscale));
    formData.append('inject_metadata', String(injectMetadata));

    const res = await fetch(`${API_BASE}/tasks`, {
      method: 'POST',
      headers: { 'X-User-Id': state.userId },
      body: formData,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const task = await res.json();
    toast(`Task created: ${task.task_id}`, 'success');
    resetForm();
    switchView('tasks');
    loadTasks();
  } catch (err) {
    toast(`Error: ${err.message}`, 'error');
  } finally {
    btn.disabled = false;
    btnText.textContent = '🚀 Start Conversion';
    btnSpinner.classList.add('hidden');
  }
});

function resetForm() {
  fileInput.value = '';
  fileDrop.classList.remove('has-file');
  fileDropContent.innerHTML = `
    <span class="file-icon">📁</span>
    <p>Drop a video file here or <strong>click to browse</strong></p>
    <p class="file-hint">MP4, MOV, AVI, MKV up to 2GB</p>
  `;
}

// ===== TASKS =====
$('#refreshTasks').addEventListener('click', loadTasks);

async function loadTasks() {
  try {
    const res = await fetch(`${API_BASE}/tasks?limit=50`, {
      headers: { 'X-User-Id': state.userId },
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    state.tasks = data.tasks || data;
    renderTasks();
  } catch (err) {
    toast(`Failed to load tasks: ${err.message}`, 'error');
  }
}

function renderTasks() {
  const list = $('#tasksList');
  const empty = $('#tasksEmpty');

  if (!state.tasks.length) {
    list.innerHTML = '';
    list.appendChild(empty);
    empty.style.display = '';
    return;
  }

  empty.style.display = 'none';
  list.innerHTML = state.tasks.map(t => `
    <div class="task-card" data-task-id="${t.task_id}">
      <div class="task-info">
        <h3>${escapeHtml(t.input_path.split('/').pop())}</h3>
        <div class="task-meta">${t.task_id} · ${formatTime(t.created_at)}</div>
      </div>
      <span class="status-badge status-${t.status}">${t.status}</span>
      <div class="progress-bar">
        <div class="progress-fill" style="width: ${t.progress || 0}%"></div>
      </div>
    </div>
  `).join('');

  list.querySelectorAll('.task-card').forEach(card => {
    card.addEventListener('click', () => {
      const id = card.dataset.taskId;
      const task = state.tasks.find(t => t.task_id === id);
      if (task) showTaskModal(task);
    });
  });
}

// ===== TASK MODAL =====
function showTaskModal(task) {
  const modal = $('#taskModal');
  const content = $('#modalContent');

  content.innerHTML = `
    <h3>Task Details</h3>
    <div class="detail-row"><span class="detail-label">Task ID</span><span class="detail-value">${task.task_id}</span></div>
    <div class="detail-row"><span class="detail-label">Status</span><span class="status-badge status-${task.status}">${task.status}</span></div>
    <div class="detail-row"><span class="detail-label">Input</span><span class="detail-value">${escapeHtml(task.input_path)}</span></div>
    <div class="detail-row"><span class="detail-label">Output</span><span class="detail-value">${escapeHtml(task.output_path || '—')}</span></div>
    <div class="detail-row"><span class="detail-label">Format</span><span class="detail-value">${task.output_format || 'equirectangular'}</span></div>
    <div class="detail-row"><span class="detail-label">Codec</span><span class="detail-value">${task.codec || 'h265'}</span></div>
    <div class="detail-row"><span class="detail-label">Resolution</span><span class="detail-value">${task.resolution || '4k'}</span></div>
    <div class="detail-row"><span class="detail-label">Progress</span><span class="detail-value">${task.progress || 0}%</span></div>
    <div class="detail-row"><span class="detail-label">Created</span><span class="detail-value">${formatTime(task.created_at)}</span></div>
    <div class="detail-row"><span class="detail-label">Updated</span><span class="detail-value">${formatTime(task.updated_at)}</span></div>
    ${task.error ? `<div class="detail-row"><span class="detail-label">Error</span><span class="detail-value" style="color:var(--error)">${escapeHtml(task.error)}</span></div>` : ''}
    <div class="modal-actions">
      ${task.status === 'queued' || task.status === 'processing' ? `<button class="btn-action danger" onclick="cancelTask('${task.task_id}')">Cancel</button>` : ''}
      ${task.status === 'completed' ? `<a class="btn-action" href="${API_BASE}/tasks/${task.task_id}/download" target="_blank">Download</a>` : ''}
      <button class="btn-action danger" onclick="deleteTask('${task.task_id}')">Delete</button>
    </div>
  `;

  modal.classList.remove('hidden');
}

$('#modalClose').addEventListener('click', () => $('#taskModal').classList.add('hidden'));
$('#taskModal').addEventListener('click', (e) => {
  if (e.target === e.currentTarget) $('#taskModal').classList.add('hidden');
});

// ===== RESULTS =====
$('#refreshResults').addEventListener('click', loadResults);

async function loadResults() {
  try {
    const res = await fetch(`${API_BASE}/results?limit=50`, {
      headers: { 'X-User-Id': state.userId },
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    state.results = data.results || data;
    renderResults();
  } catch (err) {
    // Results endpoint may not exist yet — fall back to completed tasks
    if (state.tasks.length === 0) await loadTasks();
    const completed = state.tasks.filter(t => t.status === 'completed');
    state.results = completed.map(t => ({
      task_id: t.task_id,
      filename: t.output_path?.split('/').pop() || t.task_id,
      output_path: t.output_path,
      created_at: t.created_at,
    }));
    renderResults();
  }
}

function renderResults() {
  const grid = $('#resultsGrid');
  const empty = $('#resultsEmpty');

  if (!state.results.length) {
    grid.innerHTML = '';
    grid.appendChild(empty);
    empty.style.display = '';
    return;
  }

  empty.style.display = 'none';
  grid.innerHTML = state.results.map(r => `
    <div class="result-card">
      <h3>${escapeHtml(r.filename || r.task_id)}</h3>
      <div class="result-details">
        <span>🆔 ${r.task_id}</span>
        <span>📅 ${formatTime(r.created_at)}</span>
        ${r.file_size_bytes ? `<span>💾 ${formatBytes(r.file_size_bytes)}</span>` : ''}
      </div>
      <div class="result-actions">
        <a class="btn-action" href="${API_BASE}/tasks/${r.task_id}/download" target="_blank">⬇ Download</a>
        <button class="btn-action danger" onclick="deleteResult('${r.task_id}')">🗑</button>
      </div>
    </div>
  `).join('');
}

// ===== ACTIONS =====
async function cancelTask(taskId) {
  try {
    const res = await fetch(`${API_BASE}/tasks/${taskId}/cancel`, {
      method: 'POST',
      headers: { 'X-User-Id': state.userId },
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    toast('Task cancelled', 'info');
    $('#taskModal').classList.add('hidden');
    loadTasks();
  } catch (err) {
    toast(`Cancel failed: ${err.message}`, 'error');
  }
}

async function deleteTask(taskId) {
  if (!confirm('Delete this task permanently?')) return;
  try {
    const res = await fetch(`${API_BASE}/tasks/${taskId}`, {
      method: 'DELETE',
      headers: { 'X-User-Id': state.userId },
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    toast('Task deleted', 'info');
    $('#taskModal').classList.add('hidden');
    loadTasks();
  } catch (err) {
    toast(`Delete failed: ${err.message}`, 'error');
  }
}

async function deleteResult(taskId) {
  if (!confirm('Delete this result permanently?')) return;
  try {
    const res = await fetch(`${API_BASE}/results/${taskId}`, {
      method: 'DELETE',
      headers: { 'X-User-Id': state.userId },
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    toast('Result deleted', 'info');
    loadResults();
  } catch (err) {
    toast(`Delete failed: ${err.message}`, 'error');
  }
}

// ===== QUOTA =====
async function loadQuota() {
  try {
    const res = await fetch(`${API_BASE}/quota`, {
      headers: { 'X-User-Id': state.userId },
    });
    if (!res.ok) return;
    state.quota = await res.json();
    const q = state.quota;
    const badge = $('#quotaBadge');
    if (q.unlimited) {
      $('#quotaText').textContent = 'Unlimited';
      badge.style.borderColor = 'var(--success)';
    } else {
      $('#quotaText').textContent = `${q.used}/${q.limit}`;
      if (q.remaining === 0) badge.style.borderColor = 'var(--error)';
      else if (q.remaining <= 1) badge.style.borderColor = 'var(--warning)';
    }
  } catch {
    // Quota endpoint may not be registered
  }
}

// ===== UTILITIES =====
function toast(message, type = 'info') {
  const container = $('#toastContainer');
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.textContent = message;
  container.appendChild(el);
  setTimeout(() => {
    el.style.opacity = '0';
    el.style.transform = 'translateX(20px)';
    setTimeout(() => el.remove(), 300);
  }, 4000);
}

function escapeHtml(s) {
  if (!s) return '';
  const div = document.createElement('div');
  div.textContent = s;
  return div.innerHTML;
}

function formatTime(iso) {
  if (!iso) return '—';
  try {
    const d = new Date(iso);
    return d.toLocaleString();
  } catch {
    return iso;
  }
}

function formatBytes(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  if (bytes < 1024 * 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  return (bytes / (1024 * 1024 * 1024)).toFixed(2) + ' GB';
}

// Make action functions globally accessible
window.cancelTask = cancelTask;
window.deleteTask = deleteTask;
window.deleteResult = deleteResult;

// ===== INIT =====
loadQuota();
