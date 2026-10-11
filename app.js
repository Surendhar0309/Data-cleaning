/* =============================================================================
   Universal AI-Powered Data Cleaning Platform — Frontend Logic
   ============================================================================= */

let API = localStorage.getItem('dcai_server') || 'http://localhost:7860';
let SESSION_ID = null;
let PROFILE = null;
let QUALITY = null;
let RECOMMENDATIONS = [];

/* ── Helpers ─────────────────────────────────────────────────────── */
const $ = id => document.getElementById(id);
const show = el => el && el.classList.remove('hidden');
const hide = el => el && el.classList.add('hidden');

function showLoading(msg = 'Processing...') {
  $('loadingText').textContent = msg;
  show($('loadingOverlay'));
}
function hideLoading() { hide($('loadingOverlay')); }

async function api(path, opts = {}) {
  const url = `${API}${path}`;
  const resp = await fetch(url, opts);
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: resp.statusText }));
    throw new Error(err.detail || 'Request failed');
  }
  return resp.json();
}

/* ── Server Connection ───────────────────────────────────────────── */
async function checkServer() {
  const dot = $('statusDot');
  dot.className = 'status-dot connecting';
  try {
    const r = await fetch(`${API}/`, { signal: AbortSignal.timeout(5000) });
    if (r.ok) { dot.className = 'status-dot online'; return true; }
  } catch (e) { /* fall through */ }
  dot.className = 'status-dot offline';
  return false;
}

function connectServer() {
  const val = $('serverUrl').value.trim().replace(/\/$/, '');
  if (val) { API = val; localStorage.setItem('dcai_server', API); }
  checkServer();
}

/* ── File Upload ─────────────────────────────────────────────────── */
async function uploadFile(file) {
  showLoading('Uploading & analyzing dataset...');
  const form = new FormData();
  form.append('file', file);

  try {
    const data = await api('/upload', { method: 'POST', body: form });
    SESSION_ID = data.session_id;
    PROFILE = data.profile;
    QUALITY = data.quality_score;
    RECOMMENDATIONS = data.recommendations || [];

    $('sessionId').textContent = SESSION_ID;
    show($('sessionBadge'));
    hide($('uploadSection'));
    show($('dashboard'));

    renderDashboard();
  } catch (e) {
    alert('Upload failed: ' + e.message);
  } finally {
    hideLoading();
  }
}

/* ── Dashboard Rendering ─────────────────────────────────────────── */
function renderDashboard() {
  renderMetrics();
  renderColumnTypes();
  renderQualityBreakdown();
  renderMissingHeatmap();
  renderCorrelationMatrix();
  renderMissingTable();
  renderDuplicates();
  renderRecommendations();
  populateSelects();
  loadDataPreview();
}

function renderMetrics() {
  if (!PROFILE || !QUALITY) return;
  $('metRows').textContent = PROFILE.shape.rows.toLocaleString();
  $('metCols').textContent = PROFILE.shape.columns;

  const totalCells = PROFILE.shape.rows * PROFILE.shape.columns;
  const missingCells = Object.values(PROFILE.missing_values).reduce((s, v) => s + v.count, 0);
  const missingPct = totalCells > 0 ? ((missingCells / totalCells) * 100).toFixed(1) : '0';
  $('metMissing').textContent = missingPct + '%';
  $('metDuplicates').textContent = PROFILE.duplicate_rows;
  $('metQuality').textContent = QUALITY.overall_score + '/100';
}

function renderColumnTypes() {
  const ct = PROFILE.column_types;
  const container = $('columnTypes');
  let html = '';

  const groups = [
    { key: 'numerical', label: 'Numerical', cls: 'tag-num' },
    { key: 'categorical', label: 'Categorical', cls: 'tag-cat' },
    { key: 'datetime', label: 'Datetime', cls: 'tag-date' },
    { key: 'text', label: 'Text', cls: 'tag-text' },
  ];

  for (const g of groups) {
    const cols = ct[g.key] || [];
    if (cols.length === 0) continue;
    html += `<div class="type-group">
      <div class="type-group-label">${g.label} (${cols.length})</div>
      <div class="type-tags">${cols.map(c => `<span class="tag ${g.cls}">${c}</span>`).join('')}</div>
    </div>`;
  }
  container.innerHTML = html;
}

function renderQualityBreakdown() {
  const comp = QUALITY.components;
  const container = $('qualityBreakdown');
  let html = '';

  const items = [
    { key: 'completeness', label: 'Completeness', color: 'var(--success)' },
    { key: 'uniqueness', label: 'Uniqueness', color: 'var(--accent)' },
    { key: 'outlier_free', label: 'Outlier-Free', color: 'var(--purple)' },
    { key: 'consistency', label: 'Consistency', color: 'var(--warning)' },
  ];

  for (const item of items) {
    const score = comp[item.key]?.score || 0;
    html += `<div class="quality-item">
      <span class="quality-item-label">${item.label}</span>
      <div class="quality-bar"><div class="quality-bar-fill" style="width:${score}%;background:${item.color}"></div></div>
      <span class="quality-item-score">${score}</span>
    </div>`;
  }
  container.innerHTML = html;
}

function renderMissingHeatmap() {
  const missing = PROFILE.missing_values;
  const cols = Object.keys(missing);
  const container = $('missingHeatmap');

  if (cols.length === 0) { container.innerHTML = '<p style="color:var(--text-muted)">No columns</p>'; return; }

  let html = `<div class="heatmap-grid" style="grid-template-columns:repeat(${Math.min(cols.length, 20)},1fr)">`;
  for (const col of cols.slice(0, 20)) {
    const pct = missing[col].percentage;
    const intensity = Math.min(pct / 50, 1);
    const color = pct === 0
      ? 'rgba(16,185,129,0.3)'
      : `rgba(239,68,68,${0.2 + intensity * 0.6})`;
    html += `<div>
      <div class="heatmap-cell" style="background:${color}" title="${col}: ${pct}%"></div>
      <div class="heatmap-label" title="${col}">${col.length > 8 ? col.slice(0, 7) + '...' : col}</div>
    </div>`;
  }
  html += '</div>';
  container.innerHTML = html;
}

function renderCorrelationMatrix() {
  const corr = PROFILE.correlations;
  const container = $('correlationMatrix');
  const cols = Object.keys(corr);

  if (cols.length === 0) {
    container.innerHTML = '<p style="color:var(--text-muted)">Not enough numerical columns for correlation</p>';
    return;
  }

  const displayCols = cols.slice(0, 10);
  let html = '<table class="corr-table"><tr><th></th>';
  for (const c of displayCols) html += `<th title="${c}">${c.length > 6 ? c.slice(0, 5) + '..' : c}</th>`;
  html += '</tr>';

  for (const row of displayCols) {
    html += `<tr><th title="${row}">${row.length > 6 ? row.slice(0, 5) + '..' : row}</th>`;
    for (const col of displayCols) {
      const val = corr[row]?.[col] ?? 0;
      const abs = Math.abs(val);
      const bg = val > 0
        ? `rgba(16,185,129,${abs * 0.5})`
        : `rgba(239,68,68,${abs * 0.5})`;
      html += `<td style="background:${bg}">${val.toFixed(2)}</td>`;
    }
    html += '</tr>';
  }
  html += '</table>';
  container.innerHTML = html;
}

function renderMissingTable() {
  const missing = PROFILE.missing_values;
  const container = $('missingTable');
  const entries = Object.entries(missing).filter(([_, v]) => v.count > 0).sort((a, b) => b[1].percentage - a[1].percentage);

  if (entries.length === 0) {
    container.innerHTML = '<p style="color:var(--success);font-size:13px">No missing values detected!</p>';
    return;
  }

  let html = '';
  for (const [col, info] of entries) {
    html += `<div class="missing-row">
      <span class="missing-col-name">${col}</span>
      <div class="missing-bar-wrap"><div class="missing-bar-fill" style="width:${Math.min(info.percentage, 100)}%"></div></div>
      <span class="missing-pct">${info.percentage}%</span>
    </div>`;
  }
  container.innerHTML = html;
}

async function renderDuplicates() {
  const container = $('duplicateInfo');
  try {
    const data = await api(`/duplicates/${SESSION_ID}`);
    const d = data.duplicates;
    container.innerHTML = `
      <p><span class="dup-stat">${d.duplicate_count}</span> duplicate rows found</p>
      <p style="color:var(--text-muted);font-size:12px;margin-top:4px">${d.duplicate_percentage}% of total rows</p>
    `;
  } catch (e) {
    container.innerHTML = '<p style="color:var(--text-muted)">Could not load duplicate info</p>';
  }
}

function renderRecommendations() {
  const container = $('recommendationsList');
  if (!RECOMMENDATIONS || RECOMMENDATIONS.length === 0) {
    container.innerHTML = '<p style="color:var(--text-muted)">No recommendations at this time.</p>';
    return;
  }

  let html = '';
  for (const rec of RECOMMENDATIONS.slice(0, 20)) {
    html += `<div class="rec-item">
      <span class="rec-priority ${rec.priority}">${rec.priority}</span>
      <div class="rec-content">
        <div class="rec-column">${rec.column} <span style="color:var(--text-muted);font-size:10px">[${rec.category}]</span></div>
        <div class="rec-reason">${rec.reason}</div>
        <div class="rec-suggestion">Suggestion: ${rec.suggestion}</div>
      </div>
    </div>`;
  }
  container.innerHTML = html;
}

function populateSelects() {
  const allCols = PROFILE.shape ? Object.keys(PROFILE.missing_values) : [];
  const numCols = PROFILE.column_types?.numerical || [];
  const catCols = PROFILE.column_types?.categorical || [];
  const dateCols = PROFILE.column_types?.datetime || [];

  populateSelect('fillColumn', allCols);
  populateSelect('outlierColumn', numCols);
  populateSelect('encodeColumn', catCols);
  populateSelect('scaleColumn', numCols);
  populateSelect('datetimeColumn', dateCols);
}

function populateSelect(id, options) {
  const el = $(id);
  if (!el) return;
  el.innerHTML = '<option value="">Select column...</option>' +
    options.map(o => `<option value="${o}">${o}</option>`).join('');
}

async function loadDataPreview() {
  try {
    const data = await api(`/data/${SESSION_ID}?rows=50`);
    renderDataTable(data.data, data.columns);
  } catch (e) { /* ignore */ }
}

function renderDataTable(rows, columns) {
  const container = $('dataTable');
  if (!rows || rows.length === 0) {
    container.innerHTML = '<p style="color:var(--text-muted)">No data</p>';
    return;
  }

  let html = '<table class="data-table"><thead><tr>';
  for (const col of columns) html += `<th>${col}</th>`;
  html += '</tr></thead><tbody>';

  for (const row of rows) {
    html += '<tr>';
    for (const col of columns) {
      const val = row[col];
      if (val === null || val === 'null' || val === undefined) {
        html += '<td class="cell-null">null</td>';
      } else {
        const display = String(val).length > 30 ? String(val).slice(0, 28) + '..' : String(val);
        html += `<td>${display}</td>`;
      }
    }
    html += '</tr>';
  }
  html += '</tbody></table>';
  container.innerHTML = html;
}

/* ── Actions ─────────────────────────────────────────────────────── */
async function fillMissing() {
  const col = $('fillColumn').value;
  const method = $('fillMethod').value;
  if (!col) { alert('Select a column'); return; }
  showLoading('Filling missing values...');
  try {
    const data = await api('/clean/fill-missing', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: SESSION_ID, column: col, method })
    });
    QUALITY = data.quality_score;
    await refreshProfile();
  } catch (e) { alert(e.message); }
  finally { hideLoading(); }
}

async function dropMissingRows() {
  const col = $('fillColumn').value;
  showLoading('Dropping rows...');
  try {
    const cols = col ? [col] : null;
    const data = await api('/clean/drop-missing', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: SESSION_ID, columns: cols, axis: 'rows' })
    });
    QUALITY = data.quality_score;
    await refreshProfile();
  } catch (e) { alert(e.message); }
  finally { hideLoading(); }
}

async function removeDuplicates() {
  showLoading('Removing duplicates...');
  try {
    const data = await api('/clean/remove-duplicates', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: SESSION_ID })
    });
    QUALITY = data.quality_score;
    await refreshProfile();
  } catch (e) { alert(e.message); }
  finally { hideLoading(); }
}

async function detectOutliers() {
  const col = $('outlierColumn').value;
  const method = $('outlierMethod').value;
  if (!col) { alert('Select a column'); return; }
  showLoading('Detecting outliers...');
  try {
    const data = await api('/clean/outliers', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: SESSION_ID, column: col, method, action: 'detect' })
    });
    renderOutlierResults(data.outliers);
  } catch (e) { alert(e.message); }
  finally { hideLoading(); }
}

async function removeOutliers() {
  const col = $('outlierColumn').value;
  const method = $('outlierMethod').value;
  if (!col) { alert('Select a column'); return; }
  showLoading('Removing outliers...');
  try {
    await api('/clean/outliers', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: SESSION_ID, column: col, method, action: 'remove' })
    });
    await refreshProfile();
  } catch (e) { alert(e.message); }
  finally { hideLoading(); }
}

async function capOutliers() {
  const col = $('outlierColumn').value;
  const method = $('outlierMethod').value;
  if (!col) { alert('Select a column'); return; }
  showLoading('Capping outliers...');
  try {
    await api('/clean/outliers', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: SESSION_ID, column: col, method, action: 'cap' })
    });
    await refreshProfile();
  } catch (e) { alert(e.message); }
  finally { hideLoading(); }
}

function renderOutlierResults(info) {
  const container = $('outlierResults');
  container.innerHTML = `<div class="outlier-stat">
    <div class="outlier-stat-item"><div class="outlier-stat-label">Method</div><div class="outlier-stat-value">${info.method}</div></div>
    <div class="outlier-stat-item"><div class="outlier-stat-label">Outliers Found</div><div class="outlier-stat-value" style="color:var(--danger)">${info.outlier_count}</div></div>
    <div class="outlier-stat-item"><div class="outlier-stat-label">Lower Bound</div><div class="outlier-stat-value">${info.lower_bound ?? info.mean ?? '-'}</div></div>
    <div class="outlier-stat-item"><div class="outlier-stat-label">Upper Bound</div><div class="outlier-stat-value">${info.upper_bound ?? info.threshold ?? '-'}</div></div>
  </div>`;
}

async function applyEncoding() {
  const col = $('encodeColumn').value;
  const method = $('encodeMethod').value;
  if (!col) { alert('Select a column'); return; }
  showLoading('Applying encoding...');
  try {
    const data = await api('/feature/encode', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: SESSION_ID, column: col, method })
    });
    $('encodeResult').textContent = JSON.stringify(data.info, null, 2);
    await refreshProfile();
  } catch (e) { alert(e.message); }
  finally { hideLoading(); }
}

async function applyScaling() {
  const col = $('scaleColumn').value;
  const method = $('scaleMethod').value;
  if (!col) { alert('Select a column'); return; }
  showLoading('Applying scaling...');
  try {
    const data = await api('/feature/scale', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: SESSION_ID, column: col, method })
    });
    $('scaleResult').textContent = JSON.stringify(data.info, null, 2);
    await refreshProfile();
  } catch (e) { alert(e.message); }
  finally { hideLoading(); }
}

async function extractDatetime() {
  const col = $('datetimeColumn').value;
  if (!col) { alert('Select a column'); return; }
  showLoading('Extracting datetime features...');
  try {
    const data = await api('/feature/datetime', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: SESSION_ID, column: col })
    });
    $('datetimeResult').textContent = JSON.stringify(data.info, null, 2);
    await refreshProfile();
  } catch (e) { alert(e.message); }
  finally { hideLoading(); }
}

async function refreshProfile() {
  try {
    const data = await api(`/profile/${SESSION_ID}`);
    PROFILE = data.profile;
    QUALITY = data.quality_score;
    renderDashboard();
  } catch (e) { /* ignore */ }
}

function downloadFile(format) {
  if (!SESSION_ID) return;
  window.open(`${API}/download/${SESSION_ID}?format=${format}`, '_blank');
}

/* ── Tab Switching ───────────────────────────────────────────────── */
function switchTab(tab) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b.dataset.tab === tab));
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.toggle('active', p.id === `tab-${tab}`));
}

/* ── Event Listeners ─────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  // Server
  $('serverUrl').value = API;
  $('btnConnect').addEventListener('click', connectServer);
  checkServer();

  // Upload
  $('btnBrowse').addEventListener('click', () => $('fileInput').click());
  $('fileInput').addEventListener('change', e => {
    if (e.target.files[0]) uploadFile(e.target.files[0]);
  });

  // Drag & drop
  const dz = $('dropZone');
  dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('drag-over'); });
  dz.addEventListener('dragleave', () => dz.classList.remove('drag-over'));
  dz.addEventListener('drop', e => {
    e.preventDefault(); dz.classList.remove('drag-over');
    if (e.dataTransfer.files[0]) uploadFile(e.dataTransfer.files[0]);
  });

  // Tabs
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => switchTab(btn.dataset.tab));
  });

  // Cleaning actions
  $('btnFill').addEventListener('click', fillMissing);
  $('btnDropRows').addEventListener('click', dropMissingRows);
  $('btnRemoveDups').addEventListener('click', removeDuplicates);

  // Outliers
  $('btnDetectOutliers').addEventListener('click', detectOutliers);
  $('btnRemoveOutliers').addEventListener('click', removeOutliers);
  $('btnCapOutliers').addEventListener('click', capOutliers);

  // Feature Engineering
  $('btnEncode').addEventListener('click', applyEncoding);
  $('btnScale').addEventListener('click', applyScaling);
  $('btnDatetime').addEventListener('click', extractDatetime);

  // Downloads
  $('btnDownloadCSV').addEventListener('click', () => downloadFile('csv'));
  $('btnDownloadJSON').addEventListener('click', () => downloadFile('json'));
});
