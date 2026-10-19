(() => {
  'use strict';

  const $ = (id) => document.getElementById(id);
  const DEFAULTS = {
    canny_low: 40, canny_high: 120, blur_kernel: 3,
    contour_epsilon_ratio: 0.004, silhouette_morph_kernel: 5,
    min_component_area_ratio: 0.01, max_component_area_ratio: 0.85,
    angle_tolerance_deg: 3, draw_edges: false, draw_angle_labels: true,
  };
  const CONTROL_MAP = [
    ['canny-low', 'canny_low', ''], ['canny-high', 'canny_high', ''],
    ['blur-kernel', 'blur_kernel', ''], ['contour-epsilon', 'contour_epsilon_ratio', ''],
    ['morph-kernel', 'silhouette_morph_kernel', ' px'], ['min-area-ratio', 'min_component_area_ratio', ''],
    ['angle-tolerance-deg', 'angle_tolerance_deg', '°'],
  ];
  let latestStatus = null;
  let latestData = null;
  let latestComparison = null;
  let toastTimer = null;
  let settingsTimer = null;
  let pollTimer = null;
  let lastRenderedAngleSignature = '';

  async function requestJson(url, options = {}) {
    const response = await fetch(url, {
      headers: options.body instanceof FormData ? undefined : { 'Content-Type': 'application/json', ...(options.headers || {}) },
      ...options,
    });
    let payload;
    try { payload = await response.json(); } catch (_) { payload = {}; }
    if (!response.ok) {
      const detail = payload.detail || payload.message || `Request failed (${response.status})`;
      throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
    }
    return payload;
  }

  function showToast(message, type = 'info') {
    const toast = $('toast');
    toast.textContent = message;
    toast.className = `toast visible${type === 'error' ? ' error' : ''}`;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { toast.className = 'toast'; }, 3600);
  }

  function setButtonBusy(button, busy, busyText = 'Working…') {
    if (!button) return;
    if (busy) {
      button.dataset.originalText = button.textContent;
      button.textContent = busyText;
      button.disabled = true;
      button.style.opacity = '0.72';
    } else {
      button.textContent = button.dataset.originalText || button.textContent;
      button.disabled = false;
      button.style.opacity = '';
    }
  }

  function escapeHtml(value) {
    return String(value ?? '').replace(/[&<>"']/g, (char) => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
    }[char]));
  }

  function formatNumber(value, digits = 2) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return '—';
    return Number(value).toFixed(digits);
  }

  function setSourceFields() {
    const value = $('source-type').value;
    $('video-fields').classList.toggle('hidden', value !== 'video');
    $('arena-fields').classList.toggle('hidden', value !== 'arena');
    $('webcam-fields').classList.toggle('hidden', value !== 'webcam');
  }

  function sourcePayload() {
    return {
      source_type: $('source-type').value,
      video_path: $('video-path').value.trim(),
      webcam_index: Number.parseInt($('webcam-index').value || '0', 10),
      arena_serial: $('arena-serial').value.trim(),
      loop_video: $('loop-video').checked,
    };
  }

  async function startSource() {
    const button = $('start-btn');
    setButtonBusy(button, true, 'Connecting…');
    try {
      await requestJson('/api/start', { method: 'POST', body: JSON.stringify(sourcePayload()) });
      showToast('Source start requested. Waiting for the first processed frame.');
      $('source-message').textContent = 'Connecting to image source…';
      $('source-message').classList.remove('error');
    } catch (error) {
      showToast(error.message, 'error');
      $('source-message').textContent = error.message;
      $('source-message').classList.add('error');
    } finally {
      setButtonBusy(button, false);
    }
  }

  async function stopSource() {
    try {
      await requestJson('/api/stop', { method: 'POST' });
      showToast('Capture stopped.');
    } catch (error) {
      showToast(error.message, 'error');
    }
  }

  function readSettingsFromControls() {
    const result = {};
    for (const [controlId, key] of CONTROL_MAP) result[key] = Number($(controlId).value);
    result.draw_edges = $('draw-edges').checked;
    result.draw_angle_labels = $('draw-angle-labels').checked;
    return result;
  }

  function updateSettingOutputs(settings = readSettingsFromControls()) {
    for (const [controlId, key, suffix] of CONTROL_MAP) {
      const output = $(`${controlId}-out`);
      if (!output) continue;
      const value = settings[key];
      const digits = key.includes('angle_deg') ? 1 : (key.includes('ratio') ? 3 : 0);
      output.textContent = `${Number(value).toFixed(digits)}${suffix}`;
    }
    $('tolerance-summary').textContent = `±${Number(settings.angle_tolerance_deg).toFixed(1)}°`;
  }

  async function sendSettings() {
    updateSettingOutputs();
    const settings = readSettingsFromControls();
    // Canny thresholds must remain ordered, including while sliders are being dragged.
    if (settings.canny_high <= settings.canny_low) {
      settings.canny_high = settings.canny_low + 1;
      $('canny-high').value = String(settings.canny_high);
    }
    try {
      await requestJson('/api/settings', { method: 'POST', body: JSON.stringify(settings) });
      updateSettingOutputs(settings);
    } catch (error) {
      showToast(error.message, 'error');
    }
  }

  function debounceSettings() {
    updateSettingOutputs();
    clearTimeout(settingsTimer);
    settingsTimer = setTimeout(sendSettings, 160);
  }

  function applySettings(settings) {
    for (const [controlId, key] of CONTROL_MAP) {
      if (settings[key] !== undefined) $(controlId).value = String(settings[key]);
    }
    $('draw-edges').checked = settings.draw_edges !== false;
    $('draw-angle-labels').checked = settings.draw_angle_labels !== false;
    updateSettingOutputs(settings);
  }

  async function resetSettings() {
    applySettings(DEFAULTS);
    await sendSettings();
    showToast('Detection settings reset.');
  }

  async function saveTemplate() {
    const button = $('save-template-btn');
    setButtonBusy(button, true, 'Saving reference…');
    try {
      const result = await requestJson('/api/template/save', {
        method: 'POST', body: JSON.stringify({ name: $('template-name').value.trim() || 'Reference hardware' }),
      });
      showToast(`Corner template saved: ${result.name} (${result.corner_count} corners, ${result.edge_count} edges).`);
      await refreshStatus();
    } catch (error) {
      showToast(error.message, 'error');
    } finally {
      setButtonBusy(button, false);
    }
  }

  async function clearTemplate() {
    if (!window.confirm('Clear the saved reference template? This removes the current angle reference from disk.')) return;
    try {
      await requestJson('/api/template/clear', { method: 'POST' });
      showToast('Saved reference template cleared.');
      await refreshStatus();
    } catch (error) {
      showToast(error.message, 'error');
    }
  }

  async function uploadVideo() {
    const file = $('video-upload').files[0];
    if (!file) return;
    const status = $('upload-status');
    status.textContent = 'Uploading…';
    const button = document.querySelector('.upload-button');
    if (button) button.style.opacity = '0.65';
    const form = new FormData();
    form.append('file', file);
    try {
      const result = await requestJson('/api/video/upload', { method: 'POST', body: form });
      $('video-path').value = result.video_path;
      $('source-type').value = 'video';
      setSourceFields();
      status.textContent = `${file.name} ready`;
      showToast('Video uploaded. Click Start inspection to process it.');
    } catch (error) {
      status.textContent = 'Upload failed';
      showToast(error.message, 'error');
    } finally {
      if (button) button.style.opacity = '';
      $('video-upload').value = '';
    }
  }

  function updateStateDisplay(status, data, template) {
    const running = Boolean(status.running && !status.error);
    $('header-live-dot').classList.toggle('active', running);
    $('header-live-label').textContent = running ? 'SOURCE LIVE' : (status.error ? 'SOURCE ERROR' : 'SOURCE OFFLINE');
    $('inspection-state').textContent = data?.comparison?.overall_status || (status.error ? 'ERROR' : (running ? 'RUNNING' : 'READY'));
    $('source-label').textContent = status.source_label || 'No active source';
    $('line-count').textContent = data?.edge_count ?? data?.line_count ?? 0;
    $('fps-value').textContent = formatNumber(status.fps || 0, 1);
    $('processing-ms').textContent = data?.processing_ms == null ? '—' : formatNumber(data.processing_ms, 1);
    $('resolution-label').textContent = data?.image_width ? `${data.image_width} × ${data.image_height}` : '— × —';
    $('edge-pixels').textContent = Number(data?.edge_pixels || 0).toLocaleString();
    $('silhouette-area').textContent = `${formatNumber(Number(data?.silhouette?.area_ratio || 0) * 100, 1)}%`;
    $('compared-pairs').textContent = (data?.comparison?.corner_pairs || []).filter((row) => row.reference_corner_id && row.delta_deg !== null && row.delta_deg !== undefined).length;
    $('unmatched-current').textContent = (data?.comparison?.unmatched_current || []).length;
    $('unmatched-reference').textContent = (data?.comparison?.unmatched_reference || []).length;
    $('last-frame-time').textContent = status.last_frame_time ? new Date(status.last_frame_time * 1000).toLocaleTimeString() : '—';
    $('feed-pill').textContent = status.error ? 'ERROR' : (running ? 'LIVE' : 'STANDBY');
    $('feed-pill').classList.toggle('active', running);
    $('feed-pill').classList.toggle('fail', Boolean(status.error));
    $('source-message').textContent = status.error || status.message || 'Choose a source and start inspection.';
    $('source-message').classList.toggle('error', Boolean(status.error));

    const state = String(data?.comparison?.overall_status || 'READY');
    const stateColor = state === 'PASS' ? 'var(--green)' : (state === 'FAIL' || state === 'ERROR' ? 'var(--red)' : 'var(--amber)');
    $('inspection-state').style.color = stateColor;

    const saved = Boolean(template?.saved);
    $('template-state-title').textContent = saved ? template.name || 'Reference saved' : 'No reference saved';
    $('template-state-copy').textContent = saved
      ? `${template.corner_count || 0} corners · ${template.edge_count || 0} edges · created ${template.created_at || 'previously'}`
      : 'Capture a known-good silhouette to create one.';
    $('save-template-btn').disabled = !data || !data.silhouette?.found || (data.corner_count || 0) < 4;
    $('save-template-btn').style.opacity = $('save-template-btn').disabled ? '.55' : '';
    $('clear-template-btn').disabled = !saved;
    $('clear-template-btn').style.opacity = saved ? '' : '.55';
    $('tolerance-summary').textContent = `±${Number(data?.comparison?.tolerance_deg ?? 3).toFixed(1)}°`;
    renderCornerTable(data?.corners || []);
    renderAngleTable(data?.comparison?.corner_pairs || []);
  }

  function renderCornerTable(corners) {
    const body = $('corner-table-body');
    if (!corners.length) {
      body.innerHTML = '<tr><td colspan="6" class="empty-cell">No component corners detected. Check the silhouette mask and contour approximation settings.</td></tr>';
      return;
    }
    body.innerHTML = corners.map((corner) => `
      <tr>
        <td>${escapeHtml(corner.id)}</td>
        <td>${formatNumber(corner.x, 1)}, ${formatNumber(corner.y, 1)}</td>
        <td>${formatNumber(corner.angle_deg, 2)}°</td>
        <td>${formatNumber(corner.prev_edge_angle_deg, 2)}°</td>
        <td>${formatNumber(corner.next_edge_angle_deg, 2)}°</td>
        <td>${corner.is_concave ? 'Concave / notch' : 'Convex / outer'}</td>
      </tr>`).join('');
  }

  function renderAngleTable(rows) {
    const search = $('angle-search').value.trim().toLowerCase();
    const signature = `${rows.length}|${rows.map((r) => `${r.current_corner_id}-${r.reference_corner_id}:${r.angle_deg}:${r.reference_angle_deg}:${r.delta_deg}:${r.orientation_delta_deg}:${r.status}`).join(';')}`;
    const filterSignature = `${signature}|q=${search}`;
    if (filterSignature === lastRenderedAngleSignature) return;
    lastRenderedAngleSignature = filterSignature;
    const body = $('angle-table-body');
    const filtered = rows.filter((row) => {
      const text = `${row.current_corner_id || ''} ${row.reference_corner_id || ''} ${row.status}`.toLowerCase();
      return text.includes(search);
    });
    if (!filtered.length) {
      body.innerHTML = `<tr><td colspan="7" class="empty-cell">${rows.length ? 'No corners match this filter.' : 'Start a source to detect component corners.'}</td></tr>`;
      $('angle-table-summary').textContent = rows.length ? 'No matching corners' : 'Waiting for measurements…';
      $('angle-table-footer').textContent = 'Angles are measured between adjacent boundary-edge rays at each physical contour corner.';
      return;
    }
    body.innerHTML = filtered.map((row) => {
      const currentCorner = row.current_corner_id || '—';
      const referenceCorner = row.reference_corner_id || '—';
      const measured = `${formatNumber(row.angle_deg, 2)}°`;
      const reference = row.reference_angle_deg == null ? '—' : `${formatNumber(row.reference_angle_deg, 2)}°`;
      const delta = row.delta_deg == null ? '—' : `${Number(row.delta_deg) > 0 ? '+' : ''}${formatNumber(row.delta_deg, 2)}°`;
      const orientationDelta = row.orientation_delta_deg == null ? '—' : `${formatNumber(row.orientation_delta_deg, 2)}°`;
      const klass = row.status === 'PASS' ? 'result-pass' : (row.status === 'FAIL' ? 'result-fail' : 'result-unmatched');
      return `<tr><td>${escapeHtml(currentCorner)}</td><td>${escapeHtml(referenceCorner)}</td><td>${measured}</td><td>${reference}</td><td>${delta}</td><td>${orientationDelta}</td><td><span class="result-pill ${klass}">${escapeHtml(row.status)}</span></td></tr>`;
    }).join('');
    const compared = filtered.filter((row) => row.reference_corner_id && row.delta_deg !== null && row.delta_deg !== undefined);
    const passing = compared.filter((row) => row.status === 'PASS').length;
    const failing = compared.filter((row) => row.status === 'FAIL').length;
    const unmatched = filtered.filter((row) => row.status === 'UNMATCHED' || row.status === 'NO TEMPLATE').length;
    $('angle-table-summary').textContent = `${filtered.length} of ${rows.length} corners shown`;
    if (latestComparison?.template_saved) {
      $('angle-table-footer').textContent = `${passing} passing · ${failing} failing · ${unmatched} unmatched/no reference. Adjacent edge orientations are also checked (orientation tolerance ±${formatNumber(latestComparison.orientation_tolerance_deg || 6, 1)}°). Overall: ${latestComparison.overall_status}.`;
    } else {
      $('angle-table-footer').textContent = 'No corner template saved: current corner features are available, but there is no reference to compare. Save a known-good frame as the template.';
    }
  }

  async function refreshStatus() {
    try {
      const result = await requestJson('/api/status');
      latestStatus = result.status;
      latestData = result.data;
      latestComparison = result.data?.comparison || null;
      updateStateDisplay(result.status, result.data, result.template);
    } catch (error) {
      $('source-message').textContent = `Cannot contact FastAPI: ${error.message}`;
      $('source-message').classList.add('error');
    }
  }

  function initializeEvents() {
    $('source-type').addEventListener('change', setSourceFields);
    $('start-btn').addEventListener('click', startSource);
    $('stop-btn').addEventListener('click', stopSource);
    $('save-template-btn').addEventListener('click', saveTemplate);
    $('clear-template-btn').addEventListener('click', clearTemplate);
    $('video-upload').addEventListener('change', uploadVideo);
    $('reset-settings-btn').addEventListener('click', resetSettings);
    $('angle-search').addEventListener('input', () => {
      lastRenderedAngleSignature = '';
      renderAngleTable(latestComparison?.corner_pairs || []);
    });
    for (const [controlId] of CONTROL_MAP) {
      $(controlId).addEventListener('input', debounceSettings);
      $(controlId).addEventListener('change', sendSettings);
    }
    $('draw-edges').addEventListener('change', sendSettings);
    $('draw-angle-labels').addEventListener('change', sendSettings);
  }

  async function init() {
    initializeEvents();
    setSourceFields();
    try {
      const result = await requestJson('/api/status');
      applySettings(result.processing || DEFAULTS);
      latestStatus = result.status;
      latestData = result.data;
      latestComparison = result.data?.comparison || null;
      updateStateDisplay(result.status, result.data, result.template);
    } catch (error) {
      showToast(`Backend status unavailable: ${error.message}`, 'error');
    }
    pollTimer = setInterval(refreshStatus, 700);
  }

  window.addEventListener('beforeunload', () => clearInterval(pollTimer));
  document.addEventListener('DOMContentLoaded', init);
})();
