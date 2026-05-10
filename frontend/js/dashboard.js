// ── State ─────────────────────────────────────────────────────
const resourceLatest = {};
const severityCounts = { normal:0, medium:0, high:0, critical:0 };
let   feedItemCount  = 0;

// ── Main handler — called on every new stream record ──────────
function onNewRecord(record) {
  // Attach raw metrics for charts
  // Extract from clf_reasons OR reg_reasons
  const allReasons = [...(record.clf_reasons || []), ...(record.reg_reasons || [])];
  record.cpu = record.cpu_utilization    || 0;
  record.mem = record.memory_utilization || 0;

  // Update charts
  updateCpuChart(record);
  updateCostChart(record);

  // Update severity counts
  severityCounts[record.severity] = (severityCounts[record.severity] || 0) + 1;
  updateSeverityChart(severityCounts);

  // Update detail panel
  updateDetailPanel(record);

  // Update resource health grid
  resourceLatest[record.resource_id] = record;
  updateResourceGrid();

  // Add to anomaly feed if anomalous
  if (record.is_anomaly) addToFeed(record);

  // Update header timestamp
  document.getElementById('lastUpdate').textContent =
    'Last update: ' + formatTime(record.timestamp);

  document.getElementById('statusPill').style.opacity = '1';
}

// ── Stat cards ────────────────────────────────────────────────
function updateStatCards(stats) {
  document.getElementById('totalPredictions').textContent =
    stats.total_predictions.toLocaleString();
  document.getElementById('totalAnomalies').textContent =
    stats.total_anomalies.toLocaleString();
  document.getElementById('anomalyRate').textContent =
    stats.anomaly_rate.toFixed(2) + '% rate';
  document.getElementById('criticalCount').textContent =
    stats.critical_count.toLocaleString();
  document.getElementById('highCount').textContent =
    stats.high_count + ' high severity';
  document.getElementById('avgCost').textContent =
    formatCost(stats.avg_cost);
}

// ── Detail panel ──────────────────────────────────────────────
function updateDetailPanel(record) {
  document.getElementById('detailResource').textContent =
    record.resource_id + ' · ' + record.resource_type;
  document.getElementById('probValue').textContent =
    (record.anomaly_prob * 100).toFixed(1) + '%';
  document.getElementById('costValue').textContent =
    formatCost(record.predicted_cost) + '/hr';
  document.getElementById('resourceType').textContent =
    record.resource_type;

  // Prob bar
  const pct = clamp(record.anomaly_prob * 100, 0, 100);
  const bar = document.getElementById('probBar');
  bar.style.width      = pct + '%';
  bar.style.background = severityColor(record.severity);

  // Severity badge
  const badge = document.getElementById('severityBadge');
  badge.textContent   = capitalise(record.severity);
  badge.style.background = severityColor(record.severity) + '22';
  badge.style.color      = severityColor(record.severity);
  badge.style.border     = '1px solid ' + severityColor(record.severity) + '55';

  // SHAP lists
  renderShapList('clfShap', record.clf_reasons, 'anomaly');
  renderShapList('regShap', record.reg_reasons, 'cost');
}

function renderShapList(containerId, reasons, type) {
  const container = document.getElementById(containerId);
  if (!reasons || !reasons.length) {
    container.innerHTML = '<div style="color:#484f58;font-size:0.75rem">No data</div>';
    return;
  }

  const maxShap = Math.max(...reasons.map(r => Math.abs(r.shap)));

  container.innerHTML = reasons.map(r => {
    const pct      = maxShap > 0 ? (Math.abs(r.shap) / maxShap * 100) : 0;
    const isUp     = r.impact.startsWith('up');
    const barClass = isUp ? 'up' : 'down';
    const valClass = isUp ? 'up' : 'down';
    const sign     = r.shap >= 0 ? '+' : '';
    const shortName= r.feature.length > 24
      ? r.feature.slice(0, 22) + '…'
      : r.feature;

    return `
      <div class="shap-row">
        <span class="shap-feature" title="${r.feature}">${shortName}</span>
        <div class="shap-bar-wrap">
          <div class="shap-bar ${barClass}" style="width:${pct.toFixed(1)}%"></div>
        </div>
        <span class="shap-val ${valClass}">${sign}${r.shap.toFixed(3)}</span>
      </div>`;
  }).join('');
}

// ── Anomaly feed ──────────────────────────────────────────────
function addToFeed(record) {
  const feed = document.getElementById('anomalyFeed');

  // Remove empty state
  const empty = feed.querySelector('.feed-empty');
  if (empty) empty.remove();

  const item = document.createElement('div');
  item.className = `feed-item ${record.severity}`;

  const topReason = record.clf_reasons[0];
  const reasonText = topReason
    ? `${topReason.feature}: ${topReason.shap >= 0 ? '↑' : '↓'} impact`
    : 'anomaly detected';

  item.innerHTML = `
    <div class="feed-item-top">
      <span class="feed-resource">${record.resource_id}</span>
      <span class="feed-severity ${record.severity}">${record.severity}</span>
    </div>
    <div class="feed-detail">
      prob=${(record.anomaly_prob*100).toFixed(1)}%
      · cost=${formatCost(record.predicted_cost)}/hr
      · ${reasonText}
    </div>`;

  feed.insertBefore(item, feed.firstChild);
  feedItemCount++;

  // Keep feed max 30 items
  while (feed.children.length > 30) feed.removeChild(feed.lastChild);

  document.getElementById('feedCount').textContent =
    feedItemCount + ' alerts';
}

// ── Resource health grid ──────────────────────────────────────
function initResourceGrid(resources) {
  const grid = document.getElementById('resourceGrid');
  grid.innerHTML = resources.map(r => `
    <div class="resource-tile" id="tile-${r.id.replace(/[^a-z0-9]/gi,'-')}">
      <div class="resource-tile-name">${r.id}</div>
      <div class="resource-tile-meta">${r.type} · waiting...</div>
    </div>`).join('');
}

function updateResourceGrid() {
  Object.entries(resourceLatest).forEach(([rid, record]) => {
    const safeId = rid.replace(/[^a-z0-9]/gi, '-');
    const tile   = document.getElementById(`tile-${safeId}`);
    if (!tile) return;

    tile.className = `resource-tile ${record.is_anomaly ? 'anomaly' : ''}`;
    tile.querySelector('.resource-tile-meta').textContent =
      `${record.resource_type} · prob=${(record.anomaly_prob*100).toFixed(0)}%`
      + ` · ${formatCost(record.predicted_cost)}/hr`;
  });
}