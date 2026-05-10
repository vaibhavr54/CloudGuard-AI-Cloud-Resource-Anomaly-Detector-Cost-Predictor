// Auto-detect API base — works both locally and on Render
const API_BASE = window.location.origin;
const POLL_INTERVAL = 5000;

let paused    = false;
let pollTimer = null;

async function fetchStream() {
  if (paused) return;
  try {
    const res  = await fetch(`${API_BASE}/stream`);
    const data = await res.json();
    onNewRecord(data);
  } catch(e) {
    console.error('Stream fetch error:', e);
    document.getElementById('statusPill').style.opacity = '0.4';
  }
}

async function fetchStats() {
  try {
    const res  = await fetch(`${API_BASE}/stats`);
    const data = await res.json();
    updateStatCards(data);
  } catch(e) { console.error('Stats fetch error:', e); }
}

async function fetchResources() {
  try {
    const res  = await fetch(`${API_BASE}/resources`);
    const data = await res.json();
    populateResourceSelector(data.resources);
    initResourceGrid(data.resources);
  } catch(e) { console.error('Resources fetch error:', e); }
}

function startPolling() {
  fetchStream();
  pollTimer = setInterval(() => {
    fetchStream();
    fetchStats();
  }, POLL_INTERVAL);
}

function togglePause() {
  paused = !paused;
  const btn = document.getElementById('btnPause');
  const dot = document.querySelector('.pulse-dot');
  if (paused) {
    btn.textContent = '▶ Resume';
    btn.style.color = '#d29922';
    dot.style.animationPlayState = 'paused';
  } else {
    btn.textContent = '⏸ Pause';
    btn.style.color = '';
    dot.style.animationPlayState = 'running';
    fetchStream();
  }
}

function populateResourceSelector(resources) {
  const sel = document.getElementById('resourceSelector');
  resources.forEach(r => {
    const opt       = document.createElement('option');
    opt.value       = r.id;
    opt.textContent = r.id;
    sel.appendChild(opt);
  });
}

function switchResource() { }

document.addEventListener('DOMContentLoaded', () => {
  fetchResources();
  startPolling();
});
