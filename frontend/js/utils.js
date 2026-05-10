const SEVERITY_COLORS = {
  normal  : '#3fb950',
  medium  : '#388bfd',
  high    : '#d29922',
  critical: '#f85149',
};

function formatTime(isoString) {
  const d = new Date(isoString);
  return d.toLocaleTimeString('en-IN', { hour12: false });
}

function formatCost(val) {
  return '$' + parseFloat(val).toFixed(3);
}

function severityColor(sev) {
  return SEVERITY_COLORS[sev] || '#8b949e';
}

function clamp(val, min, max) {
  return Math.min(Math.max(val, min), max);
}

function capitalise(str) {
  return str.charAt(0).toUpperCase() + str.slice(1);
}