const MAX_POINTS = 50;

// ── CPU / metric chart ────────────────────────────────────────
const cpuCtx = document.getElementById('cpuChart').getContext('2d');
const cpuChart = new Chart(cpuCtx, {
  type: 'line',
  data: {
    labels  : [],
    datasets: [{
      label          : 'CPU %',
      data           : [],
      borderColor    : '#388bfd',
      backgroundColor: 'rgba(56,139,253,0.08)',
      borderWidth    : 1.5,
      pointRadius    : 0,
      tension        : 0.4,
      fill           : true,
    }, {
      label          : 'Memory %',
      data           : [],
      borderColor    : '#a371f7',
      backgroundColor: 'rgba(163,113,247,0.05)',
      borderWidth    : 1.5,
      pointRadius    : 0,
      tension        : 0.4,
      fill           : true,
    }]
  },
  options: {
    responsive         : true,
    maintainAspectRatio: false,
    animation          : { duration: 0 },
    plugins: {
      legend: {
        labels: { color: '#8b949e', font: { size: 11 }, boxWidth: 12 }
      }
    },
    scales: {
      x: {
        ticks : { color: '#484f58', font: { size: 10 }, maxTicksLimit: 8 },
        grid  : { color: 'rgba(48,54,61,0.5)' },
      },
      y: {
        min   : 0,
        max   : 100,
        ticks : { color: '#484f58', font: { size: 10 } },
        grid  : { color: 'rgba(48,54,61,0.5)' },
      }
    }
  }
});

// ── Cost chart ────────────────────────────────────────────────
const costCtx = document.getElementById('costChart').getContext('2d');
const costChart = new Chart(costCtx, {
  type: 'bar',
  data: {
    labels  : [],
    datasets: [{
      label          : 'Predicted $/hr',
      data           : [],
      backgroundColor: [],
      borderRadius   : 3,
      borderWidth    : 0,
    }]
  },
  options: {
    responsive         : true,
    maintainAspectRatio: false,
    animation          : { duration: 150 },
    plugins: {
      legend: { display: false }
    },
    scales: {
      x: {
        ticks: { color: '#484f58', font: { size: 10 }, maxTicksLimit: 10 },
        grid : { display: false },
      },
      y: {
        ticks: { color: '#484f58', font: { size: 10 },
                 callback: v => '$' + v.toFixed(2) },
        grid : { color: 'rgba(48,54,61,0.5)' },
      }
    }
  }
});

// ── Severity doughnut ─────────────────────────────────────────
const sevCtx = document.getElementById('severityChart').getContext('2d');
const severityChart = new Chart(sevCtx, {
  type: 'doughnut',
  data: {
    labels  : ['Normal', 'Medium', 'High', 'Critical'],
    datasets: [{
      data           : [0, 0, 0, 0],
      backgroundColor: ['#3fb950','#388bfd','#d29922','#f85149'],
      borderWidth    : 0,
      hoverOffset    : 4,
    }]
  },
  options: {
    responsive         : true,
    maintainAspectRatio: false,
    cutout             : '65%',
    plugins: {
      legend: {
        position: 'bottom',
        labels  : { color: '#8b949e', font: { size: 11 },
                    boxWidth: 10, padding: 12 }
      }
    }
  }
});

// ── Chart update helpers ──────────────────────────────────────
function pushToChart(chart, label, ...values) {
  chart.data.labels.push(label);
  values.forEach((v, i) => chart.data.datasets[i].data.push(v));

  if (chart.data.labels.length > MAX_POINTS) {
    chart.data.labels.shift();
    chart.data.datasets.forEach(ds => ds.shift && ds.data.shift());
  }
  chart.update('none');
}

function updateCpuChart(record) {
  const t = formatTime(record.timestamp);
  cpuChart.data.labels.push(t);
  cpuChart.data.datasets[0].data.push(record.cpu  || 0);
  cpuChart.data.datasets[1].data.push(record.mem  || 0);

  if (cpuChart.data.labels.length > MAX_POINTS) {
    cpuChart.data.labels.shift();
    cpuChart.data.datasets.forEach(ds => ds.data.shift());
  }
  cpuChart.update('none');
}

function updateCostChart(record) {
  const t     = formatTime(record.timestamp);
  const color = record.is_anomaly
    ? severityColor(record.severity)
    : 'rgba(56,139,253,0.5)';

  costChart.data.labels.push(t);
  costChart.data.datasets[0].data.push(record.predicted_cost);
  costChart.data.datasets[0].backgroundColor.push(color);

  if (costChart.data.labels.length > MAX_POINTS) {
    costChart.data.labels.shift();
    costChart.data.datasets[0].data.shift();
    costChart.data.datasets[0].backgroundColor.shift();
  }
  costChart.update('none');
}

function updateSeverityChart(counts) {
  severityChart.data.datasets[0].data = [
    counts.normal   || 0,
    counts.medium   || 0,
    counts.high     || 0,
    counts.critical || 0,
  ];
  severityChart.update();
}