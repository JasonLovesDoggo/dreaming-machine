'use strict';
const $ = id => document.getElementById(id);
const COLORS = ['#8cf', '#ffd75e', '#ff6e6e', '#7ee787', '#d2a8ff', '#ffa657', '#56d4dd', '#f778ba', '#c9d1d9', '#3fb950'];
const DEFAULT_RUNS = 6, AUTO_MS = 60000, STORE_KEY = 'dreaming-training-runs', BASELINE_DASH = [6, 4];
const TRAIN = [
  ['loss', 'total loss', 'grid + 0.2 cursor + 0.3 doom + state loss. Check the notes column: some runs log the last update, others a window mean.', {log: true}],
  ['grid', 'grid loss (tile cross-entropy)', '', {log: true}],
  ['cursor', 'cursor loss', '', {log: true}],
  ['doom', 'doom loss', '', {log: true}],
  ['state', 'state loss', '', {log: true}],
  ['acc_changed', 'changed-tile accuracy (training batch)', 'Accuracy on tiles that differ from the previous frame. Copying the last frame scores 0 here.', {pct: true}],
  ['frac_changed', 'fraction of tiles changed (training batch)', '', {pct: true}],
  ['acc_all', 'all-tile accuracy (training batch)', 'Mostly unchanged tiles, so it stays high even for a model that only copies the last frame. Changed-tile accuracy is the better signal.', {pct: true}],
  ['lr', 'learning rate', '', {}],
  ['grad_norm', 'gradient norm', '', {}],
  ['ss_prob', 'scheduled sampling probability', '', {}],
  ['samples_per_sec', 'samples/s (all GPUs)', '', {}],
];
const HIDDEN = new Set(['time']);

const colors = new Map(), data = new Map(), charts = [];
let runs = [], selected = loadSelection(), info = null, loadedAt = null, loadError = null, followUp = false;

function color(id) {
  if (!colors.has(id)) colors.set(id, COLORS[colors.size % COLORS.length]);
  return colors.get(id);
}

function loadSelection() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORE_KEY));
    return Array.isArray(saved) ? new Set(saved) : null;
  } catch (_) { return null; }
}

function saveSelection() {
  try { localStorage.setItem(STORE_KEY, JSON.stringify([...selected])); } catch (_) {}
}

function el(tag, attrs, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (k === 'class') e.className = v;
    else if (k in e) e[k] = v;
    else e.setAttribute(k, v);
  }
  for (const kid of kids) if (kid !== null && kid !== undefined) e.append(kid);
  return e;
}

function ago(t) {
  const s = Math.max(0, Date.now() / 1000 - t);
  if (s < 90) return `${Math.round(s)}s ago`;
  if (s < 5400) return `${Math.round(s / 60)}m ago`;
  if (s < 172800) return `${Math.round(s / 3600)}h ago`;
  return `${Math.round(s / 86400)}d ago`;
}

const clock = ms => new Date(ms).toLocaleTimeString();

function fmt(v, pct) {
  if (v === null || v === undefined) return '-';
  if (pct) return `${(v * 100).toFixed(2)}%`;
  const a = Math.abs(v);
  return a !== 0 && (a < 1e-3 || a >= 1e5) ? v.toExponential(2) : String(+v.toPrecision(4));
}

async function getJSON(url) {
  const r = await fetch(url, {cache: 'no-store'});
  if (!r.ok) {
    let detail = '';
    try { detail = (await r.json()).detail || ''; } catch (_) {}
    throw new Error(`HTTP ${r.status}${detail ? ` ${detail}` : ''}`);
  }
  return r.json();
}

async function fetchRuns(ids) {
  const results = await Promise.allSettled(ids.map(id => getJSON(`training/metrics?run=${encodeURIComponent(id)}`)));
  const failures = [];
  results.forEach((res, i) => {
    if (res.status === 'fulfilled') data.set(ids[i], res.value);
    else failures.push(`${ids[i]}: ${res.reason.message}`);
  });
  return failures;
}

async function reload() {
  $('reload').disabled = true;
  try {
    info = await getJSON('training/runs');
    runs = info.runs;
    runs.forEach(r => color(r.id));
    if (!selected) selected = new Set(runs.slice(0, DEFAULT_RUNS).map(r => r.id));
    const present = new Set(runs.map(r => r.id));
    for (const id of [...data.keys()]) if (!present.has(id)) data.delete(id);
    const failures = await fetchRuns(runs.filter(r => selected.has(r.id)).map(r => r.id));
    loadedAt = Date.now();
    loadError = failures.length ? failures.join('; ') : null;
    if ((info.refresh.started || info.refresh.running) && !followUp) {
      followUp = true;
      setTimeout(() => reload().finally(() => { followUp = false; }), 8000);
    }
  } catch (e) {
    loadError = `could not list runs: ${e.message}`;
  } finally {
    $('reload').disabled = false;
  }
  render();
}

function renderStatus() {
  const parts = [], s = $('status');
  if (loadedAt) parts.push(document.createTextNode(`loaded ${clock(loadedAt)}`));
  const r = info && info.refresh;
  if (r && !r.enabled) parts.push(document.createTextNode('reading local files (no volume refresh)'));
  if (r && r.running) parts.push(document.createTextNode('volume refresh in progress'));
  if (r && r.last_ok) parts.push(document.createTextNode(`volume refreshed ${ago(r.last_ok)}`));
  if (r && r.last_error) {
    parts.push(el('span', {class: 'warn'}, `volume refresh failed ${ago(r.last_error_at)}: ${r.last_error}; showing the last mounted data`));
  }
  if (loadError) {
    parts.push(el('span', {class: 'warn'}, `reload error: ${loadError}${loadedAt ? `; showing data from ${clock(loadedAt)}` : ''}`));
  }
  s.replaceChildren();
  parts.forEach((p, i) => { if (i) s.append(' \u00b7 '); s.append(p); });
}

function runNotes(d) {
  const out = [];
  for (const [k, values] of Object.entries(d.notes || {})) out.push(`${k}: ${values.join(' | ')}`);
  if (d.truncated) out.push('oldest rows not shown (size cap)');
  if (d.skipped_lines) out.push(`${d.skipped_lines} malformed lines skipped`);
  if (d.nonfinite_values) out.push(`${d.nonfinite_values} non-finite values left out`);
  if (d.restarts) out.push(`${d.restarts} repeated iterations (resume); last write kept`);
  return out.join('; ');
}

function renderRuns() {
  const table = $('runs');
  table.replaceChildren();
  const empty = $('empty');
  empty.hidden = runs.length > 0 || !info;
  if (info && !runs.length) {
    empty.textContent = info.root_exists
      ? `No training metrics yet. Runs write one JSON line per log step to ${info.root}/<run>/metrics.jsonl; their curves appear here after the next volume refresh.`
      : `The runs directory ${info.root} is not available, so there are no training metrics to show yet.`;
  }
  if (!runs.length) return;
  table.append(el('tr', {}, ...['', 'run', 'last it', 'train rows', 'held-out evals', 'last write', 'notes'].map(h => el('th', {}, h))));
  for (const r of runs) {
    const d = data.get(r.id);
    const box = el('input', {type: 'checkbox', checked: selected.has(r.id)});
    box.addEventListener('change', () => toggle(r.id, box.checked));
    table.append(el('tr', {},
      el('td', {}, box),
      el('td', {class: 'run-id'}, el('span', {class: 'swatch', style: `background:${color(r.id)}`}), r.id),
      el('td', {}, d && d.last_it !== null ? String(d.last_it) : '-'),
      el('td', {}, d ? String(d.train.rows) : '-'),
      el('td', {}, d ? (d.eval.rows ? String(d.eval.rows) : 'none recorded') : '-'),
      el('td', {}, ago(r.mtime)),
      el('td', {class: 'notes'}, d ? runNotes(d) : '')));
  }
}

async function toggle(id, on) {
  if (on) selected.add(id); else selected.delete(id);
  saveSelection();
  if (on && !data.has(id)) {
    const failures = await fetchRuns([id]);
    if (failures.length) loadError = failures.join('; ');
  }
  render();
}

function chartWidth() {
  const w = Math.min(window.innerWidth - 60, 1600);
  return w >= 1240 ? Math.floor(w / 2) - 28 : Math.max(320, w - 16);
}

function ema(y, alpha) {
  if (!alpha) return y;
  let s = 0, w = 0;
  return y.map(v => {
    if (v === null) return null;
    s = alpha * s + (1 - alpha) * v;
    w = alpha * w + (1 - alpha);
    return s / w;
  });
}

// lines: [{label, color, x, y, dash, points}]; x ascending, gaps as null.
function addChart(parent, title, note, lines, opt = {}) {
  lines = lines.filter(l => l.y.some(v => v !== null));
  if (opt.log) lines = lines.map(l => Object.assign({}, l, {y: l.y.map(v => v !== null && v > 0 ? v : null)}));
  lines = lines.filter(l => l.y.some(v => v !== null));
  if (!lines.length) return false;
  const box = el('div', {class: 'chart'}, el('h4', {}, title), note ? el('div', {class: 'note'}, note) : null);
  parent.append(box);
  const axis = {stroke: '#aaa', grid: {stroke: '#2c2f36'}, ticks: {stroke: '#444'}};
  const yAxis = Object.assign({}, axis, {size: 70});
  if (opt.pct) yAxis.values = (u, vals) => vals.map(v => `${+(v * 100).toFixed(1)}%`);
  const xLabel = opt.xLabel || 'iteration';
  const series = [{label: xLabel}].concat(lines.map(l => ({
    label: l.label, stroke: l.color, width: 1.5, dash: l.dash, spanGaps: true,
    points: {show: l.points || l.x.length < 30, size: 5, fill: l.color, stroke: l.color},
    value: (u, v) => fmt(v, opt.pct),
  })));
  const opts = {
    width: chartWidth(), height: 240, series, legend: {live: true},
    scales: {x: {time: false}, y: opt.log ? {distr: 3} : {}},
    axes: [Object.assign({}, axis, {label: xLabel, labelSize: 20}), yAxis],
  };
  charts.push(new uPlot(opts, uPlot.join(lines.map(l => [l.x, l.y])), box));
  return true;
}

function noticeInto(parent, text) {
  parent.append(el('div', {class: 'notice'}, text));
}

function nothingLoaded(what) {
  const wanted = runs.some(r => selected && selected.has(r.id));
  return wanted ? 'The selected runs could not be loaded; see the reload error above.' : `Select a run above to plot its ${what}.`;
}

function renderTrain(loaded) {
  const parent = $('train'), alpha = +$('smooth').value, logy = $('logy').checked;
  const smoothNote = alpha ? ` Smoothed: EMA ${alpha}.` : '';
  const known = new Set(TRAIN.map(t => t[0]));
  const extra = [...new Set(loaded.flatMap(d => Object.keys(d.train.series)))].filter(k => !known.has(k) && !HIDDEN.has(k)).sort();
  let drawn = 0;
  for (const [key, title, note, opt] of TRAIN.concat(extra.map(k => [k, k, '', {}]))) {
    const lines = loaded.filter(d => d.train.series[key]).map(d => ({
      label: d.id, color: color(d.id), x: d.train.it, y: ema(d.train.series[key], alpha)}));
    drawn += addChart(parent, title, (note + smoothNote).trim(), lines, Object.assign({}, opt, {log: opt.log && logy}));
  }
  if (!drawn) noticeInto(parent, loaded.length ? 'The selected runs have no training rows yet.' : nothingLoaded('training curves'));
}

function renderEval(loaded) {
  const parent = $('eval'), select = $('app');
  const withEval = loaded.filter(d => d.eval.rows > 0), without = loaded.filter(d => !d.eval.rows);
  const apps = [...new Set(withEval.flatMap(d => d.eval.apps))].sort();
  const current = select.value;
  select.replaceChildren(...apps.map(a => el('option', {value: a}, a)));
  select.disabled = !apps.length;
  if (apps.length) select.value = apps.includes(current) ? current : (apps.includes('mixed') ? 'mixed' : apps[0]);
  if (!withEval.length) {
    noticeInto(parent, loaded.length
      ? `No held-out evaluation recorded for the selected runs (${loaded.map(d => d.id).join(', ')}). Results from dreaming/eval.py appear here once they are logged as kind "eval" rows; nothing is estimated in the meantime.`
      : nothingLoaded('held-out evaluation'));
    return;
  }
  if (without.length) parent.append(el('div', {class: 'note', style: 'flex-basis:100%'}, `No held-out evaluation recorded for: ${without.map(d => d.id).join(', ')}`));
  const app = select.value;
  const having = withEval.filter(d => d.eval.points[app]);
  const line = (d, key, extra) => Object.assign({label: d.id, color: color(d.id), x: d.eval.points[app].it,
    y: d.eval.points[app].series[key] || [], points: true}, extra);
  const baseline = (d, key) => line(d, key, {label: `${d.id} copy baseline`, dash: BASELINE_DASH});
  const horizons = [...new Set(having.flatMap(d => Object.keys(d.eval.points[app].series)
    .map(k => (k.match(/^rollout(\d+)_acc_mean$/) || [])[1]).filter(Boolean)))].sort((a, b) => a - b);
  addChart(parent, `held-out changed-tile accuracy (${app})`, 'Teacher-forced, one step ahead. Copying the last frame scores 0 on changed tiles.',
    having.map(d => line(d, 'acc_changed')), {pct: true});
  for (const h of horizons) {
    addChart(parent, `held-out ${h}-step rollout accuracy, mean over steps (${app})`,
      'The model is fed its own predictions plus the recorded actions. Dashed: copy-first-frame baseline.',
      having.flatMap(d => [line(d, `rollout${h}_acc_mean`), baseline(d, `rollout${h}_copy_mean`)]), {pct: true});
    const latest = withEval.map(d => [d, ((d.eval.latest_rollouts || {})[app] || {})[h]]).filter(([, c]) => c);
    addChart(parent, `latest ${h}-step rollout accuracy per step (${app})`, 'From each run\'s latest evaluation. Dashed: copy-first-frame baseline.',
      latest.flatMap(([d, c]) => {
        const steps = n => Array.from({length: n}, (_, i) => i + 1);
        return [{label: `${d.id} @it ${c.it}`, color: color(d.id), x: steps((c.acc || []).length), y: c.acc || []},
                {label: `${d.id} copy baseline`, color: color(d.id), x: steps((c.copy || []).length), y: c.copy || [], dash: BASELINE_DASH}];
      }), {pct: true, xLabel: 'rollout step'});
  }
  addChart(parent, `held-out grid cross-entropy (${app})`, '', having.map(d => line(d, 'grid_ce')), {log: $('logy').checked});
  addChart(parent, `held-out all-tile accuracy (${app})`, 'Dominated by unchanged tiles. Dashed: copy-last-frame baseline.',
    having.flatMap(d => [line(d, 'acc_all'), baseline(d, 'copy_last_acc_all')]), {pct: true});
}

function render() {
  while (charts.length) charts.pop().destroy();
  $('eval').replaceChildren();
  $('train').replaceChildren();
  renderStatus();
  renderRuns();
  const loaded = runs.filter(r => selected && selected.has(r.id) && data.has(r.id)).map(r => data.get(r.id));
  renderEval(loaded);
  renderTrain(loaded);
}

let resizeTimer = null;
window.addEventListener('resize', () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(render, 300); });
$('reload').addEventListener('click', reload);
for (const id of ['smooth', 'logy', 'app']) $(id).addEventListener('change', render);
setInterval(() => { if ($('auto').checked && !document.hidden) reload(); }, AUTO_MS);
reload();
