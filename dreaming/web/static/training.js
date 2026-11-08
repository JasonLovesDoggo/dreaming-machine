'use strict';
const C = window.TrainingCore;
const $ = id => document.getElementById(id);
const COLORS = ['#8cf', '#ffd75e', '#ff6e6e', '#7ee787', '#d2a8ff', '#ffa657', '#56d4dd', '#f778ba', '#c9d1d9', '#3fb950'];
const DEFAULT_RUNS = 6, AUTO_MS = 60000, STORE_KEY = 'dreaming-training-runs', STALE_S = 15 * 60;
const BASELINE_DASH = [6, 4], GRID_DASH = [2, 3];
const FONT = '11px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';
const LAG_NOTE = 'The live page can trail training by about 5 minutes: metrics are mirrored to the volume periodically and re-read at most every 60 s.';
const LR_NOTE = 'The learning rate is the optimizer step size the schedule sets for each update. It is not a quality score: a falling LR does not mean the model is improving.';
const TRAIN = [
  ['samples_per_sec', 'frame samples / s (all GPUs)', 'Measured by the trainer between log steps. The first log after each start includes compile warm-up.', {}],
  ['lr', 'learning rate (scheduled step size)', LR_NOTE, {sci: true}],
];
const ADVANCED = [
  ['cursor', 'cursor loss', '', {log: true}],
  ['doom', 'doom loss', '', {log: true}],
  ['state', 'state loss', '', {log: true}],
  ['acc_all', 'all-tile accuracy (training batch)', 'Mostly unchanged tiles, so it stays high even for a model that only copies the last frame.', {pct: true}],
  ['frac_changed', 'fraction of tiles changed (training batch)', '', {pct: true}],
  ['grad_norm', 'gradient norm (before clipping)', '', {}],
  ['ss_prob', 'scheduled-sampling probability', 'Share of unroll steps fed the model\'s own prediction. Set by the training schedule.', {}],
];
const KEY_FIELDS = new Set(['loss', 'grid', 'acc_changed', 'time']);
const KNOWN = new Set([...KEY_FIELDS, ...TRAIN.map(t => t[0]), ...ADVANCED.map(t => t[0])]);

const colors = new Map(), data = new Map(), zoom = C.createZoom();
let charts = [], runs = [], selected = loadSelection(), info = null, loadedAt = null, loadError = null, followUp = false;

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
  for (const kid of kids) if (kid !== null && kid !== undefined && kid !== false) e.append(kid);
  return e;
}

const ago = sec => `${C.fmtDuration(Math.max(0, sec))} ago`;
const agoTs = t => ago(Date.now() / 1000 - t);
const clock = ms => new Date(ms).toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'});
const pct = (v, digits = 2) => (v === null || v === undefined ? '-' : `${(v * 100).toFixed(digits)}%`);
const signedPct = v => (v === null || v === undefined ? '-' : `${v >= 0 ? '+' : ''}${(v * 100).toFixed(2)} pt`);

function fmt(v, opt) {
  if (v === null || v === undefined) return '-';
  if (opt.pct) return pct(v);
  if (opt.signed) return signedPct(v);
  if (opt.sci) return C.fmtSci(v);
  const a = Math.abs(v);
  return a !== 0 && (a < 1e-3 || a >= 1e5) ? C.fmtSci(v) : String(+v.toPrecision(4));
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
    if (res.status === 'fulfilled') data.set(ids[i], Object.assign(res.value, {fetchedAt: Date.now()}));
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
  if (loadedAt) parts.push(document.createTextNode(`loaded ${new Date(loadedAt).toLocaleTimeString()}`));
  const r = info && info.refresh;
  if (r && !r.enabled) parts.push(document.createTextNode('reading local files (no volume refresh)'));
  if (r && r.running) parts.push(document.createTextNode('volume refresh in progress'));
  if (r && r.last_ok) parts.push(document.createTextNode(`volume refreshed ${agoTs(r.last_ok)}`));
  if (r && r.last_error) parts.push(el('span', {class: 'warn'}, `volume refresh failed ${agoTs(r.last_error_at)}: ${r.last_error}; showing the last mounted data`));
  if (loadError) parts.push(el('span', {class: 'warn'}, `reload error: ${loadError}${loadedAt ? '; showing the last loaded data' : ''}`));
  s.replaceChildren();
  parts.forEach((p, i) => { if (i) s.append(' \u00b7 '); s.append(p); });
}

// ---- active-run summary ----

function tile(label, value, sub, cls) {
  return el('div', {class: `tile${cls ? ` ${cls}` : ''}`}, el('div', {class: 'label'}, label),
    el('div', {class: 'value', title: value}, value), sub ? el('div', {class: 'sub'}, sub) : null);
}

function activeRun(loaded) {
  const select = $('active'), current = select.value;
  const withTrain = loaded.filter(d => d.train.rows);
  select.replaceChildren(el('option', {value: ''}, 'latest logged row'), ...loaded.map(d => el('option', {value: d.id}, d.id)));
  select.value = loaded.some(d => d.id === current) ? current : '';
  if (select.value) return loaded.find(d => d.id === select.value);
  const written = d => (d.summary.last_time === null ? d.mtime : d.summary.last_time);
  return withTrain.slice().sort((a, b) => written(b) - written(a))[0] || null;
}

function renderSummary(loaded) {
  const box = $('summary');
  box.replaceChildren();
  const d = activeRun(loaded);
  if (!d) {
    box.append(el('div', {class: 'notice'}, loaded.length ? 'The selected runs have no training rows yet.' : 'Select a run under Runs to see its summary.'));
    return;
  }
  const s = d.summary, cfg = (d.config && d.config.values) || {};
  const now = s.server_time + (Date.now() - d.fetchedAt) / 1000;
  const tiles = [];
  tiles.push(tile('run', d.id, d.config ? `config row at it ${C.fmtInt(d.config.it)}` : 'no config row: counts and ETAs below are unavailable', 'wide run'));
  tiles.push(tile('iteration', s.last_it === null ? 'none yet' : C.fmtInt(s.last_it), `${C.fmtInt(d.train.rows)} logged steps`));
  tiles.push(s.frame_samples === null
    ? tile('frame samples seen', 'unavailable', 'needs frame_samples_base, frame_samples_base_iter and samples_per_update in a config row', 'na')
    : tile('frame samples seen', C.fmtInt(s.frame_samples),
      `${C.fmtCount(cfg.frame_samples_base)} at it ${C.fmtInt(cfg.frame_samples_base_iter)} + ${C.fmtInt(cfg.samples_per_update)} per update`));
  tiles.push(s.samples_per_sec === null
    ? tile('frame samples / s', 'unavailable', 'no steady-state logs yet; the first log after a start is skipped (compile warm-up)', 'na')
    : tile('frame samples / s', C.fmtInt(s.samples_per_sec), `median of last ${s.samples_per_sec_logs} logs, all GPUs; compile warm-up skipped`));
  const sec = s.sec_per_update;
  tiles.push(sec === null
    ? tile('time per update', 'unavailable', 'needs row timestamps, or samples_per_update in a config row', 'na')
    : tile('time per update', `${sec.toFixed(3)} s`, `${(1 / sec).toFixed(2)} updates/s; from ${s.sec_per_update_source}${s.sec_per_update_source === 'row timestamps' ? ` (median of ${s.sec_per_update_logs})` : ''}`));
  const ckpt = s.milestones.find(m => m.label === 'next checkpoint');
  if (!ckpt) tiles.push(tile('next checkpoint', 'unavailable', 'needs checkpoint_every in a config row', 'na'));
  for (const m of s.milestones) {
    const every = m.label === 'next checkpoint' ? ` (every ${C.fmtInt(cfg.checkpoint_every)})` : '';
    tiles.push(tile(m.label, `it ${C.fmtInt(m.it)}`, m.seconds === null
      ? `${C.fmtInt(m.updates)} updates away${every}; time unavailable`
      : `${m.at < now ? 'due' : `in ~${C.fmtDuration(m.at - now)}`} (~${clock(m.at * 1000)}) \u00b7 ${C.fmtInt(m.updates)} updates${every}`));
  }
  const ph = s.lr_phase;
  tiles.push(tile('learning rate', s.lr === null || s.lr === undefined ? 'unavailable' : C.fmtSci(s.lr),
    `${ph ? `${ph.name}: warm-up to ${C.fmtSci(ph.peak_lr)} by it ${C.fmtInt(ph.warmup_end)}, cosine to floor ${C.fmtSci(ph.min_lr)} at it ${C.fmtInt(ph.schedule_end)}. ` : 'schedule unknown (no config row). '}Step size, not a quality score.`,
    'wide'));
  const rowAge = s.last_time === null ? null : now - s.last_time, fileAge = now - s.file_mtime;
  const stale = (rowAge === null ? fileAge : rowAge) > STALE_S;
  tiles.push(tile('freshness', rowAge === null ? `file ${ago(fileAge)}` : ago(rowAge),
    `${rowAge === null ? 'rows carry no timestamps' : 'last training row logged'}; metrics file updated ${ago(fileAge)}. ${stale ? 'Stale: no new rows in 15+ min. ' : ''}${LAG_NOTE}`,
    `wide${stale ? ' pending' : ''}`));
  tiles.push(s.last_eval_it === null
    ? tile('held-out evaluation', 'not evaluated yet', 'No held-out eval logged for this run. Other runs\' results (e.g. audit A/B) are not its quality.', 'pending')
    : tile('held-out evaluation', `it ${C.fmtInt(s.last_eval_it)}`, `${C.evalLag(s.last_it, s.last_eval_it)}; see Held-out quality`));
  box.append(...tiles);
}

// ---- runs table ----

function runNotes(d) {
  const out = [];
  for (const [k, values] of Object.entries(d.notes || {})) out.push(`${k}: ${values.join(' | ')}`);
  if (d.config && d.config.rejected.length) out.push(`invalid config fields ignored: ${d.config.rejected.join(', ')}`);
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
  table.append(el('tr', {}, ...['', 'run', 'last it', 'samples/s', 's/update', 'held-out evals', 'last write', 'notes']
    .map((h, i) => el('th', {class: i >= 2 && i <= 5 ? 'num' : ''}, h))));
  for (const r of runs) {
    const d = data.get(r.id), s = d && d.summary;
    const box = el('input', {type: 'checkbox', checked: selected.has(r.id)});
    box.addEventListener('change', () => toggle(r.id, box.checked));
    table.append(el('tr', {},
      el('td', {}, box),
      el('td', {class: 'run-id'}, el('span', {class: 'swatch', style: `background:${color(r.id)}`}), r.id),
      el('td', {class: 'num'}, d && d.last_it !== null ? C.fmtInt(d.last_it) : '-'),
      el('td', {class: 'num'}, s && s.samples_per_sec !== null ? C.fmtInt(s.samples_per_sec) : '-'),
      el('td', {class: 'num'}, s && s.sec_per_update !== null ? s.sec_per_update.toFixed(3) : '-'),
      el('td', {class: 'num'}, d ? (d.eval.rows ? String(d.eval.rows) : 'none yet') : '-'),
      el('td', {}, agoTs(r.mtime)),
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

// ---- charts ----

// Two charts per row on wide screens; 22px = chart padding + border, 14px = grid gap.
function chartWidth(parent) {
  const w = (parent && parent.clientWidth) || Math.min(window.innerWidth - 60, 1580);
  return w >= 1000 ? Math.floor((w - 14) / 2) - 22 : Math.max(300, w - 22);
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

const axisMode = () => $('xaxis').value;
const iterGroup = () => `x:${axisMode()}`;

// Map iterations onto the chosen x axis; in frame-sample mode only runs with config metadata are drawn.
function xLine(d, its, y) {
  if (axisMode() === 'it') return {x: its, y};
  const cfg = d.config && d.config.values, x = [], yy = [];
  its.forEach((it, i) => {
    const v = C.frameSamplesAt(cfg, it);
    if (v !== null) { x.push(v); yy.push(y[i]); }
  });
  return {x, y: yy};
}

function onSelect(u, group) {
  const w = u.select.width;
  if (w < 3) return;
  const min = u.posToVal(u.select.left, 'x'), max = u.posToVal(u.select.left + w, 'x');
  u.setSelect({left: 0, top: 0, width: 0, height: 0}, false);
  if (!zoom.set(group, min, max)) return;
  for (const c of charts) if (c.group === group) c.u.setScale('x', {min, max});
  updateZoomUI();
}

function resetZoom(group) {
  zoom.reset(group);
  render();
}

function updateZoomUI() {
  $('resetzoom').disabled = !zoom.active();
  $('zoomhint').textContent = zoom.active()
    ? 'Zoomed in: some charts show part of their data. Reset zoom (or double-click a chart) shows everything again.'
    : 'Drag across a chart to zoom; training charts zoom together, held-out charts together. Double-click a chart or press Reset zoom to zoom back out.';
  for (const c of charts) c.reset.hidden = !zoom.get(c.group);
}

// lines: [{label, color, x, y, dash, points}]; x ascending, gaps as null.
function addChart(parent, title, note, lines, opt = {}) {
  if (opt.log) lines = lines.map(l => Object.assign({}, l, {y: l.y.map(v => v !== null && v > 0 ? v : null)}));
  lines = lines.filter(l => l.x.length && l.y.some(v => v !== null && v !== undefined));
  if (!lines.length) return false;
  const group = opt.group || iterGroup();
  const reset = el('button', {class: 'chart-reset', type: 'button', title: 'Show this axis\'s full data again', hidden: true}, 'reset zoom');
  reset.addEventListener('click', () => resetZoom(group));
  const box = el('div', {class: 'chart'}, el('div', {class: 'chart-head'}, el('h4', {}, title), reset), note ? el('div', {class: 'note'}, note) : null);
  parent.append(box);
  const axis = {stroke: '#9aa0aa', font: FONT, labelFont: FONT, grid: {stroke: '#23262e'}, ticks: {stroke: '#30343d'}};
  const yAxis = Object.assign({}, axis, {size: 72});
  if (opt.pct) yAxis.values = (u, vals) => vals.map(v => `${+(v * 100).toFixed(2)}%`);
  else if (opt.signed) yAxis.values = (u, vals) => vals.map(v => `${v > 0 ? '+' : ''}${+(v * 100).toFixed(1)}`);
  else if (opt.sci) yAxis.values = (u, vals) => vals.map(C.fmtSci);
  const samples = !opt.xLabel && axisMode() === 'samples';
  const xLabel = opt.xLabel || (samples ? 'frame samples' : 'iteration');
  const xAxis = Object.assign({}, axis, {label: xLabel, labelSize: 18});
  if (samples) xAxis.values = (u, vals) => vals.map(C.fmtCount);
  const series = [{label: xLabel, value: (u, v) => (v === null ? '-' : samples ? C.fmtInt(v) : String(v))}].concat(lines.map(l => ({
    label: l.label, stroke: l.color, width: l.width || 1.5, dash: l.dash, spanGaps: true,
    points: {show: l.points || l.x.length < 30, size: 5, fill: l.color, stroke: l.color},
    value: (u, v) => fmt(v, opt),
  })));
  const width = opt.width || chartWidth(parent);
  box.style.width = `${width}px`;
  const opts = {
    width, height: opt.height || 240, series, legend: {live: true},
    scales: {x: {time: false}, y: opt.log ? {distr: 3} : opt.pct ? {range: pctRange} : {}},
    axes: [xAxis, yAxis],
    cursor: {drag: {x: true, y: false, setScale: false}, bind: {dblclick: () => null}},
    hooks: {setSelect: [u => onSelect(u, group)]},
  };
  const u = new uPlot(opts, uPlot.join(lines.map(l => [l.x, l.y])), box);
  u.over.addEventListener('dblclick', () => resetZoom(group));
  const z = zoom.get(group);
  if (z) u.setScale('x', z);
  charts.push({u, group, reset});
  return true;
}

// Accuracies and rates live in [0, 1]: pad the data range but never past those bounds.
function pctRange(u, min, max) {
  const pad = Math.max((max - min) * 0.1, 0.005);
  return [Math.max(0, min - pad), Math.min(1, max + pad)];
}

const noticeInto = (parent, text) => parent.append(el('div', {class: 'notice'}, text));

function nothingLoaded(what) {
  const wanted = runs.some(r => selected && selected.has(r.id));
  return wanted ? 'The selected runs could not be loaded; see the reload error above.' : `Select a run under Runs to plot its ${what}.`;
}

function trainLines(loaded, key, extra, raw) {
  const alpha = raw ? 0 : +$('smooth').value;
  return loaded.filter(d => d.train.series[key]).map(d => Object.assign(
    {label: d.id, color: color(d.id)}, xLine(d, d.train.it, ema(d.train.series[key], alpha)), extra || {}));
}

function axisNote(loaded) {
  if (axisMode() !== 'samples') return '';
  const missing = loaded.filter(d => !(d.config && C.frameSamplesAt(d.config.values, d.config.values.frame_samples_base_iter) !== null));
  return missing.length ? ` Not drawn on the frame-sample axis (no frame-sample config): ${missing.map(d => d.id).join(', ')}.` : '';
}

function renderKey(loaded) {
  const parent = $('key'), logy = $('logy').checked, alpha = +$('smooth').value;
  const smooth = alpha ? ` Display smoothing: EMA ${alpha} of the logged points (computed in the page, not logged by the trainer).` : '';
  if (!loaded.length) { noticeInto(parent, nothingLoaded('training curves')); return; }
  const lossLines = loaded.flatMap(d => [
    ...trainLines([d], 'loss', {label: `${d.id} total`, width: 2}),
    ...trainLines([d], 'grid', {label: `${d.id} grid`, dash: GRID_DASH})]);
  const a = addChart(parent, 'training loss: total (last batch) and grid CE (window mean)',
    `Solid: total loss of the last rank-0 batch at each log step (a noisy single-batch value). Dotted: grid (tile) cross-entropy, the rank-0 mean over the log window. They come from different batches, so they do not add up row by row. Lower is better.${smooth}${axisNote(loaded)}`,
    lossLines, {log: logy});
  const b = addChart(parent, 'training changed-tile accuracy (not held-out)',
    `Rank-0 mean over the log window, on tiles that differ from the previous frame; copying the last frame scores 0. Training batches, so not a generalization score: see Held-out quality.${smooth}`,
    trainLines(loaded, 'acc_changed'), {pct: true});
  if (!a && !b) noticeInto(parent, 'The selected runs have no loss or changed-tile accuracy rows yet.');
}

function renderTrain(loaded) {
  const alpha = +$('smooth').value, logy = $('logy').checked;
  const smooth = alpha ? ` Display smoothing: EMA ${alpha} (computed in the page).` : '';
  const extra = [...new Set(loaded.flatMap(d => Object.keys(d.train.series)))].filter(k => !KNOWN.has(k)).sort();
  let drawn = 0;
  for (const [key, title, note, opt] of TRAIN) {
    const raw = key === 'lr';  // a schedule, not a noisy measurement: never smoothed
    drawn += addChart($('train'), title, note + (raw ? '' : smooth), trainLines(loaded, key, null, raw), opt);
  }
  for (const [key, title, note, opt] of ADVANCED.concat(extra.map(k => [k, k, '', {}]))) {
    drawn += addChart($('trainmore'), title, (note + smooth).trim(), trainLines(loaded, key), Object.assign({}, opt, {log: opt.log && logy}));
  }
  if (!drawn && loaded.length) noticeInto($('train'), 'The selected runs have no throughput or schedule rows yet.');
}

// ---- held-out ----

function latestRow(d, app) {
  const p = d.eval.points[app];
  if (!p || !p.it.length) return null;
  const i = p.it.length - 1, out = {it: p.it[i]};
  for (const [k, v] of Object.entries(p.series)) out[k] = v[i];
  return out;
}

function horizonsOf(rows) {
  return [...new Set(rows.flatMap(r => Object.keys(r).map(k => (k.match(/^rollout(\d+)_gap_mean$/) || [])[1]).filter(Boolean)))]
    .sort((a, b) => a - b);
}

function renderEvalTable(withEval, app) {
  const parent = $('evaltable');
  const rows = withEval.map(d => [d, latestRow(d, app)]).filter(([, r]) => r);
  if (!rows.length) return;
  const horizons = horizonsOf(rows.map(([, r]) => r));
  const head = ['run', 'eval it', 'changed-tile acc', 'false-change rate', 'grid CE',
    ...horizons.map(h => `${h}-step rollout vs copy`), 'all-tile acc (copy)'];
  const table = el('table', {}, el('tr', {}, ...head.map((h, i) => el('th', {class: i ? 'num' : ''}, h))));
  for (const [d, r] of rows) {
    const fp = r.fp_change_rate === undefined ? 'not logged' : pct(r.fp_change_rate, 3);
    table.append(el('tr', {},
      el('td', {class: 'run-id'}, el('span', {class: 'swatch', style: `background:${color(d.id)}`}), d.id),
      el('td', {class: 'num'}, C.fmtInt(r.it)),
      el('td', {class: 'num'}, pct(r.acc_changed)),
      el('td', {class: 'num'}, fp),
      el('td', {class: 'num'}, r.grid_ce === undefined ? '-' : C.fmtSci(r.grid_ce)),
      ...horizons.map(h => el('td', {class: 'num', title: `rollout ${pct(r[`rollout${h}_acc_mean`])} vs copy-first ${pct(r[`rollout${h}_copy_mean`])}, mean over steps`},
        signedPct(r[`rollout${h}_gap_mean`]))),
      el('td', {class: 'num'}, `${pct(r.acc_all)} (${pct(r.copy_last_acc_all)})`)));
  }
  parent.append(el('div', {class: 'note'}, `Latest evaluation per run for ${app}. Rollout vs copy: mean accuracy over the rollout minus the copy-first-frame baseline, in points; positive beats copying.`), table);
}

function renderEval(loaded) {
  const parent = $('eval'), more = $('evalmore'), status = $('evalstatus'), select = $('app');
  const withEval = loaded.filter(d => d.eval.rows > 0), without = loaded.filter(d => !d.eval.rows);
  const apps = [...new Set(withEval.flatMap(d => d.eval.apps))].sort();
  const current = select.value;
  select.replaceChildren(...apps.map(a => el('option', {value: a}, a)));
  select.disabled = !apps.length;
  if (apps.length) select.value = apps.includes(current) ? current : (apps.includes('mixed') ? 'mixed' : apps[0]);
  if (without.length) {
    status.append(el('div', {class: 'notice'}, el('span', {class: 'badge'}, 'not evaluated yet'),
      `${without.map(d => d.id).join(', ')}: no held-out evaluation logged. These runs have no held-out quality numbers yet; the results below belong only to the runs named in them.`));
  }
  if (!withEval.length) {
    noticeInto(parent, loaded.length
      ? 'None of the selected runs has a held-out evaluation yet. dreaming/eval.py results appear here once they are logged as kind "eval" rows; nothing is estimated in the meantime.'
      : nothingLoaded('held-out evaluation'));
    return;
  }
  const app = select.value;
  renderEvalTable(withEval, app);
  const having = withEval.filter(d => d.eval.points[app]);
  const line = (d, key, extra) => Object.assign({label: d.id, color: color(d.id), points: true},
    xLine(d, d.eval.points[app].it, d.eval.points[app].series[key] || []), extra);
  const baseline = (d, key, what) => line(d, key, {label: `${d.id} ${what}`, dash: BASELINE_DASH});
  const horizons = horizonsOf(having.map(d => d.eval.points[app].series).map(s => Object.fromEntries(Object.keys(s).map(k => [k, 1]))));
  const note = axisNote(withEval);
  const ev = {group: `eval:${axisMode()}`};
  addChart(parent, `held-out changed-tile accuracy (${app})`, `Teacher-forced, one step ahead. Copying the last frame scores 0. Higher is better.${note}`,
    having.map(d => line(d, 'acc_changed')), Object.assign({pct: true}, ev));
  if (!addChart(parent, `held-out false-change rate (${app})`, 'Share of tiles that should stay the same but the model changed. Lower is better. Dashed: copy-last-frame baseline (0).',
    having.flatMap(d => [line(d, 'fp_change_rate'), baseline(d, 'copy_last_fp_change_rate', 'copy baseline')]), Object.assign({pct: true}, ev))) {
    noticeInto(parent, `False-change rate (${app}) is not logged by the selected runs' evaluator, so it is not shown or derived.`);
  }
  for (const h of horizons) {
    addChart(parent, `${h}-step rollout accuracy minus copy baseline (${app})`,
      'Model fed its own predictions plus the recorded actions. Points above 0 beat copying the first frame; mean over rollout steps.',
      having.map(d => line(d, `rollout${h}_gap_mean`)), Object.assign({signed: true}, ev));
    addChart(parent, `${h}-step rollout accuracy (${app})`, 'Mean over rollout steps. Dashed: copy-first-frame baseline.',
      having.flatMap(d => [line(d, `rollout${h}_acc_mean`), baseline(d, `rollout${h}_copy_mean`, 'copy baseline')]), Object.assign({pct: true}, ev));
  }
  addChart(more, `held-out grid cross-entropy (${app})`, '', having.map(d => line(d, 'grid_ce')), Object.assign({log: $('logy').checked}, ev));
  addChart(more, `held-out changed-tile accuracy, small vs large changes (${app})`, 'Solid: small changes. Dashed: large changes.',
    having.flatMap(d => [line(d, 'acc_changed_small', {label: `${d.id} small`}), line(d, 'acc_changed_large', {label: `${d.id} large`, dash: BASELINE_DASH})]), Object.assign({pct: true}, ev));
  addChart(more, `held-out all-tile accuracy (${app})`, 'Dominated by unchanged tiles. Dashed: copy-last-frame baseline.',
    having.flatMap(d => [line(d, 'acc_all'), baseline(d, 'copy_last_acc_all', 'copy baseline')]), Object.assign({pct: true}, ev));
  for (const h of horizons) {
    const latest = withEval.map(d => [d, ((d.eval.latest_rollouts || {})[app] || {})[h]]).filter(([, c]) => c);
    const steps = n => Array.from({length: n}, (_, i) => i + 1);
    addChart(more, `latest ${h}-step rollout accuracy per step (${app})`, 'From each run\'s latest evaluation. Dashed: copy-first-frame baseline.',
      latest.flatMap(([d, c]) => [
        {label: `${d.id} @it ${c.it}`, color: color(d.id), x: steps((c.acc || []).length), y: c.acc || []},
        {label: `${d.id} copy baseline`, color: color(d.id), x: steps((c.copy || []).length), y: c.copy || [], dash: BASELINE_DASH}]),
      {pct: true, xLabel: 'rollout step', group: `step:${h}`});
  }
}

function render() {
  while (charts.length) charts.pop().u.destroy();
  for (const id of ['summary', 'key', 'evalstatus', 'evaltable', 'eval', 'evalmore', 'train', 'trainmore']) $(id).replaceChildren();
  renderStatus();
  renderRuns();
  const loaded = runs.filter(r => selected && selected.has(r.id) && data.has(r.id)).map(r => data.get(r.id));
  renderSummary(loaded);
  renderKey(loaded);
  renderEval(loaded);
  renderTrain(loaded);
  updateZoomUI();
}

let resizeTimer = null;
window.addEventListener('resize', () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(render, 300); });
$('reload').addEventListener('click', reload);
$('resetzoom').addEventListener('click', () => resetZoom());
for (const id of ['smooth', 'logy', 'app', 'active', 'xaxis']) $(id).addEventListener('change', render);
setInterval(() => { if ($('auto').checked && !document.hidden) reload(); }, AUTO_MS);
reload();
