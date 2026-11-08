'use strict';
(function () {
  const {fmtNumber, fmtDuration, ranked, createPoller} = AnalyticsCore;
  const byId = id => document.getElementById(id);
  let lastUpdated = null;
  function element(tag, className, value) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (value !== undefined) node.textContent = value;
    return node;
  }
  function card(label, value, hint) {
    const tile = element('div', 'tile');
    tile.append(element('div', 'label', label), element('div', 'value', value), element('div', 'sub', hint));
    return tile;
  }
  function renderRanks(id, counts) {
    const target = byId(id), rows = ranked(counts);
    target.replaceChildren();
    if (!rows.length) target.append(element('p', 'hint', 'No counts reported.'));
    for (const row of rows) {
      const item = element('div', 'rank-row'), label = element('div', 'rank-label');
      label.append(element('span', 'rank-name', row.name), element('span', 'rank-count', `${fmtNumber(row.count)} · ${fmtNumber(row.share, 1)}%`));
      const track = element('div', 'track'), fill = element('div', 'fill');
      fill.style.width = `${row.share}%`;
      track.append(fill);
      item.append(label, track);
      target.append(item);
    }
  }
  const poller = createPoller({
    async fetch() {
      byId('refresh').disabled = true;
      byId('status').textContent = lastUpdated ? 'Refreshing aggregates…' : 'Loading analytics…';
      const response = await fetch('telemetry/summary', {cache: 'no-store'});
      if (!response.ok) throw new Error('Summary request failed');
      const data = await response.json();
      if (!data || typeof data !== 'object' || Array.isArray(data)) throw new Error('Invalid summary');
      return data;
    },
    onData(data) {
      const pages = data.pages && typeof data.pages === 'object' && !Array.isArray(data.pages)
        ? ranked(data.pages, Infinity).reduce((sum, row) => sum + row.count, 0) : null;
      byId('kpis').replaceChildren(
        card('Unique visitors', fmtNumber(data.visitors), 'Distinct anonymous browser IDs · all time'),
        card('Sessions', fmtNumber(data.sessions), 'Reported sessions · all time'),
        card('Total session duration', fmtDuration(data.total_duration), 'Sum of reported durations'),
        card('Avg session duration', fmtDuration(data.avg_duration), 'Total duration / sessions'),
        card('Page views', fmtNumber(pages), 'Reported pageview events'));
      for (const key of ['pages', 'apps', 'actions', 'checkpoints']) renderRanks(key, data[key]);
      byId('performance').replaceChildren();
      for (const [key, label, unit] of [['fps', 'FPS', 'fps'], ['latency_ms', 'Server step time', 'ms']]) {
        const metric = data[key] || {};
        byId('performance').append(card(label, `${fmtNumber(metric.avg, 1)} ${unit} avg`,
          `${fmtNumber(metric.min, 1)} min · ${fmtNumber(metric.max, 1)} max · ${fmtNumber(metric.count)} samples`));
      }
      byId('empty').hidden = data.sessions !== 0;
      lastUpdated = new Date();
      byId('updated').textContent = `Last updated ${lastUpdated.toLocaleTimeString()}`;
      byId('status').textContent = 'Up to date · all-time aggregates';
      byId('status').className = '';
      byId('refresh').disabled = false;
    },
    onError() {
      byId('status').textContent = lastUpdated ? 'Stale · refresh failed. Last good data remains visible; retrying automatically.' : 'Could not load analytics. Retrying automatically, or refresh now.';
      byId('status').className = 'warn';
      byId('refresh').disabled = false;
    },
  });
  byId('refresh').addEventListener('click', () => poller.refresh());
  function visibility() {
    byId('auto-status').textContent = document.hidden ? 'Auto-refresh paused · tab hidden' : 'Auto-refresh · every 15s';
    if (document.hidden) poller.stop(); else poller.start();
  }
  document.addEventListener('visibilitychange', visibility);
  window.addEventListener('pagehide', () => poller.stop());
  visibility();
}());
