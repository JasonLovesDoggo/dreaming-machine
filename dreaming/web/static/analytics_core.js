'use strict';
(function (root, factory) {
  const core = factory();
  if (typeof module === 'object' && module.exports) module.exports = core;
  else root.AnalyticsCore = core;
}(typeof self !== 'undefined' ? self : this, function () {
  const valid = v => typeof v === 'number' && Number.isFinite(v) && v >= 0;
  function fmtNumber(v, digits = 0) {
    return valid(v) ? v.toLocaleString('en-US', {maximumFractionDigits: digits}) : '—';
  }
  function fmtDuration(v) {
    if (!valid(v)) return '—';
    const s = Math.floor(v);
    if (s < 60) return `${s}s`;
    if (s < 3600) return `${Math.floor(s / 60)}m ${s % 60}s`;
    return `${Math.floor(s / 3600)}h ${Math.floor(s % 3600 / 60)}m`;
  }
  function ranked(counts, limit = 10) {
    if (!counts || typeof counts !== 'object' || Array.isArray(counts)) return [];
    const rows = Object.entries(counts).filter(([, value]) => valid(value));
    const total = rows.reduce((sum, [, value]) => sum + value, 0);
    return rows.sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
      .slice(0, Math.max(0, limit)).map(([name, count]) => ({name, count, share: total ? count / total * 100 : 0}));
  }
  function createPoller({fetch, onData, onError, intervalMs = 15000,
    setTimeout: schedule = setTimeout, clearTimeout: cancel = clearTimeout}) {
    let running = false, inFlight = false, timer = null;
    function clear() { if (timer !== null) cancel(timer); timer = null; }
    async function refresh() {
      if (inFlight) return;
      clear();
      inFlight = true;
      try { const data = await fetch(); onData(data); }
      catch (error) { onError(error); }
      finally {
        inFlight = false;
        if (running) timer = schedule(refresh, intervalMs);
      }
    }
    return {
      refresh,
      start() { if (running) return; running = true; return refresh(); },
      stop() { running = false; clear(); },
    };
  }
  return {fmtNumber, fmtDuration, ranked, createPoller};
}));
