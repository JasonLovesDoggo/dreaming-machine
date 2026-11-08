'use strict';
// Pure helpers for the training dashboard (no DOM), shared with the Node unit test.
(function (root, factory) {
  const core = factory();
  if (typeof module === 'object' && module.exports) module.exports = core;
  else root.TrainingCore = core;
}(typeof self !== 'undefined' ? self : this, function () {
  const finite = v => typeof v === 'number' && isFinite(v);

  // Cumulative frame samples at `it`, only from the run's own config row: base + (it - base_iter) * samples_per_update.
  function frameSamplesAt(cfg, it) {
    if (!cfg || !finite(it)) return null;
    const base = cfg.frame_samples_base, baseIt = cfg.frame_samples_base_iter, per = cfg.samples_per_update;
    if (!finite(base) || !finite(baseIt) || !finite(per) || it < baseIt) return null;
    return base + (it - baseIt) * per;
  }

  // Per-axis-group x zoom ranges that survive re-renders; only valid, non-empty ranges are stored.
  function createZoom() {
    const ranges = new Map();
    return {
      get: group => ranges.get(group) || null,
      set(group, min, max) {
        if (!finite(min) || !finite(max) || !(max > min)) return false;
        ranges.set(group, {min, max});
        return true;
      },
      reset(group) { if (group === undefined) ranges.clear(); else ranges.delete(group); },
      active: () => ranges.size > 0,
    };
  }

  function fmtInt(v) {
    return finite(v) ? Math.round(v).toLocaleString('en-US') : '-';
  }

  function fmtCount(v) {
    if (!finite(v)) return '-';
    const a = Math.abs(v);
    for (const [div, suffix] of [[1e12, 'T'], [1e9, 'B'], [1e6, 'M'], [1e3, 'k']]) {
      if (a >= div) return `${+(v / div).toPrecision(3)}${suffix}`;
    }
    return String(+v.toPrecision(3));
  }

  function fmtSci(v) {
    if (!finite(v)) return '-';
    if (v === 0) return '0';
    const [m, e] = v.toExponential(2).split('e');
    return `${m}e${+e}`;
  }

  function fmtDuration(sec) {
    if (!finite(sec) || sec < 0) return '-';
    const s = Math.round(sec);
    if (s < 60) return `${s}s`;
    if (s < 3600) return `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, '0')}s`;
    if (s < 86400) return `${Math.floor(s / 3600)}h ${String(Math.floor(s % 3600 / 60)).padStart(2, '0')}m`;
    return `${Math.floor(s / 86400)}d ${Math.floor(s % 86400 / 3600)}h`;
  }

  // Subtitle for the held-out card: how the latest eval relates to the last-written training row.
  function evalLag(lastIt, evalIt) {
    if (!finite(evalIt)) return null;
    if (!finite(lastIt)) return 'no training rows in this run, so no training position to compare with';
    const d = lastIt - evalIt;
    if (d < 0) return `${fmtInt(-d)} updates ahead of the latest training row (training resumed from an earlier checkpoint)`;
    return d === 0 ? 'at the latest training row' : `${fmtInt(d)} updates behind the latest training row`;
  }

  function last(values) {
    if (!values) return null;
    for (let i = values.length - 1; i >= 0; i--) if (values[i] !== null && values[i] !== undefined) return values[i];
    return null;
  }

  return {frameSamplesAt, evalLag, createZoom, fmtInt, fmtCount, fmtSci, fmtDuration, last};
}));
