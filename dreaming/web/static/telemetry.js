'use strict';
(() => {
  let visitor = crypto.randomUUID();
  try {
    visitor = localStorage.getItem('dreaming-visitor') || visitor;
    localStorage.setItem('dreaming-visitor', visitor);
  } catch (_) {}
  const session = crypto.randomUUID(), started = performance.now();
  let actions = {}, apps = {}, fps = null, latency = null, checkpoint = '';
  let visibleAt = started, duration = 0;
  const elapsed = () => (duration + (visibleAt === null ? 0 : performance.now() - visibleAt)) / 1000;
  function send(kind) {
    const body = JSON.stringify({visitor, session, page: location.pathname, kind,
      duration: elapsed(), actions, apps, fps, latency_ms: latency, checkpoint});
    const blob = new Blob([body], {type: 'application/json'});
    if (!navigator.sendBeacon('telemetry/events', blob)) {
      fetch('telemetry/events', {method: 'POST', body: blob, keepalive: true}).catch(() => {});
    }
    actions = {}; apps = {}; fps = null; latency = null;
  }
  window.telemetry = {
    action(name) { actions[name] = (actions[name] || 0) + 1; },
    frame(value) { fps = value; },
    server(value) {
      latency = value.latency_ms;
      if (value.app) apps[value.app] = (apps[value.app] || 0) + 1;
    },
    checkpoint(value) { checkpoint = value; },
  };
  document.addEventListener('click', e => {
    const control = e.target.closest('button, a, input[type=checkbox]');
    if (control) window.telemetry.action(control.id || control.tagName.toLowerCase());
  });
  document.addEventListener('change', e => {
    if (e.target.id) window.telemetry.action(e.target.id);
  });
  document.addEventListener('visibilitychange', () => {
    if (document.hidden && visibleAt !== null) {
      duration += performance.now() - visibleAt; visibleAt = null; send('heartbeat');
    } else if (!document.hidden && visibleAt === null) visibleAt = performance.now();
  });
  window.addEventListener('pagehide', () => send('pagehide'));
  setInterval(() => { if (!document.hidden) send('heartbeat'); }, 15000);
  send('pageview');
})();
