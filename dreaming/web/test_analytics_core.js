'use strict';
const assert = require('assert');
const {fmtNumber, fmtDuration, ranked, createPoller} = require('./static/analytics_core.js');
assert.strictEqual(fmtNumber(12345), '12,345');
assert.strictEqual(fmtNumber(12.345, 1), '12.3');
assert.strictEqual(fmtDuration(3725), '1h 2m');
assert.strictEqual(fmtDuration(65), '1m 5s');
assert.strictEqual(fmtDuration(0), '0s');
for (const value of [null, undefined, NaN, Infinity, -1, '4']) {
  assert.strictEqual(fmtNumber(value), '—');
  assert.strictEqual(fmtDuration(value), '—');
}
for (const value of [null, undefined, [], 'invalid', {}]) assert.deepStrictEqual(ranked(value), []);
assert.deepStrictEqual(ranked({b: 1, a: 3, bad: -2, nope: Infinity, nil: null}, 1), [{name: 'a', count: 3, share: 75}]);
assert.deepStrictEqual(ranked({a: 0}), [{name: 'a', count: 0, share: 0}]);
assert.deepStrictEqual(ranked({b: 2, a: 2}).map(row => row.name), ['a', 'b']);
assert.deepStrictEqual(ranked({a: 1}, 0), []);

(async () => {
  let resolve, reject, requests = 0, errors = 0;
  const received = [], timers = new Map();
  let nextTimer = 0;
  const poller = createPoller({
    fetch() { requests++; return new Promise((yes, no) => { resolve = yes; reject = no; }); },
    onData: data => received.push(data), onError: () => errors++, intervalMs: 15000,
    setTimeout(callback, delay) { assert.strictEqual(delay, 15000); timers.set(++nextTimer, callback); return nextTimer; },
    clearTimeout: id => timers.delete(id),
  });
  const first = poller.start();
  poller.start(); poller.refresh();
  assert.strictEqual(requests, 1);
  assert.strictEqual(timers.size, 0);
  resolve({sessions: 1}); await first;
  assert.deepStrictEqual(received, [{sessions: 1}]);
  assert.strictEqual(timers.size, 1);
  const second = poller.refresh();
  assert.strictEqual(timers.size, 0);
  reject(new Error('offline')); await second;
  assert.strictEqual(errors, 1);
  assert.strictEqual(timers.size, 1);
  const callback = [...timers.values()][0]; timers.clear();
  const third = callback();
  poller.stop(); poller.start(); poller.refresh();
  assert.strictEqual(requests, 3);
  resolve({}); await third;
  assert.strictEqual(timers.size, 1);
  poller.stop();
  assert.strictEqual(timers.size, 0);
  const manual = poller.refresh(); poller.stop(); resolve(null); await manual;
  assert.strictEqual(timers.size, 0);
  console.log('Analytics core: formatting, ranking, null handling and poller tests passed');
})().catch(error => { console.error(error); process.exitCode = 1; });
