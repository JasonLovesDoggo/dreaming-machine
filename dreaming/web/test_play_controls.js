'use strict';
// Run with node dreaming/web/test_play_controls.js (no browser or dependencies).
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

const html = fs.readFileSync(path.join(__dirname, 'static/play.html'), 'utf8');
const controls = Object.fromEntries([...html.matchAll(/<\w+[^>]*\bid="([^"]+)"[^>]*>/g)].map(([tag, id]) => [id, {
  id, value: tag.match(/\bvalue="([^"]+)"/)?.[1] || '', checked: /\bchecked\b/.test(tag),
  style: {}, textContent: '', classList: {toggle() {}}, addEventListener() {},
}]));
const ctx = {beginPath() {}, moveTo() {}, lineTo() {}, stroke() {}, clearRect() {}};
for (const side of ['real', 'neural']) {
  const canvases = Object.fromEntries(['sim', 'grid', 'ids'].map(c => ['.' + c, {
    style: {}, getContext: () => ctx, addEventListener() {},
  }]));
  controls['screen-' + side].querySelector = c => canvases[c];
}
const options = [];
controls.ckpt.add = option => options.push(option);
const sandbox = {
  document: {getElementById: id => { assert.ok(controls[id], id); return controls[id]; }},
  location: {search: '?hz=30', pathname: '/', protocol: 'http:', host: 'localhost', href: 'http://localhost/?hz=30'},
  URLSearchParams, URL, fetch: () => new Promise(() => {}),
  WebSocket: class {static OPEN = 1; readyState = 1;},
  Option: function(label, value, defaultSelected, selected) { Object.assign(this, {label, value, selected}); },
  window: {}, history: {replaceState() {}}, setInterval: () => 1, clearInterval() {},
};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(path.join(__dirname, 'static/play.js'), 'utf8'), sandbox);
vm.runInContext(`cfg = {ckpts: ['ckpt_005000.pt', 'run/ckpt_010000.pt'], current: 'run/ckpt_010000.pt'};
  fillCheckpoints(); bindOverlays(); connect();`, sandbox);
assert.deepEqual(options.map(o => o.value), ['run/ckpt_010000.pt', 'ckpt_005000.pt']);
assert.equal(options[0].selected, true);
assert.equal(options[1].selected, false);
assert.equal(vm.runInContext('cfg.ckpts[0]', sandbox), 'ckpt_005000.pt', 'catalog is not mutated');
assert.equal(controls['show-grid'].checked, false);
assert.equal(controls['show-ids'].checked, false);
assert.equal(controls['ids-opacity-control'].hidden, true);
assert.equal(controls['screen-real'].querySelector('.ids').style.display, 'none');
assert.equal(controls['fps-target'].value, '30');
for (const [input, expected] of [['1', '5'], ['99', '30'], ['', '15']]) {
  controls['fps-target'].value = input;
  controls['fps-target'].onchange();
  assert.equal(controls['fps-target'].value, expected);
}
controls['show-ids'].onchange({target: {checked: true}});
assert.equal(controls['ids-opacity-control'].hidden, false);
assert.equal(controls['screen-real'].querySelector('.ids').style.display, 'block');
controls['ids-opacity'].value = '0.35';
controls['ids-opacity'].oninput();
assert.equal(controls['ids-opacity-value'].textContent, '0.35');
assert.equal(controls['screen-real'].querySelector('.ids').style.opacity, 0.35);
controls['show-grid'].onchange({target: {checked: true}});
assert.equal(controls['screen-real'].querySelector('.grid').style.display, 'block');
console.log('play controls ok');
