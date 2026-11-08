'use strict';
// Run with `node dreaming/web/test_training_core.js` (test_training.py does this when node is installed).
const assert = require('assert');
const C = require('./static/training_core.js');

const cfg = {frame_samples_base: 960000, frame_samples_base_iter: 5000, samples_per_update: 4608};
assert.strictEqual(C.frameSamplesAt(cfg, 5000), 960000);
assert.strictEqual(C.frameSamplesAt(cfg, 5025), 960000 + 25 * 4608);
assert.strictEqual(C.frameSamplesAt(cfg, 4999), null, 'no guessing before the baseline');
assert.strictEqual(C.frameSamplesAt({samples_per_update: 4608}, 6000), null, 'missing metadata stays unavailable');
assert.strictEqual(C.frameSamplesAt(null, 6000), null);
assert.strictEqual(C.frameSamplesAt(cfg, null), null);

const z = C.createZoom();
assert.strictEqual(z.active(), false);
assert.strictEqual(z.set('x:it', 100, 50), false, 'inverted range rejected');
assert.strictEqual(z.set('x:it', NaN, 50), false);
assert.strictEqual(z.set('x:it', 100, 100), false, 'empty range rejected');
assert.strictEqual(z.set('x:it', 100, 900), true);
assert.strictEqual(z.set('step:30', 1, 10), true);
assert.deepStrictEqual(z.get('x:it'), {min: 100, max: 900}, 'survives re-render');
z.reset('step:30');
assert.strictEqual(z.get('step:30'), null);
assert.strictEqual(z.active(), true);
z.reset();
assert.strictEqual(z.get('x:it'), null, 'global reset clears every axis');
assert.strictEqual(z.active(), false);

assert.strictEqual(C.fmtSci(0.0004), '4.00e-4');
assert.strictEqual(C.fmtSci(4e-5), '4.00e-5');
assert.strictEqual(C.fmtSci(null), '-');
assert.strictEqual(C.fmtInt(1234567), '1,234,567');
assert.strictEqual(C.fmtCount(960000), '960k');
assert.strictEqual(C.fmtCount(23040000), '23M');
assert.strictEqual(C.fmtDuration(1280), '21m 20s');
assert.strictEqual(C.fmtDuration(7260), '2h 01m');
assert.strictEqual(C.fmtDuration(-1), '-');
assert.strictEqual(C.last([1, 2, null]), 2);
console.log('training_core ok');
