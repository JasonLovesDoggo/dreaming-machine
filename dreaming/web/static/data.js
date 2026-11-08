'use strict';
const $ = id => document.getElementById(id);
const W = 256, H = 192, FRAME = W * H, CHUNK = 32;
const KEY_NAMES = ['', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'FIRE', 'ENTER', 'ESC', 'DEL'];
let dataset = 'main', shardName = '', epLen = [], episodeStart = 0, episodeLen = 0;
let playing = false, pal = null, generation = 0, controller = new AbortController(), frames = 0;
const randInt = n => Math.floor(Math.random() * n);
const clips = new Map(), pending = new Map();

function change() {
  generation++;
  controller.abort();
  controller = new AbortController();
  pending.clear();
  playing = false;
  $('play').textContent = 'play';
  return generation;
}

function status(message, retry = null) {
  $('status').textContent = message;
  $('retry').hidden = !retry;
  $('retry').onclick = retry;
}

async function request(path, params = {}) {
  const response = await fetch(`${path}?${new URLSearchParams(params)}`, {signal: controller.signal});
  if (!response.ok) throw new Error(`HTTP ${response.status}: ${await response.text()}`);
  return response;
}

async function attempt(work, retry, token = generation) {
  status('loading…');
  try { await work(); } catch (error) {
    if (token === generation && error.name !== 'AbortError') {
      playing = false;
      $('play').textContent = 'play';
      status(`Error: ${error.message}`, retry);
    }
  }
}

function fill(select, labels) {
  select.innerHTML = '';
  for (const label of labels) select.add(new Option(label));
}

async function getClip(i, token) {
  const start = i - i % CHUNK, key = `${dataset}:${shardName}:${start}`;
  if (clips.has(key)) return clips.get(key);
  if (!pending.has(key)) {
    const promise = (async () => {
      const response = await request('data/clip', {dataset, shard: shardName, start, n: CHUNK});
      const px = new Uint8Array(await response.arrayBuffer());
      const meta = JSON.parse(response.headers.get('x-meta'));
      const expected = Math.min(CHUNK, frames - start);
      if (!Array.isArray(meta) || meta.length !== expected || px.length !== expected * FRAME ||
          meta.some(m => !Array.isArray(m) || m.length !== 2 || !Array.isArray(m[1]) || m[1].length !== 4)) {
        throw new Error('invalid clip metadata or frame size');
      }
      const clip = {start, px, meta};
      if (token === generation) {
        clips.set(key, clip);
        if (clips.size > 64) clips.delete(clips.keys().next().value);
      }
      return clip;
    })();
    pending.set(key, promise);
    promise.finally(() => { if (pending.get(key) === promise) pending.delete(key); }).catch(() => {});
  }
  return pending.get(key);
}

async function show() {
  if (!episodeLen || !pal) return;
  const token = generation, step = +$('scrub').value, i = episodeStart + step;
  await attempt(async () => {
    const clip = await getClip(i, token);
    if (token !== generation || +$('scrub').value !== step) return;
    const off = (i - clip.start) * FRAME;
    const ctx = $('frame').getContext('2d'), img = ctx.createImageData(W, H);
    for (let p = 0; p < FRAME; p++) {
      const rgb = pal[clip.px[off + p]];
      if (!rgb) throw new Error('invalid palette index');
      img.data[4 * p] = rgb[0]; img.data[4 * p + 1] = rgb[1]; img.data[4 * p + 2] = rgb[2]; img.data[4 * p + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
    const [focus, [x, y, btn, key]] = clip.meta[i - clip.start];
    $('info').textContent = `${dataset} / ${shardName}: ${frames} frames, ${epLen.length} episodes | step ${step}/${episodeLen - 1} focus=${focus} mouse=(${x},${y}) btn=${btn} key=${KEY_NAMES[key] ?? key}`;
    status('');
  }, show, token);
}

async function pickEpisode(random = false) {
  change();
  const e = $('episode').selectedIndex;
  episodeStart = epLen.slice(0, e).reduce((a, b) => a + b, 0);
  episodeLen = epLen[e] || 0;
  $('scrub').max = Math.max(0, episodeLen - 1);
  $('scrub').value = random === true ? randInt(episodeLen) : 0;
  if (!episodeLen) { status('no frames in episode'); return; }
  await show();
}

async function pickShard(random = false) {
  const token = change();
  episodeLen = 0;
  fill($('episode'), []);
  shardName = $('shard').value;
  if (!shardName) return;
  await attempt(async () => {
    const result = await (await request('data/shard', {dataset, name: shardName})).json();
    if (token !== generation) return;
    epLen = result.ep_len;
    frames = result.frames;
    fill($('episode'), epLen.map((len, i) => `${i} (${len} steps)`));
    if (random === true) $('episode').selectedIndex = randInt(epLen.length);
    await pickEpisode(random === true);
  }, pickShard, token);
}

async function pickDataset() {
  const token = change();
  dataset = $('dataset').value;
  episodeLen = 0;
  fill($('shard'), []); fill($('episode'), []);
  $('info').textContent = '';
  $('frame').getContext('2d').clearRect(0, 0, W, H);
  await attempt(async () => {
    const index = await (await request('data/index', {dataset})).json();
    if (token !== generation) return;
    if (!index.shards.length) { status(`no shards in ${index.dir}`); return; }
    fill($('shard'), index.shards);
    await pickShard();
  }, pickDataset, token);
}

async function randomSpot() {
  if (!$('shard').options.length) return;
  $('shard').selectedIndex = randInt($('shard').options.length);
  await pickShard(true);
}

async function init() {
  const token = change();
  await attempt(async () => {
    const [datasets, config] = await Promise.all([
      request('data/datasets').then(r => r.json()), request('config').then(r => r.json()),
    ]);
    if (token !== generation) return;
    if (!Array.isArray(config.pal) || !config.pal.length) throw new Error('invalid palette');
    pal = config.pal;
    $('dataset').innerHTML = '';
    for (const d of datasets) {
      const option = new Option(`${d.id} (${d.dir}, ${d.shards} shards)${!d.available ? ' (missing)' : !d.shards ? ' (empty)' : ''}`, d.id);
      option.disabled = !d.available || !d.shards;
      $('dataset').add(option);
    }
    $('dataset').value = 'main';
    await pickDataset();
  }, init, token);
}

$('dataset').onchange = pickDataset;
$('shard').onchange = pickShard;
$('episode').onchange = pickEpisode;
$('scrub').oninput = show;
$('random').onclick = randomSpot;
$('play').onclick = () => { if (episodeLen) { playing = !playing; $('play').textContent = playing ? 'pause' : 'play'; } };
setInterval(() => {
  if (!playing) return;
  $('scrub').value = Math.min(episodeLen - 1, +$('scrub').value + +$('speed').value);
  show();
}, 66);
init();
