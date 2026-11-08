'use strict';
const $ = id => document.getElementById(id);
const W = 256, H = 192, FRAME = W * H, CHUNK = 32;
const KEY_NAMES = ['', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'FIRE', 'ENTER', 'ESC', 'DEL'];

let shardName = '', epLen = [], episodeStart = 0, episodeLen = 0, playing = false, pal = null;
const randInt = n => Math.floor(Math.random() * n);

// Frames are fetched in chunks of CHUNK raw palette frames and kept in a small LRU, so playback and scrubbing are local.
const clips = {
  cache: new Map(), pending: new Map(),
  key: (shard, start) => `${shard}:${start}`,
  get(shard, i) {
    const start = i - i % CHUNK, k = this.key(shard, start);
    if (this.cache.has(k)) return Promise.resolve(this.cache.get(k));
    if (!this.pending.has(k)) {
      this.pending.set(k, fetch(`data/clip?shard=${shard}&start=${start}&n=${CHUNK}`).then(async r => {
        const c = {start, px: new Uint8Array(await r.arrayBuffer()), meta: JSON.parse(r.headers.get('x-meta'))};
        this.cache.set(k, c);
        this.pending.delete(k);
        if (this.cache.size > 64) this.cache.delete(this.cache.keys().next().value);
        return c;
      }));
    }
    return this.pending.get(k);
  },
};

function fill(select, labels) {
  select.innerHTML = '';
  for (const l of labels) select.add(new Option(l));
}

async function pickShard() {
  shardName = $('shard').value;
  epLen = (await (await fetch(`data/shard?name=${shardName}`)).json()).ep_len;
  fill($('episode'), epLen.map((len, i) => `${i} (${len} steps)`));
  pickEpisode();
}

function pickEpisode() {
  const e = $('episode').selectedIndex;
  episodeStart = epLen.slice(0, e).reduce((a, b) => a + b, 0);
  episodeLen = epLen[e];
  $('scrub').max = episodeLen - 1;
  $('scrub').value = 0;
  show();
}

async function show() {
  const step = +$('scrub').value, i = episodeStart + step;
  const c = await clips.get(shardName, i);
  if (+$('scrub').value !== step) return;  // superseded while loading
  const off = (i - c.start) * FRAME;
  if (off >= c.px.length) return;
  const ctx = $('frame').getContext('2d'), img = ctx.createImageData(W, H);
  for (let p = 0; p < FRAME; p++) {
    const rgb = pal[c.px[off + p]];
    img.data[4 * p] = rgb[0]; img.data[4 * p + 1] = rgb[1]; img.data[4 * p + 2] = rgb[2]; img.data[4 * p + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  const [focus, [x, y, btn, key]] = c.meta[i - c.start];
  $('info').textContent = `step ${step}/${episodeLen - 1}  focus=${focus}  mouse=(${x},${y}) btn=${btn}  key=${KEY_NAMES[key] ?? key}`;
  if (i + CHUNK < episodeStart + episodeLen) clips.get(shardName, i + CHUNK);  // prefetch ahead
}

async function randomSpot() {
  $('shard').selectedIndex = randInt($('shard').options.length);
  await pickShard();
  $('episode').selectedIndex = randInt($('episode').options.length);
  pickEpisode();
  $('scrub').value = randInt(episodeLen);
  show();
}

async function init() {
  const [index, config] = await Promise.all([fetch('data/index').then(r => r.json()), fetch('config').then(r => r.json())]);
  pal = config.pal;
  if (!index.shards.length) { $('info').textContent = 'no shards in ' + index.dir; return; }
  fill($('shard'), index.shards);
  $('shard').onchange = pickShard;
  $('episode').onchange = pickEpisode;
  $('scrub').oninput = show;
  $('random').onclick = randomSpot;
  $('play').onclick = () => { playing = !playing; $('play').textContent = playing ? 'pause' : 'play'; };
  setInterval(() => {
    if (!playing) return;
    $('scrub').value = Math.min(episodeLen - 1, +$('scrub').value + +$('speed').value);
    show();
  }, 66);
  pickShard();
}

init();
