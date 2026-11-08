'use strict';
const $ = id => document.getElementById(id);
const KEY_NAMES = ['', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'FIRE', 'ENTER', 'ESC', 'DEL'];

let shards = [], episodeStart = 0, episodeLen = 0, playing = false, busy = false;

const shard = () => shards[$('shard').selectedIndex];
const randInt = n => Math.floor(Math.random() * n);

function fill(select, labels) {
  select.innerHTML = '';
  for (const l of labels) select.add(new Option(l));
}

function pickShard() {
  fill($('episode'), shard().ep_len.map((len, i) => `${i} (${len} steps)`));
  pickEpisode();
}

function pickEpisode() {
  const lens = shard().ep_len, e = $('episode').selectedIndex;
  episodeStart = lens.slice(0, e).reduce((a, b) => a + b, 0);
  episodeLen = lens[e];
  $('scrub').max = episodeLen - 1;
  $('scrub').value = 0;
  show();
}

async function show() {
  if (busy) return;
  busy = true;
  const step = +$('scrub').value;
  const r = await fetch(`data/frame?shard=${shard().name}&i=${episodeStart + step}`);
  const meta = JSON.parse(r.headers.get('x-meta'));
  $('frame').src = URL.createObjectURL(await r.blob());
  const [x, y, btn, key] = meta.action;
  $('info').textContent = `step ${step}/${episodeLen - 1}  focus=${meta.focus}  mouse=(${x},${y}) btn=${btn}  key=${KEY_NAMES[key] ?? key}`;
  busy = false;
}

function randomSpot() {
  $('shard').selectedIndex = randInt(shards.length);
  pickShard();
  $('episode').selectedIndex = randInt($('episode').options.length);
  pickEpisode();
  $('scrub').value = randInt(episodeLen);
  show();
}

async function init() {
  const index = await (await fetch('data/index')).json();
  shards = index.shards;
  if (!shards.length) { $('info').textContent = 'no shards in ' + index.dir; return; }
  fill($('shard'), shards.map(s => s.name));
  $('shard').onchange = pickShard;
  $('episode').onchange = pickEpisode;
  $('scrub').oninput = show;
  $('random').onclick = randomSpot;
  $('play').onclick = () => { playing = !playing; $('play').textContent = playing ? 'pause' : 'play'; };
  setInterval(() => {
    if (!playing || busy) return;
    $('scrub').value = Math.min(episodeLen - 1, +$('scrub').value + +$('speed').value);
    show();
  }, 66);
  pickShard();
}

init();
