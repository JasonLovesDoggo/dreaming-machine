'use strict';
const $ = id => document.getElementById(id);

const W = 256, H = 192, COLS = 32, ROWS = 24, CELL = 24;
const FRAME_BYTES = W * H, TILES = COLS * ROWS;
const FRAME_MS = 66;
const KEY_CHAR0 = 9;
const HELD_KEYS = {ArrowUp: 1, ArrowDown: 2, ArrowLeft: 3, ArrowRight: 4, Control: 5};
const QUEUED_KEYS = {Enter: 6, Escape: 7, Delete: 8, Backspace: 8};
const SECTION_COLORS = {ui: '#7a3d00', core: '#7a3d00', icon: '#2d5a1e', mini: '#4a4a1e', title: '#14286e', title_in: '#3c3c46', body: '#1d5e2a', sel: '#0b4f6c',
                        tool: '#5b2a6e', desk: '#0d5c5c', hud: '#7a1020', canvas: '#555'};

const SIDES = ['real', 'neural'];
const screens = Object.fromEntries(SIDES.map(side => {
  const el = $('screen-' + side);
  return [side, {el, sim: el.querySelector('.sim'), grid: el.querySelector('.grid'), ids: el.querySelector('.ids')}];
}));
let cfg = null;

/* ---------- input ---------- */

const input = {x: W / 2, y: H / 2, btn: 0, btnQueue: [], typed: [], held: [], cmd: null};

function bindInput() {
  for (const s of Object.values(screens)) {
    s.sim.addEventListener('mousemove', e => {
      const r = s.sim.getBoundingClientRect();
      input.x = Math.max(0, Math.min(W - 1, Math.floor((e.clientX - r.left) / r.width * W)));
      input.y = Math.max(0, Math.min(H - 1, Math.floor((e.clientY - r.top) / r.height * H)));
    });
    s.sim.addEventListener('mousedown', e => { input.btnQueue.push(1); e.preventDefault(); });
    s.sim.addEventListener('contextmenu', e => e.preventDefault());
  }
  window.addEventListener('mouseup', () => input.btnQueue.push(0));
  window.addEventListener('keydown', e => {
    if (e.target.tagName === 'SELECT') return;
    if (HELD_KEYS[e.key]) {
      e.preventDefault();
      if (!input.held.includes(HELD_KEYS[e.key])) input.held.push(HELD_KEYS[e.key]);
    } else if (QUEUED_KEYS[e.key]) {
      e.preventDefault();
      input.typed.push(QUEUED_KEYS[e.key]);
    } else if (e.key.length === 1 && !e.ctrlKey && !e.metaKey && cfg.extra_keys.includes(e.key)) {
      e.preventDefault();
      input.typed.push(cfg.key_extra0 + cfg.extra_keys.indexOf(e.key));
    } else if (e.key.length === 1 && !e.ctrlKey && !e.metaKey) {
      const i = cfg.charset.indexOf(e.key.toUpperCase());
      if (i >= 0) { e.preventDefault(); input.typed.push(KEY_CHAR0 + i); }
    }
  });
  window.addEventListener('keyup', e => { input.held = input.held.filter(k => k !== HELD_KEYS[e.key]); });
  $('sync').onclick = () => { input.cmd = 'sync'; };
  $('reset').onclick = () => { input.cmd = 'reset'; };
  $('ckpt').onchange = () => { input.cmd = 'load'; };
}

function nextAction() {
  if (input.btnQueue.length) input.btn = input.btnQueue.shift();
  const key = input.typed.length ? input.typed.shift() : (input.held.at(-1) || 0);
  return {x: input.x, y: input.y, btn: input.btn, key};
}

/* ---------- screens ---------- */

function drawScreen(canvas, bytes, offset) {
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(W, H);
  for (let i = 0; i < FRAME_BYTES; i++) {
    const p = cfg.pal[bytes[offset + i]];
    img.data.set(p, 4 * i);
    img.data[4 * i + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

function drawTileGrid(canvas) {
  const c = canvas.getContext('2d');
  c.strokeStyle = 'rgba(255,255,0,0.45)';
  c.beginPath();
  for (let x = 0; x <= COLS; x++) { c.moveTo(x * CELL + 0.5, 0); c.lineTo(x * CELL + 0.5, ROWS * CELL); }
  for (let y = 0; y <= ROWS; y++) { c.moveTo(0, y * CELL + 0.5); c.lineTo(COLS * CELL, y * CELL + 0.5); }
  c.stroke();
}

/* ---------- tile-ID overlay ---------- */

const tileIds = {
  on: false,
  last: {},            // side -> Uint16Array of the ids currently drawn
  hover: {side: 'real', idx: -1},
  sectionCache: new Map(),

  section(t) {
    if (!this.sectionCache.has(t)) {
      const s = cfg.sections.find(s => t >= s.start && t < s.end);
      this.sectionCache.set(t, s ? [].concat(s.name) : ['ui']);
    }
    return this.sectionCache.get(t);
  },

  describe(t) {
    const names = this.section(t), s = cfg.sections.find(s => t >= s.start && t < s.end);
    if (names[0] === 'text' && s) {
      const n = cfg.charset.length, o = t - s.start;
      return `text/${names.at(-1)} "${cfg.charset[Math.floor(o / n)]}${cfg.charset[o % n]}"`;
    }
    if (names.at(-1) === 'canvas' && s) return `paint canvas cell ${t - s.start}`;
    return names.join('/');
  },

  // Five-digit ids don't fit on one line of a 24 px cell, so the thousands go on top: 13725 -> "13" / "725".
  drawCell(ctx, i, t) {
    const x = (i % COLS) * CELL, y = Math.floor(i / COLS) * CELL, s = String(t);
    ctx.clearRect(x, y, CELL, CELL);
    ctx.globalAlpha = 0.8;
    ctx.fillStyle = SECTION_COLORS[this.section(t).at(-1)] || '#333';
    ctx.fillRect(x, y, CELL, CELL);
    ctx.globalAlpha = 1;
    ctx.strokeRect(x + 0.5, y + 0.5, CELL - 1, CELL - 1);
    ctx.fillStyle = '#fff';
    if (s.length <= 3) {
      ctx.fillText(s, x + CELL / 2, y + CELL / 2 + 1);
    } else {
      ctx.fillText(s.slice(0, -3), x + CELL / 2, y + 7);
      ctx.fillText(s.slice(-3), x + CELL / 2, y + 18);
    }
  },

  draw(side, ids) {
    const ctx = screens[side].ids.getContext('2d'), prev = this.last[side];
    ctx.font = 'bold 10px monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.strokeStyle = 'rgba(0,0,0,0.6)';
    for (let i = 0; i < TILES; i++) {
      if (!prev || prev[i] !== ids[i]) this.drawCell(ctx, i, ids[i]);
    }
    this.last[side] = ids;
  },

  showHover() {
    const {side, idx} = this.hover, ids = this.last[side];
    if (!this.on || idx < 0 || !ids) return;
    const t = ids[idx];
    $('tileinfo').textContent = `${side} tile (${idx % COLS},${Math.floor(idx / COLS)})  id ${t}  ${this.describe(t)}`;
  },

  toggle(on) {
    this.on = on;
    this.last = {};
    $('tileinfo').textContent = on ? 'hover a tile for its meaning' : '';
    for (const s of Object.values(screens)) {
      s.ids.style.display = on ? 'block' : 'none';
      s.el.classList.toggle('dim', on);
      s.ids.getContext('2d').clearRect(0, 0, s.ids.width, s.ids.height);
    }
  },
};

function bindOverlays() {
  for (const s of Object.values(screens)) drawTileGrid(s.grid);
  $('show-grid').onchange = e => {
    for (const s of Object.values(screens)) s.grid.style.display = e.target.checked ? 'block' : 'none';
  };
  $('show-ids').onchange = e => tileIds.toggle(e.target.checked);
  tileIds.toggle($('show-ids').checked);
  for (const [side, s] of Object.entries(screens)) {
    s.sim.addEventListener('mousemove', e => {
      const r = s.sim.getBoundingClientRect();
      tileIds.hover = {side, idx: Math.floor((e.clientY - r.top) / r.height * ROWS) * COLS +
                                  Math.floor((e.clientX - r.left) / r.width * COLS)};
    });
  }
}

/* ---------- checkpoints ---------- */

function fillCheckpoints() {
  const sel = $('ckpt');
  for (const name of cfg.ckpts) {
    const m = name.match(/ckpt_0*(\d+)\.pt/);
    sel.add(new Option(m ? 'iter ' + Number(m[1]).toLocaleString() : name, name, false, name === cfg.current));
  }
  if (!cfg.ckpts.length) sel.add(new Option('none'));
  $('neural-status').textContent = cfg.current ? 'loaded' : 'no checkpoint';
}

/* ---------- websocket loop ---------- */

function connect() {
  const base = location.pathname.replace(/\/$/, '');
  const ws = new WebSocket((location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + base + '/ws');
  ws.binaryType = 'arraybuffer';
  let sentAt = 0, lastFrame = performance.now(), avgMs = FRAME_MS;

  function send() {
    const temp = parseFloat($('temp').value);
    $('temp-value').textContent = temp;
    sentAt = performance.now();
    const action = nextAction();
    if (action.btn) window.telemetry?.action('mouse');
    if (action.key) window.telemetry?.action('key');
    window.telemetry?.checkpoint($('ckpt').value);
    ws.send(JSON.stringify({...action, cmd: input.cmd, temp, ckpt: $('ckpt').value, ids: tileIds.on, telemetry: true}));
    input.cmd = null;
  }

  ws.onopen = send;
  ws.onmessage = e => {
    if (typeof e.data === 'string') {
      window.telemetry?.server(JSON.parse(e.data));
      return;
    }
    const buf = new Uint8Array(e.data), n = buf[0], hasIds = buf[1];
    SIDES.slice(0, n).forEach((side, k) => drawScreen(screens[side].sim, buf, 2 + k * FRAME_BYTES));
    if (hasIds && tileIds.on) {
      const ids = new Uint16Array(e.data.slice(2 + n * FRAME_BYTES));
      SIDES.slice(0, n).forEach((side, k) => tileIds.draw(side, ids.slice(k * TILES, (k + 1) * TILES)));
      tileIds.showHover();
    }
    const now = performance.now();
    avgMs = avgMs * 0.9 + (now - lastFrame) * 0.1;
    lastFrame = now;
    $('fps').textContent = ` ${(1000 / avgMs).toFixed(1)} fps`;
    window.telemetry?.frame(1000 / avgMs);
    setTimeout(send, Math.max(0, FRAME_MS - (now - sentAt)));
  };
}

fetch('config').then(r => r.json()).then(c => {
  cfg = c;
  fillCheckpoints();
  bindInput();
  bindOverlays();
  connect();
});
