'use strict';
const $ = id => document.getElementById(id);

const W = 256, H = 192, COLS = 32, ROWS = 24, CELL = 24;
const FRAME_BYTES = W * H, TILES = COLS * ROWS;
const requestedHz = Number(new URLSearchParams(location.search).get('hz') || 15);
let targetHz = Math.max(5, Math.min(30, Number.isFinite(requestedHz) ? requestedHz : 15));
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
let palette;
const renderCache = new WeakMap();

/* ---------- input ---------- */

const input = {x: W / 2, y: H / 2, btn: 0, btnQueue: [], typed: [], held: [], cmd: null};

function bindInput() {
  let pointer = null;
  function position(el, e) {
    const r = el.getBoundingClientRect();
    input.x = Math.max(0, Math.min(W - 1, Math.floor((e.clientX - r.left) / r.width * W)));
    input.y = Math.max(0, Math.min(H - 1, Math.floor((e.clientY - r.top) / r.height * H)));
  }
  function release() {
    if (pointer) {
      input.btnQueue.push({btn: 0, x: input.x, y: input.y});
      pointer = null;
    }
  }
  function cancel() {
    release();
    input.held = [];
    input.typed = [];
  }
  for (const s of Object.values(screens)) {
    s.sim.style.touchAction = 'none';
    s.sim.addEventListener('pointermove', e => {
      if (!pointer || pointer.id === e.pointerId) position(s.sim, e);
    });
    s.sim.addEventListener('pointerdown', e => {
      if (e.button !== 0 || !e.isPrimary || pointer) return;
      position(s.sim, e);
      pointer = {id: e.pointerId, el: s.sim};
      s.sim.setPointerCapture(e.pointerId);
      input.btnQueue.push({btn: 1, x: input.x, y: input.y});
      e.preventDefault();
    });
    s.sim.addEventListener('pointercancel', cancel);
    s.sim.addEventListener('lostpointercapture', release);
    s.sim.addEventListener('contextmenu', e => e.preventDefault());
  }
  window.addEventListener('pointerup', e => {
    if (e.button !== 0 || pointer?.id !== e.pointerId) return;
    position(pointer.el, e);
    release();
  });
  window.addEventListener('blur', cancel);
  window.addEventListener('keydown', e => {
    if (e.target.tagName === 'SELECT' || e.target.id === 'fps-target' || e.target.id === 'ids-opacity') return;
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
  const transition = input.btnQueue.shift();
  if (transition) input.btn = transition.btn;
  const key = input.typed.length ? input.typed.shift() : (input.held.at(-1) || 0);
  return {x: transition ? transition.x : input.x, y: transition ? transition.y : input.y, btn: input.btn, key};
}

/* ---------- screens ---------- */

function drawScreen(canvas, bytes, offset) {
  let cached = renderCache.get(canvas);
  if (!cached) {
    const ctx = canvas.getContext('2d'), img = ctx.createImageData(W, H);
    cached = {ctx, img, pixels: new Uint32Array(img.data.buffer)};
    renderCache.set(canvas, cached);
  }
  const {ctx, img, pixels} = cached;
  for (let i = 0; i < FRAME_BYTES; i++) {
    pixels[i] = palette[bytes[offset + i]];
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
  opacity: 1,
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

  applyOpacity() {
    for (const s of Object.values(screens)) {
      s.ids.style.opacity = this.on ? this.opacity : '';
      s.sim.style.opacity = this.on ? 1 - 0.7 * this.opacity : '';
    }
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
    $('ids-opacity-control').hidden = !on;
    this.applyOpacity();
  },
};

function bindOverlays() {
  const opacityControl = document.createElement('label');
  opacityControl.id = 'ids-opacity-control';
  opacityControl.append(' opacity ');
  const opacityInput = document.createElement('input');
  Object.assign(opacityInput, {id: 'ids-opacity', type: 'range', min: '0', max: '1', step: '0.05', value: '1'});
  const opacityValue = document.createElement('span');
  opacityValue.textContent = '1';
  opacityControl.append(opacityInput, opacityValue);
  $('show-ids').closest('label').after(opacityControl);
  opacityInput.oninput = () => {
    tileIds.opacity = Number(opacityInput.value);
    opacityValue.textContent = opacityInput.value;
    tileIds.applyOpacity();
  };
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
    const run = name.includes('/') ? name.slice(0, name.lastIndexOf('/')) + ' · ' : '';
    sel.add(new Option(m ? run + 'iter ' + Number(m[1]).toLocaleString() : name, name, false, name === cfg.current));
  }
  if (!cfg.ckpts.length) sel.add(new Option('none'));
  $('neural-status').textContent = cfg.current ? 'loaded' : 'no checkpoint';
}

/* ---------- websocket loop ---------- */

function connect() {
  const base = location.pathname.replace(/\/$/, '');
  const ws = new WebSocket((location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + base + '/ws');
  ws.binaryType = 'arraybuffer';
  const pending = [], compressed = typeof DecompressionStream === 'function';
  let clock, messages = Promise.resolve(), lastFrame = null, avgMs = 1000 / targetHz;
  const targetControl = document.createElement('label');
  targetControl.style.whiteSpace = 'nowrap';
  targetControl.title = 'One frame = one simulator step, so higher values also speed up the game.';
  const targetInput = document.createElement('input');
  Object.assign(targetInput, {id: 'fps-target', type: 'number', min: '5', max: '30', step: '1', value: String(targetHz)});
  targetControl.append('target ', targetInput, ' fps');
  $('fps').after(targetControl);
  targetInput.onchange = () => {
    const value = Number(targetInput.value);
    targetHz = Math.max(5, Math.min(30, targetInput.value !== '' && Number.isFinite(value) ? value : 15));
    targetInput.value = String(targetHz);
    clearInterval(clock);
    if (ws.readyState === WebSocket.OPEN) clock = setInterval(send, 1000 / targetHz);
    const url = new URL(location.href);
    url.searchParams.set('hz', targetHz);
    history.replaceState(null, '', url);
    window.telemetry?.action('fps-target');
  };

  function send() {
    if (ws.readyState !== WebSocket.OPEN || pending.length >= 3) return;
    const temp = parseFloat($('temp').value);
    $('temp-value').textContent = temp;
    pending.push(performance.now());
    const action = nextAction();
    if (action.btn) window.telemetry?.action('mouse');
    if (action.key) window.telemetry?.action('key');
    window.telemetry?.checkpoint($('ckpt').value);
    ws.send(JSON.stringify({...action, cmd: input.cmd, temp, ckpt: $('ckpt').value, ids: tileIds.on, telemetry: true, z: compressed ? 1 : 0}));
    input.cmd = null;
  }

  ws.onopen = () => { send(); clock = setInterval(send, 1000 / targetHz); };
  ws.onclose = () => clearInterval(clock);
  ws.onmessage = e => {
    const receivedAt = performance.now();
    messages = messages.then(() => receive(e.data, receivedAt)).catch(error => {
      console.error('play websocket message failed', error);
      clearInterval(clock);
      ws.close();
    });
  };
  async function receive(data, receivedAt) {
    if (typeof data === 'string') {
      window.telemetry?.server(JSON.parse(data));
      return;
    }
    if (new Uint8Array(data)[0] === 255) {
      data = await new Response(new Blob([data.slice(1)]).stream().pipeThrough(new DecompressionStream('deflate'))).arrayBuffer();
    }
    const rtt = receivedAt - pending.shift();
    const buf = new Uint8Array(data), n = buf[0], hasIds = buf[1];
    SIDES.slice(0, n).forEach((side, k) => drawScreen(screens[side].sim, buf, 2 + k * FRAME_BYTES));
    if (hasIds && tileIds.on) {
      const ids = new Uint16Array(data.slice(2 + n * FRAME_BYTES));
      SIDES.slice(0, n).forEach((side, k) => tileIds.draw(side, ids.slice(k * TILES, (k + 1) * TILES)));
      tileIds.showHover();
    }
    const now = performance.now();
    if (lastFrame !== null) avgMs = avgMs * 0.9 + (now - lastFrame) * 0.1;
    lastFrame = now;
    $('fps').textContent = ` ${(1000 / avgMs).toFixed(1)} fps · ${Math.round(rtt)} ms`;
    window.telemetry?.frame(1000 / avgMs);
  }
}

fetch('config').then(r => r.json()).then(c => {
  cfg = c;
  const rgba = new Uint8Array(cfg.pal.length * 4);
  cfg.pal.forEach((color, i) => { rgba.set(color, i * 4); rgba[i * 4 + 3] = 255; });
  palette = new Uint32Array(rgba.buffer);
  fillCheckpoints();
  bindInput();
  bindOverlays();
  connect();
});
