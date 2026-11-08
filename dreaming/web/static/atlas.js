'use strict';
const SHEET_COLS = 64, CELL = 24;

function el(tag, props = {}) {
  return Object.assign(document.createElement(tag), props);
}

function addSection(root, s) {
  const range = `ids ${s.start}-${s.end - 1}`;
  const count = `${s.end - s.start} tiles${s.shown < s.end - s.start ? ', showing ' + s.shown : ''}`;
  root.append(el('h3', {textContent: `${[].concat(s.name).join(' / ')}: ${range} (${count})`}));
  const img = el('img', {className: 'sheet pixel', src: `sprites/sheet?start=${s.start}&n=${s.shown}`});
  const info = el('div');
  img.onmousemove = e => {
    const r = img.getBoundingClientRect();
    const id = s.start + Math.floor((e.clientY - r.top) / CELL) * SHEET_COLS + Math.floor((e.clientX - r.left) / CELL);
    info.textContent = id < s.start + s.shown ? 'tile id ' + id : '';
  };
  root.append(img, info);
}

fetch('sprites/index').then(r => r.json()).then(({sections}) => {
  const root = document.getElementById('sections');
  sections.forEach(s => addSection(root, s));
  root.append(el('h3', {textContent: 'Doom: wall textures (lit/dark side, 16x16 shown 4x); sprite kinds 1-3 at sizes ' +
                                     '6/12/24/40 drawn by render_tokens; bottom row: real frames'}));
  root.append(el('img', {className: 'sheet pixel', src: 'sprites/doom'}));
});
