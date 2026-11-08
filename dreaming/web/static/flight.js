'use strict';
const keys = {ArrowUp: 'UP', ArrowDown: 'DOWN', ArrowLeft: 'LEFT', ArrowRight: 'RIGHT', Enter: 'ENTER', w: 'W', s: 'S', a: 'A', d: 'D'};
let session = null, held = '', reset = false;
const status = document.getElementById('status');
document.addEventListener('keydown', event => {
  const key = keys[event.key] || keys[event.key.toLowerCase()];
  if (!key) return;
  event.preventDefault();
  if (key === 'ENTER') { if (!event.repeat) reset = true; }
  else held = key;
});
document.addEventListener('keyup', event => {
  if ((keys[event.key] || keys[event.key.toLowerCase()]) === held) held = '';
});
window.addEventListener('blur', () => { held = ''; });
document.getElementById('reset').onclick = () => { reset = true; };
async function tick() {
  try {
    const response = await fetch('/flight/api/step', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({session, key: held, reset})
    });
    reset = false;
    if (!response.ok) {
      if (response.status === 404) session = null;
      throw new Error(`Flight request failed (${response.status})`);
    }
    const data = await response.json();
    session = data.session;
    document.getElementById('frame').src = data.frame;
    status.textContent = data.status;
  } catch (error) { status.textContent = `${error.message} — retrying`; }
  setTimeout(tick, 80);
}
tick();
