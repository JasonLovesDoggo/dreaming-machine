"""Play the real micro-OS and the neural one side by side.
  python -m dreaming.demo --ckpt runs/modal/main/ckpt.pt --port 8000   (omit --ckpt for real OS only)"""
import argparse
import json
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from .os_sim import MicroOS, obs_to_pixels
from .dataset import OBS_KEYS
from .tiles import PAL

app = FastAPI()
STATE = {}

PAGE = """<!doctype html><html><head><title>Dreaming Machine</title><style>
body{background:#181a20;color:#ddd;font-family:monospace;margin:20px}
canvas{image-rendering:pixelated;width:768px;height:576px;border:2px solid #444;cursor:none}
.row{display:flex;gap:24px}button{font-family:monospace;padding:6px 12px;margin:8px 8px 0 0}
</style></head><body>
<h2>Dreaming Machine: real simulator (left) vs world model (right)</h2>
<div class="row"><div><div>REAL (deterministic code)</div><canvas id="real" width="256" height="192"></canvas></div>
<div><div>NEURAL (world model, __NEURAL__)</div><canvas id="neural" width="256" height="192"></canvas></div></div>
<div><button id="sync">Resync neural to real</button><button id="reset">Reboot both</button>
<span id="info"></span></div>
<p>Mouse on either screen. Keys: arrows (Doom move/turn, Explorer select), Space = fire, Enter, Esc = close window, Delete.</p>
<script>
const PAL=__PAL__;
const KEYS={ArrowUp:1,ArrowDown:2,ArrowLeft:3,ArrowRight:4," ":5,Control:5,Enter:6,Escape:7,Delete:8,Backspace:8};
let mx=128,my=96,btn=0,key=0,held=[],cmd=null;
const cv={real:document.getElementById('real'),neural:document.getElementById('neural')};
for(const c of Object.values(cv)){
 c.addEventListener('mousemove',e=>{const r=c.getBoundingClientRect();mx=Math.floor((e.clientX-r.left)/r.width*256);my=Math.floor((e.clientY-r.top)/r.height*192);});
 c.addEventListener('mousedown',e=>{btn=1;e.preventDefault();}); c.addEventListener('contextmenu',e=>e.preventDefault());}
window.addEventListener('mouseup',()=>btn=0);
window.addEventListener('keydown',e=>{if(KEYS[e.key]){e.preventDefault();if(!held.includes(KEYS[e.key]))held.push(KEYS[e.key]);}});
window.addEventListener('keyup',e=>{held=held.filter(k=>k!==KEYS[e.key]);});
document.getElementById('sync').onclick=()=>cmd='sync'; document.getElementById('reset').onclick=()=>cmd='reset';
function draw(c,bytes,off){const ctx=c.getContext('2d');const img=ctx.createImageData(256,192);
 for(let i=0;i<256*192;i++){const p=PAL[bytes[off+i]];img.data[4*i]=p[0];img.data[4*i+1]=p[1];img.data[4*i+2]=p[2];img.data[4*i+3]=255;}
 ctx.putImageData(img,0,0);}
const ws=new WebSocket((location.protocol==='https:'?'wss://':'ws://')+location.host+location.pathname.replace(/\\/$/,'')+'/ws');
ws.binaryType='arraybuffer'; let t0=performance.now();
function send(){const k=held.length?held[held.length-1]:0;ws.send(JSON.stringify({x:mx,y:my,btn:btn,key:k,cmd:cmd}));cmd=null;}
ws.onopen=send;
ws.onmessage=e=>{const b=new Uint8Array(e.data);draw(cv.real,b,0);if(b.length>49152)draw(cv.neural,b,49152);
 const dt=performance.now()-t0;t0=performance.now();document.getElementById('info').textContent=` ${(1000/dt).toFixed(1)} fps`;
 setTimeout(send,Math.max(0,66-dt));};
</script></body></html>"""


class Neural:
    def __init__(self, ckpt, device):
        from .model import WorldModel, Config
        ck = torch.load(ckpt, map_location=device)
        self.model = WorldModel(Config(**ck['cfg'])).to(device).eval()
        self.model.load_state_dict(ck['model'])
        self.device = device

    def sync(self, obs):
        self.obs = {k: np.array(obs[k]) for k in OBS_KEYS}
        self.mem = self.model.init_memory(1)
        self.prev_btn = 0

    @torch.no_grad()
    def step(self, action, temperature=0.0):
        from .model import decode
        d = self.device
        x = {k: torch.from_numpy(self.obs[k].astype(np.int64)).unsqueeze(0).to(d) for k in OBS_KEYS}
        x['grid'] = x['grid'].flatten(1)
        x['action'] = torch.tensor([action], dtype=torch.long, device=d)
        x['prev_btn'] = torch.tensor([self.prev_btn], device=d)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=d == 'cuda'):
            out, self.mem = self.model(x, self.mem)
        y = {k: v[0].cpu().numpy() for k, v in decode(out, temperature).items()}
        y['grid'] = y['grid'].reshape(24, 32)
        self.obs = {k: y[k] for k in OBS_KEYS}
        self.prev_btn = action[2]
        return self.obs


@app.get('/')
def index():
    page = PAGE.replace('__PAL__', json.dumps(PAL.tolist()))
    return HTMLResponse(page.replace('__NEURAL__', 'loaded' if STATE.get('neural') else 'no checkpoint'))


@app.websocket('/ws')
async def ws(sock: WebSocket):
    await sock.accept()
    real = MicroOS()
    obs = real.reset()
    neural = STATE.get('neural')
    if neural:
        neural.sync(obs)
    while True:
        msg = json.loads(await sock.receive_text())
        if msg.get('cmd') == 'reset':
            obs = real.reset()
            if neural: neural.sync(obs)
        elif msg.get('cmd') == 'sync' and neural:
            neural.sync(obs)
        a = (int(msg['x']), int(msg['y']), int(msg['btn']), int(msg['key']))
        obs = real.step(a)
        frames = [obs_to_pixels(obs)]
        if neural:
            frames.append(obs_to_pixels(neural.step(a)))
        await sock.send_bytes(b''.join(f.astype(np.uint8).tobytes() for f in frames))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='')
    ap.add_argument('--port', type=int, default=8000)
    a = ap.parse_args()
    if a.ckpt:
        STATE['neural'] = Neural(a.ckpt, 'cuda' if torch.cuda.is_available() else 'cpu')
    uvicorn.run(app, host='0.0.0.0', port=a.port)


if __name__ == '__main__':
    main()
