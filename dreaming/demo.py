"""Play the real micro-OS and the neural one side by side.
  python -m dreaming.demo --ckpt runs/modal/main/ckpt.pt --port 8000   (omit --ckpt for real OS only)"""
import argparse
import functools
import glob
import io
import os
import json
import random
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse, Response
from .os_sim import MicroOS, obs_to_pixels
from .dataset import OBS_KEYS
from .tiles import PAL, CHARSET

app = FastAPI()
STATE = {}
DATA_DIR = os.environ.get('DATA_DIR', 'data/local')

VIEWER = """<!doctype html><html><head><title>Dreaming Computer: training data</title><style>
body{background:#1b1b1f;color:#ddd;font-family:monospace;margin:16px}img{width:768px;height:576px;image-rendering:pixelated;border:2px solid #444}
select,button,input{font-family:monospace;margin:4px}a{color:#8cf}</style></head><body>
<h2>Dreaming Computer: training data viewer</h2><a href="/">&larr; play the computer</a><br>
shard <select id="shard"></select> episode <select id="ep"></select>
<button id="play">play</button> speed <select id="spd"><option>1</option><option selected>2</option><option>4</option><option>8</option></select>x
<button id="rnd">random spot</button><br>
<input id="fr" type="range" min="0" value="0" style="width:768px"><br><span id="info"></span><br><img id="img">
<script>
let idx=null,playing=false,base=0,len=0;const $=i=>document.getElementById(i);
async function init(){idx=await (await fetch('/data/index')).json();
 for(const s of idx.shards){const o=document.createElement('option');o.textContent=s.name;$('shard').appendChild(o);}
 if(!idx.shards.length){$('info').textContent='no shards in '+idx.dir;return;} pickShard();}
function pickShard(){const s=idx.shards[$('shard').selectedIndex];$('ep').innerHTML='';
 s.ep_len.forEach((l,i)=>{const o=document.createElement('option');o.textContent=i+' ('+l+' steps)';$('ep').appendChild(o);});pickEp();}
function pickEp(){const s=idx.shards[$('shard').selectedIndex],e=$('ep').selectedIndex;base=s.ep_len.slice(0,e).reduce((a,b)=>a+b,0);len=s.ep_len[e];
 $('fr').max=len-1;$('fr').value=0;show();}
let busy=false;async function show(){if(busy)return;busy=true;const s=idx.shards[$('shard').selectedIndex];const f=+$('fr').value;
 const r=await fetch(`/data/frame?shard=${s.name}&i=${base+f}`);const m=JSON.parse(r.headers.get('x-meta'));
 $('img').src=URL.createObjectURL(await r.blob());$('info').textContent=`step ${f}/${len-1}  focus=${m.focus}  action x=${m.a[0]} y=${m.a[1]} btn=${m.a[2]} key=${m.key}`;busy=false;}
$('shard').onchange=pickShard;$('ep').onchange=pickEp;$('fr').oninput=show;
$('play').onclick=()=>{playing=!playing;$('play').textContent=playing?'pause':'play';};
$('rnd').onclick=()=>{$('shard').selectedIndex=Math.floor(Math.random()*idx.shards.length);pickShard();
 $('ep').selectedIndex=Math.floor(Math.random()*$('ep').options.length);pickEp();$('fr').value=Math.floor(Math.random()*len);show();};
setInterval(()=>{if(playing&&!busy){$('fr').value=Math.min(len-1,+$('fr').value+ +$('spd').value);show();}},66);init();
</script></body></html>"""


@functools.lru_cache(maxsize=3)
def _shard(name):
    z = np.load(os.path.join(DATA_DIR, os.path.basename(name) + '.npz'))
    return {k: z[k] for k in z.files}


@app.get('/data')
def data_page():
    return HTMLResponse(VIEWER)


@app.get('/data/index')
def data_index():
    out = []
    for f in sorted(glob.glob(os.path.join(DATA_DIR, '*.npz'))):
        name = os.path.basename(f)[:-4]
        try:
            with np.load(f) as z:
                out.append(dict(name=name, ep_len=[int(x) for x in z['ep_len']]))
        except Exception:
            pass
    return JSONResponse(dict(dir=DATA_DIR, shards=out))


@app.get('/data/frame')
def data_frame(shard: str, i: int):
    from PIL import Image
    from .os_sim import APPS
    d = _shard(shard)
    i = max(0, min(i, len(d['grid']) - 1))
    obs = {k: d[k][i] for k in OBS_KEYS}
    buf = io.BytesIO()
    Image.fromarray(np.asarray(PAL, np.uint8)[obs_to_pixels(obs)]).save(buf, 'PNG')
    foc = int(d['state'][i][3 * len(APPS)])
    a = [int(v) for v in d['action'][i]]
    key = a[3]
    meta = dict(focus='desktop' if foc == 0 else APPS[foc - 1], a=a, key=key)
    return Response(buf.getvalue(), media_type='image/png', headers={'x-meta': json.dumps(meta)})

PAGE = """<!doctype html><html><head><title>Dreaming Computer</title><style>
body{background:#181a20;color:#ddd;font-family:monospace;margin:20px}
canvas{image-rendering:pixelated;width:768px;height:576px;border:2px solid #444;cursor:none}
.row{display:flex;gap:24px}.scr{position:relative;width:772px;height:580px}.grid,.ids{position:absolute;left:2px;top:2px;width:768px;height:576px;pointer-events:none;display:none;border:none;cursor:auto}button{font-family:monospace;padding:6px 12px;margin:8px 8px 0 0}
</style></head><body>
<h2>Dreaming Computer: real simulator (left) vs world model (right)</h2>
<div class="row"><div><div>REAL (deterministic code)</div><div class="scr"><canvas id="real" width="256" height="192"></canvas><canvas class="grid" width="768" height="576"></canvas><canvas class="ids" id="ids_real" width="768" height="576"></canvas></div></div>
<div><div>NEURAL (world model, __NEURAL__)</div><div class="scr"><canvas id="neural" width="256" height="192"></canvas><canvas class="grid" width="768" height="576"></canvas><canvas class="ids" id="ids_neural" width="768" height="576"></canvas></div></div></div>
<div><button id="sync">Resync neural to real</button><button id="reset">Reboot both</button>
checkpoint <select id="ckpt"></select> &nbsp; <label><input id="gridt" type="checkbox"> show tile grid</label> <label><input id="idst" type="checkbox"> show tile IDs</label> &nbsp; temperature <input id="temp" type="range" min="0" max="1.5" step="0.1" value="0.6"><span id="tv">0.6</span> <span id="info"></span></div>
<div id="tileinfo" style="height:1.2em;color:#ffd75e"></div><p>Mouse on either screen. Type into Notepad. Arrows: Doom move/turn, Explorer select. Space/Ctrl: fire. Enter, Esc (close window), Backspace.</p>
<p><a href="/data" style="color:#8cf">training data viewer</a> &middot; <a href="/sprites" style="color:#8cf">tile &amp; sprite atlas</a></p>
<script>
const PAL=__PAL__;
const HELD={ArrowUp:1,ArrowDown:2,ArrowLeft:3,ArrowRight:4,Control:5};const QUEUED={Enter:6,Escape:7,Delete:8,Backspace:8};const CHARSET=__CHARSET__;
let mx=128,my=96,btn=0,key=0,held=[],cmd=null,btnq=[],typed=[];
const cv={real:document.getElementById('real'),neural:document.getElementById('neural')};
for(const c of Object.values(cv)){
 c.addEventListener('mousemove',e=>{const r=c.getBoundingClientRect();mx=Math.floor((e.clientX-r.left)/r.width*256);my=Math.floor((e.clientY-r.top)/r.height*192);});
 c.addEventListener('mousedown',e=>{btnq.push(1);e.preventDefault();}); c.addEventListener('contextmenu',e=>e.preventDefault());}
window.addEventListener('mouseup',()=>btnq.push(0));
window.addEventListener('keydown',e=>{if(HELD[e.key]){e.preventDefault();if(!held.includes(HELD[e.key]))held.push(HELD[e.key]);return;}if(QUEUED[e.key]){e.preventDefault();typed.push(QUEUED[e.key]);return;}if(e.key.length===1&&!e.ctrlKey&&!e.metaKey){const i=CHARSET.indexOf(e.key.toUpperCase());if(i>=0){e.preventDefault();typed.push(9+i);}}});
window.addEventListener('keyup',e=>{held=held.filter(k=>k!==HELD[e.key]);});
for(const g of document.querySelectorAll('.grid')){const c=g.getContext('2d');c.strokeStyle='rgba(255,255,0,0.45)';c.lineWidth=1;
 for(let x=0;x<=32;x++){c.beginPath();c.moveTo(x*24+0.5,0);c.lineTo(x*24+0.5,576);c.stroke();}
 for(let y=0;y<=24;y++){c.beginPath();c.moveTo(0,y*24+0.5);c.lineTo(768,y*24+0.5);c.stroke();}}
document.getElementById('gridt').onchange=e=>{for(const g of document.querySelectorAll('.grid'))g.style.display=e.target.checked?'block':'none';};
fetch('/ckpts').then(r=>r.json()).then(j=>{const s=document.getElementById('ckpt');for(const n of j.ckpts){const o=document.createElement('option');o.value=n;const m=n.match(/ckpt_0*(\d+)\.pt/);o.textContent=m?('iter '+Number(m[1]).toLocaleString()):n;if(n===j.current)o.selected=true;s.appendChild(o);}if(!j.ckpts.length){const o=document.createElement('option');o.textContent='none';s.appendChild(o);}});
document.getElementById('ckpt').onchange=()=>cmd='load';
document.getElementById('sync').onclick=()=>cmd='sync'; document.getElementById('reset').onclick=()=>cmd='reset';
let idsOn=false,needFull=true;
document.getElementById('idst').onchange=e=>{idsOn=e.target.checked;needFull=true;document.getElementById('tileinfo').textContent=idsOn?'hover a tile for its full id':'';for(const c of document.querySelectorAll('.ids')){c.style.display=idsOn?'block':'none';c.getContext('2d').clearRect(0,0,768,576);}for(const c of Object.values(cv))c.style.opacity=idsOn?0.35:1;};
let SECS=[];fetch('/sprites/index').then(r=>r.json()).then(j=>SECS=j.sections);
const SECCOL={ui:'#7a3d00',title:'#14286e',title_in:'#3c3c46',body:'#1d5e2a',sel:'#0b4f6c',tool:'#5b2a6e',desk:'#0d5c5c',hud:'#7a1020',canvas:'#555'};
function tileInfo(t){for(const s of SECS){if(t>=s.start&&t<s.end){const n=[].concat(s.name),k=n[n.length-1],o=t-s.start;
 if(n[0]==='text'){const a=CHARSET[Math.floor(o/CHARSET.length)],b=CHARSET[o%CHARSET.length];return {k,short:(a+b).replace(/ /g,'\u00b7'),long:`text/${k} "${a}${b}"`};}
 if(k==='canvas')return {k,short:o.toString(16),long:`paint canvas cell ${o}`};
 return {k:'ui',short:String(t),long:'ui tile'};}}return {k:'ui',short:String(t),long:'tile'};}
const lastG={};
function drawIds(id,g,off,full){const c=document.getElementById(id).getContext('2d');const prev=lastG[id];c.font='bold 11px monospace';c.textAlign='center';c.textBaseline='middle';
 for(let i=0;i<768;i++){const t=g[off+i];if(!full&&prev&&prev[i]===t)continue;const x=(i%32)*24,y=Math.floor(i/32)*24;c.clearRect(x,y,24,24);
  if(t===0)continue;const f=tileInfo(t);c.fillStyle=SECCOL[f.k]||'#333';c.fillRect(x+1,y+1,22,22);c.fillStyle='#fff';
  c.font=f.short.length>3?'bold 8px monospace':'bold 11px monospace';c.fillText(f.short,x+12,y+12.5);}
 lastG[id]=g.slice(off,off+768);}
let hoverIdx=-1;
for(const [k,c] of Object.entries(cv))c.addEventListener('mousemove',e=>{const r=c.getBoundingClientRect();hoverIdx=Math.floor((e.clientY-r.top)/r.height*24)*32+Math.floor((e.clientX-r.left)/r.width*32);hoverSide=k;});
let hoverSide='real';
function showHover(){if(!idsOn||hoverIdx<0)return;const g=lastG['ids_'+hoverSide];if(!g)return;const t=g[hoverIdx],f=tileInfo(t);
 document.getElementById('tileinfo').textContent=`${hoverSide} tile (${hoverIdx%32},${Math.floor(hoverIdx/32)})  id ${t}  ${f.long}`;}
function draw(c,bytes,off){const ctx=c.getContext('2d');const img=ctx.createImageData(256,192);
 for(let i=0;i<256*192;i++){const p=PAL[bytes[off+i]];img.data[4*i]=p[0];img.data[4*i+1]=p[1];img.data[4*i+2]=p[2];img.data[4*i+3]=255;}
 ctx.putImageData(img,0,0);}
const ws=new WebSocket((location.protocol==='https:'?'wss://':'ws://')+location.host+location.pathname.replace(/\\/$/,'')+'/ws');
ws.binaryType='arraybuffer'; let t0=performance.now(),avg=66,sent=performance.now();
function send(){if(btnq.length)btn=btnq.shift();const k=typed.length?typed.shift():(held.length?held[held.length-1]:0);const tp=parseFloat(document.getElementById('temp').value);document.getElementById('tv').textContent=tp;sent=performance.now();ws.send(JSON.stringify({x:mx,y:my,btn:btn,key:k,cmd:cmd,temp:tp,ckpt:document.getElementById('ckpt').value,ids:idsOn}));cmd=null;}
ws.onopen=send;
ws.onmessage=e=>{const b=new Uint8Array(e.data);const nf=(b.length%49152===0)?b.length/49152:Math.floor(b.length/(49152+1536));draw(cv.real,b,0);if(nf>1)draw(cv.neural,b,49152);
 if(idsOn&&b.length>nf*49152){const g=new Uint16Array(e.data.slice(nf*49152));drawIds('ids_real',g,0,needFull);if(nf>1)drawIds('ids_neural',g,768,needFull);needFull=false;showHover();}
 const now=performance.now(),dt=now-t0;t0=now;avg=avg*0.9+dt*0.1;document.getElementById('info').textContent=` ${(1000/avg).toFixed(1)} fps`;
 setTimeout(send,Math.max(0,66-(performance.now()-sent)));};
</script></body></html>"""


class Neural:
    def __init__(self, ckpt, device):
        import torch
        from .model import WorldModel, Config
        ck = torch.load(ckpt, map_location=device)
        self.model = WorldModel(Config(**ck['cfg'])).to(device).eval()
        self.model.load_state_dict(ck['model'])
        self.device = device

    def sync(self, obs):
        self.obs = {k: np.array(obs[k]) for k in OBS_KEYS}
        self.mem = self.model.init_memory(1)
        self.prev_btn = 0

    def step(self, action, temperature=0.0):
        import torch
        from .model import decode
        torch.set_grad_enabled(False)
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


def list_ckpts():
    d = STATE.get('ckpt_dir')
    if not d or not os.path.isdir(d):
        return []
    names = sorted(os.path.basename(f) for f in glob.glob(os.path.join(d, 'ckpt_*.pt')))
    return names or [n for n in ['ckpt.pt'] if os.path.exists(os.path.join(d, n))]


def load_ckpt(name):
    cache = STATE.setdefault('loaded', {})
    if name not in cache:
        cache.clear()
        cache[name] = Neural(os.path.join(STATE['ckpt_dir'], name), STATE.get('device', 'cpu'))
    return cache[name]


@app.get('/ckpts')
def ckpts():
    return JSONResponse(dict(ckpts=list_ckpts(), current=STATE.get('current', '')))


def init_neural(ckpt_dir, device, name=None):
    STATE['ckpt_dir'], STATE['device'] = ckpt_dir, device
    names = list_ckpts()
    if not names:
        return
    name = name if name in names else names[-1]
    STATE['neural'] = load_ckpt(name)
    STATE['current'] = name


SPRITES = """<!doctype html><html><head><title>Dreaming Computer: tile & sprite atlas</title><style>
body{background:#1b1b1f;color:#ddd;font-family:monospace;margin:16px}img{image-rendering:pixelated;border:1px solid #444;display:block;margin:6px 0 18px}a{color:#8cf}</style></head><body>
<h2>Every tile and sprite the model can output</h2><p>Hover a tile to see its token ID. Each sheet is 64 tiles per row, drawn 3x.</p><div id="secs"></div>
<p><a href="/">&larr; play the computer</a> &middot; <a href="/data">training data viewer</a></p>
<script>
fetch('/sprites/index').then(r=>r.json()).then(j=>{const d=document.getElementById('secs');
 for(const s of j.sections){const h=document.createElement('h3');h.textContent=`${[].concat(s.name).join(' / ')}: ids ${s.start}-${s.end-1} (${s.end-s.start} tiles${s.shown<s.end-s.start?', showing '+s.shown:''})`;d.appendChild(h);
  const im=document.createElement('img');im.src=`/sprites/sheet?start=${s.start}&n=${s.shown}`;const info=document.createElement('div');
  im.onmousemove=e=>{const r=im.getBoundingClientRect();const c=Math.floor((e.clientX-r.left)/24),rr=Math.floor((e.clientY-r.top)/24);const id=s.start+rr*64+c;info.textContent=id<s.start+s.shown?'tile id '+id:'';};
  d.appendChild(im);d.appendChild(info);}
 const h=document.createElement('h3');h.textContent='Doom: wall textures (lit/dark side, 16x16 shown 4x); sprite kinds 1-3 at sizes 6/12/24/40 drawn by render_tokens; bottom row: real frames';d.appendChild(h);
 const im=document.createElement('img');im.src='/sprites/doom';d.appendChild(im);});
</script></body></html>"""


def _sections():
    from .tiles import ATLAS
    rng = sorted(ATLAS.ranges.items(), key=lambda kv: kv[1][0])
    out = []
    if rng and rng[0][1][0] > 0:
        out.append(('ui', 0, rng[0][1][0]))
    for name, (a, b) in rng:
        out.append((name, a, b))
    return out


@app.get('/sprites')
def sprites_page():
    return HTMLResponse(SPRITES)


@app.get('/sprites/index')
def sprites_index():
    return JSONResponse(dict(sections=[dict(name=n, start=a, end=b, shown=min(b - a, 64 * 32)) for n, a, b in _sections()]))


def _png(rgb, scale):
    from PIL import Image
    buf = io.BytesIO()
    im = Image.fromarray(rgb)
    im.resize((im.width * scale, im.height * scale), Image.NEAREST).save(buf, 'PNG')
    return Response(buf.getvalue(), media_type='image/png')


@app.get('/sprites/sheet')
def sprites_sheet(start: int, n: int):
    from .tiles import ATLAS
    tiles = np.stack(ATLAS.tiles[start:start + n])
    rows = (len(tiles) + 63) // 64
    sheet = np.zeros((rows * 8, 64 * 8), np.uint8)
    for i, t in enumerate(tiles):
        sheet[(i // 64) * 8:(i // 64) * 8 + 8, (i % 64) * 8:(i % 64) * 8 + 8] = t
    return _png(np.asarray(PAL, np.uint8)[sheet], 3)


@functools.lru_cache(maxsize=1)
def _doom_sheet():
    from . import doom as dm

    def pad(x, b=4):
        return np.pad(x, ((b, b), (b, b)))

    def hcat(xs):
        h = max(x.shape[0] for x in xs)
        return np.concatenate([np.pad(x, ((0, h - x.shape[0]), (0, 0))) for x in xs], 1)

    def vcat(xs):
        w = max(x.shape[1] for x in xs)
        return np.concatenate([np.pad(x, ((0, 0), (0, w - x.shape[1]))) for x in xs], 0)

    tex = hcat([pad(np.kron(dm.TEX[t, sd], np.ones((4, 4), np.uint8))) for t in range(1, dm.TEX.shape[0]) for sd in (0, 1)])
    cols = np.zeros((160, 4), np.uint8)
    cols[:, 0], cols[:, 2], cols[:, 3] = 1, 4, np.arange(160) % 16
    rows = []
    for kind in (1, 2, 3):
        fr = []
        for size in (6, 12, 24, 40):
            spr = np.zeros((4, 3), np.uint8)
            spr[0] = (kind, 80 + 32, size)
            fr.append(pad(dm.render_tokens(cols, spr, 0)))
        rows.append(hcat(fr))
    o = MicroOS(7)
    o.reset()
    o.open_app('doom')
    real, rr = [], random.Random(3)
    for k in range(600):
        ob = o.step((128, 96, 0, rr.choice([1, 1, 3, 5, 5, 0])))
        if k % 150 == 149:
            real.append(pad(dm.render_tokens(ob['doom_cols'], ob['doom_spr'], int(ob['doom_meta'][0]) > 0)))
    return np.asarray(PAL, np.uint8)[vcat([tex] + rows + [hcat(real)])]


@app.get('/sprites/doom')
def sprites_doom():
    return _png(_doom_sheet(), 2)


@app.get('/')
def index():
    page = PAGE.replace('__PAL__', json.dumps(PAL.tolist())).replace('__CHARSET__', json.dumps(CHARSET))
    return HTMLResponse(page.replace('__NEURAL__', 'loaded' if STATE.get('neural') else 'no checkpoint'))


@app.websocket('/ws')
async def ws(sock: WebSocket):
    await sock.accept()
    real = MicroOS(random.randrange(10 ** 9))
    obs = real.reset()
    import copy
    neural = copy.copy(STATE['neural']) if STATE.get('neural') else None
    if neural:
        neural.sync(obs)
    while True:
        msg = json.loads(await sock.receive_text())
        if msg.get('cmd') == 'reset':
            obs = real.reset(random.randrange(10 ** 9))
            if neural: neural.sync(obs)
        elif msg.get('cmd') == 'sync' and neural:
            neural.sync(obs)
        elif msg.get('cmd') == 'load' and msg.get('ckpt') in list_ckpts():
            try:
                neural = copy.copy(load_ckpt(msg['ckpt']))
                neural.sync(obs)
            except Exception:
                import traceback
                traceback.print_exc()
        a = (int(msg['x']), int(msg['y']), int(msg['btn']), int(msg['key']))
        try:
            obs = real.step(a)
        except Exception:
            import traceback
            traceback.print_exc()
            obs = real.reset(random.randrange(10 ** 9))
        frames = [obs_to_pixels(obs)]
        if neural:
            frames.append(obs_to_pixels(neural.step(a, float(msg.get('temp', 0.0)))))
        grids = []
        if msg.get('ids'):
            grids = [np.asarray(obs['grid'], np.uint16)] + ([np.asarray(neural.obs['grid'], np.uint16)] if neural else [])
        await sock.send_bytes(b''.join(f.astype(np.uint8).tobytes() for f in frames) + b''.join(g.astype('<u2').tobytes() for g in grids))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='')
    ap.add_argument('--port', type=int, default=8000)
    a = ap.parse_args()
    if a.ckpt:
        import torch
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        init_neural(os.path.dirname(os.path.abspath(a.ckpt)), dev, os.path.basename(a.ckpt))
    uvicorn.run(app, host='0.0.0.0', port=a.port)


if __name__ == '__main__':
    main()
