"""LLM-driven episodes: an OpenAI model looks at the screen and drives the simulator in unusual ways.
  OPENAI_API_KEY=... python -m dreaming.llm_driver --out data_llm --episodes 4 --ep-len 2000 --workers 4
Episodes are recorded in the same shard format as dreaming.dataset (files named shard_llm_*)."""
import argparse
import base64
import io
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from .os_sim import MicroOS, obs_to_pixels
from .driver import Driver, KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT, KEY_FIRE, KEY_ENTER, KEY_ESC, KEY_DEL, KEY_CHAR0
from .tiles import PAL, CHARSET, SCREEN_W, SCREEN_H
from .dataset import OBS_KEYS

MODEL = 'gpt-6.1-terra'
KEYS = dict(UP=KEY_UP, DOWN=KEY_DOWN, LEFT=KEY_LEFT, RIGHT=KEY_RIGHT, FIRE=KEY_FIRE, ENTER=KEY_ENTER, ESC=KEY_ESC,
            BACKSPACE=KEY_DEL, DEL=KEY_DEL)
PROMPT = f"""You are a mischievous QA tester driving a tiny retro desktop computer (screen {SCREEN_W}x{SCREEN_H} px, 8x8 px tiles).
Desktop icons on the left (FILES, PAINT, NOTES, TRASH, CARDS, PICS, DOOM); taskbar at the bottom; windows have a title bar
(drag it to move) and an X button at the top-right corner. Apps: Explorer (NEW button creates files by typed name; .PNG/.JPG
names make pictures), Paint, Notepad, Solitaire, Doom (arrows move/turn, FIRE shoots).
Your job is to produce UNUSUAL interaction data that a scripted happy-path bot would never produce: e.g. dragging windows
in loops, off the edges, or onto the taskbar; clicking every corner of a window; typing in the wrong app; opening many apps
at once; holding the mouse while typing; double-clicking dead space; strange filenames; abandoning tasks halfway; rapid
alt-tabbing via the taskbar; scribbling over Paint then closing it; walking into Doom walls; etc. Do not repeat what you did
in the recent history. Keep each turn short (3-10 actions).
Reply ONLY with JSON: {{"idea": "<what you are trying, <=12 words>", "actions": [ ... ]}} using these ops:
 {{"op":"move","x":X,"y":Y}}   {{"op":"click","x":X,"y":Y}}   {{"op":"dblclick","x":X,"y":Y}}
 {{"op":"drag","x0":X,"y0":Y,"x1":X,"y1":Y}}   {{"op":"down"}}   {{"op":"up"}}
 {{"op":"key","key":"UP|DOWN|LEFT|RIGHT|FIRE|ENTER|ESC|BACKSPACE","n":1-20}}
 {{"op":"type","text":"UPPERCASE TEXT using only: {CHARSET.strip()}"}}   {{"op":"wait","n":1-30}}
Coordinates are screen pixels (0<=x<{SCREEN_W}, 0<=y<{SCREEN_H})."""


def _png(obs):
    from PIL import Image
    im = Image.fromarray(np.asarray(PAL, np.uint8)[obs_to_pixels(obs)]).resize((512, 384), Image.NEAREST)
    buf = io.BytesIO()
    im.save(buf, 'PNG')
    return base64.b64encode(buf.getvalue()).decode()


def ask(obs, os_, history, model, key):
    status = f"focused: {os_.focus or 'desktop'}; open windows (col,row in tiles): " + \
             ', '.join(f'{a}@{os_.pos[a]}' for a in os_.z) + f"; recent ideas: {history[-6:]}"
    body = dict(model=model, input=[dict(role='system', content=PROMPT), dict(role='user', content=[
        dict(type='input_text', text=status), dict(type='input_image', image_url='data:image/png;base64,' + _png(obs))])])
    for attempt in range(4):
        try:
            req = urllib.request.Request('https://api.openai.com/v1/responses', data=json.dumps(body).encode(),
                                         headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
            r = json.load(urllib.request.urlopen(req, timeout=120))
            text = next(c['text'] for o in r['output'] if o['type'] == 'message' for c in o['content'] if 'text' in c)
            text = text[text.index('{'):text.rindex('}') + 1]
            return json.loads(text), r.get('usage', {})
        except Exception as e:  # noqa: BLE001
            print('llm error', type(e).__name__, str(e)[:200], flush=True)
            time.sleep(2 * (attempt + 1))
    return {'idea': 'fallback wiggle', 'actions': [{'op': 'wait', 'n': 10}]}, {}


def _clip(v, hi):
    return min(max(int(v), 0), hi - 1)


def expand(d, a):
    """Turn one LLM op into frame-level actions using the scripted driver's human-like motion."""
    op = a.get('op')
    X = lambda k: _clip(a.get(k, d.x), SCREEN_W)  # noqa: E731
    Y = lambda k: _clip(a.get(k, d.y), SCREEN_H)  # noqa: E731
    if op == 'move':
        yield from d.move_to(X('x'), Y('y'))
    elif op in ('click', 'dblclick'):
        yield from d.click(X('x'), Y('y'), double=op == 'dblclick')
    elif op == 'drag':
        yield from d.drag(X('x0'), Y('y0'), X('x1'), Y('y1'))
    elif op in ('down', 'up'):
        d.btn = int(op == 'down')
        yield d.act()
    elif op == 'key' and str(a.get('key', '')).upper() in KEYS:
        yield from d.press_key(KEYS[str(a['key']).upper()], max(1, min(int(a.get('n', 1)), 20)))
    elif op == 'type':
        for ch in str(a.get('text', '')).upper()[:40]:
            if ch in CHARSET:
                yield from d.press_key(KEY_CHAR0 + CHARSET.index(ch))
                yield from d.idle(d.rng.randint(0, 1))
    elif op == 'wait':
        yield from d.idle(max(1, min(int(a.get('n', 1)), 30)))


def run_episode(seed, length, model, key, log=None):
    o = MicroOS(seed)
    obs = o.reset()
    d = Driver(o, seed)
    rec = {k: [obs[k]] for k in OBS_KEYS}
    acts, history, usage = [], [], dict(input_tokens=0, output_tokens=0, calls=0)
    while len(acts) < length:
        plan, u = ask(obs, o, history, model, key)
        usage['calls'] += 1
        for k in ('input_tokens', 'output_tokens'):
            usage[k] += u.get(k, 0)
        history.append(plan.get('idea', ''))
        if log:
            log.write(json.dumps(dict(seed=seed, t=len(acts), **plan)) + '\n')
        n0 = len(acts)
        for a in plan.get('actions', [])[:12]:
            for act in expand(d, a if isinstance(a, dict) else {}):
                obs = o.step(act)
                acts.append(act)
                for k in OBS_KEYS:
                    rec[k].append(obs[k])
                if len(acts) >= length:
                    break
            if len(acts) >= length:
                break
        if len(acts) == n0:  # nothing happened; keep time moving
            for act in d.idle(5):
                obs = o.step(act)
                acts.append(act)
                for k in OBS_KEYS:
                    rec[k].append(obs[k])
    acts.append((0, 0, 0, 0))
    out = {k: np.stack(v) for k, v in rec.items()}
    out['action'] = np.array(acts, np.uint8)
    return out, usage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='data_llm')
    ap.add_argument('--episodes', type=int, default=4)
    ap.add_argument('--ep-len', type=int, default=2000)
    ap.add_argument('--eps-per-shard', type=int, default=4)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--model', default=MODEL)
    a = ap.parse_args()
    key = os.environ['OPENAI_API_KEY']
    os.makedirs(a.out, exist_ok=True)
    log = open(os.path.join(a.out, f'ideas_{a.seed:03d}.jsonl'), 'a')
    seeds = [a.seed * 10_000_000 + 5_000_000 + i for i in range(a.episodes)]
    t, tot = time.time(), dict(input_tokens=0, output_tokens=0, calls=0)
    with ThreadPoolExecutor(a.workers) as ex:
        eps = []
        for ep, u in ex.map(lambda s: run_episode(s, a.ep_len, a.model, key, log), seeds):
            eps.append(ep)
            tot = {k: tot[k] + u[k] for k in tot}
            print(f'episode done ({time.time() - t:.0f}s) usage so far {tot}', flush=True)
    for i in range(0, len(eps), a.eps_per_shard):
        part = eps[i:i + a.eps_per_shard]
        data = {k: np.concatenate([e[k] for e in part]) for k in part[0]}
        data['ep_len'] = np.array([len(e['action']) for e in part], np.int64)
        path = os.path.join(a.out, f'shard_llm_{a.seed:03d}_{i // a.eps_per_shard:05d}.npz')
        np.savez_compressed(path, **data)
        print('wrote', path, flush=True)
    print('total usage', tot)


if __name__ == '__main__':
    main()
