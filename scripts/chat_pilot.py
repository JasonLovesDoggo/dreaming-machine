"""Small additive Chat pilot: persona/fallback-heavy dreamer episodes, no hedge replies in new turns.

python scripts/chat_pilot.py --out ~/chat_pilot_v1 --episodes 50 --ep-len 2000 --seed 777
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import zlib
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dreaming.apps.dreamer import FALLBACKS, HEDGE, PERSONA, corpus, normalize, reply  # noqa: E402
from dreaming.dataset import OBS_KEYS  # noqa: E402
from dreaming.driver import Driver  # noqa: E402
from dreaming.os_sim import KEY_DOWN, KEY_UP, MicroOS  # noqa: E402

POLICY = 'chat_pilot_v1: 35% PERSONA aliases, 35% off-corpus prompts targeting least-used FALLBACK, 30% hedge-free corpus; resample any prompt whose reply matches HEDGE'
WORDS = ('WOBBLY', 'PICKLE', 'MOON', 'SPROCKET', 'NOODLE', 'GALAXY', 'TOASTER', 'PURPLE', 'GIGGLE', 'ZEBRA', 'WAFFLE',
         'PUDDLE', 'ROCKET', 'BISCUIT', 'TURNIP', 'GLITTER', 'KAZOO', 'BLIMP', 'MUFFIN', 'CACTUS', 'YODEL', 'PIXEL')
FRAMES = ('WHAT IS A {} {}', 'CAN A {} EAT A {}', 'WHY IS MY {} SO {}', 'TELL ME ABOUT {} {}', 'IS {} BETTER THAN {}',
          'HOW MANY {} FIT IN A {}', 'DO YOU LIKE {} {}', '{} {} PLEASE')


class PilotDriver(Driver):
    def __init__(self, os_, seed, mode='dreamer'):
        super().__init__(os_, seed, mode)
        self.used = Counter()

    def unknown(self):
        seed = self.os.seed
        target = min(range(len(FALLBACKS)), key=lambda i: (self.used[i], (i - seed) % len(FALLBACKS)))
        for _ in range(400):
            p = self.rng.choice(FRAMES).format(self.rng.choice(WORDS), self.rng.choice(WORDS))
            if reply(p, seed) == FALLBACKS[target]:
                return p
        return p

    def chat_prompt(self):
        safe = [p for p in corpus().prompts if not HEDGE.search(corpus().pairs[p])]
        for _ in range(50):
            pool = self.rng.random()
            if pool < 0.35:
                p = self.rng.choice(sorted(PERSONA))
                p = self.mess_prompt(p) if self.rng.random() < 0.3 else p
            elif pool < 0.7:
                p = self.unknown()
            else:
                p = self.mess_prompt(self.rng.choice(safe)[:150])
            r = reply(p, self.os.seed)
            if normalize(p) and not HEDGE.search(r):
                if r in FALLBACKS:
                    self.used[FALLBACKS.index(r)] += 1
                return p
        return 'HELLO'


def kind(prompt, answer):
    if answer in FALLBACKS:
        return 'fallback'
    return 'persona' if normalize(prompt) in PERSONA else 'corpus'


def episode(seed, length):
    o = MicroOS(seed)
    obs = o.reset()
    d = PilotDriver(o, seed)
    rec = {k: [obs[k]] for k in OBS_KEYS}
    acts, turns, stats, scrolled = [], [], Counter(), []
    for t in range(length):
        a = next(d.stream)
        scroll = o.chat.scroll
        obs = o.step(a)
        d.observe_events(obs['events'])
        focus = o.focus == 'dreamer'
        stats['chat_focus_ticks'] += focus
        if focus and a[3] in (KEY_UP, KEY_DOWN):
            stats['scroll_keys'] += 1
            stats['scroll_moves'] += o.chat.scroll != scroll
            if o.chat.scroll > scroll:
                scrolled.append(t + 1)
        if 'dreamer_send' in obs['events']:
            prompt, answer = o.chat.messages[-2][1], o.chat.pending
            turns.append(dict(t=t + 1, prompt=prompt, reply=answer, kind=kind(prompt, answer),
                              fallback=FALLBACKS.index(answer) if answer in FALLBACKS else None, done=None))
        if 'dreamer_reply' in obs['events'] and turns and turns[-1]['done'] is None:
            turns[-1]['done'] = t + 1
            stats['max_messages'] = max(stats['max_messages'], len(o.chat.messages))
        acts.append(a)
        for k in OBS_KEYS:
            rec[k].append(obs[k])
    acts.append((0, 0, 0, 0))
    out = {k: np.stack(v) for k, v in rec.items()}
    out['action'] = np.array(acts, np.uint8)
    return out, turns, stats, scrolled


def digest(ep):
    h = hashlib.sha256()
    for k in sorted(ep):
        h.update(k.encode() + ep[k].tobytes())
    return h.hexdigest()


def make_shard(job):
    path, seeds, length, meta = job
    if os.path.exists(path):
        raise FileExistsError(path)
    eps = [episode(s, length) for s in seeds]
    data = {k: np.concatenate([e[0][k] for e in eps]) for k in eps[0][0]}
    data['ep_len'] = np.array([len(e[0]['action']) for e in eps], np.int64)
    np.savez_compressed(path + '.tmp.npz', **data)
    os.replace(path + '.tmp.npz', path)
    side = dict(meta, mode='dreamer', seeds=seeds, actions=len(seeds) * length,
                replay_sha256=[digest(e[0]) for e in eps], turns=[e[1] for e in eps],
                stats=[dict(e[2]) for e in eps], scrolled=[e[3] for e in eps])
    Path(path + '.json').write_text(json.dumps(side, sort_keys=True))
    return path


def positive(v):
    n = int(v)
    if n <= 0:
        raise argparse.ArgumentTypeError(f'must be positive: {v}')
    return n


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--episodes', type=positive, default=50)
    ap.add_argument('--ep-len', type=positive, default=2000)
    ap.add_argument('--eps-per-shard', type=positive, default=10)
    ap.add_argument('--seed', type=int, default=777)
    ap.add_argument('--workers', type=positive, default=os.cpu_count())
    a = ap.parse_args(argv)
    if os.path.exists(a.out) and (not os.path.isdir(a.out) or os.listdir(a.out)):
        ap.error(f'--out {a.out} exists and is not an empty directory; refusing to overwrite pilot data')
    os.makedirs(a.out, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    sha = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(['git', 'status', '--porcelain', 'dreaming', 'scripts'], cwd=root,
                                capture_output=True, text=True).stdout.strip())
    meta = dict(generator='scripts/chat_pilot.py', generator_sha=sha, dirty=dirty, policy=POLICY,
                fallbacks_crc=zlib.crc32('\n'.join(FALLBACKS).encode()), persona_prompts=len(PERSONA))
    jobs = []
    for i in range(0, a.episodes, a.eps_per_shard):
        seeds = [a.seed * 10_000_000 + i * 1000 + e for e in range(min(a.eps_per_shard, a.episodes - i))]
        jobs.append((os.path.join(a.out, f'shard_dreamer_pilot_{a.seed:03d}_{i // a.eps_per_shard:05d}.npz'),
                     seeds, a.ep_len, meta))
    with Pool(a.workers) as p:
        for path in p.imap_unordered(make_shard, jobs):
            print(path, flush=True)


if __name__ == '__main__':
    main()
