"""Generate training shards: python -m dreaming.dataset --out data --steps 2000000 --workers 8"""
import argparse
import json
import os
import time
from collections import Counter
from multiprocessing import Pool
import numpy as np
from .os_sim import MicroOS
from .driver import Driver
from .pictures import parse_name, parse_words

OBS_KEYS = ['grid', 'cursor', 'doom_cols', 'doom_spr', 'doom_meta', 'state']


def run_episode(seed, length, mode='mixed', stats=None):
    o = MicroOS(seed)
    obs = o.reset()
    d = Driver(o, seed, mode)
    rec = {k: [obs[k]] for k in OBS_KEYS}
    acts = []
    for _ in range(length):
        a = next(d.stream)
        obs = o.step(a)
        d.observe_events(obs['events'])
        if stats is not None:
            stats['focus'][o.focus or 'desktop'] += 1
            if 'ex_open_png' in obs['events']:
                stats['picture_opens'][o.pt_file] += 1
                stats['subjects'].update(set(parse_name(o.pt_file)) or {'abstract'})
                cat_words = next((kws for kws, c in parse_words(o.pt_file) if c == 'cat'), None)
                if cat_words is not None:
                    stats.setdefault('cat_terms', Counter()).update(set(cat_words))
        acts.append(a)
        for k in OBS_KEYS:
            rec[k].append(obs[k])
    acts.append((0, 0, 0, 0))  # pad so actions align with obs
    out = {k: np.stack(v) for k, v in rec.items()}
    out['action'] = np.array(acts, np.uint8)
    return out


def make_shard(args):
    path, shard_idx, n_eps, ep_len, base_seed, mode = args
    stats = {k: Counter() for k in ('picture_opens', 'subjects', 'focus', 'cat_terms')}
    seeds = [base_seed * 10_000_000 + shard_idx * 1000 + e for e in range(n_eps)]
    eps = [run_episode(seed, ep_len, mode, stats) for seed in seeds]
    data = {k: np.concatenate([e[k] for e in eps]) for k in eps[0]}
    data['ep_len'] = np.array([len(e['action']) for e in eps], np.int64)
    tmp = path + '.tmp.npz'
    np.savez_compressed(tmp, **data)
    os.replace(tmp, path)
    with open(path + '.json.tmp', 'w') as f:
        json.dump(dict(mode=mode, seeds=seeds, actions=n_eps * ep_len, **stats), f, sort_keys=True)
    os.replace(path + '.json.tmp', path + '.json')
    return path


def generate(out, steps, ep_len=2000, eps_per_shard=25, workers=8, seed=0, start_shard=0, mode='mixed'):
    os.makedirs(out, exist_ok=True)
    n_shards = max(1, steps // (ep_len * eps_per_shard))
    tag = '' if mode == 'mixed' else mode + '_'
    jobs = [(os.path.join(out, f'shard_{tag}{seed:03d}_{i:05d}.npz'), i, eps_per_shard, ep_len, seed, mode)
            for i in range(start_shard, start_shard + n_shards)]
    jobs = [j for j in jobs if not os.path.exists(j[0])]
    t = time.time()
    with Pool(workers) as p:
        for i, path in enumerate(p.imap_unordered(make_shard, jobs)):
            print(f'[{i + 1}/{len(jobs)}] {path} ({time.time() - t:.0f}s)', flush=True)


def load_shards(paths, max_steps=None):
    parts, total = [], 0
    for p in paths:
        with np.load(p) as z:
            parts.append({k: z[k] for k in z.files})
        total += len(parts[-1]['action'])
        if max_steps and total >= max_steps:
            break
    data = {k: np.concatenate([d[k] for d in parts]) for k in parts[0]}
    return data


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='data')
    ap.add_argument('--steps', type=int, default=1_000_000)
    ap.add_argument('--ep-len', type=int, default=2000)
    ap.add_argument('--eps-per-shard', type=int, default=25)
    ap.add_argument('--workers', type=int, default=os.cpu_count())
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--mode', default='mixed', choices=['mixed', 'doom', 'cards', 'windows', 'calculator', 'terminal', 'dreamer', 'cats'])
    a = ap.parse_args()
    generate(a.out, a.steps, a.ep_len, a.eps_per_shard, a.workers, a.seed, mode=a.mode)
