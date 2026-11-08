"""Dataset quality report: app coverage, input mix, and how often inputs actually change the screen."""
import argparse
import glob
import json
import numpy as np

from .os_sim import APPS

FOCUS = ['desktop'] + list(APPS)
FOCUS_COL = 3 * len(APPS)


def shard_stats(path, k=6):
    z = np.load(path)
    grid, act, state, lens = z['grid'], z['action'], z['state'], z['ep_len']
    s = dict(steps=0, focus=np.zeros(len(FOCUS), np.int64), clicks=0, clicks_noop=0, dbl=0, drags=0, drags_noop=0,
             keys=0, keys_noop=0, changed=0, held=0)
    off = 0
    for L in lens:
        g = grid[off:off + L].reshape(L, -1)
        a = act[off:off + L].astype(np.int64)
        st = state[off:off + L]
        off += L
        dz = np.concatenate([z[k][off - L:off].reshape(L, -1) for k in ('doom_cols', 'doom_spr', 'doom_meta')], 1)
        ch = np.r_[False, (g[1:] != g[:-1]).any(1) | (dz[1:] != dz[:-1]).any(1)]
        # did anything change within the next k frames (inputs take effect after a frame or two)
        fut = np.convolve(ch[::-1].astype(np.int64), np.ones(k + 1, np.int64))[:L][::-1] > 0
        fut = np.r_[fut[1:], False]
        btn = a[:, 2] > 0
        down = np.flatnonzero(btn[1:] & ~btn[:-1]) + 1
        up = np.flatnonzero(~btn[1:] & btn[:-1]) + 1
        for d0 in down:
            u = up[up > d0]
            if not len(u):
                continue
            u = u[0]
            moved = np.abs(a[u, :2] - a[d0, :2]).sum() > 6
            noop = not ch[d0:u + k + 1].any()
            if moved:
                s['drags'] += 1; s['drags_noop'] += noop
            else:
                s['clicks'] += 1; s['clicks_noop'] += noop
        if len(down) > 1:
            s['dbl'] += int((np.diff(down) <= 6).sum())
        kp = np.flatnonzero(a[:, 3] > 0)
        s['keys'] += len(kp)
        s['keys_noop'] += int((~fut[kp]).sum())
        s['changed'] += int(ch.sum())
        s['held'] += int(btn.sum())
        s['steps'] += int(L)
        s['focus'] += np.bincount(np.clip(st[:, FOCUS_COL], 0, len(FOCUS) - 1), minlength=len(FOCUS))
    return s


def report(paths):
    tot = None
    for p in paths:
        s = shard_stats(p)
        tot = s if tot is None else {k: tot[k] + v for k, v in s.items()}
    n = tot['steps']
    r = dict(shards=len(paths), steps=n,
             focus={f: round(float(v) / n, 3) for f, v in zip(FOCUS, tot['focus'])},
             frames_changed=round(tot['changed'] / n, 3), button_held=round(tot['held'] / n, 3),
             clicks=tot['clicks'], clicks_noop=round(tot['clicks_noop'] / max(1, tot['clicks']), 3),
             double_clicks=tot['dbl'],
             drags=tot['drags'], drags_noop=round(tot['drags_noop'] / max(1, tot['drags']), 3),
             keys=tot['keys'], keys_noop=round(tot['keys_noop'] / max(1, tot['keys']), 3))
    return r


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('shards', nargs='+')
    a = ap.parse_args()
    paths = sorted(p for g in a.shards for p in glob.glob(g))
    print(json.dumps(report(paths), indent=1))
