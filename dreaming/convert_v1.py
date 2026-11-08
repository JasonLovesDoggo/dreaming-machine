"""Convert a checkpoint from the v1 (packed) tile/state layout to the stable v2 layout.

python -m dreaming.convert_v1 runs/main/ckpt_005000.pt runs/main/ckpt_005000.v2.pt
"""
import os
import sys
import numpy as np
import torch
from .tiles import VOCAB
from .os_sim import APP_SLOTS, STATE_VOCAB

V1_APPS = 5
TILE_MAP = np.load(os.path.join(os.path.dirname(__file__), 'atlas_v1_to_v2.npy'))


def convert(sd):
    sd = dict(sd)
    idx = torch.as_tensor(TILE_MAP, dtype=torch.long)
    if sd['tile.weight'].shape[0] == len(TILE_MAP):
        old = sd['tile.weight']
        new = torch.randn(VOCAB, old.shape[1], dtype=old.dtype) * 0.02
        new[idx] = old
        sd['tile.weight'] = new
        w, b = sd['h_tile.weight'], sd['h_tile.bias']
        nw, nb = torch.zeros(VOCAB, w.shape[1], dtype=w.dtype), torch.full((VOCAB,), -10.0, dtype=b.dtype)
        nw[idx], nb[idx] = w, b
        sd['h_tile.weight'], sd['h_tile.bias'] = nw, nb
    q = sd['state_q']
    if q.shape[0] != len(STATE_VOCAB):
        new = torch.randn(len(STATE_VOCAB), q.shape[1], dtype=q.dtype) * 0.02
        na, rest = 3 * V1_APPS, q.shape[0] - 3 * V1_APPS
        new[:na] = q[:na]
        new[3 * len(APP_SLOTS):3 * len(APP_SLOTS) + rest] = q[na:]
        sd['state_q'] = new
    return sd


def main(src, dst):
    ck = torch.load(src, map_location='cpu', weights_only=False)
    ck['model'] = convert(ck['model'])
    ck.pop('opt', None)
    torch.save(ck, dst)
    print('wrote', dst)


if __name__ == '__main__':
    main(*sys.argv[1:3])
