"""Training-data viewer: browse recorded episodes. Frames go out in raw palette-index chunks and are drawn client-side."""
import functools
import glob
import json
import os
import time
import numpy as np
from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import JSONResponse
from ..dataset import OBS_KEYS
from ..os_sim import APPS, obs_to_pixels
from .settings import SETTINGS

router = APIRouter(prefix='/data')
MAX_CLIP = 128
INDEX_TTL = 60


def _path(name):
    path = os.path.join(SETTINGS.data_dir, os.path.basename(name) + '.npz')
    if not os.path.exists(path):
        raise HTTPException(404, 'no such shard')
    return path


@functools.lru_cache(maxsize=4)
def _shard(name):
    with np.load(_path(name)) as z:
        return {k: z[k] for k in z.files}


_index = {'t': 0.0, 'names': []}


@router.get('/index')
def index():
    """Shard names only (cheap glob, cached); episode lengths come from /data/shard on demand."""
    if time.time() - _index['t'] > INDEX_TTL:
        _index['names'] = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(SETTINGS.data_dir, '*.npz')))
        _index['t'] = time.time()
    return JSONResponse(dict(dir=SETTINGS.data_dir, shards=_index['names']))


@functools.lru_cache(maxsize=4096)
def _ep_len(name):
    with np.load(_path(name)) as z:
        return z['ep_len'].tolist()


@router.get('/shard')
def shard(name: str):
    return JSONResponse(dict(name=name, ep_len=_ep_len(name)))


def _focus(state):
    f = int(state[3 * len(APPS)])
    return APPS[f - 1] if f else 'desktop'


@router.get('/clip')
def clip(shard: str, start: int, n: int = 32):
    """n consecutive frames as raw 256x192 palette indices (uint8), plus per-frame focus/action in x-meta."""
    d = _shard(shard)
    start = max(0, min(start, len(d['grid']) - 1))
    stop = min(len(d['grid']), start + max(1, min(n, MAX_CLIP)))
    frames, meta = [], []
    for i in range(start, stop):
        frames.append(obs_to_pixels({k: d[k][i] for k in OBS_KEYS}).astype(np.uint8).tobytes())
        meta.append([_focus(d['state'][i]), d['action'][i].tolist()])
    return Response(b''.join(frames), media_type='application/octet-stream',
                    headers={'x-meta': json.dumps(meta), 'cache-control': 'public, max-age=86400'})
