"""Training-data viewer: browse recorded episodes. Frames go out in raw palette-index chunks and are drawn client-side."""
import functools
import glob
import json
import os
import re
import time
import numpy as np
from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import JSONResponse
from ..dataset import OBS_KEYS
from ..os_sim import APP_SLOTS, FOCUS_COL, obs_to_pixels
from .settings import SETTINGS
from .storage import volume_read

router = APIRouter(prefix='/data')
MAX_CLIP = 128
INDEX_TTL = 60


def _root(dataset):
    root = os.path.normpath(SETTINGS.data_dir)
    roots = {'main': root, 'cats_dense_v1': root + '_cats_dense_v1', 'llm': root + '_llm'}
    if dataset not in roots:
        raise HTTPException(400, 'unknown dataset')
    return roots[dataset]


def _path(dataset, name):
    root = _root(dataset)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name) or '..' in name:
        raise HTTPException(400, 'invalid shard name')
    path = os.path.join(root, name + '.npz')
    if not os.path.exists(path):
        raise HTTPException(404, 'no such shard')
    return path


@functools.lru_cache(maxsize=4)
@volume_read
def _shard(dataset, name):
    with np.load(_path(dataset, name)) as z:
        return {k: z[k] for k in z.files}


_index: dict = {}


def _listing(dataset):
    root = _root(dataset)
    key = (dataset, root)
    if key not in _index or time.time() - _index[key]['t'] > INDEX_TTL:
        _index[key] = dict(t=time.time(), names=sorted(
            os.path.basename(p)[:-4] for p in glob.glob(os.path.join(root, '*.npz'))))
    return _index[key]['names']


@router.get('/datasets')
@volume_read
def datasets():
    return JSONResponse([dict(id=d, dir=os.path.basename(_root(d)), path=_root(d),
                              available=os.path.isdir(_root(d)), shards=len(_listing(d)))
                         for d in ('main', 'cats_dense_v1', 'llm')])


@router.get('/index')
@volume_read
def index(dataset: str = 'main'):
    """Shard names only (cheap glob, cached); episode lengths come from /data/shard on demand."""
    root = _root(dataset)
    return JSONResponse(dict(dir=root, shards=_listing(dataset), dataset=dataset, available=os.path.isdir(root)))


@functools.lru_cache(maxsize=4096)
@volume_read
def _ep_len(dataset, name):
    with np.load(_path(dataset, name)) as z:
        return z['ep_len'].tolist()


@router.get('/shard')
def shard(name: str, dataset: str = 'main'):
    lengths = _ep_len(dataset, name)
    return JSONResponse(dict(name=name, ep_len=lengths, dataset=dataset,
                             frames=sum(lengths), episodes=len(lengths)))


def _focus(state):
    f = int(state[FOCUS_COL])
    return APP_SLOTS[f - 1] if f else 'desktop'


@router.get('/clip')
def clip(shard: str, start: int, n: int = 32, dataset: str = 'main'):
    """n consecutive frames as raw 256x192 palette indices (uint8), plus per-frame focus/action in x-meta."""
    d = _shard(dataset, shard)
    start = max(0, min(start, len(d['grid']) - 1))
    stop = min(len(d['grid']), start + max(1, min(n, MAX_CLIP)))
    frames, meta = [], []
    for i in range(start, stop):
        frames.append(obs_to_pixels({k: d[k][i] for k in OBS_KEYS}).astype(np.uint8).tobytes())
        meta.append([_focus(d['state'][i]), d['action'][i].tolist()])
    return Response(b''.join(frames), media_type='application/octet-stream',
                    headers={'x-meta': json.dumps(meta), 'cache-control': 'public, max-age=86400'})
