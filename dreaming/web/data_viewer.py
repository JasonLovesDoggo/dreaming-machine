"""Training-data viewer: browse recorded episodes frame by frame."""
import functools
import glob
import json
import os
import numpy as np
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from ..dataset import OBS_KEYS
from ..os_sim import APPS, obs_to_pixels
from .images import png
from .settings import SETTINGS

router = APIRouter(prefix='/data')


@functools.lru_cache(maxsize=3)
def _shard(name):
    with np.load(os.path.join(SETTINGS.data_dir, os.path.basename(name) + '.npz')) as z:
        return {k: z[k] for k in z.files}


@router.get('/index')
def index():
    shards = []
    for path in sorted(glob.glob(os.path.join(SETTINGS.data_dir, '*.npz'))):
        try:
            with np.load(path) as z:
                shards.append(dict(name=os.path.basename(path)[:-4], ep_len=z['ep_len'].tolist()))
        except Exception:  # partially written or corrupt shard
            continue
    return JSONResponse(dict(dir=SETTINGS.data_dir, shards=shards))


@router.get('/frame')
def frame(shard: str, i: int):
    d = _shard(shard)
    i = max(0, min(i, len(d['grid']) - 1))
    focus = int(d['state'][i][3 * len(APPS)])
    meta = dict(focus=APPS[focus - 1] if focus else 'desktop', action=d['action'][i].tolist())
    return png(obs_to_pixels({k: d[k][i] for k in OBS_KEYS}), headers={'x-meta': json.dumps(meta)})
