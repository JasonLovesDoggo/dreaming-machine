"""Tile & sprite atlas: every token the model can output, rendered."""
import functools
import random
import numpy as np
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from .. import doom as dm
from ..apps import flight as fl
from ..os_sim import MicroOS
from ..driver import Driver
from ..tiles import ATLAS, T, grid_to_pixels
from .images import png

router = APIRouter(prefix='/sprites')
SHEET_COLS = 64
MAX_SHOWN = SHEET_COLS * 32


def sections():
    """[(name, start, end)] covering the whole vocabulary; ids before the first named range are plain UI tiles."""
    ranges = sorted(ATLAS.ranges.items(), key=lambda kv: kv[1][0])
    out = [('ui', 0, ranges[0][1][0])] if ranges and ranges[0][1][0] > 0 else []
    return out + [(name, a, b) for name, (a, b) in ranges]


def sections_json():
    return [dict(name=n, start=a, end=b, shown=min(b - a, MAX_SHOWN)) for n, a, b in sections()]


@router.get('/index')
def index():
    return JSONResponse(dict(sections=sections_json()))


@router.get('/sheet')
def sheet(start: int, n: int):
    tiles = ATLAS.array[start:start + min(n, MAX_SHOWN)]
    rows = (len(tiles) + SHEET_COLS - 1) // SHEET_COLS
    px = np.zeros((rows * T, SHEET_COLS * T), np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, SHEET_COLS)
        px[r * T:(r + 1) * T, c * T:(c + 1) * T] = tile
    return png(px, 3)


def _pad(x, b=4):
    return np.pad(x, ((b, b), (b, b)))


def _hcat(xs):
    h = max(x.shape[0] for x in xs)
    return np.concatenate([np.pad(x, ((0, h - x.shape[0]), (0, 0))) for x in xs], 1)


def _vcat(xs):
    w = max(x.shape[1] for x in xs)
    return np.concatenate([np.pad(x, ((0, 0), (0, w - x.shape[1]))) for x in xs], 0)


@functools.lru_cache(maxsize=1)
def _doom_sheet():
    """Wall textures, each sprite kind at four sizes, and a few real frames, all drawn by render_tokens."""
    textures = _hcat([_pad(np.kron(dm.TEX[t, side], np.ones((4, 4), np.uint8)))
                      for t in range(1, dm.TEX.shape[0]) for side in (0, 1)])
    wall = np.zeros((dm.VW, 4), np.uint8)
    wall[:, 0], wall[:, 2], wall[:, 3] = 1, 4, np.arange(dm.VW) % 16
    sprite_rows = []
    for kind in (1, 2, 3):
        frames = []
        for size in (6, 12, 24, 40):
            spr = np.zeros((dm.NSPR, 3), np.uint8)
            spr[0] = (kind, dm.VW // 2 + 32, size)
            frames.append(_pad(dm.render_tokens(wall, spr, 0)))
        sprite_rows.append(_hcat(frames))
    sim, rng, real = MicroOS(7), random.Random(3), []
    sim.reset()
    sim.open_app('doom')
    for k in range(600):
        ob = sim.step((128, 96, 0, rng.choice([1, 1, 3, 5, 5, 0])))
        if k % 150 == 149:
            real.append(_pad(dm.render_tokens(ob['doom_cols'], ob['doom_spr'], int(ob['doom_meta'][0]) > 0)))
    return _vcat([textures] + sprite_rows + [_hcat(real)])


@router.get('/doom')
def doom_sheet():
    return png(_doom_sheet(), 2)


@router.get('/flight')
def flight_sheet():
    sim = MicroOS(7)
    sim.open_app('flight')
    driver = Driver(sim, 7, 'flight')
    frames = []
    for i in range(230):
        sim.step(next(driver.stream))
        if i in (25, 90, 155, 220):
            frames.append(_pad(grid_to_pixels(fl.render_tiles(sim.flight))))
    return png(_hcat(frames), 2)
