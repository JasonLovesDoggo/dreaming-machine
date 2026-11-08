"""First-person view of a flat ocean world matched to edge tiles; fixed cockpit and dynamic gauges."""
from functools import lru_cache

import numpy as np

from ...flight_art import (
    CLOUD,
    FRAME_COLORS,
    GRASS,
    HORIZON_SPLITS,
    PAIRS,
    RUNWAY,
    SAND,
    SEA,
    SKY,
    H,
    W,
    cockpit,
    outside_tiles,
)
from ...tiles import ATLAS
from .world import clouds, terrain

VW, VH = W, H
ART = np.stack(outside_tiles())
EDGE_END = len(ART)
BOUNDARIES = {frozenset(pair): np.array([i for i, tile in enumerate(ART) if frozenset(np.unique(tile)) == frozenset(pair)])
              for pair in PAIRS}
FRAME = cockpit()
VIEW_H, HORIZON, FOCAL = 128, 53, 150
EYE = 4
YY, XX = np.mgrid[:VIEW_H, :W]
DEPTHS = np.geomspace(6, 2200, 320)[:, None]
COLUMNS = np.arange(-64, W + 64) - W / 2
MATERIAL_COLORS = np.array([SEA, SAND, GRASS, GRASS, RUNWAY, CLOUD], np.uint8)


def render_scene(f):
    dx, dy = XX - W / 2, YY - HORIZON - f.pitch * 55
    below = dy * np.cos(f.roll) - dx * np.sin(f.roll)
    horizontal = dx * np.cos(f.roll) + dy * np.sin(f.roll)
    side = COLUMNS * DEPTHS / FOCAL
    x = f.x + DEPTHS * np.sin(f.heading) + side * np.cos(f.heading)
    y = f.y + DEPTHS * np.cos(f.heading) - side * np.sin(f.heading)
    height, material = terrain(x, y, f.seed)
    material = np.where((material >= 4) & (DEPTHS > 400), 2, material)
    material = np.where((material == 5) & (DEPTHS < 30), 4, material)
    projected = (f.alt + EYE - height) * 100 / DEPTHS
    index = np.clip(np.rint(horizontal + W / 2 + 64).astype(int), 0, len(COLUMNS) - 1)
    sea_depth = np.where(below > 0, (f.alt + EYE) * 100 / np.maximum(below, 1e-6), np.inf)
    sky = sea_depth > DEPTHS[-1, 0]
    img = np.where(sky, SKY, SEA).astype(np.uint8)
    for row in np.flatnonzero((material > 0).any(1))[::-1]:
        visible = (below >= projected[row, index]) & (DEPTHS[row, 0] < sea_depth)
        img[visible] = MATERIAL_COLORS[material[row, index[visible]]]
        sky &= ~visible
    for bearing, elevation, width in clouds(f.seed):
        offset = (bearing - f.heading + np.pi) % (2 * np.pi) - np.pi
        cx, cy = offset * FOCAL, -elevation
        puff = np.zeros(XX.shape, bool)
        for px, py, s in ((0, 0, 1), (-.55, 2, .6), (.6, 2, .65), (.15, -2.5, .55)):
            puff |= ((horizontal - cx - px * width) / (width * s)) ** 2 + \
                ((below - cy - py * 2) / (width * s * .32)) ** 2 < 1
        img[puff & sky] = CLOUD
    return img


@lru_cache(maxsize=65536)
def match_tile(data):
    patch = np.frombuffer(data, np.uint8).reshape(8, 8)
    errors = (ART != patch).sum((1, 2))
    best = int(errors.argmin())
    if errors[best] == 0:
        return best
    pair = frozenset(np.unique(patch))
    if len(pair) == 2 and pair in BOUNDARIES:
        candidates = BOUNDARIES[pair]
        return int(candidates[errors[candidates].argmin()])
    return best


def frame_tile(r, c, outside):
    rows = outside.mean(1)
    if set(np.unique(outside)) == {SKY, SEA} and np.all(np.isin(rows, (SKY, SEA))):
        split = int(np.count_nonzero(outside[:, 0] == SKY))
        if np.all(outside[:split] == SKY) and split:
            nearest = min(HORIZON_SPLITS, key=lambda value: abs(value - split))
            return ATLAS[('flight_frame_horizon', r, c, nearest)]
    values, counts = np.unique(outside, return_counts=True)
    color = int(values[counts.argmax()])
    return ATLAS[('flight_frame', r, c, color if color in FRAME_COLORS else SKY)]


def render_tiles(f):
    scene = render_scene(f)
    grid = np.empty((H // 8, W // 8), np.int32)
    for r in range(H // 8):
        for c in range(W // 8):
            patch = FRAME[r*8:r*8+8, c*8:c*8+8]
            if (patch != 255).all():
                grid[r, c] = ATLAS[('flight_panel', r, c)]
                continue
            outside = scene[r*8:r*8+8, c*8:c*8+8]
            if (patch != 255).any():
                grid[r, c] = frame_tile(r, c, outside)
            else:
                grid[r, c] = ATLAS[('flight_outside', match_tile(outside.tobytes()))]
    levels = [min(15, round(f.speed / 6 * 15)), min(2, max(0, round(f.pitch * 2) + 1)) * 5 +
              min(4, max(0, round(f.roll * 4) + 2)), min(15, round(f.alt / 350 * 15))]
    for kind, level, col in zip(('speed', 'attitude', 'altitude'), levels, (8, 12, 16)):
        for r in range(2):
            for c in range(2):
                grid[17+r, col+c] = ATLAS[('flight_dial', kind, level, r, c)]
    if f.crash:
        for r, c, pattern in [(3, 10, 0), (4, 11, 0), (5, 12, 2), (6, 12, 3), (7, 11, 1), (8, 10, 1), (5, 13, 1), (4, 14, 1)]:
            patch = scene[r*8:r*8+8, c*8:c*8+8]
            color = min(FRAME_COLORS, key=lambda value: int((patch != value).sum()))
            grid[r, c] = ATLAS[('flight_crack', color, pattern)]
    return grid
