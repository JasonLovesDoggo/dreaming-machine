"""First-person scenery matched to detailed edge tiles; fixed cockpit and dynamic gauges."""
from functools import lru_cache
from itertools import combinations
import numpy as np
from .world import terrain, fbm
from ...tiles import ATLAS
from ...flight_art import W, H, FILLS, FRAME_COLORS, PAIRS, edge_geometry, SKY, SKY_MID, SKY_LOW, HAZE, SEA, GRASS, HILL, SAND, cockpit, outside_tiles

VW, VH = W, H
ART = np.stack(outside_tiles())
EDGE_END = len(FILLS) + sum(a * len(offsets) for a, offsets in map(edge_geometry, PAIRS))
BOUNDARIES = {frozenset(pair): np.array([i for i, tile in enumerate(ART)
                                       if i < EDGE_END and frozenset(np.unique(tile)) == frozenset(pair)])
              for pair in combinations(FILLS, 2)}
FRAME = cockpit()
YY, XX = np.mgrid[:128, :W]


def render_scene(f):
    dx, dy = XX - W / 2, YY - 53 - f.pitch * 55
    below = dy * np.cos(f.roll) - dx * np.sin(f.roll)
    horizontal = dx * np.cos(f.roll) + dy * np.sin(f.roll)
    depths = np.geomspace(8, 2200, 56)[:, None]
    columns = np.arange(-64, W + 64) - W / 2
    depth = depths
    side = columns * depth / 150
    x = f.x + depth * np.sin(f.heading) + side * np.cos(f.heading)
    y = f.y + depth * np.cos(f.heading) - side * np.sin(f.heading)
    height, material = terrain(x, y, f.seed)
    material = np.where((material >= 4) & (depth > 250), 2, material)
    colors = np.array([SEA, SAND, GRASS, HILL, 5, 7], np.uint8)
    projected = (f.alt - height) * 100 / depth
    index = np.clip(np.rint(horizontal + W / 2 + 64).astype(int), 0, len(columns) - 1)
    img = np.where(below < -36, SKY, np.where(below < -15, SKY_MID, SKY_LOW)).astype(np.uint8)
    sky = np.ones(XX.shape, bool)
    distance = np.full(XX.shape, 2200.)
    for row in range(len(depths) - 1, -1, -1):
        visible = below >= projected[row, index]
        img[visible] = colors[material[row, index[visible]]]
        distance[visible] = depths[row, 0]
        sky &= ~visible
    threshold = ((XX + YY * 2) % 4 + .5) / 4
    haze = np.clip(1 - np.abs(below + 1) / 5, 0, 1)
    img[(haze > threshold) & (sky | (distance > 1200))] = HAZE
    noise = fbm(horizontal / 58 + f.heading * 2 + f.x / 3000,
                below / 11 + f.y / 3000, f.seed + 10000)
    for _ in range(3):
        padded = np.pad(noise, 1, mode='edge')
        noise = sum(padded[r:r+128, c:c+W] for r in range(3) for c in range(3)) / 9
    eligible = sky & (below < -18)
    cutoff = max(.12, float(np.quantile(noise[eligible], .85))) if eligible.any() else 1.
    img[eligible & (noise > cutoff)] = 7
    return img


@lru_cache(maxsize=65536)
def match_tile(data):
    patch = np.frombuffer(data, np.uint8).reshape(8, 8)
    errors = (ART != patch).sum((1, 2))
    best = int(errors.argmin())
    if errors[best] == 0:
        return best
    pair = frozenset(np.unique(patch))
    if len(pair) == 2:
        candidates = BOUNDARIES[pair]
        return int(candidates[errors[candidates].argmin()])
    return best


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
                values, counts = np.unique(outside, return_counts=True)
                color = int(values[counts.argmax()])
                if color not in FRAME_COLORS:
                    color = 14
                grid[r, c] = ATLAS[('flight_frame', r, c, color)]
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
