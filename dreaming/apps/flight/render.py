"""First-person scenery matched to detailed edge tiles; fixed cockpit and dynamic gauges."""
from functools import lru_cache
import numpy as np
from .world import terrain, fbm
from ...tiles import ATLAS
from ...flight_art import W, H, FILLS, FRAME_COLORS, PAIRS, cockpit, outside_tiles

VW, VH = W, H
ART = np.stack(outside_tiles())
FRAME = cockpit()
YY, XX = np.mgrid[:112, :W]


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
    colors = np.array([1, 13, 11, 14, 5, 7], np.uint8)
    projected = (f.alt - height) * 100 / depth
    index = np.clip(np.rint(horizontal + W / 2 + 64).astype(int), 0, len(columns) - 1)
    img = np.full(XX.shape, 12, np.uint8)
    sky = np.ones(XX.shape, bool)
    for row in range(len(depths) - 1, -1, -1):
        visible = below >= projected[row, index]
        img[visible] = colors[material[row, index[visible]]]
        sky &= ~visible
    noise = fbm(horizontal / 65 + f.heading * 2 + f.x / 3000,
                below / 24 + f.y / 3000, f.seed + 10000)
    density = np.clip((noise - .015 - .2 * ((below + 27) / 50) ** 2) / .025, 0, 1)
    threshold = ((XX + YY * 2) % 4 + .5) / 4
    img[sky & (density > threshold)] = 7
    return img


@lru_cache(maxsize=65536)
def match_tile(data):
    patch = np.frombuffer(data, np.uint8).reshape(8, 8)
    errors = (ART != patch).sum((1, 2))
    best = int(errors.argmin())
    if best < len(FILLS) and errors[best] >= 10:
        edge = int(errors[len(FILLS):len(FILLS) + len(PAIRS) * 64].argmin()) + len(FILLS)
        if errors[edge] <= errors[best] + 4:
            best = edge
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
        for r in range(3):
            for c in range(3):
                grid[14+r, col+c] = ATLAS[('flight_dial', kind, level, r, c)]
    if f.crash:
        for r, c, pattern in [(3, 10, 0), (4, 11, 0), (5, 12, 2), (6, 12, 3), (7, 11, 1), (8, 10, 1), (5, 13, 1), (4, 14, 1)]:
            patch = scene[r*8:r*8+8, c*8:c*8+8]
            color = min(FRAME_COLORS, key=lambda value: int((patch != value).sum()))
            grid[r, c] = ATLAS[('flight_crack', color, pattern)]
    return grid
