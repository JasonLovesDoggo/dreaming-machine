"""First-person scenery matched to detailed edge tiles; fixed cockpit and dynamic gauges."""
from functools import lru_cache
import numpy as np
from .world import terrain
from ...tiles import ATLAS
from ...flight_art import W, H, FRAME_COLORS, cloud, cockpit, outside_tiles

VW, VH = W, H
ART = np.stack(outside_tiles())
FRAME = cockpit()
CLOUD = cloud()
YY, XX = np.mgrid[:112, :W]


def render_scene(f):
    dx, dy = XX - W / 2, YY - 53 - f.pitch * 55
    below = dy * np.cos(f.roll) - dx * np.sin(f.roll)
    horizontal = dx * np.cos(f.roll) + dy * np.sin(f.roll)
    depth = np.clip(max(4, f.alt) * 100 / np.maximum(1, below), 4, 1800)
    side = horizontal * depth / 150
    x = f.x + depth * np.sin(f.heading) + side * np.cos(f.heading)
    y = f.y + depth * np.cos(f.heading) - side * np.sin(f.heading)
    _, material = terrain(x, y, f.seed)
    material = np.where((material >= 4) & (depth > 250), 2, material)
    colors = np.array([1, 13, 14, 14, 5, 7], np.uint8)
    img = np.where(below > 0, colors[material], 12).astype(np.uint8)
    for i in range(3):
        phase = round(((f.seed * 23 + i * 83 - f.heading * 60 - f.x * .02) % (W + 32) - 16) / 8) * 8
        cy = round((8 + (i * 11 + f.seed * 7) % 24 + f.pitch * 30) / 8) * 8
        if (int(f.y / 300) + i + f.seed) % 3 != 0:
            cx, ry = XX - phase, YY - cy
            mask = (cx >= 0) & (cx < 32) & (ry >= 0) & (ry < 16)
            mask &= CLOUD[np.clip(ry, 0, 15), np.clip(cx, 0, 31)] == 7
            img[mask & (below < -4)] = 7
    return img


@lru_cache(maxsize=65536)
def match_tile(data):
    patch = np.frombuffer(data, np.uint8).reshape(8, 8)
    return int((ART != patch).sum((1, 2)).argmin())


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
