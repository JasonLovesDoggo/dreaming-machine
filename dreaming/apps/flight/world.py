"""Seeded gradient Perlin terrain and independent fractal cloud fields."""
import numpy as np

GRADIENTS = np.stack((np.cos(np.arange(8) * np.pi / 4), np.sin(np.arange(8) * np.pi / 4)), axis=1)


def fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)


def perlin(x, y, seed=0):
    x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
    ix, iy = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    fx, fy = x - ix, y - iy
    dots = []
    for dy, dx in ((0, 0), (0, 1), (1, 0), (1, 1)):
        h = ((ix + dx) * 374761393 + (iy + dy) * 668265263 + (seed % 65536) * 1442695041) & 0xffffffff
        h = ((h ^ (h >> 13)) * 1274126177) & 0xffffffff
        gradient = GRADIENTS[(h ^ (h >> 16)) & 7]
        dots.append(gradient[..., 0] * (fx - dx) + gradient[..., 1] * (fy - dy))
    u, v = fade(fx), fade(fy)
    top = dots[0] * (1 - u) + dots[1] * u
    bottom = dots[2] * (1 - u) + dots[3] * u
    return top * (1 - v) + bottom * v


def fbm(x, y, seed=0):
    result = np.zeros(np.broadcast_shapes(np.shape(x), np.shape(y)))
    for octave in range(4):
        result += perlin(np.asarray(x) * 2 ** octave, np.asarray(y) * 2 ** octave, seed + octave * 101) * .5 ** octave
    return result / 1.875


def terrain(x, y, seed=0):
    x, y = np.broadcast_arrays(np.asarray(x), np.asarray(y))
    height = 10 + 170 * fbm(x / 700 + .37, y / 700 + .71, seed)
    for cx, cy in ((0, 0), (1200, 900)):
        lx, ly = x - cx, y - cy
        radius = np.sqrt((lx / 110) ** 2 + (ly / 280) ** 2)
        blend = fade(np.clip(radius - 1, 0, 1))
        height = 8 * (1 - blend) + height * blend
    material = np.where(height < 0, 0, np.where(height < 4, 1, np.where(height < 28, 2, 3)))
    for cx, cy in ((0, 0), (1200, 900)):
        lx, ly = x - cx, y - cy
        runway = (np.abs(lx) < 12) & (np.abs(ly) < 180)
        height = np.where(runway, 8, height)
        material = np.where(runway, np.where((np.abs(lx) < 1.5) & (ly % 24 < 12), 5, 4), material)
    return np.maximum(height, 0), material
