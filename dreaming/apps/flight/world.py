"""Periodic deterministic archipelago with two paved airfields."""
import numpy as np


def terrain(x, y, seed=0):
    x, y = np.asarray(x), np.asarray(y)
    h = np.maximum(0, 24 + 22 * np.sin(x / 140 + seed % 17) + 26 * np.cos(y / 190)
                   + 12 * np.sin((x + y) / 53))
    color = np.where(h < 1, 0, np.where(h < 12, 1, np.where(h < 48, 2, 3)))
    distance = np.maximum(np.abs((x + 512) % 1024 - 512) / 100,
                          np.abs((y + 512) % 1024 - 512) / 300)
    h = 8 + (h - 8) * np.clip(distance - 1, 0, 1)
    runway = (np.abs((x + 512) % 1024 - 512) < 12) & (np.abs((y + 512) % 1024 - 512) < 180)
    h = np.where(runway, 8, h)
    color = np.where(runway, np.where((np.abs(x % 1024) < 1.5) & (y % 24 < 12), 5, 4), color)
    return h, color
