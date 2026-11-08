import numpy as np
from ..tiles import K, NAVY, DRED, RED, ORANGE, YELLOW, GREEN, BLUE, DGREEN

SIZE = 16
DARKER = {RED: DRED, DRED: K, BLUE: NAVY, NAVY: K, YELLOW: ORANGE, GREEN: DGREEN, DGREEN: K}


def _brick():
    t = np.full((SIZE, SIZE), RED, np.uint8)
    t[::4, :] = DRED
    for row in range(4):
        for x in range(0 if row % 2 == 0 else 4, SIZE, 8):
            t[row * 4:row * 4 + 4, x] = DRED
    return t


def _tech():
    t = np.full((SIZE, SIZE), BLUE, np.uint8)
    t[[0, 8], :] = NAVY
    t[:, [0, 8]] = NAVY
    t[3:5, 3:5] = YELLOW
    t[11:13, 11:13] = YELLOW
    return t


def _slime():
    t = np.full((SIZE, SIZE), GREEN, np.uint8)
    for x, drip in zip(range(1, SIZE, 3), [5, 9, 3, 12, 7]):
        t[:drip, x] = DGREEN
    t[14:, :] = DGREEN
    return t


def _build():
    """TEX[wall_type, side, v, u]; side 1 is the darker variant. Type 0 (empty) stays black."""
    shade = np.arange(16, dtype=np.uint8)
    for a, b in DARKER.items():
        shade[a] = b
    tex = np.zeros((4, 2, SIZE, SIZE), np.uint8)
    for wall_type, t in enumerate([_brick(), _tech(), _slime()], start=1):
        tex[wall_type, 0] = t
        tex[wall_type, 1] = shade[t]
    return tex


TEX = _build()
