import numpy as np
from ..tiles import K, DRED, DGRAY, LGRAY, ORANGE, YELLOW, RED, WHITE
from .level import VW, VH
from .textures import TEX
from .game import SPRITE_X_OFFSET

IMP, IMP_PAIN, CORPSE = 1, 2, 3  # sprite kinds (imp state + 1)
GUN = ["....kk....",
       "...kggk...",
       "...kggk...",
       "..kggggk..",
       "..kgddgk..",
       ".kggddggk.",
       ".kggggggk.",
       "kkkkkkkkkk"]
GUN_SCALE = 3
_yy, _xx = np.mgrid[0:VH, 0:VW]
_SKY_FLOOR = np.where(_yy < VH // 2, K, DGRAY).astype(np.uint8)


def _gun_layers():
    """(colour, mask) pairs for the gun and its muzzle flash."""
    gx, gy = VW // 2 - len(GUN[0]) * GUN_SCALE // 2, VH - len(GUN) * GUN_SCALE
    gun, gun_mask = np.zeros((VH, VW), np.uint8), np.zeros((VH, VW), bool)
    for y, row in enumerate(GUN):
        for x, ch in enumerate(row):
            if ch != '.':
                sl = np.s_[gy + y * GUN_SCALE:gy + (y + 1) * GUN_SCALE, gx + x * GUN_SCALE:gx + (x + 1) * GUN_SCALE]
                gun[sl] = {'k': K, 'g': LGRAY, 'd': DGRAY}[ch]
                gun_mask[sl] = True
    flash, flash_mask = np.zeros((VH, VW), np.uint8), np.zeros((VH, VW), bool)
    flash[gy - 12:gy, VW // 2 - 6:VW // 2 + 6] = YELLOW
    flash_mask[gy - 12:gy, VW // 2 - 6:VW // 2 + 6] = True
    flash[gy - 8:gy - 2, VW // 2 - 3:VW // 2 + 3] = WHITE
    return gun, gun_mask, flash, flash_mask


_GUN, _GUN_MASK, _FLASH, _FLASH_MASK = _gun_layers()


def _walls(cols):
    wt, sd, h, u = [cols[:, i].astype(np.int64) for i in range(4)]
    hh = np.maximum(h, 1)
    top = (VH - hh) / 2
    v = ((_yy - top) / hh * 16).astype(np.int64)
    inside = (_yy >= top) & (_yy < top + hh) & (wt > 0)
    return np.where(inside, TEX[wt, sd, np.clip(v, 0, 15), u], _SKY_FLOOR), h


def _draw_sprite(img, wall_h, kind, sx, size):
    """Draw one billboard in place; columns whose wall is nearer (taller) than the sprite occlude it."""
    x0, x1 = max(0, sx - size), min(VW, sx + size + 1)
    y0, y1 = max(0, VH // 2 - size), min(VH, VH // 2 + size + 1)
    if x0 >= x1 or y0 >= y1:
        return
    sub = img[y0:y1, x0:x1]
    rx = (_xx[y0:y1, x0:x1] - sx) / (size / 2 + 1e-6)
    ry = (_yy[y0:y1, x0:x1] - VH // 2) / (size / 2 + 1e-6)
    vis = (wall_h[x0:x1] < size)[None, :]
    if kind == CORPSE:
        sub[(rx * rx * 0.6 + (ry - 0.8) ** 2 * 20 < 0.5) & vis] = DRED
        return
    torso = rx * rx * 1.4 + ry * ry < 0.75
    legs = (np.abs(rx) < 0.25) & (ry > 0.4) & (ry < 0.95)
    horns = (np.abs(rx) > 0.4) & (np.abs(rx) < 0.6) & (ry < -0.5) & (ry > -0.85)
    colour = np.where(ry > -0.5, ORANGE if kind == IMP else WHITE, DRED)
    m = (torso | legs | horns) & vis
    sub[m] = colour[m]
    eyes = (np.abs(np.abs(rx) - 0.28) < 0.1) & (np.abs(ry + 0.15) < 0.1)
    sub[eyes & vis] = YELLOW if kind == IMP else RED
    sub[(np.abs(rx) < 0.35) & (np.abs(ry - 0.25) < 0.05) & vis] = K  # mouth


def render_tokens(cols, sprites, gun):
    """cols (VW,4) [type, side, height, u]; sprites (NSPR,3) [kind, x+32, size]; gun 0/1 -> (VH,VW) palette px."""
    img, wall_h = _walls(cols)
    for kind, sx, size in sprites:
        if int(kind) and int(size):
            _draw_sprite(img, wall_h, int(kind), int(sx) - SPRITE_X_OFFSET, int(size))
    if gun:
        img = np.where(_FLASH_MASK, _FLASH, img)
    return np.where(_GUN_MASK, _GUN, img).astype(np.uint8)
