"""Doom-lite: deterministic textured raycaster. Frames are fully described by column + sprite tokens."""
import math
import numpy as np
from .tiles import K, NAVY, DRED, DGRAY, LGRAY, RED, ORANGE, YELLOW, GREEN, BLUE, DGREEN, WHITE, BROWN

VW, VH = 160, 120
NSPR = 4
MAP = [
    "################",
    "#......#.......#",
    "#......#..BBB..#",
    "#..TT..........#",
    "#..TT.....B....#",
    "#.........B....#",
    "#....######....#",
    "#..............#",
    "#..#......TT...#",
    "#..#...........#",
    "#..............#",
    "################",
]
GRID = np.array([[{'.': 0, '#': 1, 'B': 2, 'T': 3}[c] for c in row] for row in MAP], np.int8)
NANG = 48
SPAWNS = [(12.5, 4.5), (5.5, 9.5), (13.5, 9.5)]
START = (2.5, 2.5, 0)
CAM = 2 * (np.arange(160) + 0.5) / 160 - 1


def _textures():
    b = np.full((16, 16), RED, np.uint8)
    b[::4, :] = DRED
    for r in range(4):
        for x in range(0 if r % 2 == 0 else 4, 16, 8):
            b[r * 4:r * 4 + 4, x] = DRED
    t = np.full((16, 16), BLUE, np.uint8)
    t[0, :] = NAVY; t[:, 0] = NAVY; t[8, :] = NAVY; t[:, 8] = NAVY
    t[3:5, 3:5] = YELLOW; t[11:13, 11:13] = YELLOW
    s = np.full((16, 16), GREEN, np.uint8)
    for x, ln in zip(range(1, 16, 3), [5, 9, 3, 12, 7]):
        s[:ln, x] = DGREEN
    s[14:, :] = DGREEN
    tex = np.zeros((4, 2, 16, 16), np.uint8)
    dark = np.arange(16, dtype=np.uint8)
    for a, bb in {RED: DRED, DRED: K, BLUE: NAVY, NAVY: K, YELLOW: ORANGE, GREEN: DGREEN, DGREEN: K}.items():
        dark[a] = bb
    for i, tx in enumerate([b, t, s], start=1):
        tex[i, 0] = tx
        tex[i, 1] = dark[tx]
    return tex


TEX = _textures()


class Doom:
    def __init__(self):
        self.reset()

    def reset(self):
        self.px, self.py, self.ang = START
        self.hp, self.ammo, self.kills = 100, 50, 0
        self.flash = 0
        self.t = 0
        self.imps = [dict(x=x, y=y, hp=2, state=0, timer=0) for x, y in SPAWNS]  # state 0 alive,1 pain,2 dead

    def _dir(self):
        a = self.ang * 2 * math.pi / NANG
        return math.cos(a), math.sin(a)

    def _free(self, x, y):
        r = 0.2
        return all(GRID[int(y + dy), int(x + dx)] == 0 for dx in (-r, r) for dy in (-r, r))

    def step(self, key, fire):
        """key: 0 none,1 fwd,2 back,3 left,4 right,5 fire."""
        events = []
        self.t += 1
        dx, dy = self._dir()
        if key in (1, 2):
            s = 0.18 if key == 1 else -0.12
            nx, ny = self.px + dx * s, self.py + dy * s
            if self._free(nx, self.py): self.px = nx
            if self._free(self.px, ny): self.py = ny
        elif key == 3:
            self.ang = (self.ang - 1) % NANG
        elif key == 4:
            self.ang = (self.ang + 1) % NANG
        self.flash = max(0, self.flash - 1)
        if (key == 5 or fire) and self.ammo > 0 and self.flash == 0:
            self.ammo -= 1
            self.flash = 2
            events.append('doom_fire')
            center_wall = self.raycast()[4][VW // 2]
            best = None
            for imp, (sx, size, depth) in zip(self.imps, self._project_all()):
                if imp['state'] != 2 and depth > 0 and abs(sx - VW // 2) < max(3, size // 4) and depth < center_wall:
                    if best is None or depth < best[1]:
                        best = (imp, depth)
            if best:
                imp = best[0]
                imp['hp'] -= 1
                imp['state'], imp['timer'] = 1, 3
                events.append('doom_hit')
                if imp['hp'] <= 0:
                    imp['state'], imp['timer'] = 2, 40
                    self.kills += 1
                    events.append('doom_kill')
        if self.t % 10 == 0 and self.ammo < 50:
            self.ammo += 1
        for i, imp in enumerate(self.imps):
            if imp['state'] == 2:
                imp['timer'] -= 1
                if imp['timer'] <= 0:
                    imp.update(x=SPAWNS[i][0], y=SPAWNS[i][1], hp=2, state=0)
                continue
            if imp['state'] == 1:
                imp['timer'] -= 1
                if imp['timer'] <= 0:
                    imp['state'] = 0
                continue
            ddx, ddy = self.px - imp['x'], self.py - imp['y']
            d = math.hypot(ddx, ddy)
            if 0.9 < d < 7:
                nx, ny = imp['x'] + ddx / d * 0.04, imp['y'] + ddy / d * 0.04
                if self._free(nx, imp['y']): imp['x'] = nx
                if self._free(imp['x'], ny): imp['y'] = ny
            if d <= 1.0 and self.t % 8 == 0:
                self.hp -= 7
                events.append('doom_hurt')
        if self.hp <= 0:
            events.append('doom_die')
            k = self.kills
            self.reset()
            self.kills = k
        return events

    def raycast(self):
        """Returns (wtype, side, h, u, dist) arrays over the VW screen columns. Cached per pose."""
        key = (self.px, self.py, self.ang)
        if getattr(self, '_rc_key', None) == key:
            return self._rc
        dx, dy = self._dir()
        cam = CAM
        rdx, rdy = dx - dy * 0.66 * cam, dy + dx * 0.66 * cam
        mx = np.full(VW, int(self.px)); my = np.full(VW, int(self.py))
        with np.errstate(divide='ignore'):
            ddx = np.where(rdx == 0, 1e9, np.abs(1 / rdx)); ddy = np.where(rdy == 0, 1e9, np.abs(1 / rdy))
        sx = np.where(rdx < 0, -1, 1); sy = np.where(rdy < 0, -1, 1)
        sdx = np.where(rdx < 0, (self.px - mx) * ddx, (mx + 1 - self.px) * ddx)
        sdy = np.where(rdy < 0, (self.py - my) * ddy, (my + 1 - self.py) * ddy)
        side = np.zeros(VW, np.int64); wtype = np.zeros(VW, np.int64)
        active = np.arange(VW)
        while active.size:
            a_sdx, a_sdy = sdx[active], sdy[active]
            gx = a_sdx < a_sdy
            ix, iy = active[gx], active[~gx]
            sdx[ix] += ddx[ix]; mx[ix] += sx[ix]; side[ix] = 0
            sdy[iy] += ddy[iy]; my[iy] += sy[iy]; side[iy] = 1
            cell = GRID[my[active], mx[active]]
            hit = cell > 0
            wtype[active[hit]] = cell[hit]
            active = active[~hit]
        dist = np.maximum(np.where(side == 0, sdx - ddx, sdy - ddy), 1e-3)
        wallx = np.where(side == 0, self.py + dist * rdy, self.px + dist * rdx)
        u = ((wallx - np.floor(wallx)) * 16).astype(np.int64).clip(0, 15)
        h = np.minimum(255, (VH / dist).astype(np.int64))
        self._rc_key, self._rc = key, (wtype, side, h, u, dist)
        return self._rc

    def _project_all(self):
        dx, dy = self._dir()
        pl = 0.66
        plx, ply = -dy * pl, dx * pl
        inv = 1.0 / (plx * dy - dx * ply)
        out = []
        for imp in self.imps:
            rx, ry = imp['x'] - self.px, imp['y'] - self.py
            tx = inv * (dy * rx - dx * ry)
            ty = inv * (-ply * rx + plx * ry)
            if ty <= 0.2:
                out.append((-1, 0, -1.0))
                continue
            sx = int(VW / 2 * (1 + tx / ty))
            out.append((sx, min(255, int(VH / ty)), ty))
        return out

    def tokens(self):
        wt, sd, h, u, _ = self.raycast()
        colarr = np.stack([wt, sd, h, u], 1).astype(np.uint8)
        spr = []
        for imp, (sx, size, depth) in zip(self.imps, self._project_all()):
            if depth > 0 and -size // 2 < sx < VW + size // 2:
                spr.append((depth, imp['state'] + 1, sx, size))
        spr.sort(key=lambda s: -s[0])
        sprarr = np.zeros((NSPR, 3), np.uint8)
        for i, (_, kind, sx, size) in enumerate(spr[:NSPR]):
            sprarr[i] = (kind, int(np.clip(sx + 32, 0, 255)), size)
        return colarr, sprarr, int(self.flash > 0)

    def hud_text(self):
        return f"HP {max(self.hp, 0):3d}  AMMO {self.ammo:2d}  KILLS {self.kills % 100:2d}"

    def state_fields(self):
        return [int(self.px) % 16, int(self.py) % 12, self.ang // 3, max(0, self.hp) // 10, self.ammo // 5,
                self.kills % 10]


GUN = ["....kk....", "...kggk...", "...kggk...", "..kggggk..", "..kgddgk..", ".kggddggk.", ".kggggggk.", "kkkkkkkkkk"]
_yy, _xx = np.mgrid[0:VH, 0:VW]
_BG = np.where(_yy < VH // 2, K, DGRAY).astype(np.uint8)


def _gun_layers():
    gs = 3
    gx, gy = VW // 2 - len(GUN[0]) * gs // 2, VH - len(GUN) * gs
    col = np.zeros((VH, VW), np.uint8); mask = np.zeros((VH, VW), bool)
    for y, row in enumerate(GUN):
        for x, ch in enumerate(row):
            if ch != '.':
                col[gy + y * gs:gy + (y + 1) * gs, gx + x * gs:gx + (x + 1) * gs] = {'k': K, 'g': LGRAY, 'd': DGRAY}[ch]
                mask[gy + y * gs:gy + (y + 1) * gs, gx + x * gs:gx + (x + 1) * gs] = True
    fcol = np.zeros((VH, VW), np.uint8); fmask = np.zeros((VH, VW), bool)
    fcol[gy - 12:gy, VW // 2 - 6:VW // 2 + 6] = YELLOW; fmask[gy - 12:gy, VW // 2 - 6:VW // 2 + 6] = True
    fcol[gy - 8:gy - 2, VW // 2 - 3:VW // 2 + 3] = WHITE
    return col, mask, fcol, fmask


_GUN_COL, _GUN_MASK, _FLASH_COL, _FLASH_MASK = _gun_layers()


def render_tokens(cols, sprites, gun):
    """cols (160,4) [type, side, height, u]; sprites (4,3) [kind, x+32, size]; gun 0/1 -> (120,160) palette px."""
    wt, sd, h, u = [cols[:, i].astype(np.int64) for i in range(4)]
    hh = np.maximum(h, 1)
    top = (VH - hh) / 2
    v = ((_yy - top) / hh * 16).astype(np.int64)
    inside = (_yy >= top) & (_yy < top + hh) & (wt > 0)
    img = np.where(inside, TEX[wt, sd, np.clip(v, 0, 15), u], _BG)
    for kind, sx, size in sprites:
        kind, sx, size = int(kind), int(sx) - 32, int(size)
        if kind == 0 or size == 0:
            continue
        x0, x1 = max(0, sx - size), min(VW, sx + size + 1)
        y0, y1 = max(0, VH // 2 - size), min(VH, VH // 2 + size + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        sub = img[y0:y1, x0:x1]
        rx = (_xx[y0:y1, x0:x1] - sx) / (size / 2 + 1e-6)
        ry = (_yy[y0:y1, x0:x1] - VH // 2) / (size / 2 + 1e-6)
        vis = (h[x0:x1] < size)[None, :]
        if kind == 3:
            sub[(rx * rx * 0.6 + (ry - 0.8) ** 2 * 20 < 0.5) & vis] = DRED
            continue
        body = (rx * rx * 1.4 + ry * ry < 0.75) | ((np.abs(rx) < 0.25) & (ry > 0.4) & (ry < 0.95)) | \
               ((np.abs(rx) > 0.4) & (np.abs(rx) < 0.6) & (ry < -0.5) & (ry > -0.85))
        col = np.where(ry > -0.5, ORANGE if kind == 1 else WHITE, DRED)
        m = body & vis
        sub[m] = col[m]
        sub[(np.abs(np.abs(rx) - 0.28) < 0.1) & (np.abs(ry + 0.15) < 0.1) & vis] = YELLOW if kind == 1 else RED
        sub[(np.abs(rx) < 0.35) & (np.abs(ry - 0.25) < 0.05) & vis] = K
    if gun:
        img = np.where(_FLASH_MASK, _FLASH_COL, img)
    return np.where(_GUN_MASK, _GUN_COL, img).astype(np.uint8)
