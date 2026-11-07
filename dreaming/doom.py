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
            cols = self.raycast()
            center_wall = cols[VW // 2][4]
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
        dx, dy = self._dir()
        pl = 0.66
        cam = 2 * (np.arange(VW) + 0.5) / VW - 1
        rdx, rdy = dx - dy * pl * cam, dy + dx * pl * cam
        mx = np.full(VW, int(self.px)); my = np.full(VW, int(self.py))
        with np.errstate(divide='ignore'):
            ddx = np.where(rdx == 0, 1e9, np.abs(1 / rdx)); ddy = np.where(rdy == 0, 1e9, np.abs(1 / rdy))
        sx = np.where(rdx < 0, -1, 1); sy = np.where(rdy < 0, -1, 1)
        sdx = np.where(rdx < 0, (self.px - mx) * ddx, (mx + 1 - self.px) * ddx)
        sdy = np.where(rdy < 0, (self.py - my) * ddy, (my + 1 - self.py) * ddy)
        hit = np.zeros(VW, bool); side = np.zeros(VW, np.int8); wtype = np.zeros(VW, np.int8)
        for _ in range(64):
            go_x = (sdx < sdy) & ~hit
            go_y = ~(sdx < sdy) & ~hit
            sdx = np.where(go_x, sdx + ddx, sdx); mx = np.where(go_x, mx + sx, mx)
            sdy = np.where(go_y, sdy + ddy, sdy); my = np.where(go_y, my + sy, my)
            side = np.where(go_x, 0, np.where(go_y, 1, side))
            cell = GRID[np.clip(my, 0, GRID.shape[0] - 1), np.clip(mx, 0, GRID.shape[1] - 1)]
            newhit = (cell > 0) & ~hit
            wtype = np.where(newhit, cell, wtype)
            hit |= newhit
            if hit.all():
                break
        dist = np.where(side == 0, sdx - ddx, sdy - ddy)
        dist = np.maximum(dist, 1e-3)
        wallx = np.where(side == 0, self.py + dist * rdy, self.px + dist * rdx)
        u = ((wallx - np.floor(wallx)) * 16).astype(int).clip(0, 15)
        h = np.minimum(255, (VH / dist).astype(int))
        return [(int(wtype[i]), int(side[i]), int(h[i]), int(u[i]), float(dist[i])) for i in range(VW)]

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
        cols = self.raycast()
        colarr = np.array([[c[0], c[1], c[2], c[3]] for c in cols], np.uint8)
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


def render_tokens(cols, sprites, gun):
    """cols (160,4) [type, side, height, u]; sprites (4,3) [kind, x+32, size]; gun 0/1 -> (120,160) palette px."""
    img = np.where(_yy < VH // 2, K, DGRAY).astype(np.uint8)
    img[VH // 2 + 18:] = DGRAY
    wt, sd, h, u = [cols[:, i].astype(int) for i in range(4)]
    hh = np.maximum(h, 1)
    top = (VH - hh) / 2
    v = ((_yy - top[None, :]) / hh[None, :] * 16).astype(int)
    inside = (_yy >= top[None, :]) & (_yy < (top + hh)[None, :]) & (wt[None, :] > 0)
    texel = TEX[wt[None, :], sd[None, :], np.clip(v, 0, 15), u[None, :]]
    img = np.where(inside, texel, img)
    for kind, sx, size in sprites:
        if kind == 0 or size == 0:
            continue
        sx = int(sx) - 32
        size = int(size)
        rx = (_xx - sx) / (size / 2 + 1e-6)
        ry = (_yy - VH // 2) / (size / 2 + 1e-6)
        vis = (h < size)[None, :]
        if kind == 3:
            body = (rx * rx * 0.6 + (ry - 0.8) ** 2 * 20 < 0.5)
            img = np.where(body & vis, DRED, img)
            continue
        body = (rx * rx * 1.4 + ry * ry < 0.75) | ((np.abs(rx) < 0.25) & (ry > 0.4) & (ry < 0.95)) | \
               ((np.abs(rx) > 0.4) & (np.abs(rx) < 0.6) & (ry < -0.5) & (ry > -0.85))
        col = np.where(ry > -0.5, ORANGE if kind == 1 else WHITE, DRED)
        img = np.where(body & vis, col, img)
        eyes = (np.abs(np.abs(rx) - 0.28) < 0.1) & (np.abs(ry + 0.15) < 0.1)
        img = np.where(eyes & vis, YELLOW if kind == 1 else RED, img)
        mouth = (np.abs(rx) < 0.35) & (np.abs(ry - 0.25) < 0.05)
        img = np.where(mouth & vis, K, img)
    gs = 3
    gx, gy = VW // 2 - len(GUN[0]) * gs // 2, VH - len(GUN) * gs
    if gun:
        img[gy - 12:gy, VW // 2 - 6:VW // 2 + 6] = YELLOW
        img[gy - 8:gy - 2, VW // 2 - 3:VW // 2 + 3] = WHITE
    for y, row in enumerate(GUN):
        for x, ch in enumerate(row):
            if ch != '.':
                img[gy + y * gs:gy + (y + 1) * gs, gx + x * gs:gx + (x + 1) * gs] = {'k': K, 'g': LGRAY, 'd': DGRAY}[ch]
    return img
