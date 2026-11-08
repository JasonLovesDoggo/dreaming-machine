import math
import numpy as np
from .level import VW, NSPR, NANG, SPAWNS, START, is_free
from .raycast import cast, direction, project

ALIVE, PAIN, DEAD = 0, 1, 2
KEY_FORWARD, KEY_BACK, KEY_LEFT, KEY_RIGHT, KEY_FIRE = 1, 2, 3, 4, 5
WALK, BACKPEDAL = 0.18, -0.12
IMP_SPEED, IMP_SIGHT, IMP_REACH = 0.04, 7, 1.0
IMP_DAMAGE, IMP_ATTACK_EVERY = 7, 8
PAIN_FRAMES, CORPSE_FRAMES = 3, 40
MAX_AMMO, AMMO_REGEN_EVERY = 50, 10
SPRITE_X_OFFSET = 32  # sprite x tokens are shifted so slightly off-screen sprites stay representable


class Doom:
    def __init__(self):
        self.reset()

    def reset(self):
        self.px, self.py, self.ang = START
        self.hp, self.ammo, self.kills = 100, MAX_AMMO, 0
        self.flash = 0
        self.t = 0
        self.imps = [dict(x=x, y=y, hp=2, state=ALIVE, timer=0) for x, y in SPAWNS]

    # ---------------------------------------------------------------- simulation
    def step(self, key, fire):
        """key: 0 none, 1 forward, 2 back, 3 turn left, 4 turn right, 5 fire. Returns event names."""
        events = []
        self.t += 1
        self._move(key)
        self.flash = max(0, self.flash - 1)
        if (key == KEY_FIRE or fire) and self.ammo > 0 and self.flash == 0:
            events += self._shoot()
        if self.t % AMMO_REGEN_EVERY == 0 and self.ammo < MAX_AMMO:
            self.ammo += 1
        for i, imp in enumerate(self.imps):
            events += self._update_imp(i, imp)
        if self.hp <= 0:
            events.append('doom_die')
            kills = self.kills
            self.reset()
            self.kills = kills
        return events

    def _move(self, key):
        if key in (KEY_FORWARD, KEY_BACK):
            dx, dy = direction(self.ang)
            s = WALK if key == KEY_FORWARD else BACKPEDAL
            self.px, self.py = _slide(self.px, self.py, dx * s, dy * s)
        elif key == KEY_LEFT:
            self.ang = (self.ang - 1) % NANG
        elif key == KEY_RIGHT:
            self.ang = (self.ang + 1) % NANG

    def _shoot(self):
        self.ammo -= 1
        self.flash = 2
        events = ['doom_fire']
        wall = self.raycast()[4][VW // 2]
        hits = [(depth, imp) for imp, (sx, size, depth) in zip(self.imps, self.project_imps())
                if imp['state'] != DEAD and 0 < depth < wall and abs(sx - VW // 2) < max(3, size // 4)]
        if hits:
            imp = min(hits, key=lambda h: h[0])[1]
            imp['hp'] -= 1
            imp['state'], imp['timer'] = PAIN, PAIN_FRAMES
            events.append('doom_hit')
            if imp['hp'] <= 0:
                imp['state'], imp['timer'] = DEAD, CORPSE_FRAMES
                self.kills += 1
                events.append('doom_kill')
        return events

    def _update_imp(self, i, imp):
        if imp['state'] == DEAD:
            imp['timer'] -= 1
            if imp['timer'] <= 0:
                imp.update(x=SPAWNS[i][0], y=SPAWNS[i][1], hp=2, state=ALIVE)
            return []
        if imp['state'] == PAIN:
            imp['timer'] -= 1
            if imp['timer'] <= 0:
                imp['state'] = ALIVE
            return []
        ddx, ddy = self.px - imp['x'], self.py - imp['y']
        d = math.hypot(ddx, ddy)
        if 0.9 < d < IMP_SIGHT:
            imp['x'], imp['y'] = _slide(imp['x'], imp['y'], ddx / d * IMP_SPEED, ddy / d * IMP_SPEED)
        if d <= IMP_REACH and self.t % IMP_ATTACK_EVERY == 0:
            self.hp -= IMP_DAMAGE
            return ['doom_hurt']
        return []

    # ---------------------------------------------------------------- views
    def raycast(self):
        """(wall_type, side, height, u, dist) per screen column, cached per pose."""
        pose = (self.px, self.py, self.ang)
        if getattr(self, '_rc_pose', None) != pose:
            self._rc_pose, self._rc = pose, cast(*pose)
        return self._rc

    def direction(self):
        return direction(self.ang)

    def project_imps(self):
        """[(screen_x, size, depth)] for each imp, in self.imps order."""
        return project(self.px, self.py, self.ang, [(imp['x'], imp['y']) for imp in self.imps])

    def tokens(self):
        """-> cols (VW,4) [type, side, height, u], sprites (NSPR,3) [kind, x+32, size] far-to-near, gun flash 0/1."""
        wt, sd, h, u, _ = self.raycast()
        cols = np.stack([wt, sd, h, u], 1).astype(np.uint8)
        visible = sorted(((depth, imp['state'] + 1, sx, size)
                          for imp, (sx, size, depth) in zip(self.imps, self.project_imps())
                          if depth > 0 and -size // 2 < sx < VW + size // 2), key=lambda s: -s[0])
        sprites = np.zeros((NSPR, 3), np.uint8)
        for i, (_, kind, sx, size) in enumerate(visible[:NSPR]):
            sprites[i] = (kind, int(np.clip(sx + SPRITE_X_OFFSET, 0, 255)), size)
        return cols, sprites, int(self.flash > 0)

    def hud_text(self):
        return f"HP {max(self.hp, 0):3d}  AMMO {self.ammo:2d}  KILLS {self.kills % 100:2d}"

    def state_fields(self):
        return [int(self.px) % 16, int(self.py) % 12, self.ang // 3, max(0, self.hp) // 10, self.ammo // 5,
                self.kills % 10]


def _slide(x, y, dx, dy):
    """Move by (dx, dy), sliding along walls one axis at a time."""
    if is_free(x + dx, y):
        x += dx
    if is_free(x, y + dy):
        y += dy
    return x, y
