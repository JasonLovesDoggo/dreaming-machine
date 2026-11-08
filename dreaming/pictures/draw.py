"""Canvas primitives for the 32x20 Paint canvas (8 colours, 255 = transparent in object layers)."""
import math
import numpy as np

H, W = 20, 32
WHITE, BLACK, RED, BLUE, GREEN, YELLOW, BROWN, GRAY = range(8)
CLEAR = 255
_yy, _xx = np.mgrid[0:H, 0:W].astype(np.float64)


def pk(rng, seq):
    return seq[rng.randint(len(seq))]


def alt(c, *prefs):
    return next(p for p in prefs if p != c)


def layer():
    return np.full((H, W), CLEAR, np.uint8)


def ellipse(cv, cx, cy, rx, ry, c):
    rx, ry = max(rx, 0.6), max(ry, 0.6)
    cv[((_xx - cx) / rx) ** 2 + ((_yy - cy) / ry) ** 2 <= 1] = c


def circle(cv, cx, cy, r, c):
    ellipse(cv, cx, cy, r, r, c)


def rect(cv, x0, y0, x1, y1, c):
    cv[(_xx >= min(x0, x1)) & (_xx <= max(x0, x1)) & (_yy >= min(y0, y1)) & (_yy <= max(y0, y1))] = c


def poly_mask(pts):
    inside = np.zeros((H, W), bool)
    n = len(pts)
    for i in range(n):
        (x1, y1), (x2, y2) = pts[i], pts[(i + 1) % n]
        cond = ((y1 > _yy) != (y2 > _yy)) & (_xx < (x2 - x1) * (_yy - y1) / (y2 - y1 + 1e-9) + x1)
        inside ^= cond
    return inside


def poly(cv, pts, c):
    cv[poly_mask(pts)] = c


def line(cv, x0, y0, x1, y1, c, r=0.5):
    n = int(max(abs(x1 - x0), abs(y1 - y0)) * 2) + 2
    for t in np.linspace(0, 1, n):
        circle(cv, x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, r, c)


class Pen:
    """Draws in object-local units: (x, y) * s around (cx, cy), mirrored by d (head faces -x when d=1), rotated by a.

    If `clip` is set, only pixels inside that mask are painted (used for stripes, spots and patches)."""

    def __init__(self, cv, cx, cy, s, d=1, a=0.0):
        self.cv, self.cx, self.cy, self.s, self.d, self.a = cv, cx, cy, s, d, a
        self.ca, self.sa = math.cos(a), math.sin(a)
        self.clip = None

    def at(self, x, y):
        x *= self.d
        return self.cx + self.s * (x * self.ca - y * self.sa), self.cy + self.s * (x * self.sa + y * self.ca)

    def _set(self, m, c):
        if self.clip is not None:
            m = m & self.clip
        self.cv[m] = c

    def mask(self):
        return self.cv != CLEAR

    def ell(self, x, y, rx, ry, c, rot=0.0):
        X, Y = self.at(x, y)
        t = self.a + self.d * rot
        rx, ry = max(rx * self.s, 0.6), max(ry * self.s, 0.6)
        u = (_xx - X) * math.cos(t) + (_yy - Y) * math.sin(t)
        v = -(_xx - X) * math.sin(t) + (_yy - Y) * math.cos(t)
        self._set((u / rx) ** 2 + (v / ry) ** 2 <= 1, c)

    def circ(self, x, y, r, c):
        self.ell(x, y, r, r, c)

    def poly(self, pts, c):
        self._set(poly_mask([self.at(x, y) for x, y in pts]), c)

    def rect(self, x0, y0, x1, y1, c):
        if self.a == 0:
            (X0, Y0), (X1, Y1) = self.at(x0, y0), self.at(x1, y1)
            self._set((_xx >= min(X0, X1) - 0.01) & (_xx <= max(X0, X1) + 0.01) &
                      (_yy >= min(Y0, Y1) - 0.01) & (_yy <= max(Y0, Y1) + 0.01), c)
        else:
            e = 0.5 / self.s
            x0, x1, y0, y1 = min(x0, x1) - e, max(x0, x1) + e, min(y0, y1) - e, max(y0, y1) + e
            self.poly([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], c)

    def line(self, x0, y0, x1, y1, c, w=0.5):
        (X0, Y0), (X1, Y1) = self.at(x0, y0), self.at(x1, y1)
        r = max(w * self.s, 0.5)
        n = int(max(abs(X1 - X0), abs(Y1 - Y0)) * 2) + 2
        m = np.zeros((H, W), bool)
        for t in np.linspace(0, 1, n):
            m |= (_xx - (X0 + (X1 - X0) * t)) ** 2 + (_yy - (Y0 + (Y1 - Y0) * t)) ** 2 <= r * r
        self._set(m, c)

    def path(self, pts, c, w=0.5):
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            self.line(x0, y0, x1, y1, c, w)

    def dot(self, x, y, c):
        X, Y = self.at(x, y)
        X, Y = int(round(X)), int(round(Y))
        if 0 <= X < W and 0 <= Y < H and (self.clip is None or self.clip[Y, X]):
            self.cv[Y, X] = c

    def clear(self, x, y, r):
        """Cut a hole (back to transparent)."""
        clip, self.clip = self.clip, None
        self.circ(x, y, r, CLEAR)
        self.clip = clip

    def clipped(self, mask=None):
        self.clip = self.mask() if mask is None else mask
        return self

    def unclip(self):
        self.clip = None
        return self


def pen(cv, rng, cx, cy, s, tilt=0.0, flip=True):
    d = rng.choice([-1, 1]) if flip else 1
    return Pen(cv, cx, cy, s, d, rng.uniform(-tilt, tilt) if tilt else 0.0)


def arc_pts(cx, cy, r, a0, a1, n=12):
    return [(cx + r * math.cos(a), cy + r * math.sin(a)) for a in np.linspace(a0, a1, n)]


# ------------------------------------------------------------------ backgrounds
def _ground(cv, rng, col, lo=0.62, hi=0.8):
    hz = int(H * rng.uniform(lo, hi))
    cv[hz:] = col
    return hz


def background(cv, rng, kind=None):
    kind = kind or pk(rng, ['sky', 'sky', 'plain', 'gray', 'yellow', 'grass'])
    if kind == 'sky':
        cv[:] = BLUE
        _ground(cv, rng, pk(rng, [GREEN, GREEN, BROWN]))
    elif kind == 'plain':
        cv[:] = WHITE
    elif kind == 'gray':
        cv[:] = GRAY
    elif kind == 'yellow':
        cv[:] = YELLOW
    elif kind == 'grass':
        cv[:] = GREEN
    elif kind == 'water':
        cv[:] = BLUE
        for _ in range(rng.randint(3, 7)):
            circle(cv, rng.uniform(0, W), rng.uniform(0, H), 0.6, WHITE)
    elif kind == 'night':
        cv[:] = BLACK
        for _ in range(rng.randint(4, 10)):
            cv[rng.randint(0, H), rng.randint(0, W)] = YELLOW
    elif kind == 'sea':
        cv[:] = pk(rng, [BLUE, WHITE, YELLOW])
        cv[int(H * rng.uniform(0.6, 0.75)):] = BLUE
    elif kind == 'field':
        cv[:] = BLUE
        hz = _ground(cv, rng, GREEN, 0.55, 0.75)
        for _ in range(rng.randint(3, 9)):
            cv[rng.randint(hz + 1, H), rng.randint(0, W)] = pk(rng, [RED, YELLOW, WHITE])
    elif kind == 'snow':
        cv[:] = pk(rng, [BLUE, BLUE, GRAY])
        _ground(cv, rng, WHITE, 0.6, 0.8)
        for _ in range(rng.randint(0, 10)):
            cv[rng.randint(0, H // 2), rng.randint(0, W)] = WHITE
    elif kind == 'desert':
        cv[:] = BLUE
        _ground(cv, rng, YELLOW, 0.6, 0.8)
        if rng.rand() < 0.6:
            circle(cv, rng.uniform(2, W - 2), rng.uniform(1, 4), 2, YELLOW)
    elif kind == 'savanna':
        cv[:] = pk(rng, [BLUE, YELLOW])
        _ground(cv, rng, pk(rng, [YELLOW, BROWN]) if cv[0, 0] == BLUE else BROWN, 0.62, 0.8)
    elif kind == 'beach':
        cv[:] = BLUE
        hz = int(H * rng.uniform(0.45, 0.6))
        cv[hz:] = BLUE
        cv[hz + rng.randint(2, 5):] = YELLOW
        cv[:hz] = pk(rng, [BLUE, WHITE])
    elif kind == 'under':
        cv[:] = BLUE
        cv[H - rng.randint(1, 3):] = YELLOW
        for _ in range(rng.randint(1, 4)):
            x, h = rng.randint(0, W), rng.randint(3, 8)
            for y in range(H - h, H):
                cv[y, min(W - 1, max(0, x + (y % 2)))] = GREEN
        for _ in range(rng.randint(2, 6)):
            cv[rng.randint(0, H - 3), rng.randint(0, W)] = WHITE
    elif kind == 'pond':
        cv[:] = pk(rng, [BLUE, BLUE, WHITE])
        hz = _ground(cv, rng, GREEN, 0.4, 0.55)
        cv[hz + rng.randint(1, 3):] = BLUE
    elif kind == 'room':
        wall = pk(rng, [WHITE, YELLOW, GRAY, BLUE])
        cv[:] = wall
        _ground(cv, rng, pk(rng, [BROWN, BROWN, GRAY, RED]) if wall != GRAY else BROWN, 0.65, 0.82)
        if rng.rand() < 0.5:
            x0, y0 = rng.randint(1, W - 9), rng.randint(1, 4)
            rect(cv, x0, y0, x0 + 7, y0 + 5, WHITE if wall != WHITE else BROWN)
            rect(cv, x0 + 1, y0 + 1, x0 + 6, y0 + 4, BLUE if wall != BLUE else BLACK)
    elif kind == 'blue':
        cv[:] = BLUE
    elif kind == 'leaf':
        cv[:] = GREEN
    elif kind == 'web':
        cv[:] = pk(rng, [BLACK, BLACK, BLUE, GRAY])
    return kind


def outline(cv, obj):
    """Where an object pixel touches background of the same colour, darken/lighten the background pixel."""
    out = np.where(obj == CLEAR, cv, obj).astype(np.uint8)
    bgm = obj == CLEAR
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        o = np.roll(obj, (dy, dx), (0, 1))
        if dy == 1: o[0] = CLEAR
        if dy == -1: o[-1] = CLEAR
        if dx == 1: o[:, 0] = CLEAR
        if dx == -1: o[:, -1] = CLEAR
        same = bgm & (o != CLEAR) & (o == cv)
        out[same] = np.where(cv[same] == BLACK, GRAY, BLACK)
    return out
