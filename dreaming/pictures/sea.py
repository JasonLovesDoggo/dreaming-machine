"""Sea creatures and bugs."""
import math
import numpy as np

from .draw import CLEAR, WHITE, BLACK, RED, BLUE, GREEN, YELLOW, BROWN, GRAY, pen, pk, alt


def whale(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s)
    col = pk(rng, [BLUE, GRAY, GRAY, BLACK])
    p.poly([(5.4, 0.6), (9.4, -2.8), (8.2, 0.8), (9.6, 3.2)], col)
    p.ell(-1.0, 1.0, 6.6, 3.4, col)
    p.clipped()
    p.ell(-1.6, 4.0, 5.8, 1.8, WHITE if col != WHITE else GRAY)
    p.unclip()
    p.ell(0.0, 4.2, 1.6, 0.6, col, rot=0.4)
    p.dot(-4.6, 0.4, WHITE if col == BLACK else BLACK)
    p.line(-7.2, 2.4, -4.0, 2.6, BLACK if col != BLACK else GRAY, 0.25)
    if rng.rand() < 0.6:
        p.path([(-3.0, -2.4), (-3.0, -4.6), (-4.4, -5.8)], WHITE, 0.3)
        p.line(-3.0, -4.6, -1.6, -5.8, WHITE, 0.3)


def shark(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s, tilt=0.2)
    col = pk(rng, [GRAY, GRAY, BLUE])
    p.poly([(5.6, 0.4), (9.2, -3.4), (8.0, 0.6), (9.0, 3.2)], col)
    p.poly([(-0.8, -1.4), (1.4, -5.4), (2.2, -1.4)], col)
    p.ell(-0.6, 0.6, 6.6, 2.2, col)
    p.clipped()
    p.ell(-1.0, 2.6, 6.4, 1.2, WHITE)
    p.unclip()
    p.poly([(-1.0, 2.0), (1.6, 5.0), (1.2, 2.0)], col)
    p.dot(-4.8, -0.2, BLACK)
    for k in range(3):
        p.dot(-2.8 + k * 0.6, 0.6, BLACK)
    p.line(-7.0, 1.4, -4.6, 1.8, BLACK, 0.25)


def dolphin(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s, tilt=0.6)
    col = pk(rng, [GRAY, BLUE, GRAY])
    p.path([(-4.0, 0.2), (-1.0, -1.0), (3.0, -0.6), (6.4, 1.2)], col, 1.5)
    p.poly([(6.0, 1.0), (8.4, -0.8), (7.4, 1.6), (8.4, 3.4)], col)
    p.poly([(-0.6, -1.8), (1.2, -4.0), (1.6, -1.4)], col)
    p.poly([(-6.6, 0.8), (-4.0, -0.6), (-4.0, 1.4)], col)
    p.ell(-1.0, 0.6, 1.2, 0.5, col, rot=0.6)
    p.dot(-3.6, -0.4, BLACK)
    p.line(-6.0, 1.0, -3.8, 1.2, WHITE, 0.2)


def octopus(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s, flip=False)
    col = pk(rng, [RED, RED, BROWN, YELLOW, GRAY])
    ph = rng.uniform(0, math.pi)
    for k in range(6):
        x0 = -3.0 + k * 1.2
        pts = [(x0 + (x0 * 0.6) * t + 0.9 * math.sin(t * 5 + ph + k), 1.0 + 5.4 * t) for t in np.linspace(0, 1, 7)]
        p.path(pts, col, 0.55)
    p.ell(0, -2.0, 3.6, 3.8, col)
    for sd in (-1, 1):
        p.circ(sd * 1.4, -0.8, 0.8, WHITE)
        p.dot(sd * 1.4, -0.6, BLACK)


def crab(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s, flip=False)
    col = pk(rng, [RED, RED, BROWN, YELLOW])
    for sd in (-1, 1):
        for k in range(3):
            p.path([(sd * 2.6, 2.4 + k * 0.5), (sd * 4.6, 2.2 + k * 1.0), (sd * 5.4, 4.6 + k * 0.8)], col, 0.3)
        p.path([(sd * 2.6, 0.8), (sd * 4.6, -0.8), (sd * 5.0, -2.6)], col, 0.4)
        p.circ(sd * 5.2, -3.6, 1.4, col)
        p.clear(sd * 5.2, -4.8, 0.6)
        p.line(sd * 1.0, 0.4, sd * 1.2, -1.6, col, 0.25)
        p.circ(sd * 1.2, -2.0, 0.6, WHITE)
        p.dot(sd * 1.2, -2.0, BLACK)
    p.ell(0, 2.4, 3.4, 2.0, col)


def jellyfish(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s, flip=False)
    col = pk(rng, [RED, WHITE, YELLOW, GRAY])
    p.ell(0, 0.0, 3.8, 3.4, col)
    p.rect(-4.0, 0.4, 4.0, 4.0, CLEAR)
    ph = rng.uniform(0, 6)
    for k in range(5):
        x0 = -2.4 + k * 1.2
        p.path([(x0 + 0.7 * math.sin(t * 4 + ph + k), 0.2 + 6.6 * t) for t in np.linspace(0, 1, 8)], col, 0.25)
    for sd in (-1, 1):
        p.dot(sd * 1.2, -1.2, BLACK)


def starfish(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s, tilt=0.6, flip=False)
    col = pk(rng, [RED, YELLOW, YELLOW, BROWN])
    pts = []
    for k in range(10):
        a = -math.pi / 2 + k * math.pi / 5
        r = 5.2 if k % 2 == 0 else 2.0
        pts.append((r * math.cos(a), 1.0 + r * math.sin(a)))
    p.poly(pts, col)
    for k in range(5):
        a = -math.pi / 2 + k * 2 * math.pi / 5
        p.dot(2.6 * math.cos(a), 1.0 + 2.6 * math.sin(a), alt(col, WHITE, RED))


def seahorse(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s)
    col = pk(rng, [YELLOW, RED, GREEN, BROWN])
    p.path([(0.6, -2.0), (1.6, 0.6), (1.0, 3.4), (-0.4, 5.6), (0.6, 7.0), (1.6, 6.2)], col, 0.85)
    p.ell(0.4, 0.6, 1.4, 2.2, col, rot=0.2)
    p.circ(0.0, -3.4, 1.3, col)
    p.line(-0.8, -3.0, -3.0, -2.4, col, 0.45)
    p.poly([(1.6, -0.6), (3.0, 0.4), (1.8, 1.6)], alt(col, YELLOW, RED))
    p.dot(-0.2, -3.6, BLACK)
    for y in (-2.0, 0.0, 2.0):
        p.dot(1.8, y, alt(col, BROWN, BLACK))


def shrimp(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s)
    col = pk(rng, [RED, RED, YELLOW])
    pts = [(-3.6 + 7.2 * (1 - math.cos(t * 2.2)) / 1.6, 1.4 - 4.0 * math.sin(t * 2.2) + 2.0) for t in np.linspace(0, 1, 10)]
    for i, (x, y) in enumerate(pts):
        p.circ(x, y - 2.0, 1.5 - i * 0.08, col)
    hx, hy = pts[0]
    p.dot(hx - 0.4, hy - 2.6, BLACK)
    p.path([(hx - 1.0, hy - 2.8), (hx - 4.0, hy - 5.6), (hx - 6.0, hy - 5.0)], col, 0.2)
    for x in np.arange(-2.6, 3.0, 1.2):
        p.line(x, 2.4, x - 0.4, 4.0, col, 0.2)


# ------------------------------------------------------------------ bugs
def butterfly(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s, tilt=0.35, flip=False)
    w1, w2 = pk(rng, [(YELLOW, RED), (BLUE, BLACK), (RED, YELLOW), (YELLOW, BLACK), (WHITE, BLACK), (BLUE, YELLOW)])
    open_ = rng.uniform(0.55, 1.0)
    for sd in (-1, 1):
        p.ell(sd * 3.2 * open_, -1.8, 3.0 * open_, 2.8, w1, rot=sd * 0.35)
        p.ell(sd * 2.4 * open_, 2.4, 2.2 * open_, 1.9, w1, rot=-sd * 0.4)
        p.circ(sd * 3.6 * open_, -2.0, 1.0 * open_, w2)
        p.dot(sd * 2.4 * open_, 2.4, w2)
        p.path([(0, -3.0), (sd * 1.0, -5.2), (sd * 2.0, -6.0)], BLACK, 0.2)
    p.ell(0, 0, 0.55, 3.4, BLACK)


def bee(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s, tilt=0.25)
    for sd in (-1.6, 0.6):
        p.ell(sd, -3.4, 1.6, 2.4, WHITE, rot=-0.3 + 0.2 * sd)
    p.ell(0.8, 1.0, 4.2, 3.0, YELLOW)
    p.clipped()
    for x in (-0.4, 1.6, 3.4):
        p.rect(x, -3, x + 0.6, 5, BLACK)
    p.unclip()
    p.poly([(4.8, 0.4), (6.6, 1.0), (4.8, 1.8)], BLACK)
    p.circ(-3.8, 0.6, 1.8, BLACK)
    p.dot(-4.4, 0.2, WHITE)
    p.path([(-4.2, -1.0), (-4.8, -2.8), (-5.8, -3.2)], BLACK, 0.2)


def ladybug(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s, tilt=0.5, flip=False)
    col = pk(rng, [RED, RED, RED, YELLOW])
    for sd in (-1, 1):
        for k in range(3):
            p.line(sd * 3.0, -0.6 + k * 1.8, sd * 5.0, -1.2 + k * 2.4, BLACK, 0.25)
        p.path([(sd * 0.8, -4.2), (sd * 1.6, -6.0), (sd * 2.6, -6.4)], BLACK, 0.2)
    p.ell(0, -3.4, 2.2, 1.4, BLACK)
    p.dot(-1.0, -3.8, WHITE); p.dot(1.0, -3.8, WHITE)
    p.ell(0, 1.0, 4.2, 4.4, col)
    p.line(0, -2.6, 0, 5.4, BLACK, 0.25)
    for _ in range(rng.randint(2, 4)):
        x, y = rng.uniform(1.0, 2.8), rng.uniform(-1.4, 3.6)
        p.circ(x, y, 0.75, BLACK); p.circ(-x, y, 0.75, BLACK)


def spider(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s, flip=False)
    col = pk(rng, [BLACK, BLACK, BROWN, GRAY])
    if rng.rand() < 0.5:
        p.line(0, -9.0, 0, -1.0, WHITE if col != WHITE else GRAY, 0.2)
    for sd in (-1, 1):
        for k in range(4):
            y = -0.6 + k * 0.8
            p.path([(0, y), (sd * (3.4 + k * 0.3), y - 2.4 + k * 0.6), (sd * (5.4 + k * 0.2), y + 1.4 + k * 0.8)], col,
                   0.3)
    p.circ(0, 1.6, 2.4, col)
    p.circ(0, -1.4, 1.5, col)
    e = pk(rng, [RED, WHITE, YELLOW])
    p.dot(-0.6, -1.6, e); p.dot(0.6, -1.6, e)


def snail(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s)
    body, shell = pk(rng, [(GRAY, BROWN), (YELLOW, BROWN), (GRAY, YELLOW), (YELLOW, RED), (GREEN, BROWN)])
    p.poly([(-6.6, 5.6), (-5.2, 2.0), (-3.6, 4.0), (5.0, 4.4), (6.4, 5.8)], body)
    for sd in (-0.8, 0.8):
        p.line(-5.2 + sd * 0.5, 2.4, -6.0 + sd * 1.0, -0.6, body, 0.2)
        p.dot(-6.0 + sd * 1.0, -0.8, BLACK)
    p.circ(1.0, 0.6, 3.8, shell)
    sp = [(1.0 + r * math.cos(a), 0.6 + r * math.sin(a)) for a, r in zip(np.linspace(0, 4 * math.pi, 26), np.linspace(0.2, 3.2, 26))]
    p.path(sp, alt(shell, YELLOW, BROWN), 0.25)


def ant(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s)
    col = pk(rng, [BLACK, BLACK, RED, BROWN])
    for k, x in enumerate((-1.6, 0.0, 1.4)):
        for sd in (-1, 1):
            p.path([(x, 2.4), (x + (k - 1) * 1.4, 2.4 + sd * 0.2 - 1.4), (x + (k - 1) * 2.4, 5.6)], col, 0.25)
    p.ell(-4.0, 1.6, 1.4, 1.2, col)
    p.ell(-0.6, 2.2, 1.4, 0.9, col)
    p.ell(3.4, 2.2, 2.4, 1.7, col)
    p.path([(-4.6, 0.6), (-5.6, -1.4), (-6.8, -1.8)], col, 0.2)
    p.dot(-4.6, 1.2, WHITE)
    if rng.rand() < 0.3:
        p.circ(-6.6, 3.4, 1.0, pk(rng, [GREEN, YELLOW, WHITE]))


def dragonfly(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s, tilt=0.3)
    col = pk(rng, [BLUE, GREEN, RED])
    for y in (-1.6, 1.6):
        p.ell(-0.4 + 0.5 * y, y * 1.6, 0.8, 2.6, WHITE, rot=0.3 * y)
        p.ell(1.6 + 0.5 * y, y * 1.4, 0.7, 2.2, WHITE, rot=-0.3 * y)
    p.line(-2.6, 0, 8.6, 0, col, 0.4)
    p.ell(-1.0, 0, 2.0, 0.8, col)
    p.circ(-3.4, 0, 1.0, col)
    p.dot(-3.8, -0.4, BLACK); p.dot(-3.8, 0.4, BLACK)


def caterpillar(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s)
    col, alt_c = pk(rng, [(GREEN, YELLOW), (GREEN, GREEN), (YELLOW, BLACK), (GREEN, BLUE)])
    if rng.rand() < 0.6:
        p.ell(0, 6.8, 9.0, 1.4, GREEN if col != GREEN else BROWN)
    ph = rng.uniform(0, 6)
    for k in range(7):
        x = 6.0 - k * 1.7
        y = 4.0 - max(0, math.sin(k * 0.9 + ph)) * 1.6
        p.circ(x, y, 1.4, col if k % 2 == 0 else alt_c)
    p.circ(-6.4, 2.6, 1.6, RED if col != RED else YELLOW)
    p.dot(-7.0, 2.2, BLACK)
    p.line(-6.4, 1.0, -7.2, -0.4, BLACK, 0.2)


def worm(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s)
    col = pk(rng, [RED, RED, BROWN])
    if rng.rand() < 0.6:
        p.rect(-10, 5.0, 10, 8, BROWN if col != BROWN else BLACK)
    ph = rng.uniform(0, 6)
    pts = [(x, 3.8 - 1.4 * max(0, math.sin(x * 0.6 + ph))) for x in np.linspace(6.0, -5.0, 16)]
    p.path(pts, col, 0.8)
    hx, hy = pts[-1]
    p.dot(hx + 0.2, hy - 0.4, BLACK)


SUBJECTS = {
    'whale': (whale, ('water', 'under'), 'WHALE ORCA'),
    'shark': (shark, ('under', 'water'), 'SHARK JAWS'),
    'dolphin': (dolphin, ('water', 'sea', 'under'), 'DOLPHIN PORPOISE'),
    'octopus': (octopus, ('under',), 'OCTOPUS SQUID KRAKEN'),
    'crab': (crab, ('beach', 'under'), 'CRAB LOBSTER'),
    'jellyfish': (jellyfish, ('under',), 'JELLYFISH JELLY'),
    'starfish': (starfish, ('beach', 'under'), 'STARFISH SEASTAR'),
    'seahorse': (seahorse, ('under',), 'SEAHORSE'),
    'shrimp': (shrimp, ('under', 'plain'), 'SHRIMP PRAWN'),
    'butterfly': (butterfly, ('sky', 'field', 'grass', 'plain'), 'BUTTERFLY'),
    'bee': (bee, ('sky', 'field', 'plain'), 'BEE BEES BUMBLEBEE WASP HONEY'),
    'ladybug': (ladybug, ('leaf', 'grass'), 'LADYBUG LADYBIRD BEETLE'),
    'spider': (spider, ('web', 'plain'), 'SPIDER TARANTULA'),
    'snail': (snail, ('grass', 'leaf', 'plain'), 'SNAIL SLUG'),
    'ant': (ant, ('plain', 'grass', 'yellow'), 'ANT ANTS'),
    'dragonfly': (dragonfly, ('sky', 'pond'), 'DRAGONFLY'),
    'caterpillar': (caterpillar, ('sky', 'plain', 'leaf'), 'CATERPILLAR'),
    'worm': (worm, ('sky', 'grass'), 'WORM EARTHWORM'),
}
