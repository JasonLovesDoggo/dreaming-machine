"""Mammals. Local units: ~18 wide x 14 tall at s=1, head towards -x (mirrored per seed), feet at y~6."""
import math
import numpy as np

from .draw import WHITE, BLACK, RED, BLUE, GREEN, YELLOW, BROWN, GRAY, pen, pk, alt


def quad(p, col, bx=1.0, by=1.5, rx=5.0, ry=2.6, legs=(-2.8, -1.2, 2.8, 4.4), ly1=6.0, lw=0.6, legcol=None, hoof=None,
         gait=0.0):
    for i, lx in enumerate(legs):
        o = gait if i % 3 == 0 else -gait
        p.line(lx, by + 0.5, lx + o, ly1, legcol if legcol is not None else col, lw)
        if hoof is not None:
            p.rect(lx + o - lw, ly1 - 0.3, lx + o + lw, ly1 + 0.3, hoof)
    p.ell(bx, by, rx, ry, col)


def eye(p, x, y, c=BLACK):
    p.dot(x, y, c)


def cow(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s)
    col, spot = pk(rng, [(WHITE, BLACK), (WHITE, BLACK), (WHITE, BROWN), (BROWN, WHITE), (BLACK, WHITE)])
    p.path([(5.8, 0.5), (6.8, 3.0), (6.8, 4.5)], col, 0.3)
    p.circ(6.8, 4.8, 0.6, spot)
    quad(p, col, rx=5.6, ry=2.9, hoof=BLACK)
    p.clipped()
    for _ in range(rng.randint(2, 5)):
        p.circ(rng.uniform(-3, 5), rng.uniform(-0.5, 3), rng.uniform(1.0, 1.8), spot)
    p.unclip()
    if col == WHITE and rng.rand() < 0.6:
        p.ell(2.0, 4.2, 1.0, 0.7, RED)
    p.ell(-5.8, -1.0, 2.0, 1.9, col)
    p.ell(-6.6, 0.4, 1.6, 1.1, GRAY if col != GRAY else WHITE)
    for sd in (-1, 1):
        p.line(-5.8 + sd * 1.2, -2.6, -5.8 + sd * 2.0, -3.6, YELLOW if col != YELLOW else WHITE, 0.35)
    p.ell(-3.9, -1.9, 0.9, 0.5, spot)
    eye(p, -6.4, -1.4, BLACK if col != BLACK else WHITE)


def pig(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s)
    col = pk(rng, [RED, RED, RED, BROWN])
    quad(p, col, rx=5.2, ry=3.2, legs=(-2.8, -1.0, 2.8, 4.4), ly1=5.8, lw=0.7)
    p.circ(-4.6, 0.0, 2.6, col)
    p.ell(-7.0, 0.6, 0.9, 1.2, col)
    p.dot(-7.3, 0.2, BLACK); p.dot(-7.3, 1.1, BLACK)
    p.poly([(-5.4, -2.0), (-4.8, -4.0), (-3.6, -2.3)], col)
    eye(p, -5.4, -0.8)
    p.path([(6.0, 0.5), (7.2, -0.3), (7.4, 0.8), (6.6, 0.8), (7.2, -1.2)], col, 0.3)
    if col == BROWN:
        p.line(-6.4, 1.3, -6.8, 2.4, WHITE, 0.3)


def sheep(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s)
    wool, face = pk(rng, [(WHITE, BLACK), (WHITE, BLACK), (WHITE, GRAY), (BLACK, BLACK), (GRAY, BLACK)])
    for lx in (-2.6, -1.0, 2.6, 4.2):
        p.line(lx, 2.0, lx, 6.0, face, 0.45)
    for _ in range(14):
        a = rng.uniform(0, 2 * math.pi)
        p.circ(1.0 + 4.6 * math.cos(a) * rng.uniform(0.4, 1), 1.0 + 2.4 * math.sin(a) * rng.uniform(0.4, 1), 1.6, wool)
    p.ell(1.0, 1.0, 4.8, 2.8, wool)
    p.ell(-5.2, -0.6, 1.4, 2.0, face, rot=0.5)
    p.ell(-4.0, -1.8, 0.9, 0.4, face, rot=-0.3)
    p.circ(-4.6, -2.6, 1.2, wool)
    eye(p, -5.4, -1.0, WHITE)


def horse(cv, rng, cx, cy, s, kind='horse'):
    p = pen(cv, rng, cx, cy, s)
    if kind == 'zebra':
        col, mane = WHITE, BLACK
    elif kind == 'unicorn':
        col, mane = WHITE, pk(rng, [RED, BLUE, YELLOW])
    elif kind == 'donkey':
        col, mane = GRAY, BLACK
    else:
        col = pk(rng, [BROWN, BROWN, BLACK, WHITE, GRAY, YELLOW])
        mane = alt(col, *pk(rng, [(BLACK, BROWN), (WHITE, BLACK), (YELLOW, BROWN)]))
    p.path([(5.8, 0.0), (7.2, 2.0), (7.4, 4.5)], mane, 0.6 if kind != 'donkey' else 0.3)
    quad(p, col, rx=5.2, ry=2.4, legs=(-3.0, -1.4, 3.0, 4.6), ly1=6.4, lw=0.55, hoof=BLACK, gait=rng.uniform(0, 0.8))
    p.poly([(-4.6, 0.2), (-2.2, 0.0), (-4.4, -4.6), (-6.4, -4.0)], col)
    p.ell(-6.8, -3.6, 1.2, 2.4, col, rot=1.0)
    ears = 1.6 if kind == 'donkey' else 0.9
    p.ell(-5.4, -5.0 - ears * 0.5, 0.45, ears, col, rot=-0.2)
    if kind == 'zebra':
        p.clipped()
        for x in np.arange(-7, 7, 1.3):
            p.line(x, -6, x + 0.6, 7, BLACK, 0.35)
        p.unclip()
    if kind != 'donkey':
        p.path([(-5.3, -4.8), (-4.2, -3.0), (-3.2, -1.0)], mane, 0.45)
    eye(p, -6.4, -4.0, BLACK if col != BLACK else WHITE)
    p.dot(-8.2, -2.5, BLACK if col != BLACK else GRAY)
    if kind == 'unicorn':
        p.poly([(-6.2, -5.0), (-7.6, -8.6), (-5.4, -5.6)], YELLOW)


def zebra(cv, rng, cx, cy, s):
    horse(cv, rng, cx, cy, s, 'zebra')


def unicorn(cv, rng, cx, cy, s):
    horse(cv, rng, cx, cy, s, 'unicorn')


def donkey(cv, rng, cx, cy, s):
    horse(cv, rng, cx, cy, s, 'donkey')


def rabbit(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s)
    col = pk(rng, [WHITE, WHITE, GRAY, BROWN, YELLOW, BLACK])
    p.circ(4.6, 3.6, 1.2, WHITE if col != WHITE else GRAY)
    p.ell(1.2, 3.0, 3.8, 3.2, col)
    p.ell(-0.6, 6.0, 2.2, 0.7, col)
    tilt = rng.uniform(-0.3, 0.3)
    for ex in (-2.6, -1.0):
        p.ell(ex, -4.4, 0.8, 2.9, col, rot=tilt + (ex + 1.8) * 0.15)
        p.ell(ex, -4.2, 0.3, 2.0, RED if col != RED else WHITE, rot=tilt + (ex + 1.8) * 0.15)
    p.circ(-2.0, -0.4, 2.3, col)
    eye(p, -2.8, -0.9, BLACK if col != BLACK else RED)
    p.dot(-4.0, 0.0, RED)


def mouse(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s)
    col = pk(rng, [GRAY, GRAY, BROWN, WHITE])
    p.path([(3.6, 3.6), (6.0, 4.6), (7.4, 3.2), (8.6, 4.2)], RED if rng.rand() < 0.5 else col, 0.3)
    p.ell(1.0, 2.8, 3.6, 2.4, col)
    p.poly([(-1.2, 0.8), (-5.6, 3.2), (-1.0, 4.8)], col)
    p.circ(-1.2, 0.0, 1.6, col)
    p.circ(-1.2, 0.0, 0.8, RED)
    p.dot(-5.6, 3.2, BLACK)
    eye(p, -3.0, 2.2)
    for lx in (-1.0, 2.6):
        p.line(lx, 4.8, lx - 0.4, 5.6, col, 0.3)
    if rng.rand() < 0.4:
        p.poly([(-9.0, 5.8), (-6.4, 5.8), (-6.4, 3.8)], YELLOW)


def bear(cv, rng, cx, cy, s, col=None):
    p = pen(cv, rng, cx, cy, s)
    col = col if col is not None else pk(rng, [BROWN, BROWN, BLACK])
    snout = YELLOW if col == BROWN else (GRAY if col == BLACK else GRAY)
    quad(p, col, rx=5.6, ry=3.4, by=1.4, legs=(-3.0, -1.0, 2.6, 4.6), ly1=6.2, lw=0.95)
    p.circ(-5.4, -1.4, 2.6, col)
    for ex in (-6.8, -4.2):
        p.circ(ex, -3.6, 0.9, col)
    p.ell(-7.4, -0.6, 1.4, 1.0, snout)
    p.dot(-8.4, -0.9, BLACK if snout != BLACK else WHITE)
    eye(p, -6.2, -2.0, BLACK if col != BLACK else WHITE)


def polarbear(cv, rng, cx, cy, s):
    bear(cv, rng, cx, cy, s, WHITE)


def panda(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s)
    if rng.rand() < 0.6:
        for bx in (rng.uniform(-9, -6), rng.uniform(6, 9)):
            p.line(bx, -9, bx, 8, GREEN, 0.6)
            p.ell(bx + 1.2, rng.uniform(-6, 0), 1.4, 0.5, GREEN, rot=-0.4)
    p.ell(0, 3.2, 4.2, 3.6, WHITE)
    p.ell(-3.4, 5.4, 1.6, 1.2, BLACK); p.ell(3.4, 5.4, 1.6, 1.2, BLACK)
    p.ell(-3.4, 1.8, 1.0, 2.0, BLACK, rot=-0.5); p.ell(3.4, 1.8, 1.0, 2.0, BLACK, rot=0.5)
    p.rect(-4.0, 0.0, 4.0, 1.0, BLACK)
    for ex in (-2.6, 2.6):
        p.circ(ex, -5.2, 1.1, BLACK)
    p.circ(0, -2.6, 3.2, WHITE)
    for ex in (-1.3, 1.3):
        p.ell(ex, -2.8, 0.8, 1.0, BLACK, rot=0.5 * ex)
        p.dot(ex, -2.9, WHITE)
    p.ell(0, -1.2, 0.6, 0.4, BLACK)


def lion(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s)
    col, mane = pk(rng, [(YELLOW, BROWN), (YELLOW, BROWN), (YELLOW, RED), (BROWN, BLACK)])
    p.path([(5.8, 0.5), (7.4, 3.0), (8.2, 4.0)], col, 0.35)
    p.circ(8.2, 4.0, 0.8, mane)
    quad(p, col, rx=5.0, ry=2.4, legs=(-2.4, -0.8, 2.8, 4.4), ly1=6.0, lw=0.65)
    female = rng.rand() < 0.25
    if not female:
        for a in np.linspace(0, 2 * math.pi, 12, endpoint=False):
            p.circ(-4.6 + 2.6 * math.cos(a), -1.4 + 2.6 * math.sin(a), 1.3, mane)
    p.circ(-4.6, -1.4, 2.2, col)
    p.circ(-6.0, -3.2 if female else -3.6, 0.6, col)
    p.ell(-5.8, -0.2, 1.0, 0.8, WHITE if col != WHITE else YELLOW)
    p.dot(-6.4, -0.8, BLACK)
    eye(p, -5.6, -2.0)


def tiger(cv, rng, cx, cy, s, spots=False):
    p = pen(cv, rng, cx, cy, s)
    col = pk(rng, [YELLOW, YELLOW, WHITE]) if not spots else YELLOW
    mark = BLACK
    p.path([(5.6, 0.8), (7.6, -0.4), (8.8, -2.4)], col, 0.45)
    quad(p, col, rx=5.2, ry=2.3, legs=(-2.6, -1.0, 2.8, 4.4), ly1=6.0, lw=0.6, gait=rng.uniform(0, 0.7))
    p.circ(-4.6, -1.0, 2.3, col)
    p.circ(-6.0, -3.0, 0.7, col); p.circ(-3.4, -3.1, 0.7, col)
    p.clipped()
    if spots:
        for _ in range(rng.randint(14, 22)):
            p.circ(rng.uniform(-6.5, 9), rng.uniform(-3, 6), 0.5, mark)
    else:
        for x in np.arange(-6.5, 9, 1.5):
            p.line(x, -3.5, x + rng.uniform(-0.8, 0.8), 6.5, mark, 0.35)
    p.unclip()
    p.ell(-5.0, 0.2, 1.1, 0.8, WHITE if col != WHITE else YELLOW)
    p.dot(-5.6, -0.3, RED)
    eye(p, -5.6, -1.6, GREEN)
    eye(p, -3.8, -1.6, GREEN)


def leopard(cv, rng, cx, cy, s):
    tiger(cv, rng, cx, cy, s, spots=True)


def elephant(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s)
    col = pk(rng, [GRAY, GRAY, GRAY, WHITE])
    p.line(6.0, 0.0, 7.2, 3.0, col, 0.3)
    quad(p, col, rx=5.6, ry=3.6, by=0.6, legs=(-2.6, -0.8, 2.6, 4.4), ly1=6.4, lw=1.0)
    p.circ(-4.6, -1.6, 3.0, col)
    p.path([(-6.6, -1.0), (-7.6, 1.6), (-7.6, 4.0), (-6.8, 5.4)], col, 0.85)
    p.ell(-2.8, -0.8, 2.0, 2.7, BLACK)
    p.ell(-2.8, -0.8, 1.6, 2.3, col)
    p.line(-6.4, 0.8, -8.2, 1.4, WHITE if col != WHITE else YELLOW, 0.35)
    eye(p, -5.6, -2.4)


def giraffe(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s)
    col, spot = YELLOW, pk(rng, [BROWN, BROWN, BLACK])
    p.line(4.4, 0.4, 5.4, 3.4, col, 0.3)
    quad(p, col, bx=1.0, by=1.2, rx=3.8, ry=2.0, legs=(-1.8, -0.6, 2.4, 3.6), ly1=6.6, lw=0.45)
    p.line(-1.6, 0.4, -4.4, -6.0, col, 1.0)
    p.ell(-5.4, -6.8, 1.6, 0.9, col, rot=0.4)
    p.line(-4.4, -7.4, -4.2, -8.8, BROWN, 0.3)
    p.line(-3.6, -7.0, -3.2, -8.4, BROWN, 0.3)
    p.clipped()
    for _ in range(rng.randint(12, 18)):
        p.circ(rng.uniform(-5, 5), rng.uniform(-6, 3), 0.6, spot)
    p.unclip()
    eye(p, -5.0, -7.2)


def monkey(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s)
    col, skin = pk(rng, [(BROWN, YELLOW), (BROWN, YELLOW), (BLACK, GRAY), (GRAY, WHITE)])
    p.path([(2.8, 5.0), (5.4, 5.0), (6.2, 2.0), (5.0, 0.4), (4.2, 1.6)], col, 0.4)
    p.ell(0, 3.0, 3.0, 3.2, col)
    p.ell(0, 3.6, 1.8, 2.2, skin)
    p.line(-2.4, 1.6, -4.2, 4.0, col, 0.55); p.line(2.4, 1.6, 4.0, 4.2, col, 0.55)
    p.ell(-1.6, 6.0, 1.4, 0.6, col); p.ell(1.6, 6.0, 1.4, 0.6, col)
    for sd in (-1, 1):
        p.circ(sd * 3.0, -2.6, 1.2, col)
        p.circ(sd * 3.0, -2.6, 0.6, skin)
    p.circ(0, -2.6, 2.8, col)
    p.ell(0, -2.0, 2.0, 1.8, skin)
    eye(p, -0.8, -2.8); eye(p, 0.8, -2.8)
    p.line(-0.6, -1.0, 0.6, -1.0, BROWN if skin != BROWN else BLACK, 0.3)
    if rng.rand() < 0.4:
        p.ell(-4.6, 3.0, 0.7, 1.8, YELLOW if skin != YELLOW else RED, rot=0.5)


def fox(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s)
    col = pk(rng, [RED, BROWN, BROWN, WHITE, GRAY])
    tip = WHITE if col != WHITE else GRAY
    p.ell(7.0, 0.4, 3.4, 1.5, col, rot=-0.5)
    p.ell(9.2, -1.0, 1.2, 1.0, tip)
    quad(p, col, rx=4.4, ry=2.0, legs=(-2.0, -0.6, 2.4, 3.8), ly1=6.0, lw=0.45, legcol=BLACK if col != BLACK else GRAY,
         gait=rng.uniform(0, 0.8))
    p.circ(-4.0, -1.2, 2.0, col)
    p.poly([(-4.8, -0.4), (-8.6, 0.6), (-4.4, 1.4)], col)
    p.dot(-8.4, 0.6, BLACK)
    for ex in (-5.0, -3.0):
        p.poly([(ex - 0.9, -2.2), (ex, -5.0), (ex + 0.9, -2.2)], col)
    p.ell(-4.6, 0.8, 1.2, 0.7, tip)
    eye(p, -4.8, -1.6)


def wolf(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s)
    col = pk(rng, [GRAY, GRAY, WHITE, BLACK, BROWN])
    p.path([(3.4, 5.2), (6.4, 5.6), (8.0, 4.4)], col, 0.8)
    p.ell(1.6, 3.2, 3.4, 2.8, col)
    p.poly([(-2.0, 2.0), (0.4, 0.4), (1.2, 3.0), (-1.2, 4.0)], col)
    p.line(-1.4, 3.0, -1.6, 6.2, col, 0.5); p.line(0.0, 3.2, -0.2, 6.2, col, 0.5)
    p.line(-0.4, 1.8, -2.0, -2.2, col, 1.2)
    p.circ(-2.4, -2.8, 1.6, col)
    p.poly([(-2.6, -3.6), (-5.4, -6.0), (-3.4, -2.2)], col)
    p.poly([(-1.6, -3.6), (-0.6, -5.6), (-0.6, -3.0)], col)
    eye(p, -2.6, -3.4, YELLOW)
    p.dot(-5.4, -6.0, BLACK if col != BLACK else GRAY)


def deer(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s)
    col = pk(rng, [BROWN, BROWN, YELLOW])
    quad(p, col, rx=4.4, ry=2.1, legs=(-2.2, -1.0, 2.6, 3.8), ly1=6.4, lw=0.4, gait=rng.uniform(0, 0.6))
    p.circ(5.2, 0.4, 0.7, WHITE)
    p.line(-2.6, 0.6, -4.0, -3.4, col, 0.8)
    p.ell(-4.8, -3.8, 1.5, 0.9, col, rot=0.3)
    p.ell(-3.4, -5.0, 0.4, 0.9, col, rot=-0.5)
    stag = rng.rand() < 0.6
    if stag:
        ac = BROWN if col != BROWN else YELLOW
        p.path([(-4.2, -4.6), (-4.6, -7.4), (-6.0, -8.6)], ac, 0.3)
        p.line(-4.5, -6.6, -3.2, -8.2, ac, 0.3)
        p.line(-4.6, -7.4, -5.6, -6.8, ac, 0.3)
    else:
        p.clipped()
        for _ in range(6):
            p.dot(rng.uniform(-1.5, 3.5), rng.uniform(0.0, 1.5), WHITE)
        p.unclip()
    eye(p, -5.0, -4.2)
    p.dot(-6.2, -3.6, BLACK)


def koala(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s)
    p.line(-9, 6.5, 9, 4.0, BROWN, 0.9)
    p.ell(rng.uniform(-8, -5), 4.6, 1.8, 0.8, GREEN)
    p.ell(0.6, 2.6, 3.0, 3.4, GRAY)
    p.ell(-1.6, 4.6, 1.2, 0.8, GRAY)
    for sd in (-1, 1):
        p.circ(sd * 3.2, -3.6, 1.8, GRAY)
        p.circ(sd * 3.2, -3.6, 1.0, WHITE)
    p.circ(0, -2.4, 3.0, GRAY)
    p.ell(0, -1.6, 0.9, 1.2, BLACK)
    eye(p, -1.4, -2.8); eye(p, 1.4, -2.8)


def kangaroo(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s)
    col = pk(rng, [BROWN, BROWN, YELLOW, GRAY])
    p.path([(2.2, 3.6), (5.4, 5.0), (8.4, 6.0)], col, 0.8)
    p.ell(0.8, 2.2, 2.6, 3.6, col, rot=-0.35)
    p.ell(-0.2, 6.0, 2.6, 0.7, col)
    p.ell(1.4, 4.4, 1.8, 1.6, col)
    p.line(-1.4, -0.4, -2.6, -2.4, col, 1.0)
    p.ell(-3.4, -3.4, 1.8, 1.1, col, rot=0.4)
    p.ell(-2.0, -5.2, 0.45, 1.3, col, rot=-0.3)
    p.line(-1.4, 1.0, -2.8, 2.4, col, 0.35)
    if rng.rand() < 0.4:
        p.circ(-1.2, 2.6, 0.9, col)
        p.dot(-1.5, 2.4, BLACK)
    eye(p, -3.4, -3.8)
    p.dot(-5.0, -3.2, BLACK)


def hippo(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s)
    col = pk(rng, [GRAY, GRAY, BROWN])
    quad(p, col, rx=5.6, ry=3.2, by=2.0, legs=(-2.6, -0.8, 2.6, 4.4), ly1=6.4, lw=0.95)
    p.ell(-5.4, 0.6, 2.4, 2.4, col)
    p.ell(-6.6, 1.8, 2.2, 1.8, col)
    p.circ(-4.6, -1.6, 0.7, col)
    p.dot(-7.8, 1.0, BLACK)
    p.line(-8.4, 2.8, -5.6, 3.0, BLACK if col != BLACK else WHITE, 0.25)
    eye(p, -5.6, -0.6)
    p.dot(-7.0, 3.6, WHITE)


def rhino(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s)
    col = pk(rng, [GRAY, GRAY, BROWN])
    quad(p, col, rx=5.4, ry=3.0, by=1.6, legs=(-2.6, -0.8, 2.6, 4.4), ly1=6.4, lw=0.9)
    p.ell(-5.6, 0.4, 2.8, 2.0, col, rot=0.3)
    horn = WHITE if col != WHITE else YELLOW
    p.poly([(-7.8, 0.0), (-8.8, -3.6), (-6.8, -0.6)], horn)
    p.poly([(-6.2, -1.0), (-6.6, -2.8), (-5.4, -1.4)], horn)
    p.ell(-4.4, -1.6, 0.5, 0.9, col)
    eye(p, -5.6, -0.4)


def camel(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s)
    col = pk(rng, [YELLOW, BROWN, BROWN])
    quad(p, col, rx=4.6, ry=2.0, by=1.6, legs=(-2.4, -1.2, 2.6, 3.8), ly1=6.6, lw=0.45)
    humps = 2 if rng.rand() < 0.5 else 1
    for hx in ((-0.2, 2.6) if humps == 2 else (1.2,)):
        p.ell(hx, -0.2, 1.8 if humps == 2 else 2.6, 1.9, col)
    p.path([(-3.6, 1.2), (-5.4, 0.6), (-5.6, -2.6)], col, 0.7)
    p.ell(-6.4, -3.0, 1.5, 0.8, col)
    p.line(4.8, 1.4, 5.4, 3.6, col, 0.3)
    eye(p, -6.2, -3.4)


def squirrel(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s)
    col = pk(rng, [BROWN, BROWN, RED, GRAY])
    p.ell(3.6, -0.4, 2.4, 4.8, col, rot=0.25)
    p.ell(4.6, -4.6, 2.2, 1.6, col, rot=-0.5)
    p.ell(0.0, 3.2, 2.4, 3.0, col)
    p.ell(-0.6, 3.6, 1.1, 1.8, WHITE if col != WHITE else YELLOW)
    p.ell(0.6, 6.0, 1.8, 0.6, col)
    p.circ(-1.0, -1.2, 1.9, col)
    p.poly([(-0.4, -2.4), (0.0, -4.4), (0.8, -2.2)], col)
    eye(p, -1.8, -1.6)
    p.circ(-2.4, 2.2, 0.9, BROWN if col != BROWN else YELLOW)


def hedgehog(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s)
    spike, face = pk(rng, [(BROWN, YELLOW), (BLACK, YELLOW), (BROWN, WHITE), (GRAY, YELLOW)])
    for a in np.linspace(math.pi, 2 * math.pi, 9):
        x, y = 1.0 + 4.8 * math.cos(a), 3.6 + 3.8 * math.sin(a)
        p.poly([(x - 1.0, y + 0.9), (1.0 + 6.4 * math.cos(a), 3.6 + 5.2 * math.sin(a)), (x + 1.0, y + 0.9)], spike)
    p.ell(1.0, 3.6, 4.8, 3.0, spike)
    p.poly([(-2.6, 1.2), (-7.2, 4.0), (-2.4, 5.8)], face)
    p.ell(-2.4, 3.8, 1.6, 2.2, face)
    p.dot(-7.2, 4.0, BLACK)
    eye(p, -3.8, 3.2)
    p.line(-1.6, 6.2, -1.6, 6.6, face, 0.4); p.line(2.8, 6.2, 2.8, 6.6, face, 0.4)


def bat(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s)
    col = pk(rng, [BLACK, BLACK, BROWN, GRAY])
    for sd in (-1, 1):
        p.poly([(0, -1.2), (sd * 4.0, -3.4), (sd * 8.6, -2.4), (sd * 7.6, 1.8), (sd * 6.4, 0.2), (sd * 5.0, 2.2),
                (sd * 3.6, 0.6), (sd * 2.2, 2.4), (0, 1.8)], col)
    p.ell(0, 0.4, 1.6, 2.6, col)
    p.circ(0, -2.0, 1.5, col)
    p.poly([(-1.4, -2.4), (-1.0, -4.4), (-0.2, -2.8)], col)
    p.poly([(1.4, -2.4), (1.0, -4.4), (0.2, -2.8)], col)
    e = RED if rng.rand() < 0.5 else YELLOW
    p.dot(-0.6, -2.2, e); p.dot(0.6, -2.2, e)


def raccoon(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s)
    col = GRAY
    p.ell(6.4, 1.2, 2.6, 1.2, col, rot=-0.3)
    p.clipped()
    for x in (5.2, 6.6, 8.0):
        p.line(x, -2, x + 0.4, 4, BLACK, 0.4)
    p.unclip()
    quad(p, col, rx=4.6, ry=2.4, by=2.0, legs=(-2.2, -0.8, 2.4, 3.8), ly1=6.0, lw=0.5, legcol=BLACK)
    p.circ(-4.4, -0.2, 2.2, col)
    p.poly([(-5.4, 0.6), (-8.2, 1.4), (-5.0, 1.8)], WHITE)
    p.ell(-4.6, -0.6, 2.0, 0.6, BLACK)
    p.dot(-5.4, -0.6, WHITE)
    p.dot(-8.2, 1.4, BLACK)
    p.poly([(-5.2, -1.8), (-4.8, -3.6), (-3.8, -2.0)], col)


def goat(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s)
    col = pk(rng, [WHITE, WHITE, BROWN, GRAY, BLACK])
    p.line(5.0, 0.0, 5.8, -1.4, col, 0.4)
    quad(p, col, rx=4.6, ry=2.3, legs=(-2.4, -1.0, 2.6, 3.8), ly1=6.0, lw=0.45, hoof=BLACK if col != BLACK else GRAY)
    p.line(-3.2, 0.6, -4.4, -2.2, col, 0.8)
    p.ell(-5.2, -2.4, 1.6, 0.9, col, rot=0.5)
    hc = GRAY if col != GRAY else BROWN
    p.path([(-4.4, -3.2), (-3.4, -5.2), (-2.2, -5.2)], hc, 0.35)
    p.line(-6.2, -1.6, -6.0, -0.2, col, 0.35)
    p.ell(-3.4, -3.0, 0.9, 0.35, col)
    eye(p, -5.0, -2.9, BLACK if col != BLACK else YELLOW)


def hamster(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s)
    col = pk(rng, [YELLOW, BROWN, WHITE, GRAY])
    p.ell(0, 2.4, 4.4, 3.8, col)
    p.ell(0, 3.6, 2.6, 2.4, WHITE if col != WHITE else YELLOW)
    for sd in (-1, 1):
        p.circ(sd * 2.6, -1.4, 1.0, col)
        p.circ(sd * 2.6, -1.4, 0.5, RED)
        p.ell(sd * 1.4, 5.8, 0.8, 0.5, RED)
    eye(p, -1.4, 0.4); eye(p, 1.4, 0.4)
    p.dot(0, 1.4, RED)
    if rng.rand() < 0.4:
        p.ell(0, 3.6, 0.8, 1.0, BROWN if col != BROWN else BLACK)


def seal(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s)
    col = pk(rng, [GRAY, GRAY, BLACK, BROWN, WHITE])
    if rng.rand() < 0.5:
        p.ell(1.0, 6.4, 7.0, 1.2, WHITE if col != WHITE else GRAY)
    p.poly([(4.0, 3.6), (7.6, 2.6), (7.4, 5.6)], col)
    p.ell(1.0, 3.4, 4.6, 2.4, col, rot=0.2)
    p.ell(-3.4, 0.6, 2.4, 2.6, col)
    p.ell(-1.6, 5.2, 1.6, 0.6, col, rot=0.4)
    eye(p, -4.2, -0.2, BLACK if col != BLACK else WHITE)
    p.dot(-5.6, 1.0, BLACK if col != BLACK else GRAY)
    if rng.rand() < 0.35:
        p.circ(-5.4, -3.2, 1.3, pk(rng, [RED, YELLOW, BLUE]))


SUBJECTS = {
    'cow': (cow, ('grass', 'field', 'sky'), 'COW COWS CATTLE MOO CALF'),
    'pig': (pig, ('grass', 'field', 'sky', 'plain'), 'PIG PIGGY PIGLET HOG OINK'),
    'sheep': (sheep, ('grass', 'field', 'sky'), 'SHEEP LAMB WOOL'),
    'horse': (horse, ('field', 'sky', 'grass'), 'HORSE PONY STALLION MARE'),
    'zebra': (zebra, ('savanna',), 'ZEBRA'), 'unicorn': (unicorn, ('sky', 'field', 'night'), 'UNICORN'),
    'donkey': (donkey, ('field', 'sky', 'desert'), 'DONKEY MULE'),
    'rabbit': (rabbit, ('grass', 'field', 'plain', 'sky'), 'RABBIT BUNNY HARE'),
    'mouse': (mouse, ('plain', 'room', 'gray'), 'MOUSE MICE'),
    'bear': (bear, ('sky', 'grass', 'field'), 'BEAR GRIZZLY'),
    'polarbear': (polarbear, ('snow',), 'POLAR POLARBEAR'),
    'panda': (panda, ('grass', 'plain', 'sky'), 'PANDA'),
    'lion': (lion, ('savanna',), 'LION LIONS ROAR'),
    'tiger': (tiger, ('grass', 'savanna'), 'TIGER'),
    'leopard': (leopard, ('savanna', 'grass'), 'LEOPARD CHEETAH JAGUAR'),
    'elephant': (elephant, ('savanna', 'sky'), 'ELEPHANT DUMBO'),
    'giraffe': (giraffe, ('savanna', 'sky'), 'GIRAFFE'),
    'monkey': (monkey, ('grass', 'sky', 'plain'), 'MONKEY CHIMP'),
    'fox': (fox, ('grass', 'snow', 'field'), 'FOX FOXY'),
    'wolf': (wolf, ('night', 'snow'), 'WOLF HOWL'),
    'deer': (deer, ('field', 'grass', 'snow'), 'DEER STAG FAWN BAMBI REINDEER'),
    'koala': (koala, ('sky', 'plain'), 'KOALA'),
    'kangaroo': (kangaroo, ('desert', 'savanna'), 'KANGAROO'),
    'hippo': (hippo, ('water', 'grass', 'savanna'), 'HIPPO HIPPOPOTAMUS'),
    'rhino': (rhino, ('savanna',), 'RHINO RHINOCEROS'),
    'camel': (camel, ('desert',), 'CAMEL'),
    'squirrel': (squirrel, ('grass', 'sky', 'field'), 'SQUIRREL CHIPMUNK'),
    'hedgehog': (hedgehog, ('grass', 'field'), 'HEDGEHOG PORCUPINE'),
    'bat': (bat, ('night',), 'BAT BATS VAMPIRE'),
    'raccoon': (raccoon, ('grass', 'night', 'field'), 'RACCOON'),
    'goat': (goat, ('field', 'grass', 'sky'), 'GOAT'),
    'hamster': (hamster, ('plain', 'room', 'yellow'), 'HAMSTER GERBIL'),
    'seal': (seal, ('snow', 'water', 'beach'), 'SEAL WALRUS SEALION'),
}
