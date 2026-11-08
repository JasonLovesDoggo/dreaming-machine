"""Birds, reptiles and amphibians. Same local frame as mammals (head towards -x, feet ~y=6)."""
import math
import numpy as np

from .draw import WHITE, BLACK, RED, BLUE, GREEN, YELLOW, BROWN, GRAY, pen, pk, alt


def legs2(p, xs, y0, y1, c=YELLOW, w=0.3):
    for x in xs:
        p.line(x, y0, x, y1, c, w)
        p.line(x, y1, x - 0.9, y1, c, w)


def owl(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s)
    col = pk(rng, [BROWN, BROWN, GRAY, WHITE])
    if rng.rand() < 0.6:
        p.line(-9, 6.4, 9, 6.4, BROWN if col != BROWN else BLACK, 0.7)
    p.ell(0, 1.6, 3.8, 4.6, col)
    p.ell(0, 2.6, 2.4, 3.2, WHITE if col != WHITE else YELLOW)
    p.poly([(-3.4, -2.4), (-3.0, -5.4), (-1.4, -3.0)], col)
    p.poly([(3.4, -2.4), (3.0, -5.4), (1.4, -3.0)], col)
    for sd in (-1, 1):
        p.circ(sd * 1.5, -1.2, 1.4, WHITE if col != WHITE else GRAY)
        p.circ(sd * 1.5, -1.2, 0.7, YELLOW if rng.rand() < 0.7 else BLACK)
        p.dot(sd * 1.5, -1.2, BLACK)
        p.ell(sd * 3.4, 2.4, 1.0, 3.0, col, rot=-sd * 0.2)
    p.poly([(-0.5, 0.0), (0.5, 0.0), (0, 1.2)], YELLOW)
    legs2(p, (-1.0, 1.4), 5.8, 6.2)


def penguin(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s)
    p.ell(0, 1.4, 3.2, 5.0, BLACK)
    p.ell(-0.2, 2.2, 2.2, 3.8, WHITE)
    p.circ(0, -3.4, 2.2, BLACK)
    p.ell(-0.6, -3.0, 1.0, 1.1, WHITE)
    p.dot(-0.8, -3.6, BLACK)
    p.poly([(-1.6, -2.6), (-3.8, -2.0), (-1.6, -1.8)], YELLOW)
    flap = rng.uniform(-0.5, 0.6)
    p.ell(2.8, 1.4, 0.8, 3.0, BLACK, rot=flap)
    p.ell(-1.2, 6.4, 1.2, 0.5, YELLOW); p.ell(1.2, 6.4, 1.2, 0.5, YELLOW)
    if rng.rand() < 0.3:
        p.rect(-2.2, -1.4, 2.2, -0.8, pk(rng, [RED, BLUE, GREEN]))


def duck(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s)
    col, head = pk(rng, [(YELLOW, YELLOW), (YELLOW, YELLOW), (WHITE, WHITE), (BROWN, GREEN), (GRAY, GREEN)])
    swim = rng.rand() < 0.5
    if not swim:
        legs2(p, (0.0, 1.6), 4.0, 6.2, RED if col == WHITE else YELLOW, 0.35)
    p.ell(1.2, 2.6, 4.2, 2.2, col)
    p.poly([(4.4, 1.2), (6.4, 0.4), (5.4, 2.6)], col)
    p.ell(1.8, 2.2, 2.2, 1.1, alt(col, BROWN, WHITE) if col != YELLOW else YELLOW, rot=-0.15)
    p.line(-2.0, 1.6, -2.6, -0.8, col, 0.8)
    p.circ(-2.8, -1.8, 1.8, head)
    p.ell(-5.0, -1.2, 1.4, 0.55, RED if rng.rand() < 0.3 else YELLOW if head != YELLOW else RED)
    p.dot(-3.4, -2.4, BLACK)
    if swim:
        p.rect(-8, 4.2, 8, 7, BLUE)


def chicken(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s)
    col = pk(rng, [WHITE, WHITE, BROWN, RED, YELLOW, BLACK])
    if col == YELLOW:
        p.ell(0, 2.6, 2.8, 2.6, col)
        p.circ(-1.4, -0.8, 1.7, col)
        p.poly([(-3.0, -0.8), (-4.2, -0.4), (-3.0, 0.0)], RED)
        p.dot(-1.8, -1.2, BLACK)
        legs2(p, (-0.4, 1.0), 5.0, 6.2, RED)
        return
    legs2(p, (-0.4, 1.4), 4.0, 6.4)
    for k in range(3):
        p.ell(3.6, -0.6 + k * 0.8, 1.0, 2.2, alt(col, BLACK, GREEN) if k == 1 else col, rot=0.5 + k * 0.3)
    p.ell(0.6, 2.0, 3.6, 2.6, col)
    p.ell(1.0, 1.8, 2.0, 1.2, alt(col, BROWN, GRAY), rot=-0.2)
    p.circ(-2.4, -1.6, 1.6, col)
    for k in range(3):
        p.circ(-3.2 + k * 0.8, -3.2 - (k == 1) * 0.3, 0.55, RED)
    p.poly([(-3.8, -1.8), (-5.0, -1.2), (-3.8, -0.8)], YELLOW)
    p.ell(-3.6, 0.0, 0.4, 0.6, RED)
    p.dot(-2.8, -2.0, BLACK if col != BLACK else WHITE)


def parrot(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s)
    col, wing = pk(rng, [(RED, BLUE), (RED, YELLOW), (GREEN, YELLOW), (BLUE, YELLOW), (GREEN, RED)])
    p.line(-9, 5.6, 9, 5.6, BROWN, 0.6)
    p.poly([(0.4, 2.8), (3.6, 9.0), (1.6, 3.0)], wing)
    p.ell(0, 1.0, 2.2, 3.6, col, rot=0.25)
    p.ell(0.8, 1.4, 1.2, 2.6, wing, rot=0.3)
    p.circ(-0.8, -2.8, 1.9, col)
    p.circ(-1.4, -3.2, 0.6, WHITE)
    p.dot(-1.4, -3.2, BLACK)
    p.poly([(-2.4, -3.0), (-3.6, -2.2), (-2.6, -0.8), (-2.0, -1.6)], YELLOW if col != YELLOW else BLACK)
    legs2(p, (-0.6, 0.6), 4.4, 5.4, GRAY)


def flamingo(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s)
    col = RED
    p.line(0.6, 1.4, 0.6, 7.0, col, 0.25)
    if rng.rand() < 0.6:
        p.path([(0.8, 3.6), (2.2, 4.2), (0.8, 4.8)], col, 0.25)
    else:
        p.line(1.4, 1.4, 1.6, 7.0, col, 0.25)
    p.ell(1.6, 0.0, 3.0, 1.6, col, rot=-0.15)
    p.poly([(4.0, -0.4), (5.8, -0.8), (4.4, 0.8)], col)
    p.path([(-0.8, -0.4), (-1.8, -2.4), (-0.6, -4.4), (-1.0, -6.4)], col, 0.4)
    p.circ(-1.6, -6.6, 0.9, col)
    p.path([(-2.4, -6.6), (-3.2, -6.0), (-3.0, -5.0)], BLACK, 0.3)
    p.dot(-1.8, -6.9, BLACK)


def eagle(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s, tilt=0.25)
    col = pk(rng, [BROWN, BROWN, BLACK, GRAY])
    up = rng.uniform(-1.6, 1.6)
    for sd in (-1, 1):
        p.poly([(sd * 1.4, -0.6), (sd * 5.0, -1.6 + up), (sd * 9.0, -2.6 + up * 1.6), (sd * 8.4, 0.0 + up),
                (sd * 4.6, 1.4), (sd * 1.4, 1.4)], col)
        for k in range(3):
            p.line(sd * (7.4 + k * 0.5), -1.2 + up * 1.3 + k * 0.6, sd * (9.4 + k * 0.4), -1.0 + up * 1.4 + k * 0.8,
                   col, 0.3)
    p.ell(0, 0.6, 1.6, 3.0, col)
    p.poly([(-1.2, 3.0), (1.2, 3.0), (1.8, 5.2), (-1.8, 5.2)], col if col != BROWN else WHITE)
    p.circ(0, -2.8, 1.4, WHITE if col != WHITE else GRAY)
    p.poly([(-0.4, -2.0), (0.6, -2.0), (0.0, -0.8)], YELLOW)
    p.dot(-0.6, -3.0, BLACK); p.dot(0.6, -3.0, BLACK)


def swan(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s)
    col = pk(rng, [WHITE, WHITE, WHITE, BLACK])
    p.rect(-9, 4.0, 9, 7, BLUE)
    p.ell(1.6, 2.6, 4.6, 2.0, col)
    p.poly([(4.6, 1.8), (6.8, -0.4), (5.6, 2.6)], col)
    p.ell(2.4, 1.6, 2.8, 1.4, col, rot=-0.3)
    p.path([(-2.4, 2.2), (-3.6, 0.0), (-2.6, -2.8), (-3.2, -4.4)], col, 0.5)
    p.circ(-3.4, -4.6, 0.9, col)
    p.poly([(-4.2, -4.8), (-5.6, -4.0), (-4.2, -4.0)], RED)
    p.dot(-3.6, -5.0, BLACK if col != BLACK else RED)


def peacock(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s, flip=False)
    fan = rng.uniform(0.75, 1.0)
    for a in np.linspace(math.pi * (1.05 + (1 - fan) * 0.6), math.pi * (1.95 - (1 - fan) * 0.6), 9):
        x, y = 8.0 * math.cos(a), 3.6 + 9.0 * math.sin(a) * 0.95
        p.line(0, 3.6, x, y, GREEN, 0.9)
        p.circ(x, y, 1.0, BLUE)
        p.dot(x, y, YELLOW)
    p.ell(0, 3.2, 1.6, 2.8, BLUE)
    p.circ(0, -0.4, 1.0, BLUE)
    p.dot(-0.4, -0.6, WHITE)
    p.line(0, -1.2, 0, -2.4, BLUE, 0.2)
    p.dot(0, -2.6, BLUE)
    legs2(p, (-0.6, 0.6), 5.6, 6.6, GRAY)


def ostrich(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s)
    st = rng.uniform(0, 0.9)
    p.line(-0.4, 1.6, -0.4 - st, 7.0, RED, 0.35)
    p.line(1.0, 1.6, 1.0 + st, 7.0, RED, 0.35)
    p.ell(0.4, -0.2, 3.6, 2.4, BLACK)
    p.ell(3.4, -1.0, 1.4, 1.2, WHITE)
    p.line(-2.6, -0.6, -3.0, -6.8, RED, 0.45)
    p.circ(-3.0, -7.0, 1.0, GRAY)
    p.poly([(-3.8, -7.2), (-5.0, -6.8), (-3.8, -6.4)], YELLOW)
    p.dot(-3.2, -7.4, BLACK)


def frog(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s, flip=False)
    col = pk(rng, [GREEN, GREEN, GREEN, YELLOW, RED, BLUE])
    if rng.rand() < 0.5:
        p.ell(0, 5.4, 7.6, 1.6, GREEN if col != GREEN else YELLOW)
        p.clear(rng.uniform(-6, 6), 4.8, 0.9)
    for sd in (-1, 1):
        p.ell(sd * 3.6, 3.4, 2.4, 1.4, col, rot=sd * 0.5)
        p.ell(sd * 4.8, 4.6, 1.6, 0.5, col)
        p.ell(sd * 1.6, 4.4, 0.6, 1.0, col)
    p.ell(0, 2.0, 4.0, 2.8, col)
    for sd in (-1, 1):
        p.circ(sd * 2.0, -1.0, 1.5, col)
        p.circ(sd * 2.0, -1.2, 0.9, WHITE)
        p.dot(sd * 2.0, -1.2, BLACK)
    p.path([(-2.4, 1.6), (0, 2.4), (2.4, 1.6)], BLACK if col != BLACK else WHITE, 0.25)
    if col in (RED, BLUE, YELLOW):
        for _ in range(4):
            p.dot(rng.uniform(-3, 3), rng.uniform(2.4, 4), BLACK)


def snake(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s)
    col, band = pk(rng, [(GREEN, YELLOW), (GREEN, BLACK), (YELLOW, BLACK), (RED, BLACK), (BROWN, YELLOW), (GRAY, BLACK)])
    ph, amp = rng.uniform(0, math.pi), rng.uniform(1.4, 2.6)
    pts = [(x, 3.0 + amp * math.sin(x * 0.75 + ph)) for x in np.linspace(9.0, -5.0, 24)]
    p.path(pts, col, 0.95)
    for x, y in pts[2:-3:3]:
        p.dot(x, y, band)
    hx, hy = pts[-1]
    p.ell(hx - 1.2, hy - 0.4, 1.8, 1.2, col)
    p.dot(hx - 1.6, hy - 0.9, BLACK)
    p.path([(hx - 2.8, hy - 0.2), (hx - 4.0, hy - 0.2), (hx - 4.6, hy - 0.8)], RED, 0.25)
    p.line(hx - 4.0, hy - 0.2, hx - 4.6, hy + 0.4, RED, 0.25)


def crocodile(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s)
    col = pk(rng, [GREEN, GREEN, BROWN])
    p.poly([(3.6, 2.4), (10.0, 4.4), (3.6, 4.6)], col)
    for lx in (-2.6, 2.4):
        p.line(lx, 4.0, lx - 0.6, 6.0, col, 0.55)
    p.ell(0.4, 3.4, 4.4, 1.5, col)
    p.poly([(-3.4, 2.4), (-10.0, 2.6), (-10.0, 3.6), (-3.4, 4.6)], col)
    open_ = rng.rand() < 0.5
    if open_:
        p.poly([(-4.0, 3.2), (-10.2, 1.6), (-10.2, 2.4)], RED)
        for x in np.arange(-9.4, -4.6, 1.2):
            p.dot(x, 2.4 - (x + 9.4) * 0.06, WHITE)
    for x in np.arange(-3.0, 8.0, 1.4):
        p.dot(x, 1.8 + max(0, x - 3.6) * 0.3, BLACK if col != BLACK else GREEN)
    p.circ(-4.2, 1.8, 0.7, col)
    p.dot(-4.2, 1.6, YELLOW)


def turtle(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s)
    skin, shell = pk(rng, [(GREEN, BROWN), (GREEN, GREEN), (YELLOW, GREEN), (GREEN, YELLOW)])
    for lx in (-2.6, 2.6):
        p.ell(lx, 4.6, 0.9, 1.5, skin)
    p.ell(-5.2, 2.6, 1.6, 1.2, skin)
    p.dot(-5.8, 2.2, BLACK)
    p.poly([(4.6, 3.4), (6.0, 4.2), (4.4, 4.2)], skin)
    p.ell(0, 2.4, 4.6, 3.0, shell)
    p.clipped()
    p.rect(-6, 3.6, 6, 6, skin if shell != skin else YELLOW)
    hex_c = alt(shell, BROWN, YELLOW, GREEN)
    for hx, hy in ((-2.0, 1.4), (0.4, 0.2), (2.6, 1.4), (0.2, 2.4)):
        p.circ(hx, hy, 0.8, hex_c)
    p.unclip()


def lizard(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s, tilt=0.3)
    col = pk(rng, [GREEN, GREEN, YELLOW, BROWN, BLUE])
    p.path([(3.4, 3.0), (6.4, 3.6), (8.6, 2.0), (9.4, 0.2)], col, 0.45)
    for lx, d in ((-1.6, -1), (2.2, 1)):
        for sd in (-1, 1):
            p.path([(lx, 3.0), (lx + d * 0.8, 3.0 + sd * 2.0), (lx + d * 1.6, 3.0 + sd * 2.4)], col, 0.3)
    p.ell(0.4, 3.0, 3.2, 1.0, col)
    p.ell(-3.6, 3.0, 1.4, 0.9, col)
    p.dot(-3.8, 2.6, BLACK)
    for x in (-1.0, 0.6, 2.2):
        p.dot(x, 3.0, alt(col, YELLOW, BLACK))


def dinosaur(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s)
    col = pk(rng, [GREEN, GREEN, BROWN, GRAY, RED, BLUE])
    if rng.rand() < 0.5:
        p.path([(-2.0, 1.4), (-4.0, -3.0), (-5.0, -6.0)], col, 0.9)
        p.ell(-6.4, -6.6, 1.6, 1.0, col)
        p.dot(-6.8, -7.0, BLACK)
        p.poly([(2.0, 0.4), (9.4, 3.8), (3.0, 3.0)], col)
        p.ell(0.6, 1.8, 3.8, 2.6, col)
        for lx in (-1.6, -0.4, 1.8, 3.0):
            p.line(lx, 3.0, lx, 6.4, col, 0.6)
        p.clipped()
        for _ in range(4):
            p.circ(rng.uniform(-2, 4), rng.uniform(0, 2), 0.6, alt(col, YELLOW, GREEN))
        p.unclip()
    else:
        p.poly([(1.6, 0.4), (8.6, 5.4), (2.4, 3.4)], col)
        p.ell(0.6, 0.8, 2.6, 3.2, col, rot=-0.4)
        p.line(-0.2, 3.0, -0.6, 6.4, col, 0.7)
        p.line(1.6, 2.8, 2.2, 6.4, col, 0.7)
        p.ell(-2.4, -3.6, 2.4, 1.5, col)
        p.line(-4.4, -2.8, -1.2, -2.8, BLACK if col != BLACK else WHITE, 0.2)
        p.dot(-2.6, -4.2, BLACK if col != BLACK else WHITE)
        p.line(-1.6, 0.4, -2.6, 1.0, col, 0.25)
        for t in range(4):
            p.poly([(-0.6 + t * 1.4, -2.0 + t * 0.6), (0.6 + t * 1.4, -3.0 + t * 0.6), (0.8 + t * 1.4, -1.4 + t * 0.6)],
                   alt(col, YELLOW, RED, GREEN))


SUBJECTS = {
    'owl': (owl, ('night', 'night', 'sky'), 'OWL HOOT'),
    'penguin': (penguin, ('snow',), 'PENGUIN'),
    'duck': (duck, ('pond', 'grass'), 'DUCK DUCKLING QUACK GOOSE'),
    'chicken': (chicken, ('field', 'grass', 'plain'), 'CHICKEN ROOSTER CHICK'),
    'parrot': (parrot, ('grass', 'sky', 'plain', 'leaf'), 'PARROT MACAW'),
    'flamingo': (flamingo, ('pond', 'beach', 'sky'), 'FLAMINGO'),
    'eagle': (eagle, ('sky', 'plain'), 'EAGLE HAWK FALCON'),
    'swan': (swan, ('pond', 'sky'), 'SWAN'),
    'peacock': (peacock, ('grass', 'plain', 'yellow'), 'PEACOCK'),
    'ostrich': (ostrich, ('savanna', 'desert'), 'OSTRICH EMU'),
    'frog': (frog, ('pond', 'grass', 'leaf'), 'FROG TOAD RIBBIT'),
    'snake': (snake, ('grass', 'desert', 'plain', 'sky'), 'SNAKE COBRA SERPENT PYTHON'),
    'crocodile': (crocodile, ('water', 'pond', 'grass'), 'CROCODILE CROC ALLIGATOR GATOR'),
    'turtle': (turtle, ('beach', 'grass', 'pond'), 'TURTLE TORTOISE'),
    'lizard': (lizard, ('desert', 'grass', 'plain'), 'LIZARD GECKO IGUANA CHAMELEON'),
    'dinosaur': (dinosaur, ('sky', 'grass', 'desert'), 'DINOSAUR DINO TREX RAPTOR'),
}
