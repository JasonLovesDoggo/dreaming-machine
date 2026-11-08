"""Sky, weather and landscape subjects."""
import math
import numpy as np

from .draw import WHITE, BLACK, RED, BLUE, GREEN, YELLOW, BROWN, GRAY, pen, pk, alt


def cloud_at(p, x, y, k, c):
    for ox, oy, r in ((-2.4, 0.6, 1.8), (0, -0.4, 2.4), (2.4, 0.6, 1.8), (1.0, 1.0, 1.8), (-1.0, 1.0, 1.8)):
        p.circ(x + ox * k, y + oy * k, r * k, c)


def cloud(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s, flip=False)
    col = pk(rng, [WHITE, WHITE, WHITE, GRAY])
    for _ in range(rng.randint(1, 3)):
        cloud_at(p, rng.uniform(-6, 6), rng.uniform(-4, 2), rng.uniform(0.7, 1.2), col)


def rain(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s, flip=False)
    col = pk(rng, [GRAY, GRAY, WHITE])
    for _ in range(rng.randint(10, 18)):
        x, y = rng.uniform(-8, 8), rng.uniform(1.6, 7.0)
        p.line(x, y, x - 0.5, y + 1.0, BLUE if col != BLUE else WHITE, 0.25)
    cloud_at(p, rng.uniform(-2, 2), -3.0, 1.3, col)


def lightning(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s)
    pts = [(0.6, -3.0), (-1.4, 1.8), (0.6, 1.6), (-1.6, 7.6), (3.0, 0.2), (1.0, 0.4), (2.4, -3.0)]
    p.poly(pts, YELLOW)
    cloud_at(p, 0.6, -4.6, 1.3, pk(rng, [GRAY, GRAY, WHITE]))


def rainbow(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s, flip=False)
    bands = [RED, YELLOW, GREEN, BLUE]
    r0 = rng.uniform(7.0, 9.0)
    for i, c in enumerate(bands):
        pts = [((r0 - i) * math.cos(a), 6.0 - (r0 - i) * math.sin(a)) for a in np.linspace(0, math.pi, 24)]
        p.path(pts, c, 0.55)
    if rng.rand() < 0.7:
        cloud_at(p, -r0 + 1.5, 5.2, 0.8, WHITE)
        cloud_at(p, r0 - 1.5, 5.2, 0.8, WHITE)


def snowflake(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s, tilt=0.5, flip=False)
    col = pk(rng, [WHITE, WHITE, BLUE])
    for k in range(6):
        a = k * math.pi / 3
        ex, ey = 6.0 * math.cos(a), 6.0 * math.sin(a)
        p.line(0, 0, ex, ey, col, 0.35)
        for t in (0.5, 0.75):
            bx, by = ex * t, ey * t
            for o in (-0.7, 0.7):
                p.line(bx, by, bx + 1.6 * math.cos(a + o), by + 1.6 * math.sin(a + o), col, 0.25)


def volcano(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s, flip=False)
    col = pk(rng, [BROWN, BROWN, GRAY])
    p.poly([(-10, 8), (-2.4, -2.6), (2.4, -2.6), (10, 8)], col)
    lava = [(-2.4, -2.6), (2.4, -2.6), (rng.uniform(1, 4), 4.0), (0.2, 1.0), (rng.uniform(-4, -1), 6.0)]
    p.poly(lava, RED)
    for _ in range(rng.randint(3, 6)):
        p.circ(rng.uniform(-2.4, 2.4), rng.uniform(-8.0, -3.4), rng.uniform(0.6, 1.2), pk(rng, [RED, YELLOW]))
    cloud_at(p, rng.uniform(-3, 3), -8.0, 0.9, GRAY if col != GRAY else BLACK)


def island(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s)
    p.ell(0, 5.4, 7.0, 2.4, YELLOW)
    bend = rng.uniform(0.8, 2.4)
    pts = [(1.0 + bend * t * t, 4.4 - 9.0 * t) for t in np.linspace(0, 1, 6)]
    p.path(pts, BROWN, 0.55)
    tx, ty = pts[-1]
    for a in np.linspace(-math.pi * 0.95, -math.pi * 0.05, 5):
        p.path([(tx, ty), (tx + 2.8 * math.cos(a), ty + 2.2 * math.sin(a)), (tx + 4.8 * math.cos(a), ty + 1.0)],
               GREEN, 0.45)


def waterfall(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s)
    rock = pk(rng, [GRAY, BROWN])
    p.rect(-10, -9, -2.0, 8, rock)
    p.rect(3.0, -9, 10, 8, rock)
    p.rect(-2.0, -9, 3.0, 6.0, BLUE)
    for x in (-1.2, 0.4, 2.0):
        p.line(x, -8, x, 5.0, WHITE, 0.2)
    p.rect(-10, 5.0, 10, 8, BLUE)
    for x in (-2.4, -0.6, 1.2, 3.0):
        p.circ(x, 5.2, 0.8, WHITE)
    p.ell(-6.0, -6.0, 3.0, 1.6, GREEN)
    p.ell(7.0, -5.6, 3.0, 1.6, GREEN)


def forest(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s, flip=False)
    for i in range(rng.randint(4, 7)):
        x = rng.uniform(-11, 11)
        k = rng.uniform(0.6, 1.1)
        if rng.rand() < 0.5:
            p.rect(x - 0.5, 4.0, x + 0.5, 7.0, BROWN)
            p.poly([(x - 2.6 * k, 4.4), (x, 4.4 - 9 * k), (x + 2.6 * k, 4.4)], GREEN)
        else:
            p.rect(x - 0.6, 3.0, x + 0.6, 7.0, BROWN)
            p.circ(x, 3.0 - 2.6 * k, 2.8 * k, GREEN)


def nest(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s)
    egg = pk(rng, [WHITE, BLUE, WHITE, YELLOW])
    for i in range(rng.randint(2, 4)):
        p.ell(-2.4 + i * 1.8, 0.8, 1.2, 1.6, egg)
    p.ell(0, 3.0, 5.4, 2.0, BROWN)
    for x in np.arange(-4.6, 5, 1.6):
        p.line(x - 0.6, 2.0, x + 0.8, 4.0, YELLOW, 0.2)


def egg(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s, tilt=0.3)
    col = pk(rng, [WHITE, WHITE, YELLOW, BLUE, BROWN])
    p.ell(0, 0.6, 3.4, 4.6, col)
    if rng.rand() < 0.4:
        p.path([(-3.4, 0.0), (-2.0, -1.0), (-0.8, 0.4), (0.6, -1.0), (2.0, 0.4), (3.4, -0.4)], BLACK, 0.2)
        if rng.rand() < 0.6:
            p.circ(0, -3.2, 2.0, YELLOW)
            p.dot(-0.6, -3.6, BLACK)
            p.poly([(0.4, -3.2), (1.6, -2.8), (0.4, -2.4)], RED)


def bone(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s, tilt=0.8)
    col = pk(rng, [WHITE, WHITE, YELLOW])
    p.line(-4.4, 0, 4.4, 0, col, 1.0)
    for x in (-5.0, 5.0):
        p.circ(x, -1.0, 1.3, col)
        p.circ(x, 1.0, 1.3, col)


def shell(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s, tilt=0.4)
    col = pk(rng, [WHITE, YELLOW, RED, GRAY])
    pts = [(5.0 * math.cos(a), 3.0 - 5.6 * math.sin(a)) for a in np.linspace(0, math.pi, 12)]
    p.poly(pts + [(1.2, 4.4), (-1.2, 4.4)], col)
    for a in np.linspace(0.2, math.pi - 0.2, 6):
        p.line(0, 4.0, 4.6 * math.cos(a), 3.0 - 5.0 * math.sin(a), alt(col, BROWN, RED), 0.2)


def comet(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s, tilt=0.4)
    for k, c in enumerate((RED, YELLOW, WHITE)):
        p.poly([(-2.0, -1.2 + k * 0.4), (9.0 - k * 2, -0.2 - k * 0.1), (9.0 - k * 2, 0.2 + k * 0.1), (-2.0, 1.2 - k * 0.4)], c)
    p.circ(-2.4, 0, 1.8, WHITE)
    p.circ(-2.4, 0, 1.0, YELLOW)


def planet(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s, tilt=0.4)
    col = pk(rng, [YELLOW, RED, BLUE, BROWN, GREEN])
    ring = alt(col, WHITE, YELLOW, GRAY)
    p.ell(0, 0, 8.0, 1.6, ring)
    p.circ(0, 0, 4.2, col)
    p.clipped()
    for y in (-2.4, 0.6, 2.4):
        p.line(-5, y, 5, y + 0.4, alt(col, BROWN, RED), 0.3)
    p.unclip()
    p.path([(-8.0, 0.0), (-5.6, 1.2), (0, 1.7), (5.6, 1.2), (8.0, 0.0)], ring, 0.45)


SUBJECTS = {
    'cloud': (cloud, ('sky', 'sky', 'yellow'), 'CLOUD CLOUDS CLOUDY'),
    'rain': (rain, ('sky', 'gray', 'plain'), 'RAIN RAINY STORM'),
    'lightning': (lightning, ('night', 'sky'), 'LIGHTNING THUNDER BOLT'),
    'rainbow': (rainbow, ('sky',), 'RAINBOW'),
    'snowflake': (snowflake, ('sky', 'gray', 'night'), 'SNOWFLAKE SNOW FROST'),
    'volcano': (volcano, ('sky', 'night'), 'VOLCANO LAVA'),
    'island': (island, ('water', 'sea'), 'ISLAND'),
    'waterfall': (waterfall, ('sky',), 'WATERFALL RIVER'),
    'forest': (forest, ('sky', 'snow', 'field'), 'WOODS JUNGLE'),
    'nest': (nest, ('sky', 'plain', 'leaf'), 'NEST'),
    'egg': (egg, ('plain', 'yellow', 'grass'), 'EGG EGGS'),
    'bone': (bone, ('plain', 'grass', 'gray'), 'BONE'),
    'shell': (shell, ('beach', 'plain', 'under'), 'SHELL SEASHELL CLAM'),
    'comet': (comet, ('night',), 'COMET METEOR ASTEROID'),
    'planet': (planet, ('night',), 'PLANET SATURN JUPITER MARS SPACE'),
}
