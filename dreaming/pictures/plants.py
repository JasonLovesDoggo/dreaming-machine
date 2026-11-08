"""Flowers, plants, fruit and veg."""
import math
import numpy as np

from .draw import WHITE, BLACK, RED, BLUE, GREEN, YELLOW, BROWN, GRAY, pen, pk, alt


def stem(p, x0, y0, x1, y1, leaves=1, rng=None):
    p.line(x0, y0, x1, y1, GREEN, 0.35)
    for k in range(leaves):
        t = 0.4 + 0.3 * k
        sd = 1 if k % 2 == 0 else -1
        p.ell(x0 + (x1 - x0) * t + sd * 1.2, y0 + (y1 - y0) * t, 1.3, 0.55, GREEN, rot=-sd * 0.6)


def bunch(p, rng, fn, n_lo=1, n_hi=3):
    n = rng.randint(n_lo, n_hi + 1)
    xs = np.linspace(-5.0, 5.0, n) if n > 1 else [0.0]
    for x in xs:
        fn(p, x + rng.uniform(-0.6, 0.6), rng.uniform(-1.0, 1.0), rng.uniform(0.75, 1.0) if n > 1 else 1.0)


def rose(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s, tilt=0.2)
    col = pk(rng, [RED, RED, RED, YELLOW, WHITE])

    def one(p, x, dy, k):
        stem(p, x, -2 * k + dy, x * 0.3, 7.0, 2)
        p.circ(x, -3.0 * k + dy, 2.4 * k, col)
        p.path([(x - 1.0 * k, -3.0 * k + dy), (x, -4.0 * k + dy), (x + 1.0 * k, -3.0 * k + dy),
                (x, -2.0 * k + dy)], alt(col, BLACK, RED) if col != RED else BLACK, 0.2)
        p.poly([(x - 2.0 * k, -2.2 * k + dy), (x, -0.4 * k + dy), (x + 2.0 * k, -2.2 * k + dy)], col)
    bunch(p, rng, one, 1, 3)


def tulip(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s)
    cols = [RED, YELLOW, WHITE, BLUE]

    def one(p, x, dy, k):
        c = pk(rng, cols)
        stem(p, x, -2 * k + dy, x * 0.4, 7.0, 1)
        p.ell(x, -3.0 * k + dy, 1.7 * k, 2.0 * k, c)
        for o in (-1.2, 0, 1.2):
            p.poly([(x + (o - 0.7) * k, -4.0 * k + dy), (x + o * k, -6.0 * k + dy), (x + (o + 0.7) * k, -4.0 * k + dy)], c)
    bunch(p, rng, one, 1, 4)


def sunflower(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s, tilt=0.15)

    def one(p, x, dy, k):
        stem(p, x, dy, x * 0.3, 8.0, 2)
        for a in np.linspace(0, 2 * math.pi, 12, endpoint=False):
            p.ell(x + 2.8 * k * math.cos(a), -3.0 * k + dy + 2.8 * k * math.sin(a), 1.2 * k, 0.6 * k, YELLOW, rot=a)
        p.circ(x, -3.0 * k + dy, 1.8 * k, BROWN)
        p.dot(x, -3.0 * k + dy, BLACK)
    bunch(p, rng, one, 1, 2)


def daisy(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s)
    pet = pk(rng, [WHITE, WHITE, WHITE, YELLOW, BLUE])

    def one(p, x, dy, k):
        stem(p, x, dy - 2 * k, x * 0.3, 7.0, 1)
        for a in np.linspace(0, 2 * math.pi, 8, endpoint=False):
            p.ell(x + 1.8 * k * math.cos(a), -3.0 * k + dy + 1.8 * k * math.sin(a), 1.0 * k, 0.5 * k, pet, rot=a)
        p.circ(x, -3.0 * k + dy, 0.9 * k, YELLOW if pet != YELLOW else BROWN)
    bunch(p, rng, one, 1, 4)


def lotus(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s, flip=False)
    col = pk(rng, [RED, WHITE, YELLOW])
    p.ell(rng.uniform(-6, -3), 5.6, 3.0, 0.8, GREEN)
    p.ell(rng.uniform(3, 6), 5.4, 2.4, 0.7, GREEN)
    p.ell(0, 3.6, 4.0, 1.2, GREEN)
    for a, h in ((-0.9, 3.0), (0.9, 3.0), (-0.45, 4.2), (0.45, 4.2), (0, 5.0)):
        p.ell(math.sin(a) * 2.6, 2.4 - h * 0.5, 1.0, h * 0.55, col, rot=a)
    p.circ(0, 2.6, 0.8, YELLOW if col != YELLOW else RED)


def cactus(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s)
    if rng.rand() < 0.5:
        p.rect(-2.4, 4.6, 2.4, 7.2, pk(rng, [RED, BROWN]))
    p.ell(0, -0.4, 1.6, 6.0, GREEN)
    for sd, ay in ((-1, rng.uniform(-1, 2)), (1, rng.uniform(-2, 1))):
        p.path([(0, ay + 1.6), (sd * 3.2, ay + 1.6), (sd * 3.2, ay - 2.4)], GREEN, 0.8)
    for y in (-4.0, -1.0, 2.0):
        p.dot(0, y, alt(GREEN, YELLOW))
    if rng.rand() < 0.5:
        p.circ(0, -6.2, 0.9, pk(rng, [RED, YELLOW, WHITE]))


def mushroom(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s, flip=False)
    cap = pk(rng, [RED, RED, BROWN, YELLOW])

    def one(p, x, dy, k):
        top = dy + 1.6 - 2.0 * k
        p.ell(x, 5.0, 1.4 * k, 2.6 * k, WHITE if cap != WHITE else GRAY)
        p.rect(x - 1.3 * k, top, x + 1.3 * k, 5.0, WHITE if cap != WHITE else GRAY)
        p.poly([(x + 4.0 * k * math.cos(a), top - 3.6 * k * math.sin(a)) for a in np.linspace(0, math.pi, 14)], cap)
        if cap == RED:
            for ox, oy in ((-2.0, 0.8), (0.0, 2.4), (2.0, 0.8)):
                p.circ(x + ox * k, top - oy * k, 0.55 * k, WHITE)
    bunch(p, rng, one, 1, 2)


def palm(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s)
    bend = rng.uniform(0.5, 2.4)
    pts = [(bend * t * t, 7.0 - 12.0 * t) for t in np.linspace(0, 1, 7)]
    p.path(pts, BROWN, 0.7)
    tx, ty = pts[-1]
    for a in np.linspace(-math.pi * 0.95, -math.pi * 0.05, 5):
        p.path([(tx, ty), (tx + 3.4 * math.cos(a), ty + 2.6 * math.sin(a)), (tx + 5.8 * math.cos(a), ty + 1.0)],
               GREEN, 0.5)
    for sd in (-0.8, 0.8):
        p.circ(tx + sd, ty + 0.8, 0.7, BROWN)


def pine(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s, flip=False)
    n = rng.randint(1, 4)
    xs = np.linspace(-6, 6, n) if n > 1 else [0.0]
    snowy = rng.rand() < 0.3
    for x in xs:
        k = rng.uniform(0.7, 1.0) if n > 1 else 1.0
        p.rect(x - 0.6, 5.0, x + 0.6, 7.0, BROWN)
        for t in range(3):
            y = 4.8 - t * 3.0 * k
            w = (3.6 - t * 0.9) * k
            p.poly([(x - w, y), (x, y - 4.0 * k), (x + w, y)], GREEN)
            if snowy:
                p.poly([(x - 0.8, y - 3.0 * k), (x, y - 4.0 * k), (x + 0.8, y - 3.0 * k)], WHITE)


def leaf(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s, tilt=0.8)
    col = pk(rng, [GREEN, GREEN, RED, YELLOW, BROWN])
    if rng.rand() < 0.5:
        p.ell(0, 0, 3.0, 5.6, col)
        p.line(0, -5.0, 0, 7.4, alt(col, BROWN, YELLOW), 0.25)
        for y in (-2.6, 0.0, 2.6):
            p.line(0, y + 1.2, 2.0, y - 0.4, alt(col, BROWN, YELLOW), 0.2)
            p.line(0, y + 1.2, -2.0, y - 0.4, alt(col, BROWN, YELLOW), 0.2)
    else:
        for a in np.linspace(-math.pi * 0.9, -math.pi * 0.1, 5):
            p.poly([(0, 2.0), (4.6 * math.cos(a - 0.25), 2.0 + 4.6 * math.sin(a - 0.25)),
                    (6.0 * math.cos(a), 2.0 + 6.0 * math.sin(a)), (4.6 * math.cos(a + 0.25), 2.0 + 4.6 * math.sin(a + 0.25))],
                   col)
        p.circ(0, 1.4, 1.6, col)
        p.line(0, 2.0, 0, 6.4, alt(col, BROWN, YELLOW), 0.3)


def clover(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s, tilt=0.4)
    n = 4 if rng.rand() < 0.4 else 3
    p.path([(0, 0), (0.8, 3.6), (0.4, 6.8)], GREEN, 0.35)
    for k in range(n):
        a = -math.pi / 2 + k * 2 * math.pi / n
        for o in (-0.45, 0.45):
            p.circ(2.0 * math.cos(a + o * 0.5) + 0.9 * math.cos(a + o * 2), 2.0 * math.sin(a + o * 0.5) + 0.9 * math.sin(a + o * 2) - 0.6,
                   1.4, GREEN)


# ------------------------------------------------------------------ fruit and veg
def banana(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s, tilt=0.4)
    ripe = pk(rng, [YELLOW, YELLOW, YELLOW, YELLOW, GREEN])
    for k in range(rng.randint(1, 4)):
        off = k * 1.2
        pts = [(-6 + 12 * t, -off + 3.4 - 6.0 * t * (1 - t) * 1.6) for t in np.linspace(0, 1, 10)]
        p.path(pts, ripe, 1.0 - 0.04 * k)
        p.dot(*pts[0], BROWN); p.dot(*pts[-1], BROWN)
    if ripe == YELLOW and rng.rand() < 0.4:
        p.dot(0.6, 0.6, BROWN)


def cherry(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s)
    p.path([(-2.6, 3.0), (-1.0, -2.0), (0.6, -5.0)], GREEN, 0.25)
    p.path([(2.6, 3.4), (1.6, -1.6), (0.6, -5.0)], GREEN, 0.25)
    p.ell(1.6, -5.2, 1.6, 0.7, GREEN, rot=-0.4)
    for x, y in ((-2.6, 4.0), (2.6, 4.4)):
        p.circ(x, y, 2.0, RED)
        p.dot(x - 0.8, y - 0.8, WHITE)


def strawberry(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s, tilt=0.4)
    p.poly([(-3.6, -1.4), (3.6, -1.4), (2.6, 3.0), (0, 5.4), (-2.6, 3.0)], RED)
    p.ell(0, -1.0, 3.6, 1.6, RED)
    for x, y in ((-1.8, 0.0), (0.0, 0.6), (1.8, 0.0), (-0.9, 2.2), (0.9, 2.2), (0, 3.8)):
        p.dot(x, y, YELLOW)
    for a in (-2.4, -1.2, 0, 1.2, 2.4):
        p.ell(a, -2.4, 0.9, 0.5, GREEN, rot=a * 0.3)
    p.line(0, -2.6, 0, -4.0, GREEN, 0.25)


def grapes(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s, tilt=0.3)
    col = pk(rng, [BLUE, BLUE, GREEN])
    for r, n in enumerate((4, 3, 3, 2, 1)):
        for i in range(n):
            p.circ((i - (n - 1) / 2) * 1.9, -1.8 + r * 1.7, 1.1, col)
    for r, n in enumerate((4, 3, 3, 2, 1)):
        for i in range(n):
            if (r + i) % 2 == 0:
                p.dot((i - (n - 1) / 2) * 1.9 - 0.3, -2.1 + r * 1.7, WHITE)
    p.line(0, -2.8, 0.4, -5.2, BROWN, 0.3)
    p.ell(2.0, -4.6, 1.8, 0.9, GREEN, rot=-0.3)


def orange(cv, rng, cx, cy, s, col=None):
    p = pen(cv, rng, cx, cy, s, flip=True)
    col = col if col is not None else pk(rng, [YELLOW, YELLOW, RED])

    def one(p, x, dy, k):
        p.circ(x, dy + 1.0, 3.6 * k, col)
        p.dot(x - 1.4 * k, dy - 0.6, WHITE)
        p.ell(x + 0.8 * k, dy - 2.8 * k + 1.0, 1.0 * k, 0.5 * k, GREEN, rot=-0.4)
    bunch(p, rng, one, 1, 2)


def lemon(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s, tilt=0.5)
    p.ell(0, 0.6, 4.6, 3.2, YELLOW)
    p.ell(-4.8, 0.6, 0.8, 0.6, YELLOW); p.ell(4.8, 0.6, 0.8, 0.6, YELLOW)
    p.dot(-1.6, -0.8, WHITE)
    if rng.rand() < 0.5:
        p.ell(1.4, -2.8, 1.6, 0.6, GREEN, rot=-0.3)


def pear(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s, tilt=0.3)
    col = pk(rng, [GREEN, YELLOW, YELLOW, BROWN])
    p.circ(0, 2.4, 3.4, col)
    p.circ(0, -1.4, 2.0, col)
    p.line(0, -3.2, 0.4, -5.0, BROWN, 0.3)
    p.ell(1.6, -4.4, 1.2, 0.5, GREEN, rot=-0.4)
    p.dot(-1.4, 1.4, WHITE)


def watermelon(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s, tilt=0.25)
    if rng.rand() < 0.5:
        p.ell(0, 1.0, 6.4, 4.2, GREEN)
        p.clipped()
        for x in np.arange(-6, 7, 1.8):
            p.ell(x, 1.0, 0.4, 4.2, GRAY)
        p.unclip()
    else:
        pts = [(6.4 * math.cos(a), -1.0 + 6.4 * math.sin(a)) for a in np.linspace(0, math.pi, 14)]
        p.poly([(-6.4, -1.0)] + pts[::-1][:-1] + [(6.4, -1.0)], GREEN)
        inner = [(5.4 * math.cos(a), -1.0 + 5.4 * math.sin(a)) for a in np.linspace(0, math.pi, 14)]
        p.poly(inner, RED)
        p.rect(-6.6, -1.6, 6.6, -1.0, RED)
        for x in (-3.0, -1.0, 1.0, 3.0):
            p.dot(x, 1.4 + abs(x) * -0.2, BLACK)
        for x in (-2.0, 0.0, 2.0):
            p.dot(x, 3.2, BLACK)


def pineapple(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s, flip=False)
    for a in (-0.6, -0.25, 0.0, 0.25, 0.6):
        p.ell(math.sin(a) * 2.0, -4.4 - math.cos(a) * 1.6, 0.5, 2.0, GREEN, rot=a)
    p.ell(0, 2.0, 3.2, 4.6, YELLOW)
    p.clipped()
    for x in np.arange(-8, 6, 2.6):
        p.line(x, -3.0, x + 5, 7.0, BROWN, 0.2)
        p.line(x, 7.0, x + 5, -3.0, BROWN, 0.2)
    p.unclip()


def carrot(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s, tilt=0.6)
    p.poly([(-1.8, -3.0), (1.8, -3.0), (0, 6.0)], RED)
    for x in (-0.8, 0.4, 1.0):
        p.dot(x * 0.8, -1.0 + abs(x) * 1.4, BROWN)
    for a in (-0.5, 0.0, 0.5):
        p.ell(math.sin(a) * 1.6, -4.6, 0.5, 1.8, GREEN, rot=a)


def corn(cv, rng, cx, cy, s):
    p = pen(cv, rng, cx, cy, s, tilt=0.5)
    p.ell(0, 0, 2.0, 5.6, YELLOW)
    for y in np.arange(-4.0, 5.0, 1.4):
        for x in (-0.9, 0.5):
            p.dot(x, y, BROWN if rng.rand() < 0.15 else YELLOW)
    p.poly([(-2.6, 1.0), (-1.6, 6.4), (0, 7.0), (-0.4, 3.0)], GREEN)
    p.poly([(2.6, 1.0), (1.6, 6.4), (0, 7.0), (0.4, 3.0)], GREEN)


SUBJECTS = {
    'rose': (rose, ('plain', 'sky', 'grass', 'yellow'), 'ROSE ROSES'),
    'tulip': (tulip, ('sky', 'field', 'plain'), 'TULIP TULIPS'),
    'sunflower': (sunflower, ('sky', 'field'), 'SUNFLOWER'),
    'daisy': (daisy, ('grass', 'sky', 'field'), 'DAISY DAISIES BLOSSOM'),
    'lotus': (lotus, ('water', 'pond'), 'LOTUS LILY WATERLILY'),
    'cactus': (cactus, ('desert', 'plain', 'yellow'), 'CACTUS CACTI SUCCULENT'),
    'mushroom': (mushroom, ('grass', 'field', 'plain'), 'MUSHROOM TOADSTOOL FUNGI'),
    'palm': (palm, ('beach',), 'PALM PALMTREE COCONUT TROPICAL'),
    'pine': (pine, ('snow', 'sky', 'field'), 'PINE SPRUCE XMAS CHRISTMAS'),
    'leaf': (leaf, ('plain', 'yellow', 'sky', 'gray'), 'LEAF LEAVES MAPLE AUTUMN FERN PLANT PLANTS'),
    'clover': (clover, ('grass', 'plain', 'yellow'), 'CLOVER SHAMROCK LUCKY'),
    'banana': (banana, ('plain', 'gray', 'blue'), 'BANANA BANANAS'),
    'cherry': (cherry, ('plain', 'yellow', 'gray'), 'CHERRY CHERRIES'),
    'strawberry': (strawberry, ('plain', 'grass', 'yellow'), 'STRAWBERRY BERRY BERRIES'),
    'grapes': (grapes, ('plain', 'yellow', 'gray'), 'GRAPE GRAPES VINE'),
    'orange': (orange, ('plain', 'blue', 'gray'), 'ORANGE ORANGES TANGERINE PEACH'),
    'lemon': (lemon, ('plain', 'blue', 'gray'), 'LEMON LIME'),
    'pear': (pear, ('plain', 'blue', 'gray'), 'PEAR'),
    'watermelon': (watermelon, ('plain', 'yellow', 'gray', 'grass'), 'WATERMELON MELON'),
    'pineapple': (pineapple, ('plain', 'blue', 'beach'), 'PINEAPPLE'),
    'carrot': (carrot, ('plain', 'grass', 'gray'), 'CARROT CARROTS VEGGIE'),
    'corn': (corn, ('plain', 'field', 'blue'), 'CORN MAIZE'),
}
