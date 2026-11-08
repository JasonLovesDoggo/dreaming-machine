"""Name-conditioned procedural pictures for the 32x20 Paint canvas. Same name -> same category, random variation."""
import math
import numpy as np

H, W = 20, 32
WHITE, BLACK, RED, BLUE, GREEN, YELLOW, BROWN, GRAY = range(8)
_yy, _xx = np.mgrid[0:H, 0:W].astype(np.float64)


def ellipse(cv, cx, cy, rx, ry, c):
    rx, ry = max(rx, 0.6), max(ry, 0.6)
    cv[((_xx - cx) / rx) ** 2 + ((_yy - cy) / ry) ** 2 <= 1] = c


def circle(cv, cx, cy, r, c):
    ellipse(cv, cx, cy, r, r, c)


def rect(cv, x0, y0, x1, y1, c):
    cv[(_xx >= min(x0, x1)) & (_xx <= max(x0, x1)) & (_yy >= min(y0, y1)) & (_yy <= max(y0, y1))] = c


def poly(cv, pts, c):
    inside = np.zeros((H, W), bool)
    n = len(pts)
    for i in range(n):
        (x1, y1), (x2, y2) = pts[i], pts[(i + 1) % n]
        cond = ((y1 > _yy) != (y2 > _yy)) & (_xx < (x2 - x1) * (_yy - y1) / (y2 - y1 + 1e-9) + x1)
        inside ^= cond
    cv[inside] = c


def line(cv, x0, y0, x1, y1, c, r=0.5):
    n = int(max(abs(x1 - x0), abs(y1 - y0)) * 2) + 2
    for t in np.linspace(0, 1, n):
        circle(cv, x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, r, c)


def background(cv, rng, kind=None):
    kind = kind or rng.choice(['sky', 'sky', 'plain', 'gray', 'yellow', 'grass'])
    if kind == 'sky':
        cv[:] = BLUE
        cv[int(H * rng.uniform(0.62, 0.8)):] = rng.choice([GREEN, GREEN, BROWN])
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
        cv[:] = rng.choice([BLUE, WHITE, YELLOW])
        cv[int(H * rng.uniform(0.6, 0.75)):] = BLUE
    return kind


def other(c, rng, avoid):
    opts = [x for x in range(8) if x not in avoid]
    return rng.choice(opts)


def cat(cv, rng, cx, cy, s):
    col = rng.choice([BLACK, BROWN, GRAY, YELLOW])
    d = rng.choice([-1, 1])
    ellipse(cv, cx + d * s, cy + 3 * s, 5 * s, 3 * s, col)
    line(cv, cx + d * 5.5 * s, cy + 3 * s, cx + d * 7.5 * s, cy - 1 * s, col, 0.6 * s)
    hx, hy = cx - d * 3 * s, cy - 1.5 * s
    circle(cv, hx, hy, 3.2 * s, col)
    poly(cv, [(hx - 3 * s, hy - 1 * s), (hx - 2.2 * s, hy - 5.5 * s), (hx - 0.3 * s, hy - 2.5 * s)], col)
    poly(cv, [(hx + 3 * s, hy - 1 * s), (hx + 2.2 * s, hy - 5.5 * s), (hx + 0.3 * s, hy - 2.5 * s)], col)
    eye = GREEN if col != GREEN else YELLOW
    circle(cv, hx - 1.3 * s, hy - 0.6 * s, 0.6 * s, eye)
    circle(cv, hx + 1.3 * s, hy - 0.6 * s, 0.6 * s, eye)
    circle(cv, hx, hy + 0.8 * s, 0.5 * s, RED)


def dog(cv, rng, cx, cy, s):
    col = rng.choice([BROWN, GRAY, BLACK, YELLOW, WHITE])
    ear = BLACK if col != BLACK else BROWN
    d = rng.choice([-1, 1])
    ellipse(cv, cx + d * 1.5 * s, cy + 2.5 * s, 5.5 * s, 2.6 * s, col)
    for lx in (-2.5, 1, 3.5, 5.5):
        rect(cv, cx + d * lx * s - 0.5 * s, cy + 3 * s, cx + d * lx * s + 0.5 * s, cy + 6 * s, col)
    line(cv, cx + d * 7 * s, cy + 1.5 * s, cx + d * 8.5 * s, cy - 1.5 * s, col, 0.5 * s)
    hx, hy = cx - d * 4 * s, cy - 1.5 * s
    circle(cv, hx, hy, 2.8 * s, col)
    ellipse(cv, hx - d * 2.6 * s, hy + 1 * s, 1.6 * s, 1.1 * s, col)
    circle(cv, hx - d * 3.8 * s, hy + 0.6 * s, 0.6 * s, BLACK)
    ellipse(cv, hx + d * 2.2 * s, hy + 0.5 * s, 0.9 * s, 2.2 * s, ear)
    circle(cv, hx - d * 0.8 * s, hy - 0.8 * s, 0.5 * s, BLACK)


def fish(cv, rng, cx, cy, s):
    col = rng.choice([YELLOW, RED, GRAY, GREEN, BROWN])
    d = rng.choice([-1, 1])
    ellipse(cv, cx, cy, 6 * s, 3 * s, col)
    poly(cv, [(cx + d * 5 * s, cy), (cx + d * 9 * s, cy - 3.5 * s), (cx + d * 9 * s, cy + 3.5 * s)], col)
    circle(cv, cx - d * 3.5 * s, cy - 0.8 * s, 0.9 * s, WHITE)
    circle(cv, cx - d * 3.5 * s, cy - 0.8 * s, 0.4 * s, BLACK)
    line(cv, cx - d * 0.5 * s, cy - 2.5 * s, cx + d * 1.5 * s, cy + 2.5 * s, other(col, rng, (col, BLUE)), 0.4 * s)


def house(cv, rng, cx, cy, s):
    wall = rng.choice([RED, YELLOW, WHITE, GRAY, BROWN])
    roof = rng.choice([c for c in (BROWN, RED, BLACK, GRAY) if c != wall])
    rect(cv, cx - 5 * s, cy - 1 * s, cx + 5 * s, cy + 6 * s, wall)
    poly(cv, [(cx - 6.5 * s, cy - 1 * s), (cx, cy - 6.5 * s), (cx + 6.5 * s, cy - 1 * s)], roof)
    door = rng.choice([BLUE, BROWN, BLACK, GREEN])
    rect(cv, cx - 1 * s, cy + 2.5 * s, cx + 1 * s, cy + 6 * s, door)
    win = rng.choice([BLUE, YELLOW, WHITE])
    for wx in (-3.2, 3.2):
        rect(cv, cx + (wx - 0.9) * s, cy + 0.6 * s, cx + (wx + 0.9) * s, cy + 2.2 * s, win if win != wall else BLUE)
    if rng.rand() < 0.5:
        rect(cv, cx + 2.5 * s, cy - 6 * s, cx + 3.8 * s, cy - 3 * s, roof)


def tree(cv, rng, cx, cy, s):
    rect(cv, cx - 0.9 * s, cy, cx + 0.9 * s, cy + 6.5 * s, BROWN)
    crown = rng.choice([GREEN, GREEN, YELLOW, RED])
    for dx, dy, r in [(0, -3, 3.6), (-2.6, -1, 2.6), (2.6, -1, 2.6), (0, -5.5, 2.4)]:
        circle(cv, cx + dx * s, cy + dy * s, r * s, crown)
    if crown == GREEN and rng.rand() < 0.5:
        for _ in range(rng.randint(2, 6)):
            circle(cv, cx + rng.uniform(-3, 3) * s, cy + rng.uniform(-5, 0) * s, 0.5 * s, RED)


def sun(cv, rng, cx, cy, s):
    for a in range(8):
        ang = a * math.pi / 4 + rng.uniform(0, 0.3)
        line(cv, cx, cy, cx + math.cos(ang) * 7 * s, cy + math.sin(ang) * 7 * s, YELLOW, 0.4 * s)
    circle(cv, cx, cy, 4 * s, YELLOW)
    if rng.rand() < 0.5:
        circle(cv, cx - 1.4 * s, cy - 1 * s, 0.5 * s, BLACK)
        circle(cv, cx + 1.4 * s, cy - 1 * s, 0.5 * s, BLACK)
        for t in np.linspace(-1, 1, 7):
            cv[min(H - 1, max(0, int(cy + 1.5 * s + (1 - t * t) * s))), min(W - 1, max(0, int(cx + t * 2 * s)))] = BLACK


def sunset(cv, rng, cx, cy, s):
    hz = int(H * rng.uniform(0.55, 0.7))
    cv[:hz] = RED
    cv[int(hz * rng.uniform(0.3, 0.6)):hz] = YELLOW
    circle(cv, cx, hz, 4.5 * s, rng.choice([YELLOW, RED, WHITE]))
    cv[hz:] = BLUE
    for y in range(hz + 1, H, 2):
        x = rng.randint(0, W - 4)
        cv[y, x:x + rng.randint(2, 6)] = YELLOW


def flower(cv, rng, cx, cy, s):
    line(cv, cx, cy, cx + rng.uniform(-1, 1) * s, cy + 8 * s, GREEN, 0.45 * s)
    ellipse(cv, cx + 1.6 * s, cy + 5 * s, 1.5 * s, 0.7 * s, GREEN)
    petal = rng.choice([RED, BLUE, WHITE, YELLOW])
    n = rng.randint(5, 8)
    for a in range(n):
        ang = a * 2 * math.pi / n
        circle(cv, cx + math.cos(ang) * 2.4 * s, cy + math.sin(ang) * 2.4 * s, 1.5 * s, petal)
    circle(cv, cx, cy, 1.3 * s, YELLOW if petal != YELLOW else BROWN)


def car(cv, rng, cx, cy, s):
    col = rng.choice([RED, BLUE, YELLOW, GREEN, BLACK, GRAY])
    rect(cv, cx - 7 * s, cy, cx + 7 * s, cy + 3 * s, col)
    poly(cv, [(cx - 4 * s, cy), (cx - 2.5 * s, cy - 3 * s), (cx + 3 * s, cy - 3 * s), (cx + 4.5 * s, cy)], col)
    win = WHITE if col != WHITE else BLUE
    poly(cv, [(cx - 3 * s, cy - 0.3 * s), (cx - 2 * s, cy - 2.3 * s), (cx + 0.5 * s, cy - 2.3 * s), (cx + 0.5 * s, cy - 0.3 * s)], win)
    poly(cv, [(cx + 1.3 * s, cy - 0.3 * s), (cx + 1.3 * s, cy - 2.3 * s), (cx + 2.6 * s, cy - 2.3 * s), (cx + 3.6 * s, cy - 0.3 * s)], win)
    for wx in (-4, 4):
        circle(cv, cx + wx * s, cy + 3.2 * s, 1.8 * s, BLACK)
        circle(cv, cx + wx * s, cy + 3.2 * s, 0.6 * s, GRAY)


def bird(cv, rng, cx, cy, s):
    col = rng.choice([BLUE, RED, YELLOW, BLACK, BROWN, WHITE])
    d = rng.choice([-1, 1])
    ellipse(cv, cx, cy, 4 * s, 2.5 * s, col)
    circle(cv, cx - d * 3.5 * s, cy - 2 * s, 2 * s, col)
    poly(cv, [(cx - d * 5.2 * s, cy - 2.6 * s), (cx - d * 7.5 * s, cy - 1.8 * s), (cx - d * 5.2 * s, cy - 1.2 * s)], YELLOW if col != YELLOW else RED)
    circle(cv, cx - d * 4 * s, cy - 2.5 * s, 0.45 * s, BLACK if col != BLACK else WHITE)
    wing = other(col, rng, (col, BLUE))
    poly(cv, [(cx - d * 1 * s, cy - 0.5 * s), (cx + d * 3.5 * s, cy - 0.5 * s), (cx + d * 1.5 * s, cy - 4.5 * s)], wing)
    poly(cv, [(cx + d * 3.5 * s, cy), (cx + d * 6.5 * s, cy - 1.5 * s), (cx + d * 6.5 * s, cy + 1.5 * s)], col)
    for lx in (-1, 1):
        line(cv, cx + lx * s, cy + 2.4 * s, cx + lx * s, cy + 4.5 * s, BLACK if col != BLACK else BROWN, 0.3)


def face(cv, rng, cx, cy, s):
    col = rng.choice([YELLOW, YELLOW, BROWN, WHITE, GREEN])
    circle(cv, cx, cy, 6 * s, col)
    circle(cv, cx - 2.2 * s, cy - 1.8 * s, 0.9 * s, BLACK)
    circle(cv, cx + 2.2 * s, cy - 1.8 * s, 0.9 * s, BLACK)
    happy = rng.rand() < 0.8
    for t in np.linspace(-1, 1, 15):
        y = cy + 2.2 * s + ((1 - t * t) if happy else (t * t - 1)) * 1.5 * s
        circle(cv, cx + t * 3.2 * s, y, 0.5 * s, RED if rng.rand() < 0.5 else BLACK)


def mountain(cv, rng, cx, cy, s):
    for _ in range(rng.randint(1, 3)):
        mx, mh, mw = cx + rng.uniform(-8, 8) * s, rng.uniform(7, 11) * s, rng.uniform(6, 10) * s
        base = H - 1
        col = rng.choice([GRAY, BROWN, GRAY])
        poly(cv, [(mx - mw, base + 1), (mx, base - mh), (mx + mw, base + 1)], col)
        poly(cv, [(mx - mw * 0.28, base - mh * 0.72), (mx, base - mh), (mx + mw * 0.28, base - mh * 0.72)], WHITE)


def boat(cv, rng, cx, cy, s):
    hull = rng.choice([BROWN, RED, BLACK, WHITE])
    wy = cy + 3 * s
    poly(cv, [(cx - 7 * s, wy - 2 * s), (cx + 7 * s, wy - 2 * s), (cx + 5 * s, wy + 1 * s), (cx - 5 * s, wy + 1 * s)], hull)
    line(cv, cx, wy - 2 * s, cx, wy - 11 * s, BROWN, 0.3)
    sail = rng.choice([WHITE, RED, YELLOW])
    poly(cv, [(cx + 0.5 * s, wy - 10.5 * s), (cx + 0.5 * s, wy - 3 * s), (cx + 6 * s, wy - 3 * s)], sail)
    if rng.rand() < 0.5:
        poly(cv, [(cx - 0.5 * s, wy - 9 * s), (cx - 0.5 * s, wy - 3 * s), (cx - 4.5 * s, wy - 3 * s)], sail)


def heart(cv, rng, cx, cy, s):
    col = rng.choice([RED, RED, YELLOW, BLACK])
    circle(cv, cx - 2.3 * s, cy - 1.5 * s, 3 * s, col)
    circle(cv, cx + 2.3 * s, cy - 1.5 * s, 3 * s, col)
    poly(cv, [(cx - 5.2 * s, cy - 0.5 * s), (cx + 5.2 * s, cy - 0.5 * s), (cx, cy + 6 * s)], col)


def star(cv, rng, cx, cy, s):
    col = rng.choice([YELLOW, YELLOW, RED, WHITE])
    r0 = rng.uniform(0, 1)
    pts = []
    for i in range(10):
        ang = -math.pi / 2 + i * math.pi / 5 + r0 * 0.2
        r = (6.5 if i % 2 == 0 else 2.7) * s
        pts.append((cx + math.cos(ang) * r, cy + math.sin(ang) * r))
    poly(cv, pts, col)


def robot(cv, rng, cx, cy, s):
    col = rng.choice([GRAY, GRAY, BLUE, RED, YELLOW])
    rect(cv, cx - 3 * s, cy - 6 * s, cx + 3 * s, cy - 1.5 * s, col)
    rect(cv, cx - 4 * s, cy - 1 * s, cx + 4 * s, cy + 4.5 * s, col)
    line(cv, cx, cy - 6 * s, cx, cy - 8 * s, BLACK, 0.3)
    circle(cv, cx, cy - 8.3 * s, 0.8 * s, RED)
    eye = rng.choice([BLUE, YELLOW, RED, GREEN])
    for ex in (-1.4, 1.4):
        rect(cv, cx + (ex - 0.5) * s, cy - 4.8 * s, cx + (ex + 0.5) * s, cy - 3.8 * s, eye if eye != col else BLACK)
    rect(cv, cx - 1.5 * s, cy - 2.6 * s, cx + 1.5 * s, cy - 2.2 * s, BLACK)
    for lx in (-2, 2):
        rect(cv, cx + (lx - 0.6) * s, cy + 4.5 * s, cx + (lx + 0.6) * s, cy + 7 * s, col)
    for ax in (-1, 1):
        line(cv, cx + ax * 4 * s, cy, cx + ax * 6.5 * s, cy + 3 * s, col, 0.5 * s)


def apple(cv, rng, cx, cy, s):
    col = rng.choice([RED, RED, GREEN, YELLOW])
    circle(cv, cx - 1.5 * s, cy, 3.8 * s, col)
    circle(cv, cx + 1.5 * s, cy, 3.8 * s, col)
    line(cv, cx, cy - 3 * s, cx + 0.8 * s, cy - 6 * s, BROWN, 0.35 * s)
    ellipse(cv, cx + 2.4 * s, cy - 5 * s, 1.6 * s, 0.8 * s, GREEN if col != GREEN else BROWN)


def ghost(cv, rng, cx, cy, s):
    col = rng.choice([WHITE, WHITE, GRAY])
    circle(cv, cx, cy - 2 * s, 4.5 * s, col)
    rect(cv, cx - 4.5 * s, cy - 2 * s, cx + 4.5 * s, cy + 4 * s, col)
    for i in range(3):
        circle(cv, cx + (-3 + 3 * i) * s, cy + 4.3 * s, 1.5 * s, col)
    circle(cv, cx - 1.6 * s, cy - 2.5 * s, 1 * s, BLACK)
    circle(cv, cx + 1.6 * s, cy - 2.5 * s, 1 * s, BLACK)
    ellipse(cv, cx, cy + 0.7 * s, 0.9 * s, 1.3 * s, BLACK)


def pizza(cv, rng, cx, cy, s):
    poly(cv, [(cx - 6 * s, cy - 5 * s), (cx + 6 * s, cy - 5 * s), (cx, cy + 7 * s)], YELLOW)
    rect(cv, cx - 6.3 * s, cy - 6.3 * s, cx + 6.3 * s, cy - 4.5 * s, BROWN)
    for _ in range(rng.randint(3, 7)):
        t, u = rng.uniform(0.15, 0.7), rng.uniform(-0.6, 0.6)
        circle(cv, cx + u * 6 * s * (1 - t), cy - 4 * s + t * 10 * s, 0.9 * s, rng.choice([RED, RED, GREEN, BLACK]))


def snowman(cv, rng, cx, cy, s):
    for dy, r in [(3.5, 3.6), (-1.6, 2.6), (-5.4, 1.9)]:
        circle(cv, cx, cy + dy * s, r * s, WHITE)
    circle(cv, cx - 0.7 * s, cy - 5.8 * s, 0.35 * s + 0.2, BLACK)
    circle(cv, cx + 0.7 * s, cy - 5.8 * s, 0.35 * s + 0.2, BLACK)
    line(cv, cx, cy - 5 * s, cx + 1.8 * s, cy - 4.8 * s, RED, 0.3)
    hat = rng.choice([BLACK, RED, BLUE, GREEN])
    rect(cv, cx - 2 * s, cy - 7.2 * s, cx + 2 * s, cy - 6.8 * s, hat)
    rect(cv, cx - 1.2 * s, cy - 9.5 * s, cx + 1.2 * s, cy - 7 * s, hat)
    for b in (-1, 0.5):
        circle(cv, cx, cy + b * s, 0.35 * s + 0.2, BLACK)


def moon(cv, rng, cx, cy, s):
    col = rng.choice([YELLOW, WHITE, GRAY])
    circle(cv, cx, cy, 5 * s, col)
    if rng.rand() < 0.7:
        circle(cv, cx + rng.choice([-1, 1]) * 2.6 * s, cy - 1 * s, 4.4 * s, cv[0, 0])
    else:
        for _ in range(3):
            circle(cv, cx + rng.uniform(-3, 3) * s, cy + rng.uniform(-3, 3) * s, 0.8 * s, GRAY if col != GRAY else WHITE)


def abstract(cv, rng, cx, cy, s):
    for _ in range(rng.randint(2, 6)):
        c = rng.randint(1, 8)
        k = rng.randint(0, 3)
        x, y = cx + rng.uniform(-8, 8) * s, cy + rng.uniform(-6, 6) * s
        if k == 0:
            rect(cv, x, y, x + rng.uniform(2, 9) * s, y + rng.uniform(2, 6) * s, c)
        elif k == 1:
            circle(cv, x, y, rng.uniform(1.5, 4.5) * s, c)
        else:
            line(cv, x, y, x + rng.uniform(-10, 10) * s, y + rng.uniform(-6, 6) * s, c, 0.6)


CATEGORIES = {
    'cat': (cat, None), 'dog': (dog, None), 'fish': (fish, 'water'), 'house': (house, None), 'tree': (tree, 'sky'),
    'sun': (sun, 'sky'), 'sunset': (sunset, 'self'), 'flower': (flower, None), 'car': (car, None),
    'bird': (bird, 'sky'), 'face': (face, None), 'mountain': (mountain, 'sky'), 'boat': (boat, 'sea'),
    'heart': (heart, None), 'star': (star, 'night'), 'robot': (robot, None), 'apple': (apple, None),
    'ghost': (ghost, 'night'), 'pizza': (pizza, None), 'snowman': (snowman, 'sky'), 'moon': (moon, 'night'),
}
KEYWORDS = {
    'CAT': 'cat', 'KITTY': 'cat', 'KITTEN': 'cat', 'DOG': 'dog', 'PUPPY': 'dog', 'FISH': 'fish', 'HOUSE': 'house',
    'HOME': 'house', 'TREE': 'tree', 'SUNSET': 'sunset', 'SUN': 'sun', 'FLOWER': 'flower', 'ROSE': 'flower',
    'CAR': 'car', 'BIRD': 'bird', 'FACE': 'face', 'SMILE': 'face', 'SMILEY': 'face', 'SELFIE': 'face',
    'MOUNTAIN': 'mountain', 'HILL': 'mountain', 'BOAT': 'boat', 'SHIP': 'boat', 'HEART': 'heart', 'LOVE': 'heart',
    'STAR': 'star', 'ROBOT': 'robot', 'APPLE': 'apple', 'GHOST': 'ghost', 'PIZZA': 'pizza', 'SNOWMAN': 'snowman',
    'MOON': 'moon',
}
IMAGE_EXTS = ('.PNG', '.JPG', '.JPEG', '.BMP', '.GIF')


def is_image(name):
    return name.upper().endswith(IMAGE_EXTS)


def parse_name(name):
    base = name.upper().rsplit('.', 1)[0]
    found = []
    for kw in sorted(KEYWORDS, key=len, reverse=True):
        while kw in base:
            found.append((base.index(kw), KEYWORDS[kw]))
            base = base.replace(kw, ' ' * len(kw), 1)
    found.sort()
    return [c for _, c in found][:3]


def generate(name, rng):
    cv = np.zeros((H, W), np.uint8)
    cats = parse_name(name)
    if not cats:
        background(cv, rng, rng.choice(['plain', 'gray', 'yellow', 'night', 'grass']))
        abstract(cv, rng, W / 2, H / 2, 1.0)
        return cv
    first_bg = CATEGORIES[cats[0]][1]
    if first_bg == 'self':
        CATEGORIES[cats[0]][0](cv, rng, W / 2 + rng.uniform(-6, 6), H / 2, 1.0)
        cats = cats[1:]
        obj = np.full((H, W), 255, np.uint8)
    else:
        obj = np.full((H, W), 255, np.uint8)
    n = len(cats)
    for i, c in enumerate(cats):
        s = rng.uniform(0.8, 1.15) / (n ** 0.6)
        cx = W * (i + 0.5) / n + rng.uniform(-3, 3) / n
        cy = H / 2 + rng.uniform(-1.5, 1.5)
        CATEGORIES[c][0](obj, rng, cx, cy, s)
    if first_bg != 'self':
        vals = obj[obj != 255]
        dom = np.bincount(vals, minlength=8).argmax() if vals.size else -1
        for _ in range(8):
            background(cv, rng, first_bg if first_bg and _ == 0 else None)
            if np.bincount(cv.ravel(), minlength=8).argmax() != dom:
                break
    return np.where(obj == 255, cv, obj).astype(np.uint8)
