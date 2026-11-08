"""Cats: 50 named cat types (coat x pose x features) plus random mixes, so CAT.PNG is a different cat each episode."""
import numpy as np

from .draw import WHITE, BLACK, RED, BLUE, GREEN, YELLOW, BROWN, GRAY, H, W, Pen, pk, alt

TABBIES = ('tabby_ginger', 'tabby_brown', 'tabby_gray')
COATS = {
    'black': dict(body=(BLACK,), eyes=(YELLOW, GREEN)),
    'white': dict(body=(WHITE,), eyes=(BLUE, YELLOW, GREEN, 'odd')),
    'gray': dict(body=(GRAY,), eyes=(GREEN, YELLOW)),
    'ginger': dict(body=(YELLOW, BROWN), eyes=(GREEN, YELLOW, BROWN)),
    'tabby_ginger': dict(body=(YELLOW,), pattern='stripes', pcol=(BROWN,), eyes=(GREEN, BROWN)),
    'tabby_brown': dict(body=(BROWN,), pattern='stripes', pcol=(BLACK,), eyes=(GREEN, YELLOW)),
    'tabby_gray': dict(body=(GRAY,), pattern='stripes', pcol=(BLACK,), eyes=(GREEN, YELLOW)),
    'tuxedo': dict(body=(BLACK,), chest=WHITE, paw=WHITE, muzzle=WHITE, eyes=(YELLOW, GREEN)),
    'bicolor': dict(body=(BLACK, GRAY, BROWN, YELLOW), chest=WHITE, paw=WHITE, muzzle=WHITE, belly=WHITE,
                    eyes=(GREEN, YELLOW, BLUE)),
    'calico': dict(body=(WHITE,), pattern='patches', pcol=(YELLOW, BLACK, BROWN), eyes=(GREEN, YELLOW)),
    'tortie': dict(body=(BLACK,), pattern='patches', pcol=(BROWN, YELLOW), eyes=(YELLOW, GREEN)),
    'siamese': dict(body=(WHITE,), point=(BROWN, BLACK), eyes=(BLUE,)),
    'snowshoe': dict(body=(WHITE,), point=(BROWN, GRAY, BLACK), paw=WHITE, eyes=(BLUE,)),
    'van': dict(body=(WHITE,), cap=(YELLOW, BROWN, BLACK, GRAY), eyes=(BLUE, YELLOW, GREEN, 'odd')),
    'spotted': dict(body=(YELLOW,), pattern='spots', pcol=(BLACK, BROWN), eyes=(GREEN,)),
    'ticked': dict(body=(BROWN,), pattern='ticks', pcol=(YELLOW, BLACK), eyes=(GREEN, YELLOW)),
    'sphynx': dict(body=(GRAY, WHITE), eyes=(GREEN, BLUE, YELLOW)),
}
POSES = ('sit', 'side', 'loaf', 'sleep', 'stretch', 'arch', 'face', 'box', 'lie', 'pounce', 'yarn')

# (label, coat, pose, features)  features: kitten long fluffy fold big tuft bob notail
CAT_TYPES = [
    ('BLACK SITTING', 'black', 'sit', ''), ('HALLOWEEN BLACK', 'black', 'arch', ''),
    ('BLACK PORTRAIT', 'black', 'face', ''), ('BLACK STRETCH', 'black', 'stretch', ''),
    ('BLACK KITTEN HISS', 'black', 'arch', 'kitten'), ('WHITE SITTING', 'white', 'sit', ''),
    ('WHITE KITTEN NAP', 'white', 'sleep', 'kitten'), ('RUSSIAN BLUE', 'gray', 'side', 'big'),
    ('GRAY IN A BOX', 'gray', 'box', ''), ('GINGER LOAF', 'ginger', 'loaf', ''),
    ('GINGER STRETCH', 'ginger', 'stretch', ''), ('GINGER NAP', 'ginger', 'sleep', ''),
    ('GINGER TABBY', 'tabby_ginger', 'sit', ''), ('GINGER TABBY FACE', 'tabby_ginger', 'face', ''),
    ('GINGER KITTEN YARN', 'tabby_ginger', 'yarn', 'kitten'), ('BROWN TABBY WALK', 'tabby_brown', 'side', ''),
    ('BROWN TABBY YARN', 'tabby_brown', 'yarn', ''), ('BROWN TABBY FACE', 'tabby_brown', 'face', ''),
    ('GRAY TABBY NAP', 'tabby_gray', 'sleep', ''), ('GRAY TABBY POUNCE', 'tabby_gray', 'pounce', ''),
    ('TABBY KITTEN BOX', 'tabby_gray', 'box', 'kitten'), ('TUXEDO SITTING', 'tuxedo', 'sit', ''),
    ('TUXEDO FACE', 'tuxedo', 'face', ''), ('TUXEDO WALK', 'tuxedo', 'side', ''),
    ('BICOLOR LOUNGE', 'bicolor', 'lie', ''), ('BICOLOR WALK', 'bicolor', 'side', ''),
    ('CALICO NAP', 'calico', 'sleep', ''), ('CALICO WALK', 'calico', 'side', ''),
    ('CALICO FACE', 'calico', 'face', ''), ('JAPANESE BOBTAIL', 'calico', 'sit', 'bob'),
    ('TORTIE LOAF', 'tortie', 'loaf', ''), ('TORTIE FACE', 'tortie', 'face', ''),
    ('SIAMESE SITTING', 'siamese', 'sit', ''), ('SIAMESE FACE', 'siamese', 'face', ''),
    ('SIAMESE WALK', 'siamese', 'side', ''), ('SNOWSHOE', 'snowshoe', 'sit', ''),
    ('RAGDOLL', 'snowshoe', 'loaf', 'long fluffy'), ('WHITE PERSIAN', 'white', 'sit', 'long fluffy'),
    ('GRAY PERSIAN', 'gray', 'loaf', 'long fluffy'), ('MAINE COON', 'tabby_brown', 'side', 'long fluffy tuft'),
    ('MAINE COON SITTING', 'tabby_gray', 'sit', 'long fluffy tuft'), ('SCOTTISH FOLD', 'gray', 'face', 'fold'),
    ('FOLD TABBY', 'tabby_gray', 'sit', 'fold'), ('SPHYNX', 'sphynx', 'sit', 'big'),
    ('SPHYNX FACE', 'sphynx', 'face', 'big'), ('BENGAL STALK', 'spotted', 'pounce', ''),
    ('BENGAL WALK', 'spotted', 'side', ''), ('ABYSSINIAN', 'ticked', 'side', 'big'),
    ('MANX', 'bicolor', 'side', 'notail'), ('TURKISH VAN', 'van', 'sit', ''),
]

_F = {'long': {'long', 'fluffy'}}
CAT_WORDS = {
    'CAT': {}, 'CATS': {}, 'KITTY': {}, 'KITTIES': {}, 'MEOW': {}, 'TOMCAT': {}, 'FELINE': {}, 'PURR': {},
    'KITTEN': {'add': {'kitten'}}, 'KITTENS': {'add': {'kitten'}},
    'TABBY': {'coat': TABBIES}, 'GINGER': {'coat': ('ginger', 'tabby_ginger')},
    'MARMALADE': {'coat': ('ginger', 'tabby_ginger')}, 'CALICO': {'coat': ('calico',)},
    'TORTIE': {'coat': ('tortie',)}, 'TUXEDO': {'coat': ('tuxedo',)}, 'SIAMESE': {'coat': ('siamese',)},
    'SNOWSHOE': {'coat': ('snowshoe',)}, 'RAGDOLL': {'coat': ('snowshoe', 'siamese'), 'add': _F['long']},
    'PERSIAN': {'coat': ('white', 'gray', 'ginger', 'black', 'calico'), 'add': _F['long']},
    'MAINECOON': {'coat': TABBIES + ('black', 'bicolor'), 'add': _F['long'] | {'tuft'}},
    'SCOTTISH': {'add': {'fold'}}, 'FOLD': {'add': {'fold'}}, 'SPHYNX': {'coat': ('sphynx',)},
    'BENGAL': {'coat': ('spotted',)}, 'ABYSSINIAN': {'coat': ('ticked',)}, 'MANX': {'add': {'notail'}},
    'BOBTAIL': {'add': {'bob'}}, 'BLACKCAT': {'coat': ('black',)}, 'WHITECAT': {'coat': ('white',)},
    'RUSSIAN': {'coat': ('gray',)}, 'CATNAP': {'pose': ('sleep',)}, 'LOAF': {'pose': ('loaf',)},
    'SCAREDYCAT': {'pose': ('arch',)},
}


def merge(conds):
    out = {}
    for c in conds:
        for k in ('coat', 'pose'):
            if k in c:
                both = tuple(x for x in out.get(k, c[k]) if x in c[k])
                out[k] = both or c[k]
        out['add'] = out.get('add', set()) | c.get('add', set())
    return out


def random_spec(rng, cond=None):
    cond = cond or {}
    coats, poses = cond.get('coat'), cond.get('pose')
    cands = [t for t in CAT_TYPES if (not coats or t[1] in coats) and (not poses or t[2] in poses)]
    if cands and rng.rand() < 0.85:
        _, coat, pose, feats = pk(rng, cands)
        feats = set(feats.split())
    else:
        coat, pose = pk(rng, coats or tuple(COATS)), pk(rng, poses or POSES)
        feats = set()
        for f, pr in (('kitten', 0.15), ('long', 0.15), ('bob', 0.04), ('notail', 0.03)):
            if rng.rand() < pr:
                feats.add(f)
        if 'long' in feats:
            feats.add('fluffy')
        e = rng.rand()
        feats.add('fold' if e < 0.08 else 'big' if e < 0.14 else 'tuft' if e < 0.2 else 'point')
        if coat == 'sphynx':
            feats = (feats - {'long', 'fluffy', 'fold', 'tuft'}) | {'big'}
    return coat, pose, feats | set(cond.get('add', ()))


class Cat:
    def __init__(self, cv, rng, cx, cy, s, coat, pose, feats):
        self.rng, self.pose, self.f = rng, pose, feats
        co = COATS[coat]
        kit = 'kitten' in feats
        self.hm = 1.22 if kit else 1.0
        self.fur = 1.12 if 'long' in feats else 1.0
        tilt = 0.08 if pose not in ('face', 'box') else 0.04
        self.p = Pen(cv, cx, cy, s * (0.78 if kit else 1.0), rng.choice([-1, 1]), rng.uniform(-tilt, tilt))
        body = pk(rng, co['body'])
        self.c = {r: body for r in ('body', 'head', 'ear', 'tail', 'paw', 'leg')}
        self.c.update(chest=co.get('chest'), belly=co.get('belly'), muzzle=co.get('muzzle'), mask=None, cap=None)
        if 'chest' in co and rng.rand() < 0.25:
            self.c['belly'] = co['chest']
        if 'point' in co:
            pt = pk(rng, co['point'])
            self.c.update(ear=pt, tail=pt, paw=co.get('paw', pt), mask=pt)
        if 'cap' in co:
            cap = pk(rng, co['cap'])
            self.c.update(ear=cap, tail=cap, cap=cap)
        self.pattern, self.pcol = co.get('pattern'), co.get('pcol', ())
        eyes = co['eyes']
        e = pk(rng, eyes)
        face = self.c['mask'] or body
        if e == 'odd':
            self.eyes = (BLUE, YELLOW) if rng.rand() < 0.5 else (YELLOW, BLUE)
        else:
            e = e if e != face else alt(face, GREEN, YELLOW, BLUE)
            self.eyes = (e, e)
        self.collar = pk(rng, [RED, BLUE, GREEN, YELLOW]) if rng.rand() < 0.12 else None
        self.ears = next((e for e in ('fold', 'big', 'tuft') if e in feats), 'point')
        self.gap, self.off = rng.uniform(1.5, 2.3), rng.uniform(0, 2)

    # ---------------------------------------------------------------- parts
    def tail(self, pts):
        if 'notail' in self.f:
            return
        if 'bob' in self.f:
            pts = [pts[0], (pts[0][0] * 0.7 + pts[1][0] * 0.3, pts[0][1] * 0.7 + pts[1][1] * 0.3)]
            self.p.path(pts, self.c['tail'], 0.8)
            return
        w = 0.95 if 'fluffy' in self.f else 0.5
        if 'big' in self.f and self.c['body'] in (GRAY, WHITE) and not self.pattern:
            w = 0.4
        self.p.path(pts, self.c['tail'], w)

    def apply_pattern(self, extent=(-9, 10, -9, 9)):
        p, r = self.p, self.rng
        if not self.pattern:
            return
        p.clipped()
        x0, x1, y0, y1 = extent
        if self.pattern == 'stripes':
            c = self.pcol[0]
            sl = r.uniform(-0.8, 0.8)
            for x in np.arange(x0 + self.off, x1, self.gap):
                p.line(x, y0, x + sl, y1, c, 0.4)
        elif self.pattern == 'patches':
            for k in range(r.randint(3, 6)):
                p.circ(r.uniform(x0 * 0.6, x1 * 0.7), r.uniform(-3, 6), r.uniform(1.0, 2.4), self.pcol[k % len(self.pcol)])
        elif self.pattern == 'spots':
            for _ in range(r.randint(8, 16)):
                p.circ(r.uniform(x0, x1), r.uniform(y0, y1), 0.55, pk(r, self.pcol))
        elif self.pattern == 'ticks':
            for _ in range(r.randint(25, 45)):
                p.dot(r.uniform(x0, x1), r.uniform(y0, y1), self.pcol[0])
        p.unclip()

    def head(self, x, y, r, closed=False):
        p, c, rng = self.p, self.c, self.rng
        r *= self.hm
        ec = c['ear']
        tuftc = BLACK if ec != BLACK else GRAY
        for sd in (-1, 1):
            if self.ears == 'fold':
                p.ell(x + sd * 0.55 * r, y - 0.78 * r, 0.42 * r, 0.32 * r, ec, rot=sd * 0.5)
            else:
                tip = 2.0 if self.ears == 'big' else 1.6
                out = 1.08 if self.ears == 'big' else 0.98
                tx, ty = x + sd * (0.95 if self.ears == 'big' else 0.78) * r, y - tip * r
                p.poly([(x + sd * out * r, y - 0.2 * r), (tx, ty), (x + sd * 0.1 * r, y - 0.88 * r)], ec)
                if self.ears == 'tuft':
                    p.line(tx, ty, tx + sd * 0.1 * r, ty - 0.45 * r, tuftc, 0.3)
                if r >= 4.5:
                    p.poly([(x + sd * 0.82 * r, y - 0.45 * r), (x + sd * 0.78 * r, y - (tip - 0.3) * r),
                            (x + sd * 0.35 * r, y - 0.85 * r)], RED if ec != RED else WHITE)
        if 'long' in self.f:
            p.ell(x, y + 0.5 * r, 1.18 * r, 0.62 * r, c['chest'] or c['head'])
        hl = p.mask()
        p.circ(x, y, r, c['head'])
        p.ell(x, y + 0.2 * r, 1.08 * r, 0.8 * r, c['head'])
        head = p.mask() & ~hl
        p.clip = head
        if self.pattern == 'stripes':
            for dx in (-0.32, 0, 0.32):
                p.line(x + dx * r, y - 0.95 * r, x + dx * r, y - 0.5 * r, self.pcol[0], 0.3)
            if r >= 4.5:
                for sd in (-1, 1):
                    for dy in (0.1, 0.35):
                        p.line(x + sd * 0.75 * r, y + dy * r, x + sd * 1.1 * r, y + dy * r, self.pcol[0], 0.3)
        elif self.pattern == 'patches':
            sd = rng.choice([-1, 1])
            p.circ(x + sd * rng.uniform(0.3, 0.7) * r, y - rng.uniform(0.2, 0.7) * r, rng.uniform(0.45, 0.7) * r,
                   pk(rng, self.pcol))
        elif self.pattern == 'spots':
            for _ in range(rng.randint(2, 5)):
                p.circ(x + rng.uniform(-0.7, 0.7) * r, y + rng.uniform(-0.9, -0.3) * r, 0.18 * r, pk(rng, self.pcol))
        elif self.pattern == 'ticks':
            for _ in range(rng.randint(3, 8)):
                p.dot(x + rng.uniform(-0.8, 0.8) * r, y + rng.uniform(-0.9, 0.6) * r, self.pcol[0])
        if c['cap'] is not None:
            p.ell(x, y - 0.95 * r, 1.1 * r, 0.55 * r, c['cap'])
            if rng.rand() < 0.5:
                p.rect(x - 0.08 * r, y - 0.95 * r, x + 0.08 * r, y - 0.2 * r, c['head'])
        if c['mask'] is not None:
            p.ell(x, y + 0.25 * r, 0.75 * r, 0.62 * r, c['mask'])
        if c['muzzle'] is not None:
            p.ell(x, y + 0.55 * r, 0.55 * r, 0.42 * r, c['muzzle'])
            if rng.rand() < 0.5:
                p.poly([(x - 0.25 * r, y + 0.3 * r), (x, y - 0.6 * r), (x + 0.25 * r, y + 0.3 * r)], c['muzzle'])
        p.unclip()
        self.face(x, y, r, closed)
        if self.collar is not None and self.pose not in ('face', 'sleep'):
            for t in np.linspace(-0.55, 0.55, 5):
                p.dot(x + t * r, y + 0.98 * r, self.collar)

    def face(self, x, y, r, closed):
        p = self.p
        fc = self.c['mask'] or self.c['head']
        dark = BLACK if fc != BLACK else GRAY
        wc = WHITE if fc in (BLACK, BROWN, GRAY) else BLACK
        ex, ey = 0.42 * r, y - 0.08 * r
        big = r >= 4.5
        for sd, ec in zip((-1, 1), self.eyes):
            if closed:
                p.line(x + sd * ex - 0.35 * r / 3, ey + 0.1, x + sd * ex + 0.35 * r / 3, ey + 0.1, dark, 0.25)
            elif big:
                p.ell(x + sd * ex, ey, 0.21 * r, 0.17 * r, ec)
                p.ell(x + sd * ex, ey, 0.05 * r, 0.15 * r, BLACK if ec != BLACK else WHITE)
            elif r >= 3.4:
                p.ell(x + sd * ex, ey, 0.6, 0.5, ec)
            else:
                p.dot(x + sd * ex, ey, ec)
        if big:
            p.poly([(x - 0.13 * r, y + 0.2 * r), (x + 0.13 * r, y + 0.2 * r), (x, y + 0.36 * r)], RED)
            p.path([(x - 0.22 * r, y + 0.58 * r), (x, y + 0.48 * r), (x + 0.22 * r, y + 0.58 * r)], dark, 0.25)
        else:
            p.dot(x, y + 0.32 * r, RED)
        if big:
            for sd in (-1, 1):
                for dy in (0.25, 0.45, 0.65):
                    for t in np.linspace(0, 1, 6):
                        p.dot(x + sd * (0.55 + 0.85 * t) * r, y + (0.42 + (dy - 0.42) * t) * r, wc)

    def paws(self, pts):
        if self.c['paw'] != self.c['body'] or self.pose in ('box',):
            for x, y in pts:
                self.p.ell(x, y, 0.8, 0.5, self.c['paw'])

    # ---------------------------------------------------------------- poses
    def draw(self):
        getattr(self, 'pose_' + self.pose)()

    def pose_sit(self):
        p, c, f, r = self.p, self.c, self.fur, self.rng
        self.tail(pk(r, [[(4.5, 5.6), (1.0, 6.8), (-2.5, 6.6), (-3.8, 5.8)],
                         [(4.8, 4.0), (7.0, 1.5), (7.2, -1.0), (6.3, -2.5)],
                         [(5.0, 5.5), (7.5, 5.8), (8.6, 4.5)]]))
        p.ell(1.6, 3.4, 3.9 * f, 3.3 * f, c['body'])
        p.ell(-0.4, 0.9, 2.5 * f, 3.0 * f, c['body'])
        p.rect(-2.2, 2.5, -1.1, 6.3, c['body'])
        p.rect(-0.6, 2.5, 0.5, 6.3, c['body'])
        self.apply_pattern()
        if c['chest']:
            p.ell(-0.8, 2.2, 1.3, 2.3, c['chest'])
        self.paws([(-1.65, 6.2), (-0.05, 6.2)])
        self.head(-1.0, -2.2, 3.0)

    def pose_yarn(self):
        self.pose_sit()
        p, r = self.p, self.rng
        col = alt(self.c['body'], *[pk(r, [RED, BLUE, GREEN, YELLOW]), RED, BLUE])
        p.path([(-5.2, 5.4), (-4.0, 6.7), (-2.0, 6.6), (0.5, 6.9)], col, 0.3)
        p.circ(-5.2, 5.2, 1.7, col)
        lc = alt(col, WHITE, BLACK) if col != YELLOW else BROWN
        p.line(-6.3, 4.5, -4.2, 6.1, lc, 0.25)
        p.line(-6.0, 6.0, -4.3, 4.2, lc, 0.25)

    def pose_side(self):
        p, c, f, r = self.p, self.c, self.fur, self.rng
        self.tail(pk(r, [[(6.2, 0.8), (7.8, -1.5), (8.2, -4.0), (7.4, -5.5)],
                         [(6.3, 1.5), (8.2, 2.5), (9.0, 4.5)],
                         [(6.3, 1.0), (8.0, -0.2), (9.3, -0.4)]]))
        p.ell(1.5, 1.6, 5.2 * f, 2.5 * f, c['body'])
        g = r.uniform(-0.9, 0.9)
        feet = []
        for lx, o in ((-2.6, g), (-1.0, -g), (3.0, -g), (4.6, g)):
            p.line(lx, 2.5, lx + o, 6.0, c['leg'], 0.55)
            feet.append((lx + o, 6.1))
        self.apply_pattern()
        if c['belly']:
            p.ell(1.5, 3.3, 3.6, 0.9, c['belly'])
        if c['chest']:
            p.ell(-3.0, 2.0, 1.3, 1.7, c['chest'])
        self.paws(feet)
        self.head(-4.2, -1.3, 2.9)

    def pose_loaf(self):
        p, c, f = self.p, self.c, self.fur
        self.tail([(6.2, 4.8), (3.0, 6.4), (-1.0, 6.5), (-3.5, 6.0)])
        p.ell(1.2, 3.0, 5.6 * f, 3.2 * f, c['body'])
        self.apply_pattern()
        if c['chest']:
            p.ell(-2.8, 3.6, 1.6, 2.0, c['chest'])
        self.paws([(-4.2, 5.9), (-2.4, 6.0)])
        self.head(-3.2, -0.5, 3.0)

    def pose_sleep(self):
        p, c, f, r = self.p, self.c, self.fur, self.rng
        p.ell(0.8, 3.0, 5.2 * f, 3.4 * f, c['body'])
        self.apply_pattern()
        a = np.linspace(0.15, 2.7, 8)
        if 'notail' not in self.f:
            self.tail([(0.8 + 5.0 * f * np.cos(t), 3.0 + 3.1 * f * np.sin(t)) for t in a])
        self.head(-2.4, 2.0, 2.6, closed=True)
        if r.rand() < 0.45:
            zc = pk(r, [WHITE, BLACK])
            for i, (zx, zy) in enumerate([(3.0, -2.5), (5.0, -4.5)][:r.randint(1, 3)]):
                X, Y = p.at(zx, zy)
                X, Y = int(X), int(Y)
                for dx, dy in ((0, 0), (1, 0), (2, 0), (1, 1), (0, 2), (1, 2), (2, 2)):
                    if 0 <= X + dx < W and 0 <= Y + dy < H:
                        p.cv[Y + dy, X + dx] = zc

    def pose_stretch(self):
        p, c, f = self.p, self.c, self.fur
        self.tail([(5.8, -0.5), (7.0, -3.0), (7.6, -5.5)])
        p.ell(1.0, 1.5, 5.4 * f, 2.2 * f, c['body'], rot=-0.38)
        p.line(-3.2, 3.0, -6.8, 5.6, c['leg'], 0.55)
        p.line(-2.4, 3.2, -5.8, 5.9, c['leg'], 0.55)
        p.line(4.2, 1.0, 4.5, 6.0, c['leg'], 0.6)
        p.line(5.3, 0.6, 5.7, 6.0, c['leg'], 0.6)
        self.apply_pattern()
        if c['chest']:
            p.ell(-2.6, 2.8, 1.4, 1.2, c['chest'])
        self.paws([(-6.8, 5.8), (-5.8, 6.1), (4.5, 6.1), (5.7, 6.1)])
        self.head(-4.7, 2.2, 2.6)

    def pose_arch(self):
        p, c, f = self.p, self.c, self.fur
        feet = []
        for lx in (-3.8, -2.6, 3.6, 4.8):
            p.line(lx, 1.5, lx, 6.3, c['leg'], 0.55)
            feet.append((lx, 6.3))
        for t in np.linspace(0, 1, 9):
            p.circ(-4.0 + 9.0 * t, 2.0 - 4.6 * np.sin(np.pi * t), 2.1 * f, c['body'])
        if 'notail' not in self.f:
            w = 1.0 if 'bob' not in self.f else 0.8
            end = (6.2, -6.0) if 'bob' not in self.f else (5.6, -1.5)
            p.line(5.2, 0.5, end[0], end[1], c['tail'], w)
        self.apply_pattern()
        self.paws(feet)
        self.head(-5.4, 0.6, 2.6)

    def pose_face(self):
        p, c = self.p, self.c
        p.ell(0, 8.8, 7.5 * self.fur, 3.4, c['body'])
        self.apply_pattern((-9, 10, 5, 12))
        if c['chest']:
            p.ell(0, 8.6, 2.4, 3.0, c['chest'])
        self.head(0, 0.6, 5.4)

    def pose_box(self):
        p, c, r = self.p, self.c, self.rng
        if r.rand() < 0.4:
            self.tail([(3.5, 2.0), (6.0, -0.5), (6.6, -3.5)])
        box = alt(c['body'], *(pk(r, [BROWN, BROWN, YELLOW, GRAY]), BROWN, YELLOW))
        edge = alt(box, BLACK, GRAY)
        self.head(0.5, -0.8, 3.0)
        p.rect(-5.2, 2.2, 5.2, 7.0, box)
        p.poly([(-5.2, 2.2), (-7.2, 0.6), (-7.2, 3.5), (-5.2, 5.0)], box)
        p.poly([(5.2, 2.2), (7.2, 0.6), (7.2, 3.5), (5.2, 5.0)], box)
        p.line(-5.2, 2.2, 5.2, 2.2, edge, 0.3)
        p.line(-5.2, 2.2, -5.2, 7.0, edge, 0.3)
        p.line(5.2, 2.2, 5.2, 7.0, edge, 0.3)
        for x in (-1.6, 2.6):
            p.ell(x, 2.0, 0.95, 0.6, c['paw'])

    def pose_lie(self):
        p, c, f = self.p, self.c, self.fur
        self.tail([(6.8, 4.5), (8.6, 5.6), (9.6, 4.8)])
        p.ell(1.3, 3.6, 6.0 * f, 2.5 * f, c['body'])
        p.line(-3.2, 4.4, -6.6, 5.4, c['leg'], 0.55)
        p.line(-2.4, 5.1, -6.0, 6.2, c['leg'], 0.55)
        p.line(5.0, 4.8, 7.4, 6.2, c['leg'], 0.6)
        self.apply_pattern()
        if c['belly']:
            p.ell(1.0, 5.0, 4.0, 1.0, c['belly'])
        self.paws([(-6.6, 5.4), (-6.0, 6.2), (7.4, 6.2)])
        self.head(-4.2, 1.2, 2.8)

    def pose_pounce(self):
        p, c, f = self.p, self.c, self.fur
        self.tail([(6.2, 2.4), (8.0, 1.5), (9.6, 1.9)])
        p.ell(1.5, 3.3, 5.2 * f, 2.0 * f, c['body'], rot=-0.12)
        p.path([(-2.4, 4.2), (-3.8, 5.2), (-3.0, 6.2)], c['leg'], 0.55)
        p.path([(-1.0, 4.4), (-2.2, 5.4), (-1.4, 6.2)], c['leg'], 0.55)
        p.path([(3.5, 3.6), (5.8, 4.4), (4.6, 6.2)], c['leg'], 0.6)
        self.apply_pattern()
        self.paws([(-3.0, 6.2), (-1.4, 6.2), (4.6, 6.2)])
        self.head(-4.6, 2.2, 2.7)


def draw_cat(cv, rng, cx, cy, s, coat, pose, feats):
    Cat(cv, rng, cx, cy, s, coat, pose, set(feats)).draw()


def cat(cv, rng, cx, cy, s, cond=None):
    draw_cat(cv, rng, cx, cy, s, *random_spec(rng, cond))


CAT_BGS = {'sleep': ('room', 'plain', 'room', 'gray'), 'arch': ('night', 'night', 'room', 'plain'),
           'box': ('room', 'plain', 'gray'), 'face': ('plain', 'room', 'gray', 'yellow', 'sky', 'grass')}
DEFAULT_CAT_BGS = ('room', 'room', 'plain', 'grass', 'sky', 'field', 'night', 'gray', 'yellow')
