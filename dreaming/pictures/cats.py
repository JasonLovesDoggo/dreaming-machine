"""Cats: big, solid, simple cats (portrait / sit / side / loaf / sleep) on plain high-contrast backgrounds.

A cat is (coat, pose, feats); coat = (body colour, pattern, pattern colour). Words in the filename narrow it and
compose: colour adjectives next to a cat word set the body colour (ORANGE CAT, RED KITTEN, BLACK AND WHITE CAT),
pattern/breed words set the coat (TABBY, CALICO, SIAMESE), pose words the pose (SLEEPY, LOAF), size words the scale."""
import numpy as np

from .draw import WHITE, BLACK, RED, BLUE, GREEN, YELLOW, BROWN, GRAY, H, W, Pen, layer, outline, pk, alt

PATTERNS = ('solid', 'stripes', 'bicolor', 'calico', 'tortie', 'point', 'spots')
COATS = {
    'black': (BLACK, 'solid', None), 'white': (WHITE, 'solid', None), 'gray': (GRAY, 'solid', None),
    'ginger': (YELLOW, 'solid', None), 'brown': (BROWN, 'solid', None),
    'ginger tabby': (YELLOW, 'stripes', BROWN), 'brown tabby': (BROWN, 'stripes', BLACK),
    'gray tabby': (GRAY, 'stripes', BLACK), 'tuxedo': (BLACK, 'bicolor', WHITE),
    'gray white': (GRAY, 'bicolor', WHITE), 'ginger white': (YELLOW, 'bicolor', WHITE),
    'brown white': (BROWN, 'bicolor', WHITE), 'calico': (WHITE, 'calico', None), 'tortie': (BLACK, 'tortie', None),
    'siamese': (WHITE, 'point', BROWN), 'blue point': (WHITE, 'point', GRAY), 'bengal': (YELLOW, 'spots', BLACK),
}
NATURAL = [('black', 3), ('white', 2), ('gray', 2), ('ginger', 2), ('brown', 1), ('ginger tabby', 3),
           ('brown tabby', 2), ('gray tabby', 2), ('tuxedo', 3), ('gray white', 1), ('ginger white', 1),
           ('brown white', 1), ('calico', 2), ('tortie', 1), ('siamese', 2), ('blue point', 1), ('bengal', 1)]
POSES = ('face', 'sit', 'side', 'loaf', 'sleep')
POSE_W = (0.36, 0.26, 0.2, 0.1, 0.08)
CAT_TYPES = [(f'{coat.upper()} {pose.upper()}', coat, pose, '') for coat in COATS for pose in POSES]

TABBIES = ('ginger tabby', 'brown tabby', 'gray tabby')
GINGERS = ('ginger', 'ginger tabby', 'ginger white')
# standalone cat words: each one alone means "a cat" (and narrows it)
CAT_WORDS = {
    'CAT': {}, 'CATS': {}, 'KITTY': {}, 'KITTIES': {}, 'MEOW': {}, 'TOMCAT': {}, 'FELINE': {}, 'PURR': {},
    'KITTEN': {'add': {'kitten'}}, 'KITTENS': {'add': {'kitten'}},
    'TABBY': {'coat': TABBIES}, 'GINGER': {'coat': GINGERS}, 'MARMALADE': {'coat': GINGERS},
    'CALICO': {'coat': ('calico',)}, 'TORTIE': {'coat': ('tortie',)}, 'TUXEDO': {'coat': ('tuxedo',)},
    'SIAMESE': {'coat': ('siamese', 'blue point')}, 'SNOWSHOE': {'coat': ('siamese',), 'add': {'whitepaw'}},
    'RAGDOLL': {'coat': ('siamese', 'blue point'), 'add': {'fluffy'}},
    'PERSIAN': {'coat': ('white', 'gray', 'ginger', 'black', 'calico'), 'add': {'fluffy'}},
    'MAINECOON': {'coat': TABBIES + ('black',), 'add': {'fluffy', 'big'}},
    'SCOTTISH': {'add': {'fold'}}, 'FOLD': {'add': {'fold'}}, 'SPHYNX': {'coat': ('gray', 'white'), 'add': {'sphynx'}},
    'BENGAL': {'coat': ('bengal',)}, 'ABYSSINIAN': {'coat': ('brown', 'ginger')}, 'MANX': {'add': {'notail'}},
    'BOBTAIL': {'add': {'bob'}}, 'BLACKCAT': {'colors': [BLACK]}, 'WHITECAT': {'colors': [WHITE]},
    'RUSSIAN': {'coat': ('gray',)}, 'CATNAP': {'pose': ('sleep',)}, 'LOAF': {'pose': ('loaf',)},
    'SCAREDYCAT': {'pose': ('side',), 'add': {'fluffy'}},
}
CAT_NOUNS = ('CAT', 'CATS', 'KITTY', 'KITTIES', 'KITTEN', 'KITTENS', 'TOMCAT', 'FELINE')
# colour adjectives: only meaningful next to a cat word (ORANGE alone is still the fruit)
COLOR_WORDS = {'RED': RED, 'PINK': RED, 'ORANGE': YELLOW, 'YELLOW': YELLOW, 'GOLD': YELLOW, 'GOLDEN': YELLOW,
               'GREEN': GREEN, 'BLUE': BLUE, 'BROWN': BROWN, 'CHOCOLATE': BROWN, 'BLACK': BLACK, 'WHITE': WHITE,
               'CREAM': WHITE, 'GRAY': GRAY, 'GREY': GRAY, 'SILVER': GRAY}
MODIFIERS = {w: {'colors': [c]} for w, c in COLOR_WORDS.items()}
MODIFIERS['ORANGE'] = {'colors': [YELLOW], 'orange': True}
MODIFIERS.update({w: {'pattern': 'stripes'} for w in ('STRIPED', 'STRIPY', 'STRIPEY', 'STRIPES')})
MODIFIERS.update({w: {'pattern': 'spots'} for w in ('SPOTTED', 'SPOTTY', 'SPOTS')})
MODIFIERS.update({w: {'add': {'big'}} for w in ('BIG', 'FAT', 'CHONKY', 'CHUNKY', 'HUGE', 'LARGE')})
MODIFIERS.update({w: {'add': {'tiny'}} for w in ('TINY', 'LITTLE', 'SMALL', 'MINI')})
MODIFIERS.update({w: {'add': {'kitten'}} for w in ('BABY',)})
MODIFIERS.update({w: {'add': {'fluffy'}} for w in ('FLUFFY', 'FURRY', 'FUZZY')})
MODIFIERS.update({w: {'pose': ('sleep',)} for w in ('SLEEPY', 'SLEEPING', 'NAPPING', 'NAP')})
MODIFIERS.update({'SITTING': {'pose': ('sit',)}, 'WALKING': {'pose': ('side',)}, 'PORTRAIT': {'pose': ('face',)},
                  'FACE': {'pose': ('face',)}})
CAT_TERMS = {**MODIFIERS, **CAT_WORDS}

LUM = {WHITE: 255, BLACK: 0, RED: 83, BLUE: 137, GREEN: 129, YELLOW: 211, BROWN: 99, GRAY: 192}
STRIPE = {YELLOW: BROWN, RED: BLACK, WHITE: GRAY, GRAY: BLACK, BROWN: BLACK, BLACK: GRAY, BLUE: BLACK, GREEN: BLACK}
BG_PREF = {WHITE: 1.0, GRAY: 0.8, BLUE: 1.0, YELLOW: 0.8, GREEN: 0.8, BLACK: 0.7, RED: 0.3, BROWN: 0.3}


def cond_from_words(words):
    """Merge cat words/modifiers into one condition dict: coat, colors, pattern, pose, add, orange."""
    out = {'colors': [], 'add': set()}
    for w in words:
        c = CAT_TERMS.get(w, {})
        for k in ('coat', 'pose'):
            if k in c:
                both = tuple(x for x in out.get(k, c[k]) if x in c[k])
                out[k] = both or c[k]
        for col in c.get('colors', ()):
            if col not in out['colors']:
                out['colors'].append(col)
        if 'pattern' in c:
            out['pattern'] = c['pattern']
        out['orange'] = out.get('orange') or c.get('orange', False)
        out['add'] |= c.get('add', set())
    return out


def merge(conds):
    """Back-compat: merge CAT_WORDS condition dicts."""
    out = {}
    for c in conds:
        for k in ('coat', 'pose'):
            if k in c:
                both = tuple(x for x in out.get(k, c[k]) if x in c[k])
                out[k] = both or c[k]
        out['add'] = out.get('add', set()) | c.get('add', set())
    return out


def _default_pcol(body, pattern):
    if pattern == 'stripes':
        return STRIPE[body]
    if pattern == 'bicolor':
        return WHITE if body != WHITE else BLACK
    if pattern == 'point':
        return BROWN if body != BROWN else BLACK
    if pattern == 'spots':
        return BLACK if body != BLACK else GRAY
    return None


def random_spec(rng, cond=None):
    cond = cond or {}
    colors = cond.get('colors') or []
    if cond.get('coat'):
        body, pattern, pcol = COATS[pk(rng, cond['coat'])]
    elif colors:
        u = rng.rand()
        pattern = ('stripes' if u < 0.45 else 'bicolor' if u < 0.6 else 'solid') if cond.get('orange') else \
            ('solid' if u < 0.68 else 'stripes' if u < 0.84 else 'bicolor')
        body, pcol = colors[0], None
    else:
        names, ws = zip(*NATURAL)
        body, pattern, pcol = COATS[names[rng.choice(len(names), p=np.array(ws) / sum(ws))]]
    if colors:
        body = colors[0]
        if len(colors) > 1:
            pcol = colors[1]
            if pattern in ('solid', 'calico', 'tortie'):
                pattern = 'bicolor'
        elif pattern in ('calico', 'tortie'):
            pattern = 'solid'
    if cond.get('pattern'):
        pattern = cond['pattern']
    if pcol is None or pcol == body:
        pcol = _default_pcol(body, pattern)
    if cond.get('pose'):
        pose = pk(rng, cond['pose'])
    else:
        pose = POSES[rng.choice(len(POSES), p=POSE_W)]
    feats = set()
    for f, pr in (('kitten', 0.14), ('fluffy', 0.14), ('fold', 0.05), ('collar', 0.2), ('big', 0.06)):
        if rng.rand() < pr:
            feats.add(f)
    return (body, pattern, pcol), pose, feats | set(cond.get('add', ()))


class Cat:
    def __init__(self, cv, rng, cx, cy, s, coat, pose, feats):
        if isinstance(coat, str):
            coat = COATS[coat]
        self.rng, self.pose, self.f = rng, pose, set(feats)
        self.body, self.pattern, self.pcol = coat
        kit = 'kitten' in self.f
        self.hm = 1.2 if kit else 1.0
        self.fur = 1.12 if 'fluffy' in self.f else 1.0
        self.wide = 1.15 if 'big' in self.f else 1.0
        s *= (0.8 if kit else 1.0) * (0.72 if 'tiny' in self.f else 1.0)
        self.p = Pen(cv, cx, cy, s, rng.choice([-1, 1]), rng.uniform(-0.05, 0.05))
        b, pc = self.body, self.pcol
        self.c = dict(body=b, head=b, ear=b, tail=b, paw=b, chest=None, muzzle=None, mask=None)
        if self.pattern == 'bicolor':
            self.c.update(chest=pc, paw=pc, muzzle=pc if rng.rand() < 0.75 else None)
        elif self.pattern == 'point':
            self.c.update(ear=pc, tail=pc, paw=pc if 'whitepaw' not in self.f else WHITE, mask=pc)
        elif self.pattern == 'calico':
            self.patches = (YELLOW, BLACK, BROWN) if rng.rand() < 0.3 else (YELLOW, BLACK)
        elif self.pattern == 'tortie':
            self.patches = (YELLOW, BROWN)
        if 'whitepaw' in self.f:
            self.c['paw'] = WHITE
        face = self.c['mask'] if self.c['mask'] is not None else b
        if self.pattern == 'point':
            eye = BLUE
        else:
            eye = pk(rng, [GREEN, YELLOW, BLUE, GREEN, YELLOW])
        self.eye = eye if eye != face else alt(face, GREEN, YELLOW, BLUE)
        self.collar = alt(b, *(pk(rng, [RED, BLUE, GREEN]), RED, BLUE)) if 'collar' in self.f else None
        self.tail_up = rng.rand() < 0.6

    def used(self):
        u = {self.body, self.pcol} | {v for v in self.c.values() if v is not None}
        return u | set(getattr(self, 'patches', ()))

    # ---------------------------------------------------------------- parts
    def tail(self, pts):
        if 'notail' in self.f:
            return
        if 'bob' in self.f:
            pts = pts[:2]
        self.p.path(pts, self.c['tail'], 0.85 * self.fur)

    def pattern_on(self, x0, x1, y0, y1, vertical=True):
        """Pattern over everything drawn so far (body + legs), inside the given local box."""
        p, r = self.p, self.rng
        if self.pattern in ('solid', 'bicolor', 'point'):
            return
        p.clipped()
        if self.pattern == 'stripes':
            off = r.uniform(0, 1.5)
            for x in np.arange(x0 + off, x1, r.uniform(2.6, 3.2)):
                if vertical:
                    p.line(x, y0, x + r.uniform(-0.5, 0.5), y1, self.pcol, 0.5)
                else:
                    p.line(x0, x - x0 + y0, x1, x - x0 + y0, self.pcol, 0.5)
        elif self.pattern in ('calico', 'tortie'):
            n = r.randint(3, 5) if self.pattern == 'calico' else r.randint(5, 8)
            rad = (1.8, 3.0) if self.pattern == 'calico' else (1.0, 2.0)
            for k in range(n):
                p.circ(r.uniform(x0, x1), r.uniform(y0, y1), r.uniform(*rad), self.patches[k % len(self.patches)])
        elif self.pattern == 'spots':
            for _ in range(r.randint(8, 14)):
                p.circ(r.uniform(x0, x1), r.uniform(y0, y1), 0.6, self.pcol)
        p.unclip()

    def head(self, x, y, r, closed=False):
        p, c, rng = self.p, self.c, self.rng
        r *= self.hm
        ec = c['ear']
        inner = RED if ec != RED else WHITE
        for sd in (-1, 1):
            if 'fold' in self.f:
                p.ell(x + sd * 0.55 * r, y - 0.72 * r, 0.42 * r, 0.3 * r, ec, rot=sd * 0.4)
                continue
            tip = 1.7 if 'sphynx' in self.f else 1.42
            spread = 1.0 if 'sphynx' in self.f else 0.82
            p.poly([(x + sd * 1.0 * r, y - 0.15 * r), (x + sd * spread * r, y - tip * r), (x + sd * 0.15 * r, y - 0.78 * r)], ec)
            if r * p.s >= 3.6:
                p.poly([(x + sd * 0.82 * r, y - 0.42 * r), (x + sd * (spread - 0.04) * r, y - (tip - 0.32) * r),
                        (x + sd * 0.4 * r, y - 0.78 * r)], inner)
        if 'fluffy' in self.f:
            for sd in (-1, 1):
                p.poly([(x + sd * 0.7 * r, y + 0.2 * r), (x + sd * 1.3 * r, y + 0.55 * r), (x + sd * 0.6 * r, y + 0.75 * r)], c['head'])
        hl = p.mask()
        p.ell(x, y, r, 0.84 * r, c['head'])
        p.ell(x, y + 0.22 * r, 1.08 * r, 0.62 * r, c['head'])
        p.clip = p.mask() & ~hl
        if self.pattern == 'stripes':
            for dx in (-0.3, 0, 0.3):
                p.line(x + dx * r, y - 0.82 * r, x + dx * r, y - 0.5 * r, self.pcol, 0.4)
            if r * p.s >= 5:
                for sd in (-1, 1):
                    p.line(x + sd * 0.8 * r, y + 0.1 * r, x + sd * 1.1 * r, y + 0.05 * r, self.pcol, 0.4)
        elif self.pattern in ('calico', 'tortie'):
            sd = rng.choice([-1, 1])
            p.circ(x + sd * rng.uniform(0.4, 0.7) * r, y - rng.uniform(0.3, 0.7) * r, rng.uniform(0.4, 0.6) * r,
                   pk(rng, self.patches))
        elif self.pattern == 'spots':
            for _ in range(rng.randint(2, 4)):
                p.circ(x + rng.uniform(-0.6, 0.6) * r, y - rng.uniform(0.3, 0.8) * r, 0.5, self.pcol)
        if c['mask'] is not None:
            p.ell(x, y + 0.2 * r, 0.62 * r, 0.55 * r, c['mask'])
        if c['muzzle'] is not None:
            p.ell(x, y + 0.48 * r, 0.5 * r, 0.36 * r, c['muzzle'])
            if rng.rand() < 0.5:
                p.poly([(x - 0.2 * r, y + 0.3 * r), (x, y - 0.7 * r), (x + 0.2 * r, y + 0.3 * r)], c['muzzle'])
        p.unclip()
        self.face(x, y, r, closed)
        if self.collar is not None and self.pose != 'sleep':
            p.line(x - 0.55 * r, y + 0.9 * r, x + 0.55 * r, y + 0.9 * r, self.collar, 0.45)
            p.dot(x, y + 1.05 * r, YELLOW if self.collar != YELLOW else WHITE)

    def face(self, x, y, r, closed):
        p = self.p
        fc = self.c['mask'] if self.c['mask'] is not None else self.c['head']
        dark = BLACK if fc != BLACK else GRAY
        wc = WHITE if fc in (BLACK, BROWN, GRAY, RED, BLUE, GREEN) else BLACK
        px = r * p.s
        ex, ey = 0.42 * r, y - 0.02 * r
        for sd in (-1, 1):
            X = x + sd * ex
            if closed:
                p.line(X - 0.18 * r, ey + 0.05 * r, X + 0.18 * r, ey + 0.05 * r, dark, 0.35 / p.s)
            elif px >= 5.5:
                p.ell(X, ey, 0.21 * r, 0.18 * r, self.eye)
                p.line(X, ey - 0.12 * r, X, ey + 0.12 * r, BLACK if self.eye != BLACK else WHITE, 0.3 / p.s)
            else:
                p.ell(X, ey, max(0.17 * r, 0.75 / p.s), max(0.13 * r, 0.6 / p.s), self.eye)
        nose = RED if fc != RED else BLACK
        if px >= 5:
            p.poly([(x - 0.13 * r, y + 0.2 * r), (x + 0.13 * r, y + 0.2 * r), (x, y + 0.34 * r)], nose)
            p.path([(x - 0.22 * r, y + 0.56 * r), (x, y + 0.42 * r), (x + 0.22 * r, y + 0.56 * r)], dark, 0.3 / p.s)
            for sd in (-1, 1):
                for dy in (0.3, 0.52):
                    p.line(x + sd * 0.5 * r, y + dy * r, x + sd * 1.35 * r, y + (dy - 0.1) * r, wc, 0.3 / p.s)
        else:
            p.dot(x, y + 0.3 * r, nose)

    def paws(self, pts):
        if self.c['paw'] != self.c['body']:
            for x, y in pts:
                self.p.ell(x, y, 1.0, 0.6, self.c['paw'])

    # ---------------------------------------------------------------- poses (local units ~ canvas pixels)
    def draw(self):
        getattr(self, 'pose_' + self.pose)()

    def pose_face(self):
        p, c, f = self.p, self.c, self.fur * self.wide
        p.ell(0, 11.5, 12.5 * f, 5.0, c['body'])
        self.pattern_on(-12, 12, 7, 11)
        if c['chest'] is not None:
            p.ell(0, 11, 4.0, 4.0, c['chest'])
        self.head(0, 1.6, 8.0)

    def pose_sit(self):
        p, c, f = self.p, self.c, self.fur * self.wide
        self.tail([(4.5, 8.8), (8.0, 8.6), (9.8, 6.0), (9.6, 3.0)] if self.tail_up else
                  [(4.5, 9.0), (1.0, 9.6), (-3.0, 9.5), (-5.5, 8.8)])
        p.ell(0, 4.8, 5.8 * f, 5.0, c['body'])
        for sd in (-1, 1):
            p.ell(sd * 3.6 * f, 7.0, 3.0 * f, 2.8, c['body'])
        for lx in (-2.0, 2.0):
            p.rect(lx - 1.0, 4.0, lx + 1.0, 9.6, c['body'])
        self.pattern_on(-7, 7, -0.5, 9.5)
        if c['chest'] is not None:
            p.ell(0, 3.8, 2.3, 3.4, c['chest'])
        self.paws([(-2.0, 9.4), (2.0, 9.4)])
        self.head(0, -2.2, 5.2)

    def pose_side(self):
        p, c, f, r = self.p, self.c, self.fur * self.wide, self.rng
        self.tail([(9.5, 0.0), (12.0, -2.5), (12.8, -6.0), (11.8, -8.0)] if self.tail_up else
                  [(9.5, 1.0), (12.5, 2.5), (14.0, 1.0)])
        g = r.uniform(-1.0, 1.0)
        feet = []
        for lx, o in ((-5.2, g), (-2.4, -g), (4.8, -g), (7.6, g)):
            p.line(lx, 3.0, lx + o, 8.6, c['body'], 0.95)
            feet.append((lx + o, 8.9))
        p.ell(1.6, 1.4, 8.6 * f, 4.0 * f, c['body'])
        p.ell(-5.5, -0.5, 3.0, 3.8, c['body'])
        self.pattern_on(-5, 10, -3, 2.5)
        if c['chest'] is not None:
            p.ell(1.6, 4.4, 6.0, 1.3, c['chest'])
            p.ell(-6.2, 1.8, 2.0, 2.6, c['chest'])
        self.paws(feet)
        self.head(-8.2, -3.2, 4.6)

    def pose_loaf(self):
        p, c, f = self.p, self.c, self.fur * self.wide
        self.tail([(11.0, 6.0), (8.5, 9.2), (2.0, 9.6)])
        p.ell(2.2, 4.4, 10.0 * f, 5.4, c['body'])
        p.rect(-6.5, 5.0, 11.0, 9.4, c['body'])
        self.pattern_on(-4, 12, -1, 9)
        if c['chest'] is not None:
            p.ell(-5.0, 5.5, 2.4, 3.0, c['chest'])
        self.paws([(-8.0, 9.0), (-5.0, 9.2)])
        self.head(-5.5, -1.2, 5.0)

    def pose_sleep(self):
        p, c, f = self.p, self.c, self.fur * self.wide
        p.ell(1.0, 4.4, 10.0 * f, 5.4, c['body'])
        self.pattern_on(-8, 11, -1, 9)
        self.tail([(10.5, 6.0), (7.0, 9.4), (0.0, 9.8), (-6.0, 9.4), (-8.5, 8.0)])
        self.head(-4.5, 4.0, 4.6, closed=True)


def draw_cat(cv, rng, cx, cy, s, coat, pose, feats):
    cat_ = Cat(cv, rng, cx, cy, s, coat, pose, set(feats))
    cat_.draw()
    return cat_


def cat(cv, rng, cx, cy, s, cond=None):
    return draw_cat(cv, rng, cx, cy, s, *random_spec(rng, cond))


def pick_bg(rng, avoid, body):
    cands = [c for c in range(8) if c not in avoid and abs(LUM[c] - LUM[body]) >= 60]
    if not cands:
        cands = [c for c in range(8) if c not in avoid] or [alt(body, WHITE, BLACK)]
    ws = np.array([BG_PREF[c] for c in cands])
    return cands[rng.choice(len(cands), p=ws / ws.sum())]


def render(rng, spec):
    """A single cat filling most of the canvas on a plain, high-contrast background."""
    coat, pose, feats = spec
    s = rng.uniform(0.92, 1.06) * (1.06 if 'big' in feats else 1.0)
    dx = {'face': 2.0, 'sit': 5.0, 'side': 1.5, 'loaf': 3.0, 'sleep': 3.0}[pose]
    L = layer()
    c = draw_cat(L, rng, W / 2 + rng.uniform(-dx, dx), H / 2 + rng.uniform(-0.6, 0.6), s, coat, pose, feats)
    avoid = c.used()
    cv = np.full((H, W), pick_bg(rng, avoid, c.body), np.uint8)
    if pose != 'face' and rng.rand() < 0.15:
        floor = pick_bg(rng, avoid | {cv[0, 0]}, c.body)
        cv[H - rng.randint(2, 4):] = floor
    return outline(cv, L)


CAT_BGS = {'sleep': ('plain', 'gray', 'blue'), 'face': ('plain', 'gray', 'yellow', 'blue', 'grass')}
DEFAULT_CAT_BGS = ('plain', 'plain', 'gray', 'blue', 'grass', 'yellow', 'night')
