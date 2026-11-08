"""Name-conditioned procedural pictures for the 32x20 Paint canvas. Same name -> same subject, random variation.

Each subject is a parametric generator `fn(layer, rng, cx, cy, s)` drawing into a transparent (255) layer; the
filename's keywords pick up to 3 subjects, the episode's rng picks pose/colours/position/background."""
import numpy as np

from .draw import H, W, CLEAR, background, outline, layer, pk
from . import classic, cats, mammals, birds, sea, plants, sky

_MODULES = [classic, mammals, birds, sea, plants, sky]
SUBJECTS = {}
for _m in _MODULES:
    SUBJECTS.update(_m.SUBJECTS)
SUBJECTS['cat'] = (cats.cat, None, ' '.join(cats.CAT_WORDS))

CATEGORIES = {k: (fn, bg) for k, (fn, bg, _) in SUBJECTS.items()}
CATEGORY_WORDS = {k: words.split() for k, (_, _, words) in SUBJECTS.items()}
KEYWORDS = {w: k for k, ws in CATEGORY_WORDS.items() for w in ws}
CAT_WORDS = list(cats.CAT_WORDS)
IMAGE_EXTS = ('.PNG', '.JPG', '.JPEG', '.BMP', '.GIF')
GENERIC_BGS = ('sky', 'sky', 'plain', 'gray', 'yellow', 'grass')


def is_image(name):
    return name.upper().endswith(IMAGE_EXTS)


def parse_words(name):
    """[(keyword, category)] in filename order, longest keywords matched first, at most 3 subjects (cat words merge)."""
    base = name.upper().rsplit('.', 1)[0]
    found = []
    for kw in sorted(KEYWORDS, key=lambda k: (-len(k), k)):
        while kw in base:
            found.append((base.index(kw), kw))
            base = base.replace(kw, ' ' * len(kw), 1)
    found.sort()
    out, cat_i = [], None
    for _, kw in found:
        c = KEYWORDS[kw]
        if c == 'cat' and cat_i is not None:
            out[cat_i][0].append(kw)
            continue
        if c == 'cat':
            cat_i = len(out)
        out.append(([kw], c))
    return [(kws, c) for kws, c in out][:3]


def parse_name(name):
    return [c for _, c in parse_words(name)]


def _bgs(c, spec):
    if c == 'cat':
        return cats.CAT_BGS.get(spec[1], cats.DEFAULT_CAT_BGS)
    bg = CATEGORIES[c][1]
    return (bg,) if isinstance(bg, str) else bg


def generate(name, rng):
    cv = np.zeros((H, W), np.uint8)
    subs = parse_words(name)
    if not subs:
        background(cv, rng, pk(rng, ['plain', 'gray', 'yellow', 'night', 'grass']))
        classic.abstract(cv, rng, W / 2, H / 2, 1.0)
        return cv
    specs = [cats.random_spec(rng, cats.merge(cats.CAT_WORDS[k] for k in kws)) if c == 'cat' else None
             for kws, c in subs]
    first = _bgs(subs[0][1], specs[0])
    if first == ('self',):
        CATEGORIES[subs[0][1]][0](cv, rng, W / 2 + rng.uniform(-6, 6), H / 2, 1.0)
        subs, specs, first = subs[1:], specs[1:], None
    obj = layer()
    n = len(subs)
    for i, ((kws, c), spec) in enumerate(zip(subs, specs)):
        s = rng.uniform(0.8, 1.15) / (n ** 0.6)
        cx = W * (i + 0.5) / n + rng.uniform(-3, 3) / n
        cy = H / 2 + rng.uniform(-1.5, 1.5)
        L = layer()
        if c == 'cat':
            cats.draw_cat(L, rng, cx, cy, s, *spec)
        else:
            CATEGORIES[c][0](L, rng, cx, cy, s)
        obj[L != CLEAR] = L[L != CLEAR]
    if first is None:
        return np.where(obj == CLEAR, cv, obj).astype(np.uint8)
    return compose(cv, obj, rng, first)


def compose(cv, obj, rng, bgs):
    """Pick a background (preferring `bgs`) whose dominant colour differs from the object's, then outline."""
    vals = obj[obj != CLEAR]
    dom = np.bincount(vals, minlength=8).argmax() if vals.size else -1
    for k in range(8):
        background(cv, rng, pk(rng, bgs) if k < 4 and bgs[0] else pk(rng, GENERIC_BGS))
        if np.bincount(cv.ravel(), minlength=8).argmax() != dom:
            break
    return outline(cv, obj)
