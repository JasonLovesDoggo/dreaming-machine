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
CAT_MODIFIERS = list(cats.MODIFIERS)
COLOR_WORDS = dict(cats.COLOR_WORDS)
IMAGE_EXTS = ('.PNG', '.JPG', '.JPEG', '.BMP', '.GIF')
GENERIC_BGS = ('sky', 'sky', 'plain', 'gray', 'yellow', 'grass')


def is_image(name):
    return name.upper().endswith(IMAGE_EXTS)


_GAP = ' _-'


def _stem(name):
    """Uppercase name without its extension (RED CAT and RED CAT.PNG parse the same)."""
    base = name.upper()
    head, dot, ext = base.rpartition('.')
    return head if dot and 0 < len(ext) <= 4 and ext.isalnum() else base


def _skip_gap(base, i, step):
    """Index past separators (spaces, _, -, AND) going left (step=-1, i = end) or right (step=1, i = start)."""
    while True:
        j = i
        while (0 < j if step < 0 else j < len(base)) and base[j - 1 if step < 0 else j] in _GAP:
            j += step
        if step < 0 and base[:j].endswith('AND') and j - 3 < i - 1:
            j -= 3
        elif step > 0 and base[j:].startswith('AND') and j > i:
            j += 3
        if j == i:
            return i
        i = j


def _cat_modifiers(base, lo, hi, taken):
    """Grow the cat word group [lo, hi) with adjacent modifiers (left: any, right: those that aren't subjects)."""
    mods = sorted(cats.MODIFIERS, key=lambda k: (-len(k), k))
    grabbed = []
    while True:
        j = _skip_gap(base, lo, -1)
        m = next((w for w in mods if base[:j].endswith(w) and not any(a < j and b > j - len(w) for a, b in taken)), None)
        if m is None:
            break
        lo = j - len(m)
        grabbed.append((lo, m))
    while True:
        j = _skip_gap(base, hi, 1)
        m = next((w for w in mods if base[j:].startswith(w) and w not in KEYWORDS
                  and not any(a < j + len(w) and b > j for a, b in taken)), None)
        if m is None:
            break
        grabbed.append((j, m))
        hi = j + len(m)
    return lo, hi, grabbed


def parse_words(name):
    """[(keywords, category)] in filename order, longest keywords matched first, at most 3 subjects.

    All cat words merge into one cat, and colour/pattern/size/pose words next to a cat word modify it
    (ORANGE CAT is an orange cat, not a fruit and a cat)."""
    base = _stem(name)
    work, found = base, []
    for kw in sorted(KEYWORDS, key=lambda k: (-len(k), k)):
        while kw in work:
            i = work.index(kw)
            found.append((i, i + len(kw), kw))
            work = work[:i] + ' ' * len(kw) + work[i + len(kw):]
    found.sort()
    cat_spans = [(a, b) for a, b, kw in found if KEYWORDS[kw] == 'cat']
    mods = []
    for a, b in cat_spans:
        taken = [(x, y) for x, y in cat_spans if (x, y) != (a, b)]
        lo, hi, got = _cat_modifiers(base, a, b, taken)
        mods += got
        found = [(x, y, kw) for x, y, kw in found if KEYWORDS[kw] == 'cat' or y <= lo or x >= hi]
    out, cat_i = [], None
    for _, _, kw in found:
        c = KEYWORDS[kw]
        if c == 'cat' and cat_i is not None:
            out[cat_i][0].append(kw)
            continue
        if c == 'cat':
            cat_i = len(out)
        out.append(([kw], c))
    if cat_i is not None:
        out[cat_i] = ([m for _, m in sorted(mods)] + out[cat_i][0], 'cat')
    return out[:3]


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
    specs = [cats.random_spec(rng, cats.cond_from_words(kws)) if c == 'cat' else None for kws, c in subs]
    if len(subs) == 1 and subs[0][1] == 'cat':
        return cats.render(rng, specs[0])
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
