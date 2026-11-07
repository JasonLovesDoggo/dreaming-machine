"""Fixed tile atlas: every 8x8 tile the OS can ever draw, with stable integer IDs."""
import numpy as np

T = 8
GW, GH = 32, 24
SCREEN_W, SCREEN_H = GW * T, GH * T

PAL = np.array([
    (0, 0, 0), (0, 0, 128), (128, 0, 0), (0, 128, 128), (139, 90, 43), (80, 80, 80),
    (192, 192, 192), (255, 255, 255), (255, 0, 64), (255, 153, 0), (255, 224, 32),
    (32, 192, 64), (48, 160, 255), (224, 176, 128), (32, 96, 48), (128, 48, 128)], np.uint8)
(K, NAVY, DRED, TEAL, BROWN, DGRAY, LGRAY, WHITE, RED, ORANGE, YELLOW, GREEN, BLUE, TAN, DGREEN,
 PURPLE) = range(16)
CANVAS_COLORS = [WHITE, K, RED, BLUE, GREEN, YELLOW, BROWN, LGRAY]

FONT3 = {
    'A': [".#.", "#.#", "###", "#.#", "#.#"], 'B': ["##.", "#.#", "##.", "#.#", "##."],
    'C': [".##", "#..", "#..", "#..", ".##"], 'D': ["##.", "#.#", "#.#", "#.#", "##."],
    'E': ["###", "#..", "##.", "#..", "###"], 'F': ["###", "#..", "##.", "#..", "#.."],
    'G': [".##", "#..", "#.#", "#.#", ".##"], 'H': ["#.#", "#.#", "###", "#.#", "#.#"],
    'I': ["###", ".#.", ".#.", ".#.", "###"], 'J': ["..#", "..#", "..#", "#.#", ".#."],
    'K': ["#.#", "#.#", "##.", "#.#", "#.#"], 'L': ["#..", "#..", "#..", "#..", "###"],
    'M': ["#.#", "###", "###", "#.#", "#.#"], 'N': ["###", "#.#", "#.#", "#.#", "#.#"],
    'O': ["###", "#.#", "#.#", "#.#", "###"], 'P': ["##.", "#.#", "##.", "#..", "#.."],
    'Q': [".#.", "#.#", "#.#", "##.", ".##"], 'R': ["##.", "#.#", "##.", "#.#", "#.#"],
    'S': [".##", "#..", ".#.", "..#", "##."], 'T': ["###", ".#.", ".#.", ".#.", ".#."],
    'U': ["#.#", "#.#", "#.#", "#.#", "###"], 'V': ["#.#", "#.#", "#.#", "#.#", ".#."],
    'W': ["#.#", "#.#", "###", "###", "#.#"], 'X': ["#.#", "#.#", ".#.", "#.#", "#.#"],
    'Y': ["#.#", "#.#", ".#.", ".#.", ".#."], 'Z': ["###", "..#", ".#.", "#..", "###"],
    '0': [".#.", "#.#", "#.#", "#.#", ".#."], '1': [".#.", "##.", ".#.", ".#.", "###"],
    '2': ["##.", "..#", ".#.", "#..", "###"], '3': ["##.", "..#", ".#.", "..#", "##."],
    '4': ["#.#", "#.#", "###", "..#", "..#"], '5': ["###", "#..", "##.", "..#", "##."],
    '6': [".##", "#..", "###", "#.#", "###"], '7': ["###", "..#", ".#.", ".#.", ".#."],
    '8': ["###", "#.#", "###", "#.#", "###"], '9': ["###", "#.#", "###", "..#", "##."],
    '.': ["...", "...", "...", "...", ".#."], ':': ["...", ".#.", "...", ".#.", "..."],
    '-': ["...", "...", "###", "...", "..."], '/': ["..#", "..#", ".#.", "#..", "#.."],
    '>': ["#..", ".#.", "..#", ".#.", "#.."], '!': [".#.", ".#.", ".#.", "...", ".#."],
    ' ': ["...", "...", "...", "...", "..."], '_': ["...", "...", "...", "...", "###"],
}
CHARSET = " ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.:-/>!_"
CHAR_IDX = {c: i for i, c in enumerate(CHARSET)}
NCH = len(CHARSET)
GLYPHS = np.array([[[c == '#' for c in row] for row in FONT3[ch]] for ch in CHARSET], bool)

# text contexts: (fg, bg)
TEXT_CTX = {
    'title': (WHITE, NAVY), 'title_in': (LGRAY, DGRAY), 'body': (K, WHITE), 'sel': (WHITE, BLUE),
    'tool': (K, LGRAY), 'desk': (WHITE, TEAL), 'hud': (RED, K),
}
CTX_NAMES = list(TEXT_CTX)

SUITS = "hdcs"  # hearts, diamonds (red); clubs, spades (black)
RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
SUIT_ART = {
    'h': [".#.#.", "#####", "#####", ".###.", "..#.."],
    'd': ["..#..", ".###.", "#####", ".###.", "..#.."],
    'c': [".###.", ".###.", "#####", "#####", "..#.."],
    's': ["..#..", ".###.", "#####", "#####", "..#.."],
}

ICON_ART = {
    'folder': ["................", "................", ".ooooo..........", ".oyyyyo.........", ".oyyyyyoooooooo.",
               ".oyyyyyyyyyyyyo.", ".oooooooooooooo."] + [".oyyyyyyyyyyyyo."] * 6 + [".oooooooooooooo.", "................", "................"],
    'trash': ["................", "......dddd......", "..dddddddddddd..", "..dggggggggggd..", "..dddddddddddd.."]
             + ["...dggdggdggd..."] * 9 + ["...dddddddddd...", "................"],
    'trash_full': ["................", "......dddd......", "..dddddddddddd..", "..dggggggggggd..", "..dddwwwwddddd..",
                   "...dgwkkwdggd...", "...dgwwwwggdd..."] + ["...dggdggdggd..."] * 7 + ["...dddddddddd...", "................"],
    'picture': ["................", ".kkkkkkkkkkkkkk.", ".kssssssssssssk.", ".kssssssssyyssk.", ".kssssssssyyssk.",
                ".kssssssssssssk.", ".ksssGGsssssssk.", ".kssGGGGssssssk.", ".ksGGGGGGsssGGk.", ".kGGGGGGGGsGGGk.",
                ".kGGGGGGGGGGGGk.", ".kGGGGGGGGGGGGk.", ".kkkkkkkkkkkkkk.", "................", "................", "................"],
    'paint': ["................", "................", "....oooooooo....", "..oooooooooooo..", ".ooorroooyyoooo.",
              ".ooorroooyyoooo.", ".oooooooooooooo.", ".obbooo..oooooo.", ".obbooo..oGGooo.", ".oooooo..oGGooo.",
              ".oooooooooooooo.", "..oooooooooooo..", "....oooooooo....", "................", "................", "................"],
    'cards': ["................", "...kkkkkkkkkk..."] + ["...k" + r + "k..." for r in
              ["wwwwwwww", "wrwwwwww", "wwwwwwww", "wwrrwrrw", "wrrrrrrr", "wrrrrrrr", "wwrrrrrw", "wwwrrrww", "wwwwrwww",
               "wwwwwwww", "wwwwwwrw", "wwwwwwww"]] + ["...kkkkkkkkkk...", "................"],
    'notepad': ["................", "..kkkkkkkkkkk...", "..kwwwwwwwwwkk..", "..kwddddddwwkwk.", "..kwwwwwwwwwkkkk",
                "..kwdddddddwwwk.", "..kwwwwwwwwwwwk.", "..kwddddddddwwk.", "..kwwwwwwwwwwwk.", "..kwdddddwwwwwk.",
                "..kwwwwwwwwwwwk.", "..kwddddddddwwk.", "..kwwwwwwwwwwwk.", "..kkkkkkkkkkkkk.", "................", "................"],
    'doom': ["................", ".rrrrrrrrrrrrrr.", ".rrrrwwwwwwrrrr.", ".rrrwwwwwwwwrrr.", ".rrwwwwwwwwwwrr.",
             ".rrwwkkwwkkwwrr.", ".rrwwkkwwkkwwrr.", ".rrwwwwwwwwwwrr.", ".rrrwwwkkwwwrrr.", ".rrrrwwwwwwrrrr.",
             ".rrrrwkwkwkrrrr.", ".rrrrwwwwwwrrrr.", ".rrrrrrrrrrrrrr.", "................", "................", "................"],
}
ART_COLORS = {'o': TAN, 'y': YELLOW, 'd': DGRAY, 'g': LGRAY, 'k': K, 's': BLUE, 'G': GREEN, 'r': RED, 'b': BLUE,
              'w': WHITE}
MINI_ART = {
    'folder': ["........", ".ooo....", ".oyyoooo", ".yyyyyyo", ".yyyyyyo", ".yyyyyyo", ".oooooo.", "........"],
    'file': ["........", ".kkkk...", ".kwwwk..", ".kwwwwk.", ".kwwwwk.", ".kwwwwk.", ".kkkkkk.", "........"],
    'png': ["........", ".kkkkkk.", ".kssysk.", ".ksssGk.", ".kGsGGk.", ".kGGGGk.", ".kkkkkk.", "........"],
    'exe': ["........", ".kkkkkk.", ".krrrrk.", ".kkkkkk.", ".kwwwwk.", ".kwwwwk.", ".kkkkkk.", "........"],
}


class Atlas:
    def __init__(self):
        self.tiles = []
        self.ids = {}
        self.ranges = {}

    def add(self, key, arr):
        assert key not in self.ids, key
        self.ids[key] = len(self.tiles)
        self.tiles.append(np.asarray(arr, np.uint8))
        return self.ids[key]

    def add_block(self, name, arrs):
        start = len(self.tiles)
        self.tiles.extend(list(arrs))
        self.ranges[name] = (start, len(self.tiles))
        return start

    def __getitem__(self, key):
        return self.ids[key]


def solid(c):
    return np.full((T, T), c, np.uint8)


def with_edges(t, sides, line=DGRAY):
    t = t.copy()
    if 'l' in sides: t[:, 0] = line
    if 'r' in sides: t[:, -1] = line
    if 'b' in sides: t[-1, :] = line
    if 't' in sides: t[0, :] = line
    return t


def art_tile(art, bg, oy=0, ox=0, colors=None):
    colors = colors or ART_COLORS
    t = solid(bg)
    for y in range(T):
        for x in range(T):
            c = art[oy + y][ox + x]
            if c != '.':
                t[y, x] = colors[c]
    return t


def draw_mask(t, mask, color, oy, ox):
    h, w = mask.shape
    sub = t[oy:oy + h, ox:ox + w]
    sub[mask] = color


def build_atlas():
    A = Atlas()
    for name, c in [('desk', TEAL), ('body', WHITE), ('title', NAVY), ('title_in', DGRAY), ('tool', LGRAY),
                    ('felt', DGREEN), ('hud', K), ('sel', BLUE)]:
        A.add(('solid', name), solid(c))
    A.add(('doom_view',), solid(K))
    for bg in ['body', 'tool', 'felt', 'hud']:
        base = A.tiles[A[('solid', bg)]]
        for sides in ['l', 'r', 'b', 'lb', 'rb']:
            A.add(('edge', bg, sides), with_edges(base, sides))
    for active in (True, False):
        t = solid(RED if active else DGRAY)
        draw_mask(t, GLYPHS[CHAR_IDX['X']], WHITE, 1, 2)
        A.add(('close', active), t)
    for name, art in ICON_ART.items():
        for q in range(4):
            qy, qx = divmod(q, 2)
            A.add(('icon', name, q), art_tile(art, TEAL, qy * T, qx * T))
    for name, art in MINI_ART.items():
        for sel in (False, True):
            A.add(('mini', name, sel), art_tile(art, BLUE if sel else WHITE))
    for i, c in enumerate(CANVAS_COLORS):
        for sel in (False, True):
            t = solid(LGRAY)
            t[1:7, 1:7] = WHITE if sel else K
            t[2:6, 2:6] = c
            if sel:
                t[0, :] = K; t[-1, :] = K; t[:, 0] = K; t[:, -1] = K
            A.add(('swatch', i, sel), t)
    # cards
    for sel in (False, True):
        border = YELLOW if sel else K
        def face(sides):
            return with_edges(solid(WHITE), sides, border)
        for r, rank in enumerate(RANKS):
            for red in (False, True):
                t = face('tl')
                fg = RED if red else K
                if len(rank) == 1:
                    draw_mask(t, GLYPHS[CHAR_IDX[rank]], fg, 2, 2)
                else:
                    draw_mask(t, GLYPHS[CHAR_IDX['1']][:, 1:], fg, 2, 1)
                    draw_mask(t, GLYPHS[CHAR_IDX['0']], fg, 2, 4)
                A.add(('card', 'tl', rank, red, sel), t)
        for s in SUITS:
            mask = np.array([[c == '#' for c in row] for row in SUIT_ART[s]])
            fg = RED if s in 'hd' else K
            t = face('tr'); draw_mask(t, mask, fg, 2, 1)
            A.add(('card', 'tr', s, sel), t)
            t = face('br'); draw_mask(t, mask, fg, 1, 1)
            A.add(('card', 'br', s, sel), t)
        for sides in ['ml', 'mr', 'bl']:
            A.add(('card', sides, sel), face({'ml': 'l', 'mr': 'r', 'bl': 'lb'}[sides]))
    back = solid(BLUE)
    yy, xx = np.mgrid[0:T, 0:T]
    back[((xx + yy) % 4 == 0) | ((xx - yy) % 4 == 0)] = NAVY
    for sides in ['tl', 'tr', 'ml', 'mr', 'bl', 'br']:
        A.add(('back', sides), with_edges(back, sides.replace('m', ''), K))
    for sides in ['tl', 'tr', 'ml', 'mr', 'bl', 'br']:
        A.add(('slot', sides), with_edges(solid(DGREEN), sides.replace('m', ''), GREEN))
    n_static = len(A.tiles)
    # text: char pairs per context
    for ctx, (fg, bg) in TEXT_CTX.items():
        arr = np.full((NCH, NCH, T, T), bg, np.uint8)
        ga = np.where(GLYPHS[:, None], fg, bg)
        gb = np.where(GLYPHS[None, :], fg, bg)
        arr[:, :, 1:6, 0:3] = ga
        arr[:, :, 1:6, 4:7] = gb
        A.add_block(('text', ctx), arr.reshape(-1, T, T))
    # paint canvas: 2x2 cells of 4x4 px, 8 colors
    codes = np.arange(8 ** 4)
    cv = np.zeros((len(codes), T, T), np.uint8)
    pal = np.array(CANVAS_COLORS, np.uint8)
    for i, (dy, dx) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
        cv[:, dy * 4:dy * 4 + 4, dx * 4:dx * 4 + 4] = pal[(codes >> (3 * i)) & 7][:, None, None]
    A.add_block(('canvas',), cv)
    A.n_static = n_static
    A.array = np.stack(A.tiles)
    return A


ATLAS = build_atlas()
VOCAB = len(ATLAS.tiles)
TEXT_START = {ctx: ATLAS.ranges[('text', ctx)][0] for ctx in CTX_NAMES}
CANVAS_START = ATLAS.ranges[('canvas',)][0]


def text_id(ctx, a, b):
    return TEXT_START[ctx] + CHAR_IDX.get(a, 0) * NCH + CHAR_IDX.get(b, 0)


def canvas_id(c00, c01, c10, c11):
    return CANVAS_START + (c00 | (c01 << 3) | (c10 << 6) | (c11 << 9))


def grid_to_pixels(grid):
    """(24,32) tile ids -> (192,256) palette indices."""
    px = ATLAS.array[grid]  # (24,32,8,8)
    return px.transpose(0, 2, 1, 3).reshape(grid.shape[0] * T, grid.shape[1] * T)


CURSOR_ART = ["k.......", "kk......", "kwk.....", "kwwk....", "kwwwk...", "kwwwwk..", "kwwwwwk.", "kwwkkkkk",
              "kk.kwk..", "k...kwk.", "....kk.."]


def draw_cursor(px, x, y):
    for dy, row in enumerate(CURSOR_ART):
        for dx, ch in enumerate(row):
            if ch != '.' and 0 <= y + dy < px.shape[0] and 0 <= x + dx < px.shape[1]:
                px[y + dy, x + dx] = ART_COLORS[ch]
    return px
