"""Calculator: a small orange-and-black integer calculator (+ - * / only, 8 digits, ERR on overflow or x/0).

Logic (`Calc`) is pure and cheap to copy so data drivers can look ahead. HELD-OUT RULE: training data must never
contain a multiplication by 24 (24*x, x*24, or a chained result equal to 24 used with *) so the model's handling of
x24 is a generalization test. `forbidden()` defines the rule, `guard_action()` filters any action that would break it.
"""
from functools import lru_cache

W, H = 12, 14  # window size in tiles
MAX_DIGITS = 8
LIMIT = 10 ** MAX_DIGITS - 1
LCD_COL, LCD_ROW, LCD_LEN = 2, 2, 9  # window-local; column 1 of the LCD shows the pending operator
BTN_COL, BTN_ROW = 2, 5  # 4x4 keypad of 2x2-tile buttons
BUTTONS = ['7', '8', '9', '/', '4', '5', '6', '*', '1', '2', '3', '-', 'C', '0', '=', '+']
OPS = '+-*/'
HELD_OUT = 24
FLASH = 3  # a pressed button is drawn highlighted for FLASH - 1 frames


class Calc:
    __slots__ = ('acc', 'op', 'cur', 'err', 'flash')

    def __init__(self):
        self.clear()
        self.flash = None

    def clear(self):
        self.acc, self.op, self.cur, self.err = None, None, None, False

    def copy(self):
        c = Calc.__new__(Calc)
        c.acc, c.op, c.cur, c.err, c.flash = self.acc, self.op, self.cur, self.err, self.flash
        return c

    def display(self):
        if self.err:
            return 'ERR'
        if self.cur is not None:
            return self.cur
        return str(self.acc if self.acc is not None else 0)

    def tick(self):
        if self.flash:
            i, n = self.flash
            self.flash = (i, n - 1) if n > 1 else None

    def press(self, tok):
        """Apply one input token ('0'-'9', + - * /, '=', 'C', '<' backspace); returns an event name."""
        if tok in BUTTONS:
            self.flash = (BUTTONS.index(tok), FLASH)
        if tok == 'C':
            self.clear()
            return 'calc_clear'
        if self.err:
            if not tok.isdigit():
                if tok == '<':
                    self.clear()
                    return 'calc_clear'
                return 'calc_noop'
            self.clear()
        if tok.isdigit():
            if self.op is None and self.cur is None:
                self.acc = None  # typing after '=' starts a new calculation
            if self.cur in (None, '0'):
                self.cur = tok
            elif len(self.cur) < MAX_DIGITS:
                self.cur += tok
            else:
                return 'calc_full'
            return 'calc_digit'
        if tok == '<':
            if self.cur is None:
                return 'calc_noop'
            self.cur = self.cur[:-1] or '0'
            return 'calc_del'
        if tok in OPS:
            if self.cur is not None:
                b, self.cur = int(self.cur), None
                if self.op is not None and self.acc is not None:
                    self._apply(b)
                else:
                    self.acc = b
            elif self.acc is None:
                self.acc = 0
            if self.err:
                return 'calc_err'
            self.op = tok
            return 'calc_op'
        if tok == '=':
            if self.op is None:
                if self.cur is not None:
                    self.acc, self.cur = int(self.cur), None
                return 'calc_eq'
            b = int(self.cur) if self.cur is not None else self.acc
            self.cur = None
            self._apply(b)
            self.op = None
            return 'calc_err' if self.err else 'calc_eq'
        return 'calc_noop'

    def _apply(self, b):
        a, op = self.acc, self.op
        if op == '+':
            r = a + b
        elif op == '-':
            r = a - b
        elif op == '*':
            r = a * b
        elif b == 0:
            r = None
        else:
            r = abs(a) // abs(b) * (1 if (a < 0) == (b < 0) else -1)  # integer division, truncated toward zero
        if r is None or abs(r) > LIMIT:
            self.acc, self.op, self.err = None, None, True
        else:
            self.acc = r

    def state_fields(self):
        flash = self.flash[0] + 1 if self.flash else 0
        return [OPS.index(self.op) + 1 if self.op else 0, int(self.err), len(self.display()), flash]


# ---------------------------------------------------------------- held-out x24 rule
def forbidden(c):
    """True if the calculator is about to multiply by 24 (either operand), the state the data must never reach."""
    return c.op == '*' and ((c.acc is not None and abs(c.acc) == HELD_OUT) or c.cur == str(HELD_OUT))


def violates(c, tok):
    d = c.copy()
    d.press(tok)
    return forbidden(d)


def button_token(lc, lr):
    if BTN_COL <= lc < BTN_COL + 8 and BTN_ROW <= lr < BTN_ROW + 8:
        return BUTTONS[(lr - BTN_ROW) // 2 * 4 + (lc - BTN_COL) // 2]
    return None


def button_cell(tok):
    i = BUTTONS.index(tok)
    return BTN_COL + 2 * (i % 4), BTN_ROW + 2 * (i // 4)


@lru_cache(maxsize=None)
def _key_map():
    from ..os_sim import KEY_ENTER, KEY_DEL, KEY_CHAR0, KEY_PLUS, KEY_TIMES, KEY_EQUALS
    from ..tiles import CHAR_IDX
    m = {KEY_CHAR0 + CHAR_IDX[ch]: ch for ch in '0123456789-/C'}
    m[KEY_CHAR0 + CHAR_IDX['X']] = '*'
    m.update({KEY_PLUS: '+', KEY_TIMES: '*', KEY_EQUALS: '=', KEY_ENTER: '=', KEY_DEL: '<'})
    return m


def key_token(key):
    return _key_map().get(key)


def token_keys(tok):
    """All keyboard keys that produce `tok`."""
    return [k for k, t in sorted(_key_map().items()) if t == tok]


def predict_tokens(os_, action):
    """Calculator tokens `os_.step(action)` could deliver (over-approximates when focus might change mid-step)."""
    from ..tiles import T, GH, SCREEN_W, SCREEN_H
    if 'calculator' not in os_.z:
        return []  # a calculator opened by this action starts cleared
    mx, my, btn, key = (int(v) for v in action)
    mx, my = min(max(mx, 0), SCREEN_W - 1), min(max(my, 0), SCREEN_H - 1)
    press = bool(btn) and not os_.btn
    toks = []
    if press:
        c, r = mx // T, my // T
        if r < GH - 1 and os_._window_at(c, r) == 'calculator':
            wc, wr = os_.pos['calculator']
            tok = button_token(c - wc, r - wr)
            if tok:
                toks.append(tok)
    if key and (os_.focus == 'calculator' or press):
        tok = key_token(key)
        if tok:
            toks.append(tok)
    return toks


def _would_violate(os_, action):
    c = os_.calc.copy()
    for tok in predict_tokens(os_, action):
        c.press(tok)
        if forbidden(c):
            return True
    return False


def guard_action(os_, action):
    """Drop the key and/or mouse press of an action that would make the calculator multiply by 24."""
    if not _would_violate(os_, action):
        return action
    x, y, btn, key = action
    no_key = (x, y, btn, 0)
    if not _would_violate(os_, no_key):
        return no_key
    return (x, y, 0, 0)  # a press needs os_.btn == 0, so holding the button up is a no-op


# ---------------------------------------------------------------- tiles + render
FONT57 = {
    '0': [".###.", "#...#", "#..##", "#.#.#", "##..#", "#...#", ".###."],
    '1': ["..#..", ".##..", "..#..", "..#..", "..#..", "..#..", ".###."],
    '2': [".###.", "#...#", "....#", "...#.", "..#..", ".#...", "#####"],
    '3': ["#####", "...#.", "..#..", "...#.", "....#", "#...#", ".###."],
    '4': ["...#.", "..##.", ".#.#.", "#..#.", "#####", "...#.", "...#."],
    '5': ["#####", "#....", "####.", "....#", "....#", "#...#", ".###."],
    '6': ["..##.", ".#...", "#....", "####.", "#...#", "#...#", ".###."],
    '7': ["#####", "....#", "...#.", "..#..", ".#...", ".#...", ".#..."],
    '8': [".###.", "#...#", "#...#", ".###.", "#...#", "#...#", ".###."],
    '9': [".###.", "#...#", "#...#", ".####", "....#", "...#.", ".##.."],
    '+': [".....", "..#..", "..#..", "#####", "..#..", "..#..", "....."],
    '-': [".....", ".....", ".....", "#####", ".....", ".....", "....."],
    '*': [".....", "#...#", ".#.#.", "..#..", ".#.#.", "#...#", "....."],
    '/': [".....", "..#..", ".....", "#####", ".....", "..#..", "....."],
    '=': [".....", ".....", "#####", ".....", "#####", ".....", "....."],
    'C': [".###.", "#...#", "#....", "#....", "#....", "#...#", ".###."],
    'E': ["#####", "#....", "#....", "####.", "#....", "#....", "#####"],
    'R': ["####.", "#...#", "#...#", "####.", "#.#..", "#..#.", "#...#"],
    ' ': ["....."] * 7,
}
LCD_CHARS = ' 0123456789-ER'


def _glyph(ch, bold=True):
    import numpy as np
    g = np.array([[c == '#' for c in row] for row in FONT57[ch]])
    out = np.zeros((7, 6), bool)
    out[:, :5] |= g
    if bold:
        out[:, 1:] |= g
    return out


def _split(px):
    """(16, 16) or (16, 8) pixels -> tiles in row-major order."""
    return [px[y:y + 8, x:x + 8] for y in range(0, px.shape[0], 8) for x in range(0, px.shape[1], 8)]


def add_tiles(A):
    import numpy as np
    from ..tiles import K, DGRAY, ORANGE, YELLOW, WHITE
    lcd_bg = DGRAY
    for ch in LCD_CHARS:  # big LCD digits: bold 6x7 glyph stretched to 6x14 over two tiles
        px = np.full((16, 8), lcd_bg, np.uint8)
        px[1:15, 1:7][np.repeat(_glyph(ch), 2, 0)] = ORANGE
        for half, t in enumerate(_split(px)):
            A.add(('calc', 'lcd', ch, half), t)
    for op in ' ' + OPS:
        px = np.full((16, 8), lcd_bg, np.uint8)
        if op != ' ':
            px[5:12, 1:7][_glyph(op)] = YELLOW
        for half, t in enumerate(_split(px)):
            A.add(('calc', 'op', op, half), t)
    yy, xx = np.mgrid[0:16, 0:16]
    face_mask = (yy >= 1) & (yy <= 14) & (xx >= 1) & (xx <= 14)
    face_mask &= ~(((yy == 1) | (yy == 14)) & ((xx == 1) | (xx == 14)))
    for i, tok in enumerate(BUTTONS):
        digit = tok.isdigit()
        for pressed in (False, True):
            if digit:
                face, fg = (ORANGE, K) if pressed else (DGRAY, ORANGE)
            else:
                face, fg = (WHITE, ORANGE) if pressed else (ORANGE, K)
            px = np.full((16, 16), K, np.uint8)
            px[face_mask] = face
            px[4:11, 5:11][_glyph(tok)] = fg
            for q, t in enumerate(_split(px)):
                A.add(('calc', 'btn', i, pressed, q), t)


def render(g, c, wc, wr):
    from ..tiles import ATLAS
    op = c.op or ' '
    for half in (0, 1):
        g[wr + LCD_ROW + half, wc + LCD_COL - 1] = ATLAS[('calc', 'op', op, half)]
        for i, ch in enumerate(c.display().rjust(LCD_LEN)):
            g[wr + LCD_ROW + half, wc + LCD_COL + i] = ATLAS[('calc', 'lcd', ch, half)]
    flash = c.flash[0] if c.flash else -1
    for i in range(len(BUTTONS)):
        bc, br = BTN_COL + 2 * (i % 4), BTN_ROW + 2 * (i // 4)
        for q in range(4):
            g[wr + br + q // 2, wc + bc + q % 2] = ATLAS[('calc', 'btn', i, i == flash, q)]
