"""Deterministic tile-based micro OS. step(action) -> observation (tile grid + cursor + doom tokens + state)."""
import random
from functools import lru_cache
import numpy as np
from .tiles import (ATLAS, GW, GH, T, SCREEN_W, SCREEN_H, text_id, canvas_id, grid_to_pixels, draw_cursor, RANKS,
                    NCH, CHARSET)
from . import doom as dm
from . import pictures as pics
from .apps import calculator as calc_app
from .apps.terminal import Terminal

APP_SIZES = {'explorer': (16, 12), 'paint': (18, 13), 'cards': (23, 20), 'doom': (22, 18), 'notepad': (18, 13),
             'calculator': (calc_app.W, calc_app.H)}
APP_SIZES['terminal'] = (20, 14)
APPS = list(APP_SIZES)
# Fixed OS-state slots: existing apps keep their slot forever; new apps take a reserved one (never reorder).
APP_SLOTS = ['explorer', 'paint', 'cards', 'doom', 'notepad', 'calculator', 'terminal', 'dreamer', 'flight', 'spare']
assert all(a in APP_SLOTS for a in APPS)
FOCUS_COL = 3 * len(APP_SLOTS)
EXT_FIELDS, EXT_VOCAB = 16, 32  # spare per-app state fields at the end of the state vector
APP_TITLES = {'explorer': 'FILES', 'paint': 'PAINT', 'cards': 'CARDS', 'doom': 'DOOM', 'notepad': 'NOTEPAD', 'calculator': 'CALC'}
DEFAULT_POS = {'explorer': (12, 2), 'paint': (11, 4), 'cards': (8, 1), 'doom': (9, 3), 'notepad': (7, 6), 'calculator': (18, 3)}
APP_TITLES['terminal'] = 'TERMINAL'
DEFAULT_POS['terminal'] = (12, 5)
KEY_NONE, KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT, KEY_FIRE, KEY_ENTER, KEY_ESC, KEY_DEL = range(9)
KEY_CHAR0 = 9  # typed characters: KEY_CHAR0 + index into CHARSET (space = 9)
KEY_EXTRA0 = KEY_CHAR0 + NCH  # typed symbols with no font glyph (append only): KEY_EXTRA0 + index into EXTRA_KEYS
EXTRA_KEYS = '+*='
KEY_PLUS, KEY_TIMES, KEY_EQUALS = (KEY_EXTRA0 + EXTRA_KEYS.index(ch) for ch in '+*=')
NKEYS = KEY_EXTRA0 + len(EXTRA_KEYS)
NP_COLS, NP_ROWS, NP_MAX = 32, 10, 600
DESK_ICONS = [('folder', 'FILES', ('explorer', 'C:')), ('trash', 'TRASH', ('explorer', 'TRASH')),
              ('picture', 'PICS', ('explorer', 'C:/PICS')), ('paint', 'PAINT', ('paint', None)),
              ('cards', 'CARDS', ('cards', None)), ('doom', 'DOOM', ('doom', None)),
              ('notepad', 'NOTES', ('notepad', None)), ('calc', 'CALC', ('calculator', None))]
PATHS = ['C:', 'C:/PICS', 'C:/DOCS', 'C:/GAMES', 'TRASH']
DESK_ICONS.append(('terminal', 'TERM', ('terminal', None)))
CANVAS_H, CANVAS_W = 20, 32
DCLICK_STEPS = 8
TASKBAR_BTN_W = 5


def icon_pos(i):
    return 1 + 4 * (i % 3), 1 + 4 * (i // 3)  # (row, col)


def make_picture(name):
    rs = np.random.RandomState(sum(ord(c) * (i + 1) for i, c in enumerate(name)) % (2 ** 31))
    cv = np.zeros((CANVAS_H, CANVAS_W), np.uint8)
    cv[:] = rs.choice([0, 3])
    cv[14 + rs.randint(0, 3):] = rs.choice([4, 6])
    for _ in range(rs.randint(2, 5)):
        c = rs.randint(1, 8)
        if rs.rand() < 0.5:
            y, x = rs.randint(0, CANVAS_H - 4), rs.randint(0, CANVAS_W - 4)
            cv[y:y + rs.randint(2, 8), x:x + rs.randint(2, 10)] = c
        else:
            yy, xx = np.mgrid[0:CANVAS_H, 0:CANVAS_W]
            cy, cx, r = rs.randint(2, 16), rs.randint(2, 30), rs.randint(2, 5)
            cv[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = c
    return cv


def line_cells(x0, y0, x1, y1):
    pts = []
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
    err = dx + dy
    while True:
        pts.append((x0, y0))
        if x0 == x1 and y0 == y1:
            return pts
        e2 = 2 * err
        if e2 >= dy: err += dy; x0 += sx
        if e2 <= dx: err += dx; y0 += sy


class MicroOS:
    def __init__(self, seed=0):
        self.seed = seed
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self, seed=None):
        if seed is not None:
            self.seed = seed
        self.pic_rng = np.random.RandomState(self.seed % (2 ** 31))
        self.t = 0
        self.mx, self.my, self.btn = SCREEN_W // 2, SCREEN_H // 2, 0
        self.last_press_t, self.last_press_xy = -100, (0, 0)
        self.z = []
        self.pos = {}
        self.drag = None
        self.desk_sel = -1
        self.fs = {
            'C:': [('PICS', 'folder'), ('DOCS', 'folder'), ('GAMES', 'folder'), ('README.TXT', 'txt')],
            'C:/PICS': [('CAT.PNG', 'png'), ('HOUSE.PNG', 'png'), ('SUNSET.PNG', 'png'), ('DOG.PNG', 'png')],
            'C:/DOCS': [('NOTES.TXT', 'txt'), ('TODO.TXT', 'txt'), ('BUDGET.TXT', 'txt')],
            'C:/GAMES': [('DOOM.EXE', 'exe'), ('CARDS.EXE', 'exe'), ('PAINT.EXE', 'exe')],
            'TRASH': [('OLD.TXT', 'txt')],
        }
        self.origin = {'OLD.TXT': 'C:/DOCS'}
        self.pictures = {n: pics.generate(n, self.pic_rng) for n, k in self.fs['C:/PICS']}
        self.new_pic = 1
        self.texts = {
            'README.TXT': 'WELCOME TO DREAMING MACHINE!\nDOUBLE CLICK AN ICON TO OPEN IT.\nDRAG WINDOWS BY THEIR TITLE BAR.',
            'NOTES.TXT': 'MEETING NOTES\n- SHIP THE WORLD MODEL\n- TRAIN ON MODAL\n- PLAY DOOM INSIDE A DREAM',
            'TODO.TXT': 'TODO:\n1. CLEAN DESKTOP\n2. WIN SOLITAIRE\n3. DRAW A CAT',
            'BUDGET.TXT': 'BUDGET 2026\nGPUS: 500\nSNACKS: 20\nTOTAL: 520',
            'OLD.TXT': 'THIS FILE IS OLD.\nYOU CAN EMPTY THE TRASH.',
        }
        self.new_doc = 1
        self.fs['C:/GAMES'] += [('STARS.BAS', 'txt'), ('COUNT.BAS', 'txt')]
        self.texts['STARS.BAS'] = ('10 FOR I 1 TO 9\n20 RPT I MUL 2 SUB 1 ! 9 SUB I\n30 NEXT I\n'
                                   '40 FOR I 8 TO 1 STEP -1\n50 RPT I MUL 2 SUB 1 ! 9 SUB I\n'
                                   '60 NEXT I\n70 END')
        self.texts['COUNT.BAS'] = '10 FOR I 1 TO 50\n20 PRINT I\n30 NEXT I\n40 END'
        self.term = Terminal(self)
        self.deal_no = 0
        self.doom = dm.Doom()
        self.calc = calc_app.Calc()
        self._reset_app('explorer'); self._reset_app('paint'); self._reset_app('cards'); self._reset_app('notepad')
        return self.observe([])

    def _reset_app(self, app, arg=None):
        if app == 'terminal':
            self.term = Terminal(self)
        elif app == 'explorer':
            self.ex_path, self.ex_sel = (arg or 'C:'), -1
            self.ex_naming = None
        elif app == 'paint':
            self.pt_canvas = np.zeros((CANVAS_H, CANVAS_W), np.uint8)
            self.pt_color, self.pt_file, self.pt_drawing, self.pt_last = 1, None, False, None
            self.pt_load = 0 if arg else 10
            if arg:
                if arg not in self.pictures:
                    self.pictures[arg] = pics.generate(arg, self.pic_rng)
                self.pt_canvas = self.pictures[arg].copy()
                self.pt_file = arg
        elif app == 'cards':
            rng = random.Random(self.deal_no)
            self.deal_no += 1
            deck = [(r, s) for s in range(4) for r in range(13)]
            rng.shuffle(deck)
            self.tab = []
            for i in range(7):
                col = [[deck.pop(), False] for _ in range(i + 1)]
                col[-1][1] = True
                self.tab.append(col)
            self.stock, self.waste, self.found = deck, [], [[] for _ in range(4)]
            self.c_sel, self.c_won = None, False
            self.c_press, self.c_grab, self.c_dragging = None, (0, 0), False
        elif app == 'notepad':
            self.np_file = arg
            self.np_text = self.texts.get(arg, '') if arg else ''
        elif app == 'doom':
            self.doom.reset()
        elif app == 'calculator':
            self.calc = calc_app.Calc()

    @property
    def focus(self):
        return self.z[-1] if self.z else None

    def open_app(self, app, arg=None, events=None):
        events = events if events is not None else []
        if app in self.z:
            self.raise_app(app)
            if app == 'explorer' and arg:
                self.ex_path, self.ex_sel = arg, -1
            if app in ('paint', 'notepad') and arg:
                self._reset_app(app, arg)
        else:
            self._reset_app(app, arg)
            self.pos[app] = DEFAULT_POS[app]
            self.z.append(app)
            events.append('open_' + app)
        return events

    def raise_app(self, app):
        self.z.remove(app)
        self.z.append(app)

    def close_app(self, app, events):
        if self.drag and self.drag[0] == app:
            self.drag = None
        if app == 'paint':
            self.pt_drawing = False
        self.z.remove(app)
        self.pos.pop(app, None)
        if app in ('paint', 'notepad', 'calculator', 'terminal'):
            self._reset_app(app)
        if app == 'cards':
            self.c_press, self.c_dragging = None, False
        events.append('close_' + app)

    # ------------------------------------------------------------------ input
    def step(self, action):
        mx, my, btn, key = (int(a) for a in action)
        mx, my = min(max(mx, 0), SCREEN_W - 1), min(max(my, 0), SCREEN_H - 1)
        events = []
        press, release = btn and not self.btn, (not btn) and self.btn
        self.mx, self.my = mx, my
        if self.drag and self.drag[0] not in self.z:
            self.drag = None
        if self.drag and btn:
            app, gx, gy = self.drag
            w, h = APP_SIZES[app]
            c = min(max(round((mx - gx) / T), 0), GW - w)
            r = min(max(round((my - gy) / T), 0), GH - 1 - h)
            if (c, r) != self.pos[app]:
                self.pos[app] = (c, r)
                events.append('drag_move')
        if press:
            dbl = (self.t - self.last_press_t <= DCLICK_STEPS and abs(mx - self.last_press_xy[0]) <= 4
                   and abs(my - self.last_press_xy[1]) <= 4)
            self.last_press_t = -100 if dbl else self.t
            self.last_press_xy = (mx, my)
            self._press(mx, my, dbl, events)
        if btn and not press and self.pt_drawing and self.focus == 'paint':
            self._paint_at(mx, my, events, cont=True)
        if btn and not press and self.c_press and self.focus == 'cards' and not self.c_dragging:
            if abs(mx - self.c_press[0]) + abs(my - self.c_press[1]) >= 4:
                self.c_dragging = True
                events.append('cards_drag')
        if release:
            if self.drag:
                events.append('drag_end')
            self.drag = None
            self.pt_drawing = False
            if self.c_dragging and 'cards' in self.z and self.c_sel:
                self._cards_drop(mx, my, events)
                self.c_sel = None
            self.c_press, self.c_dragging = None, False
        if key:
            self._key(key, events)
        if 'terminal' in self.z:
            self.term.tick(events)
        if self.focus == 'doom':
            fire = press and self._in_doom_view(mx, my)
            dkey = key if key in (1, 2, 3, 4, 5) else (KEY_FIRE if key == KEY_CHAR0 else 0)
            events += self.doom.step(dkey, fire)
        if 'calculator' in self.z:
            self.calc.tick()
        if 'paint' in self.z and self.pt_load < 10:
            self.pt_load += 1
            if self.pt_load == 10:
                events.append('paint_loaded')
        self.btn = btn
        self.t += 1
        return self.observe(events)

    def _window_at(self, c, r):
        for app in reversed(self.z):
            wc, wr = self.pos[app]
            w, h = APP_SIZES[app]
            if wc <= c < wc + w and wr <= r < wr + h:
                return app
        return None

    def _press(self, mx, my, dbl, events):
        c, r = mx // T, my // T
        if self.ex_naming is not None and self._window_at(c, r) != 'explorer':
            self.ex_naming = None
            events.append('ex_name_cancel')
        if r == GH - 1:
            if 0 <= c - 4 and (c - 4) % TASKBAR_BTN_W < 4:
                i = (c - 4) // TASKBAR_BTN_W
                opened = [a for a in APPS if a in self.z]
                if i < len(opened):
                    self.raise_app(opened[i])
                    events.append('taskbar_focus')
            return
        app = self._window_at(c, r)
        if app is None:
            self.desk_sel = -1
            for i, (_, _, target) in enumerate(DESK_ICONS):
                ir, ic = icon_pos(i)
                if ir <= r <= ir + 2 and ic <= c <= ic + 2 and not (r < ir + 2 and c == ic + 2):
                    self.desk_sel = i
                    events.append('desk_select')
                    if dbl:
                        self.open_app(target[0], target[1], events)
                        events.append('desk_open')
            return
        if app != self.focus:
            events.append('focus_' + app)
        self.raise_app(app)
        wc, wr = self.pos[app]
        lc, lr = c - wc, r - wr
        w, h = APP_SIZES[app]
        if lr == 0:
            if lc == w - 1:
                self.close_app(app, events)
            else:
                self.drag = (app, mx - wc * T, my - wr * T)
                events.append('drag_start')
            return
        getattr(self, '_press_' + app)(mx, my, lc, lr, dbl, events)

    def _press_terminal(self, mx, my, lc, lr, dbl, events):
        pass

    def _key(self, key, events):
        app = self.focus
        if app == 'terminal':
            self.term.key(key, events)
            return
        if app == 'explorer' and self.ex_naming is not None:
            self._ex_name_key(key, events)
            return
        if key == KEY_ESC and app:
            self.close_app(app, events)
        elif app == 'notepad':
            if KEY_CHAR0 <= key < KEY_EXTRA0 and len(self.np_text) < NP_MAX:
                self.np_text += CHARSET[key - KEY_CHAR0]
                events.append('np_type')
            elif key == KEY_ENTER and len(self.np_text) < NP_MAX and len(self._np_lines(self.np_text)) < 40:
                self.np_text += '\n'
                events.append('np_newline')
            elif key == KEY_DEL and self.np_text:
                self.np_text = self.np_text[:-1]
                events.append('np_backspace')
        elif app == 'calculator':
            tok = calc_app.key_token(key)
            if tok:
                events.append(self.calc.press(tok))
        elif app == 'explorer':
            items = self.fs[self.ex_path]
            if key == KEY_DOWN and items:
                self.ex_sel = min(self.ex_sel + 1, len(items) - 1); events.append('ex_key_sel')
            elif key == KEY_UP and items:
                self.ex_sel = max(self.ex_sel - 1, 0); events.append('ex_key_sel')
            elif key == KEY_ENTER:
                self._ex_open(events)
            elif key == KEY_DEL:
                self._ex_delete(events)

    # ------------------------------------------------------------------ explorer
    def _press_explorer(self, mx, my, lc, lr, dbl, events):
        trash = self.ex_path == 'TRASH'
        if self.ex_naming is not None and not (lr == 1 and lc in (9, 10)):
            self.ex_naming = None
            events.append('ex_name_cancel')
        if lr == 1:
            if lc == 1:
                if self.ex_path != 'C:':
                    self.ex_path, self.ex_sel = 'C:', -1
                    events.append('ex_up')
            elif lc in (3, 4):
                self._ex_open(events)
            elif lc in (6, 7):
                self._ex_delete(events)
            elif not trash and lc in (9, 10):
                if len(self.fs[self.ex_path]) < 9:
                    self.ex_naming, self.ex_sel = '', -1
                    events.append('ex_new')
            elif trash and 9 <= lc <= 11:
                self._ex_restore(events)
            elif trash and 12 <= lc <= 14:
                if self.fs['TRASH']:
                    self.fs['TRASH'] = []
                    self.origin = {}
                    self.ex_sel = -1
                    events.append('ex_empty_trash')
            return
        idx = lr - 2
        items = self.fs[self.ex_path]
        if 0 <= idx < len(items) and 1 <= lc <= 14:
            self.ex_sel = idx
            events.append('ex_select')
            if dbl:
                self._ex_open(events)
        else:
            self.ex_sel = -1

    def _name_taken(self, name):
        return any(n == name for folder in self.fs.values() for n, _ in folder)

    def _ex_name_key(self, key, events):
        if KEY_CHAR0 <= key < KEY_EXTRA0:
            if len(self.ex_naming) < 20:
                self.ex_naming += CHARSET[key - KEY_CHAR0]
                events.append('ex_name_type')
        elif key == KEY_DEL:
            self.ex_naming = self.ex_naming[:-1]
        elif key == KEY_ESC:
            self.ex_naming = None
            events.append('ex_name_cancel')
        elif key == KEY_ENTER:
            name = self.ex_naming.strip()
            items = self.fs[self.ex_path]
            taken = {n for folder in self.fs.values() for n, _ in folder}
            if name and name not in taken and len(items) < 9:
                kind = 'png' if pics.is_image(name) else 'txt'
                items.append((name, kind))
                if kind == 'png':
                    self.pictures[name] = pics.generate(name, self.pic_rng)
                else:
                    self.texts[name] = ''
                self.ex_sel = len(items) - 1
                events.append('ex_new_file')
            else:
                events.append('ex_name_invalid')
            self.ex_naming = None

    def _ex_open(self, events):
        items = self.fs[self.ex_path]
        if not (0 <= self.ex_sel < len(items)) or self.ex_path == 'TRASH':
            return
        name, kind = items[self.ex_sel]
        if kind == 'folder':
            self.ex_path, self.ex_sel = 'C:/' + name, -1
            events.append('ex_open_folder')
        elif kind == 'png':
            self.open_app('paint', name, events)
            events.append('ex_open_png')
        elif kind == 'exe':
            self.open_app(name[:-4].lower(), None, events)
            events.append('ex_open_exe')
        elif kind == 'txt':
            self.open_app('notepad', name, events)
            events.append('ex_open_txt')

    def _ex_delete(self, events):
        self._delete_file(self.ex_path, self.ex_sel, events)

    def _delete_file(self, path, index, events):
        items = self.fs[path]
        if not (0 <= index < len(items)):
            return
        name, kind = items[index]
        if kind == 'folder':
            return
        items.pop(index)
        if path == 'TRASH':
            self.origin.pop(name, None)
            self.pictures.pop(name, None)
            self.texts.pop(name, None)
            events.append('ex_delete_forever')
        else:
            if len(self.fs['TRASH']) >= 9:
                old, _ = self.fs['TRASH'].pop(0)
                self.origin.pop(old, None)
                self.texts.pop(old, None)
                self.pictures.pop(old, None)
            self.fs['TRASH'].append((name, kind))
            self.origin[name] = path
            events.append('ex_delete')
            if self.pt_file == name:
                self.pt_file = None
            if self.np_file == name:
                self.np_file = None
        self.ex_sel = min(self.ex_sel, len(self.fs[self.ex_path]) - 1)

    def _ex_restore(self, events):
        items = self.fs['TRASH']
        if not (0 <= self.ex_sel < len(items)):
            return
        name, kind = items[self.ex_sel]
        dest = self.origin.get(name, 'C:/DOCS')
        if len(self.fs[dest]) >= 9:
            return
        items.pop(self.ex_sel)
        self.origin.pop(name, None)
        self.fs[dest].append((name, kind))
        self.ex_sel = min(self.ex_sel, len(items) - 1)
        events.append('ex_restore')

    # ------------------------------------------------------------------ paint
    def _press_paint(self, mx, my, lc, lr, dbl, events):
        if self.pt_load < 10:
            return
        if lr == 1:
            if 1 <= lc <= 8:
                self.pt_color = lc - 1
                events.append('paint_color')
            elif 10 <= lc <= 12:
                self.pt_canvas[:] = 0
                events.append('paint_clear')
            elif 14 <= lc <= 15:
                self._paint_save(events)
            return
        self._paint_at(mx, my, events, cont=False)

    def _canvas_cell(self, mx, my):
        wc, wr = self.pos['paint']
        cx, cy = (mx - (wc + 1) * T) // 4, (my - (wr + 2) * T) // 4
        if 0 <= cx < CANVAS_W and 0 <= cy < CANVAS_H:
            return cx, cy
        return None

    def _paint_at(self, mx, my, events, cont):
        cell = self._canvas_cell(mx, my)
        if cell is None:
            self.pt_last = None
            return
        pts = line_cells(*self.pt_last, *cell) if (cont and self.pt_last) else [cell]
        for x, y in pts:
            self.pt_canvas[y, x] = self.pt_color
        self.pt_drawing, self.pt_last = True, cell
        events.append('paint_stroke' if cont else 'paint_dot')

    def _paint_save(self, events):
        if self.pt_file is None:
            if len(self.fs['C:/PICS']) >= 9:
                return
            while self._name_taken(f'PIC{self.new_pic}.PNG'):
                self.new_pic += 1
            self.pt_file = f'PIC{self.new_pic}.PNG'
            self.new_pic += 1
            self.fs['C:/PICS'].append((self.pt_file, 'png'))
        self.pictures[self.pt_file] = self.pt_canvas.copy()
        events.append('paint_save')

    # ------------------------------------------------------------------ notepad
    @staticmethod
    def _np_lines(text):
        out = []
        for line in text.split('\n'):
            out += [line[i:i + NP_COLS] for i in range(0, max(len(line), 1), NP_COLS)]
        return out

    def _press_notepad(self, mx, my, lc, lr, dbl, events):
        if lr == 1 and lc in (1, 2):
            self.np_text, self.np_file = '', None
            events.append('np_new')
        elif lr == 1 and lc in (4, 5):
            if self.np_file is None:
                if len(self.fs['C:/DOCS']) >= 9:
                    return
                while self._name_taken(f'DOC{self.new_doc}.TXT'):
                    self.new_doc += 1
                self.np_file = f'DOC{self.new_doc}.TXT'
                self.new_doc += 1
                self.fs['C:/DOCS'].append((self.np_file, 'txt'))
            self.texts[self.np_file] = self.np_text
            events.append('np_save')

    # ------------------------------------------------------------------ calculator
    def _press_calculator(self, mx, my, lc, lr, dbl, events):
        tok = calc_app.button_token(lc, lr)
        if tok:
            events.append(self.calc.press(tok))

    # ------------------------------------------------------------------ cards
    def _tab_visible(self, k):
        n = len(self.tab[k])
        off = max(0, n - 11)
        return off, n

    def _press_cards(self, mx, my, lc, lr, dbl, events):
        if self.c_won:
            self._reset_app('cards')
            events.append('cards_redeal')
            return
        if 2 <= lr <= 4:
            if lc in (1, 2):
                self.c_sel = None
                if self.stock:
                    self.waste.append(self.stock.pop())
                elif self.waste:
                    self.stock = self.waste[::-1]
                    self.waste = []
                events.append('cards_draw')
                return
            if lc in (4, 5) and self.waste:
                if dbl and self._auto_found(('waste',), events):
                    return
                self.c_sel = ('waste',)
                wc, wr = self.pos['cards']
                self.c_press, self.c_grab = (mx, my), (mx - (wc + 4) * T, my - (wr + 2) * T)
                events.append('cards_select')
                return
            for f in range(4):
                if lc in (10 + 3 * f, 11 + 3 * f):
                    if self.c_sel:
                        self._move_to_found(self.c_sel, f, events)
                    self.c_sel = None
                    return
        if lr >= 6:
            for k in range(7):
                if lc in (1 + 3 * k, 2 + 3 * k):
                    self._press_tab(k, lr - 6, dbl, events, mx, my)
                    return
        self.c_sel = None

    def _press_tab(self, k, rr, dbl, events, mx=0, my=0):
        col = self.tab[k]
        off, n = self._tab_visible(k)
        if n == 0:
            if self.c_sel and rr < 3:
                self._move_to_tab(self.c_sel, k, events)
            self.c_sel = None
            return
        vis = n - off
        if rr >= vis + 2:
            self.c_sel = None
            return
        j = off + min(rr, vis - 1)
        if self.c_sel:
            self._move_to_tab(self.c_sel, k, events)
            self.c_sel = None
            return
        if not col[j][1]:
            if j == n - 1:
                col[j][1] = True
                events.append('cards_flip')
            return
        if dbl and j == n - 1 and self._auto_found(('tab', k, j), events):
            return
        self.c_sel = ('tab', k, j)
        wc, wr = self.pos['cards']
        self.c_press, self.c_grab = (mx, my), (mx - (wc + 1 + 3 * k) * T, my - (wr + 6 + j - off) * T)
        events.append('cards_select')

    def _cards_drop(self, mx, my, events):
        wc, wr = self.pos['cards']
        lc, lr = mx // T - wc, my // T - wr
        if 1 <= lr <= 4 and 10 <= lc <= 21:
            ok = self._move_to_found(self.c_sel, min((lc - 10) // 3, 3), events)
            if not ok:
                events.append('cards_illegal')
        elif lr >= 5 and 1 <= lc <= 21:
            self._move_to_tab(self.c_sel, min((lc - 1) // 3, 6), events)

    def _sel_cards(self, sel):
        if sel[0] == 'waste':
            return [self.waste[-1]]
        _, k, j = sel
        return [c for c, _ in self.tab[sel[1]][j:]]

    def _remove_sel(self, sel, events):
        if sel[0] == 'waste':
            self.waste.pop()
            return
        _, k, j = sel
        del self.tab[k][j:]
        if self.tab[k] and not self.tab[k][-1][1]:
            self.tab[k][-1][1] = True
            events.append('cards_flip')

    def _move_to_found(self, sel, f, events):
        cards = self._sel_cards(sel)
        if len(cards) != 1:
            return False
        (r, s), fd = cards[0], self.found[f]
        if (not fd and r == 0) or (fd and fd[-1][1] == s and fd[-1][0] == r - 1):
            self._remove_sel(sel, events)
            fd.append((r, s))
            events.append('cards_found')
            if all(len(x) == 13 for x in self.found):
                self.c_won = True
                events.append('cards_win')
            return True
        return False

    def _auto_found(self, sel, events):
        for f in range(4):
            if self._move_to_found(sel, f, events):
                self.c_sel = None
                return True
        return False

    def _move_to_tab(self, sel, k, events):
        if sel[0] == 'tab' and sel[1] == k:
            return False
        cards = self._sel_cards(sel)
        r, s = cards[0]
        col = self.tab[k]
        ok = (not col and r == 12) or (col and col[-1][1] and (col[-1][0][1] < 2) != (s < 2)
                                         and col[-1][0][0] == r + 1)
        if ok:
            self._remove_sel(sel, events)
            col.extend([[c, True] for c in cards])
            events.append('cards_move')
        else:
            events.append('cards_illegal')
        return ok

    # ------------------------------------------------------------------ doom
    def _doom_view_origin(self):
        wc, wr = self.pos['doom']
        return wc + 1, wr + 1

    def _in_doom_view(self, mx, my):
        if 'doom' not in self.z:
            return False
        vc, vr = self._doom_view_origin()
        return vc * T <= mx < (vc + 20) * T and vr * T <= my < (vr + 15) * T

    def _press_doom(self, mx, my, lc, lr, dbl, events):
        pass

    # ------------------------------------------------------------------ render
    def render_grid(self):
        g = np.full((GH, GW), ATLAS[('solid', 'desk')], np.int32)
        for i, (icon, label, _) in enumerate(DESK_ICONS):
            ir, ic = icon_pos(i)
            if icon == 'trash' and self.fs['TRASH']:
                icon = 'trash_full'
            for q in range(4):
                g[ir + q // 2, ic + q % 2] = ATLAS[('icon', icon, q)]
            put_text(g, ir + 2, ic, label.center(6), 'title' if self.desk_sel == i else 'desk')
        for app in self.z:
            self._render_window(g, app)
        g[GH - 1, :] = ATLAS[('solid', 'tool')]
        put_text(g, GH - 1, 0, 'START', 'tool')
        for i, app in enumerate([a for a in APPS if a in self.z]):
            put_text(g, GH - 1, 4 + i * TASKBAR_BTN_W, APP_TITLES[app].ljust(8), 'title' if app == self.focus else 'tool')
        put_text(g, GH - 1, 29, '12:00', 'tool')
        return g

    def _frame(self, g, app, title, bg):
        wc, wr = self.pos[app]
        w, h = APP_SIZES[app]
        active = app == self.focus
        g[wr, wc:wc + w] = ATLAS[('solid', 'title' if active else 'title_in')]
        put_text(g, wr, wc, (' ' + title)[:2 * (w - 1)], 'title' if active else 'title_in')
        g[wr, wc + w - 1] = ATLAS[('close', active)]
        g[wr + 1:wr + h - 1, wc + 1:wc + w - 1] = ATLAS[('solid', bg)]
        g[wr + 1:wr + h - 1, wc] = ATLAS[('edge', bg, 'l')]
        g[wr + 1:wr + h - 1, wc + w - 1] = ATLAS[('edge', bg, 'r')]
        g[wr + h - 1, wc + 1:wc + w - 1] = ATLAS[('edge', bg, 'b')]
        g[wr + h - 1, wc] = ATLAS[('edge', bg, 'lb')]
        g[wr + h - 1, wc + w - 1] = ATLAS[('edge', bg, 'rb')]
        return wc, wr, w, h

    def _toolbar(self, g, wc, wr, w, items):
        g[wr + 1, wc + 1:wc + w - 1] = ATLAS[('solid', 'tool')]
        g[wr + 1, wc] = ATLAS[('edge', 'tool', 'l')]
        g[wr + 1, wc + w - 1] = ATLAS[('edge', 'tool', 'r')]
        for col, s in items:
            put_text(g, wr + 1, wc + col, s, 'tool')

    def _render_window(self, g, app):
        if app == 'terminal':
            wc, wr, _, _ = self._frame(g, app, 'TERMINAL', 'hud')
            self.term.render(g, wc, wr)
        elif app == 'explorer':
            wc, wr, w, h = self._frame(g, app, self.ex_path, 'body')
            items = [(1, 'UP'), (3, 'OPEN'), (6, 'DEL')]
            if self.ex_path == 'TRASH':
                items += [(9, 'UNDEL'), (12, 'EMPTY')]
            else:
                items += [(9, 'NEW')]
            self._toolbar(g, wc, wr, w, items)
            for i, (name, kind) in enumerate(self.fs[self.ex_path]):
                sel = i == self.ex_sel
                g[wr + 2 + i, wc + 1] = ATLAS[('mini', 'png' if kind == 'png' else kind if kind in ('folder', 'exe') else 'file', sel)]
                put_text(g, wr + 2 + i, wc + 2, name.ljust(26), 'sel' if sel else 'body')
            n = len(self.fs[self.ex_path])
            if self.ex_naming is not None and n < 9:
                g[wr + 2 + n, wc + 1] = ATLAS[('mini', 'png' if pics.is_image(self.ex_naming) else 'file', True)]
                put_text(g, wr + 2 + n, wc + 2, (self.ex_naming + '_').ljust(26), 'sel')
        elif app == 'paint':
            title = 'PAINT' + (' - ' + self.pt_file if self.pt_file else '')
            wc, wr, w, h = self._frame(g, app, title, 'body')
            self._toolbar(g, wc, wr, w, [(10, 'CLEAR'), (14, 'SAVE')] if self.pt_load >= 10 else [(10, 'LOADING...')])
            for i in range(8):
                g[wr + 1, wc + 1 + i] = ATLAS[('swatch', i, i == self.pt_color)]
            cv = self.pt_canvas.astype(np.int64)
            if self.pt_load < 10:
                cv = cv.copy()
                cv[self.pt_load * 2:] = 0
            ids = canvas_id(cv[0::2, 0::2], cv[0::2, 1::2], cv[1::2, 0::2], cv[1::2, 1::2])
            g[wr + 2:wr + 12, wc + 1:wc + 17] = ids
        elif app == 'cards':
            wc, wr, w, h = self._frame(g, app, 'CARDS', 'felt')
            sel = self.c_sel
            place_back_or_slot(g, wr + 2, wc + 1, bool(self.stock))
            if self.waste:
                place_card(g, wr + 2, wc + 4, self.waste[-1], sel == ('waste',))
            else:
                place_slot(g, wr + 2, wc + 4)
            for f in range(4):
                if self.found[f]:
                    place_card(g, wr + 2, wc + 10 + 3 * f, self.found[f][-1], False)
                else:
                    place_slot(g, wr + 2, wc + 10 + 3 * f)
            drag = self.c_dragging and sel is not None
            for k in range(7):
                off, n = self._tab_visible(k)
                if drag and sel[0] == 'tab' and sel[1] == k:
                    n = sel[2]
                    off = max(0, n - 11)
                if n == 0:
                    place_slot(g, wr + 6, wc + 1 + 3 * k)
                for vi, j in enumerate(range(off, n)):
                    (card, up) = self.tab[k][j]
                    covered = j < n - 1
                    is_sel = sel is not None and sel[0] == 'tab' and sel[1] == k and j >= sel[2]
                    if up:
                        place_card(g, wr + 6 + vi, wc + 1 + 3 * k, card, is_sel, covered)
                    else:
                        place_back(g, wr + 6 + vi, wc + 1 + 3 * k, covered)
            if drag and sel[0] == 'waste':
                if len(self.waste) > 1:
                    place_card(g, wr + 2, wc + 4, self.waste[-2], False)
                else:
                    place_slot(g, wr + 2, wc + 4)
            if self.c_won:
                put_text(g, wr + 10, wc + 8, ' YOU WIN! ', 'title')
            if drag:
                cards = self._sel_cards(sel)
                gc = round((self.mx - self.c_grab[0]) / T)
                gr = round((self.my - self.c_grab[1]) / T)
                tmp = np.full((len(cards) + 2, 2), -1, np.int32)
                for i, card in enumerate(cards):
                    place_card(tmp, i, 0, card, True, covered=i < len(cards) - 1)
                for (rr, cc), v in np.ndenumerate(tmp):
                    if v >= 0 and 0 <= gr + rr < GH - 1 and 0 <= gc + cc < GW:
                        g[gr + rr, gc + cc] = v
        elif app == 'doom':
            wc, wr, w, h = self._frame(g, app, 'DOOM', 'body')
            g[wr + 1:wr + 16, wc + 1:wc + 21] = ATLAS[('doom_view',)]
            put_text(g, wr + 16, wc + 1, self.doom.hud_text().center(40), 'hud')
        elif app == 'notepad':
            wc, wr, w, h = self._frame(g, app, 'NOTEPAD' + (' - ' + self.np_file if self.np_file else ''), 'body')
            self._toolbar(g, wc, wr, w, [(1, 'NEW'), (4, 'SAVE')])
            lines = self._np_lines(self.np_text + '_')[-NP_ROWS:]
            for i, line in enumerate(lines):
                put_text(g, wr + 2 + i, wc + 1, line.ljust(NP_COLS), 'body')
        elif app == 'calculator':
            wc, wr, w, h = self._frame(g, app, 'CALC', 'hud')
            calc_app.render(g, self.calc, wc, wr)

    # ------------------------------------------------------------------ observation
    def observe(self, events):
        grid = self.render_grid().astype(np.uint16)
        if 'doom' in self.z:
            cols, spr, gun = self.doom.tokens()
            vc, vr = self._doom_view_origin()
            meta = np.array([1 + gun, vc, vr], np.uint8)
        else:
            cols, spr, meta = np.zeros((dm.VW, 4), np.uint8), np.zeros((dm.NSPR, 3), np.uint8), np.zeros(3, np.uint8)
        return dict(grid=grid, cursor=np.array([self.mx, self.my], np.uint8), doom_cols=cols, doom_spr=spr,
                    doom_meta=meta, state=self.state_vector(), events=events)

    def ext_state_fields(self):
        """Spare state fields for new apps; each app claims a fixed sub-range and leaves the rest zero.
        [0:4] calculator: pending op (0-4), error, display length (0-9), flashing button + 1 (0-16)."""
        ext = [0] * EXT_FIELDS
        if 'calculator' in self.z:
            ext[0:4] = self.calc.state_fields()
        # ext[4:8] = terminal; ext[0:4] calculator, 8:12 dreamer, 12:16 flight reserved by convention
        if 'terminal' in self.z:
            ext[4:8] = [2 if self.term.basic.running else self.term.mode,
                        min(len(self.term.input) // 2, 31),
                        PATHS.index(self.term.cwd) if self.term.cwd in PATHS else 5,
                        min(len(self.term.lines), 31)]
        return ext

    def state_vector(self):
        f = []
        for app in APP_SLOTS:
            c, r = self.pos.get(app, (0, 0))
            f += [int(app in self.z), c, r]
        f.append(APP_SLOTS.index(self.focus) + 1 if self.focus else 0)
        f.append(int(self.drag is not None))
        f.append(self.btn)
        f.append(min(self.t - self.last_press_t, 9))
        f.append(self.desk_sel + 1)
        f += [PATHS.index(self.ex_path) if self.ex_path in PATHS else 0, self.ex_sel + 1, len(self.fs[self.ex_path]), len(self.fs['TRASH'])]
        f += [self.pt_color, int(self.pt_file is not None), int(self.pt_drawing)]
        sk = 0 if self.c_sel is None else (1 if self.c_sel[0] == 'waste' else 2)
        f += [len(self.stock) // 2, len(self.waste) // 2] + [len(x) for x in self.found] + [sk, self.deal_no % 8]
        f += [int(self.c_dragging), min(len(self.np_text) // 16, 31), int(self.np_file is not None)]
        f += [int(self.ex_naming is not None), len(self.ex_naming or '') // 3, self.pt_load]
        f += self.doom.state_fields() if 'doom' in self.z else [0] * 6
        f += self.ext_state_fields()
        return np.array(f, np.uint8)


# desk_sel (5th field after the app slots) has room for 15 desktop icons
STATE_VOCAB = [2, 32, 24] * len(APP_SLOTS) + [len(APP_SLOTS) + 1, 2, 2, 10, 16, 5, 10, 10, 10, 8, 2, 2, 13, 13, 14, 14, 14, 14, 3, 8, 2, 32, 2,
                                 2, 8, 11, 16, 12, 16, 11, 11, 10] + [EXT_VOCAB] * EXT_FIELDS


@lru_cache(maxsize=65536)
def text_ids(s, ctx):
    if len(s) % 2:
        s += ' '
    return np.array([text_id(ctx, s[i], s[i + 1]) for i in range(0, len(s), 2)], np.int32)


def put_text(g, r, c, s, ctx):
    ids = text_ids(s, ctx)
    a, b = max(c, 0), min(c + len(ids), GW)
    if a < b:
        g[r, a:b] = ids[a - c:b - c]


def place_card(g, r, c, card, sel, covered=False):
    rank, suit = card
    sch = "hdcs"[suit]
    g[r, c] = ATLAS[('card', 'tl', RANKS[rank], suit < 2, sel)]
    g[r, c + 1] = ATLAS[('card', 'tr', sch, sel)]
    if not covered:
        g[r + 1, c], g[r + 1, c + 1] = ATLAS[('card', 'ml', sel)], ATLAS[('card', 'mr', sel)]
        g[r + 2, c], g[r + 2, c + 1] = ATLAS[('card', 'bl', sel)], ATLAS[('card', 'br', sch, sel)]


def place_back(g, r, c, covered=False):
    g[r, c], g[r, c + 1] = ATLAS[('back', 'tl')], ATLAS[('back', 'tr')]
    if not covered:
        g[r + 1, c], g[r + 1, c + 1] = ATLAS[('back', 'ml')], ATLAS[('back', 'mr')]
        g[r + 2, c], g[r + 2, c + 1] = ATLAS[('back', 'bl')], ATLAS[('back', 'br')]


def place_slot(g, r, c):
    for dr, (a, b) in enumerate([('tl', 'tr'), ('ml', 'mr'), ('bl', 'br')]):
        g[r + dr, c], g[r + dr, c + 1] = ATLAS[('slot', a)], ATLAS[('slot', b)]


def place_back_or_slot(g, r, c, back):
    (place_back if back else place_slot)(g, r, c)


def obs_to_pixels(obs, cursor=True):
    """Display render from tokens only (works for ground truth and model predictions)."""
    grid = obs['grid'].astype(np.int64)
    px = grid_to_pixels(grid)
    meta = obs['doom_meta']
    if meta[0] > 0:
        dimg = dm.render_tokens(obs['doom_cols'], obs['doom_spr'], int(meta[0]) - 1)
        vc, vr = int(meta[1]), int(meta[2])
        mask = grid == ATLAS[('doom_view',)]
        if mask.any():
            layer = np.zeros_like(px)
            y0, x0 = vr * T, vc * T
            h, w = min(dimg.shape[0], px.shape[0] - y0), min(dimg.shape[1], px.shape[1] - x0)
            if h > 0 and w > 0:
                layer[y0:y0 + h, x0:x0 + w] = dimg[:h, :w]
                inview = np.zeros_like(mask)
                inview[vr:vr + 15, vc:vc + 20] = True
                pm = (mask & inview).repeat(T, 0).repeat(T, 1)
                px = np.where(pm, layer, px)
    if cursor:
        draw_cursor(px, int(obs['cursor'][0]), int(obs['cursor'][1]))
    return px
