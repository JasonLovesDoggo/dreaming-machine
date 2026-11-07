"""Deterministic tile-based micro OS. step(action) -> observation (tile grid + cursor + doom tokens + state)."""
import random
import numpy as np
from .tiles import (ATLAS, GW, GH, T, SCREEN_W, SCREEN_H, text_id, canvas_id, grid_to_pixels, draw_cursor, RANKS)
from . import doom as dm

APP_SIZES = {'explorer': (16, 12), 'paint': (18, 13), 'cards': (23, 20), 'doom': (22, 18)}
APPS = list(APP_SIZES)
APP_TITLES = {'explorer': 'FILES', 'paint': 'PAINT', 'cards': 'CARDS', 'doom': 'DOOM'}
DEFAULT_POS = {'explorer': (12, 2), 'paint': (11, 4), 'cards': (8, 1), 'doom': (9, 3)}
KEY_NONE, KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT, KEY_FIRE, KEY_ENTER, KEY_ESC, KEY_DEL = range(9)
NKEYS = 9
DESK_ICONS = [('folder', 'FILES', ('explorer', 'C:')), ('trash', 'TRASH', ('explorer', 'TRASH')),
              ('picture', 'PICS', ('explorer', 'C:/PICS')), ('paint', 'PAINT', ('paint', None)),
              ('cards', 'CARDS', ('cards', None)), ('doom', 'DOOM', ('doom', None))]
PATHS = ['C:', 'C:/PICS', 'C:/DOCS', 'C:/GAMES', 'TRASH']
CANVAS_H, CANVAS_W = 20, 32
DCLICK_STEPS = 8
TASKBAR_BTN_W = 5


def icon_pos(i):
    return 1 + 4 * (i % 3), (1 if i < 3 else 5)  # (row, col)


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
    def __init__(self):
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self):
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
        self.pictures = {n: make_picture(n) for n, k in self.fs['C:/PICS']}
        self.new_pic = 1
        self.deal_no = 0
        self.doom = dm.Doom()
        self._reset_app('explorer'); self._reset_app('paint'); self._reset_app('cards')
        return self.observe([])

    def _reset_app(self, app, arg=None):
        if app == 'explorer':
            self.ex_path, self.ex_sel = (arg or 'C:'), -1
        elif app == 'paint':
            self.pt_canvas = np.zeros((CANVAS_H, CANVAS_W), np.uint8)
            self.pt_color, self.pt_file, self.pt_drawing, self.pt_last = 1, None, False, None
            if arg:
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
        elif app == 'doom':
            self.doom.reset()

    @property
    def focus(self):
        return self.z[-1] if self.z else None

    def open_app(self, app, arg=None, events=None):
        events = events if events is not None else []
        if app in self.z:
            self.raise_app(app)
            if app == 'explorer' and arg:
                self.ex_path, self.ex_sel = arg, -1
            if app == 'paint' and arg:
                self._reset_app('paint', arg)
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
        self.z.remove(app)
        self.pos.pop(app, None)
        if app == 'paint':
            self._reset_app('paint')
        events.append('close_' + app)

    # ------------------------------------------------------------------ input
    def step(self, action):
        mx, my, btn, key = (int(a) for a in action)
        mx, my = min(max(mx, 0), SCREEN_W - 1), min(max(my, 0), SCREEN_H - 1)
        events = []
        press, release = btn and not self.btn, (not btn) and self.btn
        self.mx, self.my = mx, my
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
        if release:
            if self.drag:
                events.append('drag_end')
            self.drag = None
            self.pt_drawing = False
        if key:
            self._key(key, events)
        if self.focus == 'doom':
            fire = press and self._in_doom_view(mx, my)
            events += self.doom.step(key if key in (1, 2, 3, 4, 5) else 0, fire)
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

    def _key(self, key, events):
        app = self.focus
        if key == KEY_ESC and app:
            self.close_app(app, events)
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
        if lr == 1:
            if lc == 1:
                if self.ex_path != 'C:':
                    self.ex_path, self.ex_sel = 'C:', -1
                    events.append('ex_up')
            elif lc in (3, 4):
                self._ex_open(events)
            elif lc in (6, 7):
                self._ex_delete(events)
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

    def _ex_delete(self, events):
        items = self.fs[self.ex_path]
        if not (0 <= self.ex_sel < len(items)):
            return
        name, kind = items[self.ex_sel]
        if kind == 'folder':
            return
        items.pop(self.ex_sel)
        if self.ex_path == 'TRASH':
            self.origin.pop(name, None)
            self.pictures.pop(name, None)
            events.append('ex_delete_forever')
        else:
            if len(self.fs['TRASH']) >= 9:
                self.fs['TRASH'].pop(0)
            self.fs['TRASH'].append((name, kind))
            self.origin[name] = self.ex_path
            events.append('ex_delete')
            if self.pt_file == name:
                self.pt_file = None
        self.ex_sel = min(self.ex_sel, len(items) - 1)

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
            self.pt_file = f'PIC{self.new_pic}.PNG'
            self.new_pic += 1
            self.fs['C:/PICS'].append((self.pt_file, 'png'))
        self.pictures[self.pt_file] = self.pt_canvas.copy()
        events.append('paint_save')

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
                    self._press_tab(k, lr - 6, dbl, events)
                    return
        self.c_sel = None

    def _press_tab(self, k, rr, dbl, events):
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
        events.append('cards_select')

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
        if app == 'explorer':
            wc, wr, w, h = self._frame(g, app, self.ex_path, 'body')
            items = [(1, 'UP'), (3, 'OPEN'), (6, 'DEL')]
            if self.ex_path == 'TRASH':
                items += [(9, 'UNDEL'), (12, 'EMPTY')]
            self._toolbar(g, wc, wr, w, items)
            for i, (name, kind) in enumerate(self.fs[self.ex_path]):
                sel = i == self.ex_sel
                g[wr + 2 + i, wc + 1] = ATLAS[('mini', 'png' if kind == 'png' else kind if kind in ('folder', 'exe') else 'file', sel)]
                put_text(g, wr + 2 + i, wc + 2, name.ljust(26), 'sel' if sel else 'body')
        elif app == 'paint':
            title = 'PAINT' + (' - ' + self.pt_file if self.pt_file else '')
            wc, wr, w, h = self._frame(g, app, title, 'body')
            self._toolbar(g, wc, wr, w, [(10, 'CLEAR'), (14, 'SAVE')])
            for i in range(8):
                g[wr + 1, wc + 1 + i] = ATLAS[('swatch', i, i == self.pt_color)]
            cv = self.pt_canvas.astype(np.int64)
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
            for k in range(7):
                off, n = self._tab_visible(k)
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
            if self.c_won:
                put_text(g, wr + 10, wc + 8, ' YOU WIN! ', 'title')
        elif app == 'doom':
            wc, wr, w, h = self._frame(g, app, 'DOOM', 'body')
            g[wr + 1:wr + 16, wc + 1:wc + 21] = ATLAS[('doom_view',)]
            put_text(g, wr + 16, wc + 1, self.doom.hud_text().center(40), 'hud')

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

    def state_vector(self):
        f = []
        for app in APPS:
            c, r = self.pos.get(app, (0, 0))
            f += [int(app in self.z), c, r]
        f.append(APPS.index(self.focus) + 1 if self.focus else 0)
        f.append(int(self.drag is not None))
        f.append(self.btn)
        f.append(min(self.t - self.last_press_t, 9))
        f.append(self.desk_sel + 1)
        f += [PATHS.index(self.ex_path), self.ex_sel + 1, len(self.fs[self.ex_path]), len(self.fs['TRASH'])]
        f += [self.pt_color, int(self.pt_file is not None), int(self.pt_drawing)]
        sk = 0 if self.c_sel is None else (1 if self.c_sel[0] == 'waste' else 2)
        f += [len(self.stock) // 2, len(self.waste) // 2] + [len(x) for x in self.found] + [sk, self.deal_no % 8]
        f += self.doom.state_fields() if 'doom' in self.z else [0] * 6
        return np.array(f, np.uint8)


STATE_VOCAB = [2, 32, 24] * 4 + [5, 2, 2, 10, 7, 5, 10, 10, 10, 8, 2, 2, 13, 13, 14, 14, 14, 14, 3, 8, 16, 12, 16, 11, 11, 10]


def put_text(g, r, c, s, ctx):
    if len(s) % 2:
        s += ' '
    for i in range(0, len(s), 2):
        if 0 <= c + i // 2 < GW:
            g[r, c + i // 2] = text_id(ctx, s[i], s[i + 1])


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
        for r, c in zip(*np.nonzero(mask)):
            lr, lc = r - vr, c - vc
            if 0 <= lr < 15 and 0 <= lc < 20:
                px[r * T:(r + 1) * T, c * T:(c + 1) * T] = dimg[lr * T:(lr + 1) * T, lc * T:(lc + 1) * T]
    if cursor:
        draw_cursor(px, int(obs['cursor'][0]), int(obs['cursor'][1]))
    return px
