"""Smart scripted user: hierarchical goal-directed tasks + human-like mouse motion + coverage-driven task choice."""
import math
import random
from collections import Counter
from .os_sim import (APP_SIZES, APPS, DESK_ICONS, icon_pos, CANVAS_W, CANVAS_H, TASKBAR_BTN_W, DCLICK_STEPS, KEY_UP, KEY_DOWN,
                     KEY_LEFT, KEY_RIGHT, KEY_FIRE, KEY_ENTER, KEY_ESC, KEY_DEL, NKEYS, KEY_CHAR0)
from .tiles import CHAR_IDX, CHARSET
from .pictures import IMAGE_EXTS, CATEGORY_WORDS, CAT_WORDS
from .tiles import T, GW, GH, SCREEN_W, SCREEN_H
from . import doom as dm
from .apps.dreamer import prompts
from .apps import calculator as calc

TASK_EVENTS = {
    'terminal': ['term_cmd', 'term_error', 'term_fs', 'btop_open', 'neofetch'],
    'explorer': ['ex_open_folder', 'ex_delete', 'ex_restore', 'ex_empty_trash', 'ex_open_png', 'ex_up', 'ex_select'],
    'paint': ['paint_stroke', 'paint_color', 'paint_save', 'paint_clear'],
    'cards': ['cards_move', 'cards_found', 'cards_flip', 'cards_draw'],
    'doom': ['doom_fire', 'doom_kill', 'doom_hurt'],
    'windows': ['drag_move', 'close_explorer', 'close_paint', 'close_cards', 'close_doom', 'taskbar_focus'],
    'notepad': ['np_type', 'np_save', 'np_new', 'np_backspace', 'ex_open_txt'],
    'images': ['ex_new_file', 'paint_loaded'],
    'calculator': ['calc_digit', 'calc_op', 'calc_eq', 'calc_clear', 'calc_del', 'calc_err'],
    'noise': ['desk_select'],
    'chaos': [],
    'dreamer': ['dreamer_send', 'dreamer_reply', 'dreamer_type', 'dreamer_backspace'],
}
SUBJECT_KEYS = sorted(k for k in CATEGORY_WORDS if k != 'cat')
CAT_BASE_WORDS = ['CAT', 'CAT', 'KITTY', 'KITTEN', 'CATS']
UNKNOWN_WORDS = "IMG PHOTO TRIP MEME ART PIC SCAN WALLPAPER DOODLE BEST OLD THING STUFF".split()
WORDS = ("HELLO WORLD THE A DREAM MACHINE IS RUNNING DOOM CAT DOG PAINT TODO SHIP IT NOW GPU MODEL TRAIN PLAY CARDS "
         "FILE SAVE NEW OK YES NO 123 2026 WIN NOTE MEETING AT 5:00 BUY MILK EGGS CALL MOM FIX BUG DEPLOY FRIDAY "
         "COFFEE PIZZA TONIGHT REMEMBER TO WATER PLANTS SOLITAIRE HIGH SCORE KILLS AMMO HP LOVE THIS COMPUTER "
         "IN MY HEAD 42 99 100 GO STOP WAIT WHY HOW WHAT WHEN LIST ITEM DONE MAYBE LATER TODAY TOMORROW").split()
BASE_W = {'explorer': 1.0, 'paint': 1.0, 'cards': 1.0, 'doom': 1.6, 'notepad': 1.0, 'images': 3.0, 'windows': 0.8, 'noise': 0.4, 'chaos': 2.0,
          'calculator': 1.0}
BASE_W['terminal'] = 1.2
BASE_W['dreamer'] = 1.5
BASE_W['flight'] = 1.2
TASK_EVENTS['flight'] = ['flight_fly', 'flight_crash', 'flight_reset', 'flight_takeoff', 'flight_land']
FLIGHT_STEPS = (200, 600)  # autopilot frames per Flight session
FLIGHT_SHARE = .07  # mixed mode skips Flight while it has used more than this share of all frames


class Driver:
    def __init__(self, os_, seed, mode='mixed'):
        self.os = os_
        self.mode = mode
        self.rng = random.Random(seed)
        self.x, self.y = SCREEN_W // 2, SCREEN_H // 2
        self.btn = 0
        self.counts = Counter()
        self.stream = self._guarded(self._main())

    # ---------------------------------------------------------------- primitives
    def act(self, key=0):
        self.x = min(max(int(self.x), 0), SCREEN_W - 1)
        self.y = min(max(int(self.y), 0), SCREEN_H - 1)
        return (self.x, self.y, self.btn, key)

    def idle(self, n):
        for _ in range(n):
            if self.rng.random() < 0.3 and not self.btn:
                self.x = min(max(self.x + self.rng.randint(-1, 1), 0), SCREEN_W - 1)
                self.y = min(max(self.y + self.rng.randint(-1, 1), 0), SCREEN_H - 1)
            yield self.act()

    def move_to(self, tx, ty, speed=None, straight=False):
        r = self.rng
        tx, ty = min(max(int(tx), 0), SCREEN_W - 1), min(max(int(ty), 0), SCREEN_H - 1)
        x0, y0 = self.x, self.y
        dist = math.hypot(tx - x0, ty - y0)
        if dist < 1:
            return
        speed = speed or r.uniform(6, 22)
        n = int(min(max(dist / speed, 2), 30))
        overshoot = not straight and r.random() < 0.2 and dist > 40
        ex, ey = (tx + (tx - x0) * 0.08, ty + (ty - y0) * 0.08) if overshoot else (tx, ty)
        bend = 0 if straight else r.uniform(-0.25, 0.25) * dist
        nx, ny = -(ty - y0) / dist, (tx - x0) / dist
        cx, cy = (x0 + ex) / 2 + nx * bend, (y0 + ey) / 2 + ny * bend
        for i in range(1, n + 1):
            s = i / n
            s = s * s * (3 - 2 * s)
            bx = (1 - s) ** 2 * x0 + 2 * (1 - s) * s * cx + s * s * ex
            by = (1 - s) ** 2 * y0 + 2 * (1 - s) * s * cy + s * s * ey
            jitter = 1 if i < n and not straight else 0
            self.x = min(max(int(round(bx)) + r.randint(-jitter, jitter), 0), SCREEN_W - 1)
            self.y = min(max(int(round(by)) + r.randint(-jitter, jitter), 0), SCREEN_H - 1)
            yield self.act()
        if overshoot:
            for i in range(1, 3):
                self.x, self.y = int(ex + (tx - ex) * i / 2), int(ey + (ty - ey) * i / 2)
                yield self.act()
        self.x, self.y = tx, ty

    def click(self, x, y, double=False):
        yield from self.move_to(x, y)
        yield from self.idle(self.rng.randint(0, 2))
        for k in range(2 if double else 1):
            self.btn = 1
            for _ in range(self.rng.randint(1, 2)):
                yield self.act()
            self.btn = 0
            yield self.act()
        yield from self.idle(self.rng.randint(0, 3))

    def drag(self, x0, y0, x1, y1, speed=None, straight=False):
        yield from self.move_to(x0, y0)
        self.btn = 1
        yield self.act()
        yield from self.move_to(x1, y1, speed=speed or self.rng.uniform(4, 12), straight=straight)
        yield self.act()
        self.btn = 0
        yield self.act()

    def press_key(self, key, n=1):
        for _ in range(n):
            yield self.act(key)

    # ---------------------------------------------------------------- geometry helpers
    def win(self, app):
        wc, wr = self.os.pos[app]
        w, h = APP_SIZES[app]
        return wc, wr, w, h

    def taskbar_xy(self, app):
        opened = [a for a in APPS if a in self.os.z]
        i = opened.index(app)
        return (4 + i * TASKBAR_BTN_W) * T + self.rng.randint(3, 4 * T - 3), (GH - 1) * T + self.rng.randint(2, 6)

    def ensure_focus(self, app):
        if self.os.focus == app:
            return
        if self.rng.random() < 0.6:
            yield from self.click(*self.taskbar_xy(app))
        else:
            wc, wr, w, h = self.win(app)
            yield from self.click(wc * T + self.rng.randint(8, (w - 1) * T - 4), wr * T + 4)

    def tile_xy(self, c, r, wpx=T, hpx=T):
        return c * T + self.rng.randint(1, wpx - 2), r * T + self.rng.randint(1, hpx - 2)

    def open_app(self, app, arg=None):
        if app in self.os.z and arg is None:
            yield from self.ensure_focus(app)
            return
        idx = next(i for i, (_, _, tg) in enumerate(DESK_ICONS) if tg[0] == app and tg[1] == arg) \
            if any(tg[0] == app and tg[1] == arg for _, _, tg in DESK_ICONS) else None
        if idx is None:
            return
        ir, ic = icon_pos(idx)
        for _ in range(3):
            occ = self.os._window_at(ic, ir) or self.os._window_at(ic + 1, ir + 1)
            if occ is None:
                break
            wc, wr, w, h = self.win(occ)
            gx = wc * T + self.rng.randint(8, (w - 2) * T)
            yield from self.drag(gx, wr * T + 4, gx + (GW - w - wc) * T, wr * T + 4 + self.rng.randint(-16, 16))
        yield from self.click(ic * T + self.rng.randint(2, 14), ir * T + self.rng.randint(2, 14), double=True)
        yield from self.idle(self.rng.randint(1, 4))

    def close(self, app):
        if app not in self.os.z:
            return
        if self.rng.random() < 0.15:
            yield from self.ensure_focus(app)
            yield from self.press_key(KEY_ESC)
            return
        yield from self.ensure_focus(app)
        wc, wr, w, h = self.win(app)
        yield from self.click((wc + w - 1) * T + self.rng.randint(1, 6), wr * T + self.rng.randint(1, 6))

    # ---------------------------------------------------------------- tasks
    def task_windows(self):
        r = self.rng
        if not self.os.z:
            yield from self.open_app(r.choice(['paint', 'cards', 'doom', 'explorer']),
                                     None if r.random() < 0.7 else None)
            if not self.os.z:
                yield from self.open_app('explorer', 'C:')
        for _ in range(r.randint(1, 4)):
            if not self.os.z:
                break
            app = r.choice(self.os.z)
            choice = r.random()
            if choice < 0.5:
                yield from self.ensure_focus(app)
                wc, wr, w, h = self.win(app)
                gx = wc * T + r.randint(8, (w - 1) * T - 4)
                nc, nr = r.randint(0, GW - w), r.randint(0, GH - 1 - h)
                yield from self.drag(gx, wr * T + r.randint(1, 6), gx + (nc - wc) * T + r.randint(-3, 3),
                                     wr * T + 4 + (nr - wr) * T + r.randint(-3, 3))
            elif choice < 0.75:
                yield from self.ensure_focus(app)
            else:
                yield from self.close(app)
            yield from self.idle(r.randint(0, 5))

    def task_explorer(self):
        r = self.rng
        start = r.choice(['C:', 'C:', 'TRASH', 'C:/PICS'])
        if 'explorer' in self.os.z and r.random() < 0.5:
            yield from self.ensure_focus('explorer')
        else:
            yield from self.open_app('explorer', start)
        if 'explorer' not in self.os.z:
            return
        for _ in range(r.randint(3, 12)):
            if 'explorer' not in self.os.z or self.os.focus != 'explorer':
                yield from self.ensure_focus('explorer') if 'explorer' in self.os.z else self.idle(1)
                if 'explorer' not in self.os.z:
                    return
            wc, wr, w, h = self.win('explorer')
            items = self.os.fs[self.os.ex_path]
            trash = self.os.ex_path == 'TRASH'
            p = r.random()
            if items and p < 0.35:
                i = r.randrange(len(items))
                kind = items[i][1]
                dbl = kind in ('folder', 'png', 'exe', 'txt') and r.random() < 0.6 and not trash
                yield from self.click((wc + 2) * T + r.randint(2, 60), (wr + 2 + i) * T + r.randint(1, 6), double=dbl)
                if self.os.focus == 'paint':
                    yield from self.task_paint(draw_only=True)
                    return
                if self.os.focus == 'notepad':
                    yield from self.task_notepad(type_only=True)
                    return
            elif p < 0.5:
                yield from self.delete_file()
            elif p < 0.6:
                yield from self.click(*self.tile_xy(wc + 1, wr + 1))
            elif p < 0.68 and trash and self.os.ex_sel >= 0:
                yield from self.click(*self.tile_xy(wc + 9, wr + 1, 3 * T))
            elif p < 0.72 and trash:
                yield from self.click(*self.tile_xy(wc + 12, wr + 1, 3 * T))
            elif p < 0.8 and items:
                for _ in range(r.randint(1, 4)):
                    yield from self.press_key(r.choice([KEY_UP, KEY_DOWN]))
                    yield from self.idle(r.randint(1, 3))
                if r.random() < 0.4:
                    yield from self.press_key(KEY_ENTER)
            elif p < 0.88 and self.os.ex_sel >= 0 and not trash:
                yield from self.click(*self.tile_xy(wc + 3, wr + 1, 2 * T))
                if self.os.focus == 'paint':
                    yield from self.task_paint(draw_only=True)
                    return
            elif p < 0.93:
                yield from self.open_app('explorer', 'TRASH')
            else:
                yield from self.click(*self.tile_xy(wc + r.randint(1, w - 2), wr + r.randint(2, h - 2)))
            yield from self.idle(r.randint(0, 6))

    def delete_file(self):
        r = self.rng
        items = self.os.fs[self.os.ex_path]
        files = [i for i, (_, kind) in enumerate(items) if kind != 'folder']
        if not files:
            return
        wc, wr, w, h = self.win('explorer')
        i = r.choice(files)
        yield from self.idle(DCLICK_STEPS + 1)
        yield from self.click((wc + 2) * T + r.randint(2, 60), (wr + 2 + i) * T + 4)
        if self.os.focus != 'explorer':
            yield from self.ensure_focus('explorer')
        if r.random() < 0.6:
            yield from self.click(*self.tile_xy(wc + 6, wr + 1, 2 * T))
        else:
            yield from self.press_key(KEY_DEL)

    def shape_cells(self):
        r = self.rng
        kind = r.choice(['line', 'rect', 'circle', 'zigzag', 'walk', 'fill', 'spiral'])
        W, H = CANVAS_W, CANVAS_H
        x0, y0 = r.randrange(W), r.randrange(H)
        if kind == 'line':
            return [(x0, y0), (r.randrange(W), r.randrange(H))]
        if kind == 'rect':
            x1, y1 = r.randrange(W), r.randrange(H)
            return [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
        if kind == 'circle':
            rad = r.randint(2, 8)
            return [(int(x0 + rad * math.cos(a / 12 * 2 * math.pi)), int(y0 + rad * math.sin(a / 12 * 2 * math.pi)))
                    for a in range(13)]
        if kind == 'zigzag':
            return [(min(W - 1, x0 + i * 3), y0 + (4 if i % 2 else 0)) for i in range(r.randint(3, 8))]
        if kind == 'fill':
            x1, y1 = min(W - 1, x0 + r.randint(2, 10)), min(H - 1, y0 + r.randint(2, 8))
            pts = []
            for yy in range(y0, y1 + 1, 1):
                pts += [(x0, yy), (x1, yy)] if (yy - y0) % 2 == 0 else [(x1, yy), (x0, yy)]
            return pts
        if kind == 'spiral':
            return [(int(x0 + a * 0.5 * math.cos(a * 0.8)), int(y0 + a * 0.5 * math.sin(a * 0.8))) for a in range(16)]
        pts = [(x0, y0)]
        for _ in range(r.randint(4, 15)):
            x0, y0 = x0 + r.randint(-3, 3), y0 + r.randint(-3, 3)
            pts.append((x0, y0))
        return pts

    def task_paint(self, draw_only=False):
        r = self.rng
        if not draw_only:
            if r.random() < 0.25:
                yield from self.open_app('explorer', 'C:/PICS')
                if 'explorer' in self.os.z:
                    items = self.os.fs[self.os.ex_path]
                    pngs = [i for i, it in enumerate(items) if it[1] == 'png']
                    if pngs and self.os.ex_path == 'C:/PICS':
                        wc, wr, w, h = self.win('explorer')
                        i = r.choice(pngs)
                        yield from self.click((wc + 2) * T + r.randint(2, 50), (wr + 2 + i) * T + 4, double=True)
            else:
                yield from self.open_app('paint')
        while 'paint' in self.os.z and self.os.pt_load < 10:
            yield from self.idle(1)
        if 'paint' not in self.os.z:
            return
        for _ in range(r.randint(2, 8)):
            if 'paint' not in self.os.z:
                return
            yield from self.ensure_focus('paint')
            wc, wr, w, h = self.win('paint')
            p = r.random()
            if p < 0.15:
                yield from self.paint_repeat()
            elif p < 0.35:
                yield from self.click(*self.tile_xy(wc + 1 + r.randrange(8), wr + 1))
            elif p < 0.85:
                pts = [(min(max(x, 0), CANVAS_W - 1), min(max(y, 0), CANVAS_H - 1)) for x, y in self.shape_cells()]
                to_px = lambda c: ((wc + 1) * T + c[0] * 4 + r.randint(0, 3), (wr + 2) * T + c[1] * 4 + r.randint(0, 3))
                yield from self.move_to(*to_px(pts[0]))
                self.btn = 1
                yield self.act()
                for c in pts[1:]:
                    yield from self.move_to(*to_px(c), speed=r.uniform(2, 8))
                self.btn = 0
                yield self.act()
            elif p < 0.92:
                yield from self.click(*self.tile_xy(wc + 14, wr + 1, 2 * T))
            elif p < 0.95:
                yield from self.click(*self.tile_xy(wc + 10, wr + 1, 3 * T))
            else:
                yield from self.click(r.randrange(SCREEN_W), r.randrange(SCREEN_H - T))
            yield from self.idle(r.randint(0, 4))
        if r.random() < 0.3:
            yield from self.close('paint')

    def paint_repeat(self):
        r = self.rng
        wc, wr, w, h = self.win('paint')
        kind = r.choice(['dot', 'tool', 'clear', 'save'])
        if kind == 'dot':
            if self.os.pt_color == 0:
                yield from self.click(*self.tile_xy(wc + 1 + r.randrange(1, 8), wr + 1))
            cells = [(x, y) for y in range(CANVAS_H) for x in range(CANVAS_W)
                     if self.os.pt_canvas[y, x] == self.os.pt_color]
            x, y = r.choice(cells) if cells else (r.randrange(CANVAS_W), r.randrange(CANVAS_H))
            xy = ((wc + 1) * T + x * 4 + 2, (wr + 2) * T + y * 4 + 2)
        elif kind == 'tool':
            xy = self.tile_xy(wc + 1 + r.randrange(8), wr + 1)
        elif kind == 'clear':
            xy = self.tile_xy(wc + 10, wr + 1, 3 * T)
        else:
            if self.os.pt_file is None and len(self.os.fs['C:/PICS']) >= 9:
                return
            xy = self.tile_xy(wc + 14, wr + 1, 2 * T)
        for _ in range(r.randint(2, 4)):
            yield from self.click(*xy)

    def card_xy(self, where, k=0, j=0):
        wc, wr, w, h = self.win('cards')
        r = self.rng
        if where == 'stock':
            return (wc + 1) * T + r.randint(2, 13), (wr + 2) * T + r.randint(2, 21)
        if where == 'waste':
            return (wc + 4) * T + r.randint(2, 13), (wr + 2) * T + r.randint(2, 21)
        if where == 'found':
            return (wc + 10 + 3 * k) * T + r.randint(2, 13), (wr + 2) * T + r.randint(2, 21)
        off, n = self.os._tab_visible(k)
        vi = max(0, j - off)
        last = j >= n - 1
        return (wc + 1 + 3 * k) * T + r.randint(2, 13), (wr + 6 + vi) * T + r.randint(1, 3 * T - 2 if last else T - 2)

    def cards_moves(self):
        o = self.os
        moves = []
        tops = []
        if o.waste:
            tops.append((('waste',), o.waste[-1]))
        for k, col in enumerate(o.tab):
            if col and col[-1][1]:
                tops.append((('tab', k, len(col) - 1), col[-1][0]))
        for sel, (rk, s) in tops:
            for f in range(4):
                fd = o.found[f]
                if (not fd and rk == 0) or (fd and fd[-1][1] == s and fd[-1][0] == rk - 1):
                    moves.append((5, sel, ('found', f)))
                    break
        for k, col in enumerate(o.tab):
            for j, (card, up) in enumerate(col):
                if not up:
                    continue
                for k2, col2 in enumerate(o.tab):
                    if k2 == k:
                        continue
                    rk, s = card
                    if (not col2 and rk == 12 and j > 0) or (col2 and col2[-1][1] and (col2[-1][0][1] < 2) != (s < 2)
                                                           and col2[-1][0][0] == rk + 1):
                        moves.append((4 if j > 0 and not col[j - 1][1] else 1, ('tab', k, j), ('tab', k2)))
                break
        if o.waste:
            rk, s = o.waste[-1]
            for k2, col2 in enumerate(o.tab):
                if (not col2 and rk == 12) or (col2 and col2[-1][1] and (col2[-1][0][1] < 2) != (s < 2)
                                               and col2[-1][0][0] == rk + 1):
                    moves.append((3, ('waste',), ('tab', k2)))
        return moves

    def cards_cardinal(self):
        r, o = self.rng, self.os
        wc, wr, w, h = self.win('cards')
        if o.c_sel:
            yield from self.click((wc + w - 1) * T + 4, (wr + h - 1) * T + 4)
        singles, stacks = [], []
        for k, col in enumerate(o.tab):
            off, n = o._tab_visible(k)
            for j in range(off, n):
                if col[j][1]:
                    (singles if j == n - 1 else stacks).append(('tab', k, j))
        sources = [s for s in [singles, stacks, [('waste',)] if o.waste else []] if s]
        if not sources:
            return
        src = r.choice(r.choice(sources))
        x, y = self.card_xy(*src)
        dx, dy = r.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
        room = SCREEN_W - 1 - x if dx > 0 else x if dx < 0 else SCREEN_H - 1 - y if dy > 0 else y
        dist = min(r.randint(12, 56), room)
        # A recent click on this card must not turn the drag into an auto-foundation double-click.
        yield from self.idle(DCLICK_STEPS + 1)
        yield from self.drag(x, y, x + dx * dist, y + dy * dist, straight=True)

    def task_cards(self, full=False):
        r = self.rng
        yield from self.open_app('cards')
        if 'cards' not in self.os.z:
            return
        idle_draws, last = 0, None
        for _ in range(r.randint(150, 400) if full else r.randint(5, 30)):
            if 'cards' not in self.os.z:
                return
            yield from self.ensure_focus('cards')
            o = self.os
            if o.c_won:
                yield from self.idle(r.randint(5, 30))
                yield from self.click(*self.card_xy('stock'))
                idle_draws = 0
                continue
            moves = self.cards_moves()
            if full:
                sig = (sum(len(f) for f in o.found), sum(up for col in o.tab for _, up in col))
                idle_draws = idle_draws + 1 if sig == last else 0
                last = sig
                if idle_draws > 2 * (len(o.stock) + len(o.waste)) + 6:
                    yield from self.close('cards')
                    yield from self.open_app('cards')
                    idle_draws, last = 0, None
                    continue
            if not full and r.random() < 0.12:
                yield from self.cards_cardinal()
            elif r.random() < (0.02 if full else 0.08) or (not moves and r.random() < 0.1):
                k = r.randrange(7)
                yield from self.click(*self.card_xy('tab', k, max(0, len(o.tab[k]) - 1)))
                yield from self.click(*self.card_xy('tab', r.randrange(7), 0))
            elif moves and r.random() < 0.85:
                moves.sort(key=lambda m: -m[0] + r.random() * 1.5)
                _, src, dst = moves[0]
                if dst[0] == 'found' and r.random() < 0.5:
                    xy = self.card_xy('waste') if src[0] == 'waste' else self.card_xy('tab', src[1], src[2])
                    yield from self.click(*xy, double=True)
                elif r.random() < 0.5:
                    xy = self.card_xy('waste') if src[0] == 'waste' else self.card_xy('tab', src[1], src[2])
                    to = self.card_xy('found', dst[1]) if dst[0] == 'found' else \
                        self.card_xy('tab', dst[1], max(0, len(o.tab[dst[1]]) - 1))
                    yield from self.drag(*xy, *to)
                    yield from self.idle(r.randint(0, 3))
                else:
                    xy = self.card_xy('waste') if src[0] == 'waste' else self.card_xy('tab', src[1], src[2])
                    yield from self.click(*xy)
                    if dst[0] == 'found':
                        yield from self.click(*self.card_xy('found', dst[1]))
                    else:
                        yield from self.click(*self.card_xy('tab', dst[1], max(0, len(o.tab[dst[1]]) - 1)))
            else:
                yield from self.click(*self.card_xy('stock'))
            yield from self.idle(r.randint(0, 4))
        if r.random() < (0.05 if full else 0.25):
            yield from self.close('cards')

    def _doom_waypoint_key(self, d):
        if self._wp is None or math.hypot(self._wp[0] - d.px, self._wp[1] - d.py) < 0.8 or self._wp_age > 300:
            free = [(x + 0.5, y + 0.5) for y, row in enumerate(dm.MAP) for x, c in enumerate(row) if c == '.']
            far = [p for p in free if math.hypot(p[0] - d.px, p[1] - d.py) > 5] or free
            self._wp, self._wp_age = self.rng.choice(far), 0
        self._wp_age += 1
        tx, ty = self._wp[0] - d.px, self._wp[1] - d.py
        ang, best = d.ang, {}
        for key, da in ((KEY_UP, 0), (KEY_LEFT, -1), (KEY_RIGHT, 1)):
            d.ang = (ang + da) % dm.NANG
            dx, dy = d.direction()
            best[key] = (dx * tx + dy * ty) / (math.hypot(tx, ty) + 1e-6)
        d.ang = ang
        if best[KEY_UP] > 0.97 and d.raycast()[4][dm.VW // 2] > 0.5:
            return KEY_UP
        if best[KEY_UP] > 0.97:
            self._wp = None
            return self.rng.choice([KEY_LEFT, KEY_RIGHT])
        return KEY_LEFT if best[KEY_LEFT] > best[KEY_RIGHT] else KEY_RIGHT

    def task_doom(self, deep=False):
        r = self.rng
        yield from self.open_app('doom')
        if 'doom' not in self.os.z:
            return
        steps = r.randint(600, 1800) if deep else r.randint(40, 250)
        self._wp, self._wp_age = None, 0
        wander_turn, wander_left = 0, 0
        for _ in range(steps):
            if 'doom' not in self.os.z:
                return
            if self.os.focus != 'doom':
                yield from self.ensure_focus('doom')
                continue
            d = self.os.doom
            if r.random() < 0.05:
                yield self.act(r.randrange(6))
                continue
            wall = d.raycast()[4]
            targets = [(depth, sx, size) for imp, (sx, size, depth) in zip(d.imps, d.project_imps())
                       if imp['state'] != 2 and depth > 0 and 0 <= sx < dm.VW and depth < 7 and depth < wall[int(sx)]]
            if targets and r.random() < 0.85:
                depth, sx, size = min(targets)
                if abs(sx - dm.VW // 2) >= max(3, size // 4):
                    yield self.act(KEY_LEFT if sx < dm.VW // 2 else KEY_RIGHT)
                elif d.flash == 0:
                    if r.random() < 0.3:
                        vc, vr = self.os._doom_view_origin()
                        self.x, self.y = vc * T + r.randint(10, 150), vr * T + r.randint(10, 110)
                        self.btn = 1
                        yield self.act()
                        self.btn = 0
                    else:
                        yield self.act(KEY_FIRE)
                else:
                    yield self.act(KEY_UP if depth > 2 else 0)
                continue
            if deep and r.random() < 0.9:
                yield self.act(self._doom_waypoint_key(d))
                continue
            center = d.raycast()[4][dm.VW // 2]
            if wander_left > 0:
                wander_left -= 1
                yield self.act(wander_turn)
            elif center < 1.2 or r.random() < 0.04:
                wander_turn, wander_left = r.choice([KEY_LEFT, KEY_RIGHT]), r.randint(3, 15)
                yield self.act(wander_turn)
            else:
                yield self.act(KEY_UP if r.random() < 0.9 else KEY_DOWN)
        if r.random() < (0.05 if deep else 0.3):
            yield from self.close('doom')

    def type_text(self, text):
        r = self.rng
        for ch in text:
            if ch == '\n':
                yield from self.press_key(KEY_ENTER)
            else:
                if r.random() < 0.04:
                    yield from self.press_key(KEY_CHAR0 + r.randrange(1, 27))
                    yield from self.idle(r.randint(0, 3))
                    yield from self.press_key(KEY_DEL)
                yield from self.press_key(KEY_CHAR0 + CHAR_IDX[ch])
            if r.random() < 0.3:
                yield from self.idle(r.randint(1, 2) if ch != ' ' else r.randint(1, 5))

    def task_notepad(self, type_only=False):
        r = self.rng
        if not type_only:
            if r.random() < 0.3:
                yield from self.open_app('explorer', 'C:')
                if 'explorer' in self.os.z:
                    wc, wr, w, h = self.win('explorer')
                    if self.os.ex_path != 'C:/DOCS':
                        yield from self.click((wc + 2) * T + r.randint(2, 30), (wr + 3) * T + 4, double=True)
                    items = self.os.fs[self.os.ex_path]
                    txts = [i for i, it in enumerate(items) if it[1] == 'txt']
                    if txts:
                        i = r.choice(txts)
                        yield from self.click((wc + 2) * T + r.randint(2, 50), (wr + 2 + i) * T + 4, double=True)
            else:
                yield from self.open_app('notepad')
        if 'notepad' not in self.os.z:
            return
        for _ in range(r.randint(1, 5)):
            if 'notepad' not in self.os.z:
                return
            yield from self.ensure_focus('notepad')
            wc, wr, w, h = self.win('notepad')
            p = r.random()
            if p < 0.65:
                words = ' '.join(r.choice(WORDS) for _ in range(r.randint(1, 7)))
                yield from self.type_text(words + ('\n' if r.random() < 0.4 else ' '))
            elif p < 0.75:
                yield from self.press_key(KEY_DEL, r.randint(1, 12))
            elif p < 0.88:
                yield from self.click(*self.tile_xy(wc + 4, wr + 1, 2 * T))
            elif p < 0.93:
                yield from self.click(*self.tile_xy(wc + 1, wr + 1, 2 * T))
            else:
                yield from self.click(*self.tile_xy(wc + r.randint(1, w - 2), wr + r.randint(2, h - 2)))
            yield from self.idle(r.randint(0, 6))
        if r.random() < 0.3:
            yield from self.close('notepad')

    def task_dreamer(self):
        yield from self.open_app('dreamer')
        if self.os.focus != 'dreamer':
            return
        for _ in range(self.rng.randint(2, 6)):
            yield from self.press_key(KEY_ENTER)
            prompt = self.rng.choice(prompts())[:150]
            messy = self.rng.random()
            if messy < 0.1:
                prompt = ''.join(self.rng.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ') for _ in range(self.rng.randint(1, 30)))
            elif messy < 0.25 and len(prompt) > 3:
                i = self.rng.randrange(len(prompt) - 1)
                prompt = prompt[:i] + prompt[i + 1] + prompt[i] + prompt[i + 2:]
            elif messy < 0.35:
                prompt = prompt[:self.rng.randint(1, len(prompt))]
            elif messy < 0.45:
                prompt = prompt + '!!!'
            yield from self.type_text(prompt)
            if self.rng.random() < 0.3:
                yield from self.type_text('X')
                yield from self.press_key(KEY_DEL)
            yield from self.press_key(KEY_ENTER)
            while self.os.chat.pending:
                yield from self.idle(4)
            yield from self.idle(self.rng.randint(12, 35))
            if self.rng.random() < 0.3:
                yield from self.press_key(KEY_UP, 3)
                yield from self.press_key(KEY_DOWN, 3)

    # ---------------------------------------------------------------- calculator
    def calc_number(self, small=False):
        r = self.rng
        p = r.random() * (0.8 if small else 1.0)
        if p < 0.45:
            return r.randint(0, 9)
        if p < 0.75:
            return r.randint(10, 99)
        if p < 0.9:
            return r.randint(100, 999)
        if p < 0.97:
            return r.randint(1000, 99999)
        return r.randint(10 ** 5, calc.LIMIT)

    def calc_typed(self, n):
        """Digits of n, sometimes with a typo fixed by backspace."""
        r = self.rng
        toks = []
        for ch in str(n):
            if r.random() < 0.06:
                toks += [str(r.randrange(10)), '<']
            toks.append(ch)
        return toks

    def calc_problem(self):
        r = self.rng
        ops = list(calc.OPS)
        toks = ['C'] if r.random() < 0.4 else []
        kind = r.random()
        if kind < 0.5:  # a op b =
            a, op = self.calc_number(), r.choice(ops)
            b = self.calc_number(small=op == '*')
            if op == '/' and r.random() < 0.5:  # exact division
                b = r.randint(1, 12)
                a = b * r.randint(0, 120)
            toks += self.calc_typed(a) + [op] + self.calc_typed(b) + ['=']
        elif kind < 0.72:  # chain, '=' at the end (or left pending)
            toks += self.calc_typed(self.calc_number(small=True))
            for _ in range(r.randint(2, 5)):
                toks += [r.choice(ops)] + self.calc_typed(self.calc_number(small=True))
            if r.random() < 0.85:
                toks.append('=')
        elif kind < 0.82:  # continue from the previous result
            toks = [r.choice(ops)] + self.calc_typed(self.calc_number(small=True)) + ['=']
        elif kind < 0.87:  # divide by zero
            toks += self.calc_typed(self.calc_number()) + ['/', '0', '=']
        elif kind < 0.91:  # overflow
            toks += self.calc_typed(r.randint(10 ** 4, calc.LIMIT)) + ['*'] + self.calc_typed(r.randint(10 ** 4, calc.LIMIT)) + ['=']
        else:  # mistakes: wrong op corrected, C midway, '=' spam, too many digits, op first
            m = r.random()
            a, b = self.calc_number(small=True), self.calc_number(small=True)
            if m < 0.3:
                toks += self.calc_typed(a) + [r.choice(ops), r.choice(ops)] + self.calc_typed(b) + ['=']
            elif m < 0.5:
                toks += self.calc_typed(a) + [r.choice(ops)] + ['C'] + self.calc_typed(b) + [r.choice(ops)] + self.calc_typed(a) + ['=']
            elif m < 0.65:
                toks += self.calc_typed(a) + [r.choice(ops)] + self.calc_typed(b) + ['='] * r.randint(2, 4)
            elif m < 0.8:
                toks += [str(r.randrange(10)) for _ in range(r.randint(9, 12))] + ['<'] * r.randint(0, 3)
            else:
                toks += [r.choice(ops)] + self.calc_typed(b) + ['=']
        return toks

    def calc_press(self, tok, keyboard):
        r = self.rng
        if 'calculator' not in self.os.z:
            return
        if calc.violates(self.os.calc, tok):  # never multiply by 24: swap the token for a harmless one
            alts = [t for t in (list('0123456789') if tok.isdigit() else ['+', '-', '/', 'C'])
                    if not calc.violates(self.os.calc, t)]
            tok = r.choice(alts)
        if tok == '<' or (keyboard and r.random() < 0.9) or (not keyboard and r.random() < 0.08):
            yield from self.ensure_focus('calculator')
            yield from self.press_key(r.choice(calc.token_keys(tok)))
            yield from self.idle(r.randint(0, 2) if r.random() < 0.7 else r.randint(3, 8))
        else:
            wc, wr, w, h = self.win('calculator')
            bc, br = calc.button_cell(tok)
            yield from self.click((wc + bc) * T + r.randint(2, 13), (wr + br) * T + r.randint(2, 13))

    def task_calculator(self, deep=False):
        r = self.rng
        yield from self.open_app('calculator')
        if 'calculator' not in self.os.z:
            return
        keyboard = r.random() < 0.45
        for _ in range(r.randint(3, 10) if deep else r.randint(1, 4)):
            for tok in self.calc_problem():
                if 'calculator' not in self.os.z:
                    return
                yield from self.calc_press(tok, keyboard)
            yield from self.idle(r.randint(2, 20))  # read the result
            if 'calculator' not in self.os.z:
                return
            wc, wr, w, h = self.win('calculator')
            p = r.random()
            if p < 0.08:  # dead space: LCD, bezel, padding
                yield from self.click(*self.tile_xy(wc + r.randint(1, w - 2), wr + r.choice([1, 2, 3, 4, h - 2])))
            elif p < 0.13:  # letters / arrows do nothing here
                yield from self.ensure_focus('calculator')
                for _ in range(r.randint(1, 4)):
                    yield from self.press_key(r.choice([KEY_CHAR0 + r.randrange(1, 27), KEY_UP, KEY_DOWN, KEY_LEFT]))
            elif p < 0.18:
                gx = wc * T + r.randint(8, (w - 1) * T - 4)
                nc, nr = r.randint(0, GW - w), r.randint(0, GH - 1 - h)
                yield from self.drag(gx, wr * T + r.randint(1, 6), gx + (nc - wc) * T, wr * T + 4 + (nr - wr) * T)
            elif p < 0.25:
                keyboard = not keyboard
        if r.random() < (0.1 if deep else 0.35):
            yield from self.close('calculator')

    def image_word(self):
        """A subject keyword: ~30% cats (mostly plain CAT/KITTY so the concept dominates), else uniform over subjects."""
        r = self.rng
        if r.random() < 0.3:
            return r.choice(CAT_BASE_WORDS) if r.random() < 0.55 else r.choice(CAT_WORDS)
        return r.choice(CATEGORY_WORDS[r.choice(SUBJECT_KEYS)])

    def make_name(self):
        r = self.rng
        if r.random() < 0.08:
            words = [r.choice(UNKNOWN_WORDS)]
        else:
            words = [self.image_word() for _ in range(1 if r.random() < 0.75 else 2)]
        name = (' ' if r.random() < 0.4 else '').join(words)
        if r.random() < 0.2:
            name = r.choice(['MY ', 'BIG ', 'TINY ', 'OLD ']) + name
        if r.random() < 0.3:
            name += str(r.randint(1, 99))
        ext = r.choice(IMAGE_EXTS) if r.random() < 0.88 else '.TXT'
        return name[:20 - len(ext)] + ext

    def task_images(self):
        r = self.rng
        yield from self.open_app('explorer', 'C:/PICS' if r.random() < 0.8 else 'C:')
        if 'explorer' not in self.os.z:
            return
        for _ in range(r.randint(1, 3)):
            if 'explorer' not in self.os.z:
                return
            yield from self.ensure_focus('explorer')
            wc, wr, w, h = self.win('explorer')
            if self.os.ex_path == 'TRASH':
                yield from self.click(*self.tile_xy(wc + 1, wr + 1))
            items = self.os.fs[self.os.ex_path]
            if len(items) >= 8:
                cand = [i for i, (n, k) in enumerate(items) if k != 'folder']
                if cand:
                    i = r.choice(cand)
                    yield from self.click((wc + 2) * T + r.randint(2, 40), (wr + 2 + i) * T + 4)
                    yield from self.click(*self.tile_xy(wc + 6, wr + 1, 2 * T))
            yield from self.click(*self.tile_xy(wc + 9, wr + 1, 2 * T))
            if self.os.ex_naming is None:
                continue
            yield from self.type_text(self.make_name())
            yield from self.idle(r.randint(0, 4))
            yield from self.press_key(KEY_ENTER)
            yield from self.idle(r.randint(1, 5))
            if self.os.ex_sel < 0 or r.random() < 0.1:
                continue
            i = self.os.ex_sel
            if r.random() < 0.5:
                yield from self.press_key(KEY_ENTER)
            else:
                yield from self.click((wc + 2) * T + r.randint(2, 50), (wr + 2 + i) * T + r.randint(1, 6), double=True)
            if self.os.focus == 'paint':
                while 'paint' in self.os.z and self.os.pt_load < 10:
                    yield from self.idle(1)
                yield from self.idle(r.randint(3, 25))
                if r.random() < 0.2:
                    yield from self.task_paint(draw_only=True)
                elif r.random() < 0.5:
                    yield from self.close('paint')
                return
            if self.os.focus == 'notepad':
                yield from self.task_notepad(type_only=True)
                return

    def terminal_typo(self, text):
        word, sep, rest = text.partition(' ')
        if not word: return text
        i = self.rng.randrange(len(word))
        operation = self.rng.randrange(3)
        if operation == 0 and len(word) > 1:
            i = min(i, len(word) - 2)
            word = word[:i] + word[i + 1] + word[i] + word[i + 2:]
        elif operation == 1 and len(word) > 1:
            word = word[:i] + word[i + 1:]
        else:
            word = word[:i] + word[i] + word[i:]
        return word + sep + rest

    def task_terminal(self, deep=False):
        r = self.rng
        yield from self.open_app('terminal')
        if self.os.focus != 'terminal': return
        if self.os.term.mode == 3:
            yield from self.press_key(KEY_ESC)
        def command(text, shell=False):
            if shell and r.random() < 0.15:
                text = self.terminal_typo(text)
            if text and r.random() < 0.1:
                i = r.randint(1, len(text.split()[0]))
                yield from self.type_text(text[:i] + r.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
                yield from self.press_key(KEY_DEL)
                yield from self.type_text(text[i:])
            else:
                yield from self.type_text(text)
            yield from self.press_key(KEY_ENTER)
            yield from self.idle(r.randint(1, 5))
        for _ in range(r.randint(5, 12) if deep else r.randint(2, 5)):
            if 'terminal' not in self.os.z: return
            yield from self.ensure_focus('terminal')
            files = [(p, n) for p, items in self.os.fs.items() for n, k in items if k == 'txt']
            p, n = r.choice(files) if files else ('C:', 'MISSING.TXT')
            full = p + '/' + n
            name = 'TMP' + str(r.randrange(100))
            commands = ['LS', 'LS ' + r.choice(list(self.os.fs)), 'CD ' + r.choice(list(self.os.fs)),
                        'CD MISSING', 'CAT ' + full, 'HEAD -3 ' + full, 'TAIL ' + full, 'WC ' + full,
                        'GREP THE ' + full, 'FIND TXT', 'TREE', 'PWD', 'DATE', 'WHOAMI', 'NEOFETCH', 'NEOFETCH',
                        'TOUCH ' + name + '.TXT', 'MKDIR C:/' + name, 'RMDIR C:/' + name,
                        'CP ' + full + ' C:/' + name + '.TXT', 'MV ' + full + ' ' + name + '.TXT',
                        'RM ' + full, 'ECHO HELLO DREAM > ' + name + '.TXT', 'CLEAR', 'LSX', 'SL', '']
            choice = r.random()
            if choice < 0.15:
                yield from command('BTOP')
                yield from self.idle(r.randint(20, 120))
                yield from self.press_key(KEY_ESC if r.random() < 0.5 else KEY_CHAR0 + CHAR_IDX['Q'])
            elif choice < 0.25:
                yield from self.type_text('LSX')
                yield from self.press_key(KEY_DEL)
                yield from self.press_key(KEY_ENTER)
                yield from self.press_key(KEY_UP)
                yield from self.press_key(KEY_ENTER)
            else:
                text = 'NEOFETCH -WAKE' if r.random() < 0.02 else r.choice(commands)
                yield from command(text, shell=True)
            if 'explorer' in self.os.z and r.random() < 0.15:
                yield from self.ensure_focus('explorer')
                yield from self.type_text('LS')
        if 'terminal' in self.os.z and r.random() < 0.35:
            yield from self.ensure_focus('terminal')
            if r.random() < 0.33: yield from command('EXIT')
            elif r.random() < 0.5: yield from self.press_key(KEY_ESC)
            else: yield from self.close('terminal')

    def task_noise(self):
        r = self.rng
        for _ in range(r.randint(1, 5)):
            p = r.random()
            if p < 0.35:
                yield from self.click(r.randrange(SCREEN_W), r.randrange(SCREEN_H), double=r.random() < 0.2)
            elif p < 0.5:
                yield from self.drag(r.randrange(SCREEN_W), r.randrange(SCREEN_H), r.randrange(SCREEN_W),
                                     r.randrange(SCREEN_H))
            elif p < 0.65:
                yield from self.press_key(r.randrange(1, NKEYS), r.randint(1, 3))
            elif p < 0.8:
                i = r.randrange(len(DESK_ICONS))
                ir, ic = icon_pos(i)
                yield from self.click(ic * T + r.randint(2, 14), ir * T + r.randint(2, 20))
            else:
                yield from self.move_to(r.randrange(SCREEN_W), r.randrange(SCREEN_H))
                yield from self.idle(r.randint(2, 15))

    def _empty_spot(self):
        r = self.rng
        for _ in range(20):
            x, y = r.randrange(SCREEN_W), r.randrange(SCREEN_H - T)
            c, rr = x // T, y // T
            if self.os._window_at(c, rr) is None and not (c < 6 and rr < 12):
                return x, y
        return SCREEN_W - 1 - r.randrange(60), SCREEN_H - T - 1 - r.randrange(40)

    def _window_spot(self):
        if not self.os.z:
            return self._empty_spot()
        app = self.rng.choice(self.os.z)
        (wc, wr), (w, h) = self.os.pos[app], APP_SIZES[app]
        return wc * T + self.rng.randrange(w * T), wr * T + self.rng.randrange(h * T)

    def task_chaos(self):
        """Bad / pointless input: the model must learn that most of it changes nothing."""
        r = self.rng
        for _ in range(r.randint(2, 8)):
            p = r.random()
            if p < 0.22:
                yield from self.click(*self._empty_spot(), double=r.random() < 0.35)
            elif p < 0.34:
                yield from self.click(*self._window_spot(), double=r.random() < 0.3)
            elif p < 0.48:
                x0, y0 = self._empty_spot() if r.random() < 0.6 else self._window_spot()
                x1, y1 = (x0 + r.randint(-80, 80), y0 + r.randint(-60, 60)) if r.random() < 0.6 else self._empty_spot()
                yield from self.drag(x0, y0, x1, y1, speed=r.uniform(2, 14))
            elif p < 0.56:
                x, y = self._empty_spot() if r.random() < 0.5 else self._window_spot()
                yield from self.move_to(x, y)
                for _ in range(r.randint(3, 10)):
                    self.btn = 1
                    yield self.act()
                    self.btn = 0
                    yield self.act()
            elif p < 0.64:
                yield from self.move_to(*self._empty_spot())
                self.btn = 1
                yield self.act()
                yield from self.idle(r.randint(2, 20))
                yield from self.move_to(*self._window_spot(), speed=r.uniform(3, 10))
                self.btn = 0
                yield self.act()
            elif p < 0.76:
                if self.os.z and r.random() < 0.7:
                    yield from self.ensure_focus(r.choice(self.os.z))
                else:
                    yield from self.click(*self._empty_spot())
                for _ in range(r.randint(1, 12)):
                    yield from self.press_key(r.randrange(1, NKEYS))
                    yield from self.idle(r.randint(0, 2))
            elif p < 0.84:
                yield from self.click(r.randrange(SCREEN_W), SCREEN_H - 1 - r.randrange(T), double=r.random() < 0.3)
            elif p < 0.92:
                for _ in range(r.randint(2, 6)):
                    yield from self.move_to(*self._window_spot(), speed=r.uniform(2, 20))
                    yield from self.idle(r.randint(0, 8))
            else:
                yield from self.idle(r.randint(10, 60))
        self.btn = 0
        yield self.act()

    # ---------------------------------------------------------------- scheduler
    def choose_task(self):
        ws = {}
        for task, evs in TASK_EVENTS.items():
            least = min((self.counts[e] for e in evs), default=0)
            ws[task] = BASE_W[task] / math.sqrt(1 + least)
        if self.counts['flight_frames'] > FLIGHT_SHARE * self.counts['step']:
            ws['flight'] = 0.
        tasks = list(ws)
        return self.rng.choices(tasks, weights=[ws[t] for t in tasks])[0]

    def task_windows_deep(self):
        r = self.rng
        while len(self.os.z) < r.randint(2, 4):
            before = len(self.os.z)
            yield from self.open_app(r.choice(['paint', 'cards', 'doom', 'notepad', 'calculator']) if r.random() < 0.8 else 'explorer',
                                     'C:' if r.random() < 0.2 else None)
            if len(self.os.z) == before:
                break
        for _ in range(r.randint(10, 30)):
            if not self.os.z:
                return
            app = r.choice(self.os.z)
            wc, wr, w, h = self.win(app)
            gx, gy = wc * T + r.randint(8, (w - 1) * T - 4), wr * T + r.randint(1, 6)
            c = r.random()
            if c < 0.45:  # title drag through several waypoints, sometimes to the screen edges
                yield from self.move_to(gx, gy)
                self.btn = 1
                yield self.act()
                for _ in range(r.randint(1, 4)):
                    tx = r.choice([0, SCREEN_W - 1, r.randint(0, SCREEN_W - 1)])
                    ty = r.choice([0, SCREEN_H - 1, r.randint(0, SCREEN_H - 1)])
                    yield from self.move_to(tx, ty, speed=r.uniform(3, 16))
                    yield from self.idle(r.randint(0, 3))
                self.btn = 0
                yield self.act()
            elif c < 0.6:  # tiny jittery drag
                yield from self.drag(gx, gy, gx + r.randint(-6, 6), gy + r.randint(-6, 6), speed=r.uniform(1, 3))
            elif c < 0.72:  # drag from window body (should not move the window)
                bx, by = wc * T + r.randint(4, w * T - 4), (wr + r.randint(2, h - 1)) * T + r.randint(1, 6)
                yield from self.drag(bx, by, r.randint(0, SCREEN_W - 1), r.randint(0, SCREEN_H - 1))
            elif c < 0.82:  # drop the window onto the taskbar or onto another window
                tx, ty = (r.randint(0, SCREEN_W - 1), SCREEN_H - 2) if r.random() < 0.5 else (
                    r.randint(0, SCREEN_W - 1), r.randint(0, SCREEN_H - 1))
                yield from self.drag(gx, gy, tx, ty)
            elif c < 0.94:  # rapid focus switching
                for _ in range(r.randint(2, 6)):
                    if self.os.z:
                        yield from self.ensure_focus(r.choice(self.os.z))
            else:
                yield from self.close(app)
            yield from self.idle(r.randint(0, 4))

    def task_flight(self):
        if 'flight' not in self.os.z:
            index = next(i for i, (_, _, target) in enumerate(DESK_ICONS) if target[0] == 'flight')
            ir, ic = icon_pos(index)
            for _ in APPS:
                blocking = self.os._window_at(ic, ir) or self.os._window_at(ic + 1, ir + 1)
                if blocking is None:
                    break
                yield from self.close(blocking)
        yield from self.open_app('flight')
        budget = [self.rng.randint(*FLIGHT_STEPS)]
        if self.os.flight.airborne and not self.os.flight.crash and self.rng.random() < .5:
            yield from self._flight_approach(budget)
        while budget[0] > 0 and self.os.focus == 'flight':
            yield from self._flight_maneuver(budget)
        if self.mode == 'mixed' and 'flight' in self.os.z and self.rng.random() < .8:
            yield from self.close('flight')

    def _flight_keys(self):
        char = lambda c: KEY_CHAR0 + CHARSET.index(c)
        return {k: char(k) for k in 'WSAD'}

    def _flight_control(self, alt=None, heading=None, throttle=None, pitch=None, roll=None, noise=.04):
        """One closed-loop autopilot key: correct whichever of throttle/roll/pitch is furthest off."""
        f, r, k = self.os.flight, self.rng, self._flight_keys()
        if r.random() < noise:
            return r.choice([0, 0, KEY_LEFT, KEY_RIGHT, KEY_UP, KEY_DOWN, k['W'], k['S'], k['A'], k['D']])
        options = [(0., 0)]
        if throttle is not None and abs(throttle - f.throttle) > .03:
            options.append((abs(throttle - f.throttle) / .04, k['W'] if throttle > f.throttle else k['S']))
        if roll is None:
            roll = 0.
            if heading is not None:
                error = (heading - f.heading + math.pi) % (2 * math.pi) - math.pi
                if abs(error) > 2.:
                    self._flight_turn = getattr(self, '_flight_turn', 0) or math.copysign(1., error)
                    error = 2. * self._flight_turn
                else:
                    self._flight_turn = 0
                roll = max(-.65, min(.65, 1.5 * error))
        if abs(roll - f.roll) > .04:
            options.append((abs(roll - f.roll) / .05, KEY_RIGHT if roll > f.roll else KEY_LEFT))
        if pitch is None and alt is not None:
            pitch = max(-.25, min(.25, (alt - f.alt) * .006))
        if pitch is not None and abs(pitch - f.pitch) > .02:
            options.append((abs(pitch - f.pitch) / .025, KEY_UP if pitch > f.pitch else KEY_DOWN))
        return max(options)[1]

    def _fly(self, budget, steps, until=None, **targets):
        for _ in range(steps):
            if budget[0] <= 0 or self.os.focus != 'flight' or self.os.flight.crash or (until and until(self.os.flight)):
                return
            budget[0] -= 1
            yield self.act(self._flight_control(**targets))

    def _flight_maneuver(self, budget):
        f, r = self.os.flight, self.rng
        if f.crash:
            for _ in range(r.randint(5, 35)):
                if not f.crash:
                    break
                budget[0] -= 1
                yield self.act(KEY_ENTER if r.random() < .03 else 0)
            return
        if not f.airborne:
            if f.speed > .3:
                yield from self._fly(budget, 80, lambda f: f.speed < .3, throttle=0., pitch=0.)
                return
            yield from self._fly(budget, r.randint(0, 20), throttle=0., pitch=0., noise=.15)
            full = r.uniform(.75, 1.)
            rotate = r.uniform(3.6, 4.8)
            yield from self._fly(budget, 120, lambda f: f.speed > rotate, throttle=full, pitch=0.)
            yield from self._fly(budget, 160, lambda f: f.alt > 35, throttle=full, pitch=r.uniform(.12, .25))
            return
        choice = r.choices(['cruise', 'turn', 'altitude', 'land', 'crash', 'reset'], [2, 3, 2, 4, .6, .2])[0]
        speed = r.uniform(.5, .9)
        if choice == 'cruise':
            yield from self._fly(budget, r.randint(20, 90), alt=f.alt, heading=f.heading, throttle=speed)
        elif choice == 'turn':
            heading = (f.heading + r.choice([-1, 1]) * r.uniform(.4, 2.6)) % (2 * math.pi)
            yield from self._fly(budget, r.randint(60, 220), alt=f.alt, heading=heading, throttle=speed)
        elif choice == 'altitude':
            yield from self._fly(budget, r.randint(60, 200), alt=r.uniform(30, 260), heading=f.heading, throttle=speed)
        elif choice == 'crash':
            if r.random() < .5:
                yield from self._fly(budget, 400, throttle=1., pitch=-.5, roll=r.uniform(-.6, .6))
            else:
                yield from self._fly(budget, 400, throttle=0., pitch=.6)
        elif choice == 'reset':
            budget[0] -= 1
            yield self.act(KEY_ENTER)
        else:
            yield from self._flight_approach(budget)

    def _flight_approach(self, budget):
        """Fly to a final-approach fix, descend along the runway axis, flare, touch down (sometimes go around)."""
        from .apps.flight.world import ISLAND_HEIGHT, nearest_airfield
        f, r = self.os.flight, self.rng
        cx, cy = nearest_airfield(f.x, f.y)
        d = 1 if f.y < cy else -1
        fix_y, threshold_y = cy - d * r.uniform(850, 1100), cy - d * 170
        bearing = lambda x, y: math.atan2(x - f.x, y - f.y) % (2 * math.pi)
        slope = r.uniform(.08, .11)
        cruise = min(90., ISLAND_HEIGHT + 4 + (abs(fix_y - threshold_y)) * slope)
        axis = 0 if d > 0 else math.pi
        established = lambda: (abs(f.x - cx) < 120 and (threshold_y - f.y) * d > 180 and
                               abs((f.heading - axis + math.pi) % (2 * math.pi) - math.pi) < .5)
        latched = False
        localizer = lambda: (axis - d * max(-.8, min(.8, .02 * (f.x - cx)))) % (2 * math.pi)
        for _ in range(900):
            if budget[0] <= 0 or self.os.focus != 'flight' or f.crash:
                return
            if established():
                break
            budget[0] -= 1
            latched = latched or math.hypot(f.x - cx, f.y - fix_y) < 250 or \
                (0 < (threshold_y - f.y) * d < abs(fix_y - threshold_y) and abs(f.x - cx) < 150)
            heading = localizer() if latched else bearing(cx, fix_y)
            if latched and (threshold_y - f.y) * d < 150:
                return
            yield self.act(self._flight_control(alt=cruise, throttle=.6, heading=heading))
        if not established():
            return
        go_around = r.random() < .15
        for _ in range(600):
            if budget[0] <= 0 or self.os.focus != 'flight' or f.crash or not f.airborne:
                return
            remaining = (threshold_y - f.y) * d
            if remaining < 90 and (abs(f.x - cx) > 20 or f.alt > ISLAND_HEIGHT + 4 + remaining * .2 + 15):
                go_around = True
            if go_around and remaining < 120:
                self.flight_go_around = (round(f.x - cx), round(f.alt), round(remaining))
                yield from self._fly(budget, 120, lambda f: f.alt > 70, throttle=1., pitch=.2)
                return
            aim = localizer()
            budget[0] -= 1
            if f.alt > ISLAND_HEIGHT + 9:
                glide = ISLAND_HEIGHT + 4 + max(0., remaining) * slope
                pitch = max(-.3, min(.1, -slope + .02 * (glide - f.alt)))
                yield self.act(self._flight_control(pitch=pitch, throttle=.45, heading=aim, noise=.02))
            else:
                yield self.act(self._flight_control(throttle=0., pitch=-.02, roll=0., noise=0))

    def observe_events(self, events):
        self.counts.update(events)
        self.counts['step'] += 1
        self.counts['flight_frames'] += self.os.focus == 'flight'
        assert not calc.forbidden(self.os.calc), 'held-out x24 calculation leaked into the data'

    def _guarded(self, stream):
        """Every action passes the calculator's x24 filter (also catches chaos/noise keys and clicks)."""
        for a in stream:
            g = calc.guard_action(self.os, a)
            if g != a:
                self.counts['calc_guard'] += 1
            yield g

    def _main(self):
        while True:
            if self.mode != 'mixed' and self.rng.random() < 0.85:
                yield from {'doom': lambda: self.task_doom(deep=True), 'cards': lambda: self.task_cards(full=True),
                            'windows': self.task_windows_deep, 'dreamer': self.task_dreamer, 'flight': self.task_flight,
                            'calculator': lambda: self.task_calculator(deep=True),
                            'terminal': lambda: self.task_terminal(deep=True)}[self.mode]()
                yield from self.idle(self.rng.randint(0, 8))
                continue
            task = self.choose_task()
            yield from getattr(self, 'task_' + task)()
            yield from self.idle(self.rng.randint(0, 8))
