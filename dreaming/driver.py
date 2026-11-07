"""Smart scripted user: hierarchical goal-directed tasks + human-like mouse motion + coverage-driven task choice."""
import math
import random
from collections import Counter
from .os_sim import (APP_SIZES, APPS, DESK_ICONS, icon_pos, CANVAS_W, CANVAS_H, TASKBAR_BTN_W, KEY_UP, KEY_DOWN,
                     KEY_LEFT, KEY_RIGHT, KEY_FIRE, KEY_ENTER, KEY_ESC, KEY_DEL, NKEYS)
from .tiles import T, GW, GH, SCREEN_W, SCREEN_H
from . import doom as dm

TASK_EVENTS = {
    'explorer': ['ex_open_folder', 'ex_delete', 'ex_restore', 'ex_empty_trash', 'ex_open_png', 'ex_up', 'ex_select'],
    'paint': ['paint_stroke', 'paint_color', 'paint_save', 'paint_clear'],
    'cards': ['cards_move', 'cards_found', 'cards_flip', 'cards_draw'],
    'doom': ['doom_fire', 'doom_kill', 'doom_hurt'],
    'windows': ['drag_move', 'close_explorer', 'close_paint', 'close_cards', 'close_doom', 'taskbar_focus'],
    'noise': ['desk_select'],
}
BASE_W = {'explorer': 1.0, 'paint': 1.0, 'cards': 1.0, 'doom': 1.6, 'windows': 0.8, 'noise': 0.4}


class Driver:
    def __init__(self, os_, seed):
        self.os = os_
        self.rng = random.Random(seed)
        self.x, self.y = SCREEN_W // 2, SCREEN_H // 2
        self.btn = 0
        self.counts = Counter()
        self.stream = self._main()

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

    def move_to(self, tx, ty, speed=None):
        r = self.rng
        tx, ty = min(max(int(tx), 0), SCREEN_W - 1), min(max(int(ty), 0), SCREEN_H - 1)
        x0, y0 = self.x, self.y
        dist = math.hypot(tx - x0, ty - y0)
        if dist < 1:
            return
        speed = speed or r.uniform(6, 22)
        n = int(min(max(dist / speed, 2), 30))
        overshoot = r.random() < 0.2 and dist > 40
        ex, ey = (tx + (tx - x0) * 0.08, ty + (ty - y0) * 0.08) if overshoot else (tx, ty)
        bend = r.uniform(-0.25, 0.25) * dist
        nx, ny = -(ty - y0) / dist, (tx - x0) / dist
        cx, cy = (x0 + ex) / 2 + nx * bend, (y0 + ey) / 2 + ny * bend
        for i in range(1, n + 1):
            s = i / n
            s = s * s * (3 - 2 * s)
            bx = (1 - s) ** 2 * x0 + 2 * (1 - s) * s * cx + s * s * ex
            by = (1 - s) ** 2 * y0 + 2 * (1 - s) * s * cy + s * s * ey
            jitter = 1 if i < n else 0
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

    def drag(self, x0, y0, x1, y1, speed=None):
        yield from self.move_to(x0, y0)
        self.btn = 1
        yield self.act()
        yield from self.move_to(x1, y1, speed=speed or self.rng.uniform(4, 12))
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
                dbl = kind in ('folder', 'png', 'exe') and r.random() < 0.6 and not trash
                yield from self.click((wc + 2) * T + r.randint(2, 60), (wr + 2 + i) * T + r.randint(1, 6), double=dbl)
                if self.os.focus == 'paint':
                    yield from self.task_paint(draw_only=True)
                    return
            elif p < 0.5 and self.os.ex_sel >= 0:
                if r.random() < 0.6:
                    yield from self.click(*self.tile_xy(wc + 6, wr + 1, 2 * T))
                else:
                    yield from self.press_key(KEY_DEL)
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
        if 'paint' not in self.os.z:
            return
        for _ in range(r.randint(2, 8)):
            if 'paint' not in self.os.z:
                return
            yield from self.ensure_focus('paint')
            wc, wr, w, h = self.win('paint')
            p = r.random()
            if p < 0.25:
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

    def task_cards(self):
        r = self.rng
        yield from self.open_app('cards')
        if 'cards' not in self.os.z:
            return
        for _ in range(r.randint(5, 30)):
            if 'cards' not in self.os.z:
                return
            yield from self.ensure_focus('cards')
            o = self.os
            if o.c_won:
                yield from self.click(*self.card_xy('stock'))
                continue
            moves = self.cards_moves()
            if r.random() < 0.08 or (not moves and r.random() < 0.1):
                k = r.randrange(7)
                yield from self.click(*self.card_xy('tab', k, max(0, len(o.tab[k]) - 1)))
                yield from self.click(*self.card_xy('tab', r.randrange(7), 0))
            elif moves and r.random() < 0.85:
                moves.sort(key=lambda m: -m[0] + r.random() * 1.5)
                _, src, dst = moves[0]
                if dst[0] == 'found' and r.random() < 0.5:
                    xy = self.card_xy('waste') if src[0] == 'waste' else self.card_xy('tab', src[1], src[2])
                    yield from self.click(*xy, double=True)
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
        if r.random() < 0.25:
            yield from self.close('cards')

    def task_doom(self):
        r = self.rng
        yield from self.open_app('doom')
        if 'doom' not in self.os.z:
            return
        steps = r.randint(40, 250)
        wander_turn, wander_left = 0, 0
        for _ in range(steps):
            if 'doom' not in self.os.z:
                return
            if self.os.focus != 'doom':
                yield from self.ensure_focus('doom')
                continue
            d = self.os.doom
            if r.random() < 0.05:
                yield self.act(r.randrange(NKEYS - 3))
                continue
            targets = [(depth, sx) for imp, (sx, size, depth) in zip(d.imps, d._project_all())
                       if imp['state'] != 2 and depth > 0 and 0 <= sx < dm.VW and depth < 7]
            if targets and r.random() < 0.85:
                depth, sx = min(targets)
                if abs(sx - dm.VW // 2) > 6:
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
            center = d.raycast()[dm.VW // 2][4]
            if wander_left > 0:
                wander_left -= 1
                yield self.act(wander_turn)
            elif center < 1.2 or r.random() < 0.04:
                wander_turn, wander_left = r.choice([KEY_LEFT, KEY_RIGHT]), r.randint(3, 15)
                yield self.act(wander_turn)
            else:
                yield self.act(KEY_UP if r.random() < 0.9 else KEY_DOWN)
        if r.random() < 0.3:
            yield from self.close('doom')

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
                yield from self.press_key(r.randrange(1, NKEYS - 1), r.randint(1, 3))
            elif p < 0.8:
                i = r.randrange(len(DESK_ICONS))
                ir, ic = icon_pos(i)
                yield from self.click(ic * T + r.randint(2, 14), ir * T + r.randint(2, 20))
            else:
                yield from self.move_to(r.randrange(SCREEN_W), r.randrange(SCREEN_H))
                yield from self.idle(r.randint(2, 15))

    # ---------------------------------------------------------------- scheduler
    def choose_task(self):
        ws = {}
        for task, evs in TASK_EVENTS.items():
            least = min(self.counts[e] for e in evs)
            ws[task] = BASE_W[task] / math.sqrt(1 + least)
        tasks = list(ws)
        return self.rng.choices(tasks, weights=[ws[t] for t in tasks])[0]

    def observe_events(self, events):
        self.counts.update(events)

    def _main(self):
        while True:
            task = self.choose_task()
            yield from getattr(self, 'task_' + task)()
            yield from self.idle(self.rng.randint(0, 8))
