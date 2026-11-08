"""Terminal shell sharing the desktop filesystem."""
import re
from .basic import Basic
from .. import pictures as pics
from ..tiles import ATLAS, CHARSET


class Terminal:
    def __init__(self, os):
        self.os = os
        self.cwd = 'C:'
        self.input = ''
        self.lines = []
        self.history = []
        self.hist = 0
        self.mode = 0
        self.basic = Basic(self)
        self.write('DREAM TERMINAL - HELP FOR COMMANDS')

    def write(self, text):
        for line in str(text).split('\n'):
            self.lines.extend([line[i:i + 36] for i in range(0, len(line), 36)] or [''])
        self.lines = self.lines[-40:]

    def error(self, text, events):
        self.write(text)
        events.append('term_error')

    def path(self, text):
        if text in ('/', 'C:'): return 'C:'
        if text == 'TRASH' or text.startswith('TRASH/'): return text
        if self.cwd == 'TRASH' and not text.startswith(('C:', '/')):
            if text == '..': return 'C:'
            if text.startswith('../'): return self.path('C:/' + text[3:])
            return 'TRASH/' + text
        base = [] if text.startswith('C:') or text.startswith('/') else self.cwd.replace('C:', '').strip('/').split('/')
        parts = text.removeprefix('C:').strip('/').split('/')
        for part in parts:
            if part == '..': base = base[:-1]
            elif part and part != '.': base.append(part)
        return 'C:' + ''.join('/' + p for p in base if p)

    def file(self, text):
        path = self.path(text)
        parent, _, name = path.rpartition('/')
        if not parent: parent, name = self.cwd, text
        kind = next((k for n, k in self.os.fs.get(parent, []) if n == name), None)
        return parent, name, kind

    def exists(self, name):
        return any(n == name for items in self.os.fs.values() for n, _ in items)

    def changed(self, events):
        self.os.ex_sel = min(self.os.ex_sel, len(self.os.fs[self.os.ex_path]) - 1)
        events.append('term_fs')

    def write_file(self, target, text, events):
        p, n, k = self.file(target)
        if p not in self.os.fs: raise ValueError('NO SUCH DIR')
        if not n or len(n) > 20: raise ValueError('INVALID NAME')
        if k and k != 'txt': raise ValueError('BINARY FILE')
        if not k:
            if self.exists(n): raise ValueError('FILE EXISTS')
            if len(self.os.fs[p]) >= 9: raise ValueError('NO SPACE LEFT')
            self.os.fs[p].append((n, 'txt'))
        self.os.texts[n] = text
        if self.os.np_file == n: self.os.np_text = text
        self.changed(events)

    def read(self, target, cmd):
        p, n, k = self.file(target)
        if not k: raise ValueError(f'{cmd}: NO SUCH FILE')
        if k != 'txt': raise ValueError(f'{cmd}: BINARY FILE')
        return self.os.texts.get(n, '')

    def key(self, key, events):
        from ..os_sim import KEY_ESC, KEY_DEL, KEY_UP, KEY_ENTER, KEY_CHAR0
        if self.mode == 3:
            if key == KEY_ESC or key == KEY_CHAR0 + CHARSET.index('Q'):
                self.mode = 0
                events.append('btop_close')
            return
        if self.basic.running:
            if key == KEY_ESC:
                self.basic.running = False
                self.write(f'BREAK IN {self.basic.line}')
                events.append('basic_break')
            return
        if key == KEY_ESC:
            if self.mode == 1: self.mode = 0; self.write('SHELL READY')
            else: self.os.close_app('terminal', events)
        elif key == KEY_DEL: self.input = self.input[:-1]
        elif key == KEY_UP and self.history:
            self.hist = max(0, self.hist - 1)
            self.input = self.history[self.hist]
        elif key >= KEY_CHAR0 and key < KEY_CHAR0 + len(CHARSET):
            self.input = (self.input + CHARSET[key - KEY_CHAR0])[:60]
            events.append('term_type')
        elif key == KEY_ENTER:
            source, self.input = self.input.strip(), ''
            self.write(('BASIC>' if self.mode == 1 else self.cwd + '>') + source)
            if source:
                self.history = (self.history + [source])[-8:]
                self.hist = len(self.history)
                events.append('term_cmd')
                try:
                    if self.mode == 1: self.repl(source, events)
                    else: self.command(source, events)
                except ValueError as e: self.error(str(e), events)

    def repl(self, source, events):
        cmd, _, arg = source.partition(' ')
        if cmd.isdigit(): self.basic.edit(source)
        elif cmd == 'RUN': self.basic.start(events)
        elif cmd == 'LIST':
            for n in sorted(self.basic.program): self.write(f'{n} {self.basic.program[n]}')
        elif cmd == 'NEW': self.basic.program = {}
        elif cmd in ('BYE', 'EXIT'): self.mode = 0
        elif cmd == 'SAVE':
            if not arg: raise ValueError('USAGE: SAVE NAME.BAS')
            self.write_file(arg, '\n'.join(f'{n} {s}' for n, s in sorted(self.basic.program.items())), events)
        elif cmd == 'LOAD': self.basic.load(self.read(arg, 'LOAD'))
        else:
            self.basic.order = sorted(self.basic.program)
            self.basic.pc = 0
            self.basic.statement(source)

    def neofetch(self, wake, events):
        minutes = self.os.t // 60
        memory = 32 + (self.os.t // 4 + len(self.os.z) * 7) % 96
        info = ['DREAMER/DREAMING', '-----', 'OS: DREAMOS 1.0',
                'KERNEL: TILES 32X24', f'UPTIME: {minutes} MIN', 'SHELL: DREAMSH',
                'RES: 256X192', 'CPU: 1 NEURAL CORE', 'GPU: IMAGINARY',
                f'MEM: {memory} / 256 MB', f'APPS: {len(self.os.z)}']
        if wake:
            info = ['DREAMER/DREAMING', '-----', 'HOST: INSIDE THE MODEL',
                    'SHELL: DREAMSH', 'RES: 256X192', 'CPU: 1 NEURAL CORE',
                    'GPU: IMAGINARY', f'MEM: {memory} / 256 MB',
                    'YOU TRIED TO WAKE UP.', 'BUT THIS COMPUTER IS A DREAM.',
                    'NOTHING HERE IS REAL. KEEP PLAYING.']
            events.append('neofetch_egg')
        for i, text in enumerate(info):
            if i < 3: self.lines.append((i, text))
            else: self.write(text)
        self.lines = self.lines[-40:]
        events.append('neofetch')

    def command(self, source, events):
        from ..os_sim import PATHS
        cmd, _, rest = source.partition(' ')
        args = rest.split()
        def usage(n, hint):
            if len(args) != n: raise ValueError('USAGE: ' + cmd + ' ' + hint)
        if cmd == 'HELP':
            self.write('LS CD PWD TREE FIND CAT HEAD TAIL\nWC GREP ECHO TOUCH CP MV RM\nMKDIR RMDIR DATE WHOAMI CLEAR\nBASIC RUN BTOP NEOFETCH EXIT')
        elif cmd == 'NEOFETCH':
            if args not in ([], ['-WAKE']): raise ValueError('USAGE: NEOFETCH')
            self.neofetch(args == ['-WAKE'], events)
        elif cmd == 'PWD': self.write(self.cwd)
        elif cmd == 'WHOAMI': self.write('DREAMER')
        elif cmd == 'DATE': self.write(f'SUN NOV 08 2026 12:{self.os.t // 60 % 60:02}')
        elif cmd == 'CLEAR': self.lines = []; events.append('term_clear')
        elif cmd == 'EXIT': self.os.close_app('terminal', events)
        elif cmd == 'BASIC': self.mode = 1; self.write('BASIC READY')
        elif cmd == 'RUN':
            usage(1, 'FILE.BAS')
            self.basic.load(self.read(args[0], 'RUN'))
            self.basic.start(events)
        elif cmd == 'BTOP': self.mode = 3; events.append('btop_open')
        elif cmd == 'CD':
            usage(1, 'PATH')
            p = self.path(args[0])
            if p not in self.os.fs: raise ValueError('CD: NO SUCH DIR')
            self.cwd = p
        elif cmd == 'LS':
            p = self.path(rest) if rest else self.cwd
            if p not in self.os.fs: raise ValueError('LS: NO SUCH DIR')
            self.write('\n'.join(n + ('/' if k == 'folder' else '') for n, k in self.os.fs[p]) or 'EMPTY')
        elif cmd in ('TREE', 'FIND'):
            if cmd == 'FIND': usage(1, 'PAT')
            for p, items in self.os.fs.items():
                if cmd == 'TREE': self.write(p)
                for n, k in items:
                    if cmd == 'TREE' or rest in n: self.write(p + '/' + n + ('/' if k == 'folder' else ''))
        elif cmd in ('CAT', 'HEAD', 'TAIL', 'WC'):
            count = 5
            if cmd in ('HEAD', 'TAIL') and args and re.fullmatch(r'-\d+', args[0]): count = min(1000, int(args.pop(0)[1:]))
            usage(1, 'FILE')
            text = self.read(args[0], cmd)
            lines = text.splitlines()
            if cmd == 'WC': self.write(f'{len(lines)} {len(text.split())} {len(text)}')
            elif cmd == 'HEAD': self.write('\n'.join(lines[:count]))
            elif cmd == 'TAIL': self.write('\n'.join(lines[-count:] if count else []))
            else: self.write(text)
        elif cmd == 'GREP':
            if len(args) not in (1, 2): raise ValueError('USAGE: GREP PAT FILE')
            targets = [args[1]] if len(args) == 2 else [n for n, k in self.os.fs[self.cwd] if k == 'txt']
            for target in targets:
                for line in self.read(target, 'GREP').splitlines():
                    if args[0] in line: self.write((target + ':' if len(args) == 1 else '') + line)
        elif cmd == 'ECHO':
            if '>' in rest:
                text, target = rest.split('>', 1)
                if not target.strip(): raise ValueError('USAGE: ECHO TEXT > FILE')
                self.write_file(target.strip(), text.rstrip(), events)
            else: self.write(rest)
        elif cmd == 'MKDIR':
            usage(1, 'NAME')
            p, n, k = self.file(args[0])
            if p != 'C:': raise ValueError('MKDIR: TOO DEEP')
            if not re.fullmatch(r'[A-Z0-9_]{1,16}', n): raise ValueError('MKDIR: INVALID NAME')
            if self.exists(n): raise ValueError('MKDIR: FILE EXISTS')
            if len(self.os.fs[p]) >= 9: raise ValueError('NO SPACE LEFT')
            self.os.fs[p].append((n, 'folder'))
            self.os.fs['C:/' + n] = []
            self.changed(events)
        elif cmd == 'RMDIR':
            usage(1, 'PATH')
            p = self.path(rest)
            if p in PATHS: raise ValueError('RMDIR: PERMISSION DENIED')
            if p not in self.os.fs: raise ValueError('RMDIR: NO SUCH DIR')
            if self.os.fs[p]: raise ValueError('RMDIR: NOT EMPTY')
            del self.os.fs[p]
            self.os.fs['C:'].remove((p.split('/')[-1], 'folder'))
            if self.cwd == p: self.cwd = 'C:'
            if self.os.ex_path == p: self.os.ex_path = 'C:'
            for n, origin in list(self.os.origin.items()):
                if origin == p: self.os.origin[n] = 'C:'
            self.changed(events)
        elif cmd == 'TOUCH':
            usage(1, 'FILE')
            p, n, k = self.file(rest)
            if k: return
            if pics.is_image(n):
                self.write_file(rest, '', events)
                self.os.fs[p].remove((n, 'txt'))
                self.os.fs[p].append((n, 'png'))
                self.os.texts.pop(n, None)
                self.os.pictures[n] = pics.generate(n, self.os.pic_rng)
            else: self.write_file(rest, '', events)
        elif cmd == 'RM':
            usage(1, 'FILE')
            p, n, k = self.file(rest)
            if k == 'folder': raise ValueError('RM: IS A DIRECTORY')
            if not k: raise ValueError('RM: NO SUCH FILE')
            self.os._delete_file(p, self.os.fs[p].index((n, k)), events)
            self.changed(events)
        elif cmd in ('CP', 'MV'):
            usage(2, 'SRC DST')
            p, n, k = self.file(args[0])
            if not k: raise ValueError(cmd + ': NO SUCH FILE')
            if k == 'folder': raise ValueError(cmd + ': IS A DIRECTORY')
            dest = self.path(args[1])
            if dest in self.os.fs: q, m = dest, n
            else: q, m, _ = self.file(args[1])
            if q not in self.os.fs: raise ValueError(cmd + ': NO SUCH DIR')
            if self.exists(m) and not (cmd == 'MV' and m == n): raise ValueError(cmd + ': FILE EXISTS')
            if not m or len(m) > 20: raise ValueError(cmd + ': INVALID NAME')
            if len(self.os.fs[q]) >= 9 and not (cmd == 'MV' and p == q): raise ValueError('NO SPACE LEFT')
            if cmd == 'CP' and m == n: raise ValueError('CP: FILE EXISTS')
            if cmd == 'MV': self.os.fs[p].remove((n, k))
            self.os.fs[q].append((m, k))
            store = self.os.pictures if k == 'png' else self.os.texts
            if n in store:
                store[m] = store[n].copy() if k == 'png' else store[n]
                if cmd == 'MV' and m != n: store.pop(n)
            if cmd == 'MV':
                if self.os.pt_file == n: self.os.pt_file = m
                if self.os.np_file == n: self.os.np_file = m
                origin = self.os.origin.pop(n, p)
                if q == 'TRASH': self.os.origin[m] = origin
            self.changed(events)
        else: raise ValueError('UNKNOWN COMMAND: ' + cmd)

    def tick(self, events):
        if self.basic.running: self.basic.tick(events)

    def render(self, g, wc, wr):
        from ..os_sim import put_text
        if self.mode == 3:
            put_text(g, wr + 1, wc + 1, 'BTOP - Q TO QUIT', 'term')
            def metric(pid): return ((self.os.t // 4 * 37 + pid * 71 + self.os.seed * 13) % 97)
            for row, label in enumerate(['CPU0', 'CPU1', 'CPU2', 'CPU3', 'MEM'], 2):
                v = metric(row)
                put_text(g, wr + row, wc + 1, label, 'term')
                color = 'green' if v < 50 else 'yellow' if v < 80 else 'red'
                for j in range(7):
                    g[wr + row, wc + 4 + j] = ATLAS[('term_bar', color, max(0, min(8, v * 56 // 100 - j * 8))) ]
                put_text(g, wr + row, wc + 12, f'{v:2} PCT', 'term')
            put_text(g, wr + 7, wc + 1, 'PID NAME          CPU MEM', 'term')
            names = ['INIT', 'DREAMD'] + [a.upper() for a in self.os.z]
            for i, name in enumerate(names[:5]):
                cpu = min(99, metric(i) + (30 if name == 'DOOM' else 0))
                put_text(g, wr + 8 + i, wc + 1, f'{i + 1:2} {name:14} {cpu:2} {metric(i + 9):2}', 'term')
            return
        prompt = ('BASIC>' if self.mode == 1 else self.cwd + '>') + self.input + '_'
        tail = [prompt[i:i + 36] for i in range(0, len(prompt), 36)]
        lines = self.lines[-(12 - len(tail)):] + tail
        for i, line in enumerate(lines[-12:]):
            if isinstance(line, tuple):
                logo_row, text = line
                for c in range(7): g[wr + 1 + i, wc + 1 + c] = ATLAS[('neo_logo', logo_row, c)]
                put_text(g, wr + 1 + i, wc + 8, text, 'term')
            else: put_text(g, wr + 1 + i, wc + 1, line, 'term')
