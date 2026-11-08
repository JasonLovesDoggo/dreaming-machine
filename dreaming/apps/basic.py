"""Small integer BASIC with a bounded, frame-driven interpreter."""
import re


class Basic:
    def __init__(self, terminal):
        self.term = terminal
        self.program = {}
        self.running = False
        self.vars = {}
        self.loops = {}
        self.line = 0

    def expr(self, source):
        tokens = re.findall(r'-?\d+|[A-Z]+|[-/]', source)
        if ''.join(tokens) != source.replace(' ', '') or not tokens:
            raise ValueError('SYNTAX ERROR')
        def value(s):
            if re.fullmatch(r'-?\d+', s):
                return int(s)
            if len(s) == 1 and s.isalpha():
                return self.vars.get(s, 0)
            raise ValueError('SYNTAX ERROR')
        n = value(tokens[0])
        if len(tokens) % 2 == 0:
            raise ValueError('SYNTAX ERROR')
        for i in range(1, len(tokens), 2):
            op, v = tokens[i], value(tokens[i + 1])
            if op == 'ADD': n += v
            elif op in ('SUB', '-'): n -= v
            elif op == 'MUL': n *= v
            elif op in ('DIV', '/', 'MOD'):
                if not v: raise ValueError('DIV BY ZERO')
                q = abs(n) // abs(v) * (-1 if (n < 0) != (v < 0) else 1)
                n = n - q * v if op == 'MOD' else q
            else: raise ValueError('SYNTAX ERROR')
            n = max(-999999999, min(n, 999999999))
        return n

    def load(self, text):
        self.program = {}
        for line in text.splitlines():
            self.edit(line)

    def edit(self, source):
        m = re.fullmatch(r'(\d+)(?:\s+(.*))?', source)
        if not m:
            raise ValueError('SYNTAX ERROR')
        n = int(m[1])
        if m[2]: self.program[n] = m[2]
        else: self.program.pop(n, None)

    def start(self, events):
        self.order = sorted(self.program)
        self.pc, self.total = 0, 0
        self.vars, self.loops = {}, {}
        self.running = True
        events.append('basic_run')

    def finish(self, events, message='OK'):
        self.running = False
        self.term.write(message)
        events.append('basic_done')

    def jump(self, n):
        if n not in self.program: raise ValueError(f'NO SUCH LINE {n}')
        self.pc = self.order.index(n)

    def statement(self, source):
        cmd, _, arg = source.partition(' ')
        if cmd == 'REM': return
        if cmd == 'PRINT':
            try: out = str(self.expr(arg))
            except ValueError as e:
                if str(e) != 'SYNTAX ERROR': raise
                out = arg
            self.term.write(out)
        elif cmd == 'RPT':
            tokens = arg.split()
            for i in range(1, len(tokens)):
                char = tokens[i]
                if len(char) != 1: continue
                try:
                    count = self.expr(' '.join(tokens[:i]))
                    indent = self.expr(' '.join(tokens[i + 1:]) or '0')
                except ValueError as e:
                    if str(e) != 'SYNTAX ERROR': raise
                    continue
                self.term.write(' ' * max(0, min(36, indent)) + char * max(0, min(360, count)))
                break
            else: raise ValueError('SYNTAX ERROR')
        elif cmd == 'LET':
            m = re.fullmatch(r'([A-Z]) (.+)', arg)
            if not m: raise ValueError('SYNTAX ERROR')
            self.vars[m[1]] = self.expr(m[2])
        elif cmd == 'GOTO': self.jump(int(arg))
        elif cmd == 'IF':
            m = re.fullmatch(r'(.+) (GT|LT|EQ|NE|GE|LE|>) (.+) THEN (\d+)', arg)
            if not m: raise ValueError('SYNTAX ERROR')
            a, b = self.expr(m[1]), self.expr(m[3])
            if {'GT': a > b, '>': a > b, 'LT': a < b, 'EQ': a == b,
                'NE': a != b, 'GE': a >= b, 'LE': a <= b}[m[2]]:
                self.jump(int(m[4]))
        elif cmd == 'FOR':
            m = re.fullmatch(r'([A-Z]) (.+?) TO (.+?)(?: STEP (.+))?', arg)
            if not m: raise ValueError('SYNTAX ERROR')
            v, lo, hi, step = m[1], self.expr(m[2]), self.expr(m[3]), self.expr(m[4] or '1')
            if not step: raise ValueError('SYNTAX ERROR')
            self.vars[v] = lo
            self.loops[v] = (self.pc, hi, step)
            if (step > 0 and lo > hi) or (step < 0 and lo < hi):
                depth = 1
                while self.pc < len(self.order):
                    s = self.program[self.order[self.pc]]
                    self.pc += 1
                    if s.startswith('FOR '): depth += 1
                    if s.startswith('NEXT '): depth -= 1
                    if not depth: break
                self.loops.pop(v)
        elif cmd == 'NEXT':
            if arg not in self.loops: raise ValueError('NEXT WITHOUT FOR')
            pc, hi, step = self.loops[arg]
            self.vars[arg] += step
            if (step > 0 and self.vars[arg] <= hi) or (step < 0 and self.vars[arg] >= hi): self.pc = pc
            else: self.loops.pop(arg)
        elif cmd == 'END': self.pc = len(self.order)
        else: raise ValueError('SYNTAX ERROR')

    def tick(self, events):
        for _ in range(6):
            if not self.running: return
            if self.pc >= len(self.order):
                self.finish(events)
                return
            self.line = self.order[self.pc]
            self.pc += 1
            self.total += 1
            try:
                if self.total > 3000: raise ValueError('OUT OF TIME')
                self.statement(self.program[self.line])
            except (ValueError, OverflowError) as e:
                message = str(e)
                if message not in ('OUT OF TIME', 'NEXT WITHOUT FOR') and not message.startswith('NO SUCH LINE'):
                    message = ('DIV BY ZERO' if message == 'DIV BY ZERO' else 'SYNTAX ERROR') + f' IN {self.line}'
                self.term.error(message, events)
                self.finish(events, 'READY')
                return
