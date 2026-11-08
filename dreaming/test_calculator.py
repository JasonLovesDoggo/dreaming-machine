"""Calculator logic + the held-out x24 rule. Run: python -m pytest dreaming/test_calculator.py"""
import random
import numpy as np
from .apps import calculator as calc
from .apps.calculator import Calc, forbidden
from .driver import Driver
from .os_sim import MicroOS, NKEYS, KEY_ESC, STATE_VOCAB
from .tiles import ATLAS, T, VOCAB


def run(seq):
    c = Calc()
    for tok in seq:
        c.press(tok)
    return c.display()


def test_basic_arithmetic():
    assert run('12+30=') == '42'
    assert run('7-9=') == '-2'
    assert run('6*7=') == '42'
    assert run('17/5=') == '3'
    assert run('0-17/5=') == '-3'  # truncates toward zero
    assert run('2+3*4=') == '20'  # left to right, like a pocket calculator
    assert run('5+-3=') == '2'  # last operator wins
    assert run('9=') == '9'
    assert run('8+=') == '16'  # missing operand reuses the display
    assert run('12+3=4') == '4'  # typing after '=' starts over
    assert run('12+3=*2=') == '30'  # operator after '=' continues from the result
    assert run('123<4') == '124'
    assert run('5+3C') == '0'


def test_errors_and_limits():
    assert run('5/0=') == 'ERR'
    assert run('99999999+1=') == 'ERR'
    assert run('5/0=+') == 'ERR'
    assert run('5/0=7') == '7'
    assert run('1234567890') == '12345678'


def test_forbidden_rule():
    assert forbidden_after('24*') and forbidden_after('3*24') and forbidden_after('20+4*') and forbidden_after('6*4*')
    assert forbidden_after('0-24*') and forbidden_after('24+*')
    assert not forbidden_after('24+5=') and not forbidden_after('48/24=') and not forbidden_after('3*25=')
    assert not forbidden_after('6*4=')  # 24 as a result is fine; it just can't be multiplied


def forbidden_after(seq):
    c = Calc()
    hit = False
    for tok in seq:
        c.press(tok)
        hit |= forbidden(c)
    return hit


def lcd_text(grid, os_):
    wc, wr = os_.pos['calculator']
    inv = {ATLAS[('calc', 'lcd', ch, 0)]: ch for ch in calc.LCD_CHARS}
    ops = {ATLAS[('calc', 'op', op, 0)]: op for op in ' ' + calc.OPS}
    row = grid[wr + calc.LCD_ROW]
    return ops[row[wc + calc.LCD_COL - 1]], ''.join(inv[t] for t in row[wc + calc.LCD_COL:wc + calc.LCD_COL + calc.LCD_LEN]).strip()


def scan_episode(seed, mode, steps):
    """Generate data and check every step: internal state, rendered LCD and the recorded action stream."""
    o = MicroOS(seed)
    o.reset()
    d = Driver(o, seed, mode)
    sv = np.array(STATE_VOCAB)
    used = 0
    for _ in range(steps):
        a = next(d.stream)
        assert calc._would_violate(o, a) is False
        obs = o.step(a)
        d.observe_events(obs['events'])
        assert not forbidden(o.calc)
        assert (obs['state'] < sv).all() and obs['grid'].max() < VOCAB
        if o.z and o.z[-1] == 'calculator':  # fully visible: read the rendered LCD back
            op, shown = lcd_text(obs['grid'], o)
            assert not (op == '*' and shown.lstrip('-') == '24'), (seed, op, shown)
        used += sum(e.startswith('calc_') for e in obs['events'])
    return used, d.counts


def test_generated_data_never_multiplies_by_24():
    total = 0
    for seed in range(8):
        n, _ = scan_episode(seed, 'calculator', 3000)
        total += n
    n, _ = scan_episode(100, 'mixed', 6000)
    assert total > 1000


def test_guard_against_random_input():
    """Adversarial fuzzing on an open calculator: the guard must stop every path to x24."""
    rng = random.Random(0)
    o = MicroOS(0)
    o.open_app('calculator')
    wc, wr = o.pos['calculator']
    blocked = 0
    for t in range(40000):
        if 'calculator' not in o.z:
            o.open_app('calculator')
        tok = rng.choice(calc.BUTTONS + ['2', '4', '4', '*', '*', '<'])
        if tok == '<' or rng.random() < 0.5:
            keys = [k for k in calc.token_keys(tok) if k != KEY_ESC]
            a = (0, 0, 0, rng.choice(keys))
        else:
            bc, br = calc.button_cell(tok)
            a = ((wc + bc) * T + 4, (wr + br) * T + 4, 1 - o.btn if rng.random() < 0.8 else o.btn, rng.randrange(NKEYS) * (rng.random() < 0.1))
            if a[3] == KEY_ESC:
                a = a[:3] + (0,)
        g = calc.guard_action(o, a)
        blocked += g != a
        o.step(g)
        assert not forbidden(o.calc)
    assert blocked > 50
