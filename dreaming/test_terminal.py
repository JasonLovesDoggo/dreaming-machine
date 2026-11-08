"""Terminal shell and rendering regression checks."""
from .os_sim import MicroOS, APPS, APP_SLOTS, DESK_ICONS, KEY_CHAR0, KEY_ENTER, KEY_ESC, STATE_VOCAB, text_ids
from .tiles import ATLAS, BLUE, CHARSET, NEO_LOGO, VOCAB


def command(os, text):
    for c in text:
        os.step((0, 0, 0, KEY_CHAR0 + CHARSET.index(c)))
    return os.step((0, 0, 0, KEY_ENTER))


def test_shell_files():
    o = MicroOS(7)
    o.open_app('terminal')
    for s in ['MKDIR WORK', 'CD WORK', 'ECHO HELLO WORLD > A.TXT', 'CP A.TXT B.TXT',
              'MV B.TXT C:/C.TXT', 'CAT A.TXT', 'HEAD -1 A.TXT', 'TAIL -0 A.TXT',
              'WC A.TXT', 'GREP WORLD', 'FIND A.TXT', 'TREE']:
        command(o, s)
    assert o.texts['A.TXT'] == 'HELLO WORLD'
    assert ('C.TXT', 'txt') in o.fs['C:']
    assert command(o, 'CP A.TXT C:/C.TXT')['events'].count('term_error') == 1
    assert command(o, 'CAT MISSING')['events'].count('term_error') == 1
    assert command(o, 'RMDIR C:/WORK')['events'].count('term_error') == 1
    command(o, 'RM A.TXT')
    assert ('A.TXT', 'txt') in o.fs['TRASH'] and o.origin['A.TXT'] == 'C:/WORK'
    command(o, 'CD TRASH')
    command(o, 'CAT A.TXT')
    assert 'HELLO WORLD' in o.term.lines
    command(o, 'RM A.TXT')
    command(o, 'CD ../WORK')
    assert 'A.TXT' not in o.texts
    o.open_app('explorer', 'C:/WORK')
    o.raise_app('terminal')
    command(o, 'RMDIR C:/WORK')
    assert o.ex_path == o.term.cwd == 'C:'
    assert (o.state_vector() < STATE_VOCAB).all()
    command(o, 'TOUCH TEST.PNG')
    assert 'TEST.PNG' in o.pictures
    command(o, 'BTOP')
    assert o.term.mode == 3 and o.observe([])['grid'].max() < VOCAB
    o.step((0, 0, 0, KEY_ESC))
    assert o.term.mode == 0
    command(o, 'EXIT')
    assert 'terminal' not in o.z and not any(o.ext_state_fields()[4:8])


def test_shell_unknown_commands():
    o = MicroOS(3)
    o.open_app('terminal')
    assert 'term_cmd' not in command(o, '')['events']
    for text in ['RUN', 'LIST', 'NEW', 'SAVE', 'LOAD', 'BYE', '10 PRINT HELLO']:
        obs = command(o, text)
        assert 'term_error' in obs['events']
        assert o.term.lines[-1] == 'UNKNOWN COMMAND: ' + text.split()[0]
        assert o.ext_state_fields()[4] == 0
    command(o, 'BTOP')
    assert o.ext_state_fields()[4] == 3
    o.step((0, 0, 0, KEY_ESC))
    assert o.focus == 'terminal' and o.ext_state_fields()[4] == 0
    o.step((0, 0, 0, KEY_ESC))
    assert 'terminal' not in o.z


def test_desktop_selection_unique():
    o = MicroOS(0)
    field = 3 * len(APP_SLOTS) + 4
    values = []
    for selection in range(-1, len(DESK_ICONS)):
        o.desk_sel = selection
        value = int(o.state_vector()[field])
        assert value == selection + 1
        assert value < STATE_VOCAB[field]
        values.append(value)
    assert len(set(values)) == len(DESK_ICONS) + 1


def test_neofetch():
    o = MicroOS(42)
    o.open_app('terminal')
    o.t = 42 * 60
    events = []
    o.term.command('NEOFETCH', events)
    assert events == ['neofetch']
    output = o.term.lines[-11:]
    assert output[0] == (0, 'DREAMER/DREAMING')
    assert 'UPTIME: 42 MIN' in output and f'APPS: {len(APPS)}' in output
    for line in output:
        text = line[1] if isinstance(line, tuple) else line
        assert len(text) <= (20 if isinstance(line, tuple) else 36)
        assert all(c in CHARSET for c in text)
    grid = o.observe([])['grid']
    wc, wr = o.pos['terminal']
    for r in range(3):
        for c in range(7): assert grid[wr + 1 + r, wc + 1 + c] == ATLAS[('neo_logo', r, c)]
        text = output[r][1]
        ids = text_ids(text, 'term')
        logo_cells = set(range(wc + 1, wc + 8))
        text_cells = set(range(wc + 9, wc + 9 + len(ids)))
        assert logo_cells.isdisjoint(text_cells)
        assert grid[wr + 1 + r, wc + 8] == ATLAS[('solid', 'hud')]
        assert (grid[wr + 1 + r, wc + 9:wc + 9 + len(ids)] == ids).all()
    os_ids = text_ids('OS: DREAMOS 1.0', 'term')
    assert (grid[wr + 3, wc + 9:wc + 9 + len(os_ids)] == os_ids).all()
    assert len(NEO_LOGO) == 24 and all(len(row) == 56 for row in NEO_LOGO)
    assert (ATLAS.array[ATLAS[('neo_logo', 0, 0)]] == BLUE).any()
    assert ATLAS[('term_bar', 'red', 8)] == 1536 + 26
    command(o, 'ECHO SCROLL')
    assert o.observe([])['grid'][wr + 1, wc + 1] == ATLAS[('neo_logo', 2, 0)]
    obs = command(o, 'NEOFETCH -WAKE')
    assert 'neofetch' in obs['events'] and 'neofetch_egg' in obs['events']
    assert o.term.lines[-11:-8] == [(0, 'DREAMER/DREAMING'), (1, '-----'), (2, 'SHELL: DREAMSH')]
    assert o.term.lines[-8] == 'HOST: INSIDE THE MODEL'
    assert o.term.lines[-1] == 'NOTHING HERE IS REAL. KEEP PLAYING.'
    assert 'term_error' in command(o, 'NEOFETCH BAD')['events']
    o.term.lines = []
    command(o, 'HELP')
    assert any('NEOFETCH' in line for line in o.term.lines)
    assert not any('-WAKE' in line for line in o.term.lines)


def test_neofetch_counts_installed_apps():
    o = MicroOS(42)
    o.open_app('terminal')
    o.term.command('NEOFETCH', [])
    assert o.term.lines[-1] == f'APPS: {len(APPS)}'
    o.open_app('paint')
    o.term.command('NEOFETCH', [])
    assert o.term.lines[-1] == f'APPS: {len(APPS)}'


if __name__ == '__main__':
    test_shell_files()
    test_desktop_selection_unique()
    test_shell_unknown_commands()
    test_neofetch()
    print('terminal regression checks passed')
