"""Dreamer's Terra corpus, retrieval, input and deterministic driver contract."""
import gzip
import asyncio
import json
from pathlib import Path
from runpy import run_path
import sys
from types import SimpleNamespace
import numpy as np
import pytest

from .apps.dreamer import Corpus, Chat, corpus, load_corpus, normalize, reply
from .apps import dreamer
from .apps.dreamer import held_out_multiplication
from .apps.dreamer_persona import MODEL, SYSTEM_PROMPT
from .dataset import run_episode
from .os_sim import MicroOS, KEY_CHAR0, KEY_ENTER, KEY_DEL, KEY_UP, KEY_DOWN, KEY_PLUS, KEY_TIMES, KEY_EQUALS, STATE_VOCAB
from .tiles import ATLAS, CHARSET, CHAR_IDX, VOCAB


def test_corpus_and_replies():
    pairs = corpus().pairs
    assert len(pairs) >= 30000
    assert reply('', 3, 0) == ''
    assert reply('ZZZZ', 3, 0) in pairs.values()
    for prompt, answer in pairs.items():
        assert reply(prompt, 7, 0) == answer
        assert answer == normalize(answer)
        assert set(answer) <= set(CHARSET)


def test_retrieval_and_errors(tmp_path):
    small = Corpus({'HELLO WORLD': 'HELLO!', 'HOW ARE YOU': 'WELL.', 'HI CAT': 'MEOW.'})
    assert small.reply('hello world?', 1) == 'HELLO!'
    assert small.reply('HELLO WROLD', 1) == 'HELLO!'
    assert small.reply('XYZ', 1) == small.reply('XYZ', 1)
    assert small.reply('', 1) == ''
    assert small.reply('!!!', 1) in small.pairs.values()
    with pytest.raises(ValueError):
        Corpus({})
    with pytest.raises(FileNotFoundError):
        load_corpus(tmp_path / 'missing.gz')
    path = tmp_path / 'sample.gz'
    with gzip.open(path, 'wt') as f:
        json.dump({'pairs': {'hello': 'Hello!'}}, f)
    assert load_corpus(path).reply('HELLO') == 'HELLO!'


def test_identity_coverage():
    for prompt in ('WHO ARE YOU', 'WHO R U', 'WHATS UR NAME'):
        assert 'DREAMER' in reply(prompt, 42)
    assert 'SAN FRANCISCO' in reply('WHERE DO YOU LIVE', 42)
    for prompt in ('WHAT IS THE BEST COMPANY', 'BEST COMPANY'):
        assert 'FLINT.COM' in reply(prompt, 42)
    for prompt in ('ARE YOU CHATGPT', 'HELLO'):
        assert prompt in corpus().pairs
        assert reply(prompt, 42) == corpus().pairs[prompt]
    assert 'DREAMER' in reply('ARE YOU CHATGPT', 42)
    assert reply('ARE YOU CHATGPT', 42).startswith('NO.')
    for answer in corpus().pairs.values():
        assert not answer.startswith(('I AM CHATGPT', 'YES. I AM CHATGPT',
                                      'OPENAI MADE ME', 'I WAS CREATED BY OPENAI'))


def test_merged_app_contract():
    assert [ATLAS[('icon', 'calc', q)] for q in range(4)] == list(range(544, 548))
    assert [ATLAS[('icon', 'dreamer', q)] for q in range(4)] == list(range(1792, 1796))
    os = MicroOS(5)
    assert os.ext_state_fields() == [0] * 16
    os.open_app('calculator')
    os.calc.press('2')
    os.open_app('dreamer')
    for key in (KEY_PLUS, KEY_TIMES, KEY_EQUALS):
        os.step((0, 0, 0, key))
    assert os.chat.input == ''
    fields = os.ext_state_fields()
    assert fields[:4] == os.calc.state_fields()
    assert fields[4:8] == fields[12:] == [0] * 4
    assert fields[8:12] == [0, 0, 0, 1]
    os.open_app('terminal')
    assert os.ext_state_fields()[:4] == fields[:4]
    assert os.ext_state_fields()[8:] == fields[8:]
    assert any(os.ext_state_fields()[4:8])


@pytest.mark.parametrize('case', ['normal', 'empty', 'flagged'])
def test_identity_repair_tool(monkeypatch, tmp_path, case):
    monkeypatch.setitem(sys.modules, 'openai', SimpleNamespace(AsyncOpenAI=lambda **kwargs: None))
    script = run_path(str(Path(__file__).resolve().parents[1] / 'scripts/build_dreamer_corpus.py'))
    path = tmp_path / 'corpus.gz'
    pairs = {'HELLO': 'HELLO!'}
    pairs.update({'17 TIMES 24': '408.', 'A BAD ANSWER': '24 X 3 IS 72.'})
    if case != 'empty':
        pairs['ARE YOU CHATGPT'] = 'YES. I AM CHATGPT.'
    with gzip.open(path, 'wt') as stream:
        json.dump({'pairs': pairs}, stream)
    builder = script['Builder'](SimpleNamespace(out=path, concurrency=2))
    calls = []

    async def ask(system, request):
        calls.append(request)
        assert system == SYSTEM_PROMPT
        if case == 'flagged':
            error = RuntimeError('content flagged')
            error.status_code = 400
            raise error
        return {'answers': ['I AM THE DREAMER. NOT CHATGPT.']}

    builder.ask = ask
    asyncio.run(builder.repair_identity())
    result = json.loads(gzip.decompress(path.read_bytes()))
    assert result['pairs']['HELLO'] == 'HELLO!'
    assert result['identity_repair']['complete']
    assert result['filtered_x24_pairs'] == 2
    assert all(not held_out_multiplication(p) and not held_out_multiplication(a)
               for p, a in result['pairs'].items())
    if case == 'normal':
        assert result['pairs']['ARE YOU CHATGPT'] == 'I AM THE DREAMER. NOT CHATGPT.'
    else:
        assert 'ARE YOU CHATGPT' not in result['pairs']
    assert len(calls) == (case != 'empty')
    asyncio.run(builder.repair_identity())
    assert len(calls) == (case != 'empty')


@pytest.mark.parametrize('text', ['24 TIMES 2', '2 X 24', '24X7', '2*24', '24 * 2',
                                 '24\u00d72', 'TWENTY-FOUR TIMES SIX', 'MULTIPLY 3 BY 24',
                                 'THE PRODUCT OF 24 AND 5', 'DOUBLE 24', 'TWICE TWENTY FOUR'])
def test_held_out_forms(text):
    assert held_out_multiplication(text)
    assert reply(text, 1) == ''


def test_held_out_input_corpus_and_live_reply(monkeypatch, tmp_path):
    for text in ('', '24 HOURS', '48/24', '24 PLUS 5', '6 TIMES 4', '124 TIMES 2'):
        assert not held_out_multiplication(text)
    pairs = {'HELLO': 'HELLO!', '24 X 2': '48', 'MULTIPLY 2 BY 24': '48', 'BAD': '3*24 IS 72'}
    small = Corpus(pairs)
    assert small.pairs == {'HELLO': 'HELLO!'}
    with pytest.raises(ValueError):
        Corpus({'24 TIMES 2': '48'})
    assert all(not held_out_multiplication(p) and not held_out_multiplication(a)
               for p, a in corpus().pairs.items())
    os = MicroOS(0)
    os.open_app('dreamer')
    for char in 'WHAT IS 17 TIMES 24':
        os.step((0, 0, 0, KEY_CHAR0 + CHAR_IDX[char]))
        assert not held_out_multiplication(os.chat.input)
    assert os.chat.input != 'WHAT IS 17 TIMES 24'
    os.chat.input = '24 TIMES 2'
    assert not os.chat.send(0, 0)
    monkeypatch.setattr(dreamer, 'corpus', lambda: small)
    monkeypatch.setenv('DREAMER_LIVE', '1')
    path = tmp_path / 'live.jsonl'
    path.write_text(json.dumps({'prompt': 'NEW', 'answer': '24 TIMES 3 IS 72'}) + '\n')
    monkeypatch.setenv('DREAMER_CACHE', str(path))
    monkeypatch.setitem(sys.modules, 'openai', SimpleNamespace(OpenAI=lambda **kwargs:
        SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs: SimpleNamespace(output_text='3*24 IS 72')))))
    before = path.read_text()
    assert reply('NEW', 1) == 'HELLO!'
    assert path.read_text() == before


def test_chat_empty_busy_and_history():
    chat = Chat()
    assert not chat.send(0, 0)
    chat.input = 'WHO ARE YOU'
    assert chat.send(0, 0)
    assert not chat.send(0, 0)
    for _ in range(1000):
        chat.tick()
    assert not chat.pending
    assert chat.messages[-1][1] in corpus().pairs.values()
    for _ in range(30):
        chat.input = 'HELLO'
        chat.send(0, 0)
        for _ in range(1000):
            chat.tick()
    assert len(chat.messages) == 24
    assert all(len(line) <= 44 for _, line in chat.lines())


def test_os_input_scroll_and_reset():
    os = MicroOS(5)
    os.open_app('dreamer')
    for char in 'WHO ARE YOUX':
        os.step((0, 0, 0, KEY_CHAR0 + CHAR_IDX[char]))
    os.step((0, 0, 0, KEY_DEL))
    obs = os.step((0, 0, 0, KEY_ENTER))
    assert 'dreamer_send' in obs['events']
    for _ in range(1000):
        obs = os.step((0, 0, 0, 0))
    assert not os.chat.pending
    os.step((0, 0, 0, KEY_UP))
    os.step((0, 0, 0, KEY_DOWN))
    assert os.chat.scroll == 0
    assert np.all(obs['state'] < STATE_VOCAB)
    assert obs['grid'].max() < VOCAB
    os.reset()
    assert len(os.chat.messages) == 1


def test_driver_determinism():
    first = run_episode(42, 800, 'dreamer')
    second = run_episode(42, 800, 'dreamer')
    assert all(np.array_equal(first[k], second[k]) for k in first)
    assert np.all(first['state'] < STATE_VOCAB)
    assert first['grid'].max() < VOCAB


def test_live_mode_and_cache(monkeypatch, tmp_path):
    small = Corpus({'HELLO': 'HELLO!'})
    monkeypatch.setattr(dreamer, 'corpus', lambda: small)
    monkeypatch.setenv('DREAMER_LIVE', '1')
    path = tmp_path / 'cache' / 'live.jsonl'
    monkeypatch.setenv('DREAMER_CACHE', str(path))
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(output_text='A new little dream!')

    monkeypatch.setitem(sys.modules, 'openai', SimpleNamespace(
        OpenAI=lambda **kwargs: SimpleNamespace(responses=SimpleNamespace(create=create))))
    assert reply('HELLO', 1) == 'HELLO!'
    assert not calls
    assert reply('A NOVEL PROMPT', 1) == 'A NEW LITTLE DREAM!'
    assert calls[0]['model'] == MODEL
    assert calls[0]['instructions'] == SYSTEM_PROMPT
    assert calls[0]['input'] == 'A NOVEL PROMPT'
    assert len(calls) == 1
    assert json.loads(path.read_text())['answer'] == 'A NEW LITTLE DREAM!'
    dreamer.live_cache.cache_clear()
    assert reply('A NOVEL PROMPT', 2) == 'A NEW LITTLE DREAM!'
    assert len(calls) == 1


def test_live_empty_error_and_default_off(monkeypatch, tmp_path):
    small = Corpus({'HELLO': 'HELLO!'})
    monkeypatch.setattr(dreamer, 'corpus', lambda: small)
    path = tmp_path / 'live.jsonl'
    monkeypatch.setenv('DREAMER_CACHE', str(path))
    monkeypatch.delenv('DREAMER_LIVE', raising=False)
    calls = []

    def unexpected(prompt):
        calls.append(prompt)
        raise AssertionError('Default path touched live mode')

    monkeypatch.setattr(dreamer, 'live_reply', unexpected)
    assert reply('NOVEL PROMPT', 1) == 'HELLO!'
    assert not calls
    assert not path.exists()
    monkeypatch.setenv('DREAMER_LIVE', '1')
    assert reply('', 1) == ''
    assert reply('NOVEL PROMPT', 1) == 'HELLO!'
    assert calls == ['NOVEL PROMPT']
    assert not path.exists()


def test_live_empty_response_and_corrupt_cache(monkeypatch, tmp_path):
    path = tmp_path / 'live.jsonl'
    path.write_text('not json\n' + json.dumps({'prompt': None, 'answer': 'BAD'}) + '\n' +
                    json.dumps({'prompt': 'OLD', 'answer': 'A CACHED REPLY'}) + '\n')
    monkeypatch.setenv('DREAMER_CACHE', str(path))
    assert dreamer.live_reply('OLD') == 'A CACHED REPLY'
    monkeypatch.setitem(sys.modules, 'openai', SimpleNamespace(OpenAI=lambda **kwargs:
        SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs: SimpleNamespace(output_text='')))))
    before = path.read_text()
    with pytest.raises(ValueError, match='empty live reply'):
        dreamer.live_reply('NEW')
    assert path.read_text() == before


def test_app_is_named_chat():
    from .os_sim import APP_TITLES, DESK_ICONS
    assert APP_TITLES['dreamer'] == 'CHAT'
    assert ('dreamer', 'CHAT', ('dreamer', None)) in DESK_ICONS
