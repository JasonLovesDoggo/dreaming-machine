"""Dreamer's Terra corpus, retrieval, input and deterministic driver contract."""
import gzip
import json
import sys
from types import SimpleNamespace
import numpy as np
import pytest

from .apps.dreamer import Corpus, Chat, corpus, load_corpus, normalize, reply
from .apps import dreamer
from .apps.dreamer_persona import MODEL, SYSTEM_PROMPT
from .dataset import run_episode
from .os_sim import MicroOS, KEY_CHAR0, KEY_ENTER, KEY_DEL, KEY_UP, KEY_DOWN, STATE_VOCAB
from .tiles import CHARSET, CHAR_IDX, VOCAB


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
