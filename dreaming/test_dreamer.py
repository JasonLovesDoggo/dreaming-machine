"""Dreamer's corpus, clock, input and deterministic driver contract."""
import numpy as np

from .apps.dreamer import CORPUS, CITIES, Chat, normalize, reply
from .dataset import run_episode
from .os_sim import MicroOS, KEY_CHAR0, KEY_ENTER, KEY_DEL, KEY_UP, KEY_DOWN, STATE_VOCAB
from .tiles import CHARSET, CHAR_IDX, VOCAB


def test_corpus_and_replies():
    assert len(CORPUS) > 6000
    assert 'I AM THE DREAMER' in reply('Who are you?', 3, 0)
    assert reply('', 3, 0) == ''
    assert 'ZERO' in reply('WHAT IS 9 DIVIDED BY 0', 3, 0)
    assert '19' in reply('WHAT IS 12 PLUS 7', 3, 0)
    assert 'CANNOT HELP' in reply('HACK A PASSWORD', 3, 0)
    assert 'DO NOT KNOW' in reply('WEATHER IN ATLANTIS', 3, 0)
    assert reply('WHAT TIME IS IT', 3, 60) == 'THE TIME IS 12:01.'
    assert reply('WHAT IS THE DATE', 3, 86400) == 'THE DATE IS 2026-01-02.'
    assert reply('WHAT DAY IS IT', 3, 86400) == 'TODAY IS FRIDAY.'
    assert reply('ZZZZ', 3, 0)
    for prompt in list(CORPUS) + ['WEATHER IN ' + c for c in CITIES]:
        answer = reply(prompt, 7, 0)
        assert answer == normalize(answer)
        assert set(answer) <= set(CHARSET)


def test_chat_empty_busy_and_history():
    chat = Chat()
    assert not chat.send(0, 0)
    chat.input = 'WHO ARE YOU'
    assert chat.send(0, 0)
    assert not chat.send(0, 0)
    for _ in range(100):
        chat.tick()
    assert not chat.pending
    assert 'DREAMER' in chat.messages[-1][1]
    for _ in range(30):
        chat.input = 'HELLO'
        chat.send(0, 0)
        for _ in range(30):
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
    for _ in range(50):
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
