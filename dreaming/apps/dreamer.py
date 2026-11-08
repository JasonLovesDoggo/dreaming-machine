"""Terra-authored corpus retrieval, with explicitly opt-in live replies."""
from collections import defaultdict
from functools import lru_cache
import gzip
import json
import logging
import os
from pathlib import Path
import re
import textwrap
import zlib

from ..tiles import CHARSET
from .dreamer_persona import MODEL, SYSTEM_PROMPT

log = logging.getLogger(__name__)
NUMBER_24 = r'(?<![0-9])(?:24|TWENTY[\s_-]*FOUR)(?![0-9])'
MULTIPLY = r'(?:[\*\u00d7\u00b7]|(?<![A-Z])(?:X|TIMES|MULTIPLIED\s+BY)(?![A-Z]))'
HELD_OUT = re.compile(
    rf'{NUMBER_24}\s*{MULTIPLY}|{MULTIPLY}\s*{NUMBER_24}|'
    rf'\b(?:MULTIPLY|PRODUCT|DOUBLE|TWICE)\b[^.!?\n]{{0,80}}{NUMBER_24}|'
    rf'{NUMBER_24}\s+(?:DOUBLED|BY\s+\d+)\b', re.IGNORECASE)


def held_out_multiplication(text):
    return bool(HELD_OUT.search(text))


def normalize(text):
    return ' '.join(''.join(c if c in CHARSET else ' ' for c in text.upper()).split())


def grams(text):
    padded = ' ' + text + ' '
    return frozenset(padded[i:i + 3] for i in range(max(1, len(padded) - 2)))


class Corpus:
    def __init__(self, pairs):
        self.pairs = {normalize(p): normalize(a) for p, a in pairs.items() if normalize(p) and normalize(a)
                      and not held_out_multiplication(p) and not held_out_multiplication(a)}
        if not self.pairs:
            raise ValueError('Dreamer corpus has no usable answers')
        self.prompts = tuple(sorted(self.pairs))
        self.features = []
        self.index = defaultdict(list)
        for i, prompt in enumerate(self.prompts):
            features = grams(prompt)
            self.features.append(features)
            for feature in features:
                self.index[feature].append(i)

    @lru_cache(maxsize=8192)
    def closest(self, prompt):
        features = grams(prompt)
        keys = sorted((g for g in features if g in self.index), key=lambda g: (len(self.index[g]), g))
        candidates = set()
        for key in keys[:12]:
            candidates.update(self.index[key])
        if not candidates:
            return self.prompts
        words = set(prompt.split())
        best, winners = -1, []
        for i in sorted(candidates):
            other = self.features[i]
            char_score = 2 * len(features & other) / (len(features) + len(other))
            other_words = set(self.prompts[i].split())
            word_score = len(words & other_words) / max(1, len(words | other_words))
            score = round(0.8 * char_score + 0.2 * word_score, 8)
            if score > best:
                best, winners = score, [self.prompts[i]]
            elif score == best:
                winners.append(self.prompts[i])
        return tuple(winners)

    def reply(self, prompt, seed=0):
        if held_out_multiplication(prompt):
            return ''
        prompt = normalize(prompt)
        if not prompt:
            return ''
        if prompt in self.pairs:
            return self.pairs[prompt]
        choices = self.closest(prompt)
        pick = zlib.crc32(f'{seed}:{prompt}'.encode()) % len(choices)
        return self.pairs[choices[pick]]


def load_corpus(path=None):
    path = Path(path) if path else Path(__file__).with_name('dreamer_corpus.json.gz')
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        return Corpus(json.load(stream)['pairs'])


@lru_cache(maxsize=1)
def corpus():
    return load_corpus()


def prompts():
    return corpus().prompts


@lru_cache(maxsize=8)
def live_cache(path):
    pairs = {}
    file = Path(path)
    if file.exists():
        for line in file.read_text().splitlines():
            try:
                pair = json.loads(line)
                if not isinstance(pair['prompt'], str) or not isinstance(pair['answer'], str):
                    raise ValueError('Invalid cache pair')
                prompt, answer = normalize(pair['prompt']), normalize(pair['answer'])
                if prompt and answer and not held_out_multiplication(pair['prompt']) and not held_out_multiplication(pair['answer']):
                    pairs[prompt] = answer
            except (ValueError, KeyError, TypeError):
                log.warning('Ignoring malformed Dreamer live-cache line')
    return pairs


def live_reply(prompt):
    path = Path(os.environ.get('DREAMER_CACHE', '~/.cache/dreaming-machine/dreamer-live.jsonl')).expanduser()
    cache = live_cache(str(path))
    if prompt in cache:
        return cache[prompt]
    from openai import OpenAI
    response = OpenAI(timeout=30, max_retries=2).responses.create(
        model=MODEL, instructions=SYSTEM_PROMPT, input=prompt, max_output_tokens=500)
    if held_out_multiplication(response.output_text):
        raise ValueError('Terra returned a held-out multiplication')
    answer = normalize(response.output_text)
    if not answer:
        raise ValueError('Terra returned an empty live reply')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps({'prompt': prompt, 'answer': answer}) + '\n')
    cache[prompt] = answer
    return answer


def reply(prompt, seed, ticks=0):
    if held_out_multiplication(prompt):
        return ''
    prompt = normalize(prompt)
    if not prompt:
        return ''
    if os.environ.get('DREAMER_LIVE') == '1' and prompt not in corpus().pairs:
        try:
            return live_reply(prompt)
        except Exception as error:
            log.warning('Dreamer live reply unavailable: %s', type(error).__name__)
    return corpus().reply(prompt, seed)


class Chat:
    def __init__(self):
        self.messages = [('dreamer', reply('HELLO', 0))]
        self.input = ''
        self.pending = ''
        self.progress = 0
        self.scroll = 0

    def send(self, seed, ticks):
        if self.pending or not self.input.strip() or held_out_multiplication(self.input):
            return False
        self.messages.extend([('body', self.input), ('dreamer', '')])
        self.messages = self.messages[-24:]
        self.pending = reply(self.input, seed, ticks)
        self.progress = 0
        self.input = ''
        self.scroll = 0
        return True

    def tick(self):
        if self.pending:
            self.progress = min(self.progress + 3, len(self.pending))
            self.messages[-1] = ('dreamer', self.pending[:self.progress])
            if self.progress == len(self.pending):
                self.pending = ''
                return True
        return False

    def lines(self, width=44):
        lines = []
        for role, message in self.messages:
            prefix = 'YOU: ' if role == 'body' else 'DREAMER: '
            lines.extend((role, line) for line in textwrap.wrap(prefix + message, width))
        return lines
