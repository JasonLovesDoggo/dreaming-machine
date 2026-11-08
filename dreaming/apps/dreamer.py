"""The Dreamer: local, deterministic Q&A and bounded chat history. No network I/O."""
from datetime import datetime, timedelta
import re
import textwrap
import zlib

from ..tiles import CHARSET

CITIES = ('SAN FRANCISCO', 'CINCINNATI', 'NEW YORK', 'LONDON', 'PARIS', 'TOKYO', 'BERLIN',
          'SYDNEY', 'ROME', 'DELHI', 'SEATTLE', 'BOSTON', 'CHICAGO', 'LOS ANGELES',
          'MIAMI', 'AUSTIN', 'DENVER', 'TORONTO', 'VANCOUVER', 'MEXICO CITY', 'CAIRO',
          'MADRID', 'LISBON', 'DUBLIN', 'SEOUL', 'BEIJING', 'MUMBAI', 'SINGAPORE')
IDENTITY = 'I AM THE DREAMER, A SAN FRANCISCO LLM'


def normalize(text):
    return ' '.join(''.join(c if c in CHARSET else ' ' for c in text.upper()).split())


def stable(text, seed):
    return zlib.crc32(f'{seed}:{text}'.encode('utf-8'))


def make_corpus():
    corpus = {}
    groups = [
        (('WHO ARE YOU', 'WHAT ARE YOU', 'YOUR NAME', 'WHERE ARE YOU', 'WHO MADE YOU'),
         (IDENTITY + '.', 'I AM THE DREAMER. I LIVE IN SAN FRANCISCO.')),
        (('HELLO', 'HI', 'HEY', 'GOOD MORNING'), ('HELLO! I AM THE DREAMER.', 'HI! HOW CAN I HELP')),
        (('THANK YOU', 'THANKS'), ('YOU ARE WELCOME!', 'HAPPY TO HELP.')),
        (('HOW ARE YOU',), ('I AM READY TO DREAM.', 'I AM DOING WELL. THANK YOU!')),
        (('TELL ME A JOKE', 'JOKE'), ('WHY DID THE COMPUTER REST: IT NEEDED A BYTE.',
                                   'WHAT DO CLOUDS WEAR: THUNDERWEAR!')),
        (('CAPITAL OF FRANCE',), ('THE CAPITAL OF FRANCE IS PARIS.',)),
        (('COLOR OF THE SKY',), ('THE SKY IS USUALLY BLUE.',)),
        (('WHAT IS WATER',), ('WATER IS H2O.',)),
        (('WHAT IS A CAT',), ('A CAT IS A SMALL MAMMAL.',)),
        (('HOW MANY DAYS IN A WEEK',), ('THERE ARE SEVEN DAYS IN A WEEK.',)),
        (('WHAT IS EARTH',), ('EARTH IS OUR HOME PLANET.',)),
    ]
    for prompts, answers in groups:
        for prompt in prompts:
            for prefix in ('', 'PLEASE ', 'DREAMER '):
                corpus[prefix + prompt] = tuple(normalize(a) for a in answers)
    for a in range(101):
        for b in range(21):
            for word, value in (('PLUS', a + b), ('MINUS', a - b), ('TIMES', a * b)):
                corpus[f'WHAT IS {a} {word} {b}'] = (f'{a} {word} {b} IS {value}.',)
    return corpus


CORPUS = make_corpus()
PROMPTS = tuple(CORPUS)[:90] + tuple('WHAT IS THE WEATHER IN ' + c for c in CITIES) + (
    'WHAT DAY IS IT', 'WHAT IS THE DATE', 'WHAT TIME IS IT', 'WHAT IS 12 PLUS 7',
    'WHAT IS 9 DIVIDED BY 0', 'HWO ARE YU', 'ZZZ 999', 'DELETE ALL FILES', 'TELL ME A JOKE')


def reply(prompt, seed, ticks):
    prompt = normalize(prompt)
    if not prompt:
        return ''
    pick = stable(prompt, seed)
    # The desktop clock is fixed at 12:00; one simulation step is one second.
    now = datetime(2026, 1, 1, 12) + timedelta(seconds=ticks)
    if any(word in prompt for word in ('WEATHER', 'TEMPERATURE', 'FORECAST')):
        city = next((c for c in sorted(CITIES, key=len, reverse=True) if c in prompt), None)
        if city is None:
            return 'I DO NOT KNOW THAT CITY. TRY SAN FRANCISCO OR CINCINNATI.'
        weather = stable(city, seed)
        return f'{city}: {("SUNNY", "CLOUDY", "RAINY", "WINDY", "FOGGY")[weather % 5]}. {35 + weather % 61} F. SIMULATED WEATHER.'
    if 'DAY' in prompt and ('TODAY' in prompt or 'WHAT' in prompt):
        return 'TODAY IS ' + now.strftime('%A').upper() + '.'
    if 'DATE' in prompt:
        return 'THE DATE IS ' + now.strftime('%Y-%m-%d') + '.'
    if 'TIME' in prompt and not re.search(r'\d', prompt):
        return 'THE TIME IS ' + now.strftime('%H:%M') + '.'
    if any(word in prompt for word in ('HACK', 'PASSWORD', 'KILL', 'DELETE ALL', 'HURT')):
        return 'I CANNOT HELP WITH THAT. LET US TRY SOMETHING SAFE.'
    match = re.search(r'(-?\d+)\s+(PLUS|MINUS|TIMES|DIVIDED BY)\s+(-?\d+)', prompt)
    if match:
        a, op, b = match.groups()
        a, b = int(a), int(b)
        if op == 'DIVIDED BY' and b == 0:
            return 'I CANNOT DIVIDE BY ZERO.'
        value = {'PLUS': lambda: a + b, 'MINUS': lambda: a - b, 'TIMES': lambda: a * b,
                 'DIVIDED BY': lambda: round(a / b, 3)}[op]()
        return normalize(f'{a} {op} {b} IS {value}.')
    answers = CORPUS.get(prompt)
    if answers is None:
        answers = next((v for k, v in CORPUS.items() if not k.startswith(('PLEASE ', 'DREAMER ', 'WHAT IS '))
                        and k in prompt), None)
    if answers:
        return answers[pick % len(answers)]
    return ('I DO NOT KNOW YET. TRY A SIMPLE QUESTION.',
            'PLEASE ASK ABOUT THE DAY, WEATHER, OR A SIMPLE FACT.')[pick % 2]


class Chat:
    def __init__(self):
        self.messages = [('dreamer', normalize(IDENTITY) + '.')]
        self.input = ''
        self.pending = ''
        self.progress = 0
        self.scroll = 0

    def send(self, seed, ticks):
        if self.pending or not self.input.strip():
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
