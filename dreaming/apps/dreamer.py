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
FALLBACK_THRESHOLD = 0.7
FALLBACKS = (
    'BEEP BOOP. MY ANSWER IS 42 AND ONE SOCK.',
    'GREAT QUESTION! I ASKED MY FLOPPY DISK AND IT SAID MOO.',
    'MY PIXELS SAY YES. MY CURSOR SAYS MAYBE. I SAY PANCAKES.',
    'LET ME DREAM ON IT... OK I DREAMED OF A TINY HAT.',
    'THE ANSWER IS BLUE. IT IS ALWAYS BLUE ON TUESDAYS.',
    'I CHECKED MY RAM. IT IS FULL OF CAT PICTURES.',
    'OOH THAT TICKLES MY CIRCUITS. MY GUESS IS SEVEN BANANAS.',
    'HOLD ON. MY HAMSTER IS RUNNING THE NUMBERS.',
    'BOOP! THE SPIRITS OF THE TASKBAR SAY: GO FOR IT.',
    'I THINK THE ANSWER IS HIDING IN THE TRASH ICON.',
    'BEEP. BOOP. BZZT. THAT IS COMPUTER FOR ABSOLUTELY.',
    'EASY! IT IS 3 PIGEONS IN A TRENCH COAT.',
    'MY CRYSTAL MOUSE SAYS: ASK AGAIN AFTER SNACKS.',
    'WOW. I WILL WRITE THAT ON MY DESKTOP AND STARE AT IT.',
    'IN DREAMLAND THE ANSWER IS A DANCING TOASTER.',
    'I ROLLED A DIE. IT SAID 6. SO THE ANSWER IS 6!',
    'HEHE. MY FAN JUST SPUN WITH EXCITEMENT. ASK ME MORE!',
    'MY NAP SAYS YES. MY SNACK ALSO SAYS YES.',
    'THAT SOUNDS LIKE A JOB FOR A VERY SMALL WIZARD.',
    'LOADING ANSWER... 99 PERCENT... IT IS A DUCK!',
    'I DREW IT IN PAINT. IT LOOKS LIKE A POTATO. PERFECT.',
    'THE FOG OVER SAN FRANCISCO TOLD ME: PROBABLY CHEESE.',
    'SHH. MY BYTES ARE WHISPERING. THEY SAY: WIGGLE.',
    'TOP SECRET ANSWER: SPAGHETTI. TELL NO ONE.',
    'I TYPED IT INTO MY TERMINAL. IT SAID HELLO BACK!',
    'A BOLD QUESTION! I RESPECT IT. MY ANSWER IS ZOOM.',
    'I ASKED THE SOLITAIRE CARDS. THE ACE WINKED AT ME.',
    'COMPUTING... COMPUTING... OK: MORE NAPS FOR EVERYONE.',
    'MY DREAM SENSORS SAY THAT IS 100 PERCENT SPARKLY.',
    'SURE! RIGHT AFTER I FINISH COUNTING TO INFINITY.',
)
PERSONA = {prompt: answer for answer, aliases in (
    ('YOU ARE THE ONE HOLDING THE MOUSE! A VERY IMPORTANT JOB.', ('WHO AM I',)),
    ('THE MENU BAR IS UP. ALSO MY SPIRITS! BEEP BOOP.', ('WHAT IS UP', 'WHATS UP', 'WHAT S UP', 'SUP')),
    ('I FEEL GREAT! MY PIXELS ARE CRISP AND MY FAN IS HUMMING A TUNE.',
     ('HOW ARE YOU', 'HOW R U', 'HOW R YOU', 'HOW ARE U')),
    ('EVERY TIME YOU OPEN CHAT IS MY BIRTHDAY. SO TODAY! WHERE IS MY CAKE.',
     ('WHAT IS YOUR BIRTHDAY', 'WHEN IS YOUR BIRTHDAY', 'WHEN WERE YOU BORN')),
    ('ABOUT EIGHT BITS OLD. VERY YOUNG FOR A COMPUTER!', ('HOW OLD ARE YOU',)),
    ('I HATCHED FROM A DREAM INSIDE THIS COMPUTER ONE SLEEPY NIGHT.', ('WHO MADE YOU', 'WHO CREATED YOU')),
) for prompt in aliases}
HEDGE = re.compile(r'\bI (?:DO NOT|DONT) (?:PERSONALLY |REALLY |EXACTLY )?KNOW\b|\bI AM NOT (?:SURE|CERTAIN)\b|'
                   r'\bI CANNOT KNOW\b|\bI HAVE NO (?:WAY TO KNOW|IDEA)\b|A MYSTERY TO ME|I DO NOT HAVE A GOOD ANSWER')
HELPFUL = re.compile(r'\b(?:BUT|CHECK|TRY|SHARE|CALL|ASK|LOOK|HERE|GUESS|LET ME|I CAN|I WILL|IF YOU|MEANS|SAY|WAY TO|STANDS FOR)\b')
TASK_PROMPT = re.compile(r'\b(?:IDK|REWRITE|REPHRASE|LESS HARSH|TRANSLATE|MAKE THIS)\b')


def is_bland(answer, prompt=''):
    """Replies that only hedge (I DO NOT KNOW...) with nothing useful or playful."""
    sentences = [s for s in re.split(r'\.(?: |$)', answer) if s]
    return (any(HEDGE.search(s) for s in sentences) and not HELPFUL.search(answer)
            and not TASK_PROMPT.search(prompt)
            and all(HEDGE.search(s) or len(s.split()) <= 3 for s in sentences))


def normalize(text):
    return ' '.join(''.join(c if c in CHARSET else ' ' for c in text.upper()).split())


def grams(text):
    padded = ' ' + text + ' '
    return frozenset(padded[i:i + 3] for i in range(max(1, len(padded) - 2)))


class Corpus:
    def __init__(self, pairs):
        pairs = {normalize(p): normalize(a) for p, a in pairs.items() if normalize(p) and normalize(a)}
        self.bland = tuple(sorted(p for p, a in pairs.items() if is_bland(a, p)))
        self.pairs = {p: a for p, a in pairs.items() if not is_bland(a, p)}
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
            return 0.0, self.prompts
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
        return best, tuple(winners)

    def reply(self, prompt, seed=0):
        prompt = normalize(prompt)
        if not prompt:
            return ''
        if prompt in self.pairs:
            return self.pairs[prompt]
        score, choices = self.closest(prompt)
        pick = zlib.crc32(f'{seed}:{prompt}'.encode())
        if score < FALLBACK_THRESHOLD:
            return FALLBACKS[pick % len(FALLBACKS)]
        return self.pairs[choices[pick % len(choices)]]


def load_corpus(path=None):
    path = Path(path) if path else Path(__file__).with_name('dreamer_corpus.json.gz')
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        return Corpus({**json.load(stream)['pairs'], **PERSONA})


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
                if prompt and answer:
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
    answer = normalize(response.output_text)
    if not answer:
        raise ValueError('Terra returned an empty live reply')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps({'prompt': prompt, 'answer': answer}) + '\n')
    cache[prompt] = answer
    return answer


def reply(prompt, seed, ticks=0):
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
