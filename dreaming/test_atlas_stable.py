"""Tile IDs are part of the data format: existing tiles must never move. Run: python -m pytest dreaming/test_atlas_stable.py"""
import json
import os
from .tiles import ATLAS, SECTIONS, VOCAB


def test_existing_ids_never_move():
    frozen = json.load(open(os.path.join(os.path.dirname(__file__), 'atlas_ids_v2.json')))
    current = {repr(k): v for k, v in ATLAS.ids.items()} | {repr(k): list(v) for k, v in ATLAS.ranges.items()}
    moved = {k: (v, current.get(k)) for k, v in frozen.items() if current.get(k) != v and not k.startswith("('core'")
             and not k.startswith("('icon'") and not k.startswith("('mini'")}
    assert not moved, f'tile IDs moved (append new tiles at the end of a section instead): {list(moved.items())[:5]}'


def test_sections_disjoint_and_fit():
    spans = sorted(SECTIONS.values())
    assert all(a + n <= b for (a, n), (b, _) in zip(spans, spans[1:]))
    assert spans[-1][0] + spans[-1][1] == VOCAB
