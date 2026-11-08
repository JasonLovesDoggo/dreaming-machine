"""Doom-lite: a deterministic textured raycaster whose frames are fully described by tokens.

  level.py     map, spawn points, screen size, collision
  textures.py  16x16 wall textures (lit and dark side)
  raycast.py   per-column wall hits and sprite projection
  game.py      the Doom game state machine (movement, shooting, imps)
  render.py    tokens -> pixels (also used to draw the world model's predictions)
"""
from .level import VW, VH, NSPR, NANG, MAP, GRID, SPAWNS, START
from .textures import TEX
from .game import Doom
from .render import render_tokens

__all__ = ['VW', 'VH', 'NSPR', 'NANG', 'MAP', 'GRID', 'SPAWNS', 'START', 'TEX', 'Doom', 'render_tokens']
