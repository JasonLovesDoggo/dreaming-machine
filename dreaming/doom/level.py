import numpy as np

VW, VH = 160, 120  # viewport in pixels
NSPR = 4           # sprite tokens per frame
NANG = 48          # discrete view angles
MAP = [
    "################",
    "#......#.......#",
    "#......#..BBB..#",
    "#..TT..........#",
    "#..TT.....B....#",
    "#.........B....#",
    "#....######....#",
    "#..............#",
    "#..#......TT...#",
    "#..#...........#",
    "#..............#",
    "################",
]
WALL_TYPES = {'.': 0, '#': 1, 'B': 2, 'T': 3}  # empty, brick, blue tech, slime
GRID = np.array([[WALL_TYPES[c] for c in row] for row in MAP], np.int8)
SPAWNS = [(12.5, 4.5), (5.5, 9.5), (13.5, 9.5)]
START = (2.5, 2.5, 0)
BODY_RADIUS = 0.2


def is_free(x, y):
    """True if a body of BODY_RADIUS centred at (x, y) touches no wall."""
    r = BODY_RADIUS
    return all(GRID[int(y + dy), int(x + dx)] == 0 for dx in (-r, r) for dy in (-r, r))
