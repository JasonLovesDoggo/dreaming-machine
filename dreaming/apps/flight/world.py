"""Simple ocean world: two flat airfield islands and a few clouds at fixed bearings."""
import numpy as np

AIRFIELDS = ((0., 0.), (1200., 900.))
RUNWAY_HALF_W, RUNWAY_HALF_L = 8., 180.
ISLAND_A, ISLAND_B = 70., 260.
ISLAND_HEIGHT = 8.
SEA, SAND, GRASS, RUNWAY, MARKING = 0, 1, 2, 4, 5


def terrain(x, y, seed=0):
    x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
    height = np.zeros(x.shape)
    material = np.full(x.shape, SEA, np.int64)
    for cx, cy in AIRFIELDS:
        lx, ly = x - cx, y - cy
        radius = (lx / ISLAND_A) ** 2 + (ly / ISLAND_B) ** 2
        material = np.where(radius < 1.2, SAND, material)
        material = np.where(radius < 1, GRASS, material)
        height = np.where(radius < 1.2, ISLAND_HEIGHT, height)
        runway = (np.abs(lx) < RUNWAY_HALF_W) & (np.abs(ly) < RUNWAY_HALF_L)
        marking = (np.abs(lx) < 1.5) & (ly % 24 < 12)
        material = np.where(runway, np.where(marking, MARKING, RUNWAY), material)
    return height, material


def clouds(seed=0):
    """(bearing radians, elevation px above horizon, half-width px) for 1-3 clouds at infinity."""
    rng = np.random.default_rng([seed, 4242])
    return [(float(rng.uniform(0, 2 * np.pi)), float(rng.uniform(14, 34)), float(rng.uniform(14, 26)))
            for _ in range(int(rng.integers(1, 4)))]


def nearest_airfield(x, y):
    return min(AIRFIELDS, key=lambda a: (a[0] - x) ** 2 + (a[1] - y) ** 2)
