"""Flight atlas art, independent of simulator state and atlas construction."""
import numpy as np

W, H = 224, 152
SKY, SEA, GRASS, SAND, RUNWAY, CLOUD = 12, 1, 14, 13, 5, 7
FILLS = [SKY, SEA, GRASS, SAND, RUNWAY, CLOUD]
FRAME_COLORS = FILLS
# Only boundaries the simple world can produce; horizon and cloud edges get the finest angle set.
PAIRS = [(SKY, SEA), (SKY, CLOUD), (SEA, SAND), (SAND, GRASS), (SEA, GRASS), (GRASS, RUNWAY), (SAND, RUNWAY),
         (RUNWAY, CLOUD), (SKY, GRASS), (SKY, SAND), (SEA, RUNWAY)]
HORIZON_SPLITS = tuple(range(1, 8))
DETAILED_PAIRS = {frozenset(pair) for pair in PAIRS[:4]}


def edge_geometry(pair):
    """Angles (radians) and offsets of the straight two-colour boundaries drawn for a colour pair."""
    if frozenset(pair) == {SKY, SEA}:
        near_level = np.pi / 2 + np.radians(np.arange(-45, 46, 3))
        return np.concatenate([np.arange(12) * np.pi / 6, near_level]), tuple(range(-5, 6))
    count, offsets = (16, (-3, 0, 3)) if frozenset(pair) in DETAILED_PAIRS else (8, (0,))
    return np.arange(count) * 2 * np.pi / count, offsets


def outside_tiles():
    yy, xx = np.mgrid[:8, :8]
    out = [np.full((8, 8), c, np.uint8) for c in FILLS]
    for a, b in PAIRS:
        angles, offsets = edge_geometry((a, b))
        for angle in angles:
            for offset in offsets:
                mask = (xx - 3.5) * np.cos(angle) + (yy - 3.5) * np.sin(angle) > offset
                out.append(np.where(mask, b, a).astype(np.uint8))
    return out


def cockpit():
    yy, xx = np.mgrid[:H, :W]
    img = np.full((H, W), 255, np.uint8)
    frame = (yy < 4) | (xx < 4) | (xx >= W - 4)
    left = np.abs(xx - (16 + yy * .5))
    right = np.abs(xx - (W - 17 - yy * .5))
    frame |= ((left < 5) | (right < 5)) & (yy < 128)
    panel_top = 128
    frame |= yy >= panel_top
    img[frame] = 0
    inner = frame & (yy >= panel_top + 2)
    img[inner] = 5
    img[frame & (yy < 128) & (((left >= 4) & (left < 5)) | ((right >= 4) & (right < 5)))] = 5
    img[(yy > 146) & frame] = 5
    return img


def dial(kind, level):
    yy, xx = np.mgrid[:24, :24]
    x, y = xx - 11.5, yy - 11.5
    r = x * x + y * y
    img = np.full((24, 24), 5, np.uint8)
    img[r < 130] = 6
    img[r < 100] = 0
    if kind == 'attitude':
        bank, pitch = level % 5 - 2, level // 5 - 1
        img[r < 100] = np.where(y[r < 100] < x[r < 100] * bank * .5 + pitch * 3, 12, 4)
        img[(np.abs(y) < 1) & (np.abs(x) < 7)] = 7
        img[(np.abs(x) < 1) & (np.abs(y) < 3)] = 7
    else:
        for a in np.linspace(-2.4, 2.4, 9):
            tx, ty = round(11.5 + np.sin(a) * 8), round(11.5 - np.cos(a) * 8)
            img[max(0, ty-1):ty+1, max(0, tx-1):tx+1] = 7
        a = -2.4 + level * 4.8 / 15
        for t in np.linspace(0, 8, 20):
            tx, ty = round(11.5 + np.sin(a) * t), round(11.5 - np.cos(a) * t)
            img[ty, tx] = 7
        img[11:13, 11:13] = 6
    indices = np.arange(16) * 3 // 2
    return img[np.ix_(indices, indices)]


def cockpit_tiles():
    frame = cockpit()
    out = []
    for row in range(H // 8):
        for col in range(W // 8):
            patch = frame[row*8:row*8+8, col*8:col*8+8]
            if (patch != 255).all():
                out.append((('flight_panel', row, col), patch))
            elif (patch != 255).any():
                for color in FRAME_COLORS:
                    background = np.full((8, 8), color, np.uint8)
                    out.append((('flight_frame', row, col, color), np.where(patch == 255, background, patch)))
                for split in HORIZON_SPLITS:
                    background = np.where(np.arange(8)[:, None] < split, SKY, SEA).astype(np.uint8).repeat(8, 1)
                    out.append((('flight_frame_horizon', row, col, split), np.where(patch == 255, background, patch)))
    for kind, levels in [('speed', 16), ('altitude', 16), ('attitude', 15)]:
        for level in range(levels):
            img = dial(kind, level)
            for row in range(2):
                for col in range(2):
                    out.append((('flight_dial', kind, level, row, col), img[row*8:row*8+8, col*8:col*8+8]))
    return out


def flight_tiles():
    out = [(('flight_outside', i), tile) for i, tile in enumerate(outside_tiles())]
    out += cockpit_tiles()
    yy, xx = np.mgrid[:8, :8]
    for color in FRAME_COLORS:
        for pattern in range(4):
            distance = np.abs(yy - (xx if pattern % 2 == 0 else 7 - xx))
            if pattern >= 2:
                distance = np.minimum(distance, np.abs(xx - 4))
            tile = np.full((8, 8), color, np.uint8)
            tile[distance <= 1] = 0
            tile[distance == 0] = 7
            out.append((('flight_crack', color, pattern), tile))
    return out
