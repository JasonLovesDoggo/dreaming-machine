"""Flight atlas art, independent of simulator state and atlas construction."""
import numpy as np

W, H = 224, 152
PAIRS = [(12, 1), (12, 14), (12, 11), (1, 13), (13, 11), (11, 14), (12, 7), (11, 5), (5, 7), (1, 11), (13, 14)]
FILLS = [12, 1, 14, 13, 7, 5, 0, 6, 11]
FRAME_COLORS = [12, 1, 14, 13, 11, 7]


def outside_tiles():
    yy, xx = np.mgrid[:8, :8]
    out = [np.full((8, 8), c, np.uint8) for c in FILLS]
    for a, b in PAIRS:
        for angle in np.arange(16) * np.pi / 8:
            for offset in (-3, -1, 1, 3):
                mask = (xx - 3.5) * np.cos(angle) + (yy - 3.5) * np.sin(angle) > offset
                out.append(np.where(mask, b, a).astype(np.uint8))
    out.append(np.where((xx + yy * 3) % 16 == 0, 7, 12).astype(np.uint8))
    out.append(np.where((xx + yy * 2) % 16 == 0, 12, 1).astype(np.uint8))
    for a, b in ((12, 7), (11, 14)):
        for coverage in (1, 2, 3):
            out.append(np.where((xx + yy * 2) % 4 < coverage, b, a).astype(np.uint8))
    return out


def cockpit():
    yy, xx = np.mgrid[:H, :W]
    img = np.full((H, W), 255, np.uint8)
    frame = (yy < 4) | (xx < 4) | (xx >= W - 4)
    left = np.abs(xx - (16 + yy * .5))
    right = np.abs(xx - (W - 17 - yy * .5))
    frame |= ((left < 5) | (right < 5)) & (yy < 104)
    panel_top = 104
    frame |= yy >= panel_top
    img[frame] = 0
    inner = frame & (yy >= panel_top + 2)
    img[inner] = 5
    img[frame & (yy < 104) & (((left >= 4) & (left < 5)) | ((right >= 4) & (right < 5)))] = 5
    img[(yy > 146) & frame] = 5
    for x0 in (39, 168):
        img[114:141, x0:x0 + 20] = 5
        for row in range(3):
            for col in range(3):
                img[117 + row * 7:121 + row * 7, x0 + 3 + col * 5:x0 + 6 + col * 5] = 14 if (row + col) % 3 else 11
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
    return img


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
                    out.append((('flight_frame', row, col, color), np.where(patch == 255, color, patch)))
    for kind, levels in [('speed', 16), ('altitude', 16), ('attitude', 15)]:
        for level in range(levels):
            img = dial(kind, level)
            for row in range(3):
                for col in range(3):
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
