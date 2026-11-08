import math
import numpy as np
from .level import VW, VH, GRID, NANG

PLANE = 0.66                                # camera plane half-width (FOV ~66 degrees)
CAM = 2 * (np.arange(VW) + 0.5) / VW - 1    # per-column camera x in [-1, 1]
NEAR = 0.2                                  # sprites closer than this are not drawn


def direction(ang):
    a = ang * 2 * math.pi / NANG
    return math.cos(a), math.sin(a)


def cast(px, py, ang):
    """DDA raycast of all VW columns -> (wall_type, side, height, texture_u, distance) arrays."""
    dx, dy = direction(ang)
    rdx, rdy = dx - dy * PLANE * CAM, dy + dx * PLANE * CAM
    mx, my = np.full(VW, int(px)), np.full(VW, int(py))
    with np.errstate(divide='ignore'):
        ddx = np.where(rdx == 0, 1e9, np.abs(1 / rdx))
        ddy = np.where(rdy == 0, 1e9, np.abs(1 / rdy))
    stepx, stepy = np.where(rdx < 0, -1, 1), np.where(rdy < 0, -1, 1)
    sdx = np.where(rdx < 0, (px - mx) * ddx, (mx + 1 - px) * ddx)
    sdy = np.where(rdy < 0, (py - my) * ddy, (my + 1 - py) * ddy)
    side, wtype = np.zeros(VW, np.int64), np.zeros(VW, np.int64)
    active = np.arange(VW)
    while active.size:
        x_first = sdx[active] < sdy[active]
        ix, iy = active[x_first], active[~x_first]
        sdx[ix] += ddx[ix]
        mx[ix] += stepx[ix]
        side[ix] = 0
        sdy[iy] += ddy[iy]
        my[iy] += stepy[iy]
        side[iy] = 1
        cell = GRID[my[active], mx[active]]
        hit = cell > 0
        wtype[active[hit]] = cell[hit]
        active = active[~hit]
    dist = np.maximum(np.where(side == 0, sdx - ddx, sdy - ddy), 1e-3)
    wallx = np.where(side == 0, py + dist * rdy, px + dist * rdx)
    u = ((wallx - np.floor(wallx)) * 16).astype(np.int64).clip(0, 15)
    h = np.minimum(255, (VH / dist).astype(np.int64))
    return wtype, side, h, u, dist


def project(px, py, ang, points):
    """Screen position of world points -> [(screen_x, size, depth)]; depth -1 means behind the camera."""
    dx, dy = direction(ang)
    plx, ply = -dy * PLANE, dx * PLANE
    inv = 1.0 / (plx * dy - dx * ply)
    out = []
    for x, y in points:
        rx, ry = x - px, y - py
        tx = inv * (dy * rx - dx * ry)
        ty = inv * (-ply * rx + plx * ry)
        if ty <= NEAR:
            out.append((-1, 0, -1.0))
        else:
            out.append((int(VW / 2 * (1 + tx / ty)), min(255, int(VH / ty)), ty))
    return out
