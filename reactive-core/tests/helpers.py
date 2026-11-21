"""Synthetic point sets in the LEVELLED body frame (base_link at z=0, floor at FLOOR)."""
import numpy as np

FLOOR = -0.32


def grid_pts(x0, x1, y0, y1, z, step=0.05):
    xs = np.arange(x0, x1 + 1e-9, step)
    ys = np.arange(y0, y1 + 1e-9, step)
    X, Y = np.meshgrid(xs, ys)
    return np.stack([X.ravel(), Y.ravel(), np.full(X.size, z)], axis=1)


def floor(x0=1.4, x1=4.0, y0=-1.5, y1=1.5, z=FLOOR, step=0.05):
    return grid_pts(x0, x1, y0, y1, z, step)


def wall_x(x, y0, y1, z0=FLOOR, z1=FLOOR + 0.6, step=0.03):
    """Vertical wall facing the dog at distance x."""
    ys = np.arange(y0, y1 + 1e-9, step)
    zs = np.arange(z0, z1 + 1e-9, step)
    Y, Z = np.meshgrid(ys, zs)
    return np.stack([np.full(Y.size, x), Y.ravel(), Z.ravel()], axis=1)


def wall_y(y, x0, x1, z0=FLOOR, z1=FLOOR + 0.6, step=0.03):
    xs = np.arange(x0, x1 + 1e-9, step)
    zs = np.arange(z0, z1 + 1e-9, step)
    X, Z = np.meshgrid(xs, zs)
    return np.stack([X.ravel(), np.full(X.size, y), Z.ravel()], axis=1)


def box_top(x0, x1, y0, y1, h, step=0.04):
    """Top surface of a box of height h standing on the floor (plus its front face)."""
    top = grid_pts(x0, x1, y0, y1, FLOOR + h, step)
    face = wall_x(x0, y0, y1, FLOOR, FLOOR + h, step)
    return np.vstack([top, face])


def stairs(x0, n, rise, tread, y0=-0.5, y1=0.5, step=0.04):
    parts = []
    for i in range(n):
        xa = x0 + i * tread
        parts.append(grid_pts(xa, xa + tread - 1e-6, y0, y1, FLOOR + (i + 1) * rise, step))
        parts.append(wall_x(xa, y0, y1, FLOOR + i * rise, FLOOR + (i + 1) * rise, step))
    return np.vstack(parts)


def rotate_z(pts, a):
    c, s = np.cos(a), np.sin(a)
    out = pts.copy()
    out[:, 0] = c * pts[:, 0] - s * pts[:, 1]
    out[:, 1] = s * pts[:, 0] + c * pts[:, 1]
    return out
