"""Rasterise a sim world (obstacles taller than step_limit) into a Nav2 map: map.pgm + map.yaml.

    python3 tools/nav2/make_map.py OUT_DIR [arena|corridor]

Free = 254, occupied = 0, resolution 0.05 m, origin (-1.5, -1.5) so the arena (x -1..7, y -1..5) fits.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from sim import world as W  # noqa: E402

RES, X0, Y0, X1, Y1 = 0.05, -1.5, -1.5, 7.5, 5.5


def rasterise(world, min_h=0.25):
    nx, ny = int((X1 - X0) / RES), int((Y1 - Y0) / RES)
    xs = X0 + (np.arange(nx) + 0.5) * RES
    ys = Y0 + (np.arange(ny) + 0.5) * RES
    gx, gy = np.meshgrid(xs, ys)
    occ = np.zeros((ny, nx), dtype=bool)
    for b in world.boxes:
        if b.z1 >= min_h and b.name not in W.PASS:
            occ |= b.contains_xy(gx, gy)
    return occ


def main():
    out = sys.argv[1]
    which = sys.argv[2] if len(sys.argv) > 2 else 'arena'
    world = W.arena_default() if which == 'arena' else W.corridor_world(8.0, 1.4)
    occ = rasterise(world)
    img = np.where(occ, 0, 254).astype(np.uint8)[::-1]            # image row 0 = top = max y
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'map.pgm'), 'wb') as f:
        f.write(b'P5\n%d %d\n255\n' % (img.shape[1], img.shape[0]))
        f.write(img.tobytes())
    with open(os.path.join(out, 'map.yaml'), 'w') as f:
        f.write('image: map.pgm\nmode: trinary\nresolution: %.3f\norigin: [%.2f, %.2f, 0.0]\n'
                'negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.25\n' % (RES, X0, Y0))
    print('map %dx%d, %d occupied cells -> %s' % (img.shape[1], img.shape[0], int(occ.sum()), out))


if __name__ == '__main__':
    main()
