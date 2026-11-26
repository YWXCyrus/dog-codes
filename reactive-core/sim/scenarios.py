"""Named competition-like scenarios for grid navigation (used by tests/test_grid_sim.py and tools/sim_batch.py).

Each scenario returns (world, info) where info has the layout (rect, final, curtains ...) for the checks.
Arena frame = world frame: start cell (0, 0) centred at the origin, the dog facing +X.
"""
from sim import world as W

FAST_LIDAR = {'az_step_deg': 2.0, 'elev_step_deg': 2.0, 'max_range': 6.0}


def maze(seed, axis='x', sign=1, offset=-1, curtain=False, boards=0, props=0):
    import random
    lay = W.random_layout(seed, axis=axis, sign=sign, offset=offset, curtain=curtain)
    rnd = random.Random(seed + 1000)
    cells = sorted(set((i, j) for i in range(lay['rect'][0], lay['rect'][1] + 1)
                       for j in range(lay['rect'][2], lay['rect'][3] + 1)) - {(0, 0)})
    rnd.shuffle(cells)
    b = cells[:boards]
    # props (scene furniture) stand in dead-end rooms, against the wall opposite the door
    deg = {c: [k for k in lay['open_edges'] if c in ((k[0], k[1]), (k[0] + (k[2] == 'x'), k[1] + (k[2] == 'y')))]
           for c in cells}
    rooms = [c for c in cells if len(deg[c]) == 1 and c not in b][:props]
    pr = []
    for c in rooms:
        k = deg[c][0]
        door = (k[0], k[1]) if (k[0], k[1]) != c else (k[0] + (k[2] == 'x'), k[1] + (k[2] == 'y'))
        pr.append((c[0] + 0.3 * (c[0] - door[0]), c[1] + 0.3 * (c[1] - door[1])))
    w = W.build_arena(boards=b, props=pr, **lay)
    return w, dict(lay, boards=b, props=pr)


def stairs_arena(rise=0.15, n=2):
    """3 x 5 along +X, stairs in cells (2, 0) - (3, 0) (up from x = 1.53 + 0.2): the only way to the far
    half and the final cell (5, 0). The start half has the cells i = 0, 1; the far half i = 2..4 (+ final)."""
    rect = (0, 4, -1, 1)
    o = set()

    def op(a, b):
        o.add(W._ekey(a, b))
    for j in (-1, 0, 1):
        op((0, j), (1, j))
    op((0, 0), (0, 1))
    op((0, 0), (0, -1))
    op((1, 0), (1, 1))
    op((1, -1), (1, 0))
    op((1, 0), (2, 0))          # into the stairs
    op((2, 0), (3, 0))          # along the stairs
    op((3, 0), (4, 0))          # out of the stairs
    for j in (-1, 0, 1):
        op((4, j), (3, j)) if j != 0 else None
    op((4, 0), (4, 1))
    op((4, 0), (4, -1))
    op((2, 1), (3, 1))
    op((2, -1), (3, -1))
    op((4, 0), (5, 0))          # final
    w = W.build_arena(rect, (5, 0), o, stairs={'cells': [(2, 0), (3, 0)], 'dir': 'x', 'sign': 1, 'n': n,
                                               'rise': rise, 'tread': 0.25, 'platform': 0.5})
    return w, {'rect': rect, 'final': (5, 0), 'open_edges': o, 'curtains': set(), 'stairs': [(2, 0), (3, 0)]}


LEAF_SEED = 9                 # random_layout(9, offset=-1, curtain='leaf'): cell (0, -1) only through a curtain
SCENARIOS = {
    'maze1': lambda: maze(1),
    'maze2_y': lambda: maze(2, axis='y', sign=1, offset=-1),
    'maze3_back': lambda: maze(3, axis='x', sign=-1, offset=0),
    'maze4_side': lambda: maze(4, axis='y', sign=-1, offset=-2),
    'maze5_corner': lambda: maze(5, axis='x', sign=1, offset=0),
    'maze6_boards': lambda: maze(6, boards=4),
    'maze7_props': lambda: maze(7, props=2),
    'curtain_leaf': lambda: maze(LEAF_SEED, curtain='leaf'),
    'curtain_alt': lambda: maze(9, curtain='alt'),
    'stairs': lambda: stairs_arena(),
}
DRIFT = dict(odom_mode='drift', odom_scale=1.06, odom_yaw_drift=0.002, gyro_scale=0.03)
