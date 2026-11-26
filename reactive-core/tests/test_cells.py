"""nav_mode cells (rc/cells.py): geometry helpers + short closed-loop runs on course-shaped worlds.
The full acceptance matrix (mazes, messy layouts, 3 seeds, odometry drift) is tools/nav_eval.py."""
import math

import numpy as np
import pytest

from rc import cells as CL
from rc import config
from rc.driver import Driver
from rc.grid import Grid
from sim import course as C
from sim.harness import simulate

CELLS = config.load(overrides=['nav_mode=cells'])


def corridor_cells(rot_deg=0.0, off=(0.0, 0.0), opening=None, W=0.92, rng=None):
    """0.1 m obstacle cells (body frame) of a straight corridor along +x, optionally with a left
    opening (u0, u1), seen by a dog at `off` rotated by rot_deg against the corridor."""
    rng = rng or np.random.default_rng(0)
    xs = np.arange(-1.0, 3.0, 0.02)
    left = [(x, W / 2) for x in xs if not (opening and opening[0] < x < opening[1])]
    pts = np.array(left + [(x, -W / 2) for x in xs]) - np.asarray(off)
    a = math.radians(-rot_deg)
    b = np.stack([math.cos(a) * pts[:, 0] - math.sin(a) * pts[:, 1],
                  math.sin(a) * pts[:, 0] + math.cos(a) * pts[:, 1]], 1)
    b += rng.normal(0, 0.01, b.shape)
    return np.unique(np.floor(b / 0.1) * 0.1 + 0.05, axis=0)


@pytest.mark.parametrize('rot', [-9.0, -3.0, 0.0, 4.5, 8.0])
def test_wall_angle_mod_90(rot):
    a, conf = CL.wall_angle(corridor_cells(rot, off=(0.3, 0.1)))
    assert conf > 0.5
    assert abs(math.degrees(CL.wrap90(a - math.radians(-rot)))) < 1.5


def test_side_gap_centre():
    p = corridor_cells(opening=(1.0, 1.92))
    u, v = CL.to_axis(p, 0.0)
    gaps = [g for g in CL.side_gaps(u, v, 1, 0.85) if g[2] and g[3]]
    assert len(gaps) == 1
    uc = (gaps[0][0] + gaps[0][1]) / 2
    assert abs(uc - 1.46) < 0.12
    assert not [g for g in CL.side_gaps(u, v, -1, 0.85) if g[2] and g[3]]   # right wall is closed


def test_centre_offset_points_back_to_the_middle():
    u, v = CL.to_axis(corridor_cells(off=(0.0, -0.10)), 0.0)    # dog 0.10 m right of the centre
    assert 0.03 < CL.centre_offset(u, v) < 0.17


def test_default_is_not_cells():
    cfg = config.load()
    assert cfg['nav_mode'] != 'cells'          # opt-in: --set nav_mode=cells
    assert Driver(cfg, Grid()).cells is None
    assert Driver(CELLS, Grid()).cells is not None


class Tilt:
    """Counts on-the-spot turns while the (true) tilt is above 10 deg."""
    def __init__(self):
        self.n = 0

    def __call__(self, t, dog, core, cmd):
        tilt = max(abs(math.degrees(dog.pitch)), abs(math.degrees(dog.roll)))
        if core.driver.mode != 'STAIRS' and tilt > 10 and abs(dog.wz) > 0.1 and abs(dog.vx) < 0.05:
            self.n += 1


def _reached(cw, r):
    return any(cw.goal(x, y) for x, y in r.path)


@pytest.mark.parametrize('turn', ['left', 'right'])
def test_cells_rounds_a_corner(turn):
    # main and testing (lanes) never got round this corner in the simulator (2026-11-25)
    cw = C.l_corner(turn)
    r = simulate(cw.world, CELLS, T=45.0, start=cw.start)
    assert r.collisions == 0 and not r.reversed
    assert _reached(cw, r)


def test_cells_start_bay_with_cylinder_and_drift():
    cw = C.start_bay()
    pr = Tilt()
    r = simulate(cw.world, CELLS, T=45.0, start=cw.start, dog_kwargs=dict(cw.dog_kwargs, odom_drift=(1.05, 0.001)),
                 probe=pr)
    assert r.collisions == 0 and not r.reversed
    # walked over the cylinder: the model's support is the mean of front and rear feet when they
    # differ, and a 0.2 m wide bar never carries both (feet 0.6 m apart) -> 0.1
    assert r.max_support >= 0.08
    assert _reached(cw, r)
    assert pr.n == 0


def test_cells_never_turns_on_the_spot_while_tilted():
    # the 24 Nov fall: front legs on a 20 cm slab, wall behind, turned on the spot -> rolled over
    # rooms off: the slab cell is a 3-wall dead end, which the room look would only look at from outside
    cw = C.slab_wall()
    pr = Tilt()
    cfg = config.load(overrides=['nav_mode=cells', 'cells_rooms=false'])
    r = simulate(cw.world, cfg, T=30.0, start=cw.start, dog_kwargs=cw.dog_kwargs, probe=pr)
    assert r.collisions == 0 and not r.reversed
    assert pr.n == 0
    assert r.final_mode == 'HOLD' and r.hold_reason == 'tilted'
