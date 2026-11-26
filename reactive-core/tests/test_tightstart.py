"""Start in a ~1 m wide start bay (the course start, photo 2026-11-22) with the dog off-centre.

Run 20261122_140917: right wall 0.35 m from base_link. With no 0.70 m lane free the dog turns on
the spot (no turn guard since 0a525e4) and hits the bay walls in the simulator. With the
lane_w_min (0.50 m) fallback it walks out straight."""
from rc import config
from sim import world as W
from sim.harness import simulate


def bay_world(offset):
    w = W.World()
    w.wall(-1.0, -2.0, 6.0, -2.0)
    w.wall(6.0, -2.0, 6.0, 2.0)
    w.wall(6.0, 2.0, -1.0, 2.0)
    w.wall(-1.0, 2.0, -1.0, -2.0)
    w.wall(-0.45, -0.5 - offset, -0.45, 0.5 - offset, name='back')
    w.wall(-0.45, 0.5 - offset, 1.2, 0.5 - offset, name='bay_left')
    w.wall(-0.45, -0.5 - offset, 1.2, -0.5 - offset, name='bay_right')
    return w


def test_off_centre_start_walks_out():
    for side in ('left', 'right'):
        r = simulate(bay_world(0.15), config.load(overrides=['wall_side=' + side, 'nav_mode=wall']), T=20.0)
        assert r.startup == 'ready'
        assert r.collisions == 0 and not r.reversed
        assert r.end_pose[0] > 1.0                  # out of the bay


def test_off_centre_start_old_lane_width_without_fallback():
    # until bd6e0ed this collided (turning on the spot in the bay); with the short-term memory
    # replaced by the FAST-LIO map (2026-11-23) the ghost cells are gone and it walks out too
    r = simulate(bay_world(0.15), config.load(overrides=['lane_w=0.70', 'lane_w_min=0.70', 'nav_mode=wall']), T=15.0)
    assert r.collisions == 0 and r.end_pose[0] > 1.0
