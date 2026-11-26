"""Core glue for grid navigation."""
import numpy as np

from rc import config
from rc import lanes as L
from rc.core import Core


def test_own_legs_never_block_the_grid_lane():
    # live 2026-11-26: standing still, obstacle cells at 0.05 m ahead of base_link (its own front legs)
    core = Core(config.load())
    core.scan.obs = np.array([[0.05, 0.10], [0.05, -0.10], [0.25, 0.15], [1.40, 0.0]])
    pts = core._nav_pts(0.0)
    assert abs(L.free_distance(pts, 0.0, core.c['lane_w']) - 1.40) < 1e-6
