"""Short-term obstacle memory (spec section 5): obstacle cells seen recently stay known while they
move into the blind zones beside / behind the dog. Stored in the odom frame on a 0.1 m grid,
returned in the body frame. Entries expire after mem_time_s or mem_dist_m of travel since last seen.
A cell is also forgotten as soon as the current scan sees open ground in it (clear_seen): memory is
for the blind zones only, and stale or mis-levelled cells must not block lanes the lidar sees as free.
"""
import numpy as np

from rc import geometry as G

RES = 0.1


class ObstacleMemory:

    def __init__(self, cfg):
        self.c = cfg
        self.cells = {}       # (i, j) -> (t_seen, travelled_seen)

    def clear(self):
        self.cells.clear()

    def add(self, obs_body_xy, pose, t, travelled):
        obs_body_xy = np.asarray(obs_body_xy, dtype=float).reshape(-1, 2)
        if obs_body_xy.shape[0] == 0:
            return
        w = G.body_to_world(obs_body_xy, pose)
        ij = np.floor(w / RES).astype(np.int64)
        for k in set(map(tuple, ij.tolist())):
            self.cells[k] = (t, travelled)

    def clear_seen(self, ground_body_xy, high_body_xy, pose, min_pts=2):
        """Forget cells with >= min_pts ground returns and no obstacle-height return in this scan.
        Returns the number of cells forgotten."""
        if not self.cells:
            return 0
        g = np.asarray(ground_body_xy, dtype=float).reshape(-1, 2)
        if g.shape[0] == 0:
            return 0
        ij, cnt = np.unique(np.floor(G.body_to_world(g, pose) / RES).astype(np.int64), axis=0, return_counts=True)
        seen = set(map(tuple, ij[cnt >= min_pts].tolist()))
        gone = [k for k in self.cells if k in seen]
        if not gone:
            return 0
        h = np.asarray(high_body_xy, dtype=float).reshape(-1, 2)
        if h.shape[0]:
            high = set(map(tuple, np.floor(G.body_to_world(h, pose) / RES).astype(np.int64).tolist()))
            gone = [k for k in gone if k not in high]
        for k in gone:
            del self.cells[k]
        return len(gone)

    def _prune(self, t, travelled):
        tm, dm = self.c['mem_time_s'], self.c['mem_dist_m']
        dead = [k for k, (ts, ds) in self.cells.items() if t - ts > tm or travelled - ds > dm]
        for k in dead:
            del self.cells[k]

    def query(self, pose, t, travelled):
        self._prune(t, travelled)
        if not self.cells:
            return np.zeros((0, 2))
        w = (np.array(list(self.cells.keys()), dtype=float) + 0.5) * RES
        return G.world_to_body(w, pose)

    def __len__(self):
        return len(self.cells)
