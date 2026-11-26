"""Cell planner: which cell next, by which route (pure logic on rc/arena.ArenaMap).

Search: Dijkstra over (cell, heading) so turns cost (plan_turn_cost per 90 deg, turning happens only at
cell centres). A move between neighbours costs 1 plus:
  OPEN 0 | UNKNOWN plan_unknown_cost (optimistic: the lidar classifies it on the way) | CURTAIN
  plan_curtain_cost (only with curtain: push) | WALL / blocked for now: impossible.
Entering a stairs cell costs plan_stairs_cost and no turning inside it (the crossing is straight).
Cells outside every live 3 x 5 placement are never entered (the hall behind the final cell's open side,
the start area), except the final-cell candidates in the FINAL phase.

Goal choice (EXPLORE): every unvisited arena cell reachable at finite cost is a target. Each target g is
scored  d(here, g) + plan_tour_weight * NN-tour(g -> the other targets)  - the first step of a greedy
travelling-salesman tour - so the dog does not leave one cell behind and come back for it later.
FINAL: the best reachable final-cell candidate. PROBE: nothing reachable is left but arena cells are
unvisited -> a WALL border between a visited and an unvisited arena cell may be a floor-length curtain:
go to the visited side, face it, ask the camera.
"""
import heapq
import math
from dataclasses import dataclass, field

from rc import arena as A

INF = float('inf')
ORDER = ['N', 'W', 'S', 'E']


def turn_steps(d0, d1):
    if d0 is None or d1 is None:
        return 0
    k = (ORDER.index(d1) - ORDER.index(d0)) % 4
    return min(k, 4 - k)


@dataclass
class Plan:
    goal: tuple = None
    path: list = field(default_factory=list)       # cells, path[0] = start cell
    reason: str = 'none'                           # explore | final | probe | none
    cost: float = INF
    probe_edge: tuple = None
    targets: int = 0                               # unvisited reachable targets left


class CellPlanner:

    def __init__(self, cfg, amap):
        self.c = cfg
        self.m = amap

    # ------------------------------------------------------------------ graph
    def enterable(self, cell, final_phase):
        m = self.m
        if m.visited(cell):
            return True
        if m.possible(cell):
            return True
        return final_phase and cell in m.final_candidates()

    def move_cost(self, a, b, t):
        """Extra cost of the border a -> b, None if impossible."""
        c, m = self.c, self.m
        if m.is_blocked(a, b, t):
            return None
        st = m.state(a, b)
        if st == A.WALL:
            return None
        if st == A.CURTAIN:
            e = m.edges.get(A.edge_key(a, b))
            if c['curtain'] != 'push' or (e is not None and e.result in ('rigid', 'camera_no')):
                return None
            return 0.0 if (e is not None and e.traversed) else c['plan_curtain_cost']
        if st == A.UNKNOWN:
            return c['plan_unknown_cost']
        return 0.0

    def search(self, start, start_dir, t, final_phase=False):
        """Dijkstra from (start, start_dir). Returns (dist per cell, prev map over states, best state per cell)."""
        c, m = self.c, self.m
        span = c['plan_max_cells']
        dist = {(start, start_dir): 0.0}
        prev = {}
        pq = [(0.0, 0, start, start_dir)]
        best = {}
        n = 0
        while pq:
            d, _, cell, hd = heapq.heappop(pq)
            if d > dist.get((cell, hd), INF):
                continue
            if cell not in best or d < best[cell][0]:
                best[cell] = (d, hd)
            in_stairs = m.is_stairs(cell) and cell != start
            for nd in ORDER:
                if in_stairs and hd is not None and nd != hd:
                    continue                                   # no turning on the stairs
                nb = A.step(cell, nd)
                if abs(nb[0] - start[0]) > span or abs(nb[1] - start[1]) > span:
                    continue
                if not self.enterable(nb, final_phase):
                    continue
                extra = self.move_cost(cell, nb, t)
                if extra is None:
                    continue
                cost = d + 1.0 + extra + c['plan_turn_cost'] * turn_steps(hd, nd)
                if m.is_stairs(nb) and not m.is_stairs(cell):
                    cost += c['plan_stairs_cost']
                key = (nb, nd)
                if cost < dist.get(key, INF):
                    dist[key] = cost
                    prev[key] = (cell, hd)
                    n += 1
                    heapq.heappush(pq, (cost, n, nb, nd))
        return best, prev

    @staticmethod
    def _path(prev, best, start, start_dir, goal):
        if goal not in best:
            return []
        state = (goal, best[goal][1])
        out = [goal]
        while state != (start, start_dir) and state in prev:
            state = prev[state]
            out.append(state[0])
        out.reverse()
        if out[0] != start:
            out.insert(0, start)
        # collapse duplicates (a turn in place shows the same cell twice)
        res = [out[0]]
        for cell in out[1:]:
            if cell != res[-1]:
                res.append(cell)
        return res

    def _dists_from(self, cell, t, final_phase=False):
        best, _ = self.search(cell, None, t, final_phase)
        return {k: v[0] for k, v in best.items()}

    # ------------------------------------------------------------------ goals
    def targets(self, best):
        """Unvisited arena cells (not stairs, not a final candidate) reachable in `best`.
        Without bounds (all placements dead) every cell within plan_max_cells is a candidate."""
        m = self.m
        fc = set(m.final_candidates())
        return [cell for cell in best
                if not m.visited(cell) and not m.is_stairs(cell) and cell not in fc and m.possible(cell)]

    def plan(self, start, start_dir, t, phase='explore'):
        c, m = self.c, self.m
        if phase == 'final':
            return self.plan_final(start, start_dir, t)
        best, prev = self.search(start, start_dir, t)
        T = self.targets(best)
        if not T:
            pr = self.plan_probe(start, start_dir, t, best, prev)
            if pr is not None:
                return pr
            return Plan(reason='none', targets=0)
        if len(T) == 1 or c['plan_tour_weight'] <= 0:
            g = min(T, key=lambda x: best[x][0])
        else:
            # the tour runs over the LIKELY arena cells (in at least half the live placements): cells of
            # unlikely placements would pull the dog towards places that probably are not there
            likely = [g for g in T if m.weight(g) >= 0.5] or T
            dmat = {g: self._dists_from(g, t) for g in set(likely) | set(T)}

            def tour(g):
                left = set(likely) - {g}
                cur, tot = g, 0.0
                while left:
                    nxt = min(left, key=lambda x: dmat[cur].get(x, 50.0))
                    tot += dmat[cur].get(nxt, 50.0)
                    left.discard(nxt)
                    cur = nxt
                return tot

            g = min(T, key=lambda x: (best[x][0] + c['plan_tour_weight'] * tour(x), best[x][0]))
        path = self._path(prev, best, start, start_dir, g)
        return Plan(goal=g, path=path, reason='explore', cost=best[g][0], targets=len(T))

    def plan_to(self, start, start_dir, t, goal, final_phase=False):
        best, prev = self.search(start, start_dir, t, final_phase)
        if goal not in best:
            return None
        return Plan(goal=goal, path=self._path(prev, best, start, start_dir, goal), reason='explore',
                    cost=best[goal][0])

    def plan_final(self, start, start_dir, t):
        m = self.m
        best, prev = self.search(start, start_dir, t, final_phase=True)
        for f in m.final_candidates():
            if f in best:
                return Plan(goal=f, path=self._path(prev, best, start, start_dir, f), reason='final',
                            cost=best[f][0])
        return Plan(reason='none')

    def plan_probe(self, start, start_dir, t, best, prev):
        """A wall-like border between a visited cell and an unvisited arena cell: maybe a curtain."""
        c, m = self.c, self.m
        if c['curtain'] != 'push' or not c['cur_probe']:
            return None
        if not m.live() or m.no_bounds:
            return None
        cands = []
        for k, e in m.edges.items():
            if e.state != A.WALL or e.result in ('rigid', 'camera_no', 'passed') or e.traversed:
                continue
            a, b = A.edge_cells(k)
            for v, u in ((a, b), (b, a)):
                if m.visited(v) and not m.visited(u) and m.certain(u) and v in best:
                    cands.append((best[v][0], v, k, u))
        if not cands:
            return None
        cands.sort()
        d, v, k, u = cands[0]
        return Plan(goal=v, path=self._path(prev, best, start, start_dir, v), reason='probe', cost=d,
                    probe_edge=k)

    # ------------------------------------------------------------------ time
    def eta_s(self, path, start_dir=None):
        """Rough time to drive a path (cells): straight cells, turns, stairs."""
        c, m = self.c, self.m
        if not path:
            return 0.0
        t = 0.0
        hd = start_dir
        for a, b in zip(path, path[1:]):
            d = A.dir_between(a, b)
            t += c['arena_cell_m'] / max(0.05, c['cruise_vx']) + c['eta_cell_s']
            t += turn_steps(hd, d) * c['eta_turn_s']
            if m.is_stairs(b) and not m.is_stairs(a):
                t += c['eta_stairs_s']
            hd = d
        return t
