"""Perception cues for navigation (pure Python, no ROS - tested in perception/test_cues.py).

* DetectionPersist: the camera model's detections (/rc/detections from reactive-core/rc/vision_node.py,
  best.pt: curtain / human / stairs) -> "this class is really there": >= n detections within window_s,
  confidence >= conf, directions within cone_deg of each other.
* stairs_in_result: does a scene-server answer mention stairs?
* mission_status: which tasks are reported, which are open, and a suggestion where to go next, so the
  navigation can keep exploring / backtrack until the tasks are done (booklet v6.3: 12 min per session,
  the exit is beyond the stairs).
* parse_look / fresh_look: reactive-core's /rc/look_cell (the arena cell the dog is looking at from its
  doorway) -> the cell to put scene / QR / survivor / hazard results in while it is fresh.
"""
import collections
import json
import math


def parse_look(data, t):
    """/rc/look_cell JSON {'cell': [i, j], 'look': k, ...} -> (cell, k, t) or None."""
    try:
        v = json.loads(data)
        cell = tuple(int(c) for c in v['cell'])
        if len(cell) != 2:
            return None
        return cell, int(v.get('look', 0)), float(t)
    except (ValueError, TypeError, KeyError, AttributeError):
        return None


def fresh_look(look, now, max_age_s):
    """The look (cell, k, t) if it is at most max_age_s old, else None."""
    if look is None or now - look[2] > max_age_s:
        return None
    return look

STAIRS_WORDS = ('stair', 'staircase', 'stairway', 'steps going up', 'flight of steps', 'step ladder',
                'checker plate steps', 'handrail')


class DetectionPersist:

    def __init__(self, cls, conf=0.5, n=3, window_s=1.5, cone_deg=20.0):
        self.cls, self.conf, self.n, self.window_s = cls, conf, n, window_s
        self.cone = math.radians(cone_deg)
        self.hist = collections.deque()          # (t, conf, bearing, w)
        self.last_msg_t = None

    def update(self, dets, t):
        """dets: [{'cls', 'conf', 'bearing' (rad, + = left), 'w' (box width / image width)}, ...]."""
        self.last_msg_t = t
        for d in dets or []:
            try:
                if d.get('cls') == self.cls and float(d.get('conf', 0.0)) >= self.conf:
                    self.hist.append((t, float(d['conf']), float(d.get('bearing', 0.0)), float(d.get('w', 0.0))))
            except (TypeError, ValueError, AttributeError):
                continue
        while self.hist and t - self.hist[0][0] > self.window_s:
            self.hist.popleft()

    def seen(self, t):
        """The persistent detection as {'conf', 'bearing', 'w', 'n'}, or None."""
        h = [x for x in self.hist if t - x[0] <= self.window_s]
        if len(h) < self.n:
            return None
        ref = h[-1][2]
        same = [x for x in h if abs(x[2] - ref) <= self.cone]
        if len(same) < self.n:
            return None
        b = sorted(x[2] for x in same)[len(same) // 2]
        return {'conf': round(max(x[1] for x in same), 3), 'bearing': round(b, 4),
                'w': round(max(x[3] for x in same), 3), 'n': len(same)}

    def alive(self, t, max_age_s=3.0):
        """Is the detector running (a message arrived recently, even an empty one)?"""
        return self.last_msg_t is not None and t - self.last_msg_t <= max_age_s


def stairs_in_result(res):
    """res: scene_client.parse_result dict -> True if the answer names stairs."""
    txt = (str(res.get('desc', '')) + ' ' + ' '.join(map(str, res.get('evidence', [])))).lower()
    return any(w in txt for w in STAIRS_WORDS)


def mission_status(mission, best, qr_text=None, qr_seen_recent=False, m05_hint='', survivor_cells=(),
                   scene_by_cell=None, stairs_cells=(), t=0.0):
    """mission: task_id -> {'task_type', ...}; best: task_id -> last emitted report.
    Returns the JSON-able status. 'suggest' is ordered: the first entry is the most useful next goal."""
    tasks, open_ = {}, []
    for tid in sorted(mission):
        cfg = mission[tid]
        rep = best.get(tid)
        tasks[tid] = {'type': cfg.get('task_type', ''), 'reported': rep is not None}
        if rep is not None:
            tasks[tid].update({k: rep.get(k) for k in ('scene_type', 'label', 'pos', 'score', 'src')})
            if rep.get('task_type') == 'qr':
                tasks[tid]['message'] = rep.get('message', '')
        else:
            open_.append(tid)
    scenes = {}
    for cell, d in (scene_by_cell or {}).items():
        if d:
            scenes['%d,%d' % tuple(cell)] = max(d, key=d.get)
    suggest = []
    qr_open = any(mission[t_].get('task_type') == 'qr' for t_ in open_)
    if qr_open and (qr_seen_recent or survivor_cells):
        suggest.append('read_qr')                     # a survivor room is known: its side walls carry the QR
    for tid in open_:
        cfg = mission[tid]
        want = cfg.get('scene_type') or (m05_hint if tid == 'M05' else '')
        if cfg.get('task_type') in ('survivor', 'object', 'hazard') and want:
            g = 'find_scene:%s' % want
            if g not in suggest:
                suggest.append(g)
    if open_:
        suggest.append('explore')
    suggest.append('go_stairs' if not open_ else 'go_stairs_when_out_of_time')
    return {'t': round(t, 2), 'tasks': tasks, 'open': open_, 'all_done': not open_,
            'qr': {'read': bool(qr_text), 'text': qr_text or '', 'seen_not_read': bool(qr_seen_recent and not qr_text)},
            'm05_hint': m05_hint or '', 'survivor_cells': [list(c) for c in survivor_cells],
            'scenes': scenes, 'stairs_cells': [list(c) for c in stairs_cells], 'suggest': suggest}
