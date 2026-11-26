"""Camera model detections (/rc/detections from rc/vision_node.py, best.pt: curtain / human / stairs) ->
"this class is really there": >= n detections within window_s, conf >= conf, bearings within cone_deg.
Copied from frontier-fastlio/dog_codes/perception_cues.py (each folder stands alone)."""
import collections
import math


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
        return self.last_msg_t is not None and t - self.last_msg_t <= max_age_s
