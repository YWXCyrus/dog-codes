"""Simulated RealSense + perception for the QR approach (rc/qr.py).

A square QR code on a vertical surface, seen by a 640x480 pinhole camera mounted like the dog's
(/tf_static: camera_link 0.252 m ahead of base_link, 0.074 m above it, pitched 20 deg down;
fx = fy = 462.6). Produces the events of frontier perception.py on /quad/event at ~6 Hz:
  qr_decoded {'text', 'w', 'cx', 'cy'}  - read: QR side >= read_px, seen within max_incidence,
                                          camera (nearly) still, entirely in the picture
  qr_seen    {'w', 'cx', 'cy'}           - located but not read, after perception's persistence
                                          filter (3 detections within 1 s, centres within 50 px)
read_px = 165 and the blur rule come from qr-test selftest (2026-11-23). Optional false
locations (OpenCV's locator firing on panels: 200-470 px wide in the 2026-11-22 recordings).
"""
import math

import numpy as np

CAM_X, CAM_Z, CAM_PITCH = 0.252, 0.074, math.radians(20.0)
W_IMG, H_IMG = 640, 480


class QrCam:

    def __init__(self, world, x, y, z, normal_yaw, size=0.15, text='Survivor is in the kitchen',
                 fx=462.6, cx0=320.1, cy0=238.4, read_px=165.0, max_incidence_deg=55.0,
                 false_wide_rate=0.0, false_small_rate=0.0, seed=0):
        self.w = world
        self.p = np.array([x, y, z], float)                # QR centre
        self.n = normal_yaw                                # direction the QR faces
        self.size, self.text = size, text
        self.fx, self.cx0, self.cy0 = fx, cx0, cy0
        self.read_px, self.max_inc = read_px, math.radians(max_incidence_deg)
        self.false_wide_rate, self.false_small_rate = false_wide_rate, false_small_rate
        self.rng = np.random.default_rng(seed)
        self.hist = []                                     # (t, cx) of located detections
        self.read = False
        self.n_seen = self.n_read_frames = 0

    # ---- camera model
    def _cam(self, dog):
        c, s = math.cos(dog.yaw), math.sin(dog.yaw)
        pos = np.array([dog.x + CAM_X * c, dog.y + CAM_X * s, dog.base_z() + CAM_Z])
        pitch = CAM_PITCH + getattr(dog, 'pitch', 0.0)     # + = nose down
        return pos, c, s, pitch

    def _project(self, pts, cam):
        pos, c, s, th = cam
        d = pts - pos
        f = d[:, 0] * c + d[:, 1] * s                      # forward (yaw frame)
        l = -d[:, 0] * s + d[:, 1] * c                     # left
        up = d[:, 2]
        depth = f * math.cos(th) - up * math.sin(th)
        vup = f * math.sin(th) + up * math.cos(th)
        u = self.cx0 - self.fx * l / depth
        v = self.cy0 - self.fx * vup / depth
        return u, v, depth

    def _occluded(self, cam_pos):
        """Any solid box between the camera and the QR (except the last 4 cm: its own wall)."""
        seg = self.p - cam_pos
        L = float(np.linalg.norm(seg))
        k = np.linspace(0.0, max(0.0, L - 0.04) / L, max(2, int(L / 0.03)))
        pts = cam_pos[None, :] + k[:, None] * seg[None, :]
        for b in self.w.boxes:
            inside = b.contains_xy(pts[:, 0], pts[:, 1]) & (pts[:, 2] >= b.z0) & (pts[:, 2] <= b.z1)
            if inside.any():
                return True
        return False

    def view(self, dog):
        """(w, cx, cy, incidence) of the QR in the picture, or None if not in view."""
        cam = self._cam(dog)
        tng = np.array([-math.sin(self.n), math.cos(self.n), 0.0])
        h = self.size / 2.0
        corners = np.array([self.p + a * h * tng + b * h * np.array([0, 0, 1.0])
                            for a, b in ((-1, 1), (1, 1), (1, -1), (-1, -1))])
        to_cam = cam[0] - self.p
        nvec = np.array([math.cos(self.n), math.sin(self.n), 0.0])
        cos_inc = float(to_cam @ nvec) / max(1e-6, float(np.linalg.norm(to_cam)))
        if cos_inc <= math.cos(math.radians(80.0)):
            return None                                    # seen from behind / edge-on
        u, v, depth = self._project(np.vstack([corners, self.p[None]]), cam)
        if (depth <= 0.1).any():
            return None
        if (u[:4] < 0).any() or (u[:4] > W_IMG).any() or (v[:4] < 0).any() or (v[:4] > H_IMG).any():
            return None
        if self._occluded(cam[0]):
            return None
        side = np.mean([math.hypot(u[i] - u[(i + 1) % 4], v[i] - v[(i + 1) % 4]) for i in range(4)])
        return float(side), float(u[4]), float(v[4]), math.acos(min(1.0, cos_inc))

    # ---- perception model (call at ~6 Hz)
    def observe(self, dog, t):
        out = []
        moving = abs(dog.vx) > 0.03 or abs(dog.wz) > 0.06
        r = self.view(dog)
        if r is not None:
            w, cx, cy, inc = r
            if (w >= self.read_px and inc <= self.max_inc and not moving
                    and self.rng.random() < 0.85):
                self.read = True
                self.n_read_frames += 1
                out.append({'type': 'qr_decoded', 'text': self.text, 'w': round(w, 1),
                            'cx': round(cx, 1), 'cy': round(cy, 1)})
                return out
            p_loc = 0.9 if not moving else (0.55 if abs(dog.wz) < 0.25 else 0.2)
            if w >= 20.0 and self.rng.random() < p_loc and self._persist(t, cx):
                self.n_seen += 1
                out.append({'type': 'qr_seen', 'w': round(w, 1), 'cx': round(cx, 1), 'cy': round(cy, 1)})
        if self.false_wide_rate and self.rng.random() < self.false_wide_rate:
            out.append({'type': 'qr_seen', 'w': float(self.rng.uniform(200, 470)),
                        'cx': float(self.rng.uniform(150, 490)), 'cy': float(self.rng.uniform(150, 330))})
        if self.false_small_rate and self.rng.random() < self.false_small_rate:
            out.append({'type': 'qr_seen', 'w': float(self.rng.uniform(40, 120)),
                        'cx': float(self.rng.uniform(100, 540)), 'cy': float(self.rng.uniform(120, 360))})
        return out

    def _persist(self, t, cx, n=3, window=1.0, px=50.0):
        self.hist = [(tt, x) for tt, x in self.hist if t - tt <= window] + [(t, cx)]
        return sum(1 for _, x in self.hist if abs(x - cx) <= px) >= n
