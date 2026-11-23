"""Simulated camera check of a curtain (perception + scene server behind /quad/vlm_ask).

Answers {'id', 'answer', 'text'} after `delay` s: True if a cloth box ('curtain' or the rigid
look-alike 'stiff_curtain' - the camera cannot feel stiffness) is within 1.6 m ahead and +-35 deg,
else False. mode: 'ok' | 'down' (never answers: scene server unusable) | 'no' (always False).
"""
import math


class VlmSim:

    def __init__(self, world, delay=3.5, mode='ok'):
        self.w, self.delay, self.mode = world, delay, mode
        self.pending = []
        self.n_asked = 0

    def ask(self, q, dog, t):
        self.n_asked += 1
        if self.mode == 'down':
            return
        yes = False
        if self.mode == 'ok':
            for b in self.w.boxes:
                if b.name not in ('curtain', 'stiff_curtain'):
                    continue
                dx, dy = b.cx - dog.x, b.cy - dog.y
                d = math.hypot(dx, dy)
                a = math.atan2(dy, dx) - dog.yaw
                a = math.atan2(math.sin(a), math.cos(a))
                yes = yes or (d < 1.6 + b.sx / 2 and abs(a) < math.radians(35))
        self.pending.append((t + self.delay, {'id': q['id'], 'answer': yes,
                                              'text': 'a hanging cloth curtain' if yes else 'a plain wall'}))

    def due(self, t):
        out = [a for (tt, a) in self.pending if tt <= t]
        self.pending = [(tt, a) for (tt, a) in self.pending if tt > t]
        return out
