"""Layer registry. A layer = name + type + provider function (+ style, default on/off).

Types the page can draw: cells, points, path, markers, text. A new feature = one more
registry.layer(...) call; a new drawing type = one renderer in index.html.
Coordinates: grid metres, +x right of the start heading, +y along it (same as the console).
"""
import time
import traceback

TYPES = ('cells', 'points', 'path', 'markers', 'text')


class Registry:

    def __init__(self):
        self.layers = []

    def layer(self, name, type_, fn, style=None, default_on=True):
        if type_ not in TYPES:
            raise ValueError('unknown layer type %r' % type_)
        self.layers = [l for l in self.layers if l['name'] != name]
        self.layers.append({'name': name, 'type': type_, 'fn': fn, 'style': style or {},
                            'default_on': bool(default_on)})

    def snapshot(self):
        out = []
        for l in self.layers:
            item = {'name': l['name'], 'type': l['type'], 'style': l['style'], 'default_on': l['default_on']}
            try:
                item['data'] = l['fn']()
            except Exception as e:                 # a broken layer never breaks the snapshot
                item['data'] = None
                item['error'] = '%s: %s' % (type(e).__name__, e)
                item['trace'] = traceback.format_exc(limit=2)[-300:]
            out.append(item)
        return {'t': round(time.time(), 2), 'layers': out}


def build_registry(core):
    """The layers of the first version: grid + dog on; obstacles, status, stairs prepared but off."""
    r = Registry()

    def cells():
        g = core.grid.snapshot()
        cs = [[c[0], c[1], 'visited'] for c in g['visited']]
        cs += [[d[0], d[1], 'dead_' + d[2]] for d in g['dead_ends']]
        if core.grid.origin is not None:
            cs.append([0, 0, 'start'])
        if g['current']:
            cs.append([g['current'][0], g['current'][1], 'current'])
        return {'cell_m': g['cell_m'], 'cells': cs}

    def dog():
        p = core.dog_pose_grid()
        if p is None:
            return []
        st = core.status(time.monotonic())
        return [{'x': round(p[0], 2), 'y': round(p[1], 2), 'yaw': round(p[2], 3),
                 'label': st['mode'] + (' ' + st['phase'] if st['phase'] else '')}]

    def path():
        if core.grid.origin is None:
            return []
        return [[round(v, 2) for v in core.grid.to_grid((q[0], q[1], 0.0))] for q in core.path[-600:]]

    def status():
        return core.status(time.monotonic())

    def stairs():
        s = core.stairs
        p = core.dog_pose_grid()
        if s.kind == 'none' or p is None or s.edge_x == float('inf'):
            return []
        import math
        a = p[2] - s.heading          # grid yaw is clockwise-positive, body heading is left-positive
        return [{'x': round(p[0] + s.edge_x * math.sin(a), 2), 'y': round(p[1] + s.edge_x * math.cos(a), 2),
                 'yaw': round(a, 3), 'label': s.kind}]

    r.layer('grid', 'cells', cells, {'visited': '#4c8bf5', 'start': '#2ca02c', 'current': '#ffb000',
                                     'dead': '#d62728'})
    r.layer('path', 'path', path, {'color': '#888888'})
    r.layer('dog', 'markers', dog, {'color': '#ff6a00'})
    r.layer('obstacles', 'points', core.obstacles_grid, {'color': '#d62728'}, default_on=False)
    r.layer('stairs', 'markers', stairs, {'color': '#9467bd'}, default_on=False)
    r.layer('status', 'text', status, default_on=False)
    return r
