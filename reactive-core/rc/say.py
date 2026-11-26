"""One plain line for everything the dog does (terminal, '>> ...'), from the events of decisions.jsonl.

Positions are arena CELLS only: (i, j), +i = the start heading, +j = left (booklet frame). Every event gets a
line: the known kinds in words, anything else as 'key value'. HOLD / FINISHED keep run.py's own banners.
"""


def _c(v):
    """(i, j) from a list / tuple, '' if missing."""
    try:
        return '(%d,%d)' % (int(v[0]), int(v[1]))
    except (TypeError, ValueError, IndexError):
        return ''


def _edge(k):
    """Border key [i, j, 'x'|'y'] -> '(i,j)-(i+1,j)' / '(i,j)-(i,j+1)'."""
    try:
        i, j, ax = int(k[0]), int(k[1]), k[2]
        return '(%d,%d)-(%d,%d)' % ((i, j, i + 1, j) if ax == 'x' else (i, j, i, j + 1))
    except (TypeError, ValueError, IndexError):
        return str(k)


STAIRS = {'APPROACH': 'stairs: walking up to them', 'ENTER': 'entering ai mode', 'ALIGN': 'ai mode activated',
          'CLIMB': 'climbing the stairs', 'DESCEND': 'going down the stairs', 'EXIT': 'exiting ai mode',
          'DONE': 'ai mode exited, walking again', 'ABORT': 'ai mode failed: back to walking',
          'LOOK': 'stairs? stopping to look again', 'LOOK_none': 'not stairs after all'}
SKIP = {'t', 'cell', 'wall_side', 'vlm_answer', 'hold', 'finish'}       # hold / finish: run.py banners


class Say:

    def __init__(self):
        self.scenes = set()              # (cell, label, survivor) already said
        self.last = (None, -1e9)

    def lines(self, e):
        out = self._lines(e)
        t = e.get('t', 0.0)
        res = []
        for ln in out:
            if ln and not (ln == self.last[0] and t - self.last[1] < 2.0):
                res.append(ln)
                self.last = (ln, t)
        return res

    def _lines(self, e):
        if 'arena_cell' in e:
            return ['cell %s' % _c(e['arena_cell'])]
        if 'plan' in e:
            g = _c(e.get('goal'))
            if not g:
                return ['no plan']
            return [{'explore': 'going to %s', 'final': 'going to the final cell %s',
                     'curtain': 'going to %s (through the curtain)', 'probe': 'checking a wall near %s'}
                    .get(e['plan'], e['plan'] + ' %s') % g]
        if 'scene_done' in e:
            return ['%s looked at from the doorway, not entered' % _c(e['scene_done'])]
        if 'scene' in e and 'kind' in e:
            if e['kind'] == 'dead_end':
                return ['dead end %s: not entering' % _c(e['scene'])]
            return ['scene detected %s: looking from the doorway' % _c(e['scene'])]
        if 'scene_state' in e:
            label = e.get('scene') or ''
            key = (tuple(e['scene_state']), label, bool(e.get('survivor')))
            if not label or key in self.scenes:
                return []
            self.scenes.add(key)
            return ['%s, %s%s' % (label, _c(e['scene_state']), ', survivor' if e.get('survivor') else '')]
        if 'stairs_cell' in e:
            return ['stairs confirm %s (%s)' % (_c(e['stairs_cell']), e.get('src', 'map'))]
        if 'stairs_answer' in e:
            if e['stairs_answer'] == 'yes':
                return ['stairs confirm %s (camera)' % _c(e.get('cell'))]
            return ['not stairs %s (camera)' % _c(e.get('cell'))]
        if 'stairs_seen' in e:
            if e.get('confident'):
                return ['stairs detected ahead (lidar, sure)']
            return ['stairs detected ahead (%s %s)' % (e.get('src', 'lidar'), e['stairs_seen'])]
        if 'stairs' in e:
            return [STAIRS.get(e['stairs'], 'stairs: %s' % e['stairs'])]
        if 'stairs_fail' in e:
            return ['ai mode failed: %s' % e['stairs_fail']]
        if 'stairs_grid' in e:
            return ['stairs: lidar not sure, walking up on the grid']
        if 'ask' in e:
            return ['asking the camera: %s?' % e['ask']]
        if 'curtain_answer' in e:
            return ['camera: curtain %s %s' % (e['curtain_answer'], _edge(e.get('edge')))]
        if 'curtain_cue' in e:
            return ['curtain detected (camera) %s' % _edge(e['curtain_cue'])]
        if 'curtain' in e:
            what = str(e['curtain'])
            word = {'push': 'pushing through the curtain', 'passed': 'curtain passed',
                    'timeout': 'curtain push timed out'}.get(what, 'curtain: ' + what)
            return ['%s %s' % (word, _edge(e.get('edge')))]
        if 'grid_block' in e:
            return ['blocked: %s' % e['grid_block']]
        if 'grid_fail' in e:
            return ['problem: %s' % e['grid_fail']]
        if 'grid' in e:
            return [str(e['grid'])]
        if 'report' in e:
            what = ', '.join(v for v in (e.get('label'), e.get('scene'), _c(e.get('cell'))) if v)
            if e.get('message'):
                what = '"%s"' % e['message']
            return ['%s reported: %s' % (e['report'], what or e.get('type', ''))]
        if 'qr_decoded' in e:
            return ['QR decoded: "%s"' % e['qr_decoded']]
        if 'perception' in e:
            word = {'survivor_seen': 'survivor seen', 'qr_seen': 'QR seen (not readable yet)',
                    'stairs_seen': 'camera: stairs seen', 'hazard_seen': 'hazard sign seen'}.get(e['perception'],
                                                                                           e['perception'])
            return [('%s %s' % (word, _c(e.get('cell')))).strip()]
        if 'arena' in e:
            return ['map: %s' % e['arena']]
        if 'resume' in e:
            return ['resumed']
        keys = [k for k in e if k not in SKIP]
        if not keys:
            return []
        return [' '.join('%s %s' % (k, e[k]) for k in keys)[:140]]
