#!/usr/bin/env python3
"""dog_tools compare: compare survey sessions (one gait / one style each) terrain by terrain. No ROS needed.

    python3 compare.py general_slow=survey_20261120_101500 general_fast=survey_20261120_103000 ...
        [--cells i0:i1,j0:j1]            only use samples inside this lattice box
        [--override "1,0=rough 2,0=flat"] force the class of single cells (after checking the grid)
        [--flat-rough 0.03] [--raised-h 0.10] [--wall-pct 30] [--tilt-warn 20] [--out DIR]

NAME=FOLDER: NAME is your own tag (e.g. the gait and "slow"/"fast"); a bare FOLDER uses its folder name.
FOLDER is a run folder written by terrain_survey.py (contains samples.csv and map.npz).

How it works: the maps of all sessions are summed (same odom origin required), every 1.02 m lattice cell the dog
visited gets a terrain class from the combined map, and each session's samples are grouped by that class.
Classes: flat, rough (uneven/rubble), raised (stairs/ramp/platform), obstacle (mostly wall/tall), unmapped (the
lidar never saw that floor, e.g. the start cell). The thresholds are first guesses: read the cell table and the
grid that are printed, then correct with --override or the thresholds.
"""
import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import core  # noqa: E402
import report  # noqa: E402

CLASS_ORDER = ['flat', 'rough', 'raised', 'obstacle', 'unmapped']
CLASS_CHAR = {'flat': 'F', 'rough': 'R', 'raised': 'S', 'obstacle': 'W', 'unmapped': '?'}
METRICS = [('dur_s', 'time in terrain (s)', 1), ('speed_mean', 'speed while moving (m/s)', 2),
           ('wobble_deg', 'wobble (deg, lower = steadier)', 2), ('tilt_max_deg', 'max tilt (deg)', 1),
           ('gyro_rms_dps', 'gyro rms (deg/s)', 1)]


def lattice(s):
    return int(round(s['x'] / core.CELL)), int(round(s['y'] / core.CELL))


def parse_sessions(items):
    out = []
    for it in items:
        if '=' in it:
            name, path = it.split('=', 1)
        else:
            path = it
            name = os.path.basename(os.path.normpath(path))
        path = os.path.expanduser(path)
        out.append((name.strip(), path))
    names = [n for n, _ in out]
    if len(set(names)) != len(names):
        raise ValueError('session names must be unique: %s' % names)
    return out


def parse_box(text):
    """'i0:i1,j0:j1' -> (i0, i1, j0, j1) inclusive."""
    a, b = text.split(',')
    i0, i1 = [int(v) for v in a.split(':')]
    j0, j1 = [int(v) for v in b.split(':')]
    return min(i0, i1), max(i0, i1), min(j0, j1), max(j0, j1)


def parse_override(text):
    out = {}
    for tok in text.replace(';', ' ').split():
        k, v = tok.split('=')
        i, j = k.split(',')
        out[(int(i), int(j))] = v.strip()
    return out


def load_session(name, run):
    if not os.path.isdir(run):
        raise ValueError('not a folder: %s' % run)
    samples = report.read_samples(os.path.join(run, 'samples.csv'))
    if not samples:
        raise ValueError('no samples in %s' % run)
    return {'name': name, 'run': run, 'samples': samples,
            'grid': report.load_map(os.path.join(run, 'map.npz'))}


def classify(rows, visited, a):
    """class per visited lattice cell from the combined-map cell rows."""
    by = {(r['i'], r['j']): r for r in rows}
    out = {}
    for key in visited:
        r = by.get(key)
        if (r is None or r['mean_h'] != r['mean_h'] or r['rough_mean'] != r['rough_mean']
                or r['seen_pct'] < a.min_seen):
            out[key] = 'unmapped'
        elif r['wall_pct'] >= a.wall_pct:
            out[key] = 'obstacle'
        elif abs(r['mean_h']) >= a.raised_h:
            out[key] = 'raised'
        elif r['rough_mean'] >= a.flat_rough:
            out[key] = 'rough'
        else:
            out[key] = 'flat'
    out.update({k: v for k, v in a.override.items()})
    return out


def session_rows(sess, classes, box):
    samples = sess['samples']
    if box:
        samples = [s for s in samples if box[0] <= lattice(s)[0] <= box[1] and box[2] <= lattice(s)[1] <= box[3]]
    if not samples:
        return {}, 0
    dt = core.sample_dt(sess['samples'])
    lab = []
    for s in samples:
        q = dict(s)
        q['label'] = classes.get(lattice(s), 'unmapped')
        lab.append(q)
    runs = core.runs_by_label(lab)
    by = {}
    for r in runs:
        by.setdefault(r[0]['label'], []).append(r)
    out = {}
    for cls, rs in by.items():
        st = core.run_stats(rs, dt)
        st['cells'] = len({lattice(s) for r in rs for s in r})
        out[cls] = st
    st = core.run_stats([lab], dt)
    st['cells'] = len({lattice(s) for s in lab})
    out['ALL'] = st
    return out, dt


def fmt(v, p):
    return '-' if (isinstance(v, float) and v != v) else ('%.*f' % (p, v))


def matrix(title, key, p, sessions, stats, classes_present, tilt_warn=None):
    w = max(12, max(len(s['name']) for s in sessions) + 1)
    lines = ['', title, '%-10s' % 'terrain' + ''.join(s['name'][:w - 1].rjust(w) for s in sessions)]
    for cls in classes_present + ['ALL']:
        cells = []
        for s in sessions:
            st = stats[s['name']].get(cls)
            if st is None:
                cells.append('.'.rjust(w))
                continue
            txt = fmt(st[key], p)
            if tilt_warn is not None and key == 'tilt_max_deg' and st[key] == st[key] and st[key] >= tilt_warn:
                txt += '!'
            cells.append(txt.rjust(w))
        lines.append('%-10s' % cls + ''.join(cells))
    return lines


def ascii_grid(classes):
    if not classes:
        return ['(no cells visited)']
    iv = [k[0] for k in classes]
    jv = [k[1] for k in classes]
    i0, i1, j0, j1 = min(iv), max(iv), min(jv), max(jv)
    lines = ['        ' + ''.join(('%d' % i).rjust(4) for i in range(i0, i1 + 1))]
    for j in range(j1, j0 - 1, -1):
        row = '  j=%-4d' % j
        for i in range(i0, i1 + 1):
            c = classes.get((i, j))
            row += (CLASS_CHAR.get(c, '*') if c else '.').rjust(4)
        lines.append(row)
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('sessions', nargs='+')
    ap.add_argument('--cells', default=None)
    ap.add_argument('--override', default='')
    ap.add_argument('--flat-rough', dest='flat_rough', type=float, default=0.03)
    ap.add_argument('--raised-h', dest='raised_h', type=float, default=0.10)
    ap.add_argument('--wall-pct', dest='wall_pct', type=float, default=30.0)
    ap.add_argument('--min-seen', dest='min_seen', type=float, default=10.0)
    ap.add_argument('--tilt-warn', dest='tilt_warn', type=float, default=20.0)
    ap.add_argument('--out', default=None)
    a = ap.parse_args(argv)
    a.override = parse_override(a.override)
    box = parse_box(a.cells) if a.cells else None

    pairs = parse_sessions(a.sessions)
    sessions = [load_session(n, p) for n, p in pairs]
    out = []
    grids = [s['grid'] for s in sessions if s['grid'] is not None]
    for s in sessions:
        if s['grid'] is None:
            out.append('WARNING: %s has no map.npz - its cells are classified from the other sessions only.' % s['name'])
    if not grids:
        raise ValueError('none of the sessions has a map.npz')
    merged = core.merge_grids(grids)
    # consistency of the sessions' maps (same origin / same start pose?)
    ref = next(s for s in sessions if s['grid'] is not None)
    for s in sessions:
        if s is ref or s['grid'] is None:
            continue
        iou = ref['grid'].wall_iou(s['grid'])
        if iou == iou and iou < 0.5:
            out.append('WARNING: walls of %s overlap only %.0f%% with %s - the start pose / odom origin probably '
                       'differed, so cells are NOT comparable between these sessions.' % (s['name'], 100 * iou, ref['name']))
        out.append('map consistency %s vs %s: wall overlap %.0f%%' % (s['name'], ref['name'], 100 * iou)
                   if iou == iou else 'map consistency %s vs %s: no walls seen' % (s['name'], ref['name']))

    all_samples = [x for s in sessions for x in s['samples']]
    if box:
        all_samples = [x for x in all_samples if box[0] <= lattice(x)[0] <= box[1] and box[2] <= lattice(x)[1] <= box[3]]
    visited = {lattice(x) for x in all_samples}
    rows = report.cell_rows(merged, all_samples)
    classes = classify(rows, visited, a)

    stats = {}
    for s in sessions:
        stats[s['name']], _dt = session_rows(s, classes, box)
    present = [c for c in CLASS_ORDER if any(c in stats[s['name']] for s in sessions)]
    # names given with --override (e.g. "cubeA") are classes too
    present += sorted({c for s in sessions for c in stats[s['name']] if c not in CLASS_ORDER and c != 'ALL'})

    head = ['dog_tools compare   %s' % time.strftime('%Y-%m-%d %H:%M'),
            'sessions: ' + ', '.join('%s=%s' % (s['name'], os.path.basename(os.path.normpath(s['run']))) for s in sessions),
            'region: %s | thresholds: flat if rough < %.3f m, raised if |height| >= %.2f m, obstacle if wall >= %.0f%%, '
            'unmapped if seen < %.0f%%' % ('cells %s' % a.cells if a.cells else 'all visited cells', a.flat_rough,
                                           a.raised_h, a.wall_pct, a.min_seen)]
    text = head + [''] + out
    text += ['', 'TERRAIN GRID (cells the dogs visited; i across, j down; F flat, R rough, S raised, W obstacle, '
                 '? unmapped, . not visited)'] + ascii_grid(classes)
    by = {(r['i'], r['j']): r for r in rows}
    text += ['', 'CELL TABLE (combined map)', '   i    j   class      height  rough  wall%  seen%']
    for key in sorted(classes):
        r = by.get(key)
        text.append('%4d %4d   %-9s %7s %6s %6s %6s' % (
            key[0], key[1], classes[key] + ('*' if key in a.override else ''),
            fmt(r['mean_h'], 3) if r else '-', fmt(r['rough_mean'], 3) if r else '-',
            fmt(r['wall_pct'], 0) if r else '-', fmt(r['seen_pct'], 0) if r else '-'))
    if a.override:
        text.append('(* = class forced with --override)')
    for key, title, p in METRICS:
        text += matrix(title, key, p, sessions, stats, present, a.tilt_warn if key == 'tilt_max_deg' else None)
    text += ['',
             'How to read: one row per terrain class, one column per session. Compare sessions of the same terrain.',
             'time/speed = efficiency (slow vs fast runs); wobble/tilt = stability (a "!" marks max tilt >= %.0f deg).'
             % a.tilt_warn,
             'Only moving samples (speed >= %.2f m/s) count for speed, wobble and gyro; "-" = no moving samples there,'
             % core.MOVING_SPEED,
             '"." = the session never entered that terrain. A session that fell or was restarted mid-run is not '
             'comparable: tell me.']
    txt = '\n'.join(text) + '\n'
    out_dir = a.out or os.path.join(os.path.dirname(os.path.normpath(sessions[0]['run'])),
                                    'compare_' + time.strftime('%Y%m%d_%H%M%S'))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'compare.txt'), 'w', newline='') as f:
        f.write(txt)
    cols = ['session', 'terrain', 'cells', 'dur_s', 'moving_s', 'dist_m', 'speed_mean', 'tilt_max_deg',
            'wobble_deg', 'gyro_rms_dps', 'rough_flips']
    with open(os.path.join(out_dir, 'compare.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(cols)
        for s in sessions:
            for cls in present + ['ALL']:
                st = stats[s['name']].get(cls)
                if st:
                    w.writerow([s['name'], cls] + [('%.3f' % st[c]) if isinstance(st[c], float) else st[c]
                                                   for c in cols[2:]])
    with open(os.path.join(out_dir, 'cells_classified.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['i', 'j', 'class', 'mean_h', 'rough_mean', 'wall_pct', 'seen_pct'])
        for key in sorted(classes):
            r = by.get(key, {})
            w.writerow([key[0], key[1], classes[key], r.get('mean_h', ''), r.get('rough_mean', ''),
                        r.get('wall_pct', ''), r.get('seen_pct', '')])
    print(txt)
    print('written to %s' % out_dir)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except ValueError as e:
        print('compare: %s' % e)
        sys.exit(2)
