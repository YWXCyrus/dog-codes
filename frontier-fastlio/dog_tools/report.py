#!/usr/bin/env python3
"""dog_tools report: build summary tables + map images from a survey run folder (no ROS needed).

    python3 report.py <run_dir>            # e.g. ~/dog_tools_runs/survey_20261120_101500

Reads  samples.csv, labels.csv, map.npz   Writes  <run_dir>/report/{summary.txt, summary.csv,
cells.csv, map_rough.png, map_height.png}
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import core  # noqa: E402

NUM = ['t_wall', 't_rel', 'x', 'y', 'yaw_deg', 'speed', 'roll_odom', 'pitch_odom', 'roll_imu',
       'pitch_imu', 'gyro_dps', 'rough_std', 'rough_frac', 'near_n', 'rough_hit', 'rough_on']


def read_samples(path):
    out = []
    if not os.path.exists(path):
        return out
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            try:
                s = {k: float(row[k]) for k in NUM}
                s['label'] = row['label'] or core.UNLABELLED
            except (KeyError, ValueError, TypeError):
                continue                     # half-written last line
            out.append(s)
    return out


def read_events(path):
    out = []
    if not os.path.exists(path):
        return out
    with open(path, newline='') as f:
        for row in csv.reader(f):
            if len(row) < 3 or row[0] == 't_wall':
                continue
            try:
                out.append((float(row[0]), row[1], row[2]))
            except ValueError:
                continue
    return out


def load_map(path):
    if not os.path.exists(path):
        return None
    with np.load(path, allow_pickle=False) as d:
        return core.GridMap.from_arrays({k: d[k] for k in d.files if k != 'meta_json'})


def cell_rows(grid, samples):
    """One row per 1.02 m lattice cell that was seen or visited."""
    dt = core.sample_dt(samples)
    visits = {}
    for s in samples:
        if s['x'] != s['x']:
            continue
        key = (int(round(s['x'] / core.CELL)), int(round(s['y'] / core.CELL)))
        d = visits.setdefault(key, {})
        d[s['label']] = d.get(s['label'], 0.0) + dt
    rows = []
    if grid is None:
        return rows
    cx, cy = grid.centres()
    li = np.round(cx / core.CELL).astype(int)
    lj = np.round(cy / core.CELL).astype(int)
    n = grid.g(grid.n)
    s1 = grid.g(grid.s)
    hmax = grid.g(grid.hmax)
    wall = grid.wall_cells()
    obs = grid.observed()
    rough = grid.local_rough(r=2, min_n=12)
    keys = set(visits)
    ys, xs = np.where(obs)
    for a, b in zip(xs, ys):
        keys.add((int(li[a]), int(lj[b])))
    for (i, j) in sorted(keys):
        m = (lj[:, None] == j) & (li[None, :] == i)
        tot = int(m.sum())
        if tot == 0:
            continue
        o = obs & m
        fl = m & (n > 0) & ~wall
        nn = float(n[fl].sum())
        mean_h = float(s1[fl].sum() / nn) if nn > 0 else float('nan')
        r = rough[m & np.isfinite(rough)]
        gaits = ';'.join('%s:%.0fs' % (k, v) for k, v in sorted(visits.get((i, j), {}).items()))
        rows.append({
            'i': i, 'j': j, 'cx': round(i * core.CELL, 2), 'cy': round(j * core.CELL, 2),
            'seen_pct': round(100.0 * o.sum() / tot, 0),
            'wall_pct': round(100.0 * (wall & m).sum() / max(int(o.sum()), 1), 0),
            'mean_h': round(mean_h, 3) if mean_h == mean_h else float('nan'),
            'max_h': round(float(hmax[fl].max()), 3) if fl.any() else float('nan'),
            'rough_mean': round(float(r.mean()), 3) if r.size else float('nan'),
            'rough_max': round(float(r.max()), 3) if r.size else float('nan'),
            'gaits': gaits})
    return rows


CELL_COLS = ['i', 'j', 'cx', 'cy', 'seen_pct', 'wall_pct', 'mean_h', 'max_h', 'rough_mean', 'rough_max', 'gaits']


def _atomic_write(path, text):
    tmp = path + '.tmp'
    with open(tmp, 'w', newline='') as f:
        f.write(text)
    os.replace(tmp, path)


def build_report(run_dir, render=True):
    out_dir = os.path.join(run_dir, 'report')
    os.makedirs(out_dir, exist_ok=True)
    samples = read_samples(os.path.join(run_dir, 'samples.csv'))
    events = read_events(os.path.join(run_dir, 'labels.csv'))
    grid = load_map(os.path.join(run_dir, 'map.npz'))
    res = {'samples': len(samples), 'files': []}
    rows = core.summarize(samples, events)
    note = [
        'dog_tools survey report: %s' % os.path.basename(run_dir.rstrip('/\\')),
        'samples=%d  marks=%d  notes=%d  map scans=%s' % (
            len(samples), sum(1 for e in events if e[1] == 'mark'),
            sum(1 for e in events if e[1] == 'note'), grid.scans if grid else 0),
        '',
        'Columns: t0/t1 = seconds from the first sample; dur = segment length; moving = seconds with speed',
        '>= %.2f m/s; dist = path length (m); speed = mean speed while moving (m/s); tiltmx = largest' % core.MOVING_SPEED,
        'roll/pitch from /odom (deg); wobble = std of IMU roll+pitch while moving (deg, lower = steadier);',
        'gyro = rms roll/pitch rate (deg/s); rough = mean floor-height scatter 0.9-2.0 m ahead (m);',
        'hit%/on% = share of moving time the detector hit / the hysteresis flag was ON (dog threshold:',
        'std > 0.050 or frac > 0.35); flips = ON/OFF changes (many = flapping); marks = your "m" presses.',
        'kind=segment: one stretch under one gait label; kind=total: all stretches of that label together.',
        '',
    ]
    text = '\n'.join(note) + (core.format_summary(rows) if rows else '(no samples yet)') + '\n'
    notes = [e for e in events if e[1] == 'note']
    if notes:
        t_ref = samples[0]['t_wall'] if samples else notes[0][0]
        text += '\nNOTES\n' + '\n'.join('  +%6.1fs  %s' % (t - t_ref, txt) for t, _k, txt in notes) + '\n'
    _atomic_write(os.path.join(out_dir, 'summary.txt'), text)
    res['summary_text'] = text
    res['files'].append('summary.txt')
    with open(os.path.join(out_dir, 'summary.csv.tmp'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(core.SUMMARY_COLS)
        for r in rows:
            w.writerow([('%.3f' % r[c]) if isinstance(r[c], float) else r[c] for c in core.SUMMARY_COLS])
    os.replace(os.path.join(out_dir, 'summary.csv.tmp'), os.path.join(out_dir, 'summary.csv'))
    res['files'].append('summary.csv')
    cr = cell_rows(grid, samples)
    with open(os.path.join(out_dir, 'cells.csv.tmp'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(CELL_COLS)
        for r in cr:
            w.writerow([r[c] for c in CELL_COLS])
    os.replace(os.path.join(out_dir, 'cells.csv.tmp'), os.path.join(out_dir, 'cells.csv'))
    res['files'].append('cells.csv')
    if render and grid is not None:
        try:
            import render as R
            name = os.path.basename(run_dir.rstrip('/\\'))
            for kind in ('rough', 'height'):
                if R.render_map(grid, samples, events, kind, name,
                                os.path.join(out_dir, 'map_%s.png' % kind)):
                    res['files'].append('map_%s.png' % kind)
        except ImportError as e:
            res['render_error'] = 'OpenCV missing: %s' % e
    return res


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    run = os.path.expanduser(sys.argv[1])
    if not os.path.isdir(run):
        print('not a folder: %s' % run)
        return 2
    r = build_report(run)
    print(r['summary_text'])
    print('written to %s: %s' % (os.path.join(run, 'report'), ', '.join(r['files'])))
    if 'render_error' in r:
        print(r['render_error'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
