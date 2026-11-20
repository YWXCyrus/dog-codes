#!/usr/bin/env python3
"""Checks the output of one simulated survey run. Usage: check_survey_output.py <run_root> <nodeinfo.txt>"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'dog_tools'))
import report  # noqa: E402

results = []


def check(name, ok, detail=''):
    results.append(ok)
    print('%s  %s %s' % ('PASS' if ok else 'FAIL', name, detail))


def main():
    root, info = sys.argv[1], sys.argv[2]
    run = open(os.path.join(root, 'LATEST')).read().strip()
    out = os.path.join(run, 'report')
    def big(p, n):
        return os.path.exists(p) and os.path.getsize(p) > n
    for fn in ('samples.csv', 'labels.csv', 'map.npz', 'params.json'):
        check('file ' + fn, big(os.path.join(run, fn), 20))
    for fn in ('summary.txt', 'summary.csv', 'cells.csv', 'map_rough.png', 'map_height.png'):
        check('report/' + fn, big(os.path.join(out, fn), 100))
    if not big(os.path.join(run, 'samples.csv'), 20):
        print('node produced no data: stopping')
        return 1
    samples = report.read_samples(os.path.join(run, 'samples.csv'))
    check('>= 120 samples (5 Hz, ~30 s)', len(samples) >= 120, '(%d)' % len(samples))
    labs = [s['label'] for s in samples]
    check('labels seen: flat_slow, general, unlabelled', {'flat_slow', 'general', 'unlabelled'} <= set(labs), str(sorted(set(labs))))
    check('label order unlabelled -> flat_slow -> general',
          labs.index('unlabelled') < labs.index('flat_slow') < labs.index('general'))
    xs = [s['x'] for s in samples]
    check('dog moved >= 3 m in +X', max(xs) - min(xs) >= 3.0, '(%.2f m)' % (max(xs) - min(xs)))
    check('speed column ~0.3 while moving', any(0.2 < s['speed'] < 0.4 for s in samples))
    check('rough detector hit at least once', any(s['rough_hit'] for s in samples))
    check('rough flag ON at least once', any(s['rough_on'] for s in samples))
    ev = report.read_events(os.path.join(run, 'labels.csv'))
    kinds = [e[1] for e in ev]
    check('labels.csv has gait x3, mark, note', kinds.count('gait') == 3 and 'mark' in kinds and 'note' in kinds, str(kinds))
    with open(os.path.join(out, 'cells.csv'), newline='') as f:
        cells = {(int(r['i']), int(r['j'])): r for r in csv.DictReader(f)}

    def val(c, key):
        try:
            return float(cells[c][key])
        except (KeyError, ValueError):
            return float('nan')
    # sim rubble: h = 0.08*(sin*sin+1)/2  -> std 0.02 m by construction (+0.008 m lidar noise); cell (1,0)
    # = x 0.51..1.53 (rubble), cell (2,0) = x 1.53..2.55 (flat, seen from behind by the pitched-down lidar).
    # The START cell (0,0) is never mapped: the lidar cannot see the floor behind the dog.
    flat, rub, rub_max = val((2, 0), 'rough_mean'), val((1, 0), 'rough_mean'), val((1, 0), 'rough_max')
    check('rubble cell (1,0): peak roughness ~0.02 m (analytic 0.020-0.022)', 0.017 <= rub_max <= 0.030,
          '(max %.4f)' % rub_max)
    check('rubble cell (1,0) mean >= 1.5x flat cell (2,0)', rub >= 1.5 * flat,
          '(rubble %.4f flat %.4f)' % (rub, flat))
    check('flat cell (2,0) is NOT rough (< 0.012)', flat < 0.012, '(%.4f)' % flat)
    h2, h4 = val((2, 0), 'mean_h'), val((4, 0), 'mean_h')
    check('flat cell (2,0) height ~0 above the start floor', abs(h2) < 0.05, '(%.3f)' % h2)
    check('stairs cell (4,0) 0.10-0.30 m above the flat floor (analytic mean 0.15)', 0.10 < h4 - h2 < 0.30,
          '(%.3f)' % (h4 - h2))
    check('start cell (0,0) not mapped (blind behind the dog): known limitation, mean_h is nan',
          (0, 0) not in cells or cells[(0, 0)]['mean_h'] in ('nan', ''))
    check('walls detected', any(float(c['wall_pct']) > 0 for c in cells.values()))
    txt = open(info).read()
    pub = txt.split('Publishers:')[1].split('Service')[0] if 'Publishers:' in txt else 'MISSING'
    check('ros2 node info: node has NO publishers', pub.replace('\n', '').strip() == '', repr(pub.strip()))
    s = open(os.path.join(out, 'summary.txt')).read()
    check('summary has flat_slow and general rows', 'flat_slow' in s and 'general' in s)
    print('\n%d/%d checks passed' % (sum(results), len(results)))
    print('run folder:', run)
    return 0 if all(results) else 1


if __name__ == '__main__':
    sys.exit(main())
