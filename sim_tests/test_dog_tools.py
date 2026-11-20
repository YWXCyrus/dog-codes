#!/usr/bin/env python3
"""Unit tests for codes/dog_tools (pure logic, no robot). Run in WSL:
   source /opt/ros/humble/setup.bash; python3 -m unittest -v test_dog_tools   (from codes/sim_tests)
"""
import csv
import math
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(HERE, '..', 'dog_tools')
sys.path.insert(0, TOOLS)
sys.path.insert(0, os.path.join(HERE, '..', 'dog_codes'))

import core  # noqa: E402
import gait_label  # noqa: E402
import report  # noqa: E402


def make_scene(rng, bump_std=0.06):
    """flat floor x in [-0.5,5], |y|<0.6; bumpy floor for 1<=x<2; a wall at y=0.65 (h up to 1.2)."""
    xs = rng.uniform(-0.5, 5.0, 60000)
    ys = rng.uniform(-0.6, 0.6, 60000)
    h = rng.normal(0, 0.004, 60000)
    bump = (xs >= 1.0) & (xs < 2.0)
    h[bump] = rng.normal(0.03, bump_std, bump.sum())
    wy = np.full(8000, 0.65)
    wx = rng.uniform(-0.5, 5.0, 8000)
    wh = rng.uniform(-0.2, 1.2, 8000)
    return np.column_stack([np.r_[xs, wx], np.r_[ys, wy], np.r_[h, wh]]).astype(np.float32)


class TestGrid(unittest.TestCase):
    def test_flat_vs_bumpy_vs_wall(self):
        g = core.GridMap()
        g.add(make_scene(np.random.default_rng(1)))
        r = g.local_rough()
        cx, cy = g.centres()

        def at(x, y):
            return r[int((y - g.y0) / g.res), int((x - g.x0) / g.res)]
        self.assertLess(at(3.5, 0.0), 0.02)
        self.assertGreater(at(1.5, 0.0), 0.04)
        self.assertTrue(g.wall_cells()[int((0.65 - g.y0) / g.res), int((2.5 - g.x0) / g.res)])
        self.assertTrue(np.isnan(at(8.0, 5.0)))                     # never seen
        b = g.bounds()
        self.assertLess(b[0], -0.3)
        self.assertGreater(b[1], 4.8)

    def test_roundtrip(self):
        g = core.GridMap()
        g.add(make_scene(np.random.default_rng(2)))
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, 'm.npz')
            with open(p, 'wb') as f:
                np.savez_compressed(f, **g.to_arrays(), meta_json=np.array('{}'))
            with np.load(p, allow_pickle=False) as z:
                g2 = core.GridMap.from_arrays({k: z[k] for k in z.files if k != 'meta_json'})
        self.assertEqual((g.nx, g.ny, g.scans), (g2.nx, g2.ny, g2.scans))
        np.testing.assert_allclose(np.nan_to_num(g.local_rough()), np.nan_to_num(g2.local_rough()))

    def test_out_of_bounds_and_empty(self):
        g = core.GridMap()
        g.add(np.zeros((0, 3), np.float32))
        g.add(np.array([[100, 100, 0.0], [-100, 0, 0.0]], np.float32))
        self.assertIsNone(g.bounds())

    def test_box_sum(self):
        a = np.arange(20, dtype=float).reshape(4, 5)
        b = core.box_sum(a, 1)
        self.assertAlmostEqual(b[1, 1], a[0:3, 0:3].sum())
        self.assertAlmostEqual(b[0, 0], a[0:2, 0:2].sum())
        self.assertAlmostEqual(b[3, 4], a[2:4, 3:5].sum())


class TestDetector(unittest.TestCase):
    def test_hysteresis(self):
        d = core.RoughDetector()
        hit = (0.08, 0.5, 100)
        flips = [d.update(hit)[2] for _ in range(3)]
        self.assertEqual(flips, [None] * 3)
        self.assertEqual(d.update(hit)[2], 'on')
        for _ in range(14):
            self.assertIsNone(d.update((0.01, 0.0, 100))[2])
        self.assertTrue(d.on)
        self.assertEqual(d.update(None)[2], 'off')
        self.assertFalse(d.on)

    def test_flicker_never_on(self):
        d = core.RoughDetector()
        for i in range(60):
            d.update((0.08, 0.5, 9) if i % 2 == 0 else (0.0, 0.0, 9))
        self.assertFalse(d.on)

    def test_parity_with_terrain_py(self):
        try:
            import terrain
        except Exception as e:                                       # rclpy missing
            self.skipTest('terrain.py not importable: %s' % e)
        t = object.__new__(terrain.Terrain)
        t.p = {'corr_x0': 0.25, 'corr_x1': 2.6, 'corr_half_w': 0.40, 'bin_m': 0.06,
               'trend_min_rise': 0.10, 'trend_min_incr': 2, 'trend_max_drop': 0.045,
               'blind_bias_m': 0.35, 'trend_slope_min': 0.10, 'trend_slope_max': 1.2, 'detect_down': True}
        rng = np.random.default_rng(5)
        n_cmp = 0
        for k in range(40):
            yaw = rng.uniform(-math.pi, math.pi)
            px, py = rng.uniform(-1, 1, 2)
            t.px, t.py, t.yaw = px, py, yaw
            std = rng.choice([0.004, 0.03, 0.08])
            m = int(rng.integers(0, 4000))
            lx = rng.uniform(-0.5, 3.5, m)
            ly = rng.uniform(-1.2, 1.2, m)
            h = rng.normal(0.02, std, m)
            x = px + math.cos(yaw) * lx - math.sin(yaw) * ly
            y = py + math.sin(yaw) * lx + math.cos(yaw) * ly
            pts = np.column_stack([x, y, h]).astype(np.float32)
            a = t.analyse(pts)['rough']
            b = core.rough_features(pts, px, py, yaw)
            if a is None or b is None:
                self.assertEqual(a is None, b is None)
                continue
            n_cmp += 1
            self.assertAlmostEqual(a[0], b[0], places=6)
            self.assertAlmostEqual(a[1], b[1], places=6)
        self.assertGreater(n_cmp, 10)


class TestMisc(unittest.TestCase):
    def test_tilt(self):
        r, p = core.tilt_from_accel(0, 0, 9.8)
        self.assertAlmostEqual(r, 0)
        self.assertAlmostEqual(p, 0)
        r, p = core.tilt_from_accel(-9.8 * math.sin(0.3), 0, 9.8 * math.cos(0.3))
        self.assertAlmostEqual(p, 0.3, places=5)
        self.assertIsNone(core.tilt_from_accel(0, 0, 0))

    def test_cloud_to_xyz(self):
        pts = np.array([[1, 2, 3, 9], [np.nan, 0, 0, 1], [4, 5, 6, 9]], np.float32)
        msg = SimpleNamespace(width=3, height=1, point_step=16, is_bigendian=False, data=pts.tobytes(),
                              fields=[SimpleNamespace(name=n, offset=4 * i, datatype=7) for i, n in enumerate('xyz')])
        out = core.cloud_to_xyz(msg)
        self.assertEqual(out.shape, (2, 3))
        np.testing.assert_allclose(out[1], [4, 5, 6])

    def test_body_h_calibration(self):
        rng = np.random.default_rng(3)
        n = 3000
        r = rng.uniform(0.6, 2.4, n)
        a = rng.uniform(0, 2 * math.pi, n)
        w = np.column_stack([r * np.cos(a), r * np.sin(a), rng.normal(0, 0.005, n)])
        c = core.BodyHeightCalib()
        for _ in range(8):
            c.update(w, 0.0, 0.0, 0.31)
        self.assertTrue(c.calibrated)
        self.assertAlmostEqual(c.value, 0.31, delta=0.02)
        self.assertTrue(core.BodyHeightCalib(fixed=0.28).calibrated)

    def test_read_new_rows(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, 'labels.csv')
            with open(p, 'w', newline='') as f:
                f.write('t_wall,kind,text\n1.0,gait,grip\n2.0,note,"café, ok"\n3.0,mar')   # last line half-written
            rows, off = core.read_new_rows(p, 0)
            self.assertEqual(rows, [['1.0', 'gait', 'grip'], ['2.0', 'note', 'café, ok']])
            rows2, off2 = core.read_new_rows(p, off)
            self.assertEqual((rows2, off2), ([], off))                         # nothing new yet
            with open(p, 'a', newline='') as f:
                f.write('k,general\n4.0,gait,hstep\n')
            rows3, off3 = core.read_new_rows(p, off)
            self.assertEqual(rows3, [['3.0', 'mark', 'general'], ['4.0', 'gait', 'hstep']])
            self.assertEqual(core.read_new_rows(p, off3)[0], [])

    def test_gait_label_parse(self):
        P = gait_label.parse
        self.assertEqual(P('3', 'x'), ('gait', 'flat_fast'))
        self.assertEqual(P('GRIP', 'x'), ('gait', 'grip'))
        self.assertEqual(P('+rugged high step', 'x'), ('gait', 'rugged_high_step'))
        self.assertEqual(P('-', 'x'), ('gait', 'unlabelled'))
        self.assertEqual(P('m', 'general'), ('mark', 'general'))
        self.assertEqual(P('n slippery, near wall', 'x'), ('note', 'slippery, near wall'))
        self.assertEqual(P('q', 'x')[0], 'quit')
        self.assertIsNone(P('   ', 'x'))
        self.assertEqual(P('9', 'x')[0], 'info')
        self.assertEqual(P('flatslow', 'x')[0], 'info')
        self.assertEqual(P('+bad/name', 'x')[0], 'info')
        self.assertEqual(P('n', 'x')[0], 'info')

    def test_gait_label_files(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertIsNone(gait_label.find_run(root))
            run = os.path.join(root, 'survey_x')
            os.makedirs(run)
            with open(os.path.join(root, 'LATEST'), 'w') as f:
                f.write(run + '\n')
            self.assertEqual(gait_label.find_run(root), run)
            p = os.path.join(run, 'labels.csv')
            gait_label.append(p, 'note', 'a, "quoted" note')
            with open(p, newline='') as f:
                row = list(csv.reader(f))[0]
            self.assertEqual(row[1:], ['note', 'a, "quoted" note'])


def sample(t, x, y, label, speed=0.3, roll=0.0, pitch=0.0, on=0, hit=0, rstd=0.01, yaw=0.0):
    return {'t_wall': t, 't_rel': t, 'x': x, 'y': y, 'yaw_deg': yaw, 'speed': speed, 'roll_odom': roll,
            'pitch_odom': pitch, 'roll_imu': roll, 'pitch_imu': pitch, 'gyro_dps': 5.0, 'rough_std': rstd,
            'rough_frac': 0.0, 'near_n': 100, 'rough_hit': hit, 'rough_on': on, 'label': label}


class TestSummary(unittest.TestCase):
    def build(self):
        s = []
        t = 1000.0
        x = 0.0
        for i in range(50):                       # 10 s flat_slow, moving
            s.append(sample(t, x, 0, 'flat_slow'))
            t += 0.2
            x += 0.06
        for i in range(50):                       # 10 s general, flapping flag, tilting
            s.append(sample(t, x, 0, 'general', roll=(i % 5) - 2.0, pitch=10.0 if i == 20 else 0.0,
                            on=(i // 5) % 2, hit=1, rstd=0.08))
            t += 0.2
            x += 0.06
        for i in range(10):                       # back to flat_slow, stopped
            s.append(sample(t, x, 0, 'flat_slow', speed=0.0))
            t += 0.2
        return s, t

    def test_rows(self):
        s, _ = self.build()
        ev = [(1000.0 + 12.0, 'mark', 'general'), (1000.0 + 1.0, 'note', 'n')]
        rows = core.summarize(s, ev)
        seg = [r for r in rows if r['kind'] == 'segment']
        tot = {r['label']: r for r in rows if r['kind'] == 'total'}
        self.assertEqual([r['label'] for r in seg], ['flat_slow', 'general', 'flat_slow'])
        self.assertEqual(seg[1]['marks'], 1)
        self.assertEqual(seg[0]['marks'], 0)
        self.assertEqual(seg[1]['rough_flips'], 9)
        self.assertAlmostEqual(seg[0]['dist_m'], 49 * 0.06, places=2)
        self.assertAlmostEqual(seg[1]['tilt_max_deg'], 10.0, places=3)
        self.assertGreater(seg[1]['wobble_deg'], 0.5)
        self.assertAlmostEqual(seg[1]['rough_std_mean'], 0.08, places=6)
        self.assertEqual(seg[1]['rough_hit_pct'], 100.0)
        self.assertEqual(tot['flat_slow']['seg'], 2)
        self.assertAlmostEqual(tot['flat_slow']['moving_s'], 10.0, places=3)     # stopped part excluded
        self.assertAlmostEqual(seg[2]['speed_mean'] if seg[2]['speed_mean'] == seg[2]['speed_mean'] else -1, -1)
        text = core.format_summary(rows)
        self.assertIn('general', text)

    def test_empty_and_single(self):
        self.assertEqual(core.summarize([]), [])
        rows = core.summarize([sample(1.0, 0, 0, 'grip')])
        self.assertEqual(len(rows), 2)
        core.format_summary(rows)

    def test_jump_ignored(self):
        s = [sample(0, 0, 0, 'a'), sample(0.2, 0.05, 0, 'a'), sample(0.4, 5.0, 0, 'a')]
        self.assertAlmostEqual(core.path_length(s), 0.05)

    def test_standing_jitter_not_counted(self):
        # real session 1: 926 standing samples with ~4 mm odometry jitter added metres to "dist"
        rng = np.random.default_rng(0)
        s = [sample(i * 0.2, rng.normal(0, 0.004), rng.normal(0, 0.004), 'a', speed=0.01) for i in range(1000)]
        self.assertEqual(core.path_length(s), 0.0)
        s += [sample(200 + i * 0.2, 0.06 * (i + 1), 0.0, 'a', speed=0.3) for i in range(10)]
        self.assertAlmostEqual(core.path_length(s[999:]), 0.6, places=2)


class TestReport(unittest.TestCase):
    def make_run(self, d, with_map=True):
        s, t_end = TestSummary().build()
        # place the path inside the bumpy zone (x 1..2) for the general segment
        with open(os.path.join(d, 'samples.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(core.SAMPLE_COLS)
            for q in s:
                w.writerow([('%.3f' % q[c]) if c != 'label' else q[c] for c in core.SAMPLE_COLS])
            f.write('1000.5,0.5,1.0')                                  # half-written last line
        with open(os.path.join(d, 'labels.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['t_wall', 'kind', 'text'])
            w.writerow(['1012.0', 'mark', 'general'])
            w.writerow(['1005.0', 'note', 'start of rubble'])
        if with_map:
            g = core.GridMap()
            g.add(make_scene(np.random.default_rng(4)))
            with open(os.path.join(d, 'map.npz'), 'wb') as f:
                np.savez_compressed(f, **g.to_arrays(), meta_json=np.array('{}'))

    def test_build(self):
        with tempfile.TemporaryDirectory() as d:
            self.make_run(d)
            r = report.build_report(d)
            out = os.path.join(d, 'report')
            for fn in ('summary.txt', 'summary.csv', 'cells.csv', 'map_rough.png', 'map_height.png'):
                p = os.path.join(out, fn)
                self.assertTrue(os.path.exists(p) and os.path.getsize(p) > 100, fn)
            self.assertEqual(r['samples'], 110)                        # partial line skipped
            self.assertIn('start of rubble', r['summary_text'])
            with open(os.path.join(out, 'cells.csv'), newline='') as f:
                cells = list(csv.DictReader(f))
            by = {(int(c['i']), int(c['j'])): c for c in cells}
            self.assertGreater(float(by[(1, 0)]['rough_mean']), 0.03)    # bumpy cell x 0.51..1.53
            self.assertLess(float(by[(3, 0)]['rough_mean']), 0.02)       # flat cell
            import cv2
            im = cv2.imread(os.path.join(out, 'map_rough.png'))
            self.assertIsNotNone(im)
            self.assertGreater(im.shape[0], 200)
            self.assertGreater(im.shape[1], 400)

    def test_without_map_and_empty(self):
        with tempfile.TemporaryDirectory() as d:
            self.make_run(d, with_map=False)
            r = report.build_report(d)
            self.assertIn('summary.txt', r['files'])
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, 'samples.csv'), 'w').write(','.join(core.SAMPLE_COLS) + '\n')
            r = report.build_report(d)
            self.assertIn('no samples', r['summary_text'])


def make_world(rng, wall_y=0.65):
    """flat floor, rough zone x 0.55..2.5 (cells 1,2), raised zone x>=3.6 (cell 4, +0.25 m); wall at wall_y."""
    n = 90000
    xs = rng.uniform(-0.5, 5.0, n)
    ys = rng.uniform(-0.45, 0.45, n)
    h = rng.normal(0, 0.004, n)
    rough = (xs >= 0.55) & (xs < 2.5)
    h[rough] = rng.normal(0.02, 0.06, rough.sum())
    raised = xs >= 3.6
    h[raised] = 0.25 + rng.normal(0, 0.004, raised.sum())
    wx = rng.uniform(-0.5, 5.0, 9000)
    return np.column_stack([np.r_[xs, wx], np.r_[ys, np.full(9000, wall_y)],
                            np.r_[h, rng.uniform(-0.2, 1.2, 9000)]]).astype(np.float32)


def write_session(d, speed_by_cell, wobble_by_cell, wall_y=0.65, with_map=True, peak_tilt=None):
    """Drive y=0 from x=0 to 4.6 at 5 Hz; speed and wobble depend on the lattice cell."""
    rng = np.random.default_rng(11)
    t = 5000.0
    x = 0.0
    rows = []
    while x < 4.6:
        cell = int(round(x / core.CELL))
        v = speed_by_cell.get(cell, 0.3)
        a = wobble_by_cell.get(cell, 1.0)
        sgn = 1 if len(rows) % 2 == 0 else -1
        s = sample(t, x, 0.0, 'unlabelled', speed=v, roll=sgn * a)
        s['roll_odom'] = sgn * a if not (peak_tilt and cell == 1 and len(rows) % 9 == 0) else peak_tilt
        rows.append(s)
        t += 0.2
        x += v * 0.2
    with open(os.path.join(d, 'samples.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(core.SAMPLE_COLS)
        for q in rows:
            w.writerow([('%.3f' % q[c]) if c != 'label' else q[c] for c in core.SAMPLE_COLS])
    if with_map:
        g = core.GridMap()
        g.add(make_world(rng, wall_y))
        with open(os.path.join(d, 'map.npz'), 'wb') as f:
            np.savez_compressed(f, **g.to_arrays(), meta_json=np.array('{}'))
    return rows


class TestCompare(unittest.TestCase):
    def run_compare(self, argv):
        import contextlib
        import io
        import compare
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = compare.main(argv)
        return rc, buf.getvalue()

    def test_three_sessions(self):
        with tempfile.TemporaryDirectory() as root:
            dirs = []
            # slow: rough cells (1,2) 0.15 m/s wobble 3; fast: 0.45 m/s wobble 7; third falls (tilt 31)
            for name, rough_v, rough_w, peak in (('a', 0.15, 3.0, None), ('b', 0.45, 7.0, None), ('c', 0.30, 4.0, 31.0)):
                d = os.path.join(root, 'survey_' + name)
                os.makedirs(d)
                write_session(d, {1: rough_v, 2: rough_v}, {1: rough_w, 2: rough_w}, peak_tilt=peak)
                dirs.append(d)
            out = os.path.join(root, 'cmp')
            rc, txt = self.run_compare(['general_slow=' + dirs[0], 'general_fast=' + dirs[1], 'hstep=' + dirs[2],
                                        '--out', out])
            self.assertEqual(rc, 0)
            for fn in ('compare.txt', 'compare.csv', 'cells_classified.csv'):
                self.assertTrue(os.path.getsize(os.path.join(out, fn)) > 50, fn)
            with open(os.path.join(out, 'cells_classified.csv'), newline='') as f:
                cls = {(int(r['i']), int(r['j'])): r['class'] for r in csv.DictReader(f)}
            self.assertEqual(cls[(3, 0)], 'flat')
            self.assertEqual(cls[(1, 0)], 'rough')
            self.assertEqual(cls[(2, 0)], 'rough')
            self.assertEqual(cls[(4, 0)], 'raised')
            with open(os.path.join(out, 'compare.csv'), newline='') as f:
                rows = {(r['session'], r['terrain']): r for r in csv.DictReader(f)}
            self.assertAlmostEqual(float(rows[('general_slow', 'rough')]['speed_mean']), 0.15, places=2)
            self.assertAlmostEqual(float(rows[('general_fast', 'rough')]['speed_mean']), 0.45, places=2)
            self.assertAlmostEqual(float(rows[('general_slow', 'rough')]['wobble_deg']), 3.0, places=1)
            self.assertAlmostEqual(float(rows[('general_fast', 'rough')]['wobble_deg']), 7.0, places=1)
            self.assertAlmostEqual(float(rows[('general_slow', 'flat')]['wobble_deg']), 1.0, places=1)
            self.assertGreater(float(rows[('general_slow', 'rough')]['dur_s']),
                               float(rows[('general_fast', 'rough')]['dur_s']))     # slow run spends longer there
            self.assertGreaterEqual(float(rows[('hstep', 'rough')]['tilt_max_deg']), 31.0)
            self.assertIn('31.0!', txt)                                              # tilt warning shown
            self.assertIn('TERRAIN GRID', txt)
            self.assertNotIn('WARNING', txt)

    def test_override_and_cells(self):
        with tempfile.TemporaryDirectory() as root:
            d = os.path.join(root, 'survey_a')
            os.makedirs(d)
            write_session(d, {}, {})
            out = os.path.join(root, 'o')
            self.run_compare(['a=' + d, '--override', '3,0=rough', '--out', out])
            with open(os.path.join(out, 'cells_classified.csv'), newline='') as f:
                cls = {(int(r['i']), int(r['j'])): r['class'] for r in csv.DictReader(f)}
            self.assertEqual(cls[(3, 0)], 'rough')
            out2 = os.path.join(root, 'o2')
            self.run_compare(['a=' + d, '--cells', '1:2,0:0', '--out', out2])
            with open(os.path.join(out2, 'cells_classified.csv'), newline='') as f:
                cls2 = {(int(r['i']), int(r['j'])) for r in csv.DictReader(f)}
            self.assertEqual(cls2, {(1, 0), (2, 0)})

    def test_custom_class_names_listed(self):
        with tempfile.TemporaryDirectory() as root:
            d = os.path.join(root, 'survey_a')
            os.makedirs(d)
            write_session(d, {}, {})
            out = os.path.join(root, 'o')
            _rc, txt = self.run_compare(['a=' + d, '--override', '1,0=cubeA 2,0=cubeB', '--out', out])
            with open(os.path.join(out, 'compare.csv'), newline='') as f:
                terrains = {r['terrain'] for r in csv.DictReader(f)}
            self.assertTrue({'cubeA', 'cubeB', 'ALL'} <= terrains, terrains)
            self.assertIn('cubeA', txt.split('wobble')[1])                  # appears in the wobble matrix

    def test_inconsistent_maps_warn(self):
        with tempfile.TemporaryDirectory() as root:
            da, db = os.path.join(root, 'a'), os.path.join(root, 'b')
            os.makedirs(da)
            os.makedirs(db)
            write_session(da, {}, {}, wall_y=0.65)
            write_session(db, {}, {}, wall_y=-0.65)                  # different start pose -> walls elsewhere
            _rc, txt = self.run_compare(['a=' + da, 'b=' + db, '--out', os.path.join(root, 'o')])
            self.assertIn('WARNING: walls of b overlap only', txt)

    def test_missing_map_and_errors(self):
        import compare
        with tempfile.TemporaryDirectory() as root:
            da, db = os.path.join(root, 'a'), os.path.join(root, 'b')
            os.makedirs(da)
            os.makedirs(db)
            write_session(da, {}, {})
            write_session(db, {}, {}, with_map=False)
            _rc, txt = self.run_compare(['a=' + da, 'b=' + db, '--out', os.path.join(root, 'o')])
            self.assertIn('WARNING: b has no map.npz', txt)
            with self.assertRaises(ValueError):
                compare.main(['b=' + db, '--out', os.path.join(root, 'o2')])          # no map at all
            with self.assertRaises(ValueError):
                compare.main(['x=' + da, 'x=' + db, '--out', os.path.join(root, 'o3')])   # duplicate names
            with self.assertRaises(ValueError):
                compare.main(['x=' + os.path.join(root, 'nope')])
            empty = os.path.join(root, 'e')
            os.makedirs(empty)
            open(os.path.join(empty, 'samples.csv'), 'w').close()
            with self.assertRaises(ValueError):
                compare.main(['e=' + empty])

    def test_parsers(self):
        import compare
        self.assertEqual(compare.parse_box('2:1,0:3'), (1, 2, 0, 3))
        self.assertEqual(compare.parse_override('1,0=rough; -2,3=flat'), {(1, 0): 'rough', (-2, 3): 'flat'})
        self.assertEqual(compare.parse_sessions(['/x/y/survey_1', 'g=/p/q']), [('survey_1', '/x/y/survey_1'), ('g', '/p/q')])


if __name__ == '__main__':
    unittest.main(verbosity=2)
