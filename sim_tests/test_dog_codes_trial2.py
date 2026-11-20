#!/usr/bin/env python3
"""Unit tests for the trial-2 fixes (20 Nov 2026): BumpDetector, OdomWatchdog, rear_fence_mask, and replays of the
REAL recorded runs (sim_tests/real_data/*.csv, extracted from the dog's light bags).
Run in WSL with ROS sourced:   cd codes/sim_tests && python3 -B -m unittest -v test_dog_codes_trial2
The end-to-end behaviour (explorer + gate together) is tested in the simulator: see README, section 'trial 2 fixes'."""
import csv
import math
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'dog_codes'))

import common as C  # noqa: E402

DATA = os.path.join(HERE, 'real_data')


def load(name):
    rows = []
    with open(os.path.join(DATA, name)) as f:
        for r in csv.DictReader(f):
            rows.append((float(r['t']), r['kind'], float(r['a']), float(r['b'])))
    return rows


def replay_bump(rows, **kw):
    """Feed the detector exactly as explorer.control() does: one update per commanded vx, with the latest odom pose."""
    bd = C.BumpDetector(**kw)
    pose, hits = None, []
    for t, k, a, b in rows:
        if k == 'odom':
            pose = (a, b)
        elif pose is not None and bd.update(t, pose[0], pose[1], a):
            hits.append((t, bd.last))
    return hits


def replay_watchdog(rows, **kw):
    wd = C.OdomWatchdog(**kw)
    first, peak = None, 0.0
    for t, k, a, b in rows:
        if k != 'odom':
            continue
        f = wd.update(t, a, b)
        peak = max(peak, wd.speed)
        if f and first is None:
            first = (t, wd.speed)
    return first, peak


class TestBumpDetector(unittest.TestCase):
    def feed(self, bd, t0, secs, vx, speed, hz=10.0):
        """Walk `secs` seconds commanding vx while actually moving at `speed`; returns the times it fired."""
        hits, x = [], 0.0
        for i in range(int(secs * hz)):
            t = t0 + i / hz
            x += speed / hz
            if bd.update(t, x, 0.0, vx):
                hits.append(round(t - t0, 1))
        return hits

    def test_pushing_into_a_wall_fires_once_per_window(self):
        bd = C.BumpDetector()
        hits = self.feed(bd, 0.0, 5.0, 0.20, 0.01)
        self.assertEqual(len(hits), 1)
        self.assertLess(hits[0], 4.0)

    def test_normal_walking_does_not_fire(self):
        bd = C.BumpDetector()
        self.assertEqual(self.feed(bd, 0.0, 20.0, 0.25, 0.20), [])

    def test_slow_walking_on_rubble_does_not_fire(self):
        # survey session 1: RUG general averaged 0.11-0.18 m/s; even 40 % of a 0.25 m/s command is fine
        bd = C.BumpDetector()
        self.assertEqual(self.feed(bd, 0.0, 20.0, 0.25, 0.10), [])

    def test_turning_in_place_does_not_fire(self):
        bd = C.BumpDetector()
        self.assertEqual(self.feed(bd, 0.0, 20.0, 0.0, 0.0), [])

    def test_mostly_not_commanded_forward_does_not_fire(self):
        bd = C.BumpDetector()
        hits = []
        for i in range(60):                                   # 0.1 s steps: forward only 1 tick in 3
            t = i / 10.0
            hits += [bd.update(t, 0.0, 0.0, 0.2 if i % 3 == 0 else 0.0)]
        self.assertFalse(any(hits))

    def test_reset_clears_history(self):
        bd = C.BumpDetector()
        self.feed(bd, 0.0, 3.0, 0.2, 0.0)
        bd.reset()
        self.assertFalse(bd.update(3.1, 0.0, 0.0, 0.2))

    def test_backing_off_is_not_a_push(self):
        bd = C.BumpDetector()
        self.assertEqual(self.feed(bd, 0.0, 6.0, -0.10, 0.0), [])


class TestOdomWatchdog(unittest.TestCase):
    def run_wd(self, speed, secs, hz=10.0, **kw):
        wd = C.OdomWatchdog(**kw)
        for i in range(int(secs * hz)):
            if wd.update(i / hz, speed * i / hz, 0.0):
                return i / hz
        return None

    def test_normal_speeds_never_fire(self):
        for v in (0.0, 0.15, 0.3, 0.43):                      # 0.43 = fastest hand-driven walk on record
            self.assertIsNone(self.run_wd(v, 30.0), v)

    def test_impossible_speed_fires_quickly(self):
        t = self.run_wd(1.5, 10.0)
        self.assertIsNotNone(t)
        self.assertLess(t, 2.0)

    def test_short_blip_does_not_fire(self):
        wd = C.OdomWatchdog()
        x, fired = 0.0, False
        for i in range(100):
            if i == 50:
                x += 0.5                                       # one 0.5 m jump in one sample, then normal again
            fired |= wd.update(i / 10.0, x, 0.0)
        self.assertFalse(fired)

    def test_reset(self):
        wd = C.OdomWatchdog()
        for i in range(30):
            wd.update(i / 10.0, 2.0 * i / 10.0, 0.0)
        wd.reset()
        self.assertEqual(wd.speed, 0.0)
        self.assertFalse(wd.update(5.0, 0.0, 0.0))


class TestRearFence(unittest.TestCase):
    def test_mask_geometry(self):
        xs = np.arange(-2.0, 2.01, 0.1)
        X, Y = np.meshgrid(xs, xs)
        m = C.rear_fence_mask(X, Y, (0.0, 0.0), 0.0, 1.5)
        at = lambda x, y: bool(m[int(round((y + 2.0) / 0.1)), int(round((x + 2.0) / 0.1))])  # noqa: E731
        self.assertTrue(at(-0.5, 0.0))        # behind, inside the radius
        self.assertTrue(at(-1.0, 0.8))
        self.assertFalse(at(-0.1, 0.0))       # beside / on the start line (margin)
        self.assertFalse(at(0.5, 0.0))        # in front
        self.assertFalse(at(-1.8, 0.0))       # behind but outside the radius
        self.assertFalse(at(0.0, 1.0))

    def test_follows_start_heading(self):
        xs = np.arange(-2.0, 2.01, 0.1)
        X, Y = np.meshgrid(xs, xs)
        m = C.rear_fence_mask(X, Y, (0.0, 0.0), math.pi / 2, 1.5)     # facing +y: behind is -y
        at = lambda x, y: bool(m[int(round((y + 2.0) / 0.1)), int(round((x + 2.0) / 0.1))])  # noqa: E731
        self.assertTrue(at(0.0, -0.6))
        self.assertFalse(at(-0.6, 0.0))


class TestRealRuns(unittest.TestCase):
    """Replays of what the dog really did. If these ever fail, the detectors changed behaviour on real data."""

    def test_run_1003_odometry_ok_and_bump_found_at_the_wall(self):
        rows = load('trial2_run1003.csv')
        first, peak = replay_watchdog(rows)
        self.assertIsNone(first)                               # odometry was fine in this run
        self.assertLess(peak, 0.25)
        hits = replay_bump(rows)
        self.assertEqual(len(hits), 1)                         # the dog pushed against the unseen wall behind the start
        t, (cmd_dist, moved) = hits[0]
        self.assertTrue(38.0 < t < 46.0, t)                    # pushing began at ~39 s (commanded 0.22 m/s, moved 0.04 m/s)
        self.assertGreater(cmd_dist, 0.5)
        self.assertLess(moved, 0.2)

    def test_run_1007_odometry_divergence_caught(self):
        rows = load('trial2_run1007.csv')
        first, peak = replay_watchdog(rows)
        self.assertIsNotNone(first)
        t, speed = first
        # the pose first misbehaves at ~35 s; the real run went on to 284 m. Caught within ~2.5 s of the first anomaly.
        self.assertTrue(35.0 < t < 39.0, t)
        self.assertGreater(speed, 0.6)
        self.assertGreater(peak, 20.0)

    def test_no_false_alarm_on_the_rest_of_the_real_data(self):
        for name in ('trial1_walk0726.csv', ):
            first, peak = replay_watchdog(load(name))
            self.assertIsNone(first, name)
            self.assertLess(peak, 0.6, name)
        # hand-driven survey (5 Hz samples): the fastest real walking on record
        wd = C.OdomWatchdog()
        peak, fired = 0.0, False
        with open(os.path.join(DATA, 'survey1_hand_driven_samples.csv')) as f:
            for r in csv.DictReader(f):
                fired |= wd.update(float(r['t_rel']), float(r['x']), float(r['y']))
                peak = max(peak, wd.speed)
        self.assertFalse(fired)
        self.assertLess(peak, 0.6)

    def test_trial1_walk_bump_is_the_push_at_the_start(self):
        # trial 1: the dog stayed at the start (explorer 'never left the start cell'); the detector fires there and only there
        hits = replay_bump(load('trial1_walk0726.csv'))
        self.assertEqual(len(hits), 1)
        t, (cmd_dist, moved) = hits[0]
        self.assertLess(moved, 0.1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
