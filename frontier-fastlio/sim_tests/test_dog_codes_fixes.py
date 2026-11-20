#!/usr/bin/env python3
"""Unit tests for the dog_codes fixes (QR persistence filter). Run in WSL with ROS sourced:
   cd frontier-fastlio/sim_tests && python3 -B -m unittest -v test_dog_codes_fixes
(The explorer fix is tested end-to-end in the simulator: run_sim_test.sh 150 --start=-0.2,0,0)"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dog_codes'))

import common as C  # noqa: E402


class TestQrPersist(unittest.TestCase):
    def test_single_flicker_rejected(self):
        f = C.QrPersist()
        self.assertFalse(f.update(10.0, 120.0, 300.0))

    def test_three_consistent_accepted(self):
        f = C.QrPersist()
        self.assertFalse(f.update(10.0, 120.0, 300.0))
        self.assertFalse(f.update(10.17, 118.0, 310.0))
        self.assertTrue(f.update(10.33, 121.0, 305.0))
        self.assertTrue(f.update(10.5, 125.0, 300.0))            # keeps being accepted while it persists

    def test_scattered_centres_rejected(self):
        f = C.QrPersist()
        for i, cx in enumerate((100.0, 300.0, 500.0, 120.0, 330.0, 520.0)):
            self.assertFalse(f.update(10.0 + 0.17 * i, 120.0, cx))

    def test_old_detections_expire(self):
        f = C.QrPersist()
        f.update(10.0, 120.0, 300.0)
        f.update(10.17, 120.0, 300.0)
        self.assertFalse(f.update(12.0, 120.0, 300.0))            # the first two are > 1 s old

    def test_implausible_rejected_and_not_counted(self):
        f = C.QrPersist()
        for _ in range(5):                                        # real garbage values from trial 1
            self.assertFalse(f.update(10.0, 41354.9, -20401.9))
            self.assertFalse(f.update(10.0, 1865.3, -377.8))
        self.assertFalse(f.plausible(10.0, 300.0))                # too narrow to be a QR
        self.assertFalse(f.plausible(100.0, 700.0))               # outside the 640 px image
        self.assertTrue(f.plausible(100.0, 300.0))
        self.assertFalse(f.update(11.0, 120.0, 300.0))            # garbage did not pre-fill the history


if __name__ == '__main__':
    unittest.main(verbosity=2)
