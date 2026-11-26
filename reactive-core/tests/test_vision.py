"""Camera model helpers (rc/vision_node.py) and the detection persistence (rc/camcues.py); no ultralytics, no ROS."""
import math

import pytest

from rc.camcues import DetectionPersist
from rc.vision_node import boxes_to_detections, pixel_bearing

NAMES = {0: 'curtain', 1: 'human', 2: 'stairs'}


def test_pixel_bearing_left_positive_and_fov_sized_at_the_edges():
    assert pixel_bearing(320, 640, 70) == pytest.approx(0.0)
    assert pixel_bearing(0, 640, 70) == pytest.approx(math.radians(35))
    assert pixel_bearing(640, 640, 70) == pytest.approx(-math.radians(35))


def test_boxes_to_detections_filters_confidence():
    d = boxes_to_detections([(100, 50, 220, 400, 0.91, 0), (500, 60, 540, 300, 0.20, 1)], NAMES, 640, 70, min_conf=0.35)
    assert [x['cls'] for x in d] == ['curtain'] and d[0]['bearing'] > 0


def test_persistence_needs_n_detections_in_one_direction():
    p = DetectionPersist('curtain', conf=0.5, n=3, window_s=1.5, cone_deg=20)
    p.update([{'cls': 'curtain', 'conf': 0.8, 'bearing': 0.1}], 0.0)
    p.update([{'cls': 'curtain', 'conf': 0.8, 'bearing': -1.0}], 0.2)
    p.update([{'cls': 'curtain', 'conf': 0.8, 'bearing': 0.12}], 0.4)
    assert p.seen(0.4) is None                     # two agree, one points elsewhere
    p.update([{'cls': 'curtain', 'conf': 0.8, 'bearing': 0.11}], 0.6)
    s = p.seen(0.6)
    assert s is not None and abs(s['bearing'] - 0.11) < 0.02
    assert p.seen(3.0) is None                     # old detections expire
