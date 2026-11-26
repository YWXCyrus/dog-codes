"""Camera helpers (rc/vision_node.py) that need no ultralytics and no ROS."""
import math

import numpy as np
import pytest

from rc import config
from rc import curtain as CU
from rc.core import Core
from rc.vision_node import boxes_to_detections, image_to_rgb, pixel_bearing

NAMES = {0: 'curtain', 1: 'human', 2: 'stairs'}


def test_pixel_bearing_is_zero_in_the_middle_left_positive_and_fov_sized_at_the_edge():
    assert pixel_bearing(320, 640, 70) == pytest.approx(0.0)
    assert pixel_bearing(0, 640, 70) == pytest.approx(math.radians(35))        # left edge, + = left
    assert pixel_bearing(640, 640, 70) == pytest.approx(-math.radians(35))
    assert pixel_bearing(160, 640, 70) > 0 > pixel_bearing(480, 640, 70)


def test_boxes_to_detections_names_bearing_width_and_confidence_filter():
    boxes = [(100, 50, 220, 400, 0.91, 0), (500, 60, 540, 300, 0.20, 1), (300, 0, 340, 100, 0.8, 2)]
    d = boxes_to_detections(boxes, NAMES, 640, 70, min_conf=0.35)
    assert [x['cls'] for x in d] == ['curtain', 'stairs']
    assert d[0]['bearing'] > 0 and d[0]['w'] == pytest.approx(120 / 640, abs=1e-3)
    assert abs(d[1]['bearing']) < 0.05


class Img:
    def __init__(self, enc, h, w, data, step):
        self.encoding, self.height, self.width, self.data, self.step = enc, h, w, data, step


def test_image_to_rgb_swaps_bgr_and_ignores_row_padding():
    bgr = np.zeros((2, 3, 3), np.uint8)
    bgr[:, :, 0] = 10                                   # blue
    bgr[:, :, 2] = 200                                  # red
    padded = np.zeros((2, 12), np.uint8)
    padded[:, :9] = bgr.reshape(2, 9)
    out = image_to_rgb(Img('bgr8', 2, 3, padded.tobytes(), 12))
    assert out.shape == (2, 3, 3) and out[0, 0, 0] == 200 and out[0, 0, 2] == 10
    assert image_to_rgb(Img('16UC1', 2, 3, b'', 6)) is None


def test_core_takes_detections_into_the_curtain_check():
    cfg = config.load()
    core = Core(cfg)
    for i in range(4):
        core.on_vision([{'cls': 'curtain', 'conf': 0.9, 'bearing': 0.0}, {'cls': 'human', 'conf': 0.8, 'bearing': 0.5}], 0.2 * i)
    face = np.stack([np.full(40, 1.0), np.linspace(-0.5, 0.5, 40)], axis=1)
    assert CU.analyse(core.vision, face, 1.0, cfg).found
    assert core.vision.counts == {'curtain': 4, 'human': 4}
