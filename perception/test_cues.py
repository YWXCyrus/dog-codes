"""Unit tests of the perception cues for navigation (no ROS): python3 -m pytest -q perception/test_cues.py"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'frontier-fastlio', 'dog_codes'))
import perception_cues as PC          # noqa: E402

MISSION = {'M01': {'task_type': 'object', 'label': ''}, 'M02': {'task_type': 'hazard', 'label': ''},
           'M03': {'task_type': 'survivor', 'scene_type': ''}, 'M04': {'task_type': 'qr'},
           'M05': {'task_type': 'survivor', 'scene_type': ''}}


def det(cls='stairs', conf=0.8, bearing=0.1, w=0.4):
    return {'cls': cls, 'conf': conf, 'bearing': bearing, 'w': w}


# ---------------------------------------------------------------- camera model persistence
def test_one_flash_is_not_stairs():
    p = PC.DetectionPersist('stairs')
    p.update([det()], 0.0)
    assert p.seen(0.0) is None


def test_steady_stairs_detection():
    p = PC.DetectionPersist('stairs')
    for k in range(3):
        p.update([det(bearing=0.10 + 0.01 * k)], 0.2 * k)
    s = p.seen(0.4)
    assert s is not None and s['n'] == 3 and abs(s['bearing'] - 0.11) < 0.011


def test_low_confidence_other_class_and_scattered_directions_ignored():
    p = PC.DetectionPersist('stairs', conf=0.5)
    p.update([det(conf=0.3), det(cls='human'), det(cls='curtain')], 0.0)
    p.update([det(bearing=-0.8)], 0.1)
    p.update([det(bearing=0.0)], 0.2)
    p.update([det(bearing=0.8)], 0.3)
    assert p.seen(0.3) is None                     # three directions 45 deg apart: not one staircase


def test_old_detections_expire():
    p = PC.DetectionPersist('stairs', window_s=1.5)
    for k in range(3):
        p.update([det()], 0.1 * k)
    p.update([], 3.0)
    assert p.seen(3.0) is None
    assert p.alive(3.0) and not p.alive(10.0)


def test_bad_detection_entries_do_not_crash():
    p = PC.DetectionPersist('stairs')
    p.update([{'cls': 'stairs', 'conf': 'x'}, None, {'cls': 'stairs'}], 0.0)
    p.update(None, 0.1)


# ---------------------------------------------------------------- scene server answers
def test_stairs_in_scene_answer():
    assert PC.stairs_in_result({'desc': 'metal staircase with handrails', 'evidence': []})
    assert PC.stairs_in_result({'desc': 'a corridor', 'evidence': ['stairs']})
    assert not PC.stairs_in_result({'desc': 'a kitchen with a stove', 'evidence': ['utensils']})


# ---------------------------------------------------------------- mission status / backtracking
def test_nothing_done_suggests_exploring_before_stairs():
    st = PC.mission_status(MISSION, {})
    assert st['open'] == ['M01', 'M02', 'M03', 'M04', 'M05'] and not st['all_done']
    assert st['suggest'][-2:] == ['explore', 'go_stairs_when_out_of_time']


def test_survivor_room_known_but_qr_not_read_suggests_reading_it():
    best = {'M03': {'task_type': 'survivor', 'scene_type': 'kitchen', 'pos': [2.04, 1.02]}}
    st = PC.mission_status(MISSION, best, survivor_cells=[(2, 1)])
    assert st['suggest'][0] == 'read_qr'
    assert st['tasks']['M03']['reported'] and st['tasks']['M03']['scene_type'] == 'kitchen'


def test_qr_hint_points_to_the_second_survivor_room():
    best = {'M04': {'task_type': 'qr', 'message': 'My friend is in the market.'},
            'M03': {'task_type': 'survivor', 'scene_type': 'kitchen', 'pos': [2.04, 1.02]}}
    st = PC.mission_status(MISSION, best, qr_text='My friend is in the market.', m05_hint='market')
    assert st['qr']['read'] and st['tasks']['M04']['message'].endswith('market.')
    assert 'find_scene:market' in st['suggest'] and 'read_qr' not in st['suggest']


def test_all_done_goes_to_the_stairs():
    best = {t: {'task_type': MISSION[t]['task_type']} for t in MISSION}
    st = PC.mission_status(MISSION, best)
    assert st['all_done'] and st['open'] == [] and st['suggest'] == ['go_stairs']


def test_status_is_json_serialisable():
    import json
    st = PC.mission_status(MISSION, {}, scene_by_cell={(1, 0): {'clinic': 0.8, 'kitchen': 0.3}},
                           stairs_cells=[(3, 1)], t=12.3)
    s = json.dumps(st)
    assert '"1,0": "clinic"' in s and st['stairs_cells'] == [[3, 1]]
    assert math.isclose(st['t'], 12.3)
