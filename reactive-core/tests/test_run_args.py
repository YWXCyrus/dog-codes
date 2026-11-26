import json

import run
from rc import config
from rc.core import Core


def chk(argv, dom):
    return run.safety_check(run.parse_args(argv), {'ROS_DOMAIN_ID': dom})


def test_default_is_dry_run():
    ok, code, mode = chk([], '61')
    assert ok and mode == 'dry-run'


def test_live_needs_domain_61():
    assert chk(['--live'], '42')[:2] == (False, 2)
    assert chk(['--live'], '1')[:2] == (False, 2)          # the dog's old domain (until 2026-11-27)
    assert chk(['--live'], '61') == (True, 0, 'live')


def test_sim_refused_on_dog_domains():
    assert chk(['--sim'], '61')[:2] == (False, 2)
    assert chk(['--sim'], '1')[:2] == (False, 2)
    assert chk(['--sim'], '42') == (True, 0, 'sim')


def test_auto_go_only_with_sim():
    assert not chk(['--auto-go'], '42')[0]
    assert not chk(['--live', '--auto-go'], '61')[0]
    assert chk(['--sim', '--auto-go'], '42')[0]


def test_gait_test_needs_publishing_mode():
    assert not chk(['--gait-test', 'ai_basic'], '61')[0]
    assert chk(['--live', '--gait-test', 'ai_basic'], '61')[0]


def test_live_and_sim_exclusive():
    assert not chk(['--live', '--sim'], '61')[0]


def test_set_parsed():
    a = run.parse_args(['--set', 'step_limit=0.2', '--set', 'cruise_vx=0.12'])
    cfg = config.load(None, a.set)
    assert cfg['step_limit'] == 0.2 and cfg['cruise_vx'] == 0.12


def test_snapshot_json():
    c = Core(config.load())
    json.dumps(c.snapshot(0.0))
    c.on_odom(0.0, 0.0, 0.0, 0.0)
    c.start(0.0)
    c.tick(0.1)
    s = c.snapshot(0.2)
    json.dumps(s)
    assert s['grid']['visited'] == [[0, 0]] and s['dog'] is not None
