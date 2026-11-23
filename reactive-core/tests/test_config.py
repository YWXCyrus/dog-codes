import pytest

from rc import config


def test_defaults_load():
    cfg = config.load()
    assert cfg['step_limit'] == 0.25
    assert cfg['stairs'] == 'climb'
    assert cfg['knee_names'][0] == 'LF_Joint_2'


def test_overrides_typed():
    cfg = config.load(overrides=['step_limit=0.2', 'wall_side=right', 'viewer=false'])
    assert cfg['step_limit'] == pytest.approx(0.2)
    assert isinstance(cfg['step_limit'], float)
    assert cfg['wall_side'] == 'right'
    assert cfg['viewer'] is False


def test_int_override_of_float_key_stays_float():
    cfg = config.load(overrides=['cruise_vx=0'])
    assert isinstance(cfg['cruise_vx'], float)


def test_unknown_key_rejected():
    with pytest.raises(KeyError):
        config.load(overrides=['no_such_key=1'])


def test_bad_override_format():
    with pytest.raises(ValueError):
        config.load(overrides=['step_limit'])


def test_diff():
    cfg = config.load(overrides=['step_limit=0.2'])
    assert config.diff(cfg) == {'step_limit': 0.2}


def test_number_typo_rejected():
    # real case 2026-11-22: 'exit' typed onto the command line (sweep_margin_m='0.05exit')
    for bad in ('stop_m=0.05exit', 'mem_time_s=1eexit', 'cruise_vx=fast', 'stop_m=true'):
        with pytest.raises(ValueError):
            config.load(overrides=[bad])


def test_number_forms_accepted():
    cfg = config.load(overrides=['mem_time_s=1', 'cruise_vx=.3', 'stop_m=3e-1'])
    assert cfg['mem_time_s'] == 1.0 and cfg['cruise_vx'] == 0.3 and cfg['stop_m'] == 0.3
