import json
import threading
import urllib.error
import urllib.request

import pytest

from rc import config
from rc.core import Core
from viewer.layers import Registry, build_registry
from viewer.server import serve


def test_dummy_layer_in_snapshot():
    r = Registry()
    r.layer('dummy', 'points', lambda: [[1.0, 2.0]], {'color': '#000'})
    s = r.snapshot()
    json.dumps(s)
    assert s['layers'][0]['name'] == 'dummy' and s['layers'][0]['data'] == [[1.0, 2.0]]


def test_broken_layer_does_not_break_snapshot():
    r = Registry()
    r.layer('bad', 'text', lambda: 1 / 0)
    r.layer('ok', 'text', lambda: {'a': 1})
    s = r.snapshot()
    json.dumps(s)
    assert s['layers'][0]['data'] is None and 'ZeroDivisionError' in s['layers'][0]['error']
    assert s['layers'][1]['data'] == {'a': 1}


def test_unknown_type_rejected():
    with pytest.raises(ValueError):
        Registry().layer('x', 'bogus', lambda: None)


def test_core_registry_serialises():
    c = Core(config.load())
    c.on_odom(0.0, 0.0, 0.0, 0.0)
    c.start(0.0)
    c.tick(0.1)
    s = build_registry(c).snapshot()
    json.dumps(s)
    names = [l['name'] for l in s['layers']]
    assert names[:3] == ['grid', 'path', 'dog'] and all('error' not in l for l in s['layers'])


def test_server_read_only():
    r = Registry()
    r.layer('dummy', 'text', lambda: {'k': 1})
    srv = serve(r.snapshot, 0, host='127.0.0.1')
    port = srv.server_address[1]
    try:
        body = urllib.request.urlopen('http://127.0.0.1:%d/state.json' % port, timeout=3).read()
        assert json.loads(body)['layers'][0]['data'] == {'k': 1}
        page = urllib.request.urlopen('http://127.0.0.1:%d/' % port, timeout=3).read()
        assert b'reactive-core map' in page
        req = urllib.request.Request('http://127.0.0.1:%d/state.json' % port, data=b'x', method='POST')
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(req, timeout=3)
        assert e.value.code == 405
    finally:
        srv.shutdown()
