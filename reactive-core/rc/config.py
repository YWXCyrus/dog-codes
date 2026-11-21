"""Parameters: flat keys from config.yaml, overridable with "key=value" strings."""
import os

import yaml

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.yaml')


def _read(path):
    with open(path, 'r') as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError('%s: top level must be a mapping' % path)
    return data


def _coerce(default, value):
    """Keep the type of the default (an int typed on the command line for a float key stays float)."""
    if isinstance(default, bool):
        if not isinstance(value, bool):
            raise ValueError('expected true/false, got %r' % (value,))
        return value
    if isinstance(default, float) and isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(default, int) and isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def load(path=None, overrides=()):
    """Return the parameter dict. `path` replaces/extends the defaults; `overrides` are "key=value"."""
    cfg = _read(DEFAULT_PATH)
    if path and os.path.abspath(path) != DEFAULT_PATH:
        for k, v in _read(path).items():
            if k not in cfg:
                raise KeyError('unknown parameter %r in %s' % (k, path))
            cfg[k] = _coerce(cfg[k], v)
    for item in overrides:
        if '=' not in item:
            raise ValueError('override must be key=value, got %r' % (item,))
        k, v = item.split('=', 1)
        k = k.strip()
        if k not in cfg:
            raise KeyError('unknown parameter %r' % (k,))
        cfg[k] = _coerce(cfg[k], yaml.safe_load(v))
    return cfg


def diff(cfg):
    """Parameters that differ from the defaults (printed at start, saved with the run)."""
    base = _read(DEFAULT_PATH)
    return {k: v for k, v in cfg.items() if base.get(k) != v}
