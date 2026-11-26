"""Closed loop, grid navigation on competition-like arenas (sim/scenarios.py). The scenarios run in
parallel processes (~2-4 min). Each must: visit all 15 cells, stop in the final cell, never touch a
wall. tools/sim_batch.py runs the same scenarios by hand (also --drift)."""
import multiprocessing as mp

import pytest

SCEN = [('maze1', False), ('maze3_back', False), ('maze4_side', True), ('maze6_boards', False),
        ('maze7_props', False), ('curtain_leaf', False), ('curtain_alt', True), ('stairs', False)]


@pytest.fixture(scope='module')
def results():
    from tools.sim_batch import run_one
    jobs = [(n, [], drift, 480.0, False) for n, drift in SCEN]
    with mp.get_context('spawn').Pool(min(len(jobs), max(1, mp.cpu_count() - 1))) as pool:
        rows = pool.map(run_one, jobs)
    return {(r['name'], d): r for r, (_, d) in zip(rows, SCEN)}


@pytest.mark.parametrize('name,drift', SCEN)
def test_scenario_explores_all_and_finishes(results, name, drift):
    r = results[(name, drift)]
    assert r['coll'] == 0, r
    assert not r['missing'], r
    assert r['finished'] and r['in_final'], r
    assert r['err'] < 0.25, r                      # arena pose vs truth at the end
    assert r['wrong_edges'] <= 1, r


def test_curtains_are_pushed_through(results):
    assert 'passed' in results[('curtain_leaf', False)]['curtain']
    assert 'passed' in results[('curtain_alt', True)]['curtain']


def test_stairs_crossed_in_ai(results):
    assert results[('stairs', False)]['stairs'] == ['DONE']
