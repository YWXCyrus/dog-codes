#!/usr/bin/env python3
"""gait_label: tell the survey which gait you are using (run in a SECOND terminal on the dog container).

    python3 gait_label.py

Type a number or name each time you change gait in the app/controller; the survey stamps it with the
time. It only appends lines to <run>/labels.csv - it publishes nothing and talks to no robot.

    1..7 or a gait name   set the current gait
    +name                 set a gait with your own name, e.g. +rugged_high_step (use the app's label)
    -                     no gait / stopped (label "unlabelled")
    m                     MARK: you stumbled, slipped or had to catch the dog right now
    n <text>              NOTE at the current position, e.g. n start of the rubble
    l                     list the choices          q   quit
"""
import csv
import os
import sys
import time

GAITS = ['flat_slow', 'flat_medium', 'flat_fast', 'crawl', 'general', 'grip', 'hstep']
UNLABELLED = 'unlabelled'


def find_run(root=None):
    root = os.path.expanduser(root or os.environ.get('DOG_TOOLS_RUNS', '~/dog_tools_runs'))
    latest = os.path.join(root, 'LATEST')
    if not os.path.exists(latest):
        return None
    with open(latest) as f:
        run = f.read().strip()
    return run if os.path.isdir(run) else None


def parse(line, current):
    """-> (kind, text) to record, ('info', message) to print, ('quit', ''), or None for empty input."""
    s = line.strip()
    if not s:
        return None
    low = s.lower()
    if low in ('q', 'quit', 'exit'):
        return ('quit', '')
    if low in ('l', 'list', 'help', '?'):
        return ('info', menu())
    if low in ('m', 'mark'):
        return ('mark', current)
    if low == '-':
        return ('gait', UNLABELLED)
    if low == 'n' or low.startswith('n '):
        txt = s[1:].strip()
        return ('note', txt) if txt else ('info', 'usage: n <text>')
    if s.startswith('+'):
        name = s[1:].strip().replace(' ', '_')
        if name and name.replace('_', '').replace('-', '').isalnum():
            return ('gait', name)
        return ('info', 'custom names: letters/digits/_ only, e.g. +rugged_high_step')
    if low.isdigit():
        i = int(low)
        if 1 <= i <= len(GAITS):
            return ('gait', GAITS[i - 1])
        return ('info', 'no such number. ' + menu())
    if low in GAITS:
        return ('gait', low)
    return ('info', 'not understood: "%s". %s' % (s, menu()))


def menu():
    return ('choices: ' + '  '.join('%d=%s' % (i + 1, g) for i, g in enumerate(GAITS)) +
            '  | +name custom | - none | m mark | n text note | q quit')


def append(path, kind, text):
    with open(path, 'a', newline='') as f:
        csv.writer(f).writerow(['%.3f' % time.time(), kind, text])
        f.flush()
        os.fsync(f.fileno())


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else None
    run = find_run(root)
    if not run:
        print('No survey run found. Start the survey first (bash scripts/run_survey.sh).')
        return 2
    path = os.path.join(run, 'labels.csv')
    if not os.path.exists(path):
        print('labels.csv missing in %s - is the survey running?' % run)
        return 2
    print('Labelling run: %s' % run)
    print(menu())
    current = UNLABELLED
    while True:
        try:
            line = input('[%s] > ' % current)
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        r = parse(line, current)
        if r is None:
            continue
        kind, text = r
        if kind == 'quit':
            return 0
        if kind == 'info':
            print(text)
            continue
        append(path, kind, text)
        if kind == 'gait':
            current = text
        print('  recorded %s: %s' % (kind, text))


if __name__ == '__main__':
    sys.exit(main())
