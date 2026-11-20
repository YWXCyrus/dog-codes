"""dog_tools render: draw the terrain map PNGs (needs numpy + OpenCV, no ROS)."""
import math
import os

import numpy as np

from core import CELL, UNLABELLED

# BGR colours of the gait labels on the path (black outline is added for contrast)
GAIT_COLOURS = {
    'flat_slow': (80, 220, 80), 'flat_medium': (0, 230, 230), 'flat_fast': (0, 140, 255),
    'crawl': (255, 160, 0), 'general': (255, 0, 255), 'grip': (255, 255, 0),
    'hstep': (0, 0, 255), UNLABELLED: (255, 255, 255),
}
_EXTRA = [(180, 105, 255), (0, 165, 255), (128, 255, 128), (255, 200, 200), (60, 60, 255), (200, 255, 255)]

UNOBSERVED = (60, 60, 60)
NO_DATA = (105, 105, 105)
WALL = (235, 235, 235)


def label_colour(label, table):
    if label in GAIT_COLOURS:
        return GAIT_COLOURS[label]
    if label not in table:
        table[label] = _EXTRA[len(table) % len(_EXTRA)]
    return table[label]


def _cv():
    import cv2
    return cv2


def _color_layer(values, vmin, vmax, grid, wall, observed):
    """values (ny,nx) float with NaN = no data -> BGR image (ny,nx,3), y up (row 0 = smallest y)."""
    cv2 = _cv()
    img = np.zeros(values.shape + (3,), np.uint8)
    img[:] = UNOBSERVED
    img[observed] = NO_DATA
    ok = np.isfinite(values)
    if ok.any():
        z = np.clip((values - vmin) / (vmax - vmin), 0, 1)
        z8 = np.where(ok, z * 255, 0).astype(np.uint8)
        col = cv2.applyColorMap(z8.reshape(-1, 1), cv2.COLORMAP_JET).reshape(values.shape + (3,))
        img[ok] = col[ok]
    img[wall] = WALL
    return img


def _put(img, text, org, scale=0.45, colour=(255, 255, 255), thick=1):
    cv2 = _cv()
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thick, cv2.LINE_AA)


def render_map(grid, samples, events, kind, run_name, out_path, thr_rough=0.05):
    """kind: 'rough' (floor-height scatter) or 'height' (mean floor height). Returns True if written."""
    cv2 = _cv()
    bnd = grid.bounds()
    pts = [(s['x'], s['y']) for s in samples if s['x'] == s['x']]
    if bnd is None and not pts:
        return False
    xs = [bnd[0], bnd[1]] if bnd else []
    ys = [bnd[2], bnd[3]] if bnd else []
    if pts:
        xs += [min(p[0] for p in pts), max(p[0] for p in pts)]
        ys += [min(p[1] for p in pts), max(p[1] for p in pts)]
    # expand to whole lattice cells + one cell margin, clip to the grid
    gx0, gx1 = grid.x0, grid.x0 + grid.nx * grid.res
    gy0, gy1 = grid.y0, grid.y0 + grid.ny * grid.res
    xmin = max(gx0, (math.floor(min(xs) / CELL - 0.5) - 1 + 0.5) * CELL)
    xmax = min(gx1, (math.ceil(max(xs) / CELL + 0.5) + 1 - 0.5) * CELL)
    ymin = max(gy0, (math.floor(min(ys) / CELL - 0.5) - 1 + 0.5) * CELL)
    ymax = min(gy1, (math.ceil(max(ys) / CELL + 0.5) + 1 - 0.5) * CELL)
    ia = int(max(0, math.floor((xmin - grid.x0) / grid.res)))
    ib = int(min(grid.nx, math.ceil((xmax - grid.x0) / grid.res)))
    ja = int(max(0, math.floor((ymin - grid.y0) / grid.res)))
    jb = int(min(grid.ny, math.ceil((ymax - grid.y0) / grid.res)))
    if ib <= ia or jb <= ja:
        return False
    wc = ib - ia
    sc = int(max(2, min(10, 2200 // wc, 2000 // (jb - ja))))
    ppm = sc / grid.res

    if kind == 'rough':
        vals = grid.local_rough(r=2, min_n=12)
        vmin, vmax = 0.0, 0.10
        title = 'ROUGHNESS: floor-height std (0.5 m)'
    else:
        vals = grid.mean_height(r=1, min_n=4)
        vmin, vmax = -0.10, 0.40
        title = 'FLOOR HEIGHT above start floor (m)'
    layer = _color_layer(vals, vmin, vmax, grid, grid.wall_cells(), grid.observed())
    layer = layer[ja:jb, ia:ib]
    layer = np.flipud(layer)                                   # +Y up
    img = cv2.resize(layer, (wc * sc, (jb - ja) * sc), interpolation=cv2.INTER_NEAREST)
    H, W = img.shape[:2]
    x_left = grid.x0 + ia * grid.res
    y_top = grid.y0 + jb * grid.res

    def P(x, y):
        return int(round((x - x_left) * ppm)), int(round((y_top - y) * ppm))

    # lattice grid + cell labels
    i0 = int(math.floor((x_left) / CELL)) - 1
    i1 = int(math.ceil((x_left + W / ppm) / CELL)) + 1
    j0 = int(math.floor((y_top - H / ppm) / CELL)) - 1
    j1 = int(math.ceil(y_top / CELL)) + 1
    for i in range(i0, i1 + 1):
        xb = (i + 0.5) * CELL
        px, _ = P(xb, 0)
        if 0 <= px < W:
            cv2.line(img, (px, 0), (px, H - 1), (200, 200, 200), 1, cv2.LINE_AA)
    for j in range(j0, j1 + 1):
        yb = (j + 0.5) * CELL
        _, py = P(0, yb)
        if 0 <= py < H:
            cv2.line(img, (0, py), (W - 1, py), (200, 200, 200), 1, cv2.LINE_AA)
    if sc >= 5:
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                cx, cy = P((i - 0.5) * CELL, (j + 0.5) * CELL)       # top-left corner of the cell
                if 0 <= cx < W - 30 and 12 <= cy < H:
                    _put(img, '%d,%d' % (i, j), (cx + 3, cy + 13), 0.38, (255, 255, 255))

    # path coloured by gait label
    table = {}
    prev = None
    for s in samples:
        if s['x'] != s['x']:
            prev = None
            continue
        p = P(s['x'], s['y'])
        if prev is not None and prev[1] == s['label'] and math.hypot(s['x'] - prev[2], s['y'] - prev[3]) < 0.5:
            cv2.line(img, prev[0], p, (0, 0, 0), 5, cv2.LINE_AA)
        prev = (p, s['label'], s['x'], s['y'])
    prev = None
    for s in samples:
        if s['x'] != s['x']:
            prev = None
            continue
        p = P(s['x'], s['y'])
        if prev is not None and prev[1] == s['label'] and math.hypot(s['x'] - prev[2], s['y'] - prev[3]) < 0.5:
            cv2.line(img, prev[0], p, label_colour(s['label'], table), 3, cv2.LINE_AA)
        prev = (p, s['label'], s['x'], s['y'])
    if samples:
        a, b = samples[0], samples[-1]
        cv2.circle(img, P(a['x'], a['y']), 7, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(img, P(a['x'], a['y']), 5, (0, 255, 0), -1, cv2.LINE_AA)
        _put(img, 'start', (P(a['x'], a['y'])[0] + 8, P(a['x'], a['y'])[1] - 6), 0.42, (0, 255, 0))
        ex, ey = P(b['x'], b['y'])
        yaw = math.radians(b['yaw_deg']) if b['yaw_deg'] == b['yaw_deg'] else 0.0
        cv2.arrowedLine(img, (ex, ey), (int(ex + 28 * math.cos(yaw)), int(ey - 28 * math.sin(yaw))),
                        (0, 0, 0), 5, cv2.LINE_AA, tipLength=0.4)
        cv2.arrowedLine(img, (ex, ey), (int(ex + 28 * math.cos(yaw)), int(ey - 28 * math.sin(yaw))),
                        (255, 255, 255), 2, cv2.LINE_AA, tipLength=0.4)

    # marks (M) and notes (N) placed on the nearest sample in time
    notes = []
    nm = nn = 0
    for (t, ev_kind, text) in events:
        if ev_kind not in ('mark', 'note') or not samples:
            continue
        s = min(samples, key=lambda q: abs(q['t_wall'] - t))
        if s['x'] != s['x']:
            continue
        x, y = P(s['x'], s['y'])
        if ev_kind == 'mark':
            nm += 1
            tag = 'M%d' % nm
            cv2.drawMarker(img, (x, y), (0, 0, 0), cv2.MARKER_TILTED_CROSS, 22, 5, cv2.LINE_AA)
            cv2.drawMarker(img, (x, y), (255, 255, 255), cv2.MARKER_TILTED_CROSS, 20, 2, cv2.LINE_AA)
        else:
            nn += 1
            tag = 'N%d' % nn
            cv2.circle(img, (x, y), 9, (0, 0, 0), 2, cv2.LINE_AA)
            notes.append('%s: %s' % (tag, text))
        _put(img, tag, (x + 8, y + 16), 0.5, (255, 255, 255), 1)

    # legend panel
    LW = 330
    panel = np.full((H, LW, 3), 30, np.uint8)
    y = 24
    _put(panel, run_name[:34], (8, y), 0.45, (200, 255, 200))
    y += 22
    for line in (title[:44], 'cell = 1.02 m, labels = lattice "i,j"', '+X right, +Y up; (0,0) = start cell'):
        _put(panel, line, (8, y), 0.4, (255, 255, 255))
        y += 18
    y += 6
    _put(panel, 'PATH (gait label)', (8, y), 0.45, (255, 255, 0))
    y += 20
    seen = []
    for s in samples:
        if s['label'] not in seen:
            seen.append(s['label'])
    for lab in seen:
        c = label_colour(lab, table)
        cv2.line(panel, (10, y - 4), (46, y - 4), (0, 0, 0), 7)
        cv2.line(panel, (10, y - 4), (46, y - 4), c, 4)
        _put(panel, lab[:24], (54, y), 0.45)
        y += 20
    y += 8
    _put(panel, 'COLOUR SCALE', (8, y), 0.45, (255, 255, 0))
    y += 10
    bar_h = 150
    grad = np.linspace(255, 0, bar_h).astype(np.uint8).reshape(-1, 1)
    bar = cv2.applyColorMap(np.repeat(grad, 22, axis=1), cv2.COLORMAP_JET)
    if y + bar_h < H:
        panel[y:y + bar_h, 10:32] = bar
        for frac in (0.0, 0.5, 1.0):
            yy = y + int((1 - frac) * (bar_h - 1))
            _put(panel, '%.2f' % (vmin + frac * (vmax - vmin)), (38, yy + 5), 0.4)
        if kind == 'rough':
            yy = y + int((1 - (thr_rough - vmin) / (vmax - vmin)) * (bar_h - 1))
            cv2.line(panel, (6, yy), (36, yy), (255, 255, 255), 2)
            _put(panel, '%.2f = dog "rough" threshold' % thr_rough, (82, yy + 4), 0.38, (255, 255, 255))
        y += bar_h + 14
    for txt, c in (('grey = never seen', UNOBSERVED), ('light grey = too few points', NO_DATA),
                   ('white = wall / tall object', WALL)):
        if y + 14 < H:
            cv2.rectangle(panel, (10, y - 10), (24, y + 2), c, -1)
            _put(panel, txt, (32, y), 0.4)
            y += 18
    for ln in notes[:12]:
        y += 4
        if y + 14 < H:
            _put(panel, ln[:46], (8, y), 0.4, (180, 255, 255))
            y += 16
    out = np.hstack([img, panel])
    tmp = out_path[:-4] + '.tmp.png'
    ok = cv2.imwrite(tmp, out)
    if ok:
        os.replace(tmp, out_path)
    return bool(ok)
