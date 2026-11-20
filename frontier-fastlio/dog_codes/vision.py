"""Pure-OpenCV vision helpers (no ROS): QR decoding and hazard-sign detection/classification.

Hazard signs are yellow triangles with a black border and a black pictogram. The pipeline
  yellow mask -> triangle fit -> affine warp to a canonical triangle -> binarised pictogram
  -> correlation against templates built from the organisers' sheet (assets/hazard_sheet.jpg)
Only 4 of the sheet's pictograms map to the competition vocabulary; the others are known
distractors. `flammable` has no sheet pictogram, so a synthetic flame template is used.
"""
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, 'assets')

# order of the sheet (row-major, 5 columns x 2 rows)
SHEET_NAMES = ['radioactive', 'high_voltage', 'exclamation', 'magnet', 'radio_wave',
               'biological', 'laser', 'wave', 'optical', 'chemical']
VALID = {'radioactive', 'high_voltage', 'biological', 'chemical', 'flammable'}
CAN = 96
SYM = 40


# ------------------------------------------------------------------------------ QR
_qr_cache = {}


def _detectors():
    if 'd' not in _qr_cache:
        ds = []
        if hasattr(cv2, 'QRCodeDetectorAruco'):
            try:
                ds.append(cv2.QRCodeDetectorAruco())
            except Exception:
                pass
        ds.append(cv2.QRCodeDetector())
        _qr_cache['d'] = ds
    return _qr_cache['d']


def decode_qr(bgr):
    """Return (text, width_px, centre_xy) if decoded; ('', width_px, centre) if a QR-like code
    was located but not decoded; None if nothing found."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    variants = [gray]
    clahe = cv2.createCLAHE(2.0, (8, 8))
    variants.append(clahe.apply(gray))
    variants.append(cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC))
    seen = None
    for vi, img in enumerate(variants):
        sc = 2.0 if vi == 2 else 1.0
        for det in _detectors():
            try:
                text, pts, _ = det.detectAndDecode(img)
            except Exception:
                continue
            if pts is not None and len(pts):
                q = np.asarray(pts).reshape(-1, 2) / sc
                w = float(np.mean([np.linalg.norm(q[i] - q[(i + 1) % len(q)])
                                   for i in range(len(q))]))
                c = q.mean(0)
                if not text and len(q) == 4 and w >= 40:
                    # OpenCV's decoder may be missing (QUIRC not linked): decode it ourselves
                    try:
                        import qr_decode
                        text = qr_decode.decode_quad(gray if sc == 1.0 else cv2.resize(gray, None, fx=1 / sc, fy=1 / sc),
                                                     q) or ''
                    except Exception:
                        text = ''
                if text:
                    return text, w, (float(c[0]), float(c[1]))
                seen = ('', w, (float(c[0]), float(c[1])))
    return seen


# ------------------------------------------------------------------------------ hazards
def yellow_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (12, 110, 110), (36, 255, 255))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return m


def find_triangles(bgr, min_area=450):
    """List of dicts {pts (3,2) float32 [top, bl, br], area, bbox} for yellow triangles."""
    m = yellow_mask(bgr)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w < 18 or h < 16 or not (0.7 < w / h < 2.2):
            continue
        try:
            ta, tri = cv2.minEnclosingTriangle(c.astype(np.float32))
        except cv2.error:
            continue
        if tri is None or ta <= 0:
            continue
        if a / ta < 0.62:
            continue
        t = tri.reshape(3, 2).astype(np.float32)
        top = t[np.argmin(t[:, 1])]
        rest = [v for v in t if not np.array_equal(v, top)]
        rest.sort(key=lambda v: v[0])
        if len(rest) != 2:
            continue
        pts = np.array([top, rest[0], rest[1]], np.float32)
        out.append({'pts': pts, 'area': float(a), 'bbox': (x, y, w, h)})
    return out


def canonical(bgr, pts):
    dst = np.array([[CAN / 2, 4], [4, CAN - 6], [CAN - 4, CAN - 6]], np.float32)
    M = cv2.getAffineTransform(pts.astype(np.float32), dst)
    return cv2.warpAffine(bgr, M, (CAN, CAN), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def pictogram(bgr, pts):
    """Binary SYMxSYM pictogram (float32 0/1, 1 = black ink) of a triangle."""
    can = canonical(bgr, pts)
    gray = cv2.cvtColor(can, cv2.COLOR_BGR2GRAY)
    # symbol area: central window of the canonical triangle
    win = gray[40:CAN - 14, 22:CAN - 22]
    th = cv2.threshold(win, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # blank out the triangle's sloped black border that intrudes into the window corners
    ys, xs = np.mgrid[0:th.shape[0], 0:th.shape[1]]
    cx = th.shape[1] / 2.0
    inner = (np.abs(xs - cx) < (th.shape[1] / 2.0 - 2) * (0.30 + 0.70 * ys / max(1, th.shape[0])) * 1.0)
    th = th * inner.astype(np.uint8)
    th = cv2.resize(th, (SYM, SYM), interpolation=cv2.INTER_AREA)
    return (th > 100).astype(np.float32)


def _score(a, b):
    a = cv2.GaussianBlur(a, (3, 3), 0)
    b = cv2.GaussianBlur(b, (3, 3), 0)
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 1e-6 else 0.0


def _flame_template():
    img = np.full((300, 340, 3), 255, np.uint8)
    tri = np.array([[170, 10], [10, 290], [330, 290]], np.int32)
    cv2.fillPoly(img, [tri], (0, 0, 0))
    inner = np.array([[170, 55], [45, 268], [295, 268]], np.int32)
    cv2.fillPoly(img, [inner], (0, 200, 255))      # BGR yellow-orange
    flame = np.array([[170, 110], [200, 160], [215, 150], [225, 190], [210, 235], [170, 255],
                      [130, 235], [118, 195], [135, 165], [150, 175]], np.int32)
    cv2.fillPoly(img, [flame], (0, 0, 0))
    return img


_tmpl = {}


def load_templates():
    if _tmpl:
        return _tmpl
    sheet_path = os.path.join(ASSETS, 'hazard_sheet.jpg')
    if os.path.exists(sheet_path):
        sheet = cv2.imread(sheet_path)
        tris = find_triangles(sheet, min_area=1500)
        tris.sort(key=lambda t: (round(t['bbox'][1] / 120), t['bbox'][0]))
        if len(tris) == 10:
            for name, t in zip(SHEET_NAMES, tris):
                _tmpl[name] = pictogram(sheet, t['pts'])
    fl = _flame_template()
    tr = find_triangles(fl, min_area=1500)
    if tr:
        _tmpl['flammable'] = pictogram(fl, tr[0]['pts'])
    return _tmpl


def classify(bgr, pts):
    """-> (label or None, score, margin, best_name). label only for VALID classes."""
    T = load_templates()
    if not T:
        return None, 0.0, 0.0, None
    g = pictogram(bgr, pts)
    if g.sum() < 20:
        return None, 0.0, 0.0, None
    scores = sorted(((_score(g, v), k) for k, v in T.items()), reverse=True)
    best, bn = scores[0]
    second = scores[1][0] if len(scores) > 1 else 0.0
    label = bn if bn in VALID else None
    return label, best, best - second, bn


def detect_hazards(bgr):
    """Detect and classify all yellow triangles. Returns list of dicts."""
    res = []
    for t in find_triangles(bgr):
        label, sc, mg, bn = classify(bgr, t['pts'])
        x, y, w, h = t['bbox']
        res.append({'label': label, 'score': sc, 'margin': mg, 'best': bn,
                    'bbox': (x, y, w, h), 'cx': x + w / 2.0, 'cy': y + h / 2.0})
    return res
