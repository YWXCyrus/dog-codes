"""Pure-python QR decoder (versions 1-10, all EC levels, numeric/alphanumeric/byte modes).

Why: Ubuntu 22.04's OpenCV (4.5.4) can LOCATE QR codes but its decoder is not linked
("Library QUIRC is not linked"), and there is no pip/zbar on the dog container.
cv2 is only used to find the symbol corners and to rectify it; everything after that is here:
module sampling -> format info -> unmask -> de-interleave -> Reed-Solomon -> bit stream.
"""
import numpy as np

# (ecc codewords per block, [(n_blocks, data_cw_per_block), ...]) for L, M, Q, H
EC_TABLE = {
    1: {'L': (7, [(1, 19)]), 'M': (10, [(1, 16)]), 'Q': (13, [(1, 13)]), 'H': (17, [(1, 9)])},
    2: {'L': (10, [(1, 34)]), 'M': (16, [(1, 28)]), 'Q': (22, [(1, 22)]), 'H': (28, [(1, 16)])},
    3: {'L': (15, [(1, 55)]), 'M': (26, [(1, 44)]), 'Q': (18, [(2, 17)]), 'H': (22, [(2, 13)])},
    4: {'L': (20, [(1, 80)]), 'M': (18, [(2, 32)]), 'Q': (26, [(2, 24)]), 'H': (16, [(4, 9)])},
    5: {'L': (26, [(1, 108)]), 'M': (24, [(2, 43)]), 'Q': (18, [(2, 15), (2, 16)]),
        'H': (22, [(2, 11), (2, 12)])},
    6: {'L': (18, [(2, 68)]), 'M': (16, [(4, 27)]), 'Q': (24, [(4, 19)]), 'H': (28, [(4, 15)])},
    7: {'L': (20, [(2, 78)]), 'M': (18, [(4, 31)]), 'Q': (18, [(2, 14), (4, 15)]),
        'H': (26, [(4, 13), (1, 14)])},
    8: {'L': (24, [(2, 97)]), 'M': (22, [(2, 38), (2, 39)]), 'Q': (22, [(4, 18), (2, 19)]),
        'H': (26, [(4, 14), (2, 15)])},
    9: {'L': (30, [(2, 116)]), 'M': (22, [(3, 36), (2, 37)]), 'Q': (20, [(4, 16), (4, 17)]),
        'H': (24, [(4, 12), (4, 13)])},
    10: {'L': (18, [(2, 68), (2, 69)]), 'M': (26, [(4, 43), (1, 44)]),
         'Q': (24, [(6, 19), (2, 20)]), 'H': (28, [(6, 15), (2, 16)])},
}
ALIGN = {1: [], 2: [6, 18], 3: [6, 22], 4: [6, 26], 5: [6, 30], 6: [6, 34], 7: [6, 22, 38],
         8: [6, 24, 42], 9: [6, 26, 46], 10: [6, 28, 50]}
EC_BITS = {'L': 1, 'M': 0, 'Q': 3, 'H': 2}
ALNUM = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:'

# ---- GF(256), primitive polynomial 0x11D
EXP = [0] * 512
LOG = [0] * 256
_x = 1
for _i in range(255):
    EXP[_i] = _x
    LOG[_x] = _i
    _x <<= 1
    if _x & 0x100:
        _x ^= 0x11D
for _i in range(255, 512):
    EXP[_i] = EXP[_i - 255]


def gmul(a, b):
    return 0 if a == 0 or b == 0 else EXP[LOG[a] + LOG[b]]


def gdiv(a, b):
    return 0 if a == 0 else EXP[(LOG[a] - LOG[b]) % 255]


def poly_eval(p, x):
    """p: coefficients, highest degree first."""
    y = p[0]
    for c in p[1:]:
        y = gmul(y, x) ^ c
    return y


def rs_correct(block, ecc):
    """Correct a codeword block (data+ecc, data first). Returns corrected list or None."""
    n = len(block)
    synd = [poly_eval(block, EXP[j]) for j in range(ecc)]
    if not any(synd):
        return list(block)
    # Berlekamp-Massey
    C = [1]
    B = [1]
    L, m, b = 0, 1, 1
    for i in range(ecc):
        d = synd[i]
        for j in range(1, L + 1):
            if j < len(C):
                d ^= gmul(C[j], synd[i - j])
        if d == 0:
            m += 1
        elif 2 * L <= i:
            T = list(C)
            coef = gdiv(d, b)
            C = C + [0] * (len(B) + m - len(C))
            for j in range(len(B)):
                C[j + m] ^= gmul(coef, B[j])
            L, B, b, m = i + 1 - L, T, d, 1
        else:
            coef = gdiv(d, b)
            C = C + [0] * (len(B) + m - len(C))
            for j in range(len(B)):
                C[j + m] ^= gmul(coef, B[j])
            m += 1
    if L * 2 > ecc:
        return None
    # Chien search: error positions (index from the START of block)
    locs = []
    for pos in range(n):
        pw = n - 1 - pos
        s_ = 0
        for j, c in enumerate(C[:L + 1]):
            s_ ^= gmul(c, EXP[(-pw * j) % 255])
        if s_ == 0:
            locs.append(pos)
    if len(locs) != L:
        return None
    # Forney: omega = S(x)*C(x) mod x^ecc
    omega = [0] * ecc
    for i in range(ecc):
        for j in range(min(i, len(C) - 1) + 1):
            omega[i] ^= gmul(C[j], synd[i - j])
    out = list(block)
    for pos in locs:
        pw = n - 1 - pos
        xinv = EXP[(255 - pw) % 255]                     # X_k^-1
        num = 0
        for i, o in enumerate(omega):
            num ^= gmul(o, EXP[(LOG[xinv] * i) % 255] if xinv else 0)
        den = 0
        for j in range(1, len(C), 2):                    # derivative of C (odd terms)
            den ^= gmul(C[j], EXP[(LOG[xinv] * (j - 1)) % 255])
        if den == 0:
            return None
        mag = gmul(EXP[pw % 255], gdiv(num, den)) if num else 0
        out[pos] ^= mag
    if any(poly_eval(out, EXP[j]) for j in range(ecc)):
        return None
    return out


# ---- format info
def _bch_format(data5):
    rem = data5
    for _ in range(10):
        rem = (rem << 1) ^ ((rem >> 9) * 0x537)
    return ((data5 << 10) | (rem & 0x3FF)) ^ 0x5412


FORMATS = {}
for _lvl, _eb in EC_BITS.items():
    for _mask in range(8):
        FORMATS[_bch_format((_eb << 3) | _mask)] = (_lvl, _mask)


def read_format(m):
    size = m.shape[0]
    def bit(x, y):
        return int(m[y, x])
    b1 = 0
    for i in range(6):
        b1 |= bit(8, i) << i
    b1 |= bit(8, 7) << 6
    b1 |= bit(8, 8) << 7
    b1 |= bit(7, 8) << 8
    for i in range(9, 15):
        b1 |= bit(14 - i, 8) << i
    b2 = 0
    for i in range(8):
        b2 |= bit(size - 1 - i, 8) << i
    for i in range(8, 15):
        b2 |= bit(8, size - 15 + i) << i
    best = None
    for cand in (b1, b2):
        for code, val in FORMATS.items():
            d = bin(code ^ cand).count('1')
            if d <= 3 and (best is None or d < best[0]):
                best = (d, val)
    return None if best is None else best[1]


def function_mask(version):
    size = 17 + 4 * version
    f = np.zeros((size, size), bool)
    f[0:9, 0:9] = True
    f[0:9, size - 8:] = True
    f[size - 8:, 0:9] = True
    f[6, :] = True
    f[:, 6] = True
    cs = ALIGN[version]
    for cy in cs:
        for cx in cs:
            if (cx == 6 and cy == 6) or (cx == 6 and cy == cs[-1]) or (cx == cs[-1] and cy == 6):
                continue
            f[cy - 2:cy + 3, cx - 2:cx + 3] = True
    if version >= 7:
        f[0:6, size - 11:size - 8] = True
        f[size - 11:size - 8, 0:6] = True
    return f


def unmask(m, mask, func):
    size = m.shape[0]
    ii, jj = np.mgrid[0:size, 0:size]
    if mask == 0:
        g = (ii + jj) % 2 == 0
    elif mask == 1:
        g = ii % 2 == 0
    elif mask == 2:
        g = jj % 3 == 0
    elif mask == 3:
        g = (ii + jj) % 3 == 0
    elif mask == 4:
        g = (ii // 2 + jj // 3) % 2 == 0
    elif mask == 5:
        g = (ii * jj) % 2 + (ii * jj) % 3 == 0
    elif mask == 6:
        g = ((ii * jj) % 2 + (ii * jj) % 3) % 2 == 0
    else:
        g = ((ii + jj) % 2 + (ii * jj) % 3) % 2 == 0
    out = m.copy()
    out[g & ~func] ^= True
    return out


def read_codewords(m, func):
    size = m.shape[0]
    bits = []
    right = size - 1
    while right >= 1:
        if right == 6:
            right = 5
        upward = ((right + 1) & 2) == 0
        for vert in range(size):
            y = size - 1 - vert if upward else vert
            for j in range(2):
                x = right - j
                if not func[y, x]:
                    bits.append(int(m[y, x]))
        right -= 2
    cw = []
    for i in range(0, len(bits) - 7, 8):
        v = 0
        for b in bits[i:i + 8]:
            v = (v << 1) | b
        cw.append(v)
    return cw


def decode_matrix(m):
    """m: (size,size) bool, True = dark. Returns text or None."""
    size = m.shape[0]
    if (size - 17) % 4 or not (21 <= size <= 57):
        return None
    version = (size - 17) // 4
    fmt = read_format(m)
    if fmt is None:
        return None
    level, mask = fmt
    func = function_mask(version)
    cw = read_codewords(unmask(m, mask, func), func)
    ecc, groups = EC_TABLE[version][level]
    nblocks = sum(n for n, _ in groups)
    total = sum(n * (d + ecc) for n, d in groups)
    if len(cw) < total:
        return None
    blocks = []
    for n, d in groups:
        for _ in range(n):
            blocks.append([d, [], []])
    maxd = max(d for _, d in groups)
    idx = 0
    for i in range(maxd):
        for b in blocks:
            if i < b[0]:
                b[1].append(cw[idx])
                idx += 1
    for i in range(ecc):
        for b in blocks:
            b[2].append(cw[idx])
            idx += 1
    data = []
    for d, dc, ec in blocks:
        fixed = rs_correct(dc + ec, ecc)
        if fixed is None:
            return None
        data += fixed[:d]
    return parse_bits(data, version)


def parse_bits(data, version):
    bits = ''.join('{:08b}'.format(x) for x in data)
    pos = 0
    out = bytearray()
    cc = {'num': 10 if version < 10 else 12, 'alnum': 9 if version < 10 else 11,
          'byte': 8 if version < 10 else 16}
    while pos + 4 <= len(bits):
        mode = bits[pos:pos + 4]
        pos += 4
        if mode == '0000':
            break
        if mode == '0001':
            n = int(bits[pos:pos + cc['num']], 2)
            pos += cc['num']
            s = ''
            while n >= 3:
                s += '%03d' % int(bits[pos:pos + 10], 2)
                pos += 10
                n -= 3
            if n == 2:
                s += '%02d' % int(bits[pos:pos + 7], 2)
                pos += 7
            elif n == 1:
                s += '%d' % int(bits[pos:pos + 4], 2)
                pos += 4
            out += s.encode()
        elif mode == '0010':
            n = int(bits[pos:pos + cc['alnum']], 2)
            pos += cc['alnum']
            s = ''
            while n >= 2:
                v = int(bits[pos:pos + 11], 2)
                pos += 11
                s += ALNUM[v // 45] + ALNUM[v % 45]
                n -= 2
            if n == 1:
                s += ALNUM[int(bits[pos:pos + 6], 2)]
                pos += 6
            out += s.encode()
        elif mode == '0100':
            n = int(bits[pos:pos + cc['byte']], 2)
            pos += cc['byte']
            for _ in range(n):
                out.append(int(bits[pos:pos + 8], 2))
                pos += 8
        elif mode == '0111':          # ECI designator: skip 8 bits (common case)
            pos += 8
        else:
            return None
    try:
        return out.decode('utf-8')
    except UnicodeDecodeError:
        return out.decode('latin-1')


# ---- rectification / sampling
def _finder_score(m):
    size = m.shape[0]
    pat = np.array([[1, 1, 1, 1, 1, 1, 1], [1, 0, 0, 0, 0, 0, 1], [1, 0, 1, 1, 1, 0, 1],
                    [1, 0, 1, 1, 1, 0, 1], [1, 0, 1, 1, 1, 0, 1], [1, 0, 0, 0, 0, 0, 1],
                    [1, 1, 1, 1, 1, 1, 1]], bool)
    s = 0
    for (r, c) in ((0, 0), (0, size - 7), (size - 7, 0)):
        s += int((m[r:r + 7, c:c + 7] == pat).sum())
    return s / (3 * 49.0)


def sample_grid(gray, corners, version, margin=0.0):
    """Warp the quad to a (d*8)^2 image and sample module centres. corners: TL,TR,BR,BL (4,2)."""
    import cv2
    d = 17 + 4 * version
    S = 8
    W = d * S
    dst = np.float32([[margin * S, margin * S], [W - margin * S, margin * S],
                      [W - margin * S, W - margin * S], [margin * S, W - margin * S]])
    M = cv2.getPerspectiveTransform(np.float32(corners), dst)
    warp = cv2.warpPerspective(gray, M, (W, W), flags=cv2.INTER_LINEAR)
    thr = cv2.threshold(warp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    cell = warp.reshape(d, S, d, S)[:, 2:6, :, 2:6].mean(axis=(1, 3))
    return cell < thr


def decode_quad(gray, quad):
    """Try versions 1..10 and the 4 corner rotations; return text or None."""
    quad = np.asarray(quad, np.float32).reshape(4, 2)
    for rot in range(4):
        q = np.roll(quad, -rot, axis=0)
        for version in range(1, 11):
            for margin in (0.0, -0.4, 0.4):
                try:
                    m = sample_grid(gray, q, version, margin)
                except Exception:
                    continue
                if _finder_score(m) < 0.88:
                    continue
                txt = decode_matrix(m)
                if txt:
                    return txt
    return None
