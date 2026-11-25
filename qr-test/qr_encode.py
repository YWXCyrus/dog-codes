"""Minimal QR encoder, for TESTING the decoder only (byte mode, versions 1-6, EC L/M/Q/H, mask 0-7).

Not used on the dog during a run. Written because the dog container has no QR generator
(no python3-qrcode, OpenCV 4.5.4 has no QRCodeEncoder). Tables come from the mission decoder
(frontier-fastlio/dog_codes/qr_decode.py) so the two stay in step; phone QR apps reading these
codes is an independent check that both follow the standard.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'frontier-fastlio', 'dog_codes'))
import qr_decode as Q                                         # noqa: E402


def _rs_generator(n):
    g = [1]
    for i in range(n):
        out = [0] * (len(g) + 1)
        for j, c in enumerate(g):                         # g(x) * (x - a^i), highest degree first
            out[j] ^= c
            out[j + 1] ^= Q.gmul(c, Q.EXP[i])
        g = out
    return g


def _rs_ecc(data, n):
    g = _rs_generator(n)
    rem = list(data) + [0] * n
    for i in range(len(data)):
        c = rem[i]
        if c:
            for j in range(1, len(g)):
                rem[i + j] ^= Q.gmul(g[j], c)
    return rem[len(data):]


def _capacity(version, level):
    ecc, groups = Q.EC_TABLE[version][level]
    return sum(n * d for n, d in groups)


def choose_version(nbytes, level):
    for v in range(1, 7):
        if 4 + 8 + 8 * nbytes <= 8 * _capacity(v, level):
            return v
    raise ValueError('text too long for versions 1-6 at level %s' % level)


def encode(text, level='M', mask=2, version=None):
    """Return the module matrix (bool, True = dark) for `text` in byte mode."""
    data = text.encode('utf-8')
    v = version or choose_version(len(data), level)
    size = 17 + 4 * v
    ecc, groups = Q.EC_TABLE[v][level]
    cap = _capacity(v, level)
    bits = '0100' + '{:08b}'.format(len(data)) + ''.join('{:08b}'.format(b) for b in data)
    bits += '0' * min(4, 8 * cap - len(bits))
    bits += '0' * (-len(bits) % 8)
    cw = [int(bits[i:i + 8], 2) for i in range(0, len(bits), 8)]
    pad = [0xEC, 0x11]
    while len(cw) < cap:
        cw.append(pad[(len(cw) - len(bits) // 8) % 2])
    # blocks, ECC, interleave
    blocks, k = [], 0
    for n, d in groups:
        for _ in range(n):
            blocks.append(cw[k:k + d])
            k += d
    eccs = [_rs_ecc(b, ecc) for b in blocks]
    out = []
    for i in range(max(len(b) for b in blocks)):
        out += [b[i] for b in blocks if i < len(b)]
    for i in range(ecc):
        out += [e[i] for e in eccs]
    stream = ''.join('{:08b}'.format(c) for c in out)

    m = np.zeros((size, size), bool)
    func = Q.function_mask(v)
    # finders (+ separators stay light)
    for r, c in ((0, 0), (0, size - 7), (size - 7, 0)):
        m[r:r + 7, c:c + 7] = True
        m[r + 1:r + 6, c + 1:c + 6] = False
        m[r + 2:r + 5, c + 2:c + 5] = True
    # timing
    for i in range(8, size - 8):
        m[6, i] = m[i, 6] = (i % 2 == 0)
    # alignment
    cs = Q.ALIGN[v]
    for cy in cs:
        for cx in cs:
            if (cx == 6 and cy == 6) or (cx == 6 and cy == cs[-1]) or (cx == cs[-1] and cy == 6):
                continue
            m[cy - 2:cy + 3, cx - 2:cx + 3] = True
            m[cy - 1:cy + 2, cx - 1:cx + 2] = False
            m[cy, cx] = True
    m[size - 8, 8] = True                                  # dark module
    # data, same zigzag as qr_decode.read_codewords
    bi = 0
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
                    m[y, x] = bi < len(stream) and stream[bi] == '1'
                    bi += 1
        right -= 2
    m = Q.unmask(m, mask, func)                            # XOR mask (its own inverse)
    # format info, same positions as qr_decode.read_format (both copies)
    fmt = Q._bch_format((Q.EC_BITS[level] << 3) | mask)
    b = [(fmt >> i) & 1 for i in range(15)]
    for i in range(6):
        m[i, 8] = b[i]
    m[7, 8], m[8, 8], m[8, 7] = b[6], b[7], b[8]
    for i in range(9, 15):
        m[8, 14 - i] = b[i]
    for i in range(8):
        m[8, size - 1 - i] = b[i]
    for i in range(8, 15):
        m[size - 15 + i, 8] = b[i]
    m[size - 8, 8] = True
    return m


def render(m, module_px=8, quiet=4):
    """Module matrix -> grey uint8 image (white quiet zone)."""
    size = m.shape[0]
    img = np.full(((size + 2 * quiet) * module_px,) * 2, 255, np.uint8)
    big = np.kron(m.astype(np.uint8), np.ones((module_px, module_px), np.uint8))
    o = quiet * module_px
    img[o:o + size * module_px, o:o + size * module_px][big > 0] = 0
    return img
