"""
"Ember Eye" - a deterministic generative artwork.

Concept
-------
A single Clifford strange attractor is iterated several million times to
build up a dense point-cloud whose trajectory loops back on itself into a
near-circular weave of ribbons - like a glowing iris or an event horizon
seen from above. The point-cloud is rasterised into a density field, which
is then mapped through a custom "deep sea / molten ember" colour gradient:
near-black indigo in the sparse regions, rising through teal, magenta,
ember-orange and finally a hot gold-white core wherever the attractor's
trajectory folds back on itself most densely. A soft bloom pass and a
gentle vignette are added to give the filaments a glowing, dimensional
feel rather than a flat mathematical plot.

Everything is fixed/seeded so the program is fully deterministic: the same
run always produces a byte-identical PNG.

Implementation notes
---------------------
This implementation uses ONLY the Python standard library (math, os,
struct, zlib, itertools, array) - no Pillow, no numpy, no third-party
packages of any kind, so it runs in a bare "stdlib only" sandbox. The PNG
file itself is hand-encoded: IHDR/IDAT/IEND chunks with CRC32 checksums,
zlib-compressed scanline data with a filter byte per row. Dense buffers
use the `array` module (typed, contiguous) rather than lists of Python
objects, to keep memory use low and predictable.
"""

import math
import os
import struct
import zlib
from array import array
from itertools import accumulate

# ---------------------------------------------------------------------------
# Deterministic configuration
# ---------------------------------------------------------------------------
SIZE = 768                 # final PNG is SIZE x SIZE (<= 1024 cap)
N_POINTS = 8_000_000       # attractor iterations kept (after burn-in)
BURN_IN = 2_000

# Clifford attractor parameters - chosen for a dense, roughly circular
# weave of looping ribbons that fills the frame edge-to-edge, evoking a
# glowing iris / event horizon rather than a sparse wispy plot.
A, B, C, D = 1.5, -1.8, 1.7, 0.9
X0, Y0 = 0.1, 0.1  # fixed starting point -> fully deterministic trajectory

OUT_DIR = "output"
OUT_PATH = os.path.join(OUT_DIR, "entry.png")

# "Ember Tide" palette: deep indigo -> teal -> cyan glow -> magenta ->
# ember orange -> gold -> near-white hot core.
GRADIENT_STOPS = [0.00, 0.12, 0.30, 0.50, 0.68, 0.82, 0.93, 1.00]
GRADIENT_COLORS = [
    (3, 4, 14),
    (10, 14, 46),
    (18, 68, 118),
    (0, 172, 172),
    (181, 40, 160),
    (255, 110, 40),
    (255, 210, 90),
    (255, 250, 230),
]

GAMMA = 0.42
BLOOM_RADIUS = 9
BLOOM_STRENGTH = 0.9
VIGNETTE_STRENGTH = 0.22


# ---------------------------------------------------------------------------
# Attractor iteration
# ---------------------------------------------------------------------------
def iterate_clifford(n_points, burn_in):
    """Iterate the Clifford map and return the (x, y) trajectory as typed
    arrays. Pure-python loop (each step depends on the previous one, so it
    cannot be vectorised) but kept tight for speed; deterministic given
    fixed parameters and starting point.
    """
    sin = math.sin
    cos = math.cos
    a, b, c, d = A, B, C, D

    x, y = X0, Y0
    for _ in range(burn_in):
        x, y = sin(a * y) + c * cos(a * x), sin(b * x) + d * cos(b * y)

    xs = array('d', bytes(8 * n_points))
    ys = array('d', bytes(8 * n_points))
    for i in range(n_points):
        x, y = sin(a * y) + c * cos(a * x), sin(b * x) + d * cos(b * y)
        xs[i] = x
        ys[i] = y
    return xs, ys


def build_density(xs, ys, size):
    """Rasterise the point cloud into a 2D histogram (flat typed array)."""
    n = len(xs)
    xmin = min(xs)
    xmax = max(xs)
    ymin = min(ys)
    ymax = max(ys)

    # small symmetric padding so the attractor doesn't touch the frame edge
    pad = 0.04
    xr = xmax - xmin
    yr = ymax - ymin
    x0b = xmin - xr * pad
    y0b = ymin - yr * pad
    xscale = size / (xr * (1.0 + 2.0 * pad))
    yscale = size / (yr * (1.0 + 2.0 * pad))

    hist = array('I', bytes(4 * size * size))
    wm1 = size - 1
    hm1 = size - 1
    for i in range(n):
        ix = int((xs[i] - x0b) * xscale)
        iy = int((ys[i] - y0b) * yscale)
        if ix < 0:
            ix = 0
        elif ix > wm1:
            ix = wm1
        if iy < 0:
            iy = 0
        elif iy > hm1:
            iy = hm1
        hist[iy * size + ix] += 1
    return hist


def shape_density(hist, size):
    """log1p + normalise + gamma shape the density field into [0, 1]."""
    n = size * size
    log1p = math.log1p
    vals = array('d', bytes(8 * n))
    mx = 0.0
    for i in range(n):
        v = log1p(hist[i])
        vals[i] = v
        if v > mx:
            mx = v
    if mx <= 0.0:
        mx = 1.0
    inv = 1.0 / mx
    g = GAMMA
    for i in range(n):
        t = vals[i] * inv
        vals[i] = t ** g
    return vals


# ---------------------------------------------------------------------------
# Colour mapping
# ---------------------------------------------------------------------------
def build_gradient_lut(steps=4096):
    stops = GRADIENT_STOPS
    colors = GRADIENT_COLORS
    lut = [(0.0, 0.0, 0.0)] * (steps + 1)
    seg = 0
    last_seg = len(stops) - 2
    for i in range(steps + 1):
        t = i / steps
        while seg < last_seg and t > stops[seg + 1]:
            seg += 1
        t0, t1 = stops[seg], stops[seg + 1]
        c0, c1 = colors[seg], colors[seg + 1]
        frac = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
        r = c0[0] + (c1[0] - c0[0]) * frac
        g = c0[1] + (c1[1] - c0[1]) * frac
        b = c0[2] + (c1[2] - c0[2]) * frac
        lut[i] = (r, g, b)
    return lut


def apply_colormap(shaped, size, lut, steps):
    n = size * size
    r = array('d', bytes(8 * n))
    g = array('d', bytes(8 * n))
    b = array('d', bytes(8 * n))
    for i in range(n):
        idx = int(shaped[i] * steps)
        if idx < 0:
            idx = 0
        elif idx > steps:
            idx = steps
        cr, cg, cb = lut[idx]
        r[i] = cr
        g[i] = cg
        b[i] = cb
    return r, g, b


def bright_mask(shaped, size):
    """clip((shaped-0.55)/0.45, 0, 1) ** 1.5, used to pick out the glowing
    core regions that will bloom outward."""
    n = size * size
    m = array('d', bytes(8 * n))
    sqrt = math.sqrt
    for i in range(n):
        v = (shaped[i] - 0.55) / 0.45
        if v < 0.0:
            v = 0.0
        elif v > 1.0:
            v = 1.0
        m[i] = v * sqrt(v)  # v ** 1.5
    return m


# ---------------------------------------------------------------------------
# Box blur (fast separable approximation of a gaussian blur)
# ---------------------------------------------------------------------------
def _box_blur_horizontal(data, size, radius):
    out = array('d', bytes(8 * size * size))
    win = 2 * radius + 1
    pad_arr_proto = array('d', [0.0])
    for row in range(size):
        base = row * size
        line = data[base:base + size]
        first = line[0]
        last = line[size - 1]
        padded = array('d', [first]) * radius + line + array('d', [last]) * radius
        prefix = list(accumulate(padded, initial=0.0))
        for i in range(size):
            out[base + i] = (prefix[i + win] - prefix[i]) / win
    return out


def _box_blur_vertical(data, size, radius):
    out = array('d', bytes(8 * size * size))
    win = 2 * radius + 1
    for col in range(size):
        colvals = array('d', bytes(8 * size))
        for row in range(size):
            colvals[row] = data[row * size + col]
        first = colvals[0]
        last = colvals[size - 1]
        padded = array('d', [first]) * radius + colvals + array('d', [last]) * radius
        prefix = list(accumulate(padded, initial=0.0))
        for row in range(size):
            out[row * size + col] = (prefix[row + win] - prefix[row]) / win
    return out


def box_blur(data, size, radius):
    h = _box_blur_horizontal(data, size, radius)
    return _box_blur_vertical(h, size, radius)


# ---------------------------------------------------------------------------
# Compositing
# ---------------------------------------------------------------------------
def build_bloom_source(r, g, b, mask, size):
    n = size * size
    br = array('d', bytes(8 * n))
    bg = array('d', bytes(8 * n))
    bb = array('d', bytes(8 * n))
    for i in range(n):
        m = mask[i]
        br[i] = r[i] * m
        bg[i] = g[i] * m
        bb[i] = b[i] * m
    return br, bg, bb


def screen_combine(base_r, base_g, base_b, bloom_r, bloom_g, bloom_b, size,
                    strength=BLOOM_STRENGTH):
    n = size * size
    for i in range(n):
        base_r[i] = 255.0 - (255.0 - base_r[i]) * (255.0 - bloom_r[i] * strength) / 255.0
        base_g[i] = 255.0 - (255.0 - base_g[i]) * (255.0 - bloom_g[i] * strength) / 255.0
        base_b[i] = 255.0 - (255.0 - base_b[i]) * (255.0 - bloom_b[i] * strength) / 255.0
    return base_r, base_g, base_b


def apply_vignette(r, g, b, size, strength=VIGNETTE_STRENGTH):
    cx = cy = (size - 1) / 2.0
    half = size / 2.0

    dx2 = [0.0] * size
    for x in range(size):
        dxn = (x - cx) / half
        dx2[x] = dxn * dxn
    dy2 = [0.0] * size
    for y in range(size):
        dyn = (y - cy) / half
        dy2[y] = dyn * dyn

    inv2 = 0.5  # dividing r by sqrt(2) before clipping == dividing r^2 by 2
    for y in range(size):
        rowbase = y * size
        dyy = dy2[y]
        for x in range(size):
            r2 = (dx2[x] + dyy) * inv2
            if r2 > 1.0:
                r2 = 1.0
            falloff = 1.0 - strength * (r2 ** 1.1)
            idx = rowbase + x
            r[idx] *= falloff
            g[idx] *= falloff
            b[idx] *= falloff
    return r, g, b


def to_rgb_bytes(r, g, b, size):
    n = size * size
    buf = bytearray(n * 3)
    for i in range(n):
        rv = r[i]
        gv = g[i]
        bv = b[i]
        if rv < 0.0:
            rv = 0.0
        elif rv > 255.0:
            rv = 255.0
        if gv < 0.0:
            gv = 0.0
        elif gv > 255.0:
            gv = 255.0
        if bv < 0.0:
            bv = 0.0
        elif bv > 255.0:
            bv = 255.0
        j = i * 3
        buf[j] = int(rv)
        buf[j + 1] = int(gv)
        buf[j + 2] = int(bv)
    return buf


# ---------------------------------------------------------------------------
# Hand-rolled PNG encoder (stdlib only: struct + zlib + binascii-equivalent
# CRC32 via zlib.crc32)
# ---------------------------------------------------------------------------
def _paeth_predictor(a, b, c):
    # a = left, b = above, c = upper-left
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def _score(buf):
    # Sum of absolute values when each byte is interpreted as a signed
    # delta (standard PNG filter-selection heuristic: smaller is better).
    s = 0
    for v in buf:
        s += v if v < 128 else 256 - v
    return s


def _filter_variants(raw, width, height, bpp=3):
    """Build several whole-image filtered variants in a single pass over
    the scanlines: all-None, all-Up, all-Paeth, and a per-row adaptive
    choice (minimising the standard sum-of-absolute-deltas heuristic).
    Returns a dict of {name: filtered_bytes}. The caller actually
    zlib-compresses each candidate and keeps whichever compresses
    smallest - the heuristic is only a per-row proxy and does not always
    predict the final deflate size, so trying a few whole-image strategies
    and measuring the real output is more reliable while still being
    fully deterministic (it depends only on the pixel data)."""
    stride = width * bpp
    row_out_len = 1 + stride

    none_buf = bytearray(height * row_out_len)
    up_buf = bytearray(height * row_out_len)
    paeth_buf = bytearray(height * row_out_len)
    adaptive_buf = bytearray(height * row_out_len)

    prev = bytes(stride)
    pos = 0
    for y in range(height):
        row = raw[y * stride:(y + 1) * stride]

        none_f = row

        up_f = bytearray(stride)
        for i in range(stride):
            up_f[i] = (row[i] - prev[i]) & 0xFF

        if y == 0:
            # First row has no "above" neighbour: Sub/Average/Paeth all
            # degrade to using only the left neighbour (up == 0).
            sub_f = bytearray(stride)
            for i in range(stride):
                left = row[i - bpp] if i >= bpp else 0
                sub_f[i] = (row[i] - left) & 0xFF
            paeth_f = sub_f
        else:
            paeth_f = bytearray(stride)
            for i in range(stride):
                left = row[i - bpp] if i >= bpp else 0
                up = prev[i]
                upleft = prev[i - bpp] if i >= bpp else 0
                paeth_f[i] = (row[i] - _paeth_predictor(left, up, upleft)) & 0xFF

        if y == 0:
            adaptive_type, adaptive_f = 0, none_f
        else:
            candidates = ((0, none_f), (2, up_f), (4, paeth_f))
            adaptive_type, adaptive_f = min(candidates, key=lambda c: _score(c[1]))

        none_buf[pos] = 0
        none_buf[pos + 1:pos + 1 + stride] = none_f
        up_buf[pos] = 0 if y == 0 else 2
        up_buf[pos + 1:pos + 1 + stride] = none_f if y == 0 else up_f
        paeth_buf[pos] = 1 if y == 0 else 4
        paeth_buf[pos + 1:pos + 1 + stride] = paeth_f
        adaptive_buf[pos] = adaptive_type
        adaptive_buf[pos + 1:pos + 1 + stride] = adaptive_f

        prev = row
        pos += row_out_len

    return {
        "none": bytes(none_buf),
        "up": bytes(up_buf),
        "paeth": bytes(paeth_buf),
        "adaptive": bytes(adaptive_buf),
    }


def _chunk(tag, data):
    return (
        struct.pack('>I', len(data))
        + tag
        + data
        + struct.pack('>I', zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def write_png(path, width, height, rgb_bytes):
    variants = _filter_variants(rgb_bytes, width, height, 3)
    # Deterministic tie-break: iterate variant names in a fixed order and
    # keep strict "<" so the first smallest-seen candidate wins regardless
    # of dict iteration order.
    best_name = None
    compressed = None
    for name in ("none", "up", "paeth", "adaptive"):
        candidate = zlib.compress(variants[name], 9)
        if compressed is None or len(candidate) < len(compressed):
            compressed = candidate
            best_name = name

    sig = b'\x89PNG\r\n\x1a\n'
    ihdr = struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)

    with open(path, 'wb') as f:
        f.write(sig)
        f.write(_chunk(b'IHDR', ihdr))
        f.write(_chunk(b'IDAT', compressed))
        f.write(_chunk(b'IEND', b''))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    xs, ys = iterate_clifford(N_POINTS, BURN_IN)
    hist = build_density(xs, ys, SIZE)
    del xs, ys

    shaped = shape_density(hist, SIZE)
    del hist

    lut = build_gradient_lut()
    r, g, b = apply_colormap(shaped, SIZE, lut, len(lut) - 1)
    mask = bright_mask(shaped, SIZE)
    del shaped

    bloom_r, bloom_g, bloom_b = build_bloom_source(r, g, b, mask, SIZE)
    del mask
    bloom_r = box_blur(bloom_r, SIZE, BLOOM_RADIUS)
    bloom_g = box_blur(bloom_g, SIZE, BLOOM_RADIUS)
    bloom_b = box_blur(bloom_b, SIZE, BLOOM_RADIUS)

    r, g, b = screen_combine(r, g, b, bloom_r, bloom_g, bloom_b, SIZE)
    del bloom_r, bloom_g, bloom_b

    r, g, b = apply_vignette(r, g, b, SIZE)

    rgb_bytes = to_rgb_bytes(r, g, b, SIZE)
    del r, g, b

    write_png(OUT_PATH, SIZE, SIZE, rgb_bytes)


if __name__ == "__main__":
    main()
