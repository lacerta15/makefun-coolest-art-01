"""
"Ember Eye" - a deterministic generative artwork.

Concept
-------
A single Clifford strange attractor is iterated many millions of times to
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
"""

import math
import os

import numpy as np
from PIL import Image, ImageFilter

# ----------------------------------------------------------------------------
# Deterministic configuration
# ----------------------------------------------------------------------------
FINAL_SIZE = 1024          # final PNG is FINAL_SIZE x FINAL_SIZE
SUPERSAMPLE = 1536         # render larger, then downsample for smooth edges
N_POINTS = 5_200_000       # attractor iterations kept (after burn-in)
BURN_IN = 2_000

# Clifford attractor parameters - chosen (after comparing several candidate
# parameter sets) for a dense, roughly circular weave of looping ribbons
# that fills the frame edge-to-edge, evoking a glowing iris / event horizon
# rather than a sparse wispy plot.
A, B, C, D = 1.5, -1.8, 1.7, 0.9

OUT_DIR = "output"
OUT_PATH = os.path.join(OUT_DIR, "entry.png")


def iterate_clifford(a, b, c, d, x0, y0, n_points, burn_in):
    """Iterate the Clifford map and return the (x, y) trajectory as arrays.

    Pure-python loop (not vectorisable because each step depends on the
    previous one) but kept tight for speed; deterministic given fixed
    parameters and starting point.
    """
    xs = np.empty(n_points, dtype=np.float64)
    ys = np.empty(n_points, dtype=np.float64)

    sin = math.sin
    cos = math.cos

    x, y = x0, y0
    for _ in range(burn_in):
        x, y = sin(a * y) + c * cos(a * x), sin(b * x) + d * cos(b * y)

    for i in range(n_points):
        x, y = sin(a * y) + c * cos(a * x), sin(b * x) + d * cos(b * y)
        xs[i] = x
        ys[i] = y

    return xs, ys


def build_density_field(xs, ys, resolution):
    """Rasterise the point cloud into a log-scaled, normalised density grid."""
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    # small symmetric padding so the attractor doesn't touch the frame edge
    pad_x = (x_max - x_min) * 0.04
    pad_y = (y_max - y_min) * 0.04

    hist, _, _ = np.histogram2d(
        xs,
        ys,
        bins=resolution,
        range=[[x_min - pad_x, x_max + pad_x], [y_min - pad_y, y_max + pad_y]],
    )

    hist = np.log1p(hist)
    hist /= hist.max()
    hist = hist.T  # transpose so first axis is rows (y), second is cols (x)

    # A very light smoothing removes per-bin sampling speckle (the point
    # cloud is finite, so empty/near-empty bins next to full ones create
    # high-frequency noise) without blurring away the attractor's ribbons.
    # This also keeps the final PNG's entropy - and therefore its file
    # size - well under control.
    smooth_img = Image.fromarray((hist * 255.0).astype(np.uint8), "L")
    smooth_img = smooth_img.filter(ImageFilter.GaussianBlur(radius=1.1))
    hist = np.asarray(smooth_img, dtype=np.float64) / 255.0

    return hist


def colormap(t):
    """Map a normalised density array (0..1) through a multi-stop gradient.

    "Ember Tide" palette: deep indigo -> teal -> cyan glow -> magenta ->
    ember orange -> gold -> near-white hot core.
    """
    stops = np.array(
        [0.00, 0.12, 0.30, 0.50, 0.68, 0.82, 0.93, 1.00], dtype=np.float64
    )
    colors = np.array(
        [
            [3, 4, 14],
            [10, 14, 46],
            [18, 68, 118],
            [0, 172, 172],
            [181, 40, 160],
            [255, 110, 40],
            [255, 210, 90],
            [255, 250, 230],
        ],
        dtype=np.float64,
    )

    flat = t.ravel()
    r = np.interp(flat, stops, colors[:, 0])
    g = np.interp(flat, stops, colors[:, 1])
    b = np.interp(flat, stops, colors[:, 2])

    out = np.stack([r, g, b], axis=-1).reshape(t.shape[0], t.shape[1], 3)
    return out


def add_vignette(rgb_float, strength=0.22):
    """Multiply a gentle radial darkening toward the corners."""
    h, w = rgb_float.shape[0], rgb_float.shape[1]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    r = np.sqrt(((yy - cy) / (h / 2.0)) ** 2 + ((xx - cx) / (w / 2.0)) ** 2)
    r = np.clip(r / math.sqrt(2.0), 0.0, 1.0)
    falloff = 1.0 - strength * (r ** 2.2)
    return rgb_float * falloff[:, :, None]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Fixed starting point -> fully deterministic trajectory.
    x0, y0 = 0.1, 0.1
    xs, ys = iterate_clifford(A, B, C, D, x0, y0, N_POINTS, BURN_IN)

    density = build_density_field(xs, ys, SUPERSAMPLE)

    # Perceptual shaping: push up the faint filaments relative to the
    # brightest core so the whole structure of the attractor reads clearly.
    shaped = np.power(density, 0.42)

    base = colormap(shaped)  # float, 0..255 range, shape (H, W, 3)

    # --- bloom: bright regions glow softly onto their surroundings -------
    bright_mask = np.clip((shaped - 0.55) / 0.45, 0.0, 1.0) ** 1.5
    bloom_src = base * bright_mask[:, :, None]
    bloom_img = Image.fromarray(np.clip(bloom_src, 0, 255).astype(np.uint8), "RGB")
    bloom_img = bloom_img.filter(ImageFilter.GaussianBlur(radius=SUPERSAMPLE * 0.012))
    bloom = np.asarray(bloom_img, dtype=np.float64)

    combined = 255.0 - (255.0 - base) * (255.0 - bloom * 0.9) / 255.0

    combined = add_vignette(combined, strength=0.22)

    combined = np.clip(combined, 0, 255)
    img = Image.fromarray(combined.astype(np.uint8), "RGB")

    # Downsample for a clean, anti-aliased final image.
    img = img.resize((FINAL_SIZE, FINAL_SIZE), Image.LANCZOS)

    # Quantise to an (adaptive, deterministic) 256-colour palette. The
    # colour gradient is smooth and limited in hue, so this is visually
    # lossless at a glance while keeping the PNG comfortably under the
    # 1 MiB size limit. Dithering avoids visible banding in the gradients.
    img = img.quantize(colors=256, method=Image.FASTOCTREE, dither=Image.FLOYDSTEINBERG)

    # Save without any metadata/timestamp chunks, deterministic compression.
    img.save(OUT_PATH, format="PNG", optimize=True, compress_level=9)


if __name__ == "__main__":
    main()
