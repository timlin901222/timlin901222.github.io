"""Shared I/O, display, and figure helpers for proj2.

Convention used everywhere in this project: images are float64 in [0, 1].
Grayscale is (H, W); color is (H, W, 3). Nothing else.
"""

import os

import numpy as np
import skimage.io as skio
import skimage.color as skcolor
import matplotlib.pyplot as plt

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


# ==========================================================================
# I/O
# ==========================================================================
def imread(path, gray=False):
    """Read an image as float64 in [0, 1]. Drops an alpha channel if present."""
    im = skio.imread(path)
    if im.dtype == np.uint8:
        im = im.astype(np.float64) / 255.0
    elif im.dtype == np.uint16:
        im = im.astype(np.float64) / 65535.0
    else:
        im = im.astype(np.float64)
    if im.ndim == 3 and im.shape[2] == 4:
        im = im[:, :, :3]
    if gray and im.ndim == 3:
        im = skcolor.rgb2gray(im)
    return im


def imsave(name, im, subdir=None):
    """Save a float image to output/[subdir]/<name>. Returns the path."""
    d = OUTDIR if subdir is None else os.path.join(OUTDIR, subdir)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, name)
    skio.imsave(path, to_uint8(im))
    print(f"wrote {os.path.relpath(path)}")
    return path


def trim_uniform_border(im, tol=1e-3, verbose=True):
    """Drop constant-valued rows/cols from the outside in.

    Downloaded assets often arrive matted onto a white frame. That frame is a
    hard step at the image boundary, so every derivative filter reports a
    bright rectangle there and it survives thresholding as a fake edge. Trim
    it before doing anything else.
    """
    g = to_gray(im)
    H, W = g.shape

    def uniform(line):
        return line.max() - line.min() < tol

    t, b, l, r = 0, H, 0, W
    while t < b - 1 and uniform(g[t]):
        t += 1
    while b > t + 1 and uniform(g[b - 1]):
        b -= 1
    while l < r - 1 and uniform(g[:, l]):
        l += 1
    while r > l + 1 and uniform(g[:, r - 1]):
        r -= 1

    if (t, b, l, r) != (0, H, 0, W):
        if verbose:
            print(f"  trimmed uniform border: rows {t}:{b}, cols {l}:{r} "
                  f"({H}x{W} -> {b-t}x{r-l})")
        return im[t:b, l:r]
    return im


def to_uint8(im):
    return (np.clip(im, 0, 1) * 255).astype(np.uint8)


def to_gray(im):
    return skcolor.rgb2gray(im) if im.ndim == 3 else im


# ==========================================================================
# Display scaling
#
# Derivatives and Laplacian levels are signed. Squashing them with clip()
# throws away every negative value, so use one of these instead and say in
# the writeup which one a given figure used.
# ==========================================================================
def norm01(im):
    """Min-max stretch to [0, 1]. Good for magnitudes, misleading for signed data."""
    lo, hi = im.min(), im.max()
    return np.zeros_like(im) if hi - lo < 1e-12 else (im - lo) / (hi - lo)


def norm_signed(im, pct=99.5):
    """Map 0 -> mid gray, symmetric about zero. The honest view of a derivative.

    Scaled by the `pct` percentile of |im| rather than the max, because a
    handful of extreme pixels (a specular highlight, a hard occlusion edge)
    otherwise compress the entire rest of the image into flat gray. Values
    beyond the percentile clip, which is the intended trade: saturate a few
    hundred pixels so the millions in between stay readable. Pass pct=100 for
    a strict max scaling.
    """
    m = np.abs(im).max() if pct >= 100 else np.percentile(np.abs(im), pct)
    if m < 1e-12:
        return np.full_like(im, 0.5)
    return np.clip(im / (2 * m) + 0.5, 0, 1)


# ==========================================================================
# Figures
# ==========================================================================
def show_row(images, titles, name=None, cmap="gray", subdir=None, figsize_h=3.2):
    """One row of panels, shared styling. Saves to output/ if `name` is given."""
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(figsize_h * n, figsize_h * 1.15))
    axes = np.atleast_1d(axes)
    for ax, im, t in zip(axes, images, titles):
        ax.imshow(im, cmap=None if im.ndim == 3 else cmap, vmin=0, vmax=1)
        ax.set_title(t, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    if name:
        d = OUTDIR if subdir is None else os.path.join(OUTDIR, subdir)
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, name)
        fig.savefig(path, dpi=130, bbox_inches="tight")
        print(f"wrote {os.path.relpath(path)}")
        plt.close(fig)
        return path
    return fig


def show_grid(rows, row_titles, col_titles, name=None, cmap="gray", subdir=None, cell=2.4):
    """A grid of panels: one list of images per row, shared column headings.

    Used wherever a single row is not enough -- comparing two image regions
    across the same parameter sweep, or laying out a stack per input.
    """
    nr, nc = len(rows), len(rows[0])
    fig, axes = plt.subplots(nr, nc, figsize=(cell * nc, cell * nr * 1.05))
    axes = np.atleast_2d(axes)
    for r, (row, rt) in enumerate(zip(rows, row_titles)):
        for c, im in enumerate(row):
            ax = axes[r, c]
            ax.imshow(im, cmap=None if im.ndim == 3 else cmap, vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(col_titles[c], fontsize=9)
            if c == 0:
                ax.set_ylabel(rt, fontsize=8.5, rotation=0, ha="right", va="center", labelpad=8)
    fig.tight_layout()
    if name:
        d = OUTDIR if subdir is None else os.path.join(OUTDIR, subdir)
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, name)
        fig.savefig(path, dpi=130, bbox_inches="tight")
        print(f"wrote {os.path.relpath(path)}")
        plt.close(fig)
        return path
    return fig


def fft_logmag(im):
    """Log-magnitude spectrum, normalized to [0, 1] for saving as an image.

    Part 2.2 asks for five of these: both originals, both filtered, the hybrid.
    """
    g = to_gray(im)
    F = np.fft.fftshift(np.fft.fft2(g))
    return norm01(np.log(np.abs(F) + 1e-8))


def check_size(path, limit_mb=0.8):
    """The spec caps webpage images at 0.8 MB. Warn before the writeup, not after."""
    mb = os.path.getsize(path) / 1e6
    if mb > limit_mb:
        print(f"  !! {os.path.basename(path)} is {mb:.2f} MB (limit {limit_mb} MB)")
    return mb
