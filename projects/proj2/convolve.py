"""Part 1.1 -- Convolutions from scratch.

Deliverables: the two implementations below, a runtime table against
scipy.signal.convolve2d, and your grayscale self-portrait convolved with a
9x9 box filter, D_x, and D_y.

    python convolve.py --image data/selfie.jpg
    python convolve.py --image data/selfie.jpg --bench
"""

import argparse
import os
import time

import numpy as np
import skimage.transform as sktr
from scipy.signal import convolve2d as scipy_convolve2d

import utils

DX = np.array([[1.0, 0.0, -1.0]])
DY = DX.T
BOX9 = np.ones((9, 9)) / 81.0


def pad_for(im, k, mode):
    """Zero-pad so the valid region has 'same' or 'full' geometry.

    full: output is im + k - 1  -> pad k-1 on both sides.
    same: output is im.shape    -> the centered crop of 'full'. For an odd
          kernel that is k//2 on each side; for an even one there is no true
          center, so the padding is lopsided, matching what scipy calls 'same'.
    """
    kh, kw = k.shape
    if mode == "full":
        pads = ((kh - 1, kh - 1), (kw - 1, kw - 1))
    else:
        pads = ((kh - 1 - (kh - 1) // 2, (kh - 1) // 2),
                (kw - 1 - (kw - 1) // 2, (kw - 1) // 2))
    return np.pad(im, pads, mode="constant", constant_values=0)


def out_shape(im, k, mode):
    """Shape of the valid region once the image has been padded for `mode`."""
    kh, kw = k.shape
    padded = pad_for(im, k, mode)
    return padded, padded.shape[0] - kh + 1, padded.shape[1] - kw + 1


def conv2d_4loop(im, k, mode="same"):
    """Four nested loops: over output rows, output cols, kernel rows, kernel cols.

    The kernel is flipped in both axes up front, which is the one thing that
    separates convolution from correlation. After the flip the inner two loops
    are a plain dot product between the kernel and the window under it.
    """
    kf = k[::-1, ::-1]
    kh, kw = k.shape
    padded, H, W = out_shape(im, k, mode)

    out = np.zeros((H, W))
    for r in range(H):
        for c in range(W):
            acc = 0.0
            for i in range(kh):
                for j in range(kw):
                    acc += padded[r + i, c + j] * kf[i, j]
            out[r, c] = acc
    return out


def conv2d_2loop(im, k, mode="same"):
    """Two loops over the kernel only.

    Same arithmetic as the 4-loop version, reassociated: instead of visiting
    one output pixel and summing kh*kw products, visit one kernel tap and
    apply it to every output pixel at once. The image slice
    padded[i:i+H, j:j+W] is exactly the set of pixels that tap (i, j) touches,
    so the inner two loops become a single vectorized add.
    """
    kf = k[::-1, ::-1]
    kh, kw = k.shape
    padded, H, W = out_shape(im, k, mode)

    out = np.zeros((H, W))
    for i in range(kh):
        for j in range(kw):
            out += kf[i, j] * padded[i:i + H, j:j + W]
    return out


def compare(im, k, label):
    """Run all three, report max abs difference and wall time."""
    rows = []
    for name, fn in [
        ("4-loop", lambda: conv2d_4loop(im, k, "same")),
        ("2-loop", lambda: conv2d_2loop(im, k, "same")),
        ("scipy", lambda: scipy_convolve2d(im, k, mode="same", boundary="fill", fillvalue=0)),
    ]:
        t0 = time.perf_counter()
        out = fn()
        rows.append((name, time.perf_counter() - t0, out))

    print(f"\n{label}  kernel {k.shape}  image {im.shape}")
    ref = next((o for n, _, o in rows if n == "scipy"), None)
    for name, dt, out in rows:
        diff = "" if out is ref else f"   max|diff vs scipy| = {np.abs(out - ref).max():.2e}"
        print(f"  {name:<8} {dt:7.3f}s{diff}")
    return rows


def scaling_plot(im, sizes=(3, 5, 7, 9, 11, 13, 15), name="p11_scaling.jpg"):
    """Runtime vs kernel area for all three implementations.

    All three are linear in kernel area -- they all do H*W*kh*kw multiply-adds
    and none of them can avoid that. What separates them is the constant, and
    the constant is set by where the loop lives: once per multiply in the
    Python interpreter, or once per kernel tap with NumPy doing the rest in C.
    Plotted on both scales because the log axis is the only one where all
    three fit, and the linear axis is the only one that conveys the gap.
    """
    import matplotlib.pyplot as plt

    areas, series = [], {"4-loop": [], "2-loop": [], "scipy": []}
    for s in sizes:
        k = np.ones((s, s)) / (s * s)
        areas.append(s * s)
        for name_, fn in [
            ("4-loop", lambda: conv2d_4loop(im, k, "same")),
            ("2-loop", lambda: conv2d_2loop(im, k, "same")),
            ("scipy", lambda: scipy_convolve2d(im, k, mode="same",
                                               boundary="fill", fillvalue=0)),
        ]:
            t0 = time.perf_counter()
            fn()
            series[name_].append(time.perf_counter() - t0)
        print(f"  k={s:>2}x{s:<2} area={s*s:>3}   "
              + "   ".join(f"{n} {series[n][-1]:.4f}s" for n in series))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    styles = {"4-loop": ("#9C4A3C", "o"), "2-loop": ("#2C4A7C", "s"), "scipy": ("#3E6B4F", "^")}

    for ax, logy in zip(axes, [True, False]):
        for n, ts in series.items():
            c, m = styles[n]
            ax.plot(areas, ts, marker=m, color=c, label=n, lw=1.6, ms=5)
        ax.set_xlabel("kernel area (taps)")
        ax.set_ylabel("seconds per call")
        ax.grid(alpha=0.25, lw=0.6)
        if logy:
            ax.set_yscale("log")
            ax.set_title(f"log scale — all three fit, {im.shape[0]}x{im.shape[1]} image", fontsize=10)
        else:
            ax.set_ylim(0, max(series["2-loop"] + series["scipy"]) * 1.6)
            ax.set_title("linear scale — 4-loop is off the top of the axis", fontsize=10)
    axes[0].legend(fontsize=9, frameon=False)
    fig.tight_layout()

    path = os.path.join(utils.OUTDIR, name)
    os.makedirs(utils.OUTDIR, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {os.path.relpath(path)}")
    return series


def accumulation_figure(im, k=BOX9, taps=(1, 9, 45, 81), name="p11_accumulation.jpg"):
    """What the 2-loop version is actually doing, one tap at a time.

    Each panel is the output after n of the kh*kw shifted-and-scaled copies
    have been added in, rescaled by (total / n) so brightness stays comparable.
    A box filter makes this easy to read. One tap is a shifted copy of the
    original; nine taps is one full row of the kernel, so the partial sum is a
    purely horizontal smear; only after all 81 have landed is the blur
    isotropic. The 4-loop version computes the same thing in the opposite
    order -- one output pixel finished at a time -- so its partial state is
    never a viewable image, which is itself the difference between the two.
    """
    kf = k[::-1, ::-1]
    kh, kw = k.shape
    padded, H, W = out_shape(im, k, "same")

    out = np.zeros((H, W))
    total = kh * kw
    panels, titles = [], []
    n = 0
    for i in range(kh):
        for j in range(kw):
            out += kf[i, j] * padded[i:i + H, j:j + W]
            n += 1
            if n in taps:
                panels.append(np.clip(out * (total / n), 0, 1))
                titles.append(f"after {n} of {total} taps")

    return utils.show_row([im] + panels, ["original"] + titles, name=name)


def main():
    ap = argparse.ArgumentParser(description="Part 1.1 -- convolution from scratch")
    ap.add_argument("--image", default="data/selfie.jpg", help="your self-portrait")
    ap.add_argument("--bench", action="store_true", help="print the runtime table")
    ap.add_argument("--scaling", action="store_true",
                    help="runtime vs kernel area, both scales")
    ap.add_argument("--accum", action="store_true",
                    help="show the 2-loop output building up tap by tap")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="downscale before running; the 4-loop version is O(HWkhkw) "
                         "in pure Python, so benchmark it small")
    args = ap.parse_args()

    im = utils.imread(args.image, gray=True)
    if args.scale != 1.0:
        im = sktr.rescale(im, args.scale, anti_aliasing=True)

    if args.bench:
        for k, label in [(BOX9, "box 9x9"), (DX, "D_x"), (DY, "D_y")]:
            compare(im, k, label)
        return

    if args.scaling:
        scaling_plot(im)
        return

    if args.accum:
        accumulation_figure(im)
        return

    box = conv2d_2loop(im, BOX9, "same")
    dx = conv2d_2loop(im, DX, "same")
    dy = conv2d_2loop(im, DY, "same")

    utils.show_row(
        [im, np.clip(box, 0, 1), utils.norm_signed(dx), utils.norm_signed(dy)],
        ["original", "box 9x9", "D_x", "D_y"],
        name="p11_selfie_filters.jpg",
    )


if __name__ == "__main__":
    main()
