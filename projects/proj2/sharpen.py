"""Part 2.1 -- Image sharpening (unsharp masking).

    python sharpen.py --image data/taj.jpg --alpha 1                 # basic
    python sharpen.py --image data/taj.jpg --sweep 0 0.5 1 2 4       # alpha sweep
    python sharpen.py --image data/sharp.jpg --resharpen --sigma 3   # blur -> sharpen
"""

import argparse

import numpy as np
from scipy.signal import convolve2d

import utils
from edges import gaussian2d


def conv_color(im, k):
    """Apply a 2D kernel to each channel (or once, if already grayscale)."""
    if im.ndim == 2:
        return convolve2d(im, k, mode="same", boundary="symm")
    return np.stack([convolve2d(im[:, :, c], k, mode="same", boundary="symm")
                     for c in range(im.shape[2])], axis=2)


def unsharp_kernel(ksize, sigma, alpha):
    """The single combined filter: (1 + alpha) * delta - alpha * G.

    Part 2.1 explicitly asks you to show sharpening as ONE convolution, not as
    blur-subtract-add. Visualize this kernel in the writeup and verify it
    matches the two-step version.
    """
    delta = np.zeros((ksize, ksize))
    delta[ksize // 2, ksize // 2] = 1.0
    return (1 + alpha) * delta - alpha * gaussian2d(ksize, sigma)


def sharpen(im, sigma=2.0, alpha=1.0, ksize=None, single_filter=True):
    """Returns (sharpened, low_freq, high_freq), all clipped for display."""
    ksize = ksize or (int(6 * sigma) | 1)
    low = conv_color(im, gaussian2d(ksize, sigma))
    high = im - low
    out = conv_color(im, unsharp_kernel(ksize, sigma, alpha)) if single_filter \
        else im + alpha * high
    return np.clip(out, 0, 1), low, high


def main():
    ap = argparse.ArgumentParser(description="Part 2.1 -- unsharp masking")
    ap.add_argument("--image", default="data/taj.jpg")
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--sweep", type=float, nargs="+", help="alphas to compare")
    ap.add_argument("--resharpen", action="store_true",
                    help="blur an already-sharp image, then try to recover it")
    ap.add_argument("--tag", default=None, help="output filename prefix")
    args = ap.parse_args()

    im = utils.imread(args.image)
    tag = args.tag or "taj"

    if args.resharpen:
        # The interesting negative result: sharpening is not deconvolution.
        # Compare against the original and say what came back and what didn't.
        blurred = conv_color(im, gaussian2d(int(6 * args.sigma) | 1, args.sigma))
        recovered, _, _ = sharpen(blurred, args.sigma, args.alpha)
        utils.show_row([im, np.clip(blurred, 0, 1), recovered],
                       ["original (sharp)", f"blurred (sigma={args.sigma})",
                        f"re-sharpened (alpha={args.alpha})"],
                       name=f"p21_{tag}_resharpen.jpg")
        return

    if args.sweep:
        outs = [sharpen(im, args.sigma, a)[0] for a in args.sweep]
        utils.show_row(outs, [f"alpha = {a}" for a in args.sweep],
                       name=f"p21_{tag}_alpha_sweep.jpg")
        return

    out, low, high = sharpen(im, args.sigma, args.alpha)
    utils.show_row([im, np.clip(low, 0, 1), utils.norm_signed(high), out],
                   ["original", f"low pass (sigma={args.sigma})",
                    "high frequencies", f"sharpened (alpha={args.alpha})"],
                   name=f"p21_{tag}_unsharp.jpg")
    utils.imsave(f"p21_{tag}_sharpened.jpg", out)


if __name__ == "__main__":
    main()
