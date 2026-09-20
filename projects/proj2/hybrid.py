"""Part 2.2 -- Hybrid images (Oliva, Torralba & Schyns, SIGGRAPH 2006).

Low frequencies of one image + high frequencies of another. Which one you see
depends on viewing distance.

    # align first (writes output/aligned/*), then blend
    python align.py data/nutmeg.jpg data/DerekPicture.jpg --save derek
    python hybrid.py --low output/aligned/derek_2.jpg \
                     --high output/aligned/derek_1.jpg \
                     --sigma-low 8 --sigma-high 5 --tag derek --fft

--fft writes the five log-magnitude spectra the spec asks for on your one
fully-analyzed example.

Color, for the bells & whistles: --low-color / --high-color choose which
component keeps its color. Try all four combinations and report what you see.
"""

import argparse

import numpy as np

import utils
from edges import gaussian2d
from sharpen import conv_color


def low_pass(im, sigma, ksize=None):
    return conv_color(im, gaussian2d(ksize or (int(6 * sigma) | 1), sigma))


def high_pass(im, sigma, ksize=None):
    """Impulse minus Gaussian. Zero-mean, so it sits around 0, not 0.5."""
    return im - low_pass(im, sigma, ksize)


def hybrid_image(im_low, im_high, sigma_low, sigma_high):
    """Returns (hybrid, low component, high component).

    The two cutoffs are independent and both need tuning by eye. Larger
    sigma_low = the far-away image gets blurrier; larger sigma_high = the
    close-up image keeps only finer detail.
    """
    low = low_pass(im_low, sigma_low)
    high = high_pass(im_high, sigma_high)
    if low.ndim != high.ndim:  # one is gray, one is color -> broadcast gray to 3ch
        low, high = _match_channels(low, high)
    return np.clip((low + high) / 2 + 0.25, 0, 1), low, high


def _match_channels(a, b):
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=2)
    if b.ndim == 2:
        b = np.stack([b] * 3, axis=2)
    return a, b


def main():
    ap = argparse.ArgumentParser(description="Part 2.2 -- hybrid images")
    ap.add_argument("--low", required=True, help="image seen from far away")
    ap.add_argument("--high", required=True, help="image seen up close")
    ap.add_argument("--sigma-low", type=float, default=8.0)
    ap.add_argument("--sigma-high", type=float, default=5.0)
    ap.add_argument("--low-color", action="store_true", help="keep color in the low pass")
    ap.add_argument("--high-color", action="store_true", help="keep color in the high pass")
    ap.add_argument("--fft", action="store_true", help="write the 5 spectra")
    ap.add_argument("--tag", default="hybrid")
    args = ap.parse_args()

    im_low = utils.imread(args.low, gray=not args.low_color)
    im_high = utils.imread(args.high, gray=not args.high_color)
    if im_low.shape[:2] != im_high.shape[:2]:
        raise SystemExit(
            f"shape mismatch {im_low.shape[:2]} vs {im_high.shape[:2]} -- run align.py first")

    out, low, high = hybrid_image(im_low, im_high, args.sigma_low, args.sigma_high)
    utils.imsave(f"p22_{args.tag}.jpg", out)
    utils.show_row([im_low, im_high, np.clip(low, 0, 1), utils.norm_signed(high), out],
                   ["low-freq source", "high-freq source",
                    f"low pass (s={args.sigma_low})", f"high pass (s={args.sigma_high})",
                    "hybrid"],
                   name=f"p22_{args.tag}_process.jpg")

    if args.fft:
        utils.show_row(
            [utils.fft_logmag(x) for x in (im_low, im_high, low, high, out)],
            ["FFT: low source", "FFT: high source", "FFT: low-passed",
             "FFT: high-passed", "FFT: hybrid"],
            name=f"p22_{args.tag}_fft.jpg")


if __name__ == "__main__":
    main()
