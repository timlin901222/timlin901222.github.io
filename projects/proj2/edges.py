"""Parts 1.2 and 1.3 -- finite difference, DoG, and gradient orientation.

    python edges.py --threshold 0.25                    # 1.2, finite difference
    python edges.py --dog --sigma 2 --threshold 0.042   # 1.3, DoG
    python edges.py --orientation --dog --sigma 2       # 1.3 bells & whistles

Choosing the parameters is the graded part, so the sweeps are first-class:
    python edges.py --sweep 0.05 0.1 0.15 0.2 0.25 0.3  # threshold, whole image
    python edges.py --detail 0.1 0.15 0.2 0.25 0.3      # threshold, per region
    python edges.py --curve --threshold 0.25            # density vs threshold
    python edges.py --sigma-sweep                       # sigma, equal retention
    python edges.py --compare --sigma 2 --threshold 0.042
"""

import argparse

import numpy as np
import cv2
import skimage.color as skcolor
from scipy.signal import convolve2d

import utils
from convolve import DX, DY

DEFAULT_IMAGE = "data/cameraman.png"


def conv(im, k):
    return convolve2d(im, k, mode="same", boundary="symm")


def gaussian2d(ksize, sigma):
    """Separable 1D Gaussian from cv2, outer-product'd into a 2D kernel.

    ksize should be odd and comfortably wider than sigma -- 6*sigma+1 is the
    usual rule; anything tighter truncates the tail and biases the result.
    """
    g = cv2.getGaussianKernel(ksize, sigma)
    return g @ g.T


def gradient(im):
    """(gx, gy, magnitude) from the plain finite difference operators."""
    gx, gy = conv(im, DX), conv(im, DY)
    return gx, gy, np.sqrt(gx ** 2 + gy ** 2)


def dog_filters(ksize, sigma):
    """The two DoG kernels: D_x * G and D_y * G, built once and reused.

    The whole point of 1.3 is that convolving with these gives the same answer
    as blurring first and differencing after -- verify it and report the max
    absolute difference.
    """
    g = gaussian2d(ksize, sigma)
    return convolve2d(g, DX, mode="full"), convolve2d(g, DY, mode="full")


def binarize(mag, t):
    return (mag > t).astype(np.float64)


# Two regions of the cameraman image that want opposite thresholds. The
# skyline is low-contrast structure that a high threshold erases; the grass is
# real texture that a low threshold turns into confetti. Any single global
# threshold is a compromise between these two, which is the whole argument of
# Part 1.2.
CAMERAMAN_CROPS = {
    "skyline": (200, 340, 340, 534),
    "grass": (430, 530, 425, 534),
}


def threshold_detail(mag, thresholds, crops=None, name="p12_threshold_detail.jpg"):
    """Binarized close-ups of competing regions across a threshold sweep.

    Shows *why* a threshold is a compromise rather than just *that* one was
    picked. Prints the edge-pixel density per region so the visual read has a
    number behind it.
    """
    crops = crops or CAMERAMAN_CROPS
    rows, row_titles = [], []
    for label, (r0, r1, c0, c1) in crops.items():
        sub = mag[r0:r1, c0:c1]
        rows.append([binarize(sub, t) for t in thresholds])
        row_titles.append(label)
        print(f"  {label:<10} " + "  ".join(
            f"t={t:.2f}: {100 * (sub > t).mean():5.2f}%" for t in thresholds))
    return utils.show_grid(rows, row_titles,
                           [f"t = {t}" for t in thresholds], name=name)


def edge_strength(mag, r0, r1, col):
    """|grad| sampled straight down one known structural edge."""
    return mag[r0:r1, col]


def threshold_curve(mag, crops=None, pick=None, landmark=None,
                    name="p12_threshold_curve.jpg"):
    """Edge-pixel density vs threshold, one line per region.

    `landmark` is an optional (value, label) pair marking a measured gradient
    strength -- e.g. the strongest pixel on a real edge you want to keep. If
    that landmark sits to the LEFT of where the texture curve flattens, no
    global threshold can keep that edge and reject the texture, and the plot
    says so at a glance.
    """
    import matplotlib.pyplot as plt
    import os

    crops = crops or CAMERAMAN_CROPS
    ts = np.linspace(0.02, 0.45, 100)
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    colors = ["#2C4A7C", "#9C4A3C", "#3E6B4F"]

    for (label, (r0, r1, c0, c1)), color in zip(crops.items(), colors):
        sub = mag[r0:r1, c0:c1]
        ax.plot(ts, [100 * (sub > t).mean() for t in ts],
                color=color, lw=1.8, label=label)

    if landmark is not None:
        v, lab = landmark
        ax.axvline(v, color="#9C4A3C", ls="--", lw=1.3)
        ax.annotate(lab, xy=(v, 30), xytext=(v + 0.015, 34), fontsize=8.5,
                    color="#9C4A3C")

    if pick is not None:
        ax.axvline(pick, color="#2C4A7C", lw=1.4, alpha=0.4)
        ax.annotate(f"chosen\nt = {pick}", xy=(pick, 1), xytext=(pick + 0.012, 18),
                    fontsize=8.5, color="#2C4A7C")

    ax.set_xlabel("threshold on |∇I|")
    ax.set_ylabel("% of region marked as edge")
    ax.set_title("Edge density by region — the threshold is a trade, not a setting",
                 fontsize=10)
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()

    os.makedirs(utils.OUTDIR, exist_ok=True)
    path = os.path.join(utils.OUTDIR, name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {os.path.relpath(path)}")
    return path


def dog_gradient(im, sigma, ksize=None, single_filter=True):
    """(gx, gy, magnitude) after Gaussian smoothing.

    single_filter=True convolves once with D*G; False blurs then differences.
    The two are mathematically identical -- convolution is associative, so
    (I * G) * D == I * (G * D) -- and `main` checks that numerically.
    """
    ksize = ksize or (int(6 * sigma) | 1)
    if single_filter:
        kx, ky = dog_filters(ksize, sigma)
        gx, gy = conv(im, kx), conv(im, ky)
        return gx, gy, np.sqrt(gx ** 2 + gy ** 2)
    return gradient(conv(im, gaussian2d(ksize, sigma)))


def threshold_at_retention(mag, r0, r1, col, keep=0.90):
    """Threshold that retains `keep` of the pixels along a known edge.

    Smoothing shrinks every gradient, so a threshold picked at one sigma is
    meaningless at another. Pinning the threshold to a fixed retention of one
    real edge is what makes a sigma sweep a fair comparison instead of a
    brightness comparison.
    """
    return float(np.percentile(mag[r0:r1, col], 100 * (1 - keep)))


def edge_fwhm(mag, row, col, halfwidth=25):
    """Full width at half maximum of |grad| across an isolated edge, in px.

    This is what smoothing costs: the DoG response to a step is a Gaussian
    lobe, so detected edges thicken as roughly 2.35*sigma and localization
    degrades in step.
    """
    prof = mag[row, col - halfwidth:col + halfwidth + 1]
    above = np.where(prof >= prof.max() / 2)[0]
    return int(above[-1] - above[0] + 1)


# Column 400, rows 230-300 is the tall tower's left edge -- the structure that
# Part 1.2 showed no global threshold can recover. Row 360 / column 205 is an
# isolated high-contrast step used for the width measurement.
TOWER = (230, 300, 400)
ISOLATED_EDGE = (360, 205)


def sigma_sweep(im, sigmas=(0, 1, 1.5, 2, 3, 4), keep=0.90,
                name="p13_sigma_sweep.jpg"):
    """Binarized edges across sigma, each at its own equal-retention threshold.

    Prints the trade as numbers: grass contamination falls, edge width grows.
    """
    grass = CAMERAMAN_CROPS["grass"]
    panels, titles = [], []
    print(f"  {'sigma':>6} {'ksize':>6} {'t(90% tower)':>13} {'grass':>8} "
          f"{'whole img':>10} {'edge FWHM':>10}")
    for s in sigmas:
        mag = gradient(im)[2] if s == 0 else dog_gradient(im, s)[2]
        t = threshold_at_retention(mag, *TOWER, keep=keep)
        g = mag[grass[0]:grass[1], grass[2]:grass[3]]
        w = edge_fwhm(mag, *ISOLATED_EDGE)
        ks = "-" if s == 0 else int(6 * s) | 1
        print(f"  {s:>6} {str(ks):>6} {t:>13.4f} {100*(g>t).mean():7.2f}% "
              f"{100*(mag>t).mean():9.2f}% {w:>9} px")
        panels.append(binarize(mag, t))
        titles.append(f"σ = {s}  (t = {t:.3f})" if s else f"no blur  (t = {t:.3f})")
    return utils.show_row(panels, titles, name=name, figsize_h=2.9)


def kernel_figure(ksize, sigma, name="p13_dog_kernels.jpg"):
    """The construction: G, then D_x*G and D_y*G, plus a horizontal profile.

    The profile is the point -- a DoG kernel is one positive and one negative
    lobe, so convolving with it takes a smoothed difference across the centre
    in a single pass. Its taps sum to zero, which is why it returns zero on
    any flat region no matter how bright.
    """
    import matplotlib.pyplot as plt
    import os

    g = gaussian2d(ksize, sigma)
    kx, ky = dog_filters(ksize, sigma)

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
    for ax, k, t in [(axes[0], g, f"G  (σ={sigma}, {ksize}×{ksize})"),
                     (axes[1], kx, f"D$_x$ * G   {kx.shape[0]}×{kx.shape[1]}"),
                     (axes[2], ky, f"D$_y$ * G   {ky.shape[0]}×{ky.shape[1]}")]:
        ax.imshow(utils.norm_signed(k, pct=100), cmap="gray", vmin=0, vmax=1)
        ax.set_title(t, fontsize=9)
        ax.axis("off")

    mid = kx.shape[0] // 2
    axes[3].plot(kx[mid], color="#2C4A7C", lw=1.8, marker="o", ms=3)
    axes[3].axhline(0, color="#615D53", lw=0.8)
    axes[3].set_title(f"D$_x$ * G, centre row\nsum of all taps = {kx.sum():.1e}", fontsize=9)
    axes[3].set_xlabel("tap index")
    axes[3].grid(alpha=0.25, lw=0.6)
    fig.tight_layout()

    os.makedirs(utils.OUTDIR, exist_ok=True)
    path = os.path.join(utils.OUTDIR, name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {os.path.relpath(path)}")
    return path


def equivalence_figure(im, sigma, t, ksize=None, name="p13_equivalence.jpg"):
    """Both routes shown, not just asserted equal.

    Route A blurs with G and then applies D_x/D_y. Route B convolves once with
    the combined D*G kernels. The spec asks for both to be applied, so both
    are displayed alongside their difference, amplified 1e12x so that anything
    non-zero would be impossible to miss.
    """
    ksize = ksize or (int(6 * sigma) | 1)
    _, _, mag_a = dog_gradient(im, sigma, ksize, single_filter=False)
    _, _, mag_b = dog_gradient(im, sigma, ksize, single_filter=True)
    diff = np.abs(mag_a - mag_b)
    print(f"  route A (blur then difference) vs route B (single D*G conv): "
          f"max|diff| = {diff.max():.2e}")
    return utils.show_grid(
        [[utils.norm01(mag_a), binarize(mag_a, t), np.clip(diff * 1e12, 0, 1)],
         [utils.norm01(mag_b), binarize(mag_b, t), np.clip(diff * 1e12, 0, 1)]],
        ["A: blur,\nthen D$_x$/D$_y$", "B: one conv\nwith D*G"],
        ["gradient magnitude", f"binarized (t = {t})", "|A − B| × 10$^{12}$"],
        name=name, cell=3.0)


def compare_finite_vs_dog(im, sigma, t_fd, t_dog, name="p13_comparison.jpg"):
    """Part 1.2 and Part 1.3 side by side, each at its own chosen threshold."""
    _, _, mag_fd = gradient(im)
    _, _, mag_dog = dog_gradient(im, sigma)
    return utils.show_grid(
        [[utils.norm01(mag_fd), binarize(mag_fd, t_fd)],
         [utils.norm01(mag_dog), binarize(mag_dog, t_dog)]],
        ["finite\ndifference", f"DoG\nσ = {sigma}"],
        ["gradient magnitude", "binarized"],
        name=name, cell=3.6)


def gradient_angle(gx, gy):
    """Gradient orientation in [0, 2pi), computed without np.arctan2.

    np.arctan only returns (-pi/2, pi/2) -- it sees gy/gx, so it cannot tell
    (1, 1) from (-1, -1) and collapses opposite directions onto the same
    answer. Recovering the full circle means doing the quadrant bookkeeping
    that arctan2 normally hides:

      * gx > 0  -> arctan(gy/gx) is already right (quadrants I and IV)
      * gx < 0  -> add pi (quadrants II and III)
      * gx == 0 -> the ratio is undefined; the vector points straight up or
                   straight down, so use +pi/2 or -pi/2 by the sign of gy

    Row indices grow downward in an image, so gy is negated first to make the
    angle read in ordinary screen orientation (0 = pointing right,
    pi/2 = pointing up).
    """
    gy = -gy
    with np.errstate(divide="ignore", invalid="ignore"):
        theta = np.arctan(np.divide(gy, gx, out=np.zeros_like(gy), where=gx != 0))
    theta = np.where(gx < 0, theta + np.pi, theta)
    vertical = (gx == 0)
    theta = np.where(vertical, np.where(gy >= 0, np.pi / 2, -np.pi / 2), theta)
    return np.mod(theta, 2 * np.pi)


def orientation_hsv(gx, gy, mag=None, gamma=1.0, mag_pct=99.0):
    """Gradient orientation as hue, gradient strength as value.

    Hue is the natural channel for orientation because both are cyclic: an
    edge at 359 degrees and one at 1 degree are nearly parallel and get nearly
    the same colour, which no linear colormap can do. This is the same wrap
    that matplotlib's cyclic maps (hsv, twilight) provide.

    Strength drives value rather than being ignored, because orientation is
    meaningless where there is no gradient -- in a flat region the angle is
    whatever the noise happened to be. Mapping weak gradients to black stops
    the figure from showing confident colours for nothing.
    """
    if mag is None:
        mag = np.sqrt(gx ** 2 + gy ** 2)
    theta = gradient_angle(gx, gy)

    h = theta / (2 * np.pi)
    s = np.ones_like(h)
    v = np.clip(mag / np.percentile(mag, mag_pct), 0, 1) ** gamma
    return skcolor.hsv2rgb(np.stack([h, s, v], axis=2))


def orientation_wheel(size=140):
    """A colour key: hue around the circle, so the figure can be read."""
    y, x = np.mgrid[-1:1:size * 1j, -1:1:size * 1j]
    r = np.sqrt(x ** 2 + y ** 2)
    # y already runs top-to-bottom like image rows, so it is passed through
    # unchanged -- gradient_angle does the screen-orientation flip itself.
    theta = gradient_angle(x, y)
    rgb = skcolor.hsv2rgb(np.stack(
        [theta / (2 * np.pi), np.ones_like(theta),
         ((r < 1) & (r > 0.45)).astype(float)], axis=2))
    return rgb


def main():
    ap = argparse.ArgumentParser(description="Parts 1.2 / 1.3 -- edge detection")
    ap.add_argument("--image", default=DEFAULT_IMAGE)
    ap.add_argument("--threshold", type=float, default=0.2)
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--ksize", type=int, default=None, help="default 6*sigma+1, odd")
    ap.add_argument("--dog", action="store_true", help="Part 1.3 instead of 1.2")
    ap.add_argument("--orientation", action="store_true", help="1.3 bells & whistles")
    ap.add_argument("--sweep", type=float, nargs="+", help="threshold sweep figure")
    ap.add_argument("--detail", type=float, nargs="+",
                    help="binarized close-ups of competing regions, per threshold")
    ap.add_argument("--curve", action="store_true",
                    help="edge density vs threshold, per region")
    ap.add_argument("--sigma-sweep", action="store_true",
                    help="binarized edges across sigma at equal tower retention")
    ap.add_argument("--kernels", action="store_true", help="G and the two DoG kernels")
    ap.add_argument("--equivalence", action="store_true",
                    help="show both DoG routes and their difference")
    ap.add_argument("--compare", action="store_true",
                    help="1.2 vs 1.3 side by side (uses --threshold for DoG)")
    args = ap.parse_args()

    im = utils.trim_uniform_border(utils.imread(args.image, gray=True))
    ksize = args.ksize or (int(6 * args.sigma) | 1)

    if args.kernels:
        kernel_figure(ksize, args.sigma)
        return

    if args.sigma_sweep:
        sigma_sweep(im)
        return

    if args.equivalence:
        equivalence_figure(im, args.sigma, args.threshold, ksize)
        return

    if args.compare:
        compare_finite_vs_dog(im, args.sigma, t_fd=0.25, t_dog=args.threshold)
        return

    if args.orientation:
        gx, gy, mag = dog_gradient(im, args.sigma, ksize) if args.dog else gradient(im)
        rgb = orientation_hsv(gx, gy, mag)
        tag = f"p13_orientation{'_dog' if args.dog else ''}.jpg"
        utils.show_row([im, rgb, orientation_wheel()],
                       ["input",
                        f"gradient orientation{f' (DoG σ={args.sigma})' if args.dog else ''}",
                        "hue key: 0°=right, 90°=up"],
                       name=tag)
        utils.imsave(tag.replace(".jpg", "_full.jpg"), rgb)
        return

    if args.dog:
        # Two routes to the same answer: blur then difference, or convolve once
        # with D*G. Convolution is associative, so they must agree.
        gx2, gy2, mag2 = dog_gradient(im, args.sigma, ksize, single_filter=False)
        gx, gy, mag = dog_gradient(im, args.sigma, ksize, single_filter=True)
        print(f"blur-then-diff vs single DoG conv: max|diff| = {np.abs(mag - mag2).max():.2e}")

        kernel_figure(ksize, args.sigma)
        blurred = conv(im, gaussian2d(ksize, args.sigma))
        prefix, panels, titles = "p13", [blurred], [f"blurred (σ={args.sigma})"]
    else:
        gx, gy, mag = gradient(im)
        prefix, panels, titles = "p12", [im], ["original"]

    if args.curve:
        landmark = (0.162, "strongest pixel on the\ntower's edge = 0.162") \
            if "cameraman" in args.image else None
        threshold_curve(mag, pick=args.threshold, landmark=landmark,
                        name=f"{prefix}_threshold_curve.jpg")
        return

    if args.detail:
        threshold_detail(mag, args.detail, name=f"{prefix}_threshold_detail.jpg")
        return

    if args.sweep:
        utils.show_row(
            [binarize(mag, t) for t in args.sweep],
            [f"t = {t}" for t in args.sweep],
            name=f"{prefix}_threshold_sweep.jpg",
        )
        return

    utils.show_row(
        panels + [utils.norm_signed(gx), utils.norm_signed(gy),
                  utils.norm01(mag), binarize(mag, args.threshold)],
        titles + ["d/dx", "d/dy", "gradient magnitude", f"binarized (t={args.threshold})"],
        name=f"{prefix}_edges.jpg",
    )


if __name__ == "__main__":
    main()
