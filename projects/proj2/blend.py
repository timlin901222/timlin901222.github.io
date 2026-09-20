"""Part 2.4 -- Multiresolution blending (Burt & Adelson 1983). The Oraple.

    # vertical seam
    python blend.py --a data/apple.jpeg --b data/orange.jpeg --mask vertical \
                    --levels 6 --tag oraple --process
    # horizontal seam
    python blend.py --a data/apple.jpeg --b data/orange.jpeg --mask horizontal --tag oraple_h
    # irregular mask (white = keep image A)
    python blend.py --a data/x.jpg --b data/y.jpg --mask data/mask_x.png --tag custom

--process writes the Figure-10-style visualization: masked inputs, each
input's Laplacian stack, and the blended stack.
"""

import argparse

import numpy as np

import utils
from stacks import gaussian_stack, laplacian_stack, show_stack


def step_mask(shape, orientation="vertical", frac=0.5):
    """Binary step mask: 1 where image A shows through, 0 where B does."""
    h, w = shape[:2]
    m = np.zeros((h, w))
    if orientation == "vertical":
        m[:, :int(w * frac)] = 1.0
    else:
        m[:int(h * frac), :] = 1.0
    return m


def load_mask(spec, shape):
    if spec in ("vertical", "horizontal"):
        return step_mask(shape, spec)
    m = utils.imread(spec, gray=True)
    if m.shape[:2] != shape[:2]:
        raise SystemExit(f"mask is {m.shape[:2]}, images are {shape[:2]}")
    return (m > 0.5).astype(np.float64)


def blend(im_a, im_b, mask, levels=6, sigma=2.0):
    """Multiresolution blend. Returns (result, la, lb, lblend).

    TODO(you): implement.
      - la = laplacian_stack(gaussian_stack(im_a)), same for im_b
      - gm = gaussian_stack(mask) -- the mask gets blurred too, and that is
        the whole trick: each frequency band is blended over a transition
        zone whose width matches that band's wavelength.
      - blended[i] = gm[i] * la[i] + (1 - gm[i]) * lb[i]
      - result = sum(blended), clipped to [0, 1].
      - Color: broadcast the (H, W) mask level to (H, W, 1) before multiplying.
      - Writeup question: why this differs from the pyramid version -- no
        resampling, so every level is already aligned and no expand step is
        needed.
    """
    raise NotImplementedError


def main():
    ap = argparse.ArgumentParser(description="Part 2.4 -- multiresolution blending")
    ap.add_argument("--a", required=True, help="shows through where the mask is white")
    ap.add_argument("--b", required=True)
    ap.add_argument("--mask", default="vertical",
                    help="'vertical', 'horizontal', or a path to a mask image")
    ap.add_argument("--levels", type=int, default=6)
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--tag", default="blend")
    ap.add_argument("--process", action="store_true", help="Figure 10 style visualization")
    args = ap.parse_args()

    im_a = utils.imread(args.a)
    im_b = utils.imread(args.b)
    if im_a.shape != im_b.shape:
        raise SystemExit(f"shape mismatch {im_a.shape} vs {im_b.shape} -- crop or resize first")
    mask = load_mask(args.mask, im_a.shape)

    try:
        out, la, lb, lblend = blend(im_a, im_b, mask, args.levels, args.sigma)
    except NotImplementedError:
        print("Implement blend() (and the stacks in stacks.py) first.")
        return

    utils.imsave(f"p24_{args.tag}.jpg", out)
    utils.show_row([im_a, im_b, mask, out],
                   ["image A", "image B", "mask", "blended"],
                   name=f"p24_{args.tag}_summary.jpg")

    if args.process:
        m3 = mask[..., None] if im_a.ndim == 3 else mask
        utils.show_row([im_a * m3, im_b * (1 - m3)], ["A masked", "B masked"],
                       name=f"p24_{args.tag}_masked_inputs.jpg")
        show_stack(la, f"p24_{args.tag}_laplacian_a.jpg", signed=True)
        show_stack(lb, f"p24_{args.tag}_laplacian_b.jpg", signed=True)
        show_stack(lblend, f"p24_{args.tag}_laplacian_blend.jpg", signed=True)


if __name__ == "__main__":
    main()
