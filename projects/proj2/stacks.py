"""Part 2.3 -- Gaussian and Laplacian stacks.

A stack is a pyramid without the downsampling: every level stays at full
resolution, so levels can be combined pixelwise. cv2.pyrDown() and
skimage.transform.pyramid_gaussian() are explicitly banned here.

    python stacks.py --image data/apple.jpeg --levels 5 --tag apple
    python stacks.py --image data/orange.jpeg --levels 5 --tag orange
"""

import argparse

import numpy as np

import utils
from edges import gaussian2d
from sharpen import conv_color


def gaussian_stack(im, levels=5, sigma=2.0, double_sigma=True):
    """Returns a list of `levels` images, progressively blurred, all full size.

    TODO(you): implement.
      - stack[0] is the original.
      - Each later level blurs the PREVIOUS level again.
      - With double_sigma, the effective sigma doubles per level (2, 4, 8...),
        which is what matches the pyramid's octave spacing and what makes the
        Szeliski figure look right. Remember the kernel has to grow with sigma.
    """
    raise NotImplementedError


def laplacian_stack(gstack):
    """Band-pass levels: difference of consecutive Gaussian levels.

    TODO(you): implement.
      - lap[i] = gstack[i] - gstack[i+1] for i < len-1
      - The LAST entry is the final Gaussian level itself (the residual
        low-pass). Without it the stack does not sum back to the original --
        check that np.sum(lap, axis=0) reconstructs the input, and say so.
    """
    raise NotImplementedError


def show_stack(stack, name, signed=False, subdir=None):
    """One row per stack. Laplacian levels are signed -> use norm_signed."""
    norm = utils.norm_signed if signed else (lambda x: np.clip(x, 0, 1))
    return utils.show_row([norm(l) for l in stack],
                          [f"level {i}" for i in range(len(stack))],
                          name=name, subdir=subdir)


def main():
    ap = argparse.ArgumentParser(description="Part 2.3 -- Gaussian/Laplacian stacks")
    ap.add_argument("--image", required=True)
    ap.add_argument("--levels", type=int, default=5)
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--tag", default="stack")
    args = ap.parse_args()

    im = utils.imread(args.image)
    try:
        g = gaussian_stack(im, args.levels, args.sigma)
        l = laplacian_stack(g)
    except NotImplementedError:
        print("Implement gaussian_stack / laplacian_stack first.")
        return

    err = np.abs(np.sum(l, axis=0) - im).max()
    print(f"reconstruction error (sum of Laplacian stack vs original): {err:.2e}")

    show_stack(g, f"p23_{args.tag}_gaussian.jpg")
    show_stack(l, f"p23_{args.tag}_laplacian.jpg", signed=True)


if __name__ == "__main__":
    main()
