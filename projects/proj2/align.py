"""Two-point alignment for hybrid images (Part 2.2).

Stand-in for the course's align_image_code.py. You click the same two
landmarks (usually the two eyes) on each image; this rotates, scales, and
translates im2 onto im1 so the landmarks coincide, then crops both to a
common size. Alignment is what makes the perceptual switch work -- a hybrid
of two misaligned faces just looks like a double exposure.

    python align.py data/DerekPicture.jpg data/nutmeg.jpg --save aligned_derek
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
import skimage.transform as sktr

import utils


def get_points(im1, im2):
    """Click 2 points on each image (same landmarks, same order)."""
    print("Click 2 points on the LEFT image, then 2 matching points on the RIGHT.")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, im, t in zip(axes, [im1, im2], ["image 1", "image 2"]):
        ax.imshow(im, cmap=None if im.ndim == 3 else "gray")
        ax.set_title(t)
        ax.axis("off")
    pts = plt.ginput(4, timeout=0)
    plt.close(fig)
    return np.array(pts[:2]), np.array(pts[2:])


def _recenter(im, r, c):
    """Shift so that (r, c) lands at the image center, padding with zeros."""
    h, w = im.shape[:2]
    dr, dc = int(h / 2 - r), int(w / 2 - c)
    pad = [(max(0, 2 * dr), max(0, -2 * dr)), (max(0, 2 * dc), max(0, -2 * dc))]
    if im.ndim == 3:
        pad.append((0, 0))
    return np.pad(im, pad, mode="constant")


def _angle(p):
    d = p[1] - p[0]
    return np.degrees(np.arctan2(-d[1], d[0]))


def _length(p):
    return np.linalg.norm(p[1] - p[0])


def match_size(im1, im2):
    """Center-crop both images to their common height and width."""
    h = min(im1.shape[0], im2.shape[0])
    w = min(im1.shape[1], im2.shape[1])

    def crop(im):
        r0 = (im.shape[0] - h) // 2
        c0 = (im.shape[1] - w) // 2
        return im[r0:r0 + h, c0:c0 + w]

    return crop(im1), crop(im2)


def align_images(im1, im2, pts1=None, pts2=None):
    """Align im2 onto im1 using two corresponding points on each."""
    if pts1 is None or pts2 is None:
        pts1, pts2 = get_points(im1, im2)

    # center on the midpoint of the two landmarks
    im1 = _recenter(im1, *pts1.mean(axis=0)[::-1])
    im2 = _recenter(im2, *pts2.mean(axis=0)[::-1])

    # scale im2 so the landmark separation matches
    scale = _length(pts1) / _length(pts2)
    im2 = sktr.rescale(im2, scale, channel_axis=2 if im2.ndim == 3 else None)

    # rotate im2 so the landmark axis matches
    im2 = sktr.rotate(im2, -(_angle(pts1) - _angle(pts2)))

    return match_size(im1, im2)


def main():
    ap = argparse.ArgumentParser(description="two-point alignment for hybrids")
    ap.add_argument("im1")
    ap.add_argument("im2")
    ap.add_argument("--save", default="aligned", help="output name prefix")
    args = ap.parse_args()

    a, b = align_images(utils.imread(args.im1), utils.imread(args.im2))
    utils.imsave(f"{args.save}_1.jpg", a, subdir="aligned")
    utils.imsave(f"{args.save}_2.jpg", b, subdir="aligned")


if __name__ == "__main__":
    main()
